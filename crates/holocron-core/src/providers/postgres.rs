//! SQL runner for a PostgreSQL warehouse: executes generated queries and
//! introspects `information_schema` into `CREATE TABLE`-style DDL for training.

use std::collections::BTreeMap;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, Utc};
use futures::TryStreamExt;
use serde_json::Value;
use sqlx::postgres::{PgPoolOptions, PgRow};
use sqlx::{Column, PgPool, Row, TypeInfo};
use uuid::Uuid;

use crate::config::Config;
use crate::error::Result;
use crate::traits::SqlRunner;
use crate::types::QueryResult;

#[derive(Clone)]
pub struct PgSqlRunner {
    pool: PgPool,
    /// Max rows returned by `run_sql` (0 = unlimited).
    max_rows: usize,
    /// Execute queries inside a `READ ONLY` transaction (DB-enforced no-writes).
    read_only: bool,
    /// Per-query `statement_timeout` in milliseconds (0 = no limit).
    statement_timeout_ms: u64,
}

/// Build a warehouse/vector connection pool sized from config.
///
/// Shared by both Postgres-backed providers so the warehouse and vector store
/// can reuse a single pool when they point at the same database (see
/// `default_engine`). `min_connections(1)` keeps one connection warm so the
/// first query doesn't pay full TCP+TLS setup; `acquire_timeout` bounds how
/// long a query waits for a connection when the pool is saturated.
pub async fn connect_pool(
    url: &str,
    max_connections: u32,
    acquire_timeout: Duration,
) -> Result<PgPool> {
    let pool = PgPoolOptions::new()
        .max_connections(max_connections.max(1))
        .min_connections(1)
        .acquire_timeout(acquire_timeout)
        .connect(url)
        .await?;
    Ok(pool)
}

/// Build a pool from a [`Config`] (used by both providers' `connect`).
pub(crate) async fn connect_pool_from_config(config: &Config, url: &str) -> Result<PgPool> {
    connect_pool(
        url,
        config.db_max_connections,
        Duration::from_secs(config.db_acquire_timeout_secs),
    )
    .await
}

impl PgSqlRunner {
    /// Connect to the warehouse using `DATABASE_URL`.
    pub async fn connect(config: &Config) -> Result<Self> {
        let pool = connect_pool_from_config(config, &config.database_url).await?;
        // The AST gate enforces SELECT-only, so a read-only tx is also warranted
        // whenever validation is on (belt and braces).
        let read_only = config.read_only || config.validate_sql;
        Ok(Self::from_pool(
            pool,
            config.max_rows,
            read_only,
            config.statement_timeout_secs.saturating_mul(1000),
        ))
    }

    /// Wrap an existing pool (lets the warehouse + vector store share one).
    pub fn from_pool(
        pool: PgPool,
        max_rows: usize,
        read_only: bool,
        statement_timeout_ms: u64,
    ) -> Self {
        Self { pool, max_rows, read_only, statement_timeout_ms }
    }
}

/// Try to decode column `i` as `$t` into a JSON value; NULL/errors -> `null`.
macro_rules! try_val {
    ($row:expr, $i:expr, $t:ty) => {
        match $row.try_get::<Option<$t>, _>($i) {
            Ok(Some(v)) => serde_json::json!(v),
            _ => Value::Null,
        }
    };
}

/// Decode NUMERIC preserving precision: emit a JSON number when it round-trips
/// through f64, otherwise a string.
fn numeric_to_value(row: &PgRow, i: usize) -> Value {
    match row.try_get::<Option<sqlx::types::BigDecimal>, _>(i) {
        Ok(Some(v)) => {
            let s = v.to_string();
            s.parse::<f64>()
                .ok()
                .and_then(serde_json::Number::from_f64)
                .map(Value::Number)
                .unwrap_or(Value::String(s))
        }
        _ => Value::Null,
    }
}

/// A resolved decoder for a column, chosen once from its Postgres type name so
/// we don't re-match the type string for every cell in every row.
#[derive(Clone, Copy)]
enum ColKind {
    Bool,
    Int2,
    Int4,
    Int8,
    Float4,
    Float8,
    Numeric,
    Uuid,
    TimestampTz,
    Timestamp,
    Date,
    Time,
    Json,
    Text,
}

impl ColKind {
    /// Resolve a column's decoder from its Postgres type name.
    fn from_type_name(name: &str) -> Self {
        match name {
            "BOOL" => ColKind::Bool,
            "INT2" => ColKind::Int2,
            "INT4" => ColKind::Int4,
            "INT8" => ColKind::Int8,
            "FLOAT4" => ColKind::Float4,
            "FLOAT8" => ColKind::Float8,
            "NUMERIC" => ColKind::Numeric,
            "UUID" => ColKind::Uuid,
            "TIMESTAMPTZ" => ColKind::TimestampTz,
            "TIMESTAMP" => ColKind::Timestamp,
            "DATE" => ColKind::Date,
            "TIME" => ColKind::Time,
            "JSON" | "JSONB" => ColKind::Json,
            // TEXT, VARCHAR, BPCHAR, NAME, CHAR, and anything else: string.
            _ => ColKind::Text,
        }
    }
}

/// Map one cell to a JSON value using its column's pre-resolved decoder.
fn cell_to_value(row: &PgRow, i: usize, kind: ColKind) -> Value {
    match kind {
        ColKind::Bool => try_val!(row, i, bool),
        ColKind::Int2 => try_val!(row, i, i16),
        ColKind::Int4 => try_val!(row, i, i32),
        ColKind::Int8 => try_val!(row, i, i64),
        ColKind::Float4 => try_val!(row, i, f32),
        ColKind::Float8 => try_val!(row, i, f64),
        ColKind::Numeric => numeric_to_value(row, i),
        ColKind::Uuid => try_val!(row, i, Uuid),
        ColKind::TimestampTz => try_val!(row, i, DateTime<Utc>),
        ColKind::Timestamp => try_val!(row, i, NaiveDateTime),
        ColKind::Date => try_val!(row, i, NaiveDate),
        ColKind::Time => try_val!(row, i, NaiveTime),
        ColKind::Json => try_val!(row, i, Value),
        ColKind::Text => try_val!(row, i, String),
    }
}

#[async_trait]
impl SqlRunner for PgSqlRunner {
    async fn run_sql(&self, sql: &str) -> Result<QueryResult> {
        // Execute inside a transaction so we can enforce, at the database level:
        //  - `READ ONLY` - Postgres itself rejects any write (defeats writable
        //    CTEs / side-effecting functions even if the AST gate is bypassed);
        //  - `SET LOCAL statement_timeout` - caps runaway/expensive queries.
        let mut tx = self.pool.begin().await?;
        if self.read_only {
            sqlx::query("SET TRANSACTION READ ONLY").execute(&mut *tx).await?;
        }
        if self.statement_timeout_ms > 0 {
            // Integer milliseconds - safe to interpolate.
            let set = format!("SET LOCAL statement_timeout = {}", self.statement_timeout_ms);
            sqlx::query(&set).execute(&mut *tx).await?;
        }

        // Stream rows rather than buffering the whole result, so a huge result
        // set can't blow memory - and stop early once the row cap is hit.
        // Column names + per-column decoders are resolved once from the first
        // row instead of re-inspecting the type of every cell.
        let mut columns: Vec<String> = Vec::new();
        let mut kinds: Vec<ColKind> = Vec::new();
        let mut data: Vec<Vec<Value>> = Vec::new();
        {
            let mut stream = sqlx::query(sql).fetch(&mut *tx);
            while let Some(row) = stream.try_next().await? {
                if columns.is_empty() {
                    for c in row.columns() {
                        columns.push(c.name().to_string());
                        kinds.push(ColKind::from_type_name(c.type_info().name()));
                    }
                }
                if self.max_rows != 0 && data.len() >= self.max_rows {
                    tracing::warn!(cap = self.max_rows, "run_sql result truncated to row cap");
                    break;
                }
                let vals = kinds
                    .iter()
                    .enumerate()
                    .map(|(i, &k)| cell_to_value(&row, i, k))
                    .collect();
                data.push(vals);
            }
        } // stream dropped before we finish the transaction

        // Read-only: nothing to persist, so roll back.
        if self.read_only {
            tx.rollback().await?;
        } else {
            tx.commit().await?;
        }

        tracing::debug!(rows = data.len(), cols = columns.len(), "warehouse query returned");
        Ok(QueryResult { columns, rows: data })
    }

    async fn introspect_ddl(&self) -> Result<Vec<String>> {
        let rows = sqlx::query(
            "SELECT table_schema, table_name, column_name, data_type, is_nullable
             FROM information_schema.columns
             WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
               AND table_name <> 'training_data'
             ORDER BY table_schema, table_name, ordinal_position",
        )
        .fetch_all(&self.pool)
        .await?;

        // Preserve encounter order per table while grouping columns.
        let mut tables: BTreeMap<(String, String), Vec<String>> = BTreeMap::new();
        for r in rows {
            let schema: String = r.get("table_schema");
            let table: String = r.get("table_name");
            let column: String = r.get("column_name");
            let data_type: String = r.get("data_type");
            let nullable: String = r.get("is_nullable");
            let not_null = if nullable == "NO" { " NOT NULL" } else { "" };
            tables
                .entry((schema, table))
                .or_default()
                .push(format!("  {column} {data_type}{not_null}"));
        }

        let ddls: Vec<String> = tables
            .into_iter()
            .map(|((schema, table), cols)| {
                format!("CREATE TABLE {schema}.{table} (\n{}\n);", cols.join(",\n"))
            })
            .collect();
        tracing::debug!(tables = ddls.len(), "introspected warehouse schema");
        Ok(ddls)
    }
}
