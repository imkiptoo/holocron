//! Vector store backed by PostgreSQL + the `pgvector` extension.
//!
//! All training material lives in a single `training_data` table; retrieval is
//! nearest-neighbour by cosine distance (`<=>`).

use async_trait::async_trait;
use pgvector::Vector;
use sqlx::{PgPool, Row};
use uuid::Uuid;

use crate::config::Config;
use crate::error::Result;
use crate::traits::VectorStore;
use crate::types::{TrainingKind, TrainingRow};

#[derive(Clone)]
pub struct PgVectorStore {
    pool: PgPool,
    dims: usize,
}

impl PgVectorStore {
    /// Connect to the vector database using `VECTOR_DATABASE_URL`.
    pub async fn connect(config: &Config) -> Result<Self> {
        let pool =
            crate::providers::postgres::connect_pool_from_config(config, &config.vector_database_url)
                .await?;
        Ok(Self { pool, dims: config.embed_dims })
    }

    /// Wrap an existing pool (lets the vector store share the warehouse pool).
    pub fn from_pool(pool: PgPool, dims: usize) -> Self {
        Self { pool, dims }
    }

    /// Create the pgvector extension, tables, and ANN indexes.
    pub async fn init_db(&self) -> Result<()> {
        sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
            .execute(&self.pool)
            .await?;

        let create = format!(
            "CREATE TABLE IF NOT EXISTS training_data (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                kind text NOT NULL,
                question text,
                content text NOT NULL,
                embedding vector({dims}) NOT NULL,
                created_at timestamptz NOT NULL DEFAULT now()
            )",
            dims = self.dims
        );
        sqlx::query(&create).execute(&self.pool).await?;

        // Verbatim SQL cache: previously answered questions keyed by the exact
        // normalized question text (NOT by embedding similarity - see the
        // `VectorStore` cache docs). One row per question (upsert on `question`).
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS query_cache (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                question text NOT NULL,
                sql_text text NOT NULL,
                created_at timestamptz NOT NULL DEFAULT now()
            )",
        )
        .execute(&self.pool)
        .await?;
        // Migrate any earlier embedding-based cache in place: drop the vector
        // column + ANN index that the old similarity cache used.
        sqlx::query("DROP INDEX IF EXISTS query_cache_embedding_idx")
            .execute(&self.pool)
            .await?;
        sqlx::query("ALTER TABLE query_cache DROP COLUMN IF EXISTS embedding")
            .execute(&self.pool)
            .await?;
        // Collapse any duplicate question rows left by the old similarity cache
        // (keeping the newest) so the unique key below can be created.
        sqlx::query(
            "DELETE FROM query_cache a USING query_cache b
             WHERE a.question = b.question AND a.created_at < b.created_at",
        )
        .execute(&self.pool)
        .await?;
        // Exact-match lookups + upserts need a unique key on the question.
        sqlx::query(
            "CREATE UNIQUE INDEX IF NOT EXISTS query_cache_question_key ON query_cache (question)",
        )
        .execute(&self.pool)
        .await?;

        // Retrieval is always scoped to one kind, but a single HNSW index over
        // the whole table applies the `kind` filter only *after* the ANN walk -
        // hurting recall and latency when neighbours are the wrong kind. Instead
        // build one partial index per kind so each search traverses only its own
        // rows. (The query interpolates the kind as a literal so the planner can
        // match these partial-index predicates.) Drop the old combined index.
        sqlx::query("DROP INDEX IF EXISTS training_data_embedding_idx")
            .execute(&self.pool)
            .await?;
        for kind in [TrainingKind::Ddl, TrainingKind::Documentation, TrainingKind::Sql] {
            let kind = kind.as_str();
            let idx = format!(
                "CREATE INDEX IF NOT EXISTS training_data_embedding_{kind}_idx
                 ON training_data USING hnsw (embedding vector_cosine_ops)
                 WHERE kind = '{kind}'"
            );
            sqlx::query(&idx).execute(&self.pool).await?;
        }

        tracing::debug!(dims = self.dims, "pgvector schema (training_data + query_cache + indexes) ready");
        Ok(())
    }

    async fn insert(
        &self,
        kind: &str,
        question: Option<&str>,
        content: &str,
        embedding: Vec<f32>,
    ) -> Result<String> {
        let id: Uuid = sqlx::query(
            "INSERT INTO training_data (kind, question, content, embedding)
             VALUES ($1, $2, $3, $4) RETURNING id",
        )
        .bind(kind)
        .bind(question)
        .bind(content)
        .bind(Vector::from(embedding))
        .fetch_one(&self.pool)
        .await?
        .get("id");
        Ok(id.to_string())
    }

    async fn nearest(&self, kind: &str, embedding: &[f32], k: usize) -> Result<Vec<sqlx::postgres::PgRow>> {
        // `kind` is interpolated as a literal (not bound) so the planner can
        // match the per-kind partial HNSW index - a bound `$1` leaves `kind`
        // unknown at plan time and the partial index would be skipped. Safe
        // because every caller passes a fixed `TrainingKind::as_str()` value.
        debug_assert!(matches!(kind, "ddl" | "documentation" | "sql"));
        let sql = format!(
            "SELECT question, content FROM training_data
             WHERE kind = '{kind}'
             ORDER BY embedding <=> $1
             LIMIT $2"
        );
        let rows = sqlx::query(&sql)
            .bind(Vector::from(embedding.to_vec()))
            .bind(k as i64)
            .fetch_all(&self.pool)
            .await?;
        Ok(rows)
    }
}

#[async_trait]
impl VectorStore for PgVectorStore {
    async fn add_ddl(&self, ddl: &str, embedding: Vec<f32>) -> Result<String> {
        self.insert(TrainingKind::Ddl.as_str(), None, ddl, embedding).await
    }

    async fn add_documentation(&self, doc: &str, embedding: Vec<f32>) -> Result<String> {
        self.insert(TrainingKind::Documentation.as_str(), None, doc, embedding)
            .await
    }

    async fn add_question_sql(
        &self,
        question: &str,
        sql: &str,
        embedding: Vec<f32>,
    ) -> Result<String> {
        self.insert(TrainingKind::Sql.as_str(), Some(question), sql, embedding)
            .await
    }

    async fn get_related_ddl(&self, embedding: &[f32], k: usize) -> Result<Vec<String>> {
        let rows = self.nearest(TrainingKind::Ddl.as_str(), embedding, k).await?;
        Ok(rows.into_iter().map(|r| r.get::<String, _>("content")).collect())
    }

    async fn get_related_documentation(&self, embedding: &[f32], k: usize) -> Result<Vec<String>> {
        let rows = self
            .nearest(TrainingKind::Documentation.as_str(), embedding, k)
            .await?;
        Ok(rows.into_iter().map(|r| r.get::<String, _>("content")).collect())
    }

    async fn get_similar_question_sql(
        &self,
        embedding: &[f32],
        k: usize,
    ) -> Result<Vec<(String, String)>> {
        let rows = self.nearest(TrainingKind::Sql.as_str(), embedding, k).await?;
        Ok(rows
            .into_iter()
            .map(|r| {
                let q: Option<String> = r.get("question");
                let sql: String = r.get("content");
                (q.unwrap_or_default(), sql)
            })
            .collect())
    }

    /// Fetch all three kinds of retrieval context in a **single** round-trip
    /// (one connection) via `UNION ALL` of three partial-index scans, instead
    /// of three separate queries. Each branch still uses its per-kind partial
    /// HNSW index; a `tag` column lets us split the rows back out by kind while
    /// preserving each branch's most-relevant-first ordering.
    async fn get_context(
        &self,
        embedding: &[f32],
        k_ddl: usize,
        k_docs: usize,
        k_sql: usize,
    ) -> Result<(Vec<String>, Vec<String>, Vec<(String, String)>)> {
        let sql = format!(
            "(SELECT 'd' AS tag, question, content FROM training_data
                WHERE kind = 'ddl' ORDER BY embedding <=> $1 LIMIT {k_ddl})
             UNION ALL
             (SELECT 'o' AS tag, question, content FROM training_data
                WHERE kind = 'documentation' ORDER BY embedding <=> $1 LIMIT {k_docs})
             UNION ALL
             (SELECT 's' AS tag, question, content FROM training_data
                WHERE kind = 'sql' ORDER BY embedding <=> $1 LIMIT {k_sql})"
        );
        let rows = sqlx::query(&sql)
            .bind(Vector::from(embedding.to_vec()))
            .fetch_all(&self.pool)
            .await?;

        let mut ddls = Vec::new();
        let mut docs = Vec::new();
        let mut examples = Vec::new();
        for r in rows {
            let tag: String = r.get("tag");
            let content: String = r.get("content");
            match tag.as_str() {
                "d" => ddls.push(content),
                "o" => docs.push(content),
                _ => {
                    let q: Option<String> = r.get("question");
                    examples.push((q.unwrap_or_default(), content));
                }
            }
        }
        Ok((ddls, docs, examples))
    }

    async fn cache_lookup(&self, question_key: &str, ttl_secs: u64) -> Result<Option<String>> {
        // Exact match on the (already normalized) question, honouring the TTL.
        let ttl_clause = if ttl_secs == 0 {
            String::new()
        } else {
            format!("AND created_at > now() - interval '{ttl_secs} seconds'")
        };
        let sql = format!(
            "SELECT sql_text FROM query_cache WHERE question = $1 {ttl_clause} LIMIT 1"
        );
        let row = sqlx::query(&sql)
            .bind(question_key)
            .fetch_optional(&self.pool)
            .await?;
        Ok(row.map(|r| r.get::<String, _>("sql_text")))
    }

    async fn cache_put(&self, question_key: &str, sql: &str) -> Result<()> {
        // Upsert so a re-answered question refreshes its SQL + timestamp.
        sqlx::query(
            "INSERT INTO query_cache (question, sql_text) VALUES ($1, $2)
             ON CONFLICT (question) DO UPDATE
             SET sql_text = EXCLUDED.sql_text, created_at = now()",
        )
        .bind(question_key)
        .bind(sql)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn cache_clear(&self) -> Result<u64> {
        let res = sqlx::query("DELETE FROM query_cache").execute(&self.pool).await?;
        Ok(res.rows_affected())
    }

    async fn get_training_data(&self) -> Result<Vec<TrainingRow>> {
        let rows = sqlx::query(
            "SELECT id, kind, question, content FROM training_data ORDER BY created_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .into_iter()
            .map(|r| {
                let id: Uuid = r.get("id");
                let kind: String = r.get("kind");
                TrainingRow {
                    id: id.to_string(),
                    kind: match kind.as_str() {
                        "ddl" => TrainingKind::Ddl,
                        "documentation" => TrainingKind::Documentation,
                        _ => TrainingKind::Sql,
                    },
                    question: r.get("question"),
                    content: r.get("content"),
                }
            })
            .collect())
    }

    async fn remove_training_data(&self, id: &str) -> Result<bool> {
        let uuid = Uuid::parse_str(id)
            .map_err(|e| crate::error::Error::other(format!("invalid id: {e}")))?;
        let result = sqlx::query("DELETE FROM training_data WHERE id = $1")
            .bind(uuid)
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected() > 0)
    }
}
