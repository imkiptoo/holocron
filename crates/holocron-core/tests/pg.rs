//! Integration tests for the Postgres-backed providers.
//!
//! These require a real PostgreSQL instance with the `pgvector` extension.
//! They are gated on the `HOLOCRON_TEST_DATABASE_URL` environment variable and
//! quietly skip when it is unset, so the default `cargo test` run stays
//! hermetic. To run them:
//!
//! ```sh
//! createdb holocron_test
//! psql "$HOLOCRON_TEST_DATABASE_URL" -c 'CREATE EXTENSION IF NOT EXISTS vector'
//! HOLOCRON_TEST_DATABASE_URL=postgres://postgres:postgres@localhost:5432/holocron_test \
//!   cargo test -p holocron-core --test pg
//! ```

use std::sync::OnceLock;

use holocron_core::providers::pgvector::PgVectorStore;
use holocron_core::providers::postgres::PgSqlRunner;
use holocron_core::traits::{SqlRunner, VectorStore};
use holocron_core::Config;
use tokio::sync::{Mutex, MutexGuard};

const DIMS: usize = 8;

/// These tests share one database (and the `training_data` / `query_cache`
/// tables), so they must not run concurrently. Each locks this global guard.
async fn db_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().await
}

/// Build a test config from `HOLOCRON_TEST_DATABASE_URL`, or `None` to skip.
fn env_config() -> Option<Config> {
    let url = std::env::var("HOLOCRON_TEST_DATABASE_URL").ok()?;
    Some(Config {
        gemini_api_key: "unused".into(),
        gemini_chat_model: "unused".into(),
        gemini_embed_model: "unused".into(),
        embed_dims: DIMS,
        gemini_max_concurrency: 8,
        gemini_max_retries: 3,
        database_url: url.clone(),
        vector_database_url: url,
        db_max_connections: 10,
        db_acquire_timeout_secs: 30,
        top_k_ddl: 5,
        top_k_docs: 5,
        top_k_sql: 5,
        read_only: true,
        max_rows: 10_000,
        validate_sql: true,
        allow_system_schemas: false,
        statement_timeout_secs: 30,
        cache_enabled: true,
        cache_ttl_secs: 3600,
        bind_addr: "unused".into(),
        request_timeout_secs: 120,
        max_concurrent_requests: 64,
        log_level: "info".into(),
        log_format: "console".into(),
    })
}

/// One-hot embedding of width `DIMS` with `1.0` at position `i`.
fn one_hot(i: usize) -> Vec<f32> {
    let mut v = vec![0.0; DIMS];
    v[i] = 1.0;
    v
}

macro_rules! skip_if_no_db {
    () => {
        match env_config() {
            Some(c) => c,
            None => {
                eprintln!("skipping: HOLOCRON_TEST_DATABASE_URL not set");
                return;
            }
        }
    };
}

#[tokio::test]
async fn pgvector_store_roundtrip() {
    let cfg = skip_if_no_db!();
    let _guard = db_guard().await;

    // Start from a clean training_data table so dims/rows are deterministic.
    let pool = sqlx::postgres::PgPoolOptions::new()
        .connect(&cfg.vector_database_url)
        .await
        .unwrap();
    sqlx::query("DROP TABLE IF EXISTS training_data")
        .execute(&pool)
        .await
        .unwrap();

    let store = PgVectorStore::connect(&cfg).await.unwrap();
    store.init_db().await.unwrap();
    // init_db is idempotent.
    store.init_db().await.unwrap();

    // Insert two DDLs with orthogonal embeddings, plus a doc and a q/sql pair.
    let ddl0 = store
        .add_ddl("CREATE TABLE customers (id int)", one_hot(0))
        .await
        .unwrap();
    let _ddl1 = store
        .add_ddl("CREATE TABLE orders (id int)", one_hot(1))
        .await
        .unwrap();
    store
        .add_documentation("revenue = price * qty", one_hot(2))
        .await
        .unwrap();
    store
        .add_question_sql("how many customers?", "SELECT count(*) FROM customers", one_hot(3))
        .await
        .unwrap();

    // Nearest DDL to the first one-hot vector is the customers table.
    let ddls = store.get_related_ddl(&one_hot(0), 5).await.unwrap();
    assert_eq!(ddls.len(), 2, "both DDL rows are retrievable");
    assert_eq!(ddls[0], "CREATE TABLE customers (id int)");

    // Kind filtering: doc retrieval never returns DDL/SQL rows.
    let docs = store.get_related_documentation(&one_hot(2), 5).await.unwrap();
    assert_eq!(docs, vec!["revenue = price * qty".to_string()]);

    // Question/SQL retrieval returns the pair.
    let ex = store.get_similar_question_sql(&one_hot(3), 5).await.unwrap();
    assert_eq!(ex, vec![("how many customers?".to_string(), "SELECT count(*) FROM customers".to_string())]);

    // Single-round-trip get_context splits all three buckets back out by kind.
    let (c_ddls, c_docs, c_ex) = store.get_context(&one_hot(0), 5, 5, 5).await.unwrap();
    assert_eq!(c_ddls.len(), 2);
    assert_eq!(c_ddls[0], "CREATE TABLE customers (id int)");
    assert_eq!(c_docs, vec!["revenue = price * qty".to_string()]);
    assert_eq!(c_ex, vec![("how many customers?".to_string(), "SELECT count(*) FROM customers".to_string())]);

    // Listing sees all four rows.
    let rows = store.get_training_data().await.unwrap();
    assert_eq!(rows.len(), 4);

    // Removing a real id returns true; a random (valid) uuid returns false.
    assert!(store.remove_training_data(&ddl0).await.unwrap());
    assert!(!store
        .remove_training_data("00000000-0000-0000-0000-000000000000")
        .await
        .unwrap());
    assert_eq!(store.get_training_data().await.unwrap().len(), 3);

    // A malformed id is an error, not a silent miss.
    assert!(store.remove_training_data("not-a-uuid").await.is_err());

    sqlx::query("DROP TABLE IF EXISTS training_data")
        .execute(&pool)
        .await
        .unwrap();
}

#[tokio::test]
async fn sql_runner_maps_column_types_to_json() {
    let cfg = skip_if_no_db!();
    let _guard = db_guard().await;
    let runner = PgSqlRunner::connect(&cfg).await.unwrap();

    let qr = runner
        .run_sql(
            "SELECT true AS b,
                    1::int2 AS i2,
                    2::int4 AS i4,
                    3::int8 AS i8,
                    1.5::float8 AS f8,
                    9.99::numeric AS num,
                    'hi'::text AS t,
                    '2020-01-02'::date AS d,
                    '{\"a\":1}'::jsonb AS j,
                    NULL::int4 AS n",
        )
        .await
        .unwrap();

    assert_eq!(qr.row_count(), 1);
    let idx = |name: &str| qr.columns.iter().position(|c| c == name).unwrap();
    let row = &qr.rows[0];

    assert_eq!(row[idx("b")], serde_json::json!(true));
    assert_eq!(row[idx("i2")], serde_json::json!(1));
    assert_eq!(row[idx("i4")], serde_json::json!(2));
    assert_eq!(row[idx("i8")], serde_json::json!(3));
    assert_eq!(row[idx("f8")], serde_json::json!(1.5));
    assert_eq!(row[idx("num")].as_f64().unwrap(), 9.99);
    assert_eq!(row[idx("t")], serde_json::json!("hi"));
    assert_eq!(row[idx("d")], serde_json::json!("2020-01-02"));
    assert_eq!(row[idx("j")], serde_json::json!({ "a": 1 }));
    assert_eq!(row[idx("n")], serde_json::Value::Null);
}

#[tokio::test]
async fn sql_runner_handles_empty_result() {
    let cfg = skip_if_no_db!();
    let _guard = db_guard().await;
    let runner = PgSqlRunner::connect(&cfg).await.unwrap();
    let qr = runner.run_sql("SELECT 1 AS x WHERE false").await.unwrap();
    assert_eq!(qr.row_count(), 0);
    // No rows -> no column metadata is inferred.
    assert!(qr.columns.is_empty());
}

#[tokio::test]
async fn sql_runner_surfaces_query_errors() {
    let cfg = skip_if_no_db!();
    let _guard = db_guard().await;
    let runner = PgSqlRunner::connect(&cfg).await.unwrap();
    assert!(runner.run_sql("SELECT * FROM no_such_table_xyz").await.is_err());
}

#[tokio::test]
async fn introspect_ddl_emits_create_table_and_skips_training_data() {
    let cfg = skip_if_no_db!();
    let _guard = db_guard().await;
    let runner = PgSqlRunner::connect(&cfg).await.unwrap();

    let pool = sqlx::postgres::PgPoolOptions::new()
        .connect(&cfg.database_url)
        .await
        .unwrap();
    sqlx::query("CREATE SCHEMA IF NOT EXISTS holocron_it")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("DROP TABLE IF EXISTS holocron_it.widget")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("CREATE TABLE holocron_it.widget (id int NOT NULL, label text)")
        .execute(&pool)
        .await
        .unwrap();

    let ddls = runner.introspect_ddl().await.unwrap();
    let widget = ddls
        .iter()
        .find(|d| d.contains("holocron_it.widget"))
        .expect("widget table should be introspected");
    assert!(widget.contains("CREATE TABLE holocron_it.widget"));
    assert!(widget.contains("id integer NOT NULL"));
    assert!(widget.contains("label text"));
    assert!(!widget.contains("label text NOT NULL")); // nullable column has no NOT NULL

    // The engine's own bookkeeping table is never introspected as schema.
    assert!(!ddls.iter().any(|d| d.contains(".training_data ")));

    sqlx::query("DROP SCHEMA holocron_it CASCADE")
        .execute(&pool)
        .await
        .unwrap();
}

#[tokio::test]
async fn verbatim_cache_exact_match_upsert_ttl_and_clear() {
    let cfg = skip_if_no_db!();
    let _guard = db_guard().await;

    let pool = sqlx::postgres::PgPoolOptions::new()
        .connect(&cfg.vector_database_url)
        .await
        .unwrap();
    sqlx::query("DROP TABLE IF EXISTS query_cache")
        .execute(&pool)
        .await
        .unwrap();

    let store = PgVectorStore::connect(&cfg).await.unwrap();
    store.init_db().await.unwrap();

    store
        .cache_put("top 10 products by revenue", "SELECT ... ORDER BY rev DESC")
        .await
        .unwrap();

    // Exact key hits.
    let hit = store.cache_lookup("top 10 products by revenue", 0).await.unwrap();
    assert_eq!(hit.as_deref(), Some("SELECT ... ORDER BY rev DESC"));

    // The near-identical opposite question is a MISS - no similarity collision.
    let miss = store.cache_lookup("bottom 10 products by revenue", 0).await.unwrap();
    assert!(miss.is_none(), "opposite question must not hit the cache");

    // Upsert on the same key replaces the SQL (and refreshes created_at).
    store
        .cache_put("top 10 products by revenue", "SELECT v2 DESC")
        .await
        .unwrap();
    assert_eq!(
        store.cache_lookup("top 10 products by revenue", 0).await.unwrap().as_deref(),
        Some("SELECT v2 DESC")
    );
    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM query_cache")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 1, "upsert keeps a single row per question");

    // TTL: age the entry past a 60s TTL -> not returned; with no TTL (0) -> still there.
    sqlx::query("UPDATE query_cache SET created_at = now() - interval '1 hour'")
        .execute(&pool)
        .await
        .unwrap();
    assert!(store.cache_lookup("top 10 products by revenue", 60).await.unwrap().is_none());
    assert!(store.cache_lookup("top 10 products by revenue", 0).await.unwrap().is_some());

    // Clearing empties the cache.
    assert_eq!(store.cache_clear().await.unwrap(), 1);
    assert!(store.cache_lookup("top 10 products by revenue", 0).await.unwrap().is_none());

    sqlx::query("DROP TABLE IF EXISTS query_cache")
        .execute(&pool)
        .await
        .unwrap();
}

#[tokio::test]
async fn run_sql_enforces_row_cap() {
    let cfg = skip_if_no_db!();
    let _guard = db_guard().await;
    // Cap results at 3 rows regardless of how many the query would produce.
    let runner = PgSqlRunner::from_pool(
        holocron_core::providers::postgres::connect_pool(&cfg.database_url, 5, std::time::Duration::from_secs(30))
            .await
            .unwrap(),
        3,     // row cap
        true,  // read-only tx
        30_000, // statement_timeout ms
    );
    let qr = runner
        .run_sql("SELECT g FROM generate_series(1, 100) g")
        .await
        .unwrap();
    assert_eq!(qr.row_count(), 3, "streamed result stops at the row cap");
    assert_eq!(qr.columns, vec!["g".to_string()]);
}

#[tokio::test]
async fn read_only_transaction_blocks_writes_at_db_level() {
    let cfg = skip_if_no_db!();
    let _guard = db_guard().await;

    let pool = sqlx::postgres::PgPoolOptions::new()
        .connect(&cfg.database_url)
        .await
        .unwrap();
    sqlx::query("DROP TABLE IF EXISTS holo_probe").execute(&pool).await.unwrap();
    sqlx::query("CREATE TABLE holo_probe (x int)").execute(&pool).await.unwrap();
    sqlx::query("INSERT INTO holo_probe VALUES (1)").execute(&pool).await.unwrap();

    // A read-only runner (validation NOT involved here) must have Postgres
    // itself reject a write - the belt-and-braces below the AST gate.
    let runner = PgSqlRunner::from_pool(
        holocron_core::providers::postgres::connect_pool(&cfg.database_url, 5, std::time::Duration::from_secs(30))
            .await
            .unwrap(),
        0,
        true,   // READ ONLY tx
        30_000,
    );
    let res = runner.run_sql("DELETE FROM holo_probe").await;
    assert!(res.is_err(), "READ ONLY transaction must reject a DELETE");

    // ...and nothing was actually deleted.
    let n: i64 = sqlx::query_scalar("SELECT count(*) FROM holo_probe")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(n, 1);

    sqlx::query("DROP TABLE holo_probe").execute(&pool).await.unwrap();
}
