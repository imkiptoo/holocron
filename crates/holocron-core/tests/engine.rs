//! Engine orchestration tests.
//!
//! These exercise the full `train` / `generate_sql` / `run_sql` / `ask` control
//! flow against in-memory fake providers, so they run with no network or
//! database. The fakes record what the engine asked of them, letting us assert
//! on the wiring (which embedding was stored, which top-k was requested, whether
//! read-only blocked execution, how followups are parsed, etc.).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use holocron_core::{
    Config, Embedder, Engine, Error, Llm, QueryResult, SqlRunner, TrainingItem, TrainingKind,
    TrainingRow, VectorStore,
};
use holocron_core::types::Message;

// ---- Test config -------------------------------------------------------

fn test_config(read_only: bool) -> Config {
    Config {
        gemini_api_key: "k".into(),
        gemini_chat_model: "m".into(),
        gemini_embed_model: "e".into(),
        embed_dims: 3,
        gemini_max_concurrency: 8,
        gemini_max_retries: 3,
        database_url: "d".into(),
        vector_database_url: "d".into(),
        db_max_connections: 10,
        db_acquire_timeout_secs: 30,
        top_k_ddl: 2,
        top_k_docs: 3,
        top_k_sql: 4,
        read_only,
        max_rows: 10_000,
        // Validation off by default in engine tests so the legacy read-only
        // token path (and non-SQL fake queries) are exercised directly;
        // dedicated tests flip it on.
        validate_sql: false,
        allow_system_schemas: false,
        statement_timeout_secs: 30,
        // Cache off by default in engine tests so the fake store's default
        // (no-op) cache methods don't interfere with generation assertions.
        cache_enabled: false,
        cache_ttl_secs: 3600,
        bind_addr: "x".into(),
        request_timeout_secs: 120,
        max_concurrent_requests: 64,
        log_level: "info".into(),
        log_format: "console".into(),
    }
}

// ---- Fake embedder -----------------------------------------------------

/// Embeds text as a single-element vector holding its byte length, and records
/// every string it was asked to embed (in call order).
#[derive(Default)]
struct FakeEmbedder {
    calls: Mutex<Vec<String>>,
}

#[async_trait]
impl Embedder for FakeEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, Error> {
        self.calls.lock().unwrap().push(text.to_string());
        Ok(vec![text.len() as f32])
    }
    fn dims(&self) -> usize {
        1
    }
}

// ---- Scripted LLM ------------------------------------------------------

/// Returns preset replies in order; records the message lists it received.
#[derive(Default)]
struct ScriptedLlm {
    replies: Mutex<VecDeque<String>>,
    seen: Mutex<Vec<Vec<Message>>>,
}

impl ScriptedLlm {
    fn with(replies: impl IntoIterator<Item = &'static str>) -> Self {
        Self {
            replies: Mutex::new(replies.into_iter().map(String::from).collect()),
            seen: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl Llm for ScriptedLlm {
    async fn chat(&self, messages: &[Message]) -> Result<String, Error> {
        self.seen.lock().unwrap().push(messages.to_vec());
        match self.replies.lock().unwrap().pop_front() {
            Some(r) => Ok(r),
            None => Err(Error::Llm("no scripted reply".into())),
        }
    }
}

// ---- Fake vector store -------------------------------------------------

#[derive(Default)]
struct FakeStore {
    added_ddl: Mutex<Vec<(String, Vec<f32>)>>,
    added_doc: Mutex<Vec<(String, Vec<f32>)>>,
    added_qsql: Mutex<Vec<(String, String, Vec<f32>)>>,
    // Canned retrieval results.
    ret_ddl: Vec<String>,
    ret_doc: Vec<String>,
    ret_examples: Vec<(String, String)>,
    // Recorded k values for (ddl, doc, sql) retrieval.
    ks: Mutex<Vec<(String, usize)>>,
    rows: Vec<TrainingRow>,
    removed: Mutex<Vec<String>>,
    // Semantic cache fakes.
    cache_hit: Option<String>,
    cache_puts: Mutex<Vec<(String, String)>>,
    cache_cleared: AtomicUsize,
}

#[async_trait]
impl VectorStore for FakeStore {
    async fn add_ddl(&self, ddl: &str, embedding: Vec<f32>) -> Result<String, Error> {
        self.added_ddl.lock().unwrap().push((ddl.to_string(), embedding));
        Ok("ddl-id".into())
    }
    async fn add_documentation(&self, doc: &str, embedding: Vec<f32>) -> Result<String, Error> {
        self.added_doc.lock().unwrap().push((doc.to_string(), embedding));
        Ok("doc-id".into())
    }
    async fn add_question_sql(
        &self,
        question: &str,
        sql: &str,
        embedding: Vec<f32>,
    ) -> Result<String, Error> {
        self.added_qsql
            .lock()
            .unwrap()
            .push((question.to_string(), sql.to_string(), embedding));
        Ok("qsql-id".into())
    }
    async fn get_related_ddl(&self, _e: &[f32], k: usize) -> Result<Vec<String>, Error> {
        self.ks.lock().unwrap().push(("ddl".into(), k));
        Ok(self.ret_ddl.clone())
    }
    async fn get_related_documentation(&self, _e: &[f32], k: usize) -> Result<Vec<String>, Error> {
        self.ks.lock().unwrap().push(("doc".into(), k));
        Ok(self.ret_doc.clone())
    }
    async fn get_similar_question_sql(
        &self,
        _e: &[f32],
        k: usize,
    ) -> Result<Vec<(String, String)>, Error> {
        self.ks.lock().unwrap().push(("sql".into(), k));
        Ok(self.ret_examples.clone())
    }
    async fn get_training_data(&self) -> Result<Vec<TrainingRow>, Error> {
        Ok(self.rows.clone())
    }
    async fn remove_training_data(&self, id: &str) -> Result<bool, Error> {
        self.removed.lock().unwrap().push(id.to_string());
        Ok(id == "exists")
    }
    async fn cache_lookup(&self, _key: &str, _ttl: u64) -> Result<Option<String>, Error> {
        Ok(self.cache_hit.clone())
    }
    async fn cache_put(&self, key: &str, sql: &str) -> Result<(), Error> {
        self.cache_puts.lock().unwrap().push((key.to_string(), sql.to_string()));
        Ok(())
    }
    async fn cache_clear(&self) -> Result<u64, Error> {
        self.cache_cleared.fetch_add(1, Ordering::Relaxed);
        Ok(0)
    }
}

// ---- Fake SQL runner ---------------------------------------------------

#[derive(Default)]
struct FakeRunner {
    ran: Mutex<Vec<String>>,
    fail: bool,
    ddls: Vec<String>,
}

#[async_trait]
impl SqlRunner for FakeRunner {
    async fn run_sql(&self, sql: &str) -> Result<QueryResult, Error> {
        self.ran.lock().unwrap().push(sql.to_string());
        if self.fail {
            return Err(Error::other("boom"));
        }
        Ok(QueryResult {
            columns: vec!["n".into()],
            rows: vec![vec![serde_json::json!(1)]],
        })
    }
    async fn introspect_ddl(&self) -> Result<Vec<String>, Error> {
        Ok(self.ddls.clone())
    }
}

// ---- Builder -----------------------------------------------------------

/// Assemble an engine plus handles to each fake for post-hoc inspection.
struct Rig {
    engine: Engine,
    embedder: Arc<FakeEmbedder>,
    llm: Arc<ScriptedLlm>,
    store: Arc<FakeStore>,
    runner: Arc<FakeRunner>,
}

fn rig(read_only: bool, llm: ScriptedLlm, store: FakeStore, runner: FakeRunner) -> Rig {
    rig_with(test_config(read_only), llm, store, runner)
}

/// Like [`rig`] but with the semantic cache turned on.
fn rig_cached(llm: ScriptedLlm, store: FakeStore) -> Rig {
    let mut config = test_config(true);
    config.cache_enabled = true;
    rig_with(config, llm, store, FakeRunner::default())
}

fn rig_with(config: Config, llm: ScriptedLlm, store: FakeStore, runner: FakeRunner) -> Rig {
    let embedder = Arc::new(FakeEmbedder::default());
    let llm = Arc::new(llm);
    let store = Arc::new(store);
    let runner = Arc::new(runner);
    let engine = Engine::new(llm.clone(), embedder.clone(), store.clone(), runner.clone(), config);
    Rig { engine, embedder, llm, store, runner }
}

// ---- Training ----------------------------------------------------------

#[tokio::test]
async fn train_ddl_embeds_and_stores_ddl() {
    let r = rig(true, ScriptedLlm::default(), FakeStore::default(), FakeRunner::default());
    let id = r
        .engine
        .train(TrainingItem::Ddl { ddl: "CREATE TABLE t (id int)".into() })
        .await
        .unwrap();
    assert_eq!(id, "ddl-id");

    let added = r.store.added_ddl.lock().unwrap();
    assert_eq!(added.len(), 1);
    assert_eq!(added[0].0, "CREATE TABLE t (id int)");
    // Embedding was computed from the DDL text itself.
    assert_eq!(r.embedder.calls.lock().unwrap().as_slice(), &["CREATE TABLE t (id int)"]);
}

#[tokio::test]
async fn train_documentation_embeds_and_stores_doc() {
    let r = rig(true, ScriptedLlm::default(), FakeStore::default(), FakeRunner::default());
    let id = r
        .engine
        .train(TrainingItem::Documentation { documentation: "revenue = price*qty".into() })
        .await
        .unwrap();
    assert_eq!(id, "doc-id");
    assert_eq!(r.store.added_doc.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn train_sql_with_question_embeds_the_question_not_the_sql() {
    let r = rig(true, ScriptedLlm::default(), FakeStore::default(), FakeRunner::default());
    let id = r
        .engine
        .train(TrainingItem::Sql {
            question: Some("how many customers?".into()),
            sql: "SELECT count(*) FROM customers".into(),
        })
        .await
        .unwrap();
    assert_eq!(id, "qsql-id");

    let added = r.store.added_qsql.lock().unwrap();
    assert_eq!(added[0].0, "how many customers?");
    assert_eq!(added[0].1, "SELECT count(*) FROM customers");
    // Crucially, the embedding is of the QUESTION (that's what future
    // questions match against), so no LLM call was needed.
    assert_eq!(r.embedder.calls.lock().unwrap().as_slice(), &["how many customers?"]);
    assert!(r.llm.seen.lock().unwrap().is_empty());
}

#[tokio::test]
async fn train_sql_without_question_synthesizes_one_via_llm() {
    // The model reply is wrapped in quotes/whitespace to prove trimming.
    let llm = ScriptedLlm::with(["  \"How many customers are there?\"  "]);
    let r = rig(true, llm, FakeStore::default(), FakeRunner::default());
    r.engine
        .train(TrainingItem::Sql { question: None, sql: "SELECT count(*) FROM customers".into() })
        .await
        .unwrap();

    let added = r.store.added_qsql.lock().unwrap();
    assert_eq!(added[0].0, "How many customers are there?"); // quotes + spaces trimmed
    // Embedding is of the synthesized question, not the SQL.
    assert_eq!(
        r.embedder.calls.lock().unwrap().as_slice(),
        &["How many customers are there?"]
    );
    // The LLM was asked exactly once, and the SQL was in the prompt.
    let seen = r.llm.seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert!(seen[0].iter().any(|m| m.content.contains("SELECT count(*) FROM customers")));
}

#[tokio::test]
async fn train_from_information_schema_adds_every_table() {
    let runner = FakeRunner {
        ddls: vec!["CREATE TABLE a (..)".into(), "CREATE TABLE b (..)".into()],
        ..Default::default()
    };
    let r = rig(true, ScriptedLlm::default(), FakeStore::default(), runner);
    let n = r.engine.train_from_information_schema().await.unwrap();
    assert_eq!(n, 2);
    assert_eq!(r.store.added_ddl.lock().unwrap().len(), 2);
    assert_eq!(r.embedder.calls.lock().unwrap().len(), 2);
}

// ---- generate_sql ------------------------------------------------------

#[tokio::test]
async fn generate_sql_retrieves_with_configured_topk_and_extracts_sql() {
    let store = FakeStore {
        ret_ddl: vec!["CREATE TABLE customers (id int)".into()],
        ret_doc: vec!["a doc".into()],
        ret_examples: vec![("prior?".into(), "SELECT 9".into())],
        ..Default::default()
    };
    let llm = ScriptedLlm::with(["```sql\nSELECT count(*) FROM customers;\n```"]);
    let r = rig(true, llm, store, FakeRunner::default());

    let sql = r.engine.generate_sql("how many customers?").await.unwrap();
    assert_eq!(sql, "SELECT count(*) FROM customers");

    // top-k values from test_config were threaded through to the store.
    let ks = r.store.ks.lock().unwrap();
    assert!(ks.contains(&("ddl".into(), 2)));
    assert!(ks.contains(&("doc".into(), 3)));
    assert!(ks.contains(&("sql".into(), 4)));

    // The retrieved context and few-shot example reached the prompt.
    let seen = r.llm.seen.lock().unwrap();
    let sys = &seen[0][0].content;
    assert!(sys.contains("CREATE TABLE customers (id int)"));
    assert!(seen[0].iter().any(|m| m.content.contains("prior?")));
}

#[tokio::test]
async fn generate_sql_errors_when_reply_has_no_sql() {
    let llm = ScriptedLlm::with(["I'm not sure how to answer that."]);
    let r = rig(true, llm, FakeStore::default(), FakeRunner::default());
    let err = r.engine.generate_sql("???").await.unwrap_err();
    assert!(matches!(err, Error::NoSql));
}

// ---- run_sql / read-only ----------------------------------------------

#[tokio::test]
async fn run_sql_blocks_writes_in_read_only_mode() {
    let r = rig(true, ScriptedLlm::default(), FakeStore::default(), FakeRunner::default());
    let err = r.engine.run_sql("DELETE FROM customers").await.unwrap_err();
    assert!(matches!(err, Error::ReadOnly(_)));
    // The runner must never have been touched.
    assert!(r.runner.ran.lock().unwrap().is_empty());
}

#[tokio::test]
async fn run_sql_allows_selects_in_read_only_mode() {
    let r = rig(true, ScriptedLlm::default(), FakeStore::default(), FakeRunner::default());
    let qr = r.engine.run_sql("SELECT 1").await.unwrap();
    assert_eq!(qr.row_count(), 1);
    assert_eq!(r.runner.ran.lock().unwrap().as_slice(), &["SELECT 1"]);
}

#[tokio::test]
async fn run_sql_allows_writes_when_not_read_only() {
    let r = rig(false, ScriptedLlm::default(), FakeStore::default(), FakeRunner::default());
    r.engine.run_sql("DELETE FROM customers").await.unwrap();
    assert_eq!(r.runner.ran.lock().unwrap().as_slice(), &["DELETE FROM customers"]);
}

/// Build a rig with the AST validation gate enabled.
fn rig_validating() -> Rig {
    let mut config = test_config(true);
    config.validate_sql = true;
    rig_with(config, ScriptedLlm::default(), FakeStore::default(), FakeRunner::default())
}

#[tokio::test]
async fn validated_run_sql_rejects_system_schema_before_runner() {
    let r = rig_validating();
    let err = r.engine.run_sql("SELECT * FROM information_schema.columns").await.unwrap_err();
    assert!(matches!(err, Error::Rejected(_)));
    // The gate short-circuits before the query reaches the database.
    assert!(r.runner.ran.lock().unwrap().is_empty());
}

#[tokio::test]
async fn validated_run_sql_rejects_writable_cte() {
    let r = rig_validating();
    // First token is `with`, so the legacy read-only check would have passed it.
    let err = r
        .engine
        .run_sql("WITH x AS (DELETE FROM customers RETURNING *) SELECT * FROM x")
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Rejected(_)));
    assert!(r.runner.ran.lock().unwrap().is_empty());
}

#[tokio::test]
async fn validated_run_sql_allows_plain_select() {
    let r = rig_validating();
    r.engine.run_sql("SELECT count(*) FROM sales.orders").await.unwrap();
    assert_eq!(
        r.runner.ran.lock().unwrap().as_slice(),
        &["SELECT count(*) FROM sales.orders".to_string()]
    );
}

// ---- ask ---------------------------------------------------------------

#[tokio::test]
async fn ask_without_run_only_generates_sql() {
    let llm = ScriptedLlm::with(["```sql\nSELECT 1\n```"]);
    let r = rig(true, llm, FakeStore::default(), FakeRunner::default());
    let res = r.engine.ask("q", false, false, false).await.unwrap();
    assert_eq!(res.sql, "SELECT 1");
    assert!(res.result.is_none());
    assert!(res.error.is_none());
    assert!(res.followups.is_empty());
    assert!(r.runner.ran.lock().unwrap().is_empty()); // never executed
}

#[tokio::test]
async fn ask_with_run_executes_and_returns_rows() {
    let llm = ScriptedLlm::with(["```sql\nSELECT 1\n```"]);
    let r = rig(true, llm, FakeStore::default(), FakeRunner::default());
    let res = r.engine.ask("q", true, false, false).await.unwrap();
    assert_eq!(res.result.unwrap().row_count(), 1);
    assert!(res.error.is_none());
    assert!(res.answer.is_none()); // no answer requested
}

#[tokio::test]
async fn ask_answers_over_results_when_requested() {
    // First LLM reply is the SQL, second is the written answer over the rows.
    let llm = ScriptedLlm::with([
        "```sql\nSELECT 1\n```",
        "There is 1 row; the value is 1.",
    ]);
    let r = rig(true, llm, FakeStore::default(), FakeRunner::default());
    let res = r.engine.ask("what's the count?", true, true, false).await.unwrap();
    assert_eq!(res.answer.as_deref(), Some("There is 1 row; the value is 1."));
    // Two LLM calls: generation + answer.
    assert_eq!(r.llm.seen.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn ask_skips_answer_when_query_did_not_run() {
    // answer=true but run=false -> nothing to answer over, no extra LLM call.
    let llm = ScriptedLlm::with(["```sql\nSELECT 1\n```"]);
    let r = rig(true, llm, FakeStore::default(), FakeRunner::default());
    let res = r.engine.ask("q", false, true, false).await.unwrap();
    assert!(res.answer.is_none());
    assert_eq!(r.llm.seen.lock().unwrap().len(), 1); // generation only
}

#[tokio::test]
async fn ask_captures_execution_error_instead_of_failing() {
    let llm = ScriptedLlm::with(["```sql\nSELECT 1\n```"]);
    let runner = FakeRunner { fail: true, ..Default::default() };
    let r = rig(true, llm, FakeStore::default(), runner);
    let res = r.engine.ask("q", true, false, false).await.unwrap();
    // The whole call still succeeds; the error is surfaced in the result.
    assert!(res.result.is_none());
    assert!(res.error.unwrap().contains("boom"));
    assert!(res.followups.is_empty()); // no followups when execution failed
}

#[tokio::test]
async fn ask_generates_sql_and_followups_in_one_call() {
    // With followups requested, SQL + followups come back from a SINGLE combined
    // JSON reply (a 4th followup must be dropped by take(3)).
    let llm = ScriptedLlm::with([
        r#"{"sql":"SELECT 1","followups":["What about last month?","Break it down by region","Top 5 only?","a fourth"]}"#,
    ]);
    let r = rig(true, llm, FakeStore::default(), FakeRunner::default());
    let res = r.engine.ask("q", false, false, true).await.unwrap();
    assert_eq!(res.sql, "SELECT 1");
    assert_eq!(
        res.followups,
        vec![
            "What about last month?".to_string(),
            "Break it down by region".to_string(),
            "Top 5 only?".to_string(),
        ]
    );
    // Exactly one LLM call — followups were folded in, not a second round-trip.
    assert_eq!(r.llm.seen.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn ask_followups_absent_when_model_omits_them() {
    // Combined reply without a `followups` key -> SQL still parsed, no followups.
    let llm = ScriptedLlm::with([r#"{"sql":"SELECT 1"}"#]);
    let r = rig(true, llm, FakeStore::default(), FakeRunner::default());
    let res = r.engine.ask("q", false, false, true).await.unwrap();
    assert_eq!(res.sql, "SELECT 1");
    assert!(res.followups.is_empty());
    assert!(res.error.is_none());
}

#[tokio::test]
async fn ask_combined_reply_falls_back_to_plain_sql() {
    // If the model ignores the JSON instruction and returns a fenced query,
    // we still recover the SQL (with no followups).
    let llm = ScriptedLlm::with(["```sql\nSELECT 1\n```"]);
    let r = rig(true, llm, FakeStore::default(), FakeRunner::default());
    let res = r.engine.ask("q", false, false, true).await.unwrap();
    assert_eq!(res.sql, "SELECT 1");
    assert!(res.followups.is_empty());
}

// ---- training data management -----------------------------------------

#[tokio::test]
async fn list_training_passes_through_store_rows() {
    let store = FakeStore {
        rows: vec![TrainingRow {
            id: "1".into(),
            kind: TrainingKind::Ddl,
            question: None,
            content: "CREATE TABLE t()".into(),
        }],
        ..Default::default()
    };
    let r = rig(true, ScriptedLlm::default(), store, FakeRunner::default());
    let rows = r.engine.list_training().await.unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].kind, TrainingKind::Ddl);
}

#[tokio::test]
async fn remove_training_reports_hit_and_miss() {
    let r = rig(true, ScriptedLlm::default(), FakeStore::default(), FakeRunner::default());
    assert!(r.engine.remove_training("exists").await.unwrap());
    assert!(!r.engine.remove_training("missing").await.unwrap());
    assert_eq!(
        r.store.removed.lock().unwrap().as_slice(),
        &["exists".to_string(), "missing".to_string()]
    );
}

// ---- Verbatim cache ----------------------------------------------------

#[tokio::test]
async fn ask_cache_hit_skips_generation_and_embedding() {
    // The store reports cached SQL for the question, so neither the LLM nor the
    // embedder is called — a verbatim hit short-circuits before embedding.
    let store = FakeStore { cache_hit: Some("SELECT cached".into()), ..Default::default() };
    let r = rig_cached(ScriptedLlm::default(), store);
    let res = r.engine.ask("how many?", false, false, false).await.unwrap();
    assert_eq!(res.sql, "SELECT cached");
    assert!(r.llm.seen.lock().unwrap().is_empty(), "no LLM call on a cache hit");
    assert!(r.embedder.calls.lock().unwrap().is_empty(), "no embedding on a cache hit");
}

#[tokio::test]
async fn ask_cache_miss_generates_and_stores_normalized_key() {
    // Miss -> generate, then persist under the NORMALIZED question key.
    let llm = ScriptedLlm::with(["```sql\nSELECT 1\n```"]);
    let r = rig_cached(llm, FakeStore::default());
    let res = r.engine.ask("  How   MANY?  ", false, false, false).await.unwrap();
    assert_eq!(res.sql, "SELECT 1");
    let puts = r.store.cache_puts.lock().unwrap();
    // Key is trimmed, whitespace-collapsed, lower-cased.
    assert_eq!(puts.as_slice(), &[("how many?".to_string(), "SELECT 1".to_string())]);
}

#[tokio::test]
async fn normalize_question_collapses_and_lowercases() {
    use holocron_core::engine::normalize_question;
    assert_eq!(normalize_question("  Top  10  Products "), "top 10 products");
    // Crucially, opposite questions map to DIFFERENT keys (no collision).
    assert_ne!(
        normalize_question("top 10 products by revenue"),
        normalize_question("bottom 10 products by revenue"),
    );
}

#[tokio::test]
async fn generate_sql_uses_cache() {
    let store = FakeStore { cache_hit: Some("SELECT 42".into()), ..Default::default() };
    let r = rig_cached(ScriptedLlm::default(), store);
    assert_eq!(r.engine.generate_sql("q").await.unwrap(), "SELECT 42");
    assert!(r.llm.seen.lock().unwrap().is_empty());
}

#[tokio::test]
async fn training_ddl_invalidates_cache() {
    let r = rig_cached(ScriptedLlm::default(), FakeStore::default());
    r.engine
        .train(TrainingItem::Ddl { ddl: "CREATE TABLE t (id int)".into() })
        .await
        .unwrap();
    // Adding DDL changes the schema, so the cache must be cleared.
    assert_eq!(r.store.cache_cleared.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn training_docs_does_not_invalidate_cache() {
    let r = rig_cached(ScriptedLlm::default(), FakeStore::default());
    r.engine
        .train(TrainingItem::Documentation { documentation: "note".into() })
        .await
        .unwrap();
    assert_eq!(r.store.cache_cleared.load(Ordering::Relaxed), 0);
}
