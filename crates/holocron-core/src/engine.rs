//! The engine - the Rust analogue of Vanna's `VannaBase`.
//!
//! It wires an [`Llm`], [`Embedder`], [`VectorStore`], and [`SqlRunner`]
//! together and exposes the high-level `train` / `generate_sql` / `ask` flow.

use std::sync::Arc;
use std::time::Instant;

use tracing::{debug, info, instrument, warn};

use crate::config::Config;
use crate::error::{Error, Result};
use crate::logging::elapsed_ms;
use crate::prompt::{self, build_sql_prompt, extract_sql};
use crate::traits::{ChatStream, Embedder, Llm, SqlRunner, VectorStore};
use crate::types::{AskResult, Message, QueryResult, TrainingItem, TrainingRow};

/// Normalize a question into the verbatim-cache key: trimmed, lower-cased, with
/// internal whitespace collapsed. This is deliberately conservative - only
/// truly identical questions (modulo casing/spacing) share a cache entry, so
/// "top 10 …" and "bottom 10 …" never collide.
pub fn normalize_question(question: &str) -> String {
    question.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

#[derive(Clone)]
pub struct Engine {
    llm: Arc<dyn Llm>,
    embedder: Arc<dyn Embedder>,
    store: Arc<dyn VectorStore>,
    runner: Arc<dyn SqlRunner>,
    config: Config,
}

impl Engine {
    pub fn new(
        llm: Arc<dyn Llm>,
        embedder: Arc<dyn Embedder>,
        store: Arc<dyn VectorStore>,
        runner: Arc<dyn SqlRunner>,
        config: Config,
    ) -> Self {
        Self { llm, embedder, store, runner, config }
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    // ---- Training -------------------------------------------------------

    /// Add one piece of training material. For a `Sql` item without a
    /// question, the LLM synthesizes one first (Vanna's `generate_question`).
    ///
    /// Adding DDL changes the schema, which can invalidate cached SQL, so the
    /// semantic cache is cleared on DDL training.
    #[instrument(skip_all)]
    pub async fn train(&self, item: TrainingItem) -> Result<String> {
        match item {
            TrainingItem::Ddl { ddl } => {
                let emb = self.embedder.embed(&ddl).await?;
                let id = self.store.add_ddl(&ddl, emb).await?;
                info!(%id, "trained DDL");
                self.invalidate_cache().await;
                Ok(id)
            }
            TrainingItem::Documentation { documentation } => {
                let emb = self.embedder.embed(&documentation).await?;
                let id = self.store.add_documentation(&documentation, emb).await?;
                info!(%id, "trained documentation");
                Ok(id)
            }
            TrainingItem::Sql { question, sql } => {
                let question = match question {
                    Some(q) => q,
                    None => {
                        debug!("no question supplied; synthesizing one from the SQL");
                        self.generate_question(&sql).await?
                    }
                };
                // Embed on the question - that's what future questions match against.
                let emb = self.embedder.embed(&question).await?;
                let id = self.store.add_question_sql(&question, &sql, emb).await?;
                info!(%id, "trained question/SQL pair");
                Ok(id)
            }
        }
    }

    /// Auto-train from the warehouse's `information_schema`
    /// (Vanna's `get_training_plan_generic`). Returns the number of DDL rows added.
    #[instrument(skip_all)]
    pub async fn train_from_information_schema(&self) -> Result<usize> {
        let ddls = self.runner.introspect_ddl().await?;
        let count = ddls.len();
        if count == 0 {
            info!("no user tables found to auto-train");
            return Ok(0);
        }
        debug!(tables = count, "auto-training tables from information_schema");
        // Embed every table in one batch call, then store the rows.
        let embeddings = self.embedder.embed_batch(&ddls).await?;
        for (ddl, emb) in ddls.iter().zip(embeddings) {
            self.store.add_ddl(ddl, emb).await?;
        }
        self.invalidate_cache().await;
        info!(tables = count, "auto-trained tables from information_schema");
        Ok(count)
    }

    /// Drop the semantic cache (best-effort) after the schema changes.
    async fn invalidate_cache(&self) {
        if self.config.cache_enabled {
            match self.store.cache_clear().await {
                Ok(n) if n > 0 => debug!(cleared = n, "semantic cache invalidated"),
                Ok(_) => {}
                Err(e) => warn!(error = %e, "failed to clear semantic cache"),
            }
        }
    }

    /// Ask the LLM to describe what a SQL query answers, in one line.
    async fn generate_question(&self, sql: &str) -> Result<String> {
        let messages = vec![
            Message::system(
                "You write a single, concise natural-language question that the \
                 given SQL query answers. Reply with the question only - no quotes, \
                 no preamble.",
            ),
            Message::user(sql.to_string()),
        ];
        let reply = self.llm.chat(&messages).await?;
        Ok(reply.trim().trim_matches('"').to_string())
    }

    // ---- Retrieval + generation ----------------------------------------

    /// The RAG core: check the verbatim cache, else embed + retrieve + prompt
    /// the LLM. Cache-aware.
    #[instrument(skip_all, fields(q = %question))]
    pub async fn generate_sql(&self, question: &str) -> Result<String> {
        let (sql, _followups, _cached) = self.produce_sql(question, false).await?;
        Ok(sql)
    }

    /// Generate SQL with caller-supplied schema context and conversation
    /// history. This is the drop-in for the DairyBook Go API's `Generator`
    /// (formerly the Python Vanna sidecar): `extra_context` is the role-scoped
    /// curated-view DDL, `history` the recent turns. Bypasses the semantic cache
    /// because the effective schema/history varies per caller/role.
    #[instrument(skip_all, fields(q = %question))]
    pub async fn generate_sql_with(
        &self,
        question: &str,
        extra_context: &[String],
        history: &[Message],
    ) -> Result<String> {
        let q_emb = self.embed_timed(question).await?;
        let (retrieved, docs, examples) = self.retrieve(&q_emb).await?;
        // Prepend the caller's schema context ahead of holocron's own retrieval.
        let ddls: Vec<String> = if extra_context.is_empty() {
            retrieved
        } else {
            extra_context.iter().cloned().chain(retrieved).collect()
        };
        let mut messages = build_sql_prompt(question, &ddls, &docs, &examples);
        if !history.is_empty() {
            // Splice prior turns in just before the final user question.
            let question_msg = messages.pop().expect("prompt ends with the question");
            messages.extend_from_slice(history);
            messages.push(question_msg);
        }
        let reply = self.llm.chat(&messages).await?;
        extract_sql(&reply).ok_or(Error::NoSql)
    }

    /// Stream the raw model completion for a text-to-SQL request (no cache, no
    /// extraction - the caller sees the tokens as they arrive).
    #[instrument(skip_all, fields(q = %question))]
    pub async fn generate_sql_stream(&self, question: &str) -> Result<ChatStream> {
        let q_emb = self.embed_timed(question).await?;
        let (ddls, docs, examples) = self.retrieve(&q_emb).await?;
        let messages = build_sql_prompt(question, &ddls, &docs, &examples);
        self.llm.chat_stream(&messages).await
    }

    /// Embed `text`, logging how long the embedding call took (debug).
    async fn embed_timed(&self, text: &str) -> Result<Vec<f32>> {
        let t = Instant::now();
        let emb = self.embedder.embed(text).await?;
        debug!(elapsed_ms = elapsed_ms(t), "embedded text");
        Ok(emb)
    }

    /// Retrieve the three RAG buckets for a question embedding (one round-trip
    /// where the store supports it).
    async fn retrieve(
        &self,
        q_emb: &[f32],
    ) -> Result<(Vec<String>, Vec<String>, Vec<(String, String)>)> {
        let t = Instant::now();
        let ctx = self
            .store
            .get_context(q_emb, self.config.top_k_ddl, self.config.top_k_docs, self.config.top_k_sql)
            .await?;
        debug!(
            ddl = ctx.0.len(),
            docs = ctx.1.len(),
            examples = ctx.2.len(),
            elapsed_ms = elapsed_ms(t),
            "retrieved RAG context"
        );
        Ok(ctx)
    }

    /// Produce SQL for a question: check the verbatim cache first (exact match
    /// on the normalized question), and on a miss embed + generate (optionally
    /// folding follow-ups into the same call) and cache the result. Returns
    /// `(sql, followups, from_cache)`.
    ///
    /// The match is exact, not by embedding similarity: "top 10 …" and
    /// "bottom 10 …" are near-identical vectors but need opposite SQL, so a
    /// similarity cache would answer one with the other's query.
    async fn produce_sql(
        &self,
        question: &str,
        want_followups: bool,
    ) -> Result<(String, Vec<String>, bool)> {
        let key = normalize_question(question);

        if self.config.cache_enabled {
            match self.store.cache_lookup(&key, self.config.cache_ttl_secs).await {
                Ok(Some(sql)) => {
                    info!("cache hit (verbatim) - skipping generation");
                    return Ok((sql, Vec::new(), true));
                }
                Ok(None) => debug!("cache miss; generating"),
                Err(e) => warn!(error = %e, "cache lookup failed; generating"),
            }
        }

        // Cache miss: only now do we pay for the embedding + generation.
        let q_emb = self.embed_timed(question).await?;
        let (sql, followups) = self.generate_full(question, &q_emb, want_followups).await?;

        if self.config.cache_enabled {
            match self.store.cache_put(&key, &sql).await {
                Ok(()) => debug!("cached generated SQL for the exact question"),
                Err(e) => warn!(error = %e, "cache put failed"),
            }
        }
        Ok((sql, followups, false))
    }

    /// Retrieve context and prompt the LLM. When `want_followups`, uses the
    /// combined `{sql, followups}` prompt so a single call yields both.
    async fn generate_full(
        &self,
        question: &str,
        q_emb: &[f32],
        want_followups: bool,
    ) -> Result<(String, Vec<String>)> {
        let (ddls, docs, examples) = self.retrieve(q_emb).await?;
        if want_followups {
            let messages = prompt::build_combined_prompt(question, &ddls, &docs, &examples);
            let t = Instant::now();
            let reply = self.llm.chat(&messages).await?;
            let llm_ms = elapsed_ms(t);
            let (sql, followups) = prompt::extract_combined(&reply);
            let sql = sql.ok_or(Error::NoSql).inspect_err(|_| warn!("no SQL in combined reply"))?;
            debug!(sql = %sql, followups = followups.len(), llm_ms, "generated SQL (with follow-ups)");
            Ok((sql, followups))
        } else {
            let messages = build_sql_prompt(question, &ddls, &docs, &examples);
            let t = Instant::now();
            let reply = self.llm.chat(&messages).await?;
            let llm_ms = elapsed_ms(t);
            let sql = extract_sql(&reply).ok_or(Error::NoSql).inspect_err(|_| warn!("no SQL in model reply"))?;
            debug!(sql = %sql, llm_ms, "generated SQL");
            Ok((sql, Vec::new()))
        }
    }

    // ---- Execution ------------------------------------------------------

    /// Run SQL against the warehouse. Generated SQL is untrusted, so it passes
    /// the AST validation gate (`sql_guard`) first - a single read-only SELECT
    /// touching only permitted objects. (The runner additionally executes inside
    /// a `READ ONLY` transaction with a `statement_timeout` as defense in depth.)
    #[instrument(skip_all)]
    pub async fn run_sql(&self, sql: &str) -> Result<QueryResult> {
        if self.config.validate_sql {
            let policy = crate::sql_guard::SqlPolicy {
                allow_system_schemas: self.config.allow_system_schemas,
            };
            if let Err(reason) = crate::sql_guard::validate(sql, &policy) {
                warn!(%reason, "rejected generated SQL");
                return Err(Error::Rejected(reason));
            }
        } else if self.config.read_only && !prompt::is_read_only(sql) {
            warn!(sql = %sql, "rejected non-read-only statement in read-only mode");
            return Err(Error::ReadOnly(sql.to_string()));
        }
        let t = Instant::now();
        let result = self.runner.run_sql(sql).await?;
        debug!(
            rows = result.row_count(),
            cols = result.columns.len(),
            elapsed_ms = elapsed_ms(t),
            "executed SQL"
        );
        Ok(result)
    }

    /// Full pipeline: question -> SQL -> (optional) run -> (optional)
    /// natural-language answer over the rows -> (optional) followups. `answer`
    /// closes Vanna's loop: after running, the model phrases the result back
    /// into a written answer/insight (or the format the question asked for).
    #[instrument(skip_all, fields(q = %question, run, answer, followups))]
    pub async fn ask(
        &self,
        question: &str,
        run: bool,
        answer: bool,
        followups: bool,
    ) -> Result<AskResult> {
        let started = Instant::now();
        // On a cache miss with followups requested, this folds SQL + followups
        // into one LLM call. On a cache hit, `from_cache` is true and followups
        // (if wanted) are generated separately below.
        let (sql, generated_followups, from_cache) =
            self.produce_sql(question, followups).await?;

        let (result, error) = if run {
            match self.run_sql(&sql).await {
                Ok(r) => (Some(r), None),
                Err(e) => {
                    warn!(error = %e, "generated SQL failed to run");
                    (None, Some(e.to_string()))
                }
            }
        } else {
            (None, None)
        };

        // Answer over the rows (best-effort - a failed answer doesn't fail ask).
        let answer = match (answer, &result) {
            (true, Some(qr)) => match self.answer(question, qr).await {
                Ok(a) => Some(a),
                Err(e) => {
                    warn!(error = %e, "answer generation failed");
                    None
                }
            },
            _ => None,
        };

        let followups = if followups && error.is_none() {
            if from_cache {
                self.generate_followups(question, &sql).await.unwrap_or_default()
            } else {
                generated_followups
            }
        } else {
            Vec::new()
        };

        info!(
            cached = from_cache,
            executed = run,
            answered = answer.is_some(),
            errored = error.is_some(),
            followups = followups.len(),
            elapsed_ms = elapsed_ms(started),
            "answered question"
        );
        Ok(AskResult {
            question: question.to_string(),
            sql,
            result,
            error,
            answer,
            followups,
        })
    }

    /// Phrase a query result back into a written answer to the question
    /// (Vanna's `generate_summary`). Prose by default; honours an explicit
    /// output format requested in the question.
    async fn answer(&self, question: &str, result: &QueryResult) -> Result<String> {
        let t = Instant::now();
        let messages = prompt::build_answer_prompt(question, result);
        let reply = self.llm.chat(&messages).await?;
        debug!(elapsed_ms = elapsed_ms(t), "generated answer");
        Ok(reply.trim().to_string())
    }

    /// Suggest follow-up questions a user might ask next.
    async fn generate_followups(&self, question: &str, sql: &str) -> Result<Vec<String>> {
        let messages = vec![
            Message::system(
                "Given a user's question and the SQL that answered it, suggest three \
                 concise follow-up questions the user might ask next. Reply with one \
                 question per line, no numbering or bullets.",
            ),
            Message::user(format!("Question: {question}\nSQL:\n{sql}")),
        ];
        let reply = self.llm.chat(&messages).await?;
        Ok(reply
            .lines()
            .map(|l| l.trim().trim_start_matches(['-', '*', '•']).trim())
            .filter(|l| !l.is_empty())
            .take(3)
            .map(|s| s.to_string())
            .collect())
    }

    // ---- Training data management --------------------------------------

    pub async fn list_training(&self) -> Result<Vec<TrainingRow>> {
        self.store.get_training_data().await
    }

    pub async fn remove_training(&self, id: &str) -> Result<bool> {
        self.store.remove_training_data(id).await
    }
}
