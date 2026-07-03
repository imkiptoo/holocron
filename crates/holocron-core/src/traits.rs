//! The provider abstractions the [`Engine`](crate::engine::Engine) is built on.
//!
//! These map directly onto Vanna's `VannaBase` responsibilities: an LLM to
//! generate text, an embedder for RAG, a vector store for training material,
//! and a SQL runner for the warehouse being queried.

use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::error::Result;
use crate::types::{Message, QueryResult, TrainingRow};

/// A stream of generated text deltas (for streaming chat completions).
pub type ChatStream = BoxStream<'static, Result<String>>;

/// A chat LLM (Vanna: `submit_prompt`).
#[async_trait]
pub trait Llm: Send + Sync {
    async fn chat(&self, messages: &[Message]) -> Result<String>;

    /// Stream the completion as incremental text deltas. The default collapses
    /// to a single chunk by calling [`Llm::chat`]; providers that support
    /// server-side streaming should override this.
    async fn chat_stream(&self, messages: &[Message]) -> Result<ChatStream> {
        let full = self.chat(messages).await?;
        Ok(Box::pin(futures::stream::once(async move { Ok(full) })))
    }
}

/// A text embedder (Vanna: `generate_embedding`).
#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Embed several texts at once. The default embeds them concurrently in
    /// bounded batches; providers with a batch endpoint should override this.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(8) {
            let futs: Vec<_> = chunk.iter().map(|t| self.embed(t)).collect();
            out.extend(futures::future::try_join_all(futs).await?);
        }
        Ok(out)
    }

    /// Vector dimensionality - must match the pgvector column width.
    fn dims(&self) -> usize;
}

/// The vector store holding training material
/// (Vanna: `add_ddl`/`add_documentation`/`add_question_sql` + `get_*`).
#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn add_ddl(&self, ddl: &str, embedding: Vec<f32>) -> Result<String>;
    async fn add_documentation(&self, doc: &str, embedding: Vec<f32>) -> Result<String>;
    async fn add_question_sql(
        &self,
        question: &str,
        sql: &str,
        embedding: Vec<f32>,
    ) -> Result<String>;

    async fn get_related_ddl(&self, embedding: &[f32], k: usize) -> Result<Vec<String>>;
    async fn get_related_documentation(&self, embedding: &[f32], k: usize) -> Result<Vec<String>>;
    /// Returns `(question, sql)` pairs for few-shot prompting.
    async fn get_similar_question_sql(
        &self,
        embedding: &[f32],
        k: usize,
    ) -> Result<Vec<(String, String)>>;

    /// Fetch all three retrieval buckets (DDL, docs, few-shot examples) for one
    /// question embedding. The default runs the three queries concurrently;
    /// stores that can do it in a single round-trip should override this.
    #[allow(clippy::type_complexity)]
    async fn get_context(
        &self,
        embedding: &[f32],
        k_ddl: usize,
        k_docs: usize,
        k_sql: usize,
    ) -> Result<(Vec<String>, Vec<String>, Vec<(String, String)>)> {
        tokio::try_join!(
            self.get_related_ddl(embedding, k_ddl),
            self.get_related_documentation(embedding, k_docs),
            self.get_similar_question_sql(embedding, k_sql),
        )
    }

    async fn get_training_data(&self) -> Result<Vec<TrainingRow>>;
    async fn remove_training_data(&self, id: &str) -> Result<bool>;

    // ---- Verbatim SQL cache -------------------------------------------
    //
    // A store may cache previously answered `(question, sql)` pairs keyed by the
    // **exact normalized question text**, so a repeated identical question skips
    // LLM generation. Matching is exact (not by embedding similarity) on purpose:
    // semantically-close questions like "top 10 …" vs "bottom 10 …" require
    // opposite SQL, so a similarity cache would return the wrong query. The
    // defaults make caching a no-op for stores that don't implement it.

    /// Return cached SQL for the exact `question_key` (already normalized by the
    /// caller), honouring `ttl_secs` (0 = no expiry). `None` is a cache miss.
    async fn cache_lookup(&self, _question_key: &str, _ttl_secs: u64) -> Result<Option<String>> {
        Ok(None)
    }

    /// Store SQL for the exact `question_key` (upsert on the key).
    async fn cache_put(&self, _question_key: &str, _sql: &str) -> Result<()> {
        Ok(())
    }

    /// Drop all cache entries (e.g. after the schema changes). Returns the
    /// number removed.
    async fn cache_clear(&self) -> Result<u64> {
        Ok(0)
    }
}

/// The warehouse being queried (Vanna: `run_sql` + `connect_to_postgres`).
#[async_trait]
pub trait SqlRunner: Send + Sync {
    async fn run_sql(&self, sql: &str) -> Result<QueryResult>;
    /// Emit `CREATE TABLE`-style DDL for every user table, for auto-training.
    async fn introspect_ddl(&self) -> Result<Vec<String>>;
}
