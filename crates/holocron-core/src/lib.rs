//! `holocron-core` — a text-to-SQL RAG engine (Vanna-style) built on Google
//! Gemini and PostgreSQL/pgvector.
//!
//! The [`Engine`] ties four abstractions together — an [`Llm`], an
//! [`Embedder`], a [`VectorStore`], and a [`SqlRunner`] — and exposes the
//! `train` / `generate_sql` / `ask` flow. See [`traits`] for the abstractions
//! and [`providers`] for the Gemini + Postgres implementations.

pub mod cache;
pub mod config;
pub mod engine;
pub mod error;
pub mod logging;
pub mod prompt;
pub mod providers;
pub mod sql_guard;
pub mod traits;
pub mod types;

pub use config::Config;
pub use engine::Engine;
pub use error::{Error, Result};
pub use traits::{ChatStream, Embedder, Llm, SqlRunner, VectorStore};
pub use types::{
    AskResult, Message, QueryResult, Role, TrainingItem, TrainingKind, TrainingRow,
};

/// Build the default engine (Gemini + pgvector + Postgres) from a [`Config`].
///
/// Also ensures the `training_data` table exists. This is the one-call
/// constructor the CLI and server both use.
#[cfg(all(feature = "gemini", feature = "postgres"))]
pub async fn default_engine(config: Config) -> Result<Engine> {
    use std::sync::Arc;

    use cache::CachingEmbedder;
    use providers::gemini::{default_client, shared_limiter, GeminiEmbedder, GeminiLlm};
    use providers::pgvector::PgVectorStore;
    use providers::postgres::{connect_pool_from_config, PgSqlRunner};

    tracing::info!(
        chat_model = %config.gemini_chat_model,
        embed_model = %config.gemini_embed_model,
        embed_dims = config.embed_dims,
        cache = config.cache_enabled,
        read_only = config.read_only,
        "initializing holocron engine"
    );

    // One HTTP client + one concurrency limiter shared by chat + embeddings, so
    // they pool connections and share a single outbound rate-limit budget.
    let client = default_client();
    let limiter = shared_limiter(config.gemini_max_concurrency);
    let llm = Arc::new(GeminiLlm::with_client(&config, client.clone(), limiter.clone()));
    // Wrap the embedder in an exact-match cache so repeated text isn't re-embedded.
    let embedder = Arc::new(CachingEmbedder::new(
        GeminiEmbedder::with_client(&config, client, limiter),
        1024,
    ));

    // One DB pool for the warehouse; reuse it for the vector store when both
    // point at the same database (the common single-DB setup), rather than
    // opening two independent pools to the same server.
    let shared_pool = config.vector_database_url == config.database_url;
    let warehouse_pool = connect_pool_from_config(&config, &config.database_url).await?;
    let vector_pool = if shared_pool {
        warehouse_pool.clone()
    } else {
        connect_pool_from_config(&config, &config.vector_database_url).await?
    };
    tracing::debug!(shared_pool, max_connections = config.db_max_connections, "database pools ready");

    let store = PgVectorStore::from_pool(vector_pool, config.embed_dims);
    store.init_db().await?;
    let store = Arc::new(store);

    let runner = Arc::new(PgSqlRunner::from_pool(
        warehouse_pool,
        config.max_rows,
        config.read_only || config.validate_sql,
        config.statement_timeout_secs.saturating_mul(1000),
    ));

    tracing::info!("holocron engine ready");
    Ok(Engine::new(llm, embedder, store, runner, config))
}
