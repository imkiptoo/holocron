//! Runtime configuration, loaded from a `holocron.toml` file.
//!
//! The on-disk format is grouped into `[gemini]`, `[database]`, `[retrieval]`,
//! `[safety]`, and `[server]` tables; see `holocron.toml.example`. Everything
//! except the Gemini API key and the database URL has a default.

use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};

/// Fully-resolved configuration used by the engine and providers.
#[derive(Debug, Clone)]
pub struct Config {
    // Gemini
    pub gemini_api_key: String,
    pub gemini_chat_model: String,
    pub gemini_embed_model: String,
    pub embed_dims: usize,
    /// Max concurrent outbound Gemini calls (rate-limit governor).
    pub gemini_max_concurrency: usize,
    /// Retries on transient Gemini errors (429/503) before giving up.
    pub gemini_max_retries: u32,

    // Databases
    pub database_url: String,
    pub vector_database_url: String,
    /// Connection-pool ceiling per database.
    pub db_max_connections: u32,
    /// How long to wait for a pooled connection before erroring.
    pub db_acquire_timeout_secs: u64,

    // Retrieval top-k
    pub top_k_ddl: usize,
    pub top_k_docs: usize,
    pub top_k_sql: usize,

    // Safety
    pub read_only: bool,
    /// Cap on rows returned by `run_sql` (0 = unlimited).
    pub max_rows: usize,
    /// Run the AST validation gate (`sql_guard`) before executing generated SQL.
    pub validate_sql: bool,
    /// Allow generated SQL to reference `information_schema` / `pg_catalog`.
    pub allow_system_schemas: bool,
    /// Per-query `statement_timeout` in seconds (0 = no limit).
    pub statement_timeout_secs: u64,

    // Verbatim SQL cache (exact match on the normalized question)
    pub cache_enabled: bool,
    /// Cache entry lifetime in seconds (0 = no expiry).
    pub cache_ttl_secs: u64,

    // Server
    pub bind_addr: String,
    /// Per-request timeout on the HTTP server.
    pub request_timeout_secs: u64,
    /// Max in-flight requests before load-shedding (503).
    pub max_concurrent_requests: usize,

    // Logging
    /// Minimum level: `debug|info|warn|error` (overridable via `HOLOCRON_LOG`).
    pub log_level: String,
    /// Output format: `console` (pretty, dev) or `json` (structured, prod).
    pub log_format: String,
}

impl Config {
    /// Load config from the path in `HOLOCRON_CONFIG`, or `./holocron.toml` by default.
    pub fn load() -> Result<Self> {
        let path = std::env::var("HOLOCRON_CONFIG").unwrap_or_else(|_| "holocron.toml".to_string());
        Self::from_path(path)
    }

    /// Load and resolve config from a specific TOML file.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::Config(format!("cannot read {}: {e}", path.display())))?;
        let file: FileConfig = toml::from_str(&text)
            .map_err(|e| Error::Config(format!("invalid config {}: {e}", path.display())))?;
        Ok(file.resolve())
    }
}

// ---- On-disk representation --------------------------------------------

#[derive(Deserialize)]
struct FileConfig {
    gemini: GeminiSection,
    database: DatabaseSection,
    #[serde(default)]
    retrieval: RetrievalSection,
    #[serde(default)]
    safety: SafetySection,
    #[serde(default)]
    cache: CacheSection,
    #[serde(default)]
    server: ServerSection,
    #[serde(default)]
    logging: LoggingSection,
}

impl FileConfig {
    fn resolve(self) -> Config {
        let vector_database_url = self
            .database
            .vector_url
            .unwrap_or_else(|| self.database.url.clone());
        Config {
            gemini_api_key: self.gemini.api_key,
            gemini_chat_model: self.gemini.chat_model,
            gemini_embed_model: self.gemini.embed_model,
            embed_dims: self.gemini.embed_dims,
            gemini_max_concurrency: self.gemini.max_concurrency,
            gemini_max_retries: self.gemini.max_retries,
            database_url: self.database.url,
            vector_database_url,
            db_max_connections: self.database.max_connections,
            db_acquire_timeout_secs: self.database.acquire_timeout_secs,
            top_k_ddl: self.retrieval.top_k_ddl,
            top_k_docs: self.retrieval.top_k_docs,
            top_k_sql: self.retrieval.top_k_sql,
            read_only: self.safety.read_only,
            max_rows: self.safety.max_rows,
            validate_sql: self.safety.validate_sql,
            allow_system_schemas: self.safety.allow_system_schemas,
            statement_timeout_secs: self.safety.statement_timeout_secs,
            cache_enabled: self.cache.enabled,
            cache_ttl_secs: self.cache.ttl_secs,
            bind_addr: self.server.bind_addr,
            request_timeout_secs: self.server.request_timeout_secs,
            max_concurrent_requests: self.server.max_concurrent_requests,
            log_level: self.logging.level,
            log_format: self.logging.format,
        }
    }
}

#[derive(Deserialize)]
struct GeminiSection {
    api_key: String,
    #[serde(default = "default_chat_model")]
    chat_model: String,
    #[serde(default = "default_embed_model")]
    embed_model: String,
    #[serde(default = "default_embed_dims")]
    embed_dims: usize,
    #[serde(default = "default_max_concurrency")]
    max_concurrency: usize,
    #[serde(default = "default_max_retries")]
    max_retries: u32,
}

#[derive(Deserialize)]
struct DatabaseSection {
    url: String,
    #[serde(default)]
    vector_url: Option<String>,
    #[serde(default = "default_max_connections")]
    max_connections: u32,
    #[serde(default = "default_acquire_timeout_secs")]
    acquire_timeout_secs: u64,
}

#[derive(Deserialize)]
struct RetrievalSection {
    #[serde(default = "default_top_k")]
    top_k_ddl: usize,
    #[serde(default = "default_top_k")]
    top_k_docs: usize,
    #[serde(default = "default_top_k")]
    top_k_sql: usize,
}

#[derive(Deserialize)]
struct SafetySection {
    #[serde(default = "default_read_only")]
    read_only: bool,
    #[serde(default = "default_max_rows")]
    max_rows: usize,
    #[serde(default = "default_validate_sql")]
    validate_sql: bool,
    #[serde(default = "default_allow_system_schemas")]
    allow_system_schemas: bool,
    #[serde(default = "default_statement_timeout_secs")]
    statement_timeout_secs: u64,
}

#[derive(Deserialize)]
struct CacheSection {
    #[serde(default = "default_cache_enabled")]
    enabled: bool,
    #[serde(default = "default_cache_ttl_secs")]
    ttl_secs: u64,
}

#[derive(Deserialize)]
struct ServerSection {
    #[serde(default = "default_bind_addr")]
    bind_addr: String,
    #[serde(default = "default_request_timeout_secs")]
    request_timeout_secs: u64,
    #[serde(default = "default_max_concurrent_requests")]
    max_concurrent_requests: usize,
}

#[derive(Deserialize)]
struct LoggingSection {
    #[serde(default = "default_log_level")]
    level: String,
    #[serde(default = "default_log_format")]
    format: String,
}

// Section defaults (used when a whole `[table]` is omitted from the file).
impl Default for RetrievalSection {
    fn default() -> Self {
        Self { top_k_ddl: default_top_k(), top_k_docs: default_top_k(), top_k_sql: default_top_k() }
    }
}
impl Default for SafetySection {
    fn default() -> Self {
        Self {
            read_only: default_read_only(),
            max_rows: default_max_rows(),
            validate_sql: default_validate_sql(),
            allow_system_schemas: default_allow_system_schemas(),
            statement_timeout_secs: default_statement_timeout_secs(),
        }
    }
}
impl Default for CacheSection {
    fn default() -> Self {
        Self { enabled: default_cache_enabled(), ttl_secs: default_cache_ttl_secs() }
    }
}
impl Default for LoggingSection {
    fn default() -> Self {
        Self { level: default_log_level(), format: default_log_format() }
    }
}
impl Default for ServerSection {
    fn default() -> Self {
        Self {
            bind_addr: default_bind_addr(),
            request_timeout_secs: default_request_timeout_secs(),
            max_concurrent_requests: default_max_concurrent_requests(),
        }
    }
}

fn default_chat_model() -> String {
    "gemini-2.5-flash".to_string()
}
fn default_embed_model() -> String {
    "text-embedding-004".to_string()
}
fn default_embed_dims() -> usize {
    768
}
fn default_max_concurrency() -> usize {
    8
}
fn default_max_retries() -> u32 {
    3
}
fn default_max_connections() -> u32 {
    10
}
fn default_acquire_timeout_secs() -> u64 {
    30
}
fn default_top_k() -> usize {
    5
}
fn default_read_only() -> bool {
    true
}
fn default_max_rows() -> usize {
    10_000
}
fn default_validate_sql() -> bool {
    true
}
fn default_allow_system_schemas() -> bool {
    false
}
fn default_statement_timeout_secs() -> u64 {
    30
}
fn default_cache_enabled() -> bool {
    true
}
fn default_cache_ttl_secs() -> u64 {
    3600
}
fn default_bind_addr() -> String {
    "127.0.0.1:8080".to_string()
}
fn default_request_timeout_secs() -> u64 {
    120
}
fn default_max_concurrent_requests() -> usize {
    64
}
fn default_log_level() -> String {
    "info".to_string()
}
fn default_log_format() -> String {
    "console".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Write `body` to a unique temp file and return its path.
    fn write_tmp(body: &str) -> std::path::PathBuf {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "holocron-config-test-{}-{n}.toml",
            std::process::id()
        ));
        std::fs::write(&path, body).unwrap();
        path
    }

    #[test]
    fn full_config_parses_all_fields() {
        let path = write_tmp(
            r#"
            [gemini]
            api_key = "KEY"
            chat_model = "custom-chat"
            embed_model = "custom-embed"
            embed_dims = 1024
            max_concurrency = 4
            max_retries = 2

            [database]
            url = "postgres://warehouse"
            vector_url = "postgres://vectors"
            max_connections = 25
            acquire_timeout_secs = 7

            [retrieval]
            top_k_ddl = 1
            top_k_docs = 2
            top_k_sql = 3

            [safety]
            read_only = false
            max_rows = 500
            validate_sql = false
            allow_system_schemas = true
            statement_timeout_secs = 9

            [cache]
            enabled = false
            ttl_secs = 60

            [server]
            bind_addr = "0.0.0.0:9000"
            request_timeout_secs = 15
            max_concurrent_requests = 128

            [logging]
            level = "debug"
            format = "json"
            "#,
        );
        let c = Config::from_path(&path).unwrap();
        assert_eq!(c.gemini_api_key, "KEY");
        assert_eq!(c.gemini_chat_model, "custom-chat");
        assert_eq!(c.gemini_embed_model, "custom-embed");
        assert_eq!(c.embed_dims, 1024);
        assert_eq!(c.gemini_max_concurrency, 4);
        assert_eq!(c.gemini_max_retries, 2);
        assert_eq!(c.database_url, "postgres://warehouse");
        assert_eq!(c.vector_database_url, "postgres://vectors");
        assert_eq!(c.db_max_connections, 25);
        assert_eq!(c.db_acquire_timeout_secs, 7);
        assert_eq!(c.top_k_ddl, 1);
        assert_eq!(c.top_k_docs, 2);
        assert_eq!(c.top_k_sql, 3);
        assert!(!c.read_only);
        assert_eq!(c.max_rows, 500);
        assert!(!c.validate_sql);
        assert!(c.allow_system_schemas);
        assert_eq!(c.statement_timeout_secs, 9);
        assert!(!c.cache_enabled);
        assert_eq!(c.cache_ttl_secs, 60);
        assert_eq!(c.bind_addr, "0.0.0.0:9000");
        assert_eq!(c.request_timeout_secs, 15);
        assert_eq!(c.max_concurrent_requests, 128);
        assert_eq!(c.log_level, "debug");
        assert_eq!(c.log_format, "json");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn minimal_config_applies_defaults() {
        // Only the two required fields; everything else defaulted.
        let path = write_tmp(
            r#"
            [gemini]
            api_key = "KEY"

            [database]
            url = "postgres://warehouse"
            "#,
        );
        let c = Config::from_path(&path).unwrap();
        assert_eq!(c.gemini_chat_model, "gemini-2.5-flash");
        assert_eq!(c.gemini_embed_model, "text-embedding-004");
        assert_eq!(c.embed_dims, 768);
        assert_eq!(c.gemini_max_concurrency, 8);
        assert_eq!(c.gemini_max_retries, 3);
        assert_eq!(c.db_max_connections, 10);
        assert_eq!(c.db_acquire_timeout_secs, 30);
        assert_eq!(c.top_k_ddl, 5);
        assert_eq!(c.top_k_docs, 5);
        assert_eq!(c.top_k_sql, 5);
        assert!(c.read_only); // read-only defaults to true (safe)
        assert_eq!(c.max_rows, 10_000);
        assert!(c.validate_sql); // AST gate on by default
        assert!(!c.allow_system_schemas); // system schemas denied by default
        assert_eq!(c.statement_timeout_secs, 30);
        assert!(c.cache_enabled);
        assert_eq!(c.cache_ttl_secs, 3600);
        assert_eq!(c.bind_addr, "127.0.0.1:8080");
        assert_eq!(c.request_timeout_secs, 120);
        assert_eq!(c.max_concurrent_requests, 64);
        assert_eq!(c.log_level, "info");
        assert_eq!(c.log_format, "console");
        // vector_url falls back to the warehouse url.
        assert_eq!(c.vector_database_url, "postgres://warehouse");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn vector_url_defaults_to_database_url() {
        let path = write_tmp(
            r#"
            [gemini]
            api_key = "K"
            [database]
            url = "postgres://only"
            "#,
        );
        let c = Config::from_path(&path).unwrap();
        assert_eq!(c.database_url, c.vector_database_url);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_file_is_a_config_error() {
        let err = Config::from_path("/no/such/holocron.toml").unwrap_err();
        assert!(matches!(err, Error::Config(_)));
        assert!(err.to_string().contains("cannot read"));
    }

    #[test]
    fn invalid_toml_is_a_config_error() {
        let path = write_tmp("this is not = valid toml [[[");
        let err = Config::from_path(&path).unwrap_err();
        assert!(matches!(err, Error::Config(_)));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_required_field_errors() {
        // No api_key -> deserialization fails -> Config error.
        let path = write_tmp(
            r#"
            [gemini]
            chat_model = "x"
            [database]
            url = "u"
            "#,
        );
        assert!(Config::from_path(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
