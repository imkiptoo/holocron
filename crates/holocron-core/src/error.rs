//! Crate-wide error type.

use thiserror::Error;

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("configuration error: {0}")]
    Config(String),

    #[error("HTTP request to the LLM failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Gemini API returned an error: {0}")]
    Llm(String),

    #[error("database error: {0}")]
    Db(#[from] sqlx::Error),

    #[error("no SQL could be extracted from the model response")]
    NoSql,

    #[error("statement rejected in read-only mode: {0}")]
    ReadOnly(String),

    #[error("statement rejected by the SQL policy: {0}")]
    Rejected(String),

    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("{0}")]
    Other(String),
}

impl Error {
    pub fn other(msg: impl Into<String>) -> Self {
        Error::Other(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_messages_are_prefixed() {
        assert_eq!(
            Error::Config("bad".into()).to_string(),
            "configuration error: bad"
        );
        assert_eq!(
            Error::Llm("boom".into()).to_string(),
            "Gemini API returned an error: boom"
        );
        assert_eq!(
            Error::ReadOnly("DROP TABLE t".into()).to_string(),
            "statement rejected in read-only mode: DROP TABLE t"
        );
        assert_eq!(
            Error::NoSql.to_string(),
            "no SQL could be extracted from the model response"
        );
    }

    #[test]
    fn other_constructor_accepts_str_and_string() {
        assert_eq!(Error::other("x").to_string(), "x");
        assert_eq!(Error::other(String::from("y")).to_string(), "y");
    }

    #[test]
    fn serde_error_converts_via_from() {
        let serde_err = serde_json::from_str::<i32>("not json").unwrap_err();
        let err: Error = serde_err.into();
        assert!(matches!(err, Error::Serde(_)));
    }
}
