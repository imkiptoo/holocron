//! Concrete provider implementations.

#[cfg(feature = "gemini")]
pub mod gemini;

#[cfg(feature = "postgres")]
pub mod pgvector;

#[cfg(feature = "postgres")]
pub mod postgres;
