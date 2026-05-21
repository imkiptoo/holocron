//! Central logging setup — the holocron analogue of the DairyBook `./api`
//! `pkg/logger` + `clog`.
//!
//! It mirrors that style: a single entry point configured once at startup, a
//! **pretty console** format for dev and **structured JSON** for prod, RFC3339
//! timestamps, a `service` field (via a root span) on every record, and a level
//! driven by config but overridable from the environment. The Go side uses
//! zerolog; here the same shape is built on `tracing` + `tracing_subscriber`.

use std::time::Instant;

use tracing::Span;
use tracing_subscriber::fmt::time::UtcTime;
use tracing_subscriber::EnvFilter;

/// Rendering format for logs, matching `logging.format` in `./api`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// Human-friendly, colored, one line per event (dev default).
    Console,
    /// One JSON object per event (prod / log shipping).
    Json,
}

impl Format {
    /// Parse `"json"` / `"console"` (case-insensitive); anything else → console.
    pub fn parse(s: &str) -> Self {
        if s.eq_ignore_ascii_case("json") {
            Format::Json
        } else {
            Format::Console
        }
    }
}

/// Build the level filter: honour `HOLOCRON_LOG` (or `RUST_LOG`) if set,
/// otherwise apply `level` to the holocron crates (and warn for everything else).
fn env_filter(level: &str) -> EnvFilter {
    EnvFilter::try_from_env("HOLOCRON_LOG")
        .or_else(|_| EnvFilter::try_from_default_env())
        .unwrap_or_else(|_| {
            EnvFilter::new(format!(
                "holocron_core={level},holocron_cli={level},holocron_server={level},\
                 holocron_grpc={level},tower_http={level},warn"
            ))
        })
}

/// Initialize the global logger. Call **once** at process start, after loading
/// config. `level` is `debug|info|warn|error`; `format` selects console vs JSON.
///
/// Uses `try_init`, so a redundant call (e.g. across tests) is a no-op rather
/// than a panic.
pub fn init(level: &str, format: Format) {
    let filter = env_filter(level);
    // RFC3339 UTC timestamps, matching the Go logger's `time.RFC3339`.
    let timer = UtcTime::rfc_3339();

    match format {
        Format::Json => {
            let _ = tracing_subscriber::fmt()
                .json()
                .flatten_event(true)
                .with_timer(timer)
                .with_env_filter(filter)
                .try_init();
        }
        Format::Console => {
            let _ = tracing_subscriber::fmt()
                .with_timer(timer)
                .with_target(false)
                .with_env_filter(filter)
                .try_init();
        }
    }
}

/// A process-root span that stamps `service = <name>` (e.g. `"holocron-server"`)
/// onto every event emitted while it is entered — the analogue of the Go
/// logger's `.Str("service", ...)`. Enter it for the process lifetime, or
/// `.instrument()` the main future with it in async binaries.
pub fn service_span(service: &'static str) -> Span {
    tracing::info_span!("holocron", service = service)
}

/// Convenience: initialize logging and return the [`service_span`] to enter.
pub fn init_service(level: &str, format: Format, service: &'static str) -> Span {
    init(level, format);
    service_span(service)
}

// ---- Timing ------------------------------------------------------------

/// Milliseconds elapsed since `start`, as a float so sub-millisecond steps
/// (retrieval, cache lookups) still show a non-zero duration.
pub fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

/// A drop guard that logs, at **info**, how long a request took from
/// construction to drop — "from the moment it came in to the moment we
/// processed it". Because it logs on `Drop`, it also fires on early returns and
/// errors. Put one at each request boundary (gRPC RPC / HTTP handler).
///
/// The elapsed time is emitted as the `elapsed_ms` field, within whatever
/// span is current (e.g. the per-RPC span), so it inherits `service` etc.
pub struct RequestTimer {
    label: &'static str,
    start: Instant,
}

impl RequestTimer {
    /// Start timing a request labelled `label` (e.g. `"grpc.ask"`).
    pub fn new(label: &'static str) -> Self {
        Self { label, start: Instant::now() }
    }

    /// Milliseconds elapsed so far.
    pub fn elapsed_ms(&self) -> f64 {
        elapsed_ms(self.start)
    }
}

impl Drop for RequestTimer {
    fn drop(&mut self) {
        tracing::info!(elapsed_ms = self.elapsed_ms(), "{} processed", self.label);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_parse_is_case_insensitive() {
        assert_eq!(Format::parse("json"), Format::Json);
        assert_eq!(Format::parse("JSON"), Format::Json);
        assert_eq!(Format::parse("console"), Format::Console);
        assert_eq!(Format::parse("anything-else"), Format::Console);
    }

    #[test]
    fn init_is_safe_to_call_twice() {
        // try_init means neither call panics even with a global subscriber set.
        init("info", Format::Console);
        init("debug", Format::Json);
    }
}
