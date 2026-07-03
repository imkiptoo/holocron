//! `holocron-server` - a small HTTP API over the engine, mirroring Vanna's Flask
//! endpoints (ask / generate_sql / run_sql / train / training_data).

use std::time::Duration;

use axum::error_handling::HandleErrorLayer;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{BoxError, Json, Router};
use futures::{Stream, StreamExt};
use serde::Deserialize;
use serde_json::json;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::Instrument;

use holocron_core::logging::{self, Format};
use holocron_core::{default_engine, Config, Engine, TrainingItem};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = Config::load()?;
    // Configure logging (console pretty / JSON, RFC3339, leveled) from config,
    // matching the DairyBook `./api` style; stamp `service` via a root span.
    let span = logging::init_service(&config.log_level, Format::parse(&config.log_format), "holocron-server");

    async move {
        let bind_addr = config.bind_addr.clone();
        let max_concurrent = config.max_concurrent_requests;
        let request_timeout = Duration::from_secs(config.request_timeout_secs);
        let engine = default_engine(config).await?;

        // Load-shedding governor: cap in-flight requests and time each one out.
        // When at capacity the load-shed layer returns 503 instead of queueing;
        // `HandleErrorLayer` maps the shed/timeout errors back into HTTP responses
        // so the stack stays infallible.
        let governor = ServiceBuilder::new()
            .layer(HandleErrorLayer::new(handle_middleware_error))
            .load_shed()
            .concurrency_limit(max_concurrent)
            .timeout(request_timeout);

        let app = Router::new()
            .route("/api/health", get(health))
            .route("/api/ask", post(ask))
            .route("/api/generate_sql", post(generate_sql))
            .route("/api/generate_sql/stream", post(generate_sql_stream))
            .route("/api/run_sql", post(run_sql))
            .route("/api/train", post(train))
            .route("/api/training_data", get(list_training))
            .route("/api/training_data/{id}", delete(remove_training))
            .layer(governor)
            .layer(TraceLayer::new_for_http())
            .layer(CorsLayer::permissive())
            .with_state(engine);

        let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
        tracing::info!("holocron-server listening on http://{bind_addr}");
        axum::serve(listener, app).await?;
        Ok(())
    }
    .instrument(span)
    .await
}

/// Turn load-shed / timeout middleware errors into HTTP status codes.
async fn handle_middleware_error(err: BoxError) -> (StatusCode, String) {
    if err.is::<tower::load_shed::error::Overloaded>() {
        (StatusCode::SERVICE_UNAVAILABLE, "server overloaded, try again".into())
    } else if err.is::<tower::timeout::error::Elapsed>() {
        (StatusCode::REQUEST_TIMEOUT, "request timed out".into())
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("middleware error: {err}"))
    }
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({ "status": "ok" }))
}

#[derive(Deserialize)]
struct AskBody {
    question: String,
    #[serde(default = "default_true")]
    run: bool,
    /// Phrase the result back into a written answer (defaults on).
    #[serde(default = "default_true")]
    answer: bool,
    #[serde(default)]
    followups: bool,
}

fn default_true() -> bool {
    true
}

async fn ask(
    State(engine): State<Engine>,
    Json(body): Json<AskBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    let _timer = logging::RequestTimer::new("http.ask");
    tracing::debug!(
        question = %body.question,
        run = body.run,
        answer = body.answer,
        followups = body.followups,
        "POST /api/ask"
    );
    let result = engine.ask(&body.question, body.run, body.answer, body.followups).await?;
    Ok(Json(serde_json::to_value(result)?))
}

#[derive(Deserialize)]
struct QuestionBody {
    question: String,
}

async fn generate_sql(
    State(engine): State<Engine>,
    Json(body): Json<QuestionBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    let _timer = logging::RequestTimer::new("http.generate_sql");
    tracing::debug!(question = %body.question, "POST /api/generate_sql");
    let sql = engine.generate_sql(&body.question).await?;
    Ok(Json(json!({ "sql": sql })))
}

/// Stream the model's SQL generation token-by-token as Server-Sent Events,
/// for low time-to-first-token in interactive clients.
async fn generate_sql_stream(
    State(engine): State<Engine>,
    Json(body): Json<QuestionBody>,
) -> Result<Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>>, AppError> {
    let _timer = logging::RequestTimer::new("http.generate_sql_stream");
    tracing::debug!(question = %body.question, "POST /api/generate_sql/stream");
    let deltas = engine.generate_sql_stream(&body.question).await?;
    let events = deltas.map(|chunk| {
        Ok(match chunk {
            Ok(delta) => Event::default().data(delta),
            Err(e) => Event::default().event("error").data(e.to_string()),
        })
    });
    Ok(Sse::new(events).keep_alive(KeepAlive::default()))
}

#[derive(Deserialize)]
struct SqlBody {
    sql: String,
}

async fn run_sql(
    State(engine): State<Engine>,
    Json(body): Json<SqlBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    let _timer = logging::RequestTimer::new("http.run_sql");
    tracing::debug!("POST /api/run_sql");
    let result = engine.run_sql(&body.sql).await?;
    Ok(Json(serde_json::to_value(result)?))
}

async fn train(
    State(engine): State<Engine>,
    Json(item): Json<TrainingItem>,
) -> Result<Json<serde_json::Value>, AppError> {
    let _timer = logging::RequestTimer::new("http.train");
    let id = engine.train(item).await?;
    Ok(Json(json!({ "id": id })))
}

async fn list_training(
    State(engine): State<Engine>,
) -> Result<Json<serde_json::Value>, AppError> {
    let _timer = logging::RequestTimer::new("http.list_training");
    let rows = engine.list_training().await?;
    Ok(Json(serde_json::to_value(rows)?))
}

async fn remove_training(
    State(engine): State<Engine>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let _timer = logging::RequestTimer::new("http.remove_training");
    let removed = engine.remove_training(&id).await?;
    Ok(Json(json!({ "removed": removed })))
}

/// Error wrapper that turns any engine error into a JSON 500 response.
struct AppError(anyhow::Error);

impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        AppError(err.into())
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        tracing::warn!(error = %self.0, "request failed");
        let body = Json(json!({ "error": self.0.to_string() }));
        (StatusCode::INTERNAL_SERVER_ERROR, body).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ask_body_defaults_run_true_followups_false() {
        // The HTTP contract: `run` defaults to true, `followups` to false.
        let body: AskBody = serde_json::from_str(r#"{"question":"q"}"#).unwrap();
        assert_eq!(body.question, "q");
        assert!(body.run);
        assert!(!body.followups);
    }

    #[test]
    fn ask_body_honours_explicit_flags() {
        let body: AskBody =
            serde_json::from_str(r#"{"question":"q","run":false,"followups":true}"#).unwrap();
        assert!(!body.run);
        assert!(body.followups);
    }

    #[test]
    fn default_true_is_true() {
        assert!(default_true());
    }

    #[test]
    fn app_error_becomes_json_500() {
        let err: AppError = anyhow::anyhow!("kaboom").into();
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn middleware_error_maps_overloaded_to_503() {
        let err: BoxError = Box::new(tower::load_shed::error::Overloaded::new());
        let (status, _) = handle_middleware_error(err).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn middleware_error_maps_timeout_to_408() {
        let err: BoxError = Box::new(tower::timeout::error::Elapsed::new());
        let (status, _) = handle_middleware_error(err).await;
        assert_eq!(status, StatusCode::REQUEST_TIMEOUT);
    }

    #[tokio::test]
    async fn middleware_error_maps_other_to_500() {
        let err: BoxError = "boom".into();
        let (status, _) = handle_middleware_error(err).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    }
}
