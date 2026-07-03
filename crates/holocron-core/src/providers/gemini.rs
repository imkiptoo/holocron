//! Google Gemini provider: chat completion + text embeddings over the
//! Generative Language REST API (`generativelanguage.googleapis.com/v1beta`).

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::Semaphore;

use crate::config::Config;
use crate::error::{Error, Result};
use crate::traits::{ChatStream, Embedder, Llm};
use crate::types::{Message, Role};

const API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Per-request timeout so a hung Gemini call can't block a request forever.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// Build the default HTTP client (with a request timeout). Callers that make
/// both chat and embedding calls should build one and share it so they reuse a
/// single connection pool.
pub fn default_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .unwrap_or_default()
}

/// Build a shared concurrency governor sized to `max_concurrency`.
pub fn shared_limiter(max_concurrency: usize) -> Arc<Semaphore> {
    Arc::new(Semaphore::new(max_concurrency.max(1)))
}

/// Shared HTTP client + credentials for both chat and embedding calls.
#[derive(Clone)]
struct GeminiHttp {
    client: reqwest::Client,
    api_key: String,
    /// API root, e.g. `https://generativelanguage.googleapis.com/v1beta`.
    /// A field (rather than the `API_BASE` const directly) so tests can point
    /// it at a mock server.
    base: String,
    /// Caps concurrent outbound calls across chat + embeddings (rate-limit
    /// governor). Shared so a burst of requests can't stampede the API.
    limiter: Arc<Semaphore>,
    /// Retries on transient errors (429/503, timeouts) before giving up.
    max_retries: u32,
}

impl GeminiHttp {
    fn new(api_key: String, client: reqwest::Client, limiter: Arc<Semaphore>, max_retries: u32) -> Self {
        Self { client, api_key, base: API_BASE.to_string(), limiter, max_retries }
    }

    /// POST `body` as JSON and deserialize the response, holding a concurrency
    /// permit and retrying transient failures (429 / 5xx / timeout / connect)
    /// with exponential backoff. The response body is parsed regardless of
    /// status because Gemini returns its error payload as JSON.
    async fn post_json<T: DeserializeOwned>(
        &self,
        url: &str,
        body: &serde_json::Value,
    ) -> Result<T> {
        let _permit = self.limiter.acquire().await.expect("limiter never closed");
        let mut attempt = 0u32;
        loop {
            match self.client.post(url).json(body).send().await {
                Ok(resp) => {
                    let status = resp.status();
                    let transient = status.as_u16() == 429 || status.is_server_error();
                    if transient && attempt < self.max_retries {
                        attempt += 1;
                        tracing::warn!(status = status.as_u16(), attempt, "gemini transient error; retrying after backoff");
                        backoff(attempt).await;
                        continue;
                    }
                    return Ok(resp.json::<T>().await?);
                }
                Err(e) => {
                    let transient = e.is_timeout() || e.is_connect();
                    if transient && attempt < self.max_retries {
                        attempt += 1;
                        tracing::warn!(error = %e, attempt, "gemini request failed; retrying after backoff");
                        backoff(attempt).await;
                        continue;
                    }
                    return Err(e.into());
                }
            }
        }
    }
}

/// Exponential backoff: 400ms, 800ms, 1600ms, ... capped at 8s.
async fn backoff(attempt: u32) {
    let ms = (200u64 << attempt).min(8_000);
    tokio::time::sleep(Duration::from_millis(ms)).await;
}

// ---- Chat ---------------------------------------------------------------

#[derive(Clone)]
pub struct GeminiLlm {
    http: GeminiHttp,
    model: String,
}

impl GeminiLlm {
    pub fn new(config: &Config) -> Self {
        Self::with_client(config, default_client(), shared_limiter(config.gemini_max_concurrency))
    }

    /// Build using a caller-supplied client + shared concurrency limiter
    /// (share both across the LLM + embedder so they pool connections and
    /// share one rate-limit budget).
    pub fn with_client(config: &Config, client: reqwest::Client, limiter: Arc<Semaphore>) -> Self {
        Self {
            http: GeminiHttp::new(
                config.gemini_api_key.clone(),
                client,
                limiter,
                config.gemini_max_retries,
            ),
            model: config.gemini_chat_model.clone(),
        }
    }

    /// Build the Gemini request body from chat messages: system turns collapse
    /// into `system_instruction`; user/assistant map to user/model contents.
    fn build_body(&self, messages: &[Message]) -> serde_json::Value {
        let mut system_text = String::new();
        let mut contents: Vec<Content> = Vec::new();
        for m in messages {
            match m.role {
                Role::System => {
                    if !system_text.is_empty() {
                        system_text.push_str("\n\n");
                    }
                    system_text.push_str(&m.content);
                }
                Role::User => contents.push(Content {
                    role: "user",
                    parts: vec![Part { text: &m.content }],
                }),
                Role::Assistant => contents.push(Content {
                    role: "model",
                    parts: vec![Part { text: &m.content }],
                }),
            }
        }
        let mut body = json!({ "contents": contents });
        if !system_text.is_empty() {
            body["system_instruction"] = json!({ "parts": [{ "text": system_text }] });
        }
        body
    }
}

/// Concatenate all text parts across a response's candidates.
fn response_text(resp: GenerateResponse) -> String {
    resp.candidates
        .into_iter()
        .filter_map(|c| c.content)
        .flat_map(|c| c.parts)
        .map(|p| p.text)
        .collect::<Vec<_>>()
        .join("")
}

#[derive(Serialize)]
struct Part<'a> {
    text: &'a str,
}

#[derive(Serialize)]
struct Content<'a> {
    role: &'a str,
    parts: Vec<Part<'a>>,
}

#[derive(Deserialize)]
struct GenerateResponse {
    #[serde(default)]
    candidates: Vec<Candidate>,
    #[serde(default)]
    error: Option<ApiError>,
}

#[derive(Deserialize)]
struct Candidate {
    content: Option<CandidateContent>,
}

#[derive(Deserialize)]
struct CandidateContent {
    #[serde(default)]
    parts: Vec<TextPart>,
}

#[derive(Deserialize)]
struct TextPart {
    #[serde(default)]
    text: String,
}

#[derive(Deserialize)]
struct ApiError {
    message: String,
}

#[async_trait]
impl Llm for GeminiLlm {
    async fn chat(&self, messages: &[Message]) -> Result<String> {
        tracing::debug!(model = %self.model, turns = messages.len(), "gemini chat request");
        let body = self.build_body(messages);
        let url = format!(
            "{}/models/{}:generateContent?key={}",
            self.http.base, self.model, self.http.api_key
        );
        let resp: GenerateResponse = self.http.post_json(&url, &body).await?;

        if let Some(err) = &resp.error {
            return Err(Error::Llm(err.message.clone()));
        }
        let text = response_text(resp);
        if text.is_empty() {
            return Err(Error::Llm("empty response from Gemini".into()));
        }
        Ok(text)
    }

    async fn chat_stream(&self, messages: &[Message]) -> Result<ChatStream> {
        use futures::StreamExt;

        tracing::debug!(model = %self.model, turns = messages.len(), "gemini streaming chat request");
        let body = self.build_body(messages);
        // `alt=sse` makes Gemini emit Server-Sent Events (`data: {json}`) rather
        // than one buffered JSON array, so we can forward deltas as they arrive.
        let url = format!(
            "{}/models/{}:streamGenerateContent?alt=sse&key={}",
            self.http.base, self.model, self.http.api_key
        );
        // Hold a concurrency permit for the whole stream's lifetime.
        let permit = self
            .http
            .limiter
            .clone()
            .acquire_owned()
            .await
            .expect("limiter never closed");
        let resp = self.http.client.post(url).json(&body).send().await?;

        let stream = async_stream::try_stream! {
            let _permit = permit; // released when the stream is dropped
            let mut bytes = resp.bytes_stream();
            let mut buf: Vec<u8> = Vec::new();
            while let Some(chunk) = bytes.next().await {
                buf.extend_from_slice(&chunk?);
                // Emit every complete `\n`-terminated line we have buffered.
                while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                    let line: Vec<u8> = buf.drain(..=nl).collect();
                    let line = String::from_utf8_lossy(&line);
                    let line = line.trim();
                    let Some(data) = line.strip_prefix("data:") else { continue };
                    let data = data.trim();
                    if data.is_empty() || data == "[DONE]" {
                        continue;
                    }
                    if let Ok(resp) = serde_json::from_str::<GenerateResponse>(data) {
                        if let Some(err) = &resp.error {
                            Err(Error::Llm(err.message.clone()))?;
                        }
                        let delta = response_text(resp);
                        if !delta.is_empty() {
                            yield delta;
                        }
                    }
                }
            }
        };
        Ok(Box::pin(stream))
    }
}

// ---- Embeddings ---------------------------------------------------------

#[derive(Clone)]
pub struct GeminiEmbedder {
    http: GeminiHttp,
    model: String,
    dims: usize,
}

impl GeminiEmbedder {
    pub fn new(config: &Config) -> Self {
        Self::with_client(config, default_client(), shared_limiter(config.gemini_max_concurrency))
    }

    /// Build using a caller-supplied client + shared concurrency limiter.
    pub fn with_client(config: &Config, client: reqwest::Client, limiter: Arc<Semaphore>) -> Self {
        Self {
            http: GeminiHttp::new(
                config.gemini_api_key.clone(),
                client,
                limiter,
                config.gemini_max_retries,
            ),
            model: config.gemini_embed_model.clone(),
            dims: config.embed_dims,
        }
    }

    /// One `content` request payload for embed / batch-embed.
    fn embed_request(&self, text: &str) -> serde_json::Value {
        // `outputDimensionality` lets models like gemini-embedding-001 (default
        // 3072) emit a smaller vector that matches our pgvector column width.
        json!({
            "model": format!("models/{}", self.model),
            "content": { "parts": [{ "text": text }] },
            "outputDimensionality": self.dims,
        })
    }
}

#[derive(Deserialize)]
struct EmbedResponse {
    embedding: Option<EmbeddingValues>,
    #[serde(default)]
    error: Option<ApiError>,
}

#[derive(Deserialize)]
struct BatchEmbedResponse {
    #[serde(default)]
    embeddings: Vec<EmbeddingValues>,
    #[serde(default)]
    error: Option<ApiError>,
}

#[derive(Deserialize)]
struct EmbeddingValues {
    values: Vec<f32>,
}

#[async_trait]
impl Embedder for GeminiEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        tracing::trace!(model = %self.model, dims = self.dims, "gemini embed request");
        let url = format!(
            "{}/models/{}:embedContent?key={}",
            self.http.base, self.model, self.http.api_key
        );
        let resp: EmbedResponse = self.http.post_json(&url, &self.embed_request(text)).await?;

        if let Some(err) = resp.error {
            return Err(Error::Llm(err.message));
        }
        resp.embedding
            .map(|e| e.values)
            .ok_or_else(|| Error::Llm("embedding response had no values".into()))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        tracing::debug!(model = %self.model, count = texts.len(), "gemini batch embed request");
        // One `batchEmbedContents` call instead of N `embedContent` round-trips.
        let url = format!(
            "{}/models/{}:batchEmbedContents?key={}",
            self.http.base, self.model, self.http.api_key
        );
        let requests: Vec<serde_json::Value> =
            texts.iter().map(|t| self.embed_request(t)).collect();
        let body = json!({ "requests": requests });
        let resp: BatchEmbedResponse = self.http.post_json(&url, &body).await?;

        if let Some(err) = resp.error {
            return Err(Error::Llm(err.message));
        }
        if resp.embeddings.len() != texts.len() {
            return Err(Error::Llm(format!(
                "batch embed returned {} vectors for {} inputs",
                resp.embeddings.len(),
                texts.len()
            )));
        }
        Ok(resp.embeddings.into_iter().map(|e| e.values).collect())
    }

    fn dims(&self) -> usize {
        self.dims
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{Embedder, Llm};
    use crate::types::Message;
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn http_at(base: &str, max_retries: u32) -> GeminiHttp {
        GeminiHttp {
            client: reqwest::Client::new(),
            api_key: "test-key".into(),
            base: format!("{base}/v1beta"),
            limiter: shared_limiter(8),
            max_retries,
        }
    }

    /// A `GeminiLlm` pointed at a mock server's base URL.
    fn llm_at(base: &str, model: &str) -> GeminiLlm {
        GeminiLlm { http: http_at(base, 0), model: model.into() }
    }

    fn embedder_at(base: &str, model: &str, dims: usize) -> GeminiEmbedder {
        GeminiEmbedder { http: http_at(base, 0), model: model.into(), dims }
    }

    #[tokio::test]
    async fn chat_maps_roles_and_returns_text() {
        let server = MockServer::start().await;
        // The mock only responds if the request body has the expected shape:
        // system messages become `system_instruction`, user -> "user",
        // assistant -> "model".
        Mock::given(method("POST"))
            .and(path("/v1beta/models/test-model:generateContent"))
            .and(query_param("key", "test-key"))
            .and(body_partial_json(json!({
                "system_instruction": { "parts": [{ "text": "you are helpful" }] },
                "contents": [
                    { "role": "user", "parts": [{ "text": "hello" }] },
                    { "role": "model", "parts": [{ "text": "hi there" }] },
                    { "role": "user", "parts": [{ "text": "again" }] },
                ],
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "candidates": [
                    { "content": { "parts": [{ "text": "Hello " }, { "text": "world!" }] } }
                ]
            })))
            .mount(&server)
            .await;

        let llm = llm_at(&server.uri(), "test-model");
        let out = llm
            .chat(&[
                Message::system("you are helpful"),
                Message::user("hello"),
                Message::assistant("hi there"),
                Message::user("again"),
            ])
            .await
            .unwrap();
        // Multiple parts are concatenated.
        assert_eq!(out, "Hello world!");
    }

    #[tokio::test]
    async fn chat_joins_multiple_system_messages() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1beta/models/m:generateContent"))
            .and(body_partial_json(json!({
                "system_instruction": { "parts": [{ "text": "a\n\nb" }] },
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "candidates": [ { "content": { "parts": [{ "text": "ok" }] } } ]
            })))
            .mount(&server)
            .await;

        let llm = llm_at(&server.uri(), "m");
        let out = llm
            .chat(&[Message::system("a"), Message::system("b"), Message::user("q")])
            .await
            .unwrap();
        assert_eq!(out, "ok");
    }

    #[tokio::test]
    async fn chat_surfaces_api_error_message() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(429).set_body_json(json!({
                "error": { "message": "quota exceeded" }
            })))
            .mount(&server)
            .await;

        let llm = llm_at(&server.uri(), "m");
        let err = llm.chat(&[Message::user("q")]).await.unwrap_err();
        match err {
            Error::Llm(msg) => assert_eq!(msg, "quota exceeded"),
            other => panic!("expected Llm error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_errors_on_empty_candidates() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "candidates": [] })))
            .mount(&server)
            .await;

        let llm = llm_at(&server.uri(), "m");
        let err = llm.chat(&[Message::user("q")]).await.unwrap_err();
        assert!(matches!(err, Error::Llm(m) if m.contains("empty response")));
    }

    #[tokio::test]
    async fn embed_sends_output_dimensionality_and_returns_values() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1beta/models/test-embed:embedContent"))
            .and(query_param("key", "test-key"))
            .and(body_partial_json(json!({
                "model": "models/test-embed",
                "content": { "parts": [{ "text": "embed me" }] },
                "outputDimensionality": 4,
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "embedding": { "values": [0.1, 0.2, 0.3, 0.4] }
            })))
            .mount(&server)
            .await;

        let embedder = embedder_at(&server.uri(), "test-embed", 4);
        let v = embedder.embed("embed me").await.unwrap();
        assert_eq!(v, vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(embedder.dims(), 4);
    }

    #[tokio::test]
    async fn embed_surfaces_api_error_message() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(400).set_body_json(json!({
                "error": { "message": "bad model" }
            })))
            .mount(&server)
            .await;

        let embedder = embedder_at(&server.uri(), "m", 4);
        let err = embedder.embed("x").await.unwrap_err();
        assert!(matches!(err, Error::Llm(m) if m == "bad model"));
    }

    #[tokio::test]
    async fn embed_errors_when_response_has_no_values() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
            .mount(&server)
            .await;

        let embedder = embedder_at(&server.uri(), "m", 4);
        let err = embedder.embed("x").await.unwrap_err();
        assert!(matches!(err, Error::Llm(m) if m.contains("no values")));
    }

    #[tokio::test]
    async fn embed_batch_uses_batch_endpoint() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1beta/models/test-embed:batchEmbedContents"))
            .and(body_partial_json(json!({
                "requests": [
                    { "content": { "parts": [{ "text": "a" }] }, "outputDimensionality": 2 },
                    { "content": { "parts": [{ "text": "b" }] }, "outputDimensionality": 2 },
                ]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "embeddings": [ { "values": [1.0, 2.0] }, { "values": [3.0, 4.0] } ]
            })))
            .mount(&server)
            .await;

        let embedder = embedder_at(&server.uri(), "test-embed", 2);
        let out = embedder
            .embed_batch(&["a".to_string(), "b".to_string()])
            .await
            .unwrap();
        assert_eq!(out, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    #[tokio::test]
    async fn embed_batch_empty_makes_no_call() {
        // No mock mounted: if it tried to call, it would fail.
        let embedder = embedder_at("http://127.0.0.1:1", "m", 2);
        assert!(embedder.embed_batch(&[]).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn retries_then_succeeds_on_transient_429() {
        let server = MockServer::start().await;
        // First response 429, then 200 - with max_retries=2 the call should
        // recover and return the eventual success.
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(429).set_body_json(json!({
                "error": { "message": "slow down" }
            })))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "candidates": [ { "content": { "parts": [{ "text": "recovered" }] } } ]
            })))
            .mount(&server)
            .await;

        let mut llm = llm_at(&server.uri(), "m");
        llm.http.max_retries = 2;
        let out = llm.chat(&[Message::user("q")]).await.unwrap();
        assert_eq!(out, "recovered");
    }

    #[tokio::test]
    async fn chat_stream_yields_sse_deltas() {
        use futures::StreamExt;

        let server = MockServer::start().await;
        // Gemini SSE: each event is a `data: {json}` line.
        let sse = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hel\"}]}}]}\n\n\
                   data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"lo\"}]}}]}\n\n\
                   data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/v1beta/models/m:streamGenerateContent"))
            .and(query_param("alt", "sse"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&server)
            .await;

        let llm = llm_at(&server.uri(), "m");
        let mut stream = llm.chat_stream(&[Message::user("hi")]).await.unwrap();
        let mut got = String::new();
        while let Some(delta) = stream.next().await {
            got.push_str(&delta.unwrap());
        }
        assert_eq!(got, "Hello");
    }
}
