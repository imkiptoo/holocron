# holocron

A **text-to-SQL RAG engine** in Rust — a Vanna-style system that turns natural-language
questions into SQL over your PostgreSQL database, using **Google Gemini** for the LLM and
embeddings and **pgvector** for retrieval.

You *train* it on your schema (DDL), documentation, and known question→SQL pairs. For a
new question it embeds the question, retrieves the most relevant training snippets, builds
a few-shot prompt, asks Gemini to write SQL, runs it, and returns the rows.

## Workspace layout

| Crate | What it is |
|-------|------------|
| `crates/holocron-core` | The engine: `Llm` / `Embedder` / `VectorStore` / `SqlRunner` traits, the `Engine`, RAG prompt assembly, and Gemini + pgvector + Postgres providers. |
| `crates/holocron-cli` | `holocron` command-line tool. |
| `crates/holocron-server` | `holocron-server` HTTP API (axum). |
| `crates/holocron-grpc` | `holocron-grpc` gRPC API (tonic) — same operations as the HTTP server; the DairyBook Go API calls `GenerateSql` here in place of the Python Vanna sidecar. |

## Setup

1. **Postgres with pgvector** (one DB can be both the warehouse and the vector store).
   Either a native install (`brew install pgvector` against your `postgresql`) or Docker:

   ```sh
   docker run -d --name holocron-pg -e POSTGRES_PASSWORD=pg -p 5432:5432 pgvector/pgvector:pg16
   ```

2. **Demo data** (optional but recommended): a realistic 4-schema business database
   (`sales`, `hr`, `inventory`, `finance`) with ~1.5M rows, generated in seconds:

   ```sh
   createdb holocron_demo   # or: psql -c 'CREATE DATABASE holocron_demo'
   psql "postgres://postgres:postgres@localhost:5432/holocron_demo" -f demo/seed.sql
   ```

3. **Config**: copy `holocron.toml.example` to `holocron.toml` and fill in your
   `[gemini].api_key` and `[database].url`. Config is a TOML file (no `.env`); the
   engine loads `./holocron.toml`, overridable with the `HOLOCRON_CONFIG` env var. Keys:

   - `[gemini] chat_model` (default `gemini-2.5-flash`), `embed_model`
     (`gemini-embedding-001`), `embed_dims` (768 — must match the pgvector column),
     `max_concurrency` (8) + `max_retries` (3) — outbound rate-limit governor.
   - `[database] url` — the warehouse queried by generated SQL;
     `vector_url` — where embeddings live (defaults to `url`); `max_connections`
     (10), `acquire_timeout_secs` (30) — pool sizing.
   - `[retrieval] top_k_ddl/top_k_docs/top_k_sql` — retrieval depth.
   - `[safety]` — the SQL guardrails: `validate_sql` (on) runs the AST gate,
     `allow_system_schemas` (off) blocks `information_schema`/`pg_catalog`,
     `statement_timeout_secs` (30) caps each query, `max_rows` (10000) caps rows,
     `read_only` (true) is the legacy fallback check. See **Security** below.
   - `[cache] enabled/ttl_secs` — the verbatim (exact-match) SQL cache.
   - `[server] bind_addr` (default `127.0.0.1:8080`), `request_timeout_secs` (120),
     `max_concurrent_requests` (64 — load-shed beyond this).
   - `[logging] level` (`debug|info|warn|error`, default `info`; overridable via
     `HOLOCRON_LOG`/`RUST_LOG`) + `format` (`console` for dev, `json` for prod).
     At `info`, each request logs its total time (`… processed elapsed_ms=…`);
     at `debug` every subprocess (embed / retrieve / LLM / execute) logs its own
     `elapsed_ms` too.

4. **Build**: `cargo build --workspace`

## CLI usage

```sh
# One-time: create the pgvector table + index
cargo run -p holocron-cli -- init-db

# Train
cargo run -p holocron-cli -- train --auto                       # from information_schema
cargo run -p holocron-cli -- train --ddl schema.sql             # a file, '-' for stdin, or inline
cargo run -p holocron-cli -- train --doc "revenue = price * qty"
cargo run -p holocron-cli -- train --sql "SELECT ..." --question "top customers?"

# Ask
cargo run -p holocron-cli -- ask "how many customers are there?" --run
cargo run -p holocron-cli -- ask "top 5 products by revenue" --run --followups --json

# Inspect / manage
cargo run -p holocron-cli -- run "SELECT now()"
cargo run -p holocron-cli -- list
cargo run -p holocron-cli -- remove <uuid>
```

## HTTP API

```sh
cargo run -p holocron-server        # listens on [server].bind_addr (default 127.0.0.1:8080)
```

| Method & path | Body → response |
|---------------|-----------------|
| `GET /api/health` | `{status:"ok"}` |
| `POST /api/ask` | `{question, run?, answer?, followups?}` → `AskResult` (`sql`, `result`, `answer`, `followups`) |
| `POST /api/generate_sql` | `{question}` → `{sql}` |
| `POST /api/generate_sql/stream` | `{question}` → SSE stream of generated text deltas |
| `POST /api/run_sql` | `{sql}` → `{columns, rows}` |
| `POST /api/train` | `{kind:"ddl", ddl}` / `{kind:"documentation", documentation}` / `{kind:"sql", question?, sql}` → `{id}` |
| `GET /api/training_data` | → `[TrainingRow]` |
| `DELETE /api/training_data/{id}` | → `{removed}` |

```sh
curl -s localhost:8080/api/ask -H 'content-type: application/json' \
  -d '{"question":"top 5 products by revenue"}' | jq
```

## gRPC API

The same operations are exposed over gRPC by `holocron-grpc` (proto:
[`crates/holocron-grpc/proto/holocron.proto`](crates/holocron-grpc/proto/holocron.proto),
package `holocron.v1`, service `Holocron`): `GenerateSql` (+ `GenerateSqlStream`),
`Ask`, `RunSql`, `Train`, `ListTrainingData`, `RemoveTrainingData`, `Health`.

```sh
HOLOCRON_GRPC_ADDR=127.0.0.1:50051 cargo run -p holocron-grpc   # default 127.0.0.1:50051
```

`GenerateSql` is the drop-in the **DairyBook Go API** (`../api/pkg/ai`) calls in
place of the Python Vanna sidecar: its request carries `question`,
`schema_context` (role-scoped view DDL), `allowed_views`, and conversation
`history`; holocron returns candidate SQL and the Go side validates + executes it
under its read-only role. Select it there with `ai.grpc_url` (or `AI_GRPC_URL`);
the Go client + generated stubs live in `../api/pkg/ai/{holocron_generator.go,holocronpb}`.

## Testing

```sh
cargo test --workspace          # unit + engine tests (no network, no DB)
```

The suite has three layers:

- **Unit tests** (inline `#[cfg(test)]`): config loading + defaults, prompt
  assembly / SQL extraction / combined-reply parsing / read-only detection, the
  type serde contracts, error formatting, the CLI/server helpers (including the
  load-shed/timeout error mapping), the embedding cache, and the Gemini provider
  — request mapping, response/error parsing, **batch embedding**, **SSE
  streaming**, and **429 retry** are all tested against a
  [`wiremock`](https://docs.rs/wiremock) mock HTTP server, so no real API key or
  network is needed.
- **Engine tests** (`crates/holocron-core/tests/engine.rs`): the full
  `train` / `generate_sql` / `run_sql` / `ask` flow driven against in-memory
  fake providers — no Gemini or Postgres. These assert the wiring: which text
  gets embedded, that the configured top-k reaches the store, read-only mode
  blocks writes before they hit the runner, execution errors are captured (not
  raised), followups fold into one call, and the **verbatim cache** hits/misses
  /stores and is invalidated on DDL training.
- **Postgres integration tests** (`crates/holocron-core/tests/pg.rs`): the pgvector
  store round-trip, single-round-trip `get_context`, the verbatim cache
  (lookup/TTL/clear), the SQL runner's type mapping / row cap / introspection —
  against a real database. Gated on `HOLOCRON_TEST_DATABASE_URL`, they skip when it
  is unset (so the default run stays hermetic) and serialize via a shared lock
  since they share one database.

  ```sh
  createdb holocron_test
  psql "postgres://postgres:postgres@localhost:5432/holocron_test" -c 'CREATE EXTENSION IF NOT EXISTS vector'
  HOLOCRON_TEST_DATABASE_URL=postgres://postgres:postgres@localhost:5432/holocron_test \
    cargo test -p holocron-core --test pg
  ```

## How it maps to Vanna

| Vanna (`VannaBase`) | holocron |
|---------------------|-------|
| `add_ddl` / `add_documentation` / `add_question_sql` | `VectorStore` trait (`providers::pgvector`) |
| `get_related_ddl` / `get_similar_question_sql` | nearest-neighbour by cosine distance (`<=>`) |
| `generate_embedding` | `Embedder` (`GeminiEmbedder`) |
| `submit_prompt` / `generate_sql` | `Llm` (`GeminiLlm`) + `Engine::generate_sql` |
| `get_sql_prompt` / `extract_sql` | `prompt::build_sql_prompt` / `prompt::extract_sql` |
| `run_sql` / `connect_to_postgres` | `SqlRunner` (`providers::postgres`) |
| `train` (plan from information_schema) | `Engine::train_from_information_schema` |
| `generate_followup_questions` | `Engine::ask(.., followups=true)` |

## Performance notes

A few deliberate choices in the hot paths:

Latency in an `ask` is dominated by the LLM calls (generation ≫ embedding ≫
retrieval/execution), so the design attacks those first:

- **Verbatim SQL cache.** Before generating, `ask`/`generate_sql` check
  `query_cache` for the **exact** normalized question (`Engine::normalize_question`
  — trimmed, lower-cased, whitespace-collapsed), honouring `ttl_secs`. A hit
  returns the cached SQL and **skips embedding *and* generation** (~2 s → one
  indexed lookup). Matching is exact, *not* by embedding similarity: "top 10 …"
  and "bottom 10 …" are near-identical vectors but need opposite SQL, so a
  similarity cache would answer one with the other's query. Cleared
  automatically when DDL is trained (schema change ⇒ cached SQL may be stale).
  Semantically-similar past questions still help — as few-shot examples in the
  prompt (`get_similar_question_sql`), where the LLM regenerates, like Vanna.
- **Streaming generation.** `Llm::chat_stream` + `Engine::generate_sql_stream`
  use Gemini's `streamGenerateContent?alt=sse` and surface tokens through the
  `POST /api/generate_sql/stream` SSE endpoint, cutting time-to-first-token.
- **Folded follow-ups.** When follow-ups are requested on a cache miss, a single
  combined prompt returns `{sql, followups}` in one call instead of a second
  serial LLM round-trip (`prompt::build_combined_prompt` / `extract_combined`).
- **Single-round-trip retrieval.** `VectorStore::get_context` fetches all three
  buckets in one `UNION ALL` query (one connection), each branch still using its
  per-kind partial HNSW index — instead of three separate queries.
- **Per-kind partial HNSW indexes.** `init_db` builds one partial index per kind
  (`... USING hnsw (embedding vector_cosine_ops) WHERE kind = '...'`); retrieval
  interpolates the kind as a **literal** (not a bound `$1`) so the planner can
  match those predicates — a single whole-table index would apply the `kind`
  filter only after the ANN walk, hurting recall and latency.
- **Outbound rate-limit governor + retries.** A shared `Semaphore`
  (`gemini.max_concurrency`) caps concurrent Gemini calls so a request burst
  can't stampede the API into 429s; transient failures (429/5xx/timeout) retry
  with exponential backoff (`gemini.max_retries`).
- **Load-shedding server.** `tower` `load_shed` + `concurrency_limit` + `timeout`
  layers return `503`/`408` under overload instead of unbounded queueing.
- **Batch embeddings.** `Embedder::embed_batch` uses Gemini's
  `batchEmbedContents`, so auto-training embeds the whole schema in one call.
- **Embedding cache + shared client/pool.** A `CachingEmbedder` decorator
  memoizes exact-match text; `default_engine` shares one `reqwest::Client`
  (with a request timeout) and one Postgres pool (warehouse + vector store when
  they point at the same DB).
- **Streaming, row-capped execution.** `run_sql` streams rows and stops at
  `safety.max_rows`, and resolves each column's type decoder once (not per cell).

## Security

**The LLM is an untrusted SQL generator** — its input is untrusted natural
language (potentially prompt-injected), so its output is never trusted. Defense
in depth, strongest first:

1. **Least-privilege DB role (the real boundary).** Point `[database].url` at a
   read-only role granted `SELECT` on only your analytics schemas — then no
   generated query can read data it wasn't granted, regardless of the SQL. See
   [`demo/readonly_role.sql`](demo/readonly_role.sql). **Never** run against a
   superuser (the demo's `postgres`) with untrusted input — a `SELECT
   pg_read_file(…)` could read server files.
2. **AST validation gate** ([`sql_guard`](crates/holocron-core/src/sql_guard.rs)):
   every query is parsed and must be a single read-only `SELECT` — no writes/DDL
   *anywhere* (incl. writable CTEs like `WITH … DELETE`), no `information_schema`
   /`pg_catalog`/`pg_*` refs, no file/dblink/backend functions. Fails closed.
   Replaces the first-token `is_read_only` check, which passed those.
3. **Hardened execution.** Each query runs in a `BEGIN READ ONLY` transaction
   with `SET LOCAL statement_timeout` and a row cap.

The **answer/summarizer step still sees the result rows** — sending them to
hosted Gemini is a cross-border transfer; self-host or mask for sensitive data.
And the HTTP/gRPC servers have **no auth yet** — add authN/Z, rate limits, and
audit before exposing them, and scope the cache per tenant.

When holocron is the DairyBook API's *generator*, all of this is moot there: the
API sends schema only (never data) and re-validates + executes the SQL itself
under its own `dairybook_ai` role — see `../api/pkg/ai`.

## Notes

- Ports Vanna's classic RAG "base class" model. Plotting is out of scope for v1;
  follow-up questions and auto-training are in.
- On `sqlx 0.8`: the `pgvector` crate (our vector-store binding) has no `sqlx 0.9`
  support yet, and sqlx 0.9 also adds an anti-injection guard that would require
  reworking every dynamic query. Everything else is on the latest release.
- No server auth yet — it's a localhost dev tool. Add a bearer-token middleware layer
  in `holocron-server` before exposing it.
