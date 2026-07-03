//! `holocron-grpc` - a gRPC surface over the holocron engine, alongside the HTTP
//! `holocron-server`. `GenerateSql` is the drop-in the DairyBook Go API calls in
//! place of the Python Vanna sidecar.

use std::pin::Pin;

use futures::{Stream, StreamExt};
use tonic::{transport::Server, Request, Response, Status};
use tracing::Instrument;

use holocron_core::logging::{self, Format};
use holocron_core::types::Message;
use holocron_core::{default_engine, Config, Engine, Error, TrainingItem};

/// Generated protobuf types + service traits (`holocron.v1`).
pub mod pb {
    tonic::include_proto!("holocron.v1");
}

use pb::holocron_server::{Holocron, HolocronServer};
use pb::{
    train_request::Item, AskRequest, AskResponse, GenerateSqlRequest, GenerateSqlResponse,
    HealthRequest, HealthResponse, ListTrainingDataRequest, ListTrainingDataResponse,
    RemoveTrainingDataRequest, RemoveTrainingDataResponse, RunSqlRequest, RunSqlResponse, SqlDelta,
    TrainRequest, TrainResponse, TrainingRow,
};

/// Map a holocron [`Error`] onto a gRPC [`Status`], logging it on the way out so
/// every failed RPC leaves a trace (invalid-argument at debug, the rest at warn).
fn to_status(e: Error) -> Status {
    let status = match e {
        Error::NoSql => Status::invalid_argument("could not turn that question into a query"),
        Error::ReadOnly(sql) => Status::failed_precondition(format!("statement is not read-only: {sql}")),
        Error::Rejected(reason) => Status::failed_precondition(format!("rejected by SQL policy: {reason}")),
        other => Status::internal(other.to_string()),
    };
    if status.code() == tonic::Code::InvalidArgument {
        tracing::debug!(code = ?status.code(), "rpc rejected: {}", status.message());
    } else {
        tracing::warn!(code = ?status.code(), "rpc failed: {}", status.message());
    }
    status
}

/// Turn proto `Turn`s into engine chat messages.
fn history_to_messages(history: &[pb::Turn]) -> Vec<Message> {
    history
        .iter()
        .map(|t| match t.role.as_str() {
            "assistant" => Message::assistant(t.content.clone()),
            _ => Message::user(t.content.clone()),
        })
        .collect()
}

struct HolocronService {
    engine: Engine,
}

#[tonic::async_trait]
impl Holocron for HolocronService {
    #[tracing::instrument(skip_all, name = "grpc.generate_sql")]
    async fn generate_sql(
        &self,
        request: Request<GenerateSqlRequest>,
    ) -> Result<Response<GenerateSqlResponse>, Status> {
        let _timer = logging::RequestTimer::new("grpc.generate_sql");
        let req = request.into_inner();
        tracing::debug!(
            question = %req.question,
            history = req.history.len(),
            views = req.allowed_views.len(),
            "GenerateSql"
        );
        let history = history_to_messages(&req.history);
        let extra = if req.schema_context.trim().is_empty() {
            Vec::new()
        } else {
            vec![req.schema_context]
        };
        let sql = self
            .engine
            .generate_sql_with(&req.question, &extra, &history)
            .await
            .map_err(to_status)?;
        Ok(Response::new(GenerateSqlResponse { sql }))
    }

    type GenerateSqlStreamStream =
        Pin<Box<dyn Stream<Item = Result<SqlDelta, Status>> + Send + 'static>>;

    #[tracing::instrument(skip_all, name = "grpc.generate_sql_stream")]
    async fn generate_sql_stream(
        &self,
        request: Request<GenerateSqlRequest>,
    ) -> Result<Response<Self::GenerateSqlStreamStream>, Status> {
        let _timer = logging::RequestTimer::new("grpc.generate_sql_stream");
        let req = request.into_inner();
        tracing::debug!(question = %req.question, "GenerateSqlStream");
        let deltas = self
            .engine
            .generate_sql_stream(&req.question)
            .await
            .map_err(to_status)?;
        let stream = deltas.map(|chunk| match chunk {
            Ok(text) => Ok(SqlDelta { text }),
            Err(e) => Err(to_status(e)),
        });
        Ok(Response::new(Box::pin(stream)))
    }

    #[tracing::instrument(skip_all, name = "grpc.ask")]
    async fn ask(&self, request: Request<AskRequest>) -> Result<Response<AskResponse>, Status> {
        let _timer = logging::RequestTimer::new("grpc.ask");
        let req = request.into_inner();
        tracing::debug!(
            question = %req.question,
            run = req.run,
            answer = req.answer,
            followups = req.followups,
            "Ask"
        );
        let res = self
            .engine
            .ask(&req.question, req.run, req.answer, req.followups)
            .await
            .map_err(to_status)?;
        let result_json = match &res.result {
            Some(qr) => serde_json::to_string(qr).map_err(|e| Status::internal(e.to_string()))?,
            None => String::new(),
        };
        Ok(Response::new(AskResponse {
            sql: res.sql,
            result_json,
            error: res.error.unwrap_or_default(),
            followups: res.followups,
            answer: res.answer.unwrap_or_default(),
        }))
    }

    #[tracing::instrument(skip_all, name = "grpc.run_sql")]
    async fn run_sql(
        &self,
        request: Request<RunSqlRequest>,
    ) -> Result<Response<RunSqlResponse>, Status> {
        let _timer = logging::RequestTimer::new("grpc.run_sql");
        let req = request.into_inner();
        tracing::debug!("RunSql");
        let qr = self.engine.run_sql(&req.sql).await.map_err(to_status)?;
        let result_json = serde_json::to_string(&qr).map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(RunSqlResponse { result_json }))
    }

    #[tracing::instrument(skip_all, name = "grpc.train")]
    async fn train(
        &self,
        request: Request<TrainRequest>,
    ) -> Result<Response<TrainResponse>, Status> {
        let _timer = logging::RequestTimer::new("grpc.train");
        tracing::debug!("Train");
        let item = request
            .into_inner()
            .item
            .ok_or_else(|| Status::invalid_argument("train: item is required"))?;
        let training = match item {
            Item::Ddl(ddl) => TrainingItem::Ddl { ddl },
            Item::Documentation(documentation) => TrainingItem::Documentation { documentation },
            Item::Sql(s) => TrainingItem::Sql {
                question: if s.question.trim().is_empty() { None } else { Some(s.question) },
                sql: s.sql,
            },
        };
        let id = self.engine.train(training).await.map_err(to_status)?;
        Ok(Response::new(TrainResponse { id }))
    }

    #[tracing::instrument(skip_all, name = "grpc.list_training_data")]
    async fn list_training_data(
        &self,
        _request: Request<ListTrainingDataRequest>,
    ) -> Result<Response<ListTrainingDataResponse>, Status> {
        let _timer = logging::RequestTimer::new("grpc.list_training_data");
        tracing::debug!("ListTrainingData");
        let rows = self.engine.list_training().await.map_err(to_status)?;
        let rows = rows
            .into_iter()
            .map(|r| TrainingRow {
                id: r.id,
                kind: r.kind.as_str().to_string(),
                question: r.question.unwrap_or_default(),
                content: r.content,
            })
            .collect();
        Ok(Response::new(ListTrainingDataResponse { rows }))
    }

    #[tracing::instrument(skip_all, name = "grpc.remove_training_data")]
    async fn remove_training_data(
        &self,
        request: Request<RemoveTrainingDataRequest>,
    ) -> Result<Response<RemoveTrainingDataResponse>, Status> {
        let _timer = logging::RequestTimer::new("grpc.remove_training_data");
        let id = request.into_inner().id;
        tracing::debug!(%id, "RemoveTrainingData");
        let removed = self.engine.remove_training(&id).await.map_err(to_status)?;
        Ok(Response::new(RemoveTrainingDataResponse { removed }))
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse { status: "ok".into() }))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = Config::load()?;
    // Logging in the shared DairyBook `./api` style (console/JSON, RFC3339,
    // leveled), configured from config; `service` stamped via a root span.
    let span = logging::init_service(&config.log_level, Format::parse(&config.log_format), "holocron-grpc");

    async move {
        let engine = default_engine(config).await?;

        // The gRPC listen address is its own env var so it can run beside the HTTP
        // server without colliding on `[server].bind_addr`.
        let addr = std::env::var("HOLOCRON_GRPC_ADDR")
            .unwrap_or_else(|_| "127.0.0.1:50051".to_string())
            .parse()?;

        tracing::info!("holocron-grpc listening on {addr}");
        Server::builder()
            .add_service(HolocronServer::new(HolocronService { engine }))
            .serve(addr)
            .await?;
        Ok(())
    }
    .instrument(span)
    .await
}
