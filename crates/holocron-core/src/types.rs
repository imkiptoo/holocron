//! Core data types shared across the engine, providers, and surfaces.

use serde::{Deserialize, Serialize};

/// A single chat message handed to the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: Role::System, content: content.into() }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: content.into() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

/// The kind of training material stored in the vector store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrainingKind {
    Ddl,
    Documentation,
    Sql,
}

impl TrainingKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            TrainingKind::Ddl => "ddl",
            TrainingKind::Documentation => "documentation",
            TrainingKind::Sql => "sql",
        }
    }
}

/// A request to add training material. `Sql` without a question triggers
/// LLM-generated question synthesis (Vanna's `generate_question`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum TrainingItem {
    Ddl { ddl: String },
    Documentation { documentation: String },
    Sql { question: Option<String>, sql: String },
}

/// A stored training row, as returned by `get_training_data`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRow {
    pub id: String,
    pub kind: TrainingKind,
    pub question: Option<String>,
    pub content: String,
}

/// The result of running a SQL statement: column names + rows of JSON values.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QueryResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<serde_json::Value>>,
}

impl QueryResult {
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }
}

/// The full answer to a natural-language question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AskResult {
    pub question: String,
    pub sql: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<QueryResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// The model's answer synthesized from `result` - prose insight by default,
    /// but the phrasing is prompt-driven, so it can equally be the rows rendered
    /// as CSV/JSON/etc. Present only when an answer was requested and the query
    /// ran successfully.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,
    #[serde(default)]
    pub followups: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn message_constructors_set_roles() {
        assert_eq!(Message::system("s").role, Role::System);
        assert_eq!(Message::user("u").role, Role::User);
        assert_eq!(Message::assistant("a").role, Role::Assistant);
        assert_eq!(Message::user("hi").content, "hi");
    }

    #[test]
    fn role_serializes_lowercase() {
        assert_eq!(serde_json::to_value(Role::System).unwrap(), json!("system"));
        assert_eq!(serde_json::to_value(Role::User).unwrap(), json!("user"));
        assert_eq!(
            serde_json::to_value(Role::Assistant).unwrap(),
            json!("assistant")
        );
    }

    #[test]
    fn training_kind_as_str_and_serde_agree() {
        for (k, s) in [
            (TrainingKind::Ddl, "ddl"),
            (TrainingKind::Documentation, "documentation"),
            (TrainingKind::Sql, "sql"),
        ] {
            assert_eq!(k.as_str(), s);
            assert_eq!(serde_json::to_value(k).unwrap(), json!(s));
        }
    }

    #[test]
    fn training_item_is_internally_tagged_by_kind() {
        // The HTTP /api/train body relies on this tagging.
        let ddl = TrainingItem::Ddl { ddl: "CREATE TABLE t()".into() };
        assert_eq!(
            serde_json::to_value(&ddl).unwrap(),
            json!({ "kind": "ddl", "ddl": "CREATE TABLE t()" })
        );

        let sql: TrainingItem =
            serde_json::from_value(json!({ "kind": "sql", "sql": "SELECT 1" })).unwrap();
        match sql {
            TrainingItem::Sql { question, sql } => {
                assert_eq!(question, None); // optional question defaults to None
                assert_eq!(sql, "SELECT 1");
            }
            _ => panic!("expected Sql variant"),
        }
    }

    #[test]
    fn query_result_row_count_matches_rows() {
        let qr = QueryResult {
            columns: vec!["n".into()],
            rows: vec![vec![json!(1)], vec![json!(2)]],
        };
        assert_eq!(qr.row_count(), 2);
        assert_eq!(QueryResult::default().row_count(), 0);
    }

    #[test]
    fn ask_result_omits_empty_optionals() {
        let r = AskResult {
            question: "q".into(),
            sql: "SELECT 1".into(),
            result: None,
            error: None,
            answer: None,
            followups: vec![],
        };
        let v = serde_json::to_value(&r).unwrap();
        // result/error/answer are skipped when None; followups always present.
        assert!(v.get("result").is_none());
        assert!(v.get("error").is_none());
        assert!(v.get("answer").is_none());
        assert_eq!(v["followups"], json!([]));
        assert_eq!(v["sql"], json!("SELECT 1"));
    }

    #[test]
    fn ask_result_round_trips() {
        let r = AskResult {
            question: "q".into(),
            sql: "SELECT 1".into(),
            result: Some(QueryResult {
                columns: vec!["c".into()],
                rows: vec![vec![json!("v")]],
            }),
            error: None,
            answer: Some("There is one row.".into()),
            followups: vec!["next?".into()],
        };
        let back: AskResult =
            serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        assert_eq!(back.sql, "SELECT 1");
        assert_eq!(back.followups, vec!["next?".to_string()]);
        assert_eq!(back.result.unwrap().row_count(), 1);
    }
}
