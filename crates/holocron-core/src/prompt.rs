//! Prompt assembly and SQL extraction — the RAG glue.
//!
//! Ports Vanna's `get_sql_prompt` (system prompt from retrieved DDL + docs +
//! few-shot question/SQL pairs) and `extract_sql` (pull SQL back out of the
//! model's reply).

use serde::Deserialize;

use crate::types::Message;

/// Append the retrieved DDL + documentation sections to a system prompt.
fn append_context(sys: &mut String, ddls: &[String], docs: &[String]) {
    if !ddls.is_empty() {
        sys.push_str("\n===== Database schema (DDL) =====\n");
        for d in ddls {
            sys.push_str(d.trim());
            sys.push_str("\n\n");
        }
    }
    if !docs.is_empty() {
        sys.push_str("\n===== Additional documentation =====\n");
        for d in docs {
            sys.push_str("- ");
            sys.push_str(d.trim());
            sys.push('\n');
        }
    }
}

/// Append few-shot `(question, sql)` pairs and the real question as turns.
fn append_examples_and_question(
    messages: &mut Vec<Message>,
    examples: &[(String, String)],
    question: &str,
) {
    for (q, sql) in examples {
        messages.push(Message::user(q.clone()));
        messages.push(Message::assistant(format!("```sql\n{}\n```", sql.trim())));
    }
    messages.push(Message::user(question.to_string()));
}

/// Build the message list sent to the LLM for a text-to-SQL request.
///
/// * `ddls` — retrieved schema snippets (most relevant first).
/// * `docs` — retrieved documentation snippets.
/// * `examples` — retrieved `(question, sql)` few-shot pairs.
pub fn build_sql_prompt(
    question: &str,
    ddls: &[String],
    docs: &[String],
    examples: &[(String, String)],
) -> Vec<Message> {
    let mut sys = String::new();
    sys.push_str(
        "You are a PostgreSQL expert. Given the database context below, write a \
         single valid PostgreSQL query that answers the user's question.\n\n\
         Rules:\n\
         - Respond with SQL only, wrapped in a ```sql code block.\n\
         - Use only tables and columns that appear in the provided schema.\n\
         - Do not invent columns. If the question cannot be answered from the \
         schema, return a query that selects a clear error message string.\n\
         - Prefer explicit JOINs and qualified column names.\n",
    );
    append_context(&mut sys, ddls, docs);

    let mut messages = vec![Message::system(sys)];
    append_examples_and_question(&mut messages, examples, question);
    messages
}

/// Like [`build_sql_prompt`], but asks the model to return the SQL *and* three
/// follow-up questions in one structured JSON reply — folding what used to be a
/// second LLM round-trip into the generation call.
pub fn build_combined_prompt(
    question: &str,
    ddls: &[String],
    docs: &[String],
    examples: &[(String, String)],
) -> Vec<Message> {
    let mut sys = String::new();
    sys.push_str(
        "You are a PostgreSQL expert. Given the database context below, write a \
         single valid PostgreSQL query that answers the user's question, and \
         suggest follow-up questions.\n\n\
         Rules:\n\
         - Respond with ONLY a JSON object of the form \
         {\"sql\": \"<query>\", \"followups\": [\"q1\", \"q2\", \"q3\"]}.\n\
         - `sql` must be a single valid PostgreSQL query using only tables and \
         columns from the provided schema; do not invent columns.\n\
         - `followups` is up to three concise questions the user might ask next.\n\
         - Do not wrap the JSON in prose.\n",
    );
    append_context(&mut sys, ddls, docs);

    let mut messages = vec![Message::system(sys)];
    append_examples_and_question(&mut messages, examples, question);
    messages
}

/// Parse a combined `{sql, followups}` reply. Falls back to [`extract_sql`]
/// (with no followups) if the reply isn't the expected JSON.
pub fn extract_combined(reply: &str) -> (Option<String>, Vec<String>) {
    #[derive(Deserialize)]
    struct Combined {
        sql: String,
        #[serde(default)]
        followups: Vec<String>,
    }

    // The JSON may be bare or fenced in a ``` block.
    let candidate = fenced_block(reply).unwrap_or_else(|| reply.to_string());
    if let Ok(c) = serde_json::from_str::<Combined>(candidate.trim()) {
        let sql = strip_trailing_semicolon(c.sql.trim());
        let followups = c
            .followups
            .into_iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .take(3)
            .collect();
        let sql = if sql.is_empty() { None } else { Some(sql) };
        return (sql, followups);
    }
    (extract_sql(reply), Vec::new())
}

/// Max result rows sent to the answer model — keeps the prompt within token
/// limits; the true row count is stated so aggregates stay honest.
const ANSWER_ROW_LIMIT: usize = 50;

/// Build the prompt that turns a question + its query result into a written
/// answer (Vanna's `generate_summary`). The instruction is intentionally
/// generic: by default it asks for a concise prose answer with brief,
/// data-grounded recommendations, but honours an explicit format the user asked
/// for in the question (e.g. "as CSV", "as JSON").
pub fn build_answer_prompt(question: &str, result: &crate::types::QueryResult) -> Vec<Message> {
    let total = result.row_count();
    let shown = total.min(ANSWER_ROW_LIMIT);
    let preview: Vec<&Vec<serde_json::Value>> = result.rows.iter().take(ANSWER_ROW_LIMIT).collect();
    let rows_json = serde_json::json!({ "columns": result.columns, "rows": preview });

    let mut sys = String::from(
        "You are a data analyst. Given a user's question and the result of a SQL \
         query over their database (columns + rows as JSON), write a direct answer \
         to the question. Ground every figure in the provided rows — never invent \
         numbers. If the user asked for the data in a specific format (CSV, JSON, a \
         list, a table), return exactly that. Otherwise reply in concise prose: state \
         the key findings, and if they asked for advice or what to do, add one to \
         three brief, actionable recommendations supported only by the data. No \
         preamble, no markdown headings.",
    );
    if total > shown {
        sys.push_str(&format!(
            "\n\nOnly the first {shown} of {total} rows are shown; base any totals or \
             counts on the stated total of {total}, not on the rows you can see.",
        ));
    }

    vec![
        Message::system(sys),
        Message::user(format!("Question: {question}\n\nResult:\n{rows_json}")),
    ]
}

/// Extract a SQL statement from a raw LLM reply.
///
/// Prefers the contents of a fenced code block (```sql … ``` or ``` … ```);
/// otherwise falls back to the substring starting at the first SQL keyword.
/// Returns `None` if nothing SQL-shaped is found.
pub fn extract_sql(response: &str) -> Option<String> {
    // 1) Fenced code block, optionally tagged `sql`.
    if let Some(block) = fenced_block(response) {
        let trimmed = block.trim();
        if !trimmed.is_empty() {
            return Some(strip_trailing_semicolon(trimmed));
        }
    }

    // 2) Fall back to the first SQL statement in free text.
    let lower = response.to_lowercase();
    for kw in ["with ", "select ", "insert ", "update ", "delete "] {
        if let Some(idx) = lower.find(kw) {
            let candidate = response[idx..].trim();
            return Some(strip_trailing_semicolon(candidate));
        }
    }

    None
}

/// Return the inside of the first fenced code block, if any.
fn fenced_block(text: &str) -> Option<String> {
    let start_fence = text.find("```")?;
    let after = &text[start_fence + 3..];
    // Skip an optional language tag on the same line (e.g. `sql`).
    let body_start = match after.find('\n') {
        Some(nl) => nl + 1,
        None => return None,
    };
    let body = &after[body_start..];
    let end = body.find("```")?;
    Some(body[..end].to_string())
}

fn strip_trailing_semicolon(sql: &str) -> String {
    sql.trim().trim_end_matches(';').trim().to_string()
}

/// Is this statement read-only (SELECT / WITH / EXPLAIN / SHOW)?
/// Used to enforce `READ_ONLY` before executing generated SQL.
pub fn is_read_only(sql: &str) -> bool {
    // Compare the first whitespace-delimited token so `SELECT\n...` and
    // `select\t...` are handled the same as `SELECT ...`.
    match sql.split_whitespace().next() {
        Some(word) => matches!(
            word.to_lowercase().as_str(),
            "select" | "with" | "explain" | "show"
        ),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Role;

    #[test]
    fn extracts_from_sql_fence() {
        let r = "Here you go:\n```sql\nSELECT 1;\n```\nHope that helps.";
        assert_eq!(extract_sql(r).unwrap(), "SELECT 1");
    }

    #[test]
    fn extracts_from_bare_fence() {
        let r = "```\nSELECT count(*) FROM customers\n```";
        assert_eq!(extract_sql(r).unwrap(), "SELECT count(*) FROM customers");
    }

    #[test]
    fn extracts_from_free_text() {
        let r = "The query is SELECT * FROM orders WHERE total > 10";
        assert_eq!(
            extract_sql(r).unwrap(),
            "SELECT * FROM orders WHERE total > 10"
        );
    }

    #[test]
    fn read_only_detection() {
        assert!(is_read_only("SELECT * FROM t"));
        assert!(is_read_only("SELECT\n  a,\n  b\nFROM t")); // newline after keyword
        assert!(is_read_only("  with x as (select 1) select * from x"));
        assert!(!is_read_only("DROP TABLE t"));
        assert!(!is_read_only("update t set a = 1"));
    }

    #[test]
    fn read_only_allows_explain_and_show() {
        assert!(is_read_only("EXPLAIN SELECT 1"));
        assert!(is_read_only("show search_path"));
    }

    #[test]
    fn read_only_rejects_empty_and_dml() {
        assert!(!is_read_only(""));
        assert!(!is_read_only("   \n\t "));
        assert!(!is_read_only("insert into t values (1)"));
        assert!(!is_read_only("delete from t"));
        // A token that merely *starts* with select must still be rejected.
        assert!(!is_read_only("selectfoo bar"));
    }

    #[test]
    fn extract_prefers_fence_over_free_text() {
        // Both a fence and a stray keyword exist; the fenced block wins.
        let r = "delete everything? No. ```sql\nSELECT 1\n``` and then select 2";
        assert_eq!(extract_sql(r).unwrap(), "SELECT 1");
    }

    #[test]
    fn extract_strips_only_trailing_semicolons_and_whitespace() {
        let r = "```sql\n  SELECT 1 ;  \n```";
        assert_eq!(extract_sql(r).unwrap(), "SELECT 1");
    }

    #[test]
    fn extract_free_text_is_case_insensitive_keyword() {
        let r = "here: with cte as (select 1) select * from cte";
        assert_eq!(
            extract_sql(r).unwrap(),
            "with cte as (select 1) select * from cte"
        );
    }

    #[test]
    fn extract_returns_none_without_sql() {
        assert!(extract_sql("I don't know how to answer that.").is_none());
        assert!(extract_sql("").is_none());
    }

    #[test]
    fn extract_empty_fence_falls_through_to_none() {
        // An empty fenced block should not be returned as SQL.
        assert!(extract_sql("```sql\n\n```").is_none());
    }

    #[test]
    fn extract_unterminated_fence_falls_back_to_keyword() {
        // No closing fence -> fenced_block returns None -> keyword fallback.
        let r = "```sql\nSELECT 42 FROM t";
        assert_eq!(extract_sql(r).unwrap(), "SELECT 42 FROM t");
    }

    #[test]
    fn build_prompt_minimal_has_system_and_user_only() {
        let msgs = build_sql_prompt("how many?", &[], &[], &[]);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs[1].role, Role::User);
        assert_eq!(msgs[1].content, "how many?");
        // No optional sections when nothing is retrieved.
        assert!(!msgs[0].content.contains("Database schema"));
        assert!(!msgs[0].content.contains("Additional documentation"));
    }

    #[test]
    fn build_prompt_includes_ddl_and_docs() {
        let ddls = vec!["CREATE TABLE t (id int)".to_string()];
        let docs = vec!["revenue = price * qty".to_string()];
        let msgs = build_sql_prompt("q", &ddls, &docs, &[]);
        let sys = &msgs[0].content;
        assert!(sys.contains("Database schema (DDL)"));
        assert!(sys.contains("CREATE TABLE t (id int)"));
        assert!(sys.contains("Additional documentation"));
        assert!(sys.contains("- revenue = price * qty"));
    }

    #[test]
    fn build_prompt_examples_become_alternating_turns() {
        let examples = vec![
            ("top customers?".to_string(), "SELECT 1".to_string()),
            ("worst product?".to_string(), "SELECT 2".to_string()),
        ];
        let msgs = build_sql_prompt("q", &[], &[], &examples);
        // system + (user, assistant) * 2 + final user = 6
        assert_eq!(msgs.len(), 6);
        assert_eq!(msgs[1].role, Role::User);
        assert_eq!(msgs[1].content, "top customers?");
        assert_eq!(msgs[2].role, Role::Assistant);
        assert_eq!(msgs[2].content, "```sql\nSELECT 1\n```");
        assert_eq!(msgs[5].role, Role::User);
        assert_eq!(msgs[5].content, "q"); // question always comes last
    }
}
