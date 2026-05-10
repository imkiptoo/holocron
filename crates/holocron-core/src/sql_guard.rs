//! Deterministic SQL validation gate — the security boundary in front of query
//! execution.
//!
//! The LLM is an **untrusted** SQL generator (its input is untrusted natural
//! language, potentially prompt-injected), so its output is parsed and checked
//! against an allow/deny policy *before* it ever reaches the database — never
//! trusting the model's own "I won't do that". This replaces the old first-token
//! `is_read_only` check, which passed writable CTEs (`WITH … DELETE`) and
//! file-reading functions (`SELECT pg_read_file(…)`).
//!
//! This is defense-in-depth, **not** the primary boundary — a least-privilege,
//! read-only DB role granted `SELECT` on only the analytics schemas is. The gate
//! adds a deterministic layer and good error messages on top of that role.

use std::ops::ControlFlow;

use sqlparser::ast::{visit_expressions, visit_relations, visit_statements, Expr, Statement};
use sqlparser::dialect::PostgreSqlDialect;
use sqlparser::parser::Parser;

/// Policy the [`validate`] gate enforces.
#[derive(Debug, Clone, Default)]
pub struct SqlPolicy {
    /// Allow references to `information_schema` / `pg_catalog` / `pg_*`. Off by
    /// default (`false`): schema enumeration is reconnaissance and those catalogs
    /// can expose more than the analytics data.
    pub allow_system_schemas: bool,
}

/// Postgres functions that can read files, reach other systems, or control the
/// server — never appropriate in a generated analytics query.
const DENIED_FUNCTIONS: &[&str] = &[
    "pg_read_file",
    "pg_read_binary_file",
    "pg_ls_dir",
    "pg_stat_file",
    "lo_import",
    "lo_export",
    "lo_get",
    "dblink",
    "dblink_exec",
    "dblink_open",
    "dblink_connect",
    "pg_sleep",
    "pg_terminate_backend",
    "pg_cancel_backend",
    "copy",
];

const SYSTEM_SCHEMAS: &[&str] = &["information_schema", "pg_catalog", "pg_toast"];

/// Validate generated SQL against `policy`. Returns `Ok(())` if it is a single,
/// read-only `SELECT` touching only permitted objects; otherwise an `Err` with a
/// user-facing reason. Fails **closed**: SQL that doesn't parse is rejected.
pub fn validate(sql: &str, policy: &SqlPolicy) -> Result<(), String> {
    let statements = Parser::parse_sql(&PostgreSqlDialect {}, sql)
        .map_err(|e| format!("could not parse SQL: {e}"))?;

    // Exactly one statement — no stacked `SELECT …; DROP …`.
    match statements.len() {
        1 => {}
        0 => return Err("no statement found".into()),
        n => return Err(format!("expected exactly one statement, found {n}")),
    }
    if !matches!(statements[0], Statement::Query(_)) {
        return Err("only a single read-only SELECT query is allowed".into());
    }

    // No write/DDL statement *anywhere* — this catches writable CTEs, whose
    // INSERT/UPDATE/DELETE body is a nested statement in the AST.
    if let ControlFlow::Break(reason) = visit_statements(&statements, |s| match s {
        Statement::Query(_) => ControlFlow::Continue(()),
        _ => ControlFlow::Break(
            "only read-only SELECT is allowed (found a data-modifying or DDL statement)"
                .to_string(),
        ),
    }) {
        return Err(reason);
    }

    // No references to system catalogs unless explicitly allowed. Reject both
    // qualified system schemas (`information_schema.columns`, `pg_catalog.*`) and
    // unqualified system relations reachable via `search_path` (`pg_stat_activity`,
    // `pg_tables`, …) — Postgres reserves the `pg_` prefix for system objects.
    if !policy.allow_system_schemas {
        if let ControlFlow::Break(reason) = visit_relations(&statements, |name| {
            let full = name.to_string();
            if let Some(schema) = schema_of(&full) {
                if SYSTEM_SCHEMAS.contains(&schema.as_str()) || schema.starts_with("pg_") {
                    return ControlFlow::Break(format!(
                        "access to the system schema `{schema}` is not allowed"
                    ));
                }
            }
            let table = last_segment(&full).to_lowercase();
            if table.starts_with("pg_") {
                return ControlFlow::Break(format!(
                    "access to the system relation `{table}` is not allowed"
                ));
            }
            ControlFlow::Continue(())
        }) {
            return Err(reason);
        }
    }

    // No file/dblink/server-control functions.
    if let ControlFlow::Break(reason) = visit_expressions(&statements, |e| {
        if let Expr::Function(f) = e {
            let fname = last_segment(&f.name.to_string()).to_lowercase();
            if DENIED_FUNCTIONS.contains(&fname.as_str()) {
                return ControlFlow::Break(format!("function `{fname}` is not allowed"));
            }
        }
        ControlFlow::Continue(())
    }) {
        return Err(reason);
    }

    Ok(())
}

/// Schema part of a dotted object name (`schema.table` or `db.schema.table`),
/// lower-cased and unquoted. `None` for an unqualified name.
fn schema_of(dotted: &str) -> Option<String> {
    let parts: Vec<&str> = dotted.split('.').collect();
    if parts.len() < 2 {
        return None;
    }
    Some(parts[parts.len() - 2].trim_matches('"').to_lowercase())
}

/// Last dotted segment (the function/object name), unquoted.
fn last_segment(dotted: &str) -> String {
    dotted.rsplit('.').next().unwrap_or(dotted).trim_matches('"').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(sql: &str) {
        assert!(validate(sql, &SqlPolicy::default()).is_ok(), "should allow: {sql}");
    }
    fn rejected(sql: &str) {
        assert!(validate(sql, &SqlPolicy::default()).is_err(), "should reject: {sql}");
    }

    #[test]
    fn allows_plain_selects() {
        ok("SELECT 1");
        ok("SELECT count(*) FROM sales.orders WHERE total > 10");
        // Window functions, joins, group by, order by, limit — real generated SQL.
        ok("SELECT p.name, SUM(oi.quantity * oi.unit_price) AS rev, \
            PERCENT_RANK() OVER (ORDER BY SUM(oi.quantity * oi.unit_price) DESC) \
            FROM sales.products p JOIN sales.order_items oi ON p.product_id = oi.product_id \
            GROUP BY p.product_id, p.name ORDER BY rev DESC LIMIT 10");
        // Read-only CTE is fine.
        ok("WITH t AS (SELECT 1 AS x) SELECT * FROM t");
        // The model's safe error fallback.
        ok("SELECT 'Error: not available' AS error_message");
    }

    #[test]
    fn rejects_writes_and_ddl() {
        rejected("DELETE FROM customers");
        rejected("UPDATE users SET admin = true");
        rejected("INSERT INTO t VALUES (1)");
        rejected("DROP TABLE customers");
        rejected("TRUNCATE customers");
        rejected("GRANT SELECT ON t TO public");
    }

    #[test]
    fn rejects_writable_cte() {
        // First token is `with` — the old check passed this.
        rejected("WITH x AS (DELETE FROM t RETURNING *) SELECT * FROM x");
        rejected("WITH x AS (INSERT INTO t VALUES (1) RETURNING *) SELECT * FROM x");
    }

    #[test]
    fn rejects_stacked_statements() {
        rejected("SELECT 1; DROP TABLE customers");
    }

    #[test]
    fn rejects_system_schema_access() {
        rejected("SELECT * FROM information_schema.columns");
        rejected("SELECT count(*) FROM information_schema.tables");
        rejected("SELECT * FROM pg_catalog.pg_user");
        rejected("SELECT * FROM pg_stat_activity"); // pg_ prefix
    }

    #[test]
    fn allows_system_schema_when_policy_permits() {
        let policy = SqlPolicy { allow_system_schemas: true };
        assert!(validate("SELECT count(*) FROM information_schema.tables", &policy).is_ok());
    }

    #[test]
    fn rejects_dangerous_functions() {
        rejected("SELECT pg_read_file('/etc/passwd')");
        rejected("SELECT * FROM dblink('host=x', 'SELECT 1') AS t(a int)");
        rejected("SELECT pg_sleep(10)");
        rejected("SELECT lo_import('/etc/passwd')");
    }

    #[test]
    fn rejects_unparseable() {
        rejected("this is not sql");
        rejected("SELECT FROM WHERE");
    }
}
