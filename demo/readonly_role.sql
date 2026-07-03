-- A least-privilege, read-only role for holocron - the REAL security boundary.
--
-- The AST validation gate (sql_guard) and the READ ONLY transaction are
-- defense-in-depth, but the LLM is untrusted: only the database can guarantee
-- that a generated query cannot read data it was never granted. Point
-- holocron.toml's [database].url at this role in any shared/exposed deployment.
--
--   psql "postgres://postgres:postgres@localhost:5432/holocron_demo" -f demo/readonly_role.sql
--   # then set [database].url = "postgres://holocron_ro:<pw>@localhost:5432/holocron_demo"
--
-- Adjust the schema list (sales, inventory, finance) to your analytics schemas.

BEGIN;

-- 1) The role. Use a real password (or SCRAM/cert auth) in production.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'holocron_ro') THEN
        CREATE ROLE holocron_ro LOGIN PASSWORD 'change-me';
    END IF;
END $$;

-- 2) Connect, but nothing else at the database level.
REVOKE ALL ON DATABASE holocron_demo FROM holocron_ro;
GRANT CONNECT ON DATABASE holocron_demo TO holocron_ro;

-- 3) No object creation anywhere (esp. the public schema).
REVOKE ALL ON SCHEMA public FROM holocron_ro;

-- 4) SELECT-only, and ONLY on the curated analytics schemas. Everything the
--    role wasn't explicitly granted (other schemas, catalogs' underlying data,
--    a future `users.password_hash`) is unreadable - regardless of the SQL.
GRANT USAGE ON SCHEMA sales, inventory, finance TO holocron_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA sales, inventory, finance TO holocron_ro;
-- Cover tables created later, too.
ALTER DEFAULT PRIVILEGES IN SCHEMA sales, inventory, finance
    GRANT SELECT ON TABLES TO holocron_ro;

-- 5) Belt and braces: this role is not a superuser, so file/dblink/backend
--    functions (pg_read_file, lo_import, dblink, pg_terminate_backend, COPY …
--    TO PROGRAM) are already unavailable to it. Never run holocron as a
--    superuser (the demo's postgres/postgres) against untrusted input.
--
-- Optional: prefer exposing only curated *views* and revoking SELECT on the
-- base tables, so even column names of sensitive tables stay hidden.

COMMIT;
