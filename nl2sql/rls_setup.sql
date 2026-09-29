-- nl2sql/rls_setup.sql
--
-- Step 11 of the build sequence: row-level access control, enforced by
-- Postgres itself (native Row-Level Security), not by anything our
-- Python code decides to add to a query. Once enabled, ANY query against
-- `orders` - whether it came from the semantic layer's fixed templates or
-- the LLM's free-form SQL generation - is silently restricted to the
-- allowed rows by Postgres, before our app ever sees the result.
--
-- Run once, manually, against the northwind database:
--     psql -U nexus_user -d northwind -f nl2sql/rls_setup.sql

ALTER TABLE orders ENABLE ROW LEVEL SECURITY;

-- IMPORTANT: RLS does NOT apply to a table's OWNER by default - and our
-- app connects as nexus_user, which owns this table (it ran the restore).
-- Without FORCE, nexus_user would silently bypass every policy below,
-- and RLS would look like it's "on" while doing nothing at all.
ALTER TABLE orders FORCE ROW LEVEL SECURITY;

-- One policy: an "admin" session (app.user_role = 'admin') sees every
-- row; any other session only sees orders where employee_id matches its
-- own app.current_employee_id. If NEITHER session variable was ever set
-- (a connection that forgot to identify itself), this evaluates to false
-- for every row - default-deny, not an error and not "see everything".
--
-- NOTE 1: the variable is named "app.user_role", NOT "app.current_role" -
-- CURRENT_ROLE is a reserved SQL keyword, and Postgres's parser rejects
-- `SET app.current_role = ...` even though it's namespaced with "app.".
--
-- NOTE 2: NULLIF(..., '')::int, NOT a separate "<> ''" guard before the
-- cast - Postgres's AND/OR do not GUARANTEE left-to-right short-circuit
-- evaluation order for every query plan, so a guard clause written as
-- "x <> '' AND employee_id = x::int" can still attempt the cast on an
-- empty string in some plans (confirmed - this actually happened, even
-- for an admin session where the OR's left side was already true).
-- NULLIF converts '' to a real NULL BEFORE the cast ever runs, so
-- casting can never fail no matter how the planner orders evaluation.
DROP POLICY IF EXISTS orders_access_policy ON orders;

CREATE POLICY orders_access_policy ON orders
USING (
    current_setting('app.user_role', true) = 'admin'
    OR employee_id = NULLIF(current_setting('app.current_employee_id', true), '')::int
);
