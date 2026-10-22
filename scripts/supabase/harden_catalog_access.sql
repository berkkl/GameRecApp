-- Prepared September 30, 2026 for the recovered GameRecApp database.
-- Reviewed baseline: RLS disabled on all three tables; anon has CRUD grants.
-- games already has Public read access, Service insert, and Service update policies.
-- game_groups and searches have no policies. No rows are modified by this script.
-- Run as the database owner. The backend must use a server-held service-role key.

BEGIN;

-- Abort before changing permissions if the trusted backend cannot retain access.
DO $$
DECLARE
    target_table text;
    operation text;
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_roles
        WHERE rolname = 'service_role' AND rolbypassrls
    ) THEN
        RAISE EXCEPTION 'service_role must retain its existing RLS bypass';
    END IF;
    FOREACH target_table IN ARRAY ARRAY[
        'public.games', 'public.game_groups', 'public.searches'
    ] LOOP
        FOREACH operation IN ARRAY ARRAY['SELECT', 'INSERT', 'UPDATE', 'DELETE'] LOOP
            IF NOT has_table_privilege('service_role', target_table, operation) THEN
                RAISE EXCEPTION 'Missing backend privilege: % on %', operation, target_table;
            END IF;
        END LOOP;
    END LOOP;
END
$$;

ALTER TABLE public.games ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.game_groups ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.searches ENABLE ROW LEVEL SECURITY;

-- Keep the existing public games SELECT policy; restrict catalog mutations.
REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER
    ON TABLE public.games, public.game_groups
    FROM PUBLIC, anon, authenticated;

-- History has no client access policy: only the trusted backend/owner can use it.
REVOKE ALL PRIVILEGES ON TABLE public.searches
    FROM PUBLIC, anon, authenticated;

-- Preserve public franchise/group reads while RLS protects their mutations.
DROP POLICY IF EXISTS "GameRec public group read" ON public.game_groups;
CREATE POLICY "GameRec public group read"
    ON public.game_groups FOR SELECT TO PUBLIC USING (true);

COMMIT;

-- Post-apply verification. These statements do not change data.
SELECT c.relname AS table_name, c.relrowsecurity AS rls_enabled,
       has_table_privilege('anon', c.oid, 'SELECT') AS anonymous_read,
       has_table_privilege('anon', c.oid, 'INSERT') AS anonymous_insert,
       has_table_privilege('anon', c.oid, 'UPDATE') AS anonymous_update,
       has_table_privilege('anon', c.oid, 'DELETE') AS anonymous_delete
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public'
  AND c.relname IN ('games', 'game_groups', 'searches')
ORDER BY c.relname;

SELECT count(*) AS games,
       count(*) FILTER (WHERE review_count >= 50 AND rating_pct > 60)
           AS meets_review_threshold
FROM public.games;
