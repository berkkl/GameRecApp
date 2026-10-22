# Deployment

Build and validate the catalog/index locally before deployment. Git ignores data, so a build from a clean checkout will not contain the catalog.

Use service `gamerecapp`, region `europe-west1`, and Artifact Registry repository `gamerec` consistently. The historical Container Registry backend stopped accepting writes in 2025; migrated gcr.io repositories remain supported. This configuration uses explicit Artifact Registry paths. [Google transition documentation](https://docs.cloud.google.com/artifact-registry/docs/transition/transition-from-gcr).

## Prerequisites

Install Docker, Google Cloud CLI, and Firebase CLI. Select your own project and billing configuration. Provision the Docker-format `gamerec` Artifact Registry repository in `europe-west1` and enable Cloud Run, Artifact Registry, and Cloud Build APIs. The build/deploy identity needs repository write, Cloud Run deploy, and service-account act-as permissions. Provision these intentionally; the scripts do not grant IAM roles.

The setup script provisions APIs and the repository when explicitly run:

```bash
./scripts/setup_gcp.sh YOUR_PROJECT_ID
```

## Build and run locally

For the legacy catalog:

```bash
docker build -t gamerecapp .
docker run --rm -p 8000:8000 -v "$(pwd)/data:/app/data" gamerecapp
```

For multilingual RAG, build with the same encoder and exact revision recorded in the completed index. Supply the matching database and index at runtime. This example uses a locally validated DuckDB bundle; change all paths together for a different snapshot:

```bash
docker build --build-arg EMBED_MODEL=intfloat/multilingual-e5-small --build-arg EMBED_REVISION=614241f622f53c4eeff9890bdc4f31cfecc418b3 -t gamerecapp .
docker run --rm -p 8000:8000 -v "$(pwd)/data:/app/data" -e DB_PATH=/app/data/catalogs/my-snapshot/games.duckdb -e INDEX_PATH=/app/data/catalogs/my-snapshot/rag/games.index gamerecapp
```

The image caches the build-time encoder and runs offline. It does not ingest or create sample data. Check `/health`, `/datasets`, and a real `/recommend` request before deployment. Health JSON must report `components_loaded: true`; a running process alone is insufficient.

## Cloud Build and Firebase

`cloudbuild.yaml` builds/pushes to Artifact Registry and deploys Cloud Run. Its defaults use the legacy MiniLM index. To deploy RAG, pass non-secret substitutions:

```bash
gcloud builds submit --project YOUR_PROJECT_ID --config cloudbuild.yaml --substitutions=_EMBED_MODEL=intfloat/multilingual-e5-small,_EMBED_REVISION=614241f622f53c4eeff9890bdc4f31cfecc418b3,_DB_PATH=/app/data/catalogs/my-snapshot/games.duckdb,_INDEX_PATH=/app/data/catalogs/my-snapshot/rag/games.index
firebase deploy --only hosting --project YOUR_PROJECT_ID
```

The catalog/index upload from the local workspace is intentional. Review `.gcloudignore` and `.dockerignore` before a build; credential files, search histories/logs, and audit snapshots are excluded. Do not submit a SQLite catalog containing private feedback; use a clean deployment export. Container writes are ephemeral and are not durable feedback storage.

Firebase API rewrites point to `gamerecapp` in `europe-west1`, preserving same-origin requests. Update all configs together if changing the region or service. This follows [Firebase's Cloud Run rewrite configuration](https://firebase.google.com/docs/hosting/cloud-run).

`scripts/deploy.ps1` invokes Cloud Build followed by Firebase. `scripts/deploy_gcp.sh` delegates to the same Cloud Build configuration. These commands modify cloud resources and should be run only when deployment is intended.

## Runtime credentials and rollback

Optional translation/AI feedback and Supabase history need runtime credentials from Secret Manager. Never put secret values in build substitutions or tracked environment files. History/logs/cloud history remain off by default. Remote history reads need an explicit bearer token; a proxy's loopback connection must not be mistaken for an end-user loopback client.

List revisions using your project and `--region europe-west1`. Restore traffic to a previously validated revision for rollback, and keep its matching catalog/model artifacts. Check the browser's recommendation and feedback flows through Hosting after deployment. Deployment and external-service permissions remain unverified until those checks are run.
