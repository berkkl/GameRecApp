#!/bin/bash
# Provision only when explicitly preparing a deployment.
set -euo pipefail
PROJECT_ID="${1:?Usage: setup_gcp.sh PROJECT_ID}"
REGION="europe-west1"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com --project "$PROJECT_ID"
if ! gcloud artifacts repositories describe gamerec --location "$REGION" --project "$PROJECT_ID" >/dev/null 2>&1; then
    gcloud artifacts repositories create gamerec --repository-format docker --location "$REGION" --project "$PROJECT_ID"
fi
gcloud auth configure-docker "$REGION-docker.pkg.dev"
echo "Validate your catalog/index and build/deploy permissions before deploying."
