#!/bin/bash
# Delegate to the same configuration used by PowerShell.
set -euo pipefail
PROJECT_ID="${1:?Usage: deploy_gcp.sh PROJECT_ID}"
gcloud builds submit --project "$PROJECT_ID" --config cloudbuild.yaml
gcloud run services describe gamerecapp --project "$PROJECT_ID" --region europe-west1 --format 'value(status.url)'
