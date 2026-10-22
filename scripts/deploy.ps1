# Run only when a cloud deployment is intended.
param(
    [string]$ProjectId = "",
    [string]$EmbeddingModel = "sentence-transformers/all-MiniLM-L6-v2",
    [string]$EmbeddingRevision = "",
    [string]$DatabasePath = "/app/data/games.db",
    [string]$IndexPath = "/app/data/games.index"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not $ProjectId) {
    $ProjectId = (gcloud config get-value project).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $ProjectId -or $ProjectId -eq "(unset)") {
        throw "Provide -ProjectId or configure a Google Cloud project."
    }
}
gcloud builds submit --project $ProjectId --config cloudbuild.yaml "--substitutions=_EMBED_MODEL=$EmbeddingModel,_EMBED_REVISION=$EmbeddingRevision,_DB_PATH=$DatabasePath,_INDEX_PATH=$IndexPath"
if ($LASTEXITCODE -ne 0) { throw "Cloud Build deployment failed." }
gcloud run services describe gamerecapp --project $ProjectId --region europe-west1 "--format=value(status.url)"
if ($LASTEXITCODE -ne 0) { throw "Could not verify the Cloud Run service." }
npx firebase-tools deploy --only hosting --project $ProjectId
if ($LASTEXITCODE -ne 0) { throw "Firebase deployment failed; the backend was deployed." }
Write-Host "Deployment commands completed. Verify /health and browser search through Hosting."
