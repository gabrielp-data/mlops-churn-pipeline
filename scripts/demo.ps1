param([switch]$TrainOnly)
$ErrorActionPreference = "Stop"
Push-Location (Join-Path $PSScriptRoot "..")
$previousDataPath = $env:CHURN_DATA_PATH
try {
    $env:CHURN_DATA_PATH = "data/telco-public.csv"
    uv sync --frozen
    if ($LASTEXITCODE -ne 0) { throw "Falha ao preparar ambiente." }
    foreach ($module in @("churn.fetch_data", "churn.model", "churn.registry", "churn.export_champion", "churn.predict")) {
        uv run python -m $module
        if ($LASTEXITCODE -ne 0) { throw "Falha em $module." }
    }
    if (-not $TrainOnly) {
        Write-Output "Painel: http://localhost:8501 | API: http://localhost:8000/docs"
        uv run python -m churn.serve
        if ($LASTEXITCODE -ne 0) { throw "Falha ao iniciar aplicação." }
    }
} finally {
    $env:CHURN_DATA_PATH = $previousDataPath
    Pop-Location
}
