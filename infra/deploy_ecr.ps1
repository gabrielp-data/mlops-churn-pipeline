param(
    [Parameter(Mandatory=$true)][string]$AccountId,
    [string]$Region = "us-east-2",
    [string]$Repository = "mlops-churn",
    [Parameter(Mandatory=$true)][string]$Tag,
    [string]$AwsProfile
)
$ErrorActionPreference = "Stop"
if ($AccountId -notmatch '^\d{12}$') { throw "AccountId precisa ter 12 dígitos." }
if ($Tag -notmatch '^[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,127}$') { throw "Tag inválida." }
$profileArgs = @()
if ($AwsProfile) { $profileArgs = @("--profile", $AwsProfile) }
Push-Location (Join-Path $PSScriptRoot "..")
try {
    uv run python -m churn.export_champion
    if ($LASTEXITCODE -ne 0) { throw "Falha ao exportar campeão." }
    $registry = "$AccountId.dkr.ecr.$Region.amazonaws.com"
    $image = "${registry}/${Repository}:${Tag}"
    # O repositório ECR deve existir; o script não cria infraestrutura.
    aws ecr describe-repositories --repository-names $Repository --region $Region @profileArgs | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Repositório ECR indisponível." }
    $password = aws ecr get-login-password --region $Region @profileArgs
    if ($LASTEXITCODE -ne 0) { throw "Falha na autenticação AWS." }
    $password | docker login --username AWS --password-stdin $registry
    if ($LASTEXITCODE -ne 0) { throw "Falha no login ECR." }
    docker build --target serving --tag $image .
    if ($LASTEXITCODE -ne 0) { throw "Falha no build." }
    docker push $image
    if ($LASTEXITCODE -ne 0) { throw "Falha no push." }
    Write-Output "Imagem publicada: $image"
} finally {
    Pop-Location
}
