# Build local da imagem da aplicação + push para o Amazon ECR (Windows / PowerShell).
#   .\infra\deploy_ecr.ps1 -AwsProfile churn
param (
    [string]$Region = "us-east-2",
    [string]$RepoName = "app-churn",
    [string]$AwsProfile = ""   # perfil criado com `aws configure --profile churn`
)

$ErrorActionPreference = "Stop"
$ProfileArgs = if ($AwsProfile) { @("--profile", $AwsProfile) } else { @() }

if (-not (Test-Path "data/churn.csv")) { throw "data/churn.csv ausente. Rode 'uv run dvc pull' antes do build." }

Write-Host "--> Obtendo Account ID da AWS..." -ForegroundColor Cyan
$AccountId = (aws sts get-caller-identity --query Account --output text @ProfileArgs).Trim()
$EcrUri = "$AccountId.dkr.ecr.$Region.amazonaws.com"
$Image = "$EcrUri/$($RepoName):latest"

Write-Host "--> 0. Exportando o @champion do MLflow Model Registry..." -ForegroundColor Cyan
uv run python -m churn.registry --export
if ($LASTEXITCODE -ne 0) { throw "Falha ao exportar o @champion. Rode churn.experiments e churn.registry antes." }

Write-Host "--> 1. Garantindo o repositório $RepoName no ECR..." -ForegroundColor Cyan
aws ecr describe-repositories --region $Region --repository-names $RepoName @ProfileArgs *> $null
if ($LASTEXITCODE -ne 0) { aws ecr create-repository --region $Region --repository-name $RepoName @ProfileArgs | Out-Null }

Write-Host "--> 2. Autenticando o Docker no ECR ($Region)..." -ForegroundColor Cyan
aws ecr get-login-password --region $Region @ProfileArgs | docker login --username AWS --password-stdin $EcrUri

Write-Host "--> 3. Build da imagem (Dockerfile.app)..." -ForegroundColor Cyan
docker build -f Dockerfile.app -t "$($RepoName):latest" .
docker tag "$($RepoName):latest" $Image

Write-Host "--> 4. Push para o ECR..." -ForegroundColor Cyan
docker push $Image

Write-Host "`nPublicação concluída com sucesso! URI: $Image" -ForegroundColor Green
