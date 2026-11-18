# Deploy AWS: aplicação e PostgreSQL

O repositório prepara a implantação; nenhuma conta ou instância é criada automaticamente.

## Arquitetura

- ECR: imagem com FastAPI, Streamlit e pacote imutável do campeão.
- EC2 de aplicação: executa a imagem; consulta o banco pelo IP privado.
- EC2 de dados: PostgreSQL 16 com volume persistente.
- A exportação preserva versão, run, contrato, métricas e hash dos dados.
- Alterar o alias no registry não altera uma imagem já publicada: exporte e publique outra tag.

## Pré-requisitos

Docker e AWS CLI instalados nas máquinas correspondentes; duas EC2 Linux na mesma VPC.
Na aplicação, associe uma IAM Role com leitura do ECR, usando a trust policy deste diretório.
Na máquina de publicação, use um perfil AWS com acesso de push ao repositório ECR.

Configure o Security Group do banco para aceitar 5432 somente do Security Group da aplicação.
Para demonstração, restrinja 8501 ao seu IP. A API fica vinculada a localhost na EC2.
O exemplo não implementa autenticação de usuários nem TLS; publicação ampla requer um proxy autenticado com HTTPS.

## 1. Banco

Crie um arquivo protegido (chmod 600) na EC2 de dados:

```dotenv
POSTGRES_DB=churn
POSTGRES_USER=churn
POSTGRES_PASSWORD=uma-senha-longa-e-aleatoria
```

```bash
DB_ENV_FILE=$HOME/postgres.env PRIVATE_IP=10.0.1.10 bash infra/setup_db_ec2.sh
```

O script não apaga banco existente. A tabela é criada pelo importador Python.

## 2. Publicação

Depois de treinar e promover um campeão, crie o repositório ECR na sua conta.
Na máquina local com Docker, uv e AWS CLI:

```powershell
.\infra\deploy_ecr.ps1 -AccountId 123456789012 -Region us-east-2 -Repository mlops-churn -Tag v0.2.0 -AwsProfile meu-perfil
```

O script exporta o campeão, constrói e publica a imagem. Use uma tag nova a cada publicação.

## 3. Importação e aplicação

Na EC2 da aplicação, crie um arquivo chmod 600:

```dotenv
CHURN_DATABASE_URL=postgresql+psycopg://churn:SENHA_URL_ENCODED@10.0.1.10:5432/churn
CHURN_API_URL=http://127.0.0.1:8000
```

Faça login no ECR com a role e baixe a imagem:

```bash
export AWS_REGION=us-east-2
export IMAGE_URI=123456789012.dkr.ecr.us-east-2.amazonaws.com/mlops-churn:v0.2.0
export ENV_FILE=$HOME/churn.env
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "${IMAGE_URI%%/*}"
docker pull "$IMAGE_URI"
```

Transfira o CSV versionado para a EC2 da aplicação e importe antes de iniciar o serviço:

```bash
docker run --rm --env-file "$ENV_FILE" -v "$PWD/data:/input:ro" "$IMAGE_URI" python -m churn.db --input /input/churn.csv
bash infra/setup_ec2.sh
```

A carga é transacional e não sobrescreve IDs existentes. Não repita em uma tabela já carregada.
Acesse http://IP_DA_APLICACAO:8501 a partir do IP permitido no Security Group.

## Verificação e rollback

Confira /health no localhost:8000 da EC2 e /model-info para identificar o run servido.
O script aguarda os health checks de API e painel e retorna erro se falharem.
Para rollback, execute setup_ec2.sh com IMAGE_URI apontando para a tag anterior.
O volume do PostgreSQL é independente do container da aplicação; mantenha backups do banco.
