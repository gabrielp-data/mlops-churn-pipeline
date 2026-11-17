# Infraestrutura e implantação na AWS

Scripts para reproduzir a arquitetura do projeto na AWS (região `us-east-2`).

| Arquivo | Função |
| --- | --- |
| `../Dockerfile.app` + `../supervisord.conf` | Imagem única da aplicação: FastAPI (:8000) + Streamlit (:8501) |
| `deploy_ecr.sh` / `deploy_ecr.ps1` | Exporta o `@champion`, faz o build e o push para o Amazon ECR (Linux / Windows) |
| `iam_trust_ec2.json` | Trust policy da role `EC2-ECR-ReadRole` (a EC2 lê o ECR sem chaves fixas) |
| `setup_ec2.sh` | Deploy da **camada de aplicação** na EC2 `churn-app-host` |
| `setup_db_ec2.sh` | Provisiona a **camada de dados** na EC2 `churn-db-host` (Postgres em Docker + carga do CSV) |
| `db/init_customers.sql` | DDL da tabela `customers` + `\copy` do `churn.csv` (CSV → postgres) |

---

## 1. Evolução da arquitetura

| Etapa | App | Onde ficam os dados dos clientes | Infra |
| --- | --- | --- | --- |
| 1 | app-churn-v1 | `data/batch_predictions.parquet` embutido na imagem (só consulta, sem inferência) | 1 EC2 |
| 2 | app-churn-v2 | parquet (consulta) + inferência online via `POST /predict` | 1 EC2 |
| **3** | **app-churn-v2** | **`churn.csv` bruto no PostgreSQL em Docker numa 2ª EC2**; a API faz `SELECT` pela rede | **2 EC2** |

A mesma imagem serve as duas versões, escolhidas por variável de ambiente:

| versão | `API_MODULE` | `FRONTEND` |
| --- | --- | --- |
| v1 | `churn.api_lookup` | `src/churn/app_churn_v1.py` |
| v2 (padrão) | `churn.api_v2` | `src/churn/app_churn_v2.py` |

## 2. Topologia em duas camadas

```text
[Navegador]
     │ HTTP :8501
     ▼
┌── AWS us-east-2 (VPC padrão) ─────────────────────────────────────────────────┐
│                                                                               │
│  EC2 churn-app-host                              EC2 churn-db-host            │
│  role IAM EC2-ECR-ReadRole (sem chaves fixas)                                 │
│  ┌── container churn_app ────────────┐          ┌── container churn_postgres ┐│
│  │ Supervisord                       │  SQL     │ PostgreSQL 15              ││
│  │  ├─ Streamlit :8501 app_churn_v2  │  :5432   │ banco churn_db             ││
│  │  └─ FastAPI   :8000 api_v2 ───────┼─────────►│ tabela customers           ││
│  │     + @champion em memória        │ IP priv. │ volume pgdata              ││
│  └───────────────────────────────────┘          └────────────────────────────┘│
│                ▲ docker pull                                                  │
│        Amazon ECR: app-churn:latest                                           │
└───────────────────────────────────────────────────────────────────────────────┘
```

As máquinas conversam pelo **IP privado** da VPC. A senha do banco nunca vai para a
imagem nem para o Git: entra no container em tempo de execução via `--env-file`.

## 3. Endpoints da API v2 (`src/churn/api_v2.py`)

| Método | Rota | O que faz |
| --- | --- | --- |
| GET | `/health` | status do modelo e da conexão com o Postgres (com contagem de registros) |
| GET | `/model-info` | versão, run, métricas e linhagem do `@champion` |
| GET | `/customers/{customerID}` | consulta pura: registro do cliente como está no banco |
| GET | `/results/{customerID}` | busca o cliente no banco e calcula o score do `@champion` |
| POST | `/predict` | inferência online para cliente novo (não depende do banco) |

Banco fora do ar → as consultas respondem `503` e o `/predict` continua funcionando.
Registro fora do contrato do modelo (ex.: `Contract = "Vitalicio"`) → `422`.

---

## 4. Passo a passo

### 4.0 Uma vez: role IAM e repositório
```bash
aws iam create-role --role-name EC2-ECR-ReadRole \
  --assume-role-policy-document file://infra/iam_trust_ec2.json
aws iam attach-role-policy --role-name EC2-ECR-ReadRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
aws iam create-instance-profile --instance-profile-name EC2-ECR-ReadRole
aws iam add-role-to-instance-profile --instance-profile-name EC2-ECR-ReadRole --role-name EC2-ECR-ReadRole
```
Associe o instance profile à EC2 `churn-app-host` (Ações → Segurança → Modificar função do IAM).
O repositório do ECR é criado pelo próprio `deploy_ecr` se não existir.

### 4.1 Camada de dados (`churn-db-host`)
1. EC2 `t3.micro` Ubuntu, mesma VPC. Security Group com a porta **5432** liberada só para o
   Security Group da aplicação (evite `0.0.0.0/0`).
2. Clone o repositório e coloque o `churn.csv` em `data/`.
3. Rode:
   ```bash
   export DB_PASSWORD='sua-senha'
   bash infra/setup_db_ec2.sh
   ```
   Saída esperada: `COPY 7043`.

### 4.2 Imagem da aplicação (sua máquina)
```bash
uv run dvc pull                                # garante data/churn.csv
uv run python -m churn.experiments             # grid → mlflow.db
uv run python -m churn.registry                # melhor run → @champion
AWS_PROFILE=churn bash infra/deploy_ecr.sh     # export + build + push  (Windows: .\infra\deploy_ecr.ps1 -AwsProfile churn)
```
Espere `Publicação concluída com sucesso!` antes do 4.3 — se a EC2 baixar a imagem antes do
push terminar, ela recebe a versão anterior (confira o `Digest:` do pull).

### 4.3 Camada de aplicação (`churn-app-host`)
1. EC2 `t3.micro` Ubuntu com a role `EC2-ECR-ReadRole`; Security Group com a **8501** aberta.
2. Crie `~/churn.env` (modelo em `.env.example`) com `DB_HOST=<IP privado da churn-db-host>` e a senha.
3. Copie o `infra/setup_ec2.sh` para a máquina e rode `bash setup_ec2.sh`.
   O script sobe o container e imprime o `/health`, que deve mostrar o `@champion` e `"records": 7043`.
4. Acesse `http://<IP público da churn-app-host>:8501`.

Pouco disco (~7 GB na `t3.micro`)? Antes de um novo deploy: `sudo docker system prune -a -f`.

## 5. Teste local

**Com Docker (as duas camadas):**
```bash
uv run python -m churn.registry --export
docker compose up --build app-v2       # Postgres + app; painel em http://localhost:8502
```

**Sem Docker, apontando para o Postgres da AWS:**
```bash
cp .env.example .env                   # DB_HOST = IP PÚBLICO da churn-db-host + senha
uv run python -m churn.registry --export
uv run uvicorn churn.api_v2:app --port 8000 --reload
uv run streamlit run src/churn/app_churn_v2.py
```
Para testar de fora da AWS, a regra da 5432 precisa aceitar o seu IP público.

## 6. Custos

Instâncias `t3.micro` consomem créditos enquanto estão **em execução**. Pare as duas ao final.
Ao religar, o IP **público** muda, mas o **privado** se mantém — o `DB_HOST` da aplicação continua válido.
