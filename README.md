# prj_churn · Curso de MLOps — IESB

Projeto-fio-condutor do curso: previsão de **churn** (classificação binária tabular).  
O arco é do `train_churn.py` monolítico até um sistema completo de MLOps.

---

## O dado e o problema

Classificação binária: prever se um cliente vai cancelar o serviço (`Churn = Yes/No`).
Dataset e modelo base (Random Forest) fixados pela disciplina.

| característica | valor |
| --- | --- |
| arquivo | `data/churn.csv` (versionado com DVC, não com Git) |
| linhas × colunas | 7.043 clientes × 21 colunas |
| identificador | `customerID` (único; ex.: `7590-VHVEG`) |
| alvo | `Churn`: 4.752 `No` / 2.291 `Yes` (≈ 32,5 % de churn) |
| numéricas | `tenure` (0–72 meses), `MonthlyCharges`, `TotalCharges` (texto no CSV; 11 em branco) |
| categóricas | 16 colunas (contrato, serviços, forma de pagamento etc.) |

---

## Estrutura

```
prj_churn/
├─ src/churn/
│  ├─ config.py        # Settings (Pydantic) — caminhos, hiperparâmetros
│  ├─ data.py          # carga, limpeza, validação (pandera)
│  ├─ schema.py        # ChurnSchema — contrato de dados
│  ├─ profiling.py     # retrato do dataset (tipos, estatísticas, nulos, alvo)
│  ├─ schema_demo.py   # quebra proposital do contrato (pandera barrando dado ruim)
│  ├─ features.py      # transformações puras + encoder sem vazamento
│  ├─ model.py         # Pipeline sklearn (features → encoder → RF) + MLflow
│  ├─ evaluate.py      # métricas
│  ├─ experiments.py   # grid rastreado (MLflow)
│  ├─ lineage.py       # data_md5() e git_commit()
│  ├─ registry.py      # Model Registry — @champion, metadados e export
│  ├─ test_champion.py # consumo de models:/churn@champion (pyfunc)
│  ├─ predict.py       # inferência em lote via @champion (pyfunc)
│  ├─ app.py           # API FastAPI servindo o @champion
│  ├─ dashboard.py     # painel Streamlit que consome a API
│  ├─ scoring.py       # AWS: ChampionModel (export) + contrato de entrada + risco
│  ├─ batch_scoring.py # AWS v1: pontuação em lote → parquet
│  ├─ api_lookup.py    # AWS v1: API que só consulta o parquet
│  ├─ app_churn_v1.py  # AWS v1: tela de consulta
│  ├─ db.py            # AWS v2: conexão com o PostgreSQL via variáveis de ambiente
│  ├─ api_v2.py        # AWS v2: SELECT no PostgreSQL + inferência online
│  └─ app_churn_v2.py  # AWS v2: tela (consulta no banco + nova inferência)
├─ tests/              # pytest (data, features, schema, registry, app, apps da AWS)
├─ infra/              # deploy AWS: ECR, EC2 da app, EC2 do Postgres (ver infra/README.md)
├─ data/
│  ├─ churn.csv.dvc    # ponteiro DVC (o .csv fica no remote)
│  └─ .gitignore
├─ Dockerfile          # imagem com uv (enc. 3)
├─ Dockerfile.app      # imagem da AWS: FastAPI + Streamlit via Supervisord
├─ supervisord.conf    # processos do container da AWS (backend :8000, frontend :8501)
├─ .env.example        # modelo das variáveis do banco (o .env real é ignorado)
├─ compose.yml         # treino, grid, api, painel + postgres e app-v2 (AWS local)
├─ dvc.yaml            # treino_e_tuning → promover_modelo
└─ pyproject.toml      # dependências travadas (uv)
```

---

## Bloco 0 — Fundação (Encontros 1–3)

### Rodando localmente

```bash
uv sync                              # cria o ambiente a partir do pyproject
uv run python -m churn.model         # treina e imprime métricas
uv run pytest                        # testa
```

### Rodando no container (Docker)

```bash
docker build -t churn:0.1 .
docker run -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts churn:0.1
```

Ou com compose:

```bash
docker compose up treino
```

---

## Bloco 1 — Data Pipeline (Encontros 4–5)

### Encontro 4 — Pandera

```bash
uv run python -m churn.profiling     # retrato do dado (tipos, estatísticas, nulos, alvo)
uv run python -m churn.schema_demo   # quebra 3 linhas reais de propósito e mostra o pandera barrando
```

Saída esperada do `schema_demo`:

```text
customerID duplicado: ['7590-VHVEG']
        column                                            check failure_case
        tenure                        less_than_or_equal_to(72)          999
      Contract isin(['Month-to-month', 'One year', 'Two year'])    Vitalicio
MonthlyCharges                      greater_than_or_equal_to(0)        -15.0
```

O `data.py` valida o dataset antes do treino com o `ChurnSchema` (pandera).  
Dado ruim para o pipeline **cedo e com barulho**, não silenciosamente após a métrica cair.

```python
# para forçar falha e ver o contrato em ação:
from churn.schema import ChurnSchema
ChurnSchema.validate(df_ruim, lazy=True)  # SchemaErrors com todas as violações
```

### Encontro 5 — DVC

O `churn.csv` não vai para o Git — só o ponteiro `churn.csv.dvc` vai.  
Os bytes ficam em um remote (local ou S3).

```bash
dvc pull           # baixa o dado do remote
dvc repro          # re-executa só o que mudou (dado ou código)
dvc push           # sobe a nova versão do dado
```

---

## Bloco 2 — ML Pipeline (Encontros 6–8)

### Encontro 6 — MLflow Tracking

```bash
uv run python -m churn.model          # um run rastreado
uv run python -m churn.experiments    # grid: 9 combinações (n_estimators × max_depth)
uv run mlflow ui                      # UI em localhost:5000
```

Achar o melhor run por código:

```python
import mlflow
runs = mlflow.search_runs(experiment_names=["churn"], order_by=["metrics.roc_auc DESC"])
print(runs.iloc[0][["params.n_estimators", "params.max_depth", "metrics.roc_auc"]])
```

> **Dica:** use `--backend-store-uri sqlite:///mlflow.db` para um backend persistente:
> ```bash
> MLFLOW_TRACKING_URI=sqlite:///mlflow.db uv run python -m churn.model
> MLFLOW_TRACKING_URI=sqlite:///mlflow.db uv run mlflow ui
> ```

### Encontro 7 — Model Registry

Registra a melhor run e promove o alias `@champion`:

```bash
uv run python -m churn.registry
```

Saída esperada:
```
Melhor run: abc12345  roc_auc=0.8195
Modelo registrado → versão 1
Alias @champion → v1

Versões registradas:
  v1  run=abc12345  status=READY  aliases=['champion']
```

Consumir o campeão como faria qualquer cliente (3 clientes reais da base):

```bash
uv run python -m churn.test_champion
```

Carregar o campeão em código:

```python
from churn.registry import load_champion
model = load_champion()            # pyfunc — agnóstico de framework
preds = model.predict(features_df)   # dado cru: o pipeline faz o pré-processamento
```

### Encontro 8 — Empacotamento (MLflow Models)

O modelo registrado é um **`sklearn.Pipeline` completo**, não só o Random Forest:

```
dado cru (19 colunas) → features derivadas → encoding das categóricas → RandomForest
```

Quem consome (lote, API, painel) entrega o **dado cru** e recebe a predição —
treino e inferência usam exatamente as mesmas transformações, sem reimplementar nada.
Cada run (inclusive as 9 do grid) loga o pipeline com **assinatura**, **exemplo de entrada**
e as tags de linhagem `data_md5` (DVC) e `git_commit`:

```python
mlflow.sklearn.log_model(pipe, name="model", signature=signature,
                         input_example=x_train.iloc[:5],
                         skops_trusted_types=TRUSTED_TYPES)  # MLflow 3 serializa com skops
```

Inferência em lote com o `@champion`:

```bash
uv run python -m churn.predict --input data/churn.csv --output predictions.csv
```

### Orquestração do ML Pipeline com DVC

O `dvc.yaml` encadeia as duas etapas — `dvc repro` só re-executa o que mudou:

| etapa | comando | saída |
|---|---|---|
| `treino_e_tuning` | `python -m churn.experiments` (grid de 9 runs) | `mlflow.db` |
| `promover_modelo` | `python -m churn.registry` + `--export` | `artifacts/champion/` |

---

## O que mudou em relação ao `train_churn.py`

| Script monolítico | Projeto modular |
|---|---|
| caminho hardcoded `C:/Users/ana/...` | `config.py` (Pydantic, sobrescrevível por env) |
| `fillna(2200)` mágico | `TotalCharges` em branco → `0.0` (cliente novo) |
| normalização por números mágicos | removida — Random Forest não precisa de escala |
| `LabelEncoder` no dataset inteiro (vazamento) | `CategoricalEncoder` com `fit` só no treino |
| sem `random_state` (acurácia instável) | semente fixa em split e modelo |
| só `accuracy` | 5 métricas: accuracy, precision, recall, f1, roc_auc |
| sem validação de dados | `ChurnSchema` (pandera) — falha cedo e com mensagem |
| dataset solto no disco | `churn.csv.dvc` — versionado pelo DVC |
| `print(accuracy)` e esquece | MLflow registra params, métricas e artefatos |
| modelo em arquivo pkl sem rastreamento | Model Registry com aliases `@champion`/`@challenger` |
| inferência acoplada ao código de treino | `@champion` é um pipeline completo: lote, API e painel recebem o dado cru |
| modelo sem interface | FastAPI (`/predict`, `/model-info`) + painel Streamlit |

## Bloco 3 — Serviço do modelo (Encontro 12)

### FastAPI servindo o `@champion`

```bash
uv run python -m churn.experiments             # 1. grid de 9 runs
uv run python -m churn.registry                # 2. melhor run → @champion
uv run uvicorn churn.app:app --reload --port 8000   # 3. API (lê direto do Registry)
```

- **Docs interativas:** http://localhost:8000/docs

| rota | o que faz |
|---|---|
| `GET /health` | API no ar? modelo e base de clientes carregados? |
| `GET /model-info` | versão, run, métricas, hiperparâmetros e linhagem (DVC + git) do `@champion` |
| `GET /predict/{customer_id}` | predição para um cliente da base histórica (+ churn real) |
| `POST /predict/{customer_id}` | simulação "e se": troca `tenure`/`MonthlyCharges`/`TotalCharges` |
| `POST /predict` | cliente novo, informando todas as colunas |

```bash
curl -s localhost:8000/predict/7590-VHVEG
curl -s -X POST localhost:8000/predict/7590-VHVEG \
  -H "Content-Type: application/json" -d '{"tenure": 48, "TotalCharges": 1430.0}'
```

Resposta (valores ilustrativos):

```json
{"churn": false, "churn_probability": 0.21, "label": "No", "risk": "Baixo",
 "model_version": 1, "customer_id": "7590-VHVEG", "actual_churn": "No",
 "customer_source": "base_historica_com_simulacao",
 "simulated_values": {"tenure": 48, "TotalCharges": 1430.0}}
```

O contrato de entrada é o mesmo do `ChurnSchema`: categoria fora do contrato
(ex.: `"Contract": "Vitalicio"`) ou `tenure` fora de 0–72 → **422**, sem chute.
Sem modelo carregado a API sobe mesmo assim: `/health` mostra `model_loaded: false`
e as predições respondem **503** com o motivo.

### De onde a API carrega o modelo

| `CHURN_MODEL_URI` | uso |
|---|---|
| `models:/churn@champion` (padrão) | dev local — direto do Model Registry (`mlflow.db`) |
| `artifacts/champion` | container — cópia exportada com `python -m churn.registry --export` |

O Registry vive no `mlflow.db` local, com caminhos absolutos para o `mlruns/`. Por isso,
antes de subir o container, o `@champion` é **exportado** para `artifacts/champion/`
(pacote MLflow + `champion_info.json` com versão, métricas e linhagem).

### Painel Streamlit

```bash
uv run streamlit run src/churn/dashboard.py    # http://localhost:8501
```

Três abas: **consulta por cliente** (com simulação "e se"), **novo cliente** (formulário
completo) e **modelo @champion** (versão, métricas, linhagem e hiperparâmetros).
O painel só conversa com a API (`API_URL`); ele não carrega o modelo.

### Docker Compose

```bash
uv run python -m churn.registry --export   # gera artifacts/champion/
docker compose up --build api painel       # API :8000 + painel :8501
```

---

## Deploy na AWS (duas camadas)

```text
[Navegador]
     │ HTTP :8501
     ▼
┌── AWS us-east-2 (VPC padrão) ─────────────────────────────────────────────────┐
│  EC2 churn-app-host (role EC2-ECR-ReadRole)      EC2 churn-db-host            │
│  ┌── container (imagem do ECR) ──────┐          ┌── container churn_postgres ┐│
│  │ Supervisord                       │  SQL     │ PostgreSQL 15              ││
│  │  ├─ Streamlit :8501 app_churn_v2  │  :5432   │ banco churn_db             ││
│  │  └─ FastAPI   :8000 api_v2 ───────┼─────────►│ tabela customers           ││
│  │     + @champion em memória        │ IP priv. │ (churn.csv bruto)          ││
│  └───────────────────────────────────┘          └────────────────────────────┘│
│                ▲ docker pull                                                  │
│        Amazon ECR: app-churn:latest                                           │
└───────────────────────────────────────────────────────────────────────────────┘
```

* **Camada de aplicação:** um único container roda backend (FastAPI) e frontend (Streamlit),
  gerenciados pelo Supervisord. A imagem (`Dockerfile.app`) vai para o ECR e a EC2 a baixa
  com uma **IAM Role** — credenciais temporárias, nenhuma chave gravada em disco.
* **Camada de dados:** PostgreSQL em Docker numa segunda EC2, com o `churn.csv` bruto na
  tabela `customers` (`infra/db/init_customers.sql`).
* As máquinas conversam pelo **IP privado**; a senha do banco entra só em tempo de
  execução (`--env-file`), nunca no Git nem na imagem.

### app-churn-v1 — consulta de resultados prontos (sem inferência)

* `churn.batch_scoring` aplica o `@champion` a todos os clientes e grava `data/batch_predictions.parquet`
  (roda dentro do `docker build`).
* `churn.api_lookup` (FastAPI) só consulta esse arquivo: `GET /results/{customerID}`.
* `src/churn/app_churn_v1.py` é a tela.

### app-churn-v2 — inferência em tempo real + PostgreSQL

* `churn.api_v2`: `GET /customers/{id}` e `GET /results/{id}` fazem `SELECT` no Postgres e o
  `@champion` pontua o registro; `POST /predict` faz inferência online para cliente novo
  (não depende do banco); `GET /health` mostra modelo e banco; `GET /model-info`, a linhagem.
* `src/churn/app_churn_v2.py`: bloco 1 (consulta no banco) + bloco 2 (nova inferência).

A mesma imagem serve as duas versões (`API_MODULE` / `FRONTEND`, padrão v2).

### Um modelo, do treino à produção

```text
churn.csv (DVC) → churn.experiments (grid MLflow) → churn.registry (@champion)
               → churn.registry --export (artifacts/champion/) → docker build → ECR → EC2
```

* **Contrato de entrada.** O `@champion` é o pipeline completo do treino, e antes de pontuar
  o registro é checado contra o mesmo `ChurnSchema` (pandera): um cliente vindo do Postgres
  ou do formulário com categoria desconhecida (`Contract = "Vitalicio"`) é recusado com **422**.
* **Mesmo ambiente.** A imagem instala as dependências do mesmo `uv.lock` que treinou o modelo.
* **Linhagem.** `champion_info.json` leva versão, run, métricas, `data_md5` (DVC) e `git_commit`.

### Como publicar

```bash
uv run python -m churn.experiments && uv run python -m churn.registry
AWS_PROFILE=churn bash infra/deploy_ecr.sh          # Windows: .\infra\deploy_ecr.ps1 -AwsProfile churn
```

Passo a passo das duas EC2, role IAM, Security Groups e custos: **[`infra/README.md`](infra/README.md)**.

Teste local das duas camadas, sem AWS:

```bash
uv run python -m churn.registry --export
docker compose up --build app-v2        # Postgres + app; painel v2 em http://localhost:8502
```

---

## Observações

* **Dependências travadas:** treino, API e imagem da AWS usam o mesmo `uv.lock`.
* **Testes:** `uv run pytest` cobre dados, features, schema, profiling, registry, API e as apps da AWS (v1 e v2).
* **IP público das EC2:** muda a cada parada/início. A comunicação entre as máquinas usa o IP privado.
* **Airflow:** não implementado neste repositório; o ML Pipeline é orquestrado pelo `dvc.yaml`.

## Próximos encontros

- **Enc. 9–11** — Orquestração com Airflow.
- **Enc. 13–14** — CI/CD com GitHub Actions.
- **Enc. 15** — Monitoramento com Evidently.
