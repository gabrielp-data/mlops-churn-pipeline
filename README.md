# prj_churn · Curso de MLOps — IESB

Projeto-fio-condutor do curso: previsão de **churn** (classificação binária tabular).  
O arco é do `train_churn.py` monolítico até um sistema completo de MLOps.

---

## Estrutura

```
prj_churn/
├─ src/churn/
│  ├─ config.py        # Settings (Pydantic) — caminhos, hiperparâmetros
│  ├─ data.py          # carga, limpeza, validação (pandera)
│  ├─ schema.py        # ChurnSchema — contrato de dados
│  ├─ features.py      # transformações puras + encoder sem vazamento
│  ├─ model.py         # treino + MLflow tracking + signature
│  ├─ evaluate.py      # métricas
│  ├─ experiments.py   # grid rastreado (MLflow)
│  ├─ lineage.py       # data_md5() e git_commit()
│  ├─ registry.py      # Model Registry — register, aliases @champion/@challenger
│  ├─ predict.py       # inferência em lote via @champion (pyfunc)
│  ├─ pipeline.py      # tarefas do DAG (Python puro, testável sem Airflow)
│  └─ app.py           # API REST FastAPI (GET /health, POST /predict)
├─ dags/
│  └─ churn_training.py # DAG do Airflow (enc. 9–11)
├─ tests/              # pytest (data, features, schema, pipeline, dag, app)
├─ data/
│  ├─ churn.csv.dvc    # ponteiro DVC (o .csv fica no remote)
│  └─ .gitignore
├─ Dockerfile          # imagem com uv (enc. 3)
├─ Dockerfile.airflow  # Airflow + pacote churn (enc. 11)
├─ compose.yml         # docker compose (enc. 3, 11, 12)
├─ dvc.yaml            # pipeline reproduzível (enc. 5)
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

Carregar o campeão em código:

```python
from churn.registry import load_champion
model = load_champion()            # pyfunc — agnóstico de framework
preds = model.predict(features_df)
```

### Encontro 8 — Empacotamento (MLflow Models)

O `model.py` agora registra **assinatura** e **exemplo de entrada** junto ao artefato:

```python
signature = infer_signature(x_train_enc, model.predict(x_train_enc))
mlflow.sklearn.log_model(model, artifact_path="model",
                         signature=signature, input_example=x_train_enc.iloc[:5])
```

Inferência em lote com o `@champion`:

```bash
uv run python -m churn.predict --input data/churn.csv --output predictions.csv
```

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
| inferência acoplada ao código de treino | `predict.py` carrega `@champion` via pyfunc (agnóstico) |
| retreino manual, "quando alguém lembra" | DAG semanal no Airflow com quality gate e champion × challenger |

## Bloco 3 — Orquestração (Encontros 9–11)

O retreino vira um **DAG** do Airflow (`dags/churn_training.py`):

```
validate_data → train_candidate → quality_gate → decide_promotion
                                                 ├─ promote_champion     (candidato > campeão)
                                                 └─ register_challenger  (caso contrário)
```

A lógica das tarefas mora em `src/churn/pipeline.py` — Python puro, sem `import airflow`.
O DAG só descreve **ordem, agendamento e política de falha**. Assim as tarefas são
testadas com pytest e o pipeline roda localmente sem scheduler:

```bash
MLFLOW_TRACKING_URI=sqlite:///mlflow.db uv run python -m churn.pipeline
```

```
[validate] 7043 linhas, churn=26.5%
[train]    run=abc12345 roc_auc=0.8195
[registry] v1 → @champion
```

### Encontro 9 — DAG e tarefas

- TaskFlow API (`@dag`, `@task`): o retorno de uma tarefa vira entrada da próxima.
- `schedule="@weekly"`, `catchup=False`, `max_active_runs=1`.

### Encontro 10 — XCom, branching e quality gate

- **XCom** carrega só o que é pequeno e serializável: `run_id` e métricas. Modelo e
  DataFrame ficam no MLflow/disco.
- **Quality gate:** `roc_auc < CHURN_MIN_ROC_AUC` (padrão 0.75) falha o DAG antes de
  tocar no Registry. Essa tarefa tem `retries=0` — reprovar é decisão, não erro transitório.
- **Branching (`@task.branch`):** o candidato vira `@champion` só se superar o `roc_auc`
  do campeão atual; senão é registrado como `@challenger`. O ramo não escolhido fica `skipped`.
- Demais tarefas: `retries=2`, `retry_delay=5min`.

### Encontro 11 — Airflow em container

```bash
docker compose up airflow                         # UI em http://localhost:8080
docker compose logs airflow | grep -i password    # senha do usuário admin
```

O serviço monta `dags/`, `src/`, `data/`, `artifacts/` e `mlruns/` (o backend do
MLflow fica em `mlruns/mlflow.db`). Ative o DAG `churn_training` na UI ou dispare:

```bash
docker compose exec airflow airflow dags trigger churn_training
```

Testando o DAG fora do container (Airflow fica fora do `pyproject` de propósito — é
pesado e tem *constraints* próprias):

```bash
pip install "apache-airflow==2.10.5" -e .   # em um venv separado, Python 3.11
airflow dags test churn_training 2026-01-01   # executa o DAG inteiro, sem scheduler
pytest tests/test_dag.py                      # integridade (pulado se não houver Airflow)
```

---

## Bloco 4 — Serving (Encontro 12)

### FastAPI

```bash
# 1. treine o modelo primeiro (gera artifacts/churn_model.pkl)
uv run python -m churn.model          # ou: uv run python -m churn.pipeline / DAG do Airflow

# 2. suba a API
uv run uvicorn churn.app:app --reload --port 8000
```

- **Docs interativas:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health

Exemplo de chamada:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "DSL",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 55.5, "TotalCharges": 666.0
  }'
```

Resposta:
```json
{"churn": true, "churn_probability": 0.72, "label": "Yes"}
```

Ou com Docker Compose:

```bash
docker compose up api
```

---

## Próximos encontros

- **Enc. 13–14** — CI/CD.
- **Enc. 15** — Monitoramento com Evidently.
