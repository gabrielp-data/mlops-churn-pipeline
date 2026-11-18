# Churn · do treinamento à aplicação

Pipeline de previsão de cancelamento com **MLflow, FastAPI, Streamlit, PostgreSQL, DVC e Docker**.
O modelo, o encoder e a engenharia de atributos são um único pipeline. A API, o lote
e a consulta por cliente utilizam o mesmo pacote exportado do Model Registry.

A aplicação tem dois modos:
- **Online:** consulta de clientes no PostgreSQL (ou CSV local) e nova inferência.
- **Lote:** consulta de um Parquet previamente calculado, sem executar o modelo.

## Demonstração rápida no Windows

Pré-requisitos: Python 3.12 e [uv](https://docs.astral.sh/uv/getting-started/installation/).

```powershell
uv sync --frozen
.\scripts\demo.ps1
```

O script baixa uma amostra pública IBM com checksum verificado, treina, promove,
exporta, calcula o lote e inicia o painel em **http://localhost:8501**.
A API fica em **http://localhost:8000/docs**. Use Ctrl+C para encerrar os dois serviços.
Para apenas preparar os artefatos: ` .\scripts\demo.ps1 -TrainOnly `.

**Origem dos dados:** o ponteiro original `data/churn.csv.dvc` foi preservado.
Seu remote não está configurado no repositório, então `dvc pull` requer a configuração
do armazenamento original. A demonstração usa separadamente
[data/telco-public.csv da IBM](https://github.com/IBM/watsonx-ai-samples/blob/e7345a897d3a440e0e8ee465150fe895bd5b29cb/cpd4.5/data/customer_churn/WA_FnUseC_TelcoCustomerChurn.csv),
com MD5 `3a27ec1ff5c0a90dd61cefa8a13184ab`. Não é uma recuperação da versão original
`d390bd07b5514a2256a2396993b8b0e3`.

## Executar por etapas

Copie `.env.example` para `.env` e ajuste o caminho dos dados.
Com a base pública, defina `CHURN_DATA_PATH=data/telco-public.csv`.

```powershell
uv sync --frozen
uv run python -m churn.fetch_data
uv run python -m churn.model
# Opcional: substitua o treino único pelo grid de 9 combinações.
uv run python -m churn.experiments
uv run python -m churn.registry
uv run python -m churn.export_champion
uv run python -m churn.predict
uv run python -m churn.serve
```

No Linux/macOS, os comandos `uv run` são os mesmos. O demo.ps1 é apenas um atalho Windows.

Para abrir os serviços separadamente:
```sh
uv run uvicorn churn.app:app --port 8000
uv run streamlit run streamlit_app.py
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

O MLflow UI usa a mesma URI configurada em CHURN_TRACKING_URI. Os exemplos usam SQLite local.

## Fluxo do modelo

```text
CSV → validação → split estratificado 60% treino / 20% validação / 20% teste
    → features + encoder + Random Forest → MLflow
    → seleção por val_roc_auc → registry @champion
    → models/champion/ → lote / API / imagem Docker
```

- Encoder ajustado exclusivamente no treino.
- Contrato valida categorias, intervalos e colunas; entrada nova não exige Churn.
- O grid salva todos os modelos e compara ROC-AUC de validação.
- O conjunto final de teste é avaliado somente para o vencedor do grid.
- Métricas: accuracy, precision, recall, F1 e ROC-AUC.
- Cada run salva o hash dos bytes do dado, commit, parâmetros, assinatura e contrato.
- A promoção usa candidatos da execução atual, evitando runs antigos incompletos.
- Exportação resolve uma versão imutável; metadados identificam o modelo servido.
- O limiar de classificação é 0,5. Faixas de risco 0,3 e 0,6 são heurísticas de interface,
  não decisões de negócio calibradas.
- Probabilidades do lote e da API são verificadas contra o pipeline carregado.

`models/champion/` contém o modelo MLflow, código, requirements, contrato de features
e champion_info.json. Mudanças de alias só chegam ao serviço após nova exportação/reinício,
ou novo build/deploy na AWS.

## Endpoints

| Rota | Resultado |
|---|---|
| GET /health | Modelo carregado e disponibilidade do banco configurado |
| GET /model-info | Versão, run, métricas, parâmetros e linhagem |
| GET /customers/{id} | Registro do cliente |
| GET /results/{id} | Registro, previsão e churn observado quando disponível |
| POST /predict | Previsão de um cliente novo |

Campos de POST /predict: os 19 atributos do Telco, sem customerID ou Churn.
Consulte o formulário e /docs. TotalCharges deve ser informado; zero não é substituído
por uma estimativa. Categorias inválidas retornam 422; cliente inexistente, 404;
falha do banco, 503. A nova inferência permanece disponível sem o banco.

### Lote e aplicação v1

```sh
uv run python -m churn.predict --input data/telco-public.csv --output data/batch_predictions.parquet
uv run uvicorn churn.api_lookup:app --port 8001
```

O lote pode receber um CSV sem alvo e sem ID. A API de consulta exige IDs únicos.
Ela não oferece POST /predict. Para painel e API v1 juntos, use CHURN_SERVING_MODE=batch
e CHURN_API_URL apontando para a API do lote.

## PostgreSQL e Docker

Copie `.env.example` para `.env`, escolha POSTGRES_PASSWORD e configure:
`CHURN_DATABASE_URL=postgresql+psycopg://churn:SENHA_URL_ENCODED@db:5432/churn`.

Depois do treino, promoção e exportação locais:

```sh
docker compose --profile database up -d --wait db
docker compose build app
docker compose run --rm app python -m churn.db --input /app/data/telco-public.csv
docker compose up -d app
```

A importação cria a tabela e insere em uma transação; não sobrescreve IDs existentes.
Para CSV local, deixe CHURN_DATABASE_URL ausente e ajuste CHURN_DATA_PATH. O Compose
aceita o caminho em `data/` definido no .env e monta a pasta somente para leitura.

A imagem serving roda FastAPI e Streamlit juntos, como a referência. Encerra ambos
se um processo falhar e tem health check dos dois. Modelos são exportados antes do build;
o banco MLflow não vai dentro da imagem.

```sh
# Variante de consulta em lote, nas portas 8001 e 8502:
docker compose --profile batch up -d lookup
# Treino isolado, sem exigir exportação prévia:
docker compose --profile training run --rm treino
```

O treino em container mantém seu registry em .state/mlflow.db e os artefatos em mlruns/.
Para promover e exportar esse treino, use o mesmo serviço:
```sh
docker compose --profile training run --rm treino python -m churn.registry
docker compose --profile training run --rm treino python -m churn.export_champion
```

Deploy ECR + EC2 da aplicação + EC2 do PostgreSQL: [infra/README.md](infra/README.md).
Os scripts não provisionam recursos AWS automaticamente.

## DVC

Com o CSV original disponível em data/churn.csv e uv no PATH:
```sh
uv run dvc repro
```

As dependências conectam tuning → promoção → exportação → lote.
A etapa de treino é sempre executada, pois cria estado no MLflow externo ao cache DVC.
Não use a amostra pública como se fosse o dado do ponteiro original. Para adotá-la como
nova versão oficial, atualize explicitamente o caminho/ponteiro e configure um remote DVC.

## Testes e CI

```sh
uv run ruff check src tests streamlit_app.py
uv run pytest -q
```

A suíte inclui validação, API com modelo real, painel Streamlit, falha de banco,
importação transacional e ciclo completo com SQLite/MLflow, registry, exportação e lote.
O CI executa os testes em Python 3.12 no Linux. SQLite valida a integração de dados
nos testes; PostgreSQL e containers precisam de verificação no ambiente de implantação.

Airflow, retreino automático, monitoramento de drift, autenticação e implantação contínua
não fazem parte desta versão.

## Estrutura

```text
src/churn/
  data.py, schema.py, features.py  contrato e transformações
  model.py, experiments.py         treino e avaliação
  registry.py, export_champion.py  seleção e empacotamento
  inference.py, predict.py         scoring compartilhado
  app.py, api_lookup.py            APIs online e de lote
  db.py                           clientes e importação
  serve.py                        FastAPI + Streamlit
  fetch_data.py                   amostra pública verificada
streamlit_app.py                   painel
tests/                            testes unitários e de integração
infra/                            ECR/EC2/PostgreSQL
scripts/demo.ps1                  demonstração local
Dockerfile, compose.yml            execução em containers
dvc.yaml                          reprodução do fluxo
```

O escopo de aplicação segue a referência
[caiorj2004/mlops-churn-project](https://github.com/caiorj2004/mlops-churn-project),
com implementação integrada ao pacote e aos testes deste repositório.
