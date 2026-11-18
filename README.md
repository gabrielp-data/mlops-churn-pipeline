# Previsão de Churn — Projeto Prático de MLOps

**[Abrir o painel público: churn-gabriel.streamlit.app](https://churn-gabriel.streamlit.app)**

[![Tests](https://github.com/gabrielp-data/mlops-churn-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/gabrielp-data/mlops-churn-pipeline/actions/workflows/ci.yml)

Projeto de Gabriel para a disciplina **Engenharia de Machine Learning Aplicada — princípios de MLOps**, do IESB. A implementação segue como referência o [projeto de Caio](https://github.com/caiorj2004/mlops-churn-project). O plano da disciplina está em [grade-mlops-60h.md](grade-mlops-60h.md).

O painel publicado consulta clientes no PostgreSQL, calcula previsões com um Random Forest exportado do MLflow e grava novas simulações no histórico. A aplicação roda no Streamlit Community Cloud e o banco fica em outro servidor. Não depende do computador do autor estar ligado.

## Publicação atual

```text
Navegador (HTTPS)
       │
       ▼
Streamlit Community Cloud — streamlit_cloud/app.py
       ├── modelo exportado + contrato de features
       └── PostgreSQL gerenciado (conexão via Secrets)
                 ├── customers: base pública IBM Telco
                 └── predictions: histórico de simulações
```

O painel executa a inferência diretamente no processo do Streamlit. A API FastAPI e os containers são alternativas de execução local/AWS; não são serviços adicionais dessa publicação.

| Item | Estado verificado em 18/11/2026 |
| --- | --- |
| Site | [churn-gabriel.streamlit.app](https://churn-gabriel.streamlit.app) |
| Clientes no banco | 7.043 |
| Modelo publicado | `churn-model`, versão 1, exportado do alias `@champion` |
| Run | `rf-n500-d8` · `ad80f992061a43b48663a5fe53cc6dc6` |
| Random Forest | 500 árvores, profundidade máxima 8 |
| Acurácia registrada no artefato | 80,13% |
| Consulta `7590-VHVEG` | 54,39% de probabilidade, classe 1, risco médio; churn observado `No` |
| Simulação com os valores iniciais do formulário | 67,66%, classe 1, risco alto; registrada em `predictions` |

A acurácia acima vem de [champion_info.json](streamlit_cloud/champion/champion_info.json); não é uma avaliação nova sobre o banco de produção. O primeiro exemplo é um erro de classificação do modelo, não uma promessa de previsão correta para cada cliente. O limite da classe é 50%; as faixas de risco são baixo (<30%), médio (30% a <60%) e alto (≥60%).

## Dados e rastreabilidade

A publicação inicializa `customers` com a [base pública IBM Telco em um commit fixo](https://github.com/IBM/watsonx-ai-samples/blob/e7345a897d3a440e0e8ee465150fe895bd5b29cb/cpd4.5/data/customer_churn/WA_FnUseC_TelcoCustomerChurn.csv): 7.043 registros, 21 colunas, 5.174 `No` e 1.869 `Yes` (26,54% de churn). Exemplos de códigos: `7590-VHVEG` e `5575-GNVDE`.

O MD5 desse CSV é `3a27ec1ff5c0a90dd61cefa8a13184ab`. O ponteiro histórico [data/raw/churn.csv.dvc](data/raw/churn.csv.dvc) registra outro hash (`d390bd07b5514a2256a2396993b8b0e3`). O metadado do campeão também registra esse hash histórico. **Não são evidência de que a base pública seja o mesmo arquivo do treino**; reproduzir a linhagem original requer recuperar a versão correspondente. Não há remote DVC configurado neste checkout.

O artefato publicado inclui o modelo e o mapeamento/ordem de categorias em `feature_contract.json`. A transformação de inferência está em [src/features.py](src/features.py); categorias desconhecidas são recusadas. A aplicação usa a cópia exportada, e não acompanha alterações do alias no MLflow em tempo real.

## Estrutura

| Caminho | Responsabilidade |
| --- | --- |
| `streamlit_cloud/app.py` | Painel público, consulta ao banco e histórico |
| `streamlit_cloud/champion/` | Modelo publicado, contrato e metadados |
| `streamlit_cloud/requirements.txt` | Dependências do painel; bibliotecas do modelo com versões fixas |
| `src/data.py`, `src/schema.py`, `src/features.py` | Carga, validação e transformações |
| `src/experiments.py`, `src/register_model.py` | Grid de treinamento e promoção no MLflow |
| `src/export_champion.py`, `src/inference.py` | Exportação e consumo do campeão |
| `src/api_v2.py`, `src/db.py` | API e conexão ao PostgreSQL para execução fora do Cloud |
| `src/batch_scoring.py`, `src/api_lookup.py` | Previsões em lote e API de consulta |
| `app_churn_v1.py`, `app_churn_v2.py`, `streamlit_app.py` | Interfaces para as APIs |
| `tests/`, `.github/workflows/ci.yml` | Testes automáticos e integração com PostgreSQL |
| `infra/`, `Dockerfile*`, `docker-compose.yml` | Alternativas de empacotamento e implantação AWS |
| `dvc.yaml`, `dvc.lock`, `data/raw/churn.csv.dvc` | Pipeline e ponteiro DVC |

## Executar o painel publicado localmente

Python 3.12, na raiz do repositório:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r streamlit_cloud/requirements.txt
streamlit run streamlit_cloud/app.py
```

Configure `.streamlit/secrets.toml` (ignorado pelo Git) com uma conexão PostgreSQL dedicada à demonstração:

```toml
DB_URL = "postgresql://USUARIO:SENHA@HOST/BANCO?sslmode=require"
```

O painel cria `customers` e `predictions` se ainda não existirem. Usa o CSV público para a carga inicial; não sobrescreve uma tabela já existente. A conexão precisa ter permissão para criar tabelas, consultar clientes e inserir previsões.

## Publicar no Streamlit Community Cloud

1. Crie um banco PostgreSQL gerenciado e obtenha a conexão com SSL.
2. Em [share.streamlit.io](https://share.streamlit.io), selecione este repositório, a branch `claude/great-thompson-kkpvpv` e o arquivo `streamlit_cloud/app.py`.
3. Em **Advanced settings → Secrets**, configure `DB_URL`. Use Python 3.12.
4. Publique e verifique consulta, nova inferência e histórico.

Atualizações na branch conectada são refletidas no aplicativo. Para trocar o modelo, exporte o campeão validado e atualize `streamlit_cloud/champion/` e as dependências correspondentes. Nunca publique credenciais no código, README ou histórico Git.

## Treinamento e execução com API

Instale `requirements.txt` em um ambiente próprio para desenvolvimento. Disponibilize a versão correta do dado em `data/raw/churn.csv` antes de executar:

```powershell
python -m src.profiling
python run_pipeline.py
python -m src.experiments
python -m src.register_model
python -m src.export_champion
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

`dvc repro` encadeia treinamento/tuning e promoção; recuperar o dado via DVC depende de configurar o armazenamento da versão original. Usar outra base exige atualizar e documentar sua versão, sem reutilizar o hash antigo como se fosse equivalente.

Para a API v2, configure as variáveis de `.env.example`, mantenha o modelo em `models/champion/` e execute:

```powershell
python -m uvicorn src.api_v2:app --port 8000
streamlit run app_churn_v2.py
```

O fluxo AWS proposto usa ECR, uma EC2 da aplicação (FastAPI + Streamlit) e outra EC2 do PostgreSQL. **Não foi implantado nesta conta AWS.** Os passos estão em [infra/README.md](infra/README.md). As imagens em `docs/img/aws_*` pertencem ao projeto de referência e não comprovam uma implantação AWS deste repositório.

## Testes automáticos

```powershell
python -m pip install -r requirements-test.txt
python -m pytest -q
```

A suíte usa o modelo efetivamente publicado e cobre:

- validação de dados inválidos, IDs duplicados, categorias e clientes com tempo zero;
- consistência da previsão entre API e painel;
- cliente ausente, IDs com caracteres SQL, entrada inválida e falha do banco;
- consulta e simulação pelo Streamlit, gravação do histórico sem duplicar a cada atualização;
- histórico vazio e preenchido sem expor objetos internos do Streamlit.

Localmente os testes usam SQLite temporário. O [GitHub Actions](https://github.com/gabrielp-data/mlops-churn-pipeline/actions/workflows/ci.yml) executa a mesma suíte contra PostgreSQL 16 real, com banco efêmero e esquema isolado por teste. Não usa o banco nem os segredos de produção. Para repetir com um PostgreSQL de teste, configure `TEST_DATABASE_URL` em uma base descartável com permissão para criar esquemas.

`pytest.ini` limita a coleta a `tests/`: `test_schema.py` e `src/test_champion.py` são scripts didáticos que dependem de dados/registry locais, não testes isolados.

## Limitações conhecidas

- Demonstração acadêmica com dados públicos, sem autenticação: qualquer visitante pode simular e visualizar o histórico recente. Não inserir dados pessoais reais.
- O formulário fixa alguns atributos de perfil e usa `meses × mensalidade` quando o total é zero; os resultados representam essas hipóteses.
- O pipeline atual seleciona o campeão pela acurácia de teste. Uma avaliação imparcial requer separar validação para escolha e um teste final reservado.
- A linhagem do modelo exportado precisa ser reconciliada com os bytes do dado de treino; os testes de serving não revalidam a métrica de treinamento.
- Airflow, retreinamento automático e monitoramento de drift não estão implementados.
