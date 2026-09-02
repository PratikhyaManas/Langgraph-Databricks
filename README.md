# langgraph-databricks-cicd

A modular, CI/CD-ready version of a LangGraph SQL workflow agent deployed to
Databricks Model Serving via MLflow + Unity Catalog. Based on the pattern in
[LanggraphInDatabricks](https://github.com/PragatiGupta97/LanggraphInDatabricks),
restructured out of a single notebook into a proper Python package with
automated deploy pipelines.

## Structure

```
.
├── databricks.yml                     # Databricks Asset Bundle (dev/staging/prod targets)
├── src/
│   ├── agent/
│   │   ├── state.py                   # LangGraph state schema
│   │   ├── tools.py                   # SQL / catalog tools the graph nodes call
│   │   ├── nodes.py                   # Individual graph node functions
│   │   ├── graph.py                   # build_graph() — wires nodes into a LangGraph
│   │   └── responses_agent.py         # MLflow ResponsesAgent wrapper around the graph
│   └── utils/
│       ├── config.py                  # Typed config loaded from env vars / bundle vars
│       └── logging_utils.py           # Shared logger setup
├── deploy/
│   ├── log_and_register_model.py      # Logs model to MLflow, registers to Unity Catalog
│   ├── promote_model.py               # Moves an alias (e.g. "champion") to a version
│   └── create_or_update_endpoint.py   # Creates/updates the Model Serving endpoint
├── notebooks/
│   └── 00_manual_run.py               # Thin Databricks notebook that just calls the package
│                                       # (kept only for interactive debugging in the workspace)
├── tests/
│   ├── test_workflow_unit.py          # Local tests, no Databricks connection needed
│   └── test_endpoint_integration.py   # Hits the live serving endpoint after deploy
├── pyproject.toml                     # Runtime and development dependencies
├── uv.lock                            # Reproducible dependency lockfile
├── .env.example
└── .github/workflows/
    ├── ci.yml                         # Lint + unit tests on every PR
    └── deploy.yml                     # Bundle deploy + model registration + promotion
```

## Why this layout

The original notebook mixed config, graph definition, and deployment steps in
one file that only runs interactively. Here:

- **`src/agent`** is a plain importable Python package — testable locally with
  `pytest`, no Databricks connection required for unit tests.
- **`deploy/`** scripts are what CI actually calls (`databricks bundle run`),
  replacing "run notebook cells 8 through 16 by hand."
- **`databricks.yml`** encodes per-environment config (catalog, schema,
  endpoint name, foundation model) instead of hardcoded notebook widgets.
- The **notebook that remains** (`notebooks/00_manual_run.py`) is optional —
  useful if you want to poke at the graph interactively in the Databricks
  workspace — but it's no longer where the source of truth lives.

## Local setup

```bash
uv sync
cp .env.example .env   # fill in catalog/schema/endpoint/model values
uv run pytest tests/test_workflow_unit.py -v
```

## Deploying by hand (before wiring CI/CD secrets)

```bash
databricks auth login --host https://<your-workspace>.azuredatabricks.net
databricks bundle deploy -t dev
databricks bundle run deploy_agent -t dev
```

## CI/CD

See `.github/workflows/ci.yml` and `.github/workflows/deploy.yml`. Summary:

| Trigger                     | Job          | What happens |
|------------------------------|--------------|---------------|
| PR opened/updated            | `ci.yml`     | Lint (ruff) + unit tests (mocked LLM, no Databricks call) |
| Push to `main`               | `deploy.yml` | `bundle deploy -t dev` → register model → set `champion` alias → create/update dev endpoint → integration test against live dev endpoint |
| GitHub Release published     | `deploy.yml` | Promote the **same** already-tested model version's alias into staging/prod catalogs, update prod endpoint |

### Required GitHub secrets

Set these under **Settings → Environments** for both `dev` and `prod` environments
(add required reviewers on `prod` for a manual approval gate):

- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN` (or configure OIDC federation instead — recommended over static PATs)

## Config

All environment-specific values (catalog, schema, endpoint name, foundation
model) live in `databricks.yml` under `targets.<env>.variables`, not hardcoded
in Python. `src/utils/config.py` reads them from environment variables that
the bundle job injects at runtime.
