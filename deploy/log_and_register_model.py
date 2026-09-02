"""
Script version of the notebook's "log + register model" cells.

Called headlessly by CI via:
    databricks bundle run deploy_agent -t <env>

Logs the SQLWorkflowResponsesAgent to MLflow, registers it in Unity Catalog,
and points the configured alias (default "champion") at the new version.
"""
from __future__ import annotations

import argparse
import sys

import mlflow
from mlflow.models import ModelSignature
from mlflow.types.responses import RESPONSES_AGENT_INPUT_SCHEMA, RESPONSES_AGENT_OUTPUT_SCHEMA

sys.path.insert(0, ".")  # allow `python deploy/log_and_register_model.py` from repo root

from src.utils.config import AgentConfig
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=False, help="Overrides CATALOG_NAME env var")
    parser.add_argument("--endpoint", required=False, help="Overrides ENDPOINT_NAME env var")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AgentConfig.from_env()
    if args.catalog:
        config = config.__class__(**{**config.__dict__, "catalog_name": args.catalog})
    if args.endpoint:
        config = config.__class__(**{**config.__dict__, "endpoint_name": args.endpoint})

    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(experiment_id=config.experiment_id)

    signature = ModelSignature(
        inputs=RESPONSES_AGENT_INPUT_SCHEMA, outputs=RESPONSES_AGENT_OUTPUT_SCHEMA
    )

    with mlflow.start_run(run_name=f"deploy-{config.endpoint_name}"):
        logged_model = mlflow.pyfunc.log_model(
            python_model="src/agent/responses_agent.py",
            artifact_path="agent",
            signature=signature,
            pip_requirements=[
                "databricks-sdk>=0.30.0",
                "databricks-sql-connector>=3.4.0",
                "langchain-community>=0.3.0",
                "langgraph>=1.0.0,<2.0.0",
                "mlflow>=2.17.0",
            ],
        )
        logger.info("Logged model: %s", logged_model.model_uri)

        registered = mlflow.register_model(
            model_uri=logged_model.model_uri,
            name=config.full_model_name,
        )
        logger.info(
            "Registered %s as version %s", config.full_model_name, registered.version
        )

        client = mlflow.MlflowClient()
        client.set_registered_model_alias(
            name=config.full_model_name,
            alias=config.alias,
            version=registered.version,
        )
        logger.info(
            "Set alias '%s' -> version %s for %s",
            config.alias,
            registered.version,
            config.full_model_name,
        )

    # Emit the version so downstream CI steps (e.g. endpoint creation, or a
    # later promote step) can reference the exact artifact that was tested.
    print(f"::set-output name=model_version::{registered.version}")


if __name__ == "__main__":
    main()
