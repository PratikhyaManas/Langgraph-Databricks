"""
Promotes a model that already passed dev integration tests into
staging/prod, by copying the exact tested version rather than re-running
log_and_register_model.py against a different catalog.

Usage:
    python deploy/promote_model.py \
        --source-model dev_catalog.gold.sql_workflow_agent \
        --source-alias champion \
        --target-model prod_catalog.gold.sql_workflow_agent \
        --target-alias champion
"""
from __future__ import annotations

import argparse

import mlflow

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--source-alias", default="champion")
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--target-alias", default="champion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mlflow.set_registry_uri("databricks-uc")
    client = mlflow.MlflowClient()

    source_version = client.get_model_version_by_alias(
        args.source_model, args.source_alias
    )
    logger.info(
        "Copying %s@%s (version %s) -> %s",
        args.source_model,
        args.source_alias,
        source_version.version,
        args.target_model,
    )

    copied = client.copy_model_version(
        src_model_uri=f"models:/{args.source_model}/{source_version.version}",
        dst_name=args.target_model,
    )

    client.set_registered_model_alias(
        name=args.target_model,
        alias=args.target_alias,
        version=copied.version,
    )
    logger.info(
        "Set alias '%s' -> version %s on %s",
        args.target_alias,
        copied.version,
        args.target_model,
    )


if __name__ == "__main__":
    main()
