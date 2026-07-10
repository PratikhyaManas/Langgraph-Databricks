"""
Creates the Model Serving endpoint if it doesn't exist, or updates it to
serve the version currently pointed at by the configured alias.

Run after log_and_register_model.py, per environment:
    python deploy/create_or_update_endpoint.py
"""
from __future__ import annotations

import sys

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)

sys.path.insert(0, ".")

from src.utils.config import AgentConfig
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    config = AgentConfig.from_env()
    ws = WorkspaceClient()

    served_entity = ServedEntityInput(
        entity_name=config.full_model_name,
        entity_version=None,  # None + alias below tracks the alias automatically in UC
        scale_to_zero_enabled=True,
        workload_size="Small",
    )

    existing = None
    for ep in ws.serving_endpoints.list():
        if ep.name == config.endpoint_name:
            existing = ep
            break

    if existing is None:
        logger.info("Creating new serving endpoint: %s", config.endpoint_name)
        ws.serving_endpoints.create(
            name=config.endpoint_name,
            config=EndpointCoreConfigInput(served_entities=[served_entity]),
        )
    else:
        logger.info("Updating existing serving endpoint: %s", config.endpoint_name)
        ws.serving_endpoints.update_config(
            name=config.endpoint_name,
            served_entities=[served_entity],
        )

    logger.info("Endpoint '%s' deployment triggered.", config.endpoint_name)


if __name__ == "__main__":
    main()
