"""
Centralized, typed configuration.

All values are read from environment variables so the exact same code runs
unchanged in dev/staging/prod — only the *values* differ, injected by
databricks.yml (bundle variables) or GitHub Actions env blocks.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _require(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise EnvironmentError(
            f"Missing required environment variable '{name}'. "
            f"Set it directly or via databricks.yml bundle variables."
        )
    return value


@dataclass(frozen=True)
class AgentConfig:
    catalog_name: str
    schema_name: str
    experiment_id: str
    foundation_model_name: str
    endpoint_name: str
    registered_model_name: str
    alias: str = "champion"

    @property
    def full_model_name(self) -> str:
        return f"{self.catalog_name}.{self.schema_name}.{self.registered_model_name}"

    @classmethod
    def from_env(cls) -> "AgentConfig":
        catalog_name = _require("CATALOG_NAME")
        schema_name = _require("SCHEMA_NAME", "default")
        return cls(
            catalog_name=catalog_name,
            schema_name=schema_name,
            experiment_id=_require("EXPERIMENT_ID"),
            foundation_model_name=_require(
                "FOUNDATION_MODEL_NAME", "databricks-meta-llama-3-1-70b-instruct"
            ),
            endpoint_name=_require("ENDPOINT_NAME"),
            registered_model_name=os.environ.get(
                "REGISTERED_MODEL_NAME", "sql_workflow_agent"
            ),
            alias=os.environ.get("MODEL_ALIAS", "champion"),
        )
