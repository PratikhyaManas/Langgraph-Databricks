"""
Integration test — runs only in CI's deploy job, after the endpoint has just
been created/updated. Requires real DATABRICKS_HOST / DATABRICKS_TOKEN and a
live endpoint, so it's excluded from the fast PR test run (test_workflow_unit.py).
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABRICKS_HOST"),
    reason="requires a live Databricks workspace; run only in the deploy pipeline",
)


def test_endpoint_responds():
    from databricks.sdk import WorkspaceClient

    ws = WorkspaceClient()
    endpoint_name = os.environ["ENDPOINT_NAME"]

    response = ws.serving_endpoints.query(
        name=endpoint_name,
        inputs={
            "input": [{"role": "user", "content": "Show me top 5 customers by revenue"}]
        },
    )

    assert response is not None
    output = getattr(response, "output", None) or response.predictions
    assert output, f"Endpoint '{endpoint_name}' returned an empty response"
