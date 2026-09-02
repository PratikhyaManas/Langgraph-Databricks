"""
Unit tests for the LangGraph workflow. Runs fully locally — the LLM and SQL
execution are faked so this suite needs no Databricks credentials and stays
fast enough to run on every PR.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.agent.graph import build_graph


class FakeLLM:
    """Returns canned responses so tests don't hit a real endpoint."""

    def __init__(self, responses: list[str]):
        self._responses = iter(responses)

    def invoke(self, _messages):
        return SimpleNamespace(content=next(self._responses))


@pytest.fixture
def fake_rows():
    return [{"customer": "Acme", "revenue": 100000}]


def test_happy_path(fake_rows):
    llm = FakeLLM(
        responses=[
            "SELECT customer, revenue FROM gold.sales ORDER BY revenue DESC LIMIT 5",
            "Acme is your top customer with $100,000 in revenue.",
        ]
    )
    graph = build_graph(llm)

    with patch("src.agent.nodes.run_sql", return_value=fake_rows):
        result = graph.invoke(
            {"messages": [{"role": "user", "content": "Who is my top customer by revenue?"}]}
        )

    assert result["generated_sql"].upper().startswith("SELECT")
    assert result["query_result"] == fake_rows
    assert "Acme" in result["final_answer"]
    assert result.get("error") is None


def test_sql_generation_failure_routes_to_error_node():
    class RaisingLLM:
        def invoke(self, _messages):
            raise RuntimeError("model unavailable")

    graph = build_graph(RaisingLLM())
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "Who is my top customer?"}]}
    )

    assert result["error"].startswith("sql_generation_failed")
    assert "went wrong" in result["final_answer"]


def test_sql_execution_failure_is_captured():
    llm = FakeLLM(responses=["SELECT * FROM does_not_exist"])
    graph = build_graph(llm)

    with patch(
        "src.agent.nodes.run_sql", side_effect=RuntimeError("table not found")
    ):
        result = graph.invoke(
            {"messages": [{"role": "user", "content": "Show me a nonexistent table"}]}
        )

    assert result["error"].startswith("sql_execution_failed")
