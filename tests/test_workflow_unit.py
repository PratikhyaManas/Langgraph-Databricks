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
from src.agent.tools import enforce_sql_limit, validate_safe_select_sql
from src.utils.config import AgentConfig


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
    assert result["final_answer"].startswith("Sorry, something went wrong:")


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


def test_dict_style_user_message_is_supported(fake_rows):
    llm = FakeLLM(
        responses=[
            "SELECT customer, revenue FROM gold.sales ORDER BY revenue DESC LIMIT 5",
            "Acme is your top customer with $100,000 in revenue.",
        ]
    )
    graph = build_graph(llm)

    with patch("src.agent.nodes.run_sql", return_value=fake_rows):
        result = graph.invoke(
            {
                "messages": [
                    {"role": "user", "content": "Who is my top customer by revenue?"}
                ]
            }
        )

    assert result["question"] == "Who is my top customer by revenue?"
    assert result["query_result"] == fake_rows
    assert "Acme" in result["final_answer"]


def test_sql_markdown_fence_is_cleaned_before_execution(fake_rows):
    llm = FakeLLM(
        responses=[
            "```sql\nSELECT customer, revenue FROM gold.sales ORDER BY revenue DESC LIMIT 5\n```",
            "Acme is your top customer with $100,000 in revenue.",
        ]
    )
    graph = build_graph(llm)

    with patch("src.agent.nodes.run_sql", return_value=fake_rows) as mock_run_sql:
        graph.invoke(
            {
                "messages": [
                    {"role": "user", "content": "Who is my top customer by revenue?"}
                ]
            }
        )

    assert mock_run_sql.call_count == 1
    executed_sql = mock_run_sql.call_args.args[0]
    assert executed_sql == "SELECT customer, revenue FROM gold.sales ORDER BY revenue DESC LIMIT 5"


def test_sql_guardrails_block_non_select():
    with pytest.raises(ValueError, match="only SELECT statements are allowed"):
        validate_safe_select_sql("DELETE FROM gold.sales")


def test_sql_guardrails_append_default_limit():
    sql = "SELECT customer FROM gold.sales"
    assert enforce_sql_limit(sql).endswith("LIMIT 1000")


def test_sql_allowlist_accepts_configured_table(monkeypatch):
    monkeypatch.setenv("SQL_ALLOWED_TABLES", "gold.sales,gold.customers")
    sql = "SELECT customer FROM gold.sales"
    assert validate_safe_select_sql(sql) == sql


def test_sql_allowlist_rejects_disallowed_table(monkeypatch):
    monkeypatch.setenv("SQL_ALLOWED_TABLES", "gold.sales")
    with pytest.raises(ValueError, match="disallowed tables"):
        validate_safe_select_sql("SELECT customer FROM gold.secret_table")


def test_sql_allowlist_supports_quoted_identifiers(monkeypatch):
    monkeypatch.setenv("SQL_ALLOWED_TABLES", "gold.sales")
    sql = 'SELECT customer FROM "gold"."sales"'
    assert validate_safe_select_sql(sql) == sql


def test_sql_allowlist_ignores_cte_alias(monkeypatch):
    monkeypatch.setenv("SQL_ALLOWED_TABLES", "gold.sales")
    sql = "WITH recent AS (SELECT * FROM gold.sales) SELECT * FROM recent"
    assert validate_safe_select_sql(sql) == sql


def test_generate_sql_uses_schema_context(fake_rows):
    captured_messages = []

    class CapturingLLM:
        def invoke(self, messages):
            captured_messages.append(messages)
            if len(captured_messages) == 1:
                return SimpleNamespace(content="SELECT customer FROM gold.sales")
            return SimpleNamespace(content="Acme")

    graph = build_graph(CapturingLLM())
    with patch("src.agent.nodes.run_sql", return_value=fake_rows):
        graph.invoke(
            {
                "messages": [{"role": "user", "content": "Top customer?"}],
                "schema_context": "dev_catalog.gold.sales(customer STRING, revenue DOUBLE)",
            }
        )

    prompt_payload = captured_messages[0][1]["content"]
    assert "Schema context:" in prompt_payload
    assert "dev_catalog.gold.sales" in prompt_payload


def test_agent_config_rejects_dotted_names(monkeypatch):
    monkeypatch.setenv("CATALOG_NAME", "dev.catalog")
    monkeypatch.setenv("SCHEMA_NAME", "gold")
    monkeypatch.setenv("EXPERIMENT_ID", "123")
    monkeypatch.setenv("FOUNDATION_MODEL_NAME", "databricks-meta-llama-3-1-70b-instruct")
    monkeypatch.setenv("ENDPOINT_NAME", "sql-workflow-agent-dev")
    monkeypatch.setenv("REGISTERED_MODEL_NAME", "sql_workflow_agent")

    with pytest.raises(ValueError, match="must be simple names without dots"):
        AgentConfig.from_env()
