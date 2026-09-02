"""
Individual LangGraph node functions. Each node takes and returns a partial
SQLWorkflowState. Kept as small, pure-ish functions so they're easy to unit
test with a fake/mocked LLM client.
"""
from __future__ import annotations

import re

from src.agent.state import SQLWorkflowState
from src.agent.tools import infer_schema_context, run_sql, validate_safe_select_sql
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

SQL_GENERATION_SYSTEM_PROMPT = (
    "You are a SQL analyst. Given a user question and table schema, write a "
    "single valid SQL SELECT statement that answers the question. "
    "Rules: produce read-only SQL only (SELECT), avoid DDL/DML, and return only SQL text."
)

ANSWER_SYSTEM_PROMPT = (
    "You turn SQL query results into a short, direct natural-language answer "
    "for a business user. Do not mention SQL or tables explicitly."
)

_SQL_CODE_BLOCK_RE = re.compile(r"```(?:sql)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def _message_role_and_content(message) -> tuple[str | None, str]:
    """Normalize graph message objects and dict-style chat messages."""
    if isinstance(message, dict):
        role = message.get("role")
        content = message.get("content", "")
        return role, str(content)
    role = getattr(message, "type", None)
    content = getattr(message, "content", "")
    return role, str(content)


def _extract_sql_text(raw_text: str) -> str:
    """Extract SQL safely from plain text or fenced markdown code blocks."""
    match = _SQL_CODE_BLOCK_RE.search(raw_text)
    if match:
        return match.group(1).strip()
    return raw_text.strip()


def extract_question(state: SQLWorkflowState, llm=None) -> dict:
    """Pull the latest user question out of the message history."""
    last_user_msg = None
    for m in reversed(state.get("messages", [])):
        role, content = _message_role_and_content(m)
        if role in {"human", "user"}:
            last_user_msg = content
            break

    question = last_user_msg or ""
    logger.info("Extracted question: %s", question)
    return {"question": question}


def generate_sql(state: SQLWorkflowState, llm) -> dict:
    """Ask the LLM to turn the question into SQL."""
    try:
        schema_context = state.get("schema_context") or infer_schema_context()
        user_prompt = (
            f"Question: {state['question']}\n"
            f"Schema context:\n{schema_context or 'No schema context provided.'}"
        )
        response = llm.invoke(
            [
                {"role": "system", "content": SQL_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        sql_text = _extract_sql_text(response.content)
        safe_sql = validate_safe_select_sql(sql_text)
        return {"generated_sql": safe_sql, "schema_context": schema_context}
    except Exception as exc:  # noqa: BLE001
        logger.exception("SQL generation failed")
        return {"error": f"sql_generation_failed: {exc}"}


def execute_sql(state: SQLWorkflowState) -> dict:
    """Run the generated SQL and capture results (or an error)."""
    if state.get("error"):
        return {}
    if not state.get("generated_sql"):
        return {"error": "sql_execution_failed: empty generated SQL"}
    try:
        rows = run_sql(state["generated_sql"])
        return {"query_result": rows}
    except Exception as exc:  # noqa: BLE001
        logger.exception("SQL execution failed")
        return {"error": f"sql_execution_failed: {exc}"}


def summarize_result(state: SQLWorkflowState, llm) -> dict:
    """Turn query results into a natural-language answer."""
    if state.get("error"):
        return {"final_answer": f"Sorry, something went wrong: {state['error']}"}
    try:
        response = llm.invoke(
            [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Question: {state['question']}\n"
                        f"Query result: {state['query_result']}"
                    ),
                },
            ]
        )
        return {"final_answer": response.content.strip()}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Result summarization failed")
        return {"final_answer": f"Sorry, something went wrong: summarization_failed: {exc}"}


def route_after_generate_sql(state: SQLWorkflowState) -> str:
    """Conditional edge: skip execution if SQL generation already failed."""
    return "error" if state.get("error") else "execute_sql"
