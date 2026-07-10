"""
Individual LangGraph node functions. Each node takes and returns a partial
SQLWorkflowState. Kept as small, pure-ish functions so they're easy to unit
test with a fake/mocked LLM client.
"""
from __future__ import annotations

from src.agent.state import SQLWorkflowState
from src.agent.tools import run_sql
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

SQL_GENERATION_SYSTEM_PROMPT = (
    "You are a SQL analyst. Given a user question and table schema, write a "
    "single valid SQL SELECT statement that answers the question. "
    "Return only the SQL, no explanation."
)

ANSWER_SYSTEM_PROMPT = (
    "You turn SQL query results into a short, direct natural-language answer "
    "for a business user. Do not mention SQL or tables explicitly."
)


def extract_question(state: SQLWorkflowState, llm=None) -> dict:
    """Pull the latest user question out of the message history."""
    last_user_msg = next(
        (m for m in reversed(state["messages"]) if getattr(m, "type", None) == "human"),
        None,
    )
    question = last_user_msg.content if last_user_msg else ""
    logger.info("Extracted question: %s", question)
    return {"question": question}


def generate_sql(state: SQLWorkflowState, llm) -> dict:
    """Ask the LLM to turn the question into SQL."""
    try:
        response = llm.invoke(
            [
                {"role": "system", "content": SQL_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": state["question"]},
            ]
        )
        sql_text = response.content.strip().strip("```sql").strip("```").strip()
        return {"generated_sql": sql_text}
    except Exception as exc:  # noqa: BLE001
        logger.exception("SQL generation failed")
        return {"error": f"sql_generation_failed: {exc}"}


def execute_sql(state: SQLWorkflowState) -> dict:
    """Run the generated SQL and capture results (or an error)."""
    if state.get("error"):
        return {}
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


def route_after_generate_sql(state: SQLWorkflowState) -> str:
    """Conditional edge: skip execution if SQL generation already failed."""
    return "error" if state.get("error") else "execute_sql"
