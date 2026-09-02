"""Graph state definition for the SQL workflow agent."""
from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph.message import add_messages


class SQLWorkflowState(TypedDict):
    """State threaded through every node in the graph."""

    # Full chat history, using LangGraph's built-in message reducer.
    messages: Annotated[list[Any], add_messages]

    # The natural-language question extracted from the latest user message.
    question: Optional[str]

    # SQL generated for the question.
    generated_sql: Optional[str]

    # Optional schema context injected into SQL generation prompts.
    schema_context: Optional[str]

    # Result of executing generated_sql (list of row dicts, or an error string).
    query_result: Optional[Any]

    # Final natural-language answer to return to the caller.
    final_answer: Optional[str]

    # Populated if a node hits an unrecoverable error; routes to the error node.
    error: Optional[str]
