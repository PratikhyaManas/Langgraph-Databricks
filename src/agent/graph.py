"""Builds the LangGraph StateGraph from the node functions in nodes.py."""
from __future__ import annotations

from functools import partial

from langgraph.graph import END, StateGraph

from src.agent.state import SQLWorkflowState
from src.agent import nodes


def build_graph(llm):
    """Construct and compile the SQL workflow graph.

    Parameters
    ----------
    llm : a LangChain-compatible chat model (e.g. ChatDatabricks) bound to the
        foundation model configured in AgentConfig.foundation_model_name.
    """
    graph = StateGraph(SQLWorkflowState)

    graph.add_node("extract_question", nodes.extract_question)
    graph.add_node("generate_sql", partial(nodes.generate_sql, llm=llm))
    graph.add_node("execute_sql", nodes.execute_sql)
    graph.add_node("summarize_result", partial(nodes.summarize_result, llm=llm))
    graph.add_node("error", lambda state: {"final_answer": state.get("error")})

    graph.set_entry_point("extract_question")
    graph.add_edge("extract_question", "generate_sql")
    graph.add_conditional_edges(
        "generate_sql",
        nodes.route_after_generate_sql,
        {"execute_sql": "execute_sql", "error": "error"},
    )
    graph.add_edge("execute_sql", "summarize_result")
    graph.add_edge("summarize_result", END)
    graph.add_edge("error", END)

    return graph.compile()
