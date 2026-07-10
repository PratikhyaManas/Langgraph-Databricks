"""
MLflow ResponsesAgent wrapper — the bridge that lets Databricks Model Serving
invoke the LangGraph workflow as a standard chat-completions-style endpoint.

This is what gets logged with mlflow.pyfunc.log_model(...) in
deploy/log_and_register_model.py.
"""
from __future__ import annotations

from typing import Any

import mlflow
from langchain_community.chat_models import ChatDatabricks
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from src.agent.graph import build_graph
from src.utils.config import AgentConfig


class SQLWorkflowResponsesAgent(ResponsesAgent):
    """Wraps the compiled LangGraph graph in MLflow's ResponsesAgent interface."""

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig.from_env()
        self._graph = None  # built lazily in load_context / first predict

    def _ensure_graph(self):
        if self._graph is None:
            llm = ChatDatabricks(endpoint=self.config.foundation_model_name)
            self._graph = build_graph(llm)
        return self._graph

    def load_context(self, context):  # noqa: D401  (mlflow pyfunc hook)
        self._ensure_graph()

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        graph = self._ensure_graph()
        messages = [{"role": m.role, "content": m.content} for m in request.input]
        result = graph.invoke({"messages": messages})
        return ResponsesAgentResponse(
            output=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": result["final_answer"]}
                    ],
                }
            ]
        )


mlflow.models.set_model(SQLWorkflowResponsesAgent())
