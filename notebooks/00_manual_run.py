# Databricks notebook source
# MAGIC %md
# MAGIC # Manual / interactive run
# MAGIC This notebook is **not** part of the deploy pipeline anymore — CI calls
# MAGIC the scripts in `deploy/` directly. Keep this around only for poking at
# MAGIC the graph interactively inside the Databricks workspace (e.g. while
# MAGIC iterating on prompts).

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys
sys.path.insert(0, "..")

from langchain_community.chat_models import ChatDatabricks

from src.agent.graph import build_graph
from src.utils.config import AgentConfig

config = AgentConfig.from_env()
llm = ChatDatabricks(endpoint=config.foundation_model_name)
graph = build_graph(llm)

# COMMAND ----------

result = graph.invoke(
    {"messages": [{"role": "user", "content": "Who are my top 5 customers by revenue?"}]}
)
print(result["final_answer"])
