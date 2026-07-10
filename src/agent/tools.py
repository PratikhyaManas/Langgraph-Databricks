"""
Tools the graph nodes call out to: schema lookup and SQL execution against
Unity Catalog. Kept separate from nodes.py so they can be unit tested and
mocked independently of the LLM calls.
"""
from __future__ import annotations

from typing import Any

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def get_table_schema(catalog: str, schema: str, table: str) -> str:
    """Return a compact DDL-like description of a table for prompting the LLM.

    In a real deployment this queries information_schema via the Databricks
    SQL connector / Spark session available inside the serving container.
    """
    from databricks import sql as databricks_sql  # local import: only needed at runtime

    query = f"""
        SELECT column_name, data_type
        FROM {catalog}.information_schema.columns
        WHERE table_schema = '{schema}' AND table_name = '{table}'
        ORDER BY ordinal_position
    """
    with _get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
    columns = ", ".join(f"{r.column_name} {r.data_type}" for r in rows)
    return f"{catalog}.{schema}.{table}({columns})"


def run_sql(query: str) -> list[dict[str, Any]]:
    """Execute a SQL query against the workspace SQL warehouse and return rows."""
    logger.info("Executing generated SQL: %s", query)
    with _get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            columns = [c[0] for c in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _get_connection():
    """Lazily build a Databricks SQL connector connection.

    Imported lazily / wrapped so unit tests can monkeypatch this function
    instead of needing real workspace credentials.
    """
    import os

    from databricks import sql as databricks_sql

    return databricks_sql.connect(
        server_hostname=os.environ["DATABRICKS_SQL_HOST"],
        http_path=os.environ["DATABRICKS_SQL_HTTP_PATH"],
        access_token=os.environ.get("DATABRICKS_TOKEN"),
    )
