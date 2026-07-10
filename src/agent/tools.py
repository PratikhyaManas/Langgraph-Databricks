"""
Tools the graph nodes call out to: schema lookup and SQL execution against
Unity Catalog. Kept separate from nodes.py so they can be unit tested and
mocked independently of the LLM calls.
"""
from __future__ import annotations

import os
import re
from typing import Any

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_BLOCKED_SQL_PATTERNS = (
    r"\bINSERT\b",
    r"\bUPDATE\b",
    r"\bDELETE\b",
    r"\bMERGE\b",
    r"\bDROP\b",
    r"\bALTER\b",
    r"\bTRUNCATE\b",
    r"\bCREATE\b",
    r"\bGRANT\b",
    r"\bREVOKE\b",
)

_IDENTIFIER_SEGMENT = r"(?:`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*)"
_TABLE_REF_RE = re.compile(
    rf"\b(?:FROM|JOIN)\s+(({_IDENTIFIER_SEGMENT})(?:\s*\.\s*{_IDENTIFIER_SEGMENT}){{0,2}})",
    re.IGNORECASE,
)
_CTE_NAME_RE = re.compile(r"(?:\bWITH\b|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(", re.IGNORECASE)


def infer_schema_context() -> str:
    """Build schema context from env-configured tables for safer SQL generation."""
    catalog = os.environ.get("CATALOG_NAME")
    schema = os.environ.get("SCHEMA_NAME")
    tables_raw = os.environ.get("SQL_CONTEXT_TABLES", "")
    tables = [t.strip() for t in tables_raw.split(",") if t.strip()]

    if not catalog or not schema or not tables:
        return ""

    schema_bits: list[str] = []
    for table in tables:
        try:
            schema_bits.append(get_table_schema(catalog=catalog, schema=schema, table=table))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Schema lookup failed for %s.%s.%s: %s", catalog, schema, table, exc)
    return "\n".join(schema_bits)


def validate_safe_select_sql(query: str) -> str:
    """Validate SQL safety constraints and normalize trailing semicolons."""
    normalized = query.strip().rstrip(";")
    if not normalized:
        raise ValueError("empty SQL")

    if not re.match(r"(?is)^\s*(?:WITH\b[\s\S]*?\bSELECT\b|SELECT\b)", normalized):
        raise ValueError("only SELECT statements are allowed")

    for pattern in _BLOCKED_SQL_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            raise ValueError("non-read-only SQL statement detected")

    _validate_allowed_tables(normalized)

    return normalized


def _validate_allowed_tables(query: str) -> None:
    """Ensure referenced tables are in SQL_ALLOWED_TABLES when configured."""
    allowed_raw = os.environ.get("SQL_ALLOWED_TABLES", "")
    allowed = {t.strip().lower() for t in allowed_raw.split(",") if t.strip()}
    if not allowed:
        return

    referenced_tables = _extract_referenced_tables(query)
    disallowed = sorted(t for t in referenced_tables if t not in allowed)
    if disallowed:
        raise ValueError(
            "query references disallowed tables: " + ", ".join(disallowed)
        )


def _extract_referenced_tables(query: str) -> set[str]:
    """Extract normalized table refs from FROM/JOIN clauses, excluding CTE aliases."""
    cte_names = _extract_cte_names(query)
    tables: set[str] = set()
    for match in _TABLE_REF_RE.finditer(query):
        table_ref = _normalize_table_identifier(match.group(1))
        if table_ref and table_ref not in cte_names:
            tables.add(table_ref)
    return tables


def _extract_cte_names(query: str) -> set[str]:
    """Collect CTE alias names defined in WITH clauses."""
    return {m.group(1).strip().lower() for m in _CTE_NAME_RE.finditer(query)}


def _normalize_table_identifier(identifier: str) -> str:
    """Normalize quoted identifiers to lowercase dot notation for matching."""
    parts = [part.strip() for part in re.split(r"\s*\.\s*", identifier.strip()) if part.strip()]
    normalized_parts: list[str] = []
    for part in parts:
        if (part.startswith("`") and part.endswith("`")) or (
            part.startswith('"') and part.endswith('"')
        ):
            part = part[1:-1]
        elif part.startswith("[") and part.endswith("]"):
            part = part[1:-1]
        normalized_parts.append(part.strip().lower())
    return ".".join(normalized_parts)


def enforce_sql_limit(query: str, max_rows: int = 1000) -> str:
    """Append LIMIT when not present to reduce runaway queries."""
    if re.search(r"\bLIMIT\s+\d+\b", query, flags=re.IGNORECASE):
        return query
    return f"{query} LIMIT {max_rows}"


def get_table_schema(catalog: str, schema: str, table: str) -> str:
    """Return a compact DDL-like description of a table for prompting the LLM.

    In a real deployment this queries information_schema via the Databricks
    SQL connector / Spark session available inside the serving container.
    """
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
    safe_query = validate_safe_select_sql(query)
    bounded_query = enforce_sql_limit(safe_query)
    logger.info("Executing generated SQL: %s", bounded_query)
    with _get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(bounded_query)
            columns = [c[0] for c in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _get_connection():
    """Lazily build a Databricks SQL connector connection.

    Imported lazily / wrapped so unit tests can monkeypatch this function
    instead of needing real workspace credentials.
    """
    from databricks import sql as databricks_sql

    return databricks_sql.connect(
        server_hostname=os.environ["DATABRICKS_SQL_HOST"],
        http_path=os.environ["DATABRICKS_SQL_HTTP_PATH"],
        access_token=os.environ.get("DATABRICKS_TOKEN"),
    )
