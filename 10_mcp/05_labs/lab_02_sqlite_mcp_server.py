"""
lab_02_sqlite_mcp_server.py — SQLite MCP Server
================================================
A complete database interface via MCP that lets an LLM:
  - Create tables with custom schemas
  - Insert, update, and delete records
  - Query data with safe parameterised SQL
  - Inspect the database schema

Security practices:
  - Only SELECT allowed via the generic query() tool
  - DDL (CREATE/DROP) only through explicit tools
  - All data mutations go through validated tools

Tools:
  create_table(name, columns)   → create a new table
  insert_row(table, data)       → insert a JSON record
  update_rows(table, data, where) → update matching rows
  delete_rows(table, where)     → delete matching rows
  query(sql)                    → run a SELECT query
  list_tables()                 → list all tables
  describe_table(name)          → show column schema
  drop_table(name)              → delete a table

Resources:
  db://schema                   → full database schema
  db://tables/{name}            → all rows from a table

Run:
    python 10_mcp/05_labs/lab_02_sqlite_mcp_server.py
"""

import json
import sqlite3
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("sqlite-server")

# Database file lives next to this script
DB_PATH = Path(__file__).parent / "lab_database.db"


# ── DB helper ─────────────────────────────────────────────────────────────────

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row  # enables dict-like access
    return conn


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def create_table(name: str, columns: str) -> str:
    """
    Create a new SQLite table.

    Args:
        name: Table name (alphanumeric and underscores only).
        columns: Column definitions as a SQL string.
                 Example: "id INTEGER PRIMARY KEY, name TEXT, age INTEGER"

    Returns:
        Confirmation message.
    """
    # Validate table name (prevent injection)
    if not name.replace("_", "").isalnum():
        raise ValueError(f"Invalid table name: '{name}'. Use alphanumeric and underscores only.")

    sql = f"CREATE TABLE IF NOT EXISTS {name} ({columns})"
    with get_conn() as conn:
        conn.execute(sql)

    return f"✅ Table '{name}' created (or already exists)."


@mcp.tool()
def insert_row(table: str, data: str) -> str:
    """
    Insert a row into a table.

    Args:
        table: Target table name.
        data: JSON object with column names as keys.
              Example: '{"name": "Alice", "age": 30}'

    Returns:
        The row ID of the inserted record.
    """
    if not table.replace("_", "").isalnum():
        raise ValueError(f"Invalid table name: '{table}'")

    record = json.loads(data)
    columns = ", ".join(record.keys())
    placeholders = ", ".join("?" for _ in record)
    values = list(record.values())

    sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
    with get_conn() as conn:
        cursor = conn.execute(sql, values)
        return f"✅ Inserted row with id={cursor.lastrowid} into '{table}'."


@mcp.tool()
def update_rows(table: str, data: str, where: str) -> str:
    """
    Update rows in a table matching a WHERE condition.

    Args:
        table: Target table name.
        data: JSON object with columns to update.
              Example: '{"age": 31}'
        where: SQL WHERE clause (without the WHERE keyword).
               Example: "name = 'Alice'"

    Returns:
        Number of rows updated.
    """
    if not table.replace("_", "").isalnum():
        raise ValueError(f"Invalid table name: '{table}'")

    record = json.loads(data)
    set_clause = ", ".join(f"{col} = ?" for col in record.keys())
    values = list(record.values())
    sql = f"UPDATE {table} SET {set_clause} WHERE {where}"

    with get_conn() as conn:
        cursor = conn.execute(sql, values)
        return f"✅ Updated {cursor.rowcount} row(s) in '{table}'."


@mcp.tool()
def delete_rows(table: str, where: str) -> str:
    """
    Delete rows from a table matching a WHERE condition.

    Args:
        table: Target table name.
        where: SQL WHERE clause (without WHERE keyword).
               Example: "id = 5" or "age < 18"

    Returns:
        Number of rows deleted.
    """
    if not table.replace("_", "").isalnum():
        raise ValueError(f"Invalid table name: '{table}'")

    sql = f"DELETE FROM {table} WHERE {where}"
    with get_conn() as conn:
        cursor = conn.execute(sql)
        return f"🗑️ Deleted {cursor.rowcount} row(s) from '{table}'."


@mcp.tool()
def query(sql: str) -> str:
    """
    Run a SELECT query and return results as JSON.

    Only SELECT statements are allowed for safety.

    Args:
        sql: A SQL SELECT statement.
             Example: "SELECT * FROM users WHERE age > 25"

    Returns:
        JSON array of matching rows.
    """
    sql_stripped = sql.strip().upper()
    if not sql_stripped.startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed through this tool. "
                         "Use insert_row, update_rows, or delete_rows for mutations.")

    with get_conn() as conn:
        cursor = conn.execute(sql)
        rows = [dict(row) for row in cursor.fetchall()]

    return json.dumps(rows, indent=2, default=str)


@mcp.tool()
def list_tables() -> str:
    """
    List all tables in the database with their row counts.

    Returns:
        JSON array of table names and row counts.
    """
    with get_conn() as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()

        result = []
        for (name,) in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            result.append({"table": name, "rows": count})

    if not result:
        return "No tables found. Create one with create_table."
    return json.dumps(result, indent=2)


@mcp.tool()
def describe_table(name: str) -> str:
    """
    Show the schema (columns, types, constraints) for a table.

    Args:
        name: Table name to describe.

    Returns:
        JSON array of column definitions.
    """
    if not name.replace("_", "").isalnum():
        raise ValueError(f"Invalid table name: '{name}'")

    with get_conn() as conn:
        columns = conn.execute(f"PRAGMA table_info({name})").fetchall()

    if not columns:
        raise ValueError(f"Table '{name}' not found.")

    schema = [
        {
            "column_id": col[0],
            "name": col[1],
            "type": col[2],
            "not_null": bool(col[3]),
            "default": col[4],
            "primary_key": bool(col[5]),
        }
        for col in columns
    ]
    return json.dumps(schema, indent=2)


@mcp.tool()
def drop_table(name: str) -> str:
    """
    Delete a table and all its data permanently.

    Args:
        name: Table name to delete.

    Returns:
        Confirmation message.
    """
    if not name.replace("_", "").isalnum():
        raise ValueError(f"Invalid table name: '{name}'")

    with get_conn() as conn:
        conn.execute(f"DROP TABLE IF EXISTS {name}")
    return f"🗑️ Table '{name}' dropped."


# ── Resources ─────────────────────────────────────────────────────────────────

@mcp.resource("db://schema")
def get_schema() -> str:
    """
    Return the full database schema as SQL CREATE statements.
    Resource URI: db://schema
    """
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()

    if not rows:
        return "-- Empty database (no tables yet)"
    return "\n\n".join(f"-- Table: {name}\n{sql};" for name, sql in rows if sql)


@mcp.resource("db://tables/{name}")
def get_table_data(name: str) -> str:
    """
    Return all rows from a table as JSON.
    Resource URI: db://tables/{name}
    """
    if not name.replace("_", "").isalnum():
        return f"Invalid table name: '{name}'"
    try:
        with get_conn() as conn:
            rows = conn.execute(f"SELECT * FROM {name}").fetchall()
        return json.dumps([dict(row) for row in rows], indent=2, default=str)
    except sqlite3.OperationalError:
        return f"Table '{name}' not found."


if __name__ == "__main__":
    print(f"🗄️  SQLite MCP Server starting...")
    print(f"   Database: {DB_PATH}")
    print("   Tools: create_table, insert_row, update_rows, delete_rows,")
    print("          query, list_tables, describe_table, drop_table")
    print("   Resources: db://schema, db://tables/{{name}}")
    mcp.run()
