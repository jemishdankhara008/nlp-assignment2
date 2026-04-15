"""
mcp_server.py — MCP (Model Context Protocol) server for the Research Assistant Agent.

Exposes three tools over stdio:
  - save_research  : Persist a research note (title, summary, sources) to SQLite.
  - list_research  : Return all saved research entries (title + date).
  - search_research: Keyword search over saved research entries.

Run directly:
    python mcp_server.py

The agent launches this as a subprocess and communicates via stdin/stdout.
"""

import sqlite3
import json
from datetime import datetime
from typing import Any

from mcp.server.fastmcp import FastMCP
import config

# ---------------------------------------------------------------------------
# Server initialisation
# ---------------------------------------------------------------------------

server = FastMCP("research-assistant-mcp")


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _get_connection() -> sqlite3.Connection:
    """Open (or create) the SQLite database and ensure the table exists."""
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS research (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            title     TEXT    NOT NULL,
            summary   TEXT    NOT NULL,
            sources   TEXT    NOT NULL,
            timestamp TEXT    NOT NULL
        )
        """
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# MCP tool definitions
# ---------------------------------------------------------------------------

@server.tool()
def save_research(
    title: str,
    summary: str,
    sources: list[str],
    timestamp: str = "",
) -> dict[str, Any]:
    """
    Save a research note to the local SQLite database.

    Args:
        title:     Short descriptive title for the research topic.
        summary:   Full LLM-generated summary text.
        sources:   List of source URLs referenced in the summary.
        timestamp: ISO-format datetime string. Defaults to current UTC time.

    Returns:
        A dict with 'status' ('ok' or 'error') and an optional 'message'.
    """
    if not timestamp:
        timestamp = datetime.utcnow().isoformat()
    try:
        conn = _get_connection()
        conn.execute(
            "INSERT INTO research (title, summary, sources, timestamp) VALUES (?, ?, ?, ?)",
            (title, summary, json.dumps(sources), timestamp),
        )
        conn.commit()
        conn.close()
        return {"status": "ok", "message": f"Research '{title}' saved successfully."}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@server.tool()
def list_research() -> dict[str, Any]:
    """
    Return all saved research entries (title and date only).

    Returns:
        A dict with 'entries' (list of {id, title, timestamp}) or 'error'.
    """
    try:
        conn = _get_connection()
        rows = conn.execute(
            "SELECT id, title, timestamp FROM research ORDER BY id DESC"
        ).fetchall()
        conn.close()
        entries = [{"id": r[0], "title": r[1], "timestamp": r[2]} for r in rows]
        return {"status": "ok", "entries": entries}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@server.tool()
def search_research(keyword: str) -> dict[str, Any]:
    """
    Search saved research entries by keyword (matches title or summary).

    Args:
        keyword: The search term to look for (case-insensitive).

    Returns:
        A dict with 'matches' (list of {id, title, timestamp, summary}) or 'error'.
    """
    try:
        conn = _get_connection()
        pattern = f"%{keyword}%"
        rows = conn.execute(
            """
            SELECT id, title, timestamp, summary
            FROM research
            WHERE title LIKE ? OR summary LIKE ?
            ORDER BY id DESC
            """,
            (pattern, pattern),
        ).fetchall()
        conn.close()
        matches = [
            {"id": r[0], "title": r[1], "timestamp": r[2], "summary": r[3]}
            for r in rows
        ]
        return {"status": "ok", "matches": matches}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    server.run(transport="stdio")
