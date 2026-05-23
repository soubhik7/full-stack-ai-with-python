"""
03_resources_server.py — Tools + Resources + Prompts
=====================================================
Demonstrates all three MCP primitives:

  TOOLS     — callable actions (the LLM decides when to call)
  RESOURCES — read-only data the client can fetch
  PROMPTS   — reusable prompt templates with parameters

This server manages a simple in-memory note-taking system.

Tools:
  create_note(title, content)
  update_note(title, content)
  delete_note(title)
  list_notes()

Resources:
  notes://all           → all notes as JSON
  notes://{title}       → single note content

Prompts:
  summarise_notes()     → prompt to summarise all notes
  explain_note(title)   → prompt to explain a specific note
"""

import json
from datetime import datetime

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("notes-server")

# ── In-memory store ───────────────────────────────────────────────────────────
# (In production, this would be a database)
_notes: dict[str, dict] = {}


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS — the LLM calls these to perform write operations
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def create_note(title: str, content: str) -> str:
    """
    Create a new note with a title and content.

    Args:
        title: Unique title for the note (used as the key).
        content: The body text of the note.

    Returns:
        Confirmation message with the note title.
    """
    if title in _notes:
        raise ValueError(f"Note '{title}' already exists. Use update_note to modify it.")
    _notes[title] = {
        "content": content,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
    return f"✅ Note '{title}' created successfully."


@mcp.tool()
def update_note(title: str, content: str) -> str:
    """
    Update the content of an existing note.

    Args:
        title: Title of the note to update.
        content: New content to replace the existing content.

    Returns:
        Confirmation message.
    """
    if title not in _notes:
        raise ValueError(f"Note '{title}' not found. Use create_note to create it.")
    _notes[title]["content"] = content
    _notes[title]["updated_at"] = datetime.now().isoformat()
    return f"✅ Note '{title}' updated successfully."


@mcp.tool()
def delete_note(title: str) -> str:
    """
    Delete a note by title.

    Args:
        title: Title of the note to delete.

    Returns:
        Confirmation message.
    """
    if title not in _notes:
        raise ValueError(f"Note '{title}' not found.")
    del _notes[title]
    return f"🗑️ Note '{title}' deleted."


@mcp.tool()
def list_notes() -> str:
    """
    List all existing note titles and their creation dates.

    Returns:
        JSON array of note metadata (title + timestamps).
    """
    if not _notes:
        return "No notes yet. Create one with create_note."
    summary = [
        {"title": t, "created_at": v["created_at"], "updated_at": v["updated_at"]}
        for t, v in _notes.items()
    ]
    return json.dumps(summary, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# RESOURCES — read-only data views
# ══════════════════════════════════════════════════════════════════════════════

@mcp.resource("notes://all")
def get_all_notes() -> str:
    """
    Return all notes as a JSON object.
    Resource URI: notes://all
    """
    return json.dumps(_notes, indent=2)


@mcp.resource("notes://{title}")
def get_note(title: str) -> str:
    """
    Return the content of a specific note.
    Resource URI: notes://{title}

    Args:
        title: The note title (from the URI path).
    """
    if title not in _notes:
        return f"Note '{title}' not found."
    note = _notes[title]
    return f"# {title}\n\n{note['content']}\n\n---\nCreated: {note['created_at']}"


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS — reusable prompt templates
# ══════════════════════════════════════════════════════════════════════════════

@mcp.prompt()
def summarise_notes() -> str:
    """
    Generate a prompt to summarise all current notes.
    Returns a ready-to-use prompt string.
    """
    if not _notes:
        return "There are no notes to summarise yet."

    notes_text = "\n\n".join(
        f"### {title}\n{data['content']}"
        for title, data in _notes.items()
    )

    return f"""Please summarise the following notes concisely:

{notes_text}

Provide:
1. A one-paragraph overall summary
2. Key themes across all notes
3. Any action items or important dates mentioned"""


@mcp.prompt()
def explain_note(title: str) -> str:
    """
    Generate a prompt to explain a specific note in simple terms.

    Args:
        title: The title of the note to explain.
    """
    if title not in _notes:
        return f"Note '{title}' not found."

    content = _notes[title]["content"]
    return f"""Please explain the following note in simple terms that a non-expert could understand:

Title: {title}

Content:
{content}

Provide:
1. A plain-language explanation
2. Why this might be important
3. Any follow-up questions worth considering"""


if __name__ == "__main__":
    # Pre-populate with a couple of demo notes
    _notes["Meeting Notes"] = {
        "content": "Discussed Q3 roadmap. Key priorities: MCP integration, RAG improvements, UI overhaul.",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
    _notes["Ideas"] = {
        "content": "Build a tool that auto-generates MCP servers from OpenAPI specs.",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

    print("Notes MCP server starting (2 demo notes pre-loaded)...")
    print("Primitives: tools (create/update/delete/list), resources (all/single), prompts (summarise/explain)")
    mcp.run()
