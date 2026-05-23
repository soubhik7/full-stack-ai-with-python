"""
App 14 — MCP Task Manager Server
==================================
A production-ready MCP server for managing tasks.
Connects to Claude Desktop, Claude Code, or client.py in this folder.

Run: python 08_ai_apps/14_mcp_server/server.py
Inspect: mcp dev 08_ai_apps/14_mcp_server/server.py
"""

import json
import uuid
from datetime import datetime
from typing import Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("task-manager")

# ── In-memory task store ──────────────────────────────────────────────────────

_tasks: dict[str, dict] = {}

PRIORITIES = {"low", "medium", "high", "urgent"}


def _make_task(title: str, priority: str = "medium", notes: str = "") -> dict:
    return {
        "id": str(uuid.uuid4())[:8],
        "title": title,
        "priority": priority,
        "status": "pending",
        "notes": notes,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
    }


# Pre-populate with sample tasks
_tasks["task1"] = _make_task("Review MCP documentation", "high")
_tasks["task1"]["id"] = "task1"
_tasks["task2"] = _make_task("Build calculator MCP server", "medium")
_tasks["task2"]["id"] = "task2"
_tasks["task3"] = _make_task("Write unit tests", "low")
_tasks["task3"]["id"] = "task3"


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def add_task(title: str, priority: str = "medium", notes: str = "") -> str:
    """
    Add a new task to the task list.

    Args:
        title: Clear, action-oriented task title.
        priority: Task priority — one of: low, medium, high, urgent.
        notes: Optional additional details or context.

    Returns:
        Confirmation with the new task's ID.
    """
    if priority not in PRIORITIES:
        raise ValueError(f"Priority must be one of: {', '.join(sorted(PRIORITIES))}")

    task = _make_task(title, priority, notes)
    _tasks[task["id"]] = task
    return f"✅ Task created: [{task['id']}] {title} (priority: {priority})"


@mcp.tool()
def complete_task(task_id: str) -> str:
    """
    Mark a task as completed.

    Args:
        task_id: The task's unique ID (e.g. 'task1' or the short UUID).

    Returns:
        Confirmation message.
    """
    task = _tasks.get(task_id)
    if not task:
        raise ValueError(f"Task '{task_id}' not found. Use list_tasks() to see available IDs.")

    if task["status"] == "done":
        return f"ℹ️ Task '{task_id}' is already completed."

    task["status"] = "done"
    task["completed_at"] = datetime.now().isoformat()
    return f"✅ Completed: [{task_id}] {task['title']}"


@mcp.tool()
def list_tasks(status: str = "all") -> str:
    """
    List tasks filtered by status.

    Args:
        status: Filter tasks by status. One of: 'pending', 'done', 'all'.

    Returns:
        JSON array of matching tasks sorted by priority.
    """
    if status not in ("pending", "done", "all"):
        raise ValueError("Status must be 'pending', 'done', or 'all'.")

    PRIORITY_ORDER = {"urgent": 0, "high": 1, "medium": 2, "low": 3}

    tasks = list(_tasks.values())
    if status != "all":
        tasks = [t for t in tasks if t["status"] == status]

    tasks.sort(key=lambda t: PRIORITY_ORDER.get(t["priority"], 4))

    if not tasks:
        return f"No {status} tasks found."
    return json.dumps(tasks, indent=2)


@mcp.tool()
def update_task(task_id: str, title: Optional[str] = None, priority: Optional[str] = None, notes: Optional[str] = None) -> str:
    """
    Update an existing task's title, priority, or notes.

    Args:
        task_id: ID of the task to update.
        title: New title (leave None to keep current).
        priority: New priority (leave None to keep current).
        notes: New notes (leave None to keep current).

    Returns:
        Confirmation with updated fields.
    """
    task = _tasks.get(task_id)
    if not task:
        raise ValueError(f"Task '{task_id}' not found.")

    updated = []
    if title is not None:
        task["title"] = title
        updated.append("title")
    if priority is not None:
        if priority not in PRIORITIES:
            raise ValueError(f"Priority must be one of: {', '.join(sorted(PRIORITIES))}")
        task["priority"] = priority
        updated.append("priority")
    if notes is not None:
        task["notes"] = notes
        updated.append("notes")

    if not updated:
        return "No fields updated (all parameters were None)."
    return f"✅ Updated task '{task_id}': {', '.join(updated)}"


@mcp.tool()
def delete_task(task_id: str) -> str:
    """
    Permanently delete a task.

    Args:
        task_id: ID of the task to delete.

    Returns:
        Confirmation message.
    """
    task = _tasks.pop(task_id, None)
    if not task:
        raise ValueError(f"Task '{task_id}' not found.")
    return f"🗑️ Deleted task: [{task_id}] {task['title']}"


@mcp.tool()
def get_stats() -> str:
    """
    Get task completion statistics and a productivity summary.

    Returns:
        JSON with total, pending, done counts, and completion rate.
    """
    all_tasks = list(_tasks.values())
    pending = [t for t in all_tasks if t["status"] == "pending"]
    done = [t for t in all_tasks if t["status"] == "done"]

    priority_breakdown = {}
    for t in pending:
        p = t["priority"]
        priority_breakdown[p] = priority_breakdown.get(p, 0) + 1

    stats = {
        "total_tasks": len(all_tasks),
        "pending": len(pending),
        "completed": len(done),
        "completion_rate": f"{len(done) / len(all_tasks) * 100:.0f}%" if all_tasks else "0%",
        "pending_by_priority": priority_breakdown,
        "urgent_tasks": [t["title"] for t in pending if t["priority"] == "urgent"],
    }
    return json.dumps(stats, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# RESOURCES
# ══════════════════════════════════════════════════════════════════════════════

@mcp.resource("tasks://all")
def get_all_tasks() -> str:
    """All tasks as JSON. URI: tasks://all"""
    return json.dumps(list(_tasks.values()), indent=2)


@mcp.resource("tasks://pending")
def get_pending_tasks() -> str:
    """All pending tasks. URI: tasks://pending"""
    pending = [t for t in _tasks.values() if t["status"] == "pending"]
    return json.dumps(pending, indent=2)


@mcp.resource("tasks://stats")
def get_task_stats() -> str:
    """Task statistics. URI: tasks://stats"""
    return get_stats()


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

@mcp.prompt()
def daily_standup() -> str:
    """
    Generate a daily standup update from current tasks.
    Returns a prompt for Claude to write a standup based on task state.
    """
    pending = [t for t in _tasks.values() if t["status"] == "pending"]
    done_today = [
        t for t in _tasks.values()
        if t["status"] == "done" and t.get("completed_at", "")[:10] == datetime.now().date().isoformat()
    ]

    pending_str = "\n".join(f"- [{t['priority'].upper()}] {t['title']}" for t in pending)
    done_str = "\n".join(f"- {t['title']}" for t in done_today) if done_today else "None completed today"

    return f"""Based on this task state, write a concise daily standup update:

COMPLETED TODAY:
{done_str}

PENDING TASKS:
{pending_str}

Write the standup in first person. Include:
1. What I completed (if anything)
2. What I'm working on today
3. Any blockers or concerns (based on urgent/high priority items)

Keep it under 100 words. Use a professional, direct tone."""


@mcp.prompt()
def prioritise_tasks() -> str:
    """
    Generate a prompt to help Claude prioritise the current task list.
    """
    all_tasks = [t for t in _tasks.values() if t["status"] == "pending"]
    task_str = "\n".join(
        f"- [{t['id']}] {t['title']} (current priority: {t['priority']})"
        for t in all_tasks
    )

    return f"""Here is my current task list:

{task_str}

Please help me prioritise these tasks. Consider:
1. Impact vs effort (which gives the most value for the least time?)
2. Dependencies (which tasks unblock others?)
3. Urgency vs importance (Eisenhower matrix)

Suggest: (a) the top 3 tasks to focus on today, and (b) any tasks that should be delegated or dropped."""


if __name__ == "__main__":
    print("🗂️  Task Manager MCP Server")
    print(f"   Pre-loaded: {len(_tasks)} sample tasks")
    print("   Tools: add_task, complete_task, list_tasks, update_task, delete_task, get_stats")
    print("   Resources: tasks://all, tasks://pending, tasks://stats")
    print("   Prompts: daily_standup, prioritise_tasks")
    print("\n   Waiting for MCP client connection (stdio)...")
    mcp.run()
