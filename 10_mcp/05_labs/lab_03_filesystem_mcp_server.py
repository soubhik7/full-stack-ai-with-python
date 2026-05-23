"""
lab_03_filesystem_mcp_server.py — Safe Filesystem MCP Server
=============================================================
A filesystem MCP server that enforces a security sandbox:
  - All operations are restricted to ALLOWED_ROOT
  - Path traversal attacks (../../etc/passwd) are blocked
  - Large files (> 10 MB) are rejected
  - Binary files are detected and refused

Tools:
  read_file(path)               → file contents as text
  write_file(path, content)     → write/overwrite a file
  append_to_file(path, content) → append lines to a file
  list_directory(path)          → list files and folders
  create_directory(path)        → create a directory
  delete_file(path)             → delete a file
  file_info(path)               → size, dates, permissions
  search_files(pattern, root)   → find files matching a glob

Resources:
  file://{path}                 → read a file as a resource

Run:
    python 10_mcp/05_labs/lab_03_filesystem_mcp_server.py
"""

import json
import os
import stat
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("filesystem-server")

# ── Security sandbox ──────────────────────────────────────────────────────────
# Only allow operations inside this directory.
# Change this to restrict to a specific project folder.
ALLOWED_ROOT = (Path(__file__).parent / "sandbox").resolve()
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Create the sandbox directory if it doesn't exist
ALLOWED_ROOT.mkdir(exist_ok=True)
(ALLOWED_ROOT / "README.txt").write_text(
    "This is the MCP filesystem sandbox.\n"
    "All file operations are restricted to this directory.\n",
    encoding="utf-8",
)


def _safe_path(path: str) -> Path:
    """
    Resolve a relative path inside the sandbox.
    Raises ValueError if the path escapes ALLOWED_ROOT.
    """
    # Strip leading slashes to treat path as relative to sandbox
    clean = path.lstrip("/").lstrip("\\")
    full = (ALLOWED_ROOT / clean).resolve()

    # Path traversal check
    if not str(full).startswith(str(ALLOWED_ROOT)):
        raise ValueError(
            f"Access denied: '{path}' is outside the allowed directory.\n"
            f"Allowed root: {ALLOWED_ROOT}"
        )
    return full


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def read_file(path: str) -> str:
    """
    Read the contents of a text file from the sandbox.

    Args:
        path: Relative path from the sandbox root.
              Example: "notes.txt" or "subdir/data.json"

    Returns:
        The text contents of the file.
    """
    full = _safe_path(path)

    if not full.exists():
        raise FileNotFoundError(f"File not found: '{path}'")
    if not full.is_file():
        raise ValueError(f"'{path}' is a directory, not a file.")

    size = full.stat().st_size
    if size > MAX_FILE_SIZE:
        raise ValueError(f"File too large: {size / 1024 / 1024:.1f} MB (max 10 MB)")

    # Detect binary files
    try:
        content = full.read_text(encoding="utf-8")
        return content
    except UnicodeDecodeError:
        raise ValueError(f"'{path}' appears to be a binary file. Only text files are supported.")


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """
    Write or overwrite a text file in the sandbox.
    Creates parent directories if they don't exist.

    Args:
        path: Relative path from the sandbox root.
        content: Text content to write to the file.

    Returns:
        Confirmation with bytes written.
    """
    full = _safe_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")
    return f"✅ Written {len(content.encode('utf-8'))} bytes to '{path}'."


@mcp.tool()
def append_to_file(path: str, content: str) -> str:
    """
    Append text to an existing file (or create it if it doesn't exist).

    Args:
        path: Relative path from sandbox root.
        content: Text to append. A newline is added automatically.

    Returns:
        Confirmation message.
    """
    full = _safe_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    with full.open("a", encoding="utf-8") as f:
        f.write(content + "\n")
    return f"✅ Appended {len(content)} characters to '{path}'."


@mcp.tool()
def list_directory(path: str = "") -> str:
    """
    List the contents of a directory in the sandbox.

    Args:
        path: Relative path to the directory (empty string = sandbox root).

    Returns:
        JSON array of directory entries with name, type, and size.
    """
    full = _safe_path(path) if path else ALLOWED_ROOT

    if not full.exists():
        raise FileNotFoundError(f"Directory not found: '{path or '/'}'")
    if not full.is_dir():
        raise ValueError(f"'{path}' is a file, not a directory.")

    entries = []
    for item in sorted(full.iterdir()):
        info = {
            "name": item.name,
            "type": "directory" if item.is_dir() else "file",
            "size_bytes": item.stat().st_size if item.is_file() else None,
            "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
        }
        entries.append(info)

    return json.dumps({
        "directory": str(path or "/"),
        "items": entries,
        "total": len(entries),
    }, indent=2)


@mcp.tool()
def create_directory(path: str) -> str:
    """
    Create a directory (and any missing parent directories) in the sandbox.

    Args:
        path: Relative path of the directory to create.

    Returns:
        Confirmation message.
    """
    full = _safe_path(path)
    full.mkdir(parents=True, exist_ok=True)
    return f"✅ Directory '{path}' created."


@mcp.tool()
def delete_file(path: str) -> str:
    """
    Delete a file from the sandbox. (Directories must be empty to delete.)

    Args:
        path: Relative path of the file to delete.

    Returns:
        Confirmation message.
    """
    full = _safe_path(path)

    if not full.exists():
        raise FileNotFoundError(f"'{path}' not found.")

    if full.is_dir():
        # Only delete empty directories
        items = list(full.iterdir())
        if items:
            raise ValueError(f"'{path}' is a non-empty directory. Delete files inside it first.")
        full.rmdir()
        return f"🗑️ Directory '{path}' deleted."

    full.unlink()
    return f"🗑️ File '{path}' deleted."


@mcp.tool()
def file_info(path: str) -> str:
    """
    Get detailed information about a file or directory.

    Args:
        path: Relative path from sandbox root.

    Returns:
        JSON with size, timestamps, and permissions.
    """
    full = _safe_path(path)

    if not full.exists():
        raise FileNotFoundError(f"'{path}' not found.")

    s = full.stat()
    mode = oct(stat.S_IMODE(s.st_mode))

    info = {
        "path": str(path),
        "type": "directory" if full.is_dir() else "file",
        "size_bytes": s.st_size,
        "size_human": f"{s.st_size / 1024:.2f} KB" if s.st_size > 1024 else f"{s.st_size} B",
        "created": datetime.fromtimestamp(s.st_ctime).isoformat(),
        "modified": datetime.fromtimestamp(s.st_mtime).isoformat(),
        "permissions": mode,
        "readable": os.access(full, os.R_OK),
        "writable": os.access(full, os.W_OK),
    }
    return json.dumps(info, indent=2)


@mcp.tool()
def search_files(pattern: str, directory: str = "") -> str:
    """
    Search for files matching a glob pattern in the sandbox.

    Args:
        pattern: Glob pattern to match.
                 Examples: "*.txt", "**/*.py", "data_*.json"
        directory: Subdirectory to search in (empty = sandbox root).

    Returns:
        JSON array of matching file paths.
    """
    search_root = _safe_path(directory) if directory else ALLOWED_ROOT

    if not search_root.is_dir():
        raise ValueError(f"'{directory}' is not a directory.")

    matches = list(search_root.glob(pattern))

    # Filter to only files within sandbox
    safe_matches = [
        str(m.relative_to(ALLOWED_ROOT))
        for m in matches
        if str(m.resolve()).startswith(str(ALLOWED_ROOT))
    ]

    return json.dumps({
        "pattern": pattern,
        "search_root": str(directory or "/"),
        "matches": safe_matches,
        "count": len(safe_matches),
    }, indent=2)


# ── Resources ─────────────────────────────────────────────────────────────────

@mcp.resource("file://{path}")
def read_file_resource(path: str) -> str:
    """
    Read a file as an MCP resource.
    URI: file://{path}  (path is relative to sandbox)
    """
    try:
        return read_file(path)
    except (FileNotFoundError, ValueError) as e:
        return f"Error: {e}"


if __name__ == "__main__":
    print(f"📁 Filesystem MCP Server starting...")
    print(f"   Sandbox: {ALLOWED_ROOT}")
    print("   Tools: read_file, write_file, append_to_file, list_directory,")
    print("          create_directory, delete_file, file_info, search_files")
    print("   Resources: file://{{path}}")
    print("\n   ⚠️  All operations are sandboxed to the directory above.")
    mcp.run()
