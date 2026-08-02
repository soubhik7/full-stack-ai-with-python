# Standalone copy of unique_code_snippets.ipynb's "Build a local MCP server with
# fastmcp" cell (section 6), saved as a real file so the notebook's stdio-client
# cell in the same section has an actual server.py-equivalent to launch as a
# subprocess when this notebook is run from this folder.
from fastmcp import FastMCP

mcp = FastMCP(name="Inventory")

@mcp.tool()
def get_inventory_levels() -> dict:
    """Returns current inventory for all products."""
    return {"Moisturizer": 6, "Shampoo": 8}

if __name__ == "__main__":
    mcp.run(show_banner=False)
