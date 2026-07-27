# =============================================================================
# LAB 8 (part B - local MCP server): the SIMPLEST possible MCP server, built
# with the `fastmcp` framework. It exposes two "tools" that any MCP client
# (including client.py in this same folder) can discover and call.
#
# This file is NOT meant to be run directly by you - client.py starts it
# automatically as a subprocess (see StdioServerParameters in client.py).
# It communicates with its client over "stdio" (standard input/output pipes)
# rather than a network port, which is the simplest MCP transport.
#
# Requires: pip install fastmcp (not in the root requirements.txt).
# =============================================================================

# Add references
# Add references
from fastmcp import FastMCP   # A lightweight framework for building MCP servers with plain Python decorators

# Create an MCP server
# Create an MCP server
# "Inventory" is just a human-readable name for this server (shown to
# whatever client connects to it).
mcp = FastMCP(name="Inventory")

# Add an inventory check mcp tool
@mcp.tool()
# The @mcp.tool() decorator is what turns this plain function into a tool
# that MCP clients can discover (via session.list_tools()) and call (via
# session.call_tool("get_inventory_levels", {})). The docstring becomes the
# tool's description, and the return type hint (-> dict) becomes part of its
# schema.
def get_inventory_levels() -> dict:
    """Returns current inventory for all products."""
    # Hardcoded sample data - a real server would query a database here.
    return {
        "Moisturizer": 6,
        "Shampoo": 8,
        "Body Spray": 28,
        "Hair Gel": 5,
        "Lip Balm": 12,
        "Skin Serum": 9,
        "Cleanser": 30,
        "Conditioner": 3,
        "Setting Powder": 17,
        "Dry Shampoo": 45
    }

# Add a weekly sales mcp tool

@mcp.tool()
def get_weekly_sales() -> dict:
    """Returns number of units sold last week."""
    # Hardcoded sample data, paired with get_inventory_levels() above -
    # together they let the agent in client.py reason about restock vs.
    # clearance decisions (see the agent's instructions in client.py).
    return {
        "Moisturizer": 22,
        "Shampoo": 18,
        "Body Spray": 3,
        "Hair Gel": 2,
        "Lip Balm": 14,
        "Skin Serum": 19,
        "Cleanser": 4,
        "Conditioner": 1,
        "Setting Powder": 13,
        "Dry Shampoo": 17
    }



# Run the MCP server
# Run the MCP server
# This starts the server's event loop, listening for MCP requests on
# stdin/stdout. `show_banner=False` just suppresses fastmcp's startup ASCII
# art so it doesn't clutter the client's console output. This call blocks
# forever (until the process is killed) - which is fine here since
# client.py manages this file's entire lifecycle as a subprocess.
mcp.run(show_banner=False)
