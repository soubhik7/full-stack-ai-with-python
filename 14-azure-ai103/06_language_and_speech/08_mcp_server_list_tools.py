# pip3 install mcp
import asyncio
import httpx
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config

LANGUAGE_MCP_URL = (
    f"https://{config.language_resource_name}.cognitiveservices.azure.com"
    f"/language/mcp?api-version=2025-11-15-preview"
)


async def list_language_mcp_tools():
    headers = {
        "Ocp-Apim-Subscription-Key": config.language_key
    }

    timeout = httpx.Timeout(
        timeout=30.0,
        read=300.0
    )

    async with httpx.AsyncClient(headers=headers, timeout=timeout) as http_client:
        async with streamable_http_client(
            url=LANGUAGE_MCP_URL,
            http_client=http_client,
            terminate_on_close=True,
        ) as (read_stream, write_stream, _):

            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                result = await session.list_tools()

                print(f"Found {len(result.tools)} tool(s) on the Azure Language MCP server:\n")

                for tool in result.tools:
                    print(f"• {tool.name}")
                    print(f"  {tool.description}")
                    print()


if __name__ == "__main__":
    asyncio.run(list_language_mcp_tools())