import sys
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects.models import PromptAgentDefinition, FunctionTool

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config

PROJECT_ENDPOINT = config.project_endpoint
AGENT_NAME = config.agent_name("IT-HelpDesk-Agent")
DEPLOYMENT_NAME = config.model_deployment_or("gpt-4.1-mini")

client=AIProjectClient(
    endpoint=PROJECT_ENDPOINT,
    credential=DefaultAzureCredential()
)

tools = [
    FunctionTool(
        name="get_password_reset_steps",
        description="Get the company password reset steps.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        strict=True,
    ),
    FunctionTool(
        name="get_vpn_troubleshooting_steps",
        description="Get troubleshooting steps for VPN connection issues.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        strict=True,
    ),
    FunctionTool(
        name="get_software_install_guide",
        description="Get installation instructions for a supported software package.",
        parameters={
            "type": "object",
            "properties": {
                "software_name": {
                    "type": "string",
                    "description": "The software name, for example Slack, Zoom, or VS Code."
                }
            },
            "required": ["software_name"],
            "additionalProperties": False,
        },
        strict=True,
    ),
]

agent=client.agents.create_version(
    agent_name=AGENT_NAME,
    definition=PromptAgentDefinition(
        model=DEPLOYMENT_NAME,
        instructions=(
            "You are an IT support assistant for a company. "
            "Help users with password resets, VPN issues, and software installation. "
            "Give clear, step-by-step answers. "
            "If the question is outside IT support topics, politely say so."
        ),
        tools=tools
    )
)

print(f"Agent created:")
print(f"  ID      : {agent.id}")
print(f"  Name    : {agent.name}")
print(f"  Version : {agent.version}")