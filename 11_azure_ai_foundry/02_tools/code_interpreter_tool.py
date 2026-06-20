####################################################################################################
# LAB 2b — CODE INTERPRETER TOOL (built-in, sandboxed Python execution)
#
# WHY THIS MATTERS
#   LLMs are unreliable at exact arithmetic — they predict plausible-looking numbers, not
#   computed ones. The code interpreter tool is Azure's built-in fix: the agent writes real
#   Python, Azure runs it in a sandbox, and the agent reports the actual computed result.
#
# HOW IT LINKS TO THE REST OF THE CHAPTER
#   Contrast with 02_tools/function_tool.py: that tool is custom code *you* wrote ahead of time;
#   this tool is generic — the agent writes the code itself, on the fly, per question.
#   03_toolbox/ combines this exact tool with custom function tools in a single ToolSet.
#
# HOW IT WORKS
#   CodeInterpreterTool() needs no setup (no functions to register) — just pass its
#   `.definitions` to create_agent(tools=...) and the Agent Service handles sandboxing and
#   execution entirely server-side. Printing the full message list (not just the last reply)
#   shows both the user's question and the agent's computed answer in order.
####################################################################################################
import os

from azure.ai.agents import AgentsClient
from azure.ai.agents.models import CodeInterpreterTool, ListSortOrder
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = os.environ["AZURE_AI_PROJECT_ENDPOINT"]
DEPLOYMENT = os.environ["AZURE_AI_MODEL_DEPLOYMENT"]


def main():
    agents_client = AgentsClient(endpoint=ENDPOINT, credential=DefaultAzureCredential())
    code_interpreter = CodeInterpreterTool()

    agent = agents_client.create_agent(
        model=DEPLOYMENT,
        name="code-interpreter-demo",
        instructions="You are a Crystal Hotels billing assistant. Write and run Python to compute exact totals.",
        tools=code_interpreter.definitions,
    )
    thread = agents_client.threads.create()
    agents_client.messages.create(
        thread_id=thread.id,
        role="user",
        content="A Deluxe room costs $189/night for 5 nights, plus 12% tax. Compute the total with Python.",
    )
    agents_client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

    print("Conversation:")
    for message in agents_client.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING):
        if message.text_messages:
            print(f"  [{message.role}] {message.text_messages[-1].text.value}")

    agents_client.delete_agent(agent.id)
    agents_client.threads.delete(thread.id)


if __name__ == "__main__":
    main()
