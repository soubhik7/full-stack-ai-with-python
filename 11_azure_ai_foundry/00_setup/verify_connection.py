####################################################################################################
# LAB 0 — SETUP: VERIFY CONNECTION
#
# WHY THIS MATTERS
#   Every other lab in this chapter (01_workflow through 07_knowledge_rag) talks to the same
#   live Azure AI Foundry project. If auth or the endpoint is wrong, every later lab fails in
#   the same way — so this script exists purely to catch that early, in isolation.
#
# HOW IT LINKS TO THE REST OF THE CHAPTER
#   AZURE_AI_PROJECT_ENDPOINT and AZURE_AI_MODEL_DEPLOYMENT (read from .env here) are the same
#   two environment variables every other lab in 11_azure_ai_foundry/ reads. Run this script
#   first; if it prints "connection ok", every other lab's setup is already correct.
#
# HOW IT WORKS
#   1. AIProjectClient(endpoint, credential) — the top-level handle to your Foundry project.
#   2. DefaultAzureCredential() — reuses your `az login` session (Entra ID), no API key in .env.
#   3. project_client.get_openai_client() — hands back a standard OpenAI SDK client whose
#      base_url/auth are already pointed at your project. This is the "Playground" component
#      from the study notes (§7.2/Component 2) made programmatic: one stateless chat call.
####################################################################################################
import os

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

endpoint = os.environ["AZURE_AI_PROJECT_ENDPOINT"]
deployment = os.environ["AZURE_AI_MODEL_DEPLOYMENT"]

project_client = AIProjectClient(endpoint=endpoint, credential=DefaultAzureCredential())
openai_client = project_client.get_openai_client()

response = openai_client.chat.completions.create(
    model=deployment,
    messages=[{"role": "user", "content": "Reply with exactly: connection ok"}],
)

print(f"Project endpoint: {endpoint}")
print(f"Model deployment: {deployment}")
print(f"Model response:   {response.choices[0].message.content}")
