# pip3 install langchain langchain-openai
import sys
from pathlib import Path

from langchain_openai import ChatOpenAI

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config

endpoint = config.openai_endpoint
deployment_name = config.openai_deployment
api_key = config.openai_api_key

client=ChatOpenAI(
    base_url=endpoint,
    api_key=api_key,
    model=deployment_name
)

response=client.invoke("What are the three main benefits of using managed AI endpoints in the cloud?")
print(response.content)