import sys
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config

search_client = SearchClient(
    endpoint=config.search_endpoint,
    index_name=config.search_index_name,
    credential=DefaultAzureCredential(),
)

results = search_client.search(
    search_text="refund",
    select=["chunk", "title"],
)

for result in results:
    print(f"Score:  {result['@search.score']:.4f}")
    print(f"Source: {result['title']}")
    print(f"Text:   {result['chunk']}")
    print("---")