# =============================================================================
# LAB 26 (NEW — added to fill an AI-103 syllabus gap, not an original course
# file): AZURE AI SEARCH — "bring your own retrieval" RAG, Domain 5's
# "knowledge mining" topic. Adapted from this repo's own real, working
# `02_foundry_agent_service/08_ai_search.py` and `10_customer_rag_client.py`.
#
# Worth knowing going in: `AzureAISearchTool` (a Foundry-native "search as a
# tool" class some docs describe) is never actually called in ANY .py file
# in this repo - the real, working pattern here queries `SearchClient`
# directly in your own code and manually stuffs the results into the
# prompt, which is exactly what this lab does. (If you want Search wired in
# as a native Foundry AGENT tool instead of client-side code, the only
# proven-working example of "search via a tool" in this repo is Lab 8's
# MCPTool pointed at a knowledge base's MCP endpoint - a different, MCP-
# based mechanism, not AzureAISearchTool.)
#
# Prerequisites: an Azure AI Search resource with an index ALREADY created
# and populated (index creation/ingestion is out of scope for this lab -
# see the Azure AI Search quickstart docs, or chapter 07_rag/ elsewhere in
# this repo for a from-scratch RAG pipeline) + `az login` + .env with
# AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX_NAME, PROJECT_ENDPOINT, and
# MODEL_DEPLOYMENT_NAME.
# Requires: pip install azure-search-documents (not in root requirements.txt).
# =============================================================================

import os                      # Env vars, clear-screen command
import sys
from pathlib import Path

from azure.identity import DefaultAzureCredential           # Authenticates using your `az login` session — same credential for both Search and Foundry
from azure.search.documents import SearchClient               # The Azure AI Search query client
from azure.search.documents.models import VectorizableTextQuery  # Lets Search embed your query text for you server-side ("integrated vectorization")
from azure.ai.projects import AIProjectClient                 # Talks to your Foundry project, for the final answer-generation call

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config


def retrieve_context(search_client, question, top=3):
    """Hybrid search: a keyword ('search_text') query AND a vector query,
    merged server-side via Reciprocal Rank Fusion (RRF) - this tends to beat
    either keyword-only or vector-only search alone, since keyword search
    catches exact terms (product codes, names) that embeddings can blur,
    while vector search catches semantically-related phrasing that keyword
    search misses entirely.
    """
    results = search_client.search(
        search_text=question,                                  # keyword side of the hybrid query
        vector_queries=[
            # k_nearest_neighbors = how many candidates the VECTOR side
            # contributes before merging with the keyword side's results.
            VectorizableTextQuery(text=question, k_nearest_neighbors=top, fields="text_vector")
        ],
        select=["chunk", "title"],   # only pull back the fields we actually need to build a prompt
        top=top,                     # final number of results returned after merging
    )

    # Build a plain-text "sources" block to paste into the prompt - this IS
    # the "bring your own retrieval" style: nothing here is a Foundry-native
    # tool, it's just string building in your own code.
    sources = []
    for result in results:
        title = result.get("title", "unknown source")
        chunk = result.get("chunk", "")
        score = result["@search.score"]
        print(f"  [score {score:.4f}] {title}")
        sources.append(f"Source ({title}):\n{chunk}")
    return "\n\n".join(sources)


def main():
    # Clear the console
    os.system('cls' if os.name == 'nt' else 'clear')

    try:
        search_endpoint = config.search_endpoint
        index_name = config.search_index_name
        project_endpoint = config.project_endpoint
        model_deployment = config.model_deployment

        credential = DefaultAzureCredential()

        search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=index_name,
            credential=credential,
        )

        project_client = AIProjectClient(endpoint=project_endpoint, credential=credential)
        openai_client = project_client.get_openai_client()

        question = input("Ask a question to answer using your Azure AI Search index: ")

        print("\nRetrieving relevant chunks from the index...")
        sources_text = retrieve_context(search_client, question)

        if not sources_text:
            print("No matching content found in the index for that question.")
            return

        # Manually stuff the retrieved chunks into the prompt, along with a
        # grounding instruction - the standard hallucination-mitigation
        # pattern: "answer only from these sources, say you don't know
        # otherwise."
        prompt = f"""Answer the question using ONLY the sources below. If the sources don't contain
the answer, say you don't know - do not make anything up.

{sources_text}

Question: {question}
Answer:"""

        response = openai_client.responses.create(
            model=model_deployment,
            input=prompt,
        )
        print(f"\nAnswer:\n{response.output_text}")

    except Exception as ex:
        # Catch-all: print any error instead of crashing
        print(ex)


if __name__ == "__main__":
    main()
