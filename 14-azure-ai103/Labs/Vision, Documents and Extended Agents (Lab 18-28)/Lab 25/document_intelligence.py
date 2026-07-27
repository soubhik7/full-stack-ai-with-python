# =============================================================================
# LAB 25 (NEW — added to fill an AI-103 syllabus gap, not an original course
# file): AZURE AI DOCUMENT INTELLIGENCE — Domain 5 of the AI-103 exam
# blueprint ("Implement information extraction solutions"). Unlike Content
# Understanding (Lab 24), this is a long-established, stable, real Azure SDK
# package - adapted directly from the proven, working
# `08. Section Code/01_document_intelligence.py`.
#
# Prerequisites: an Azure AI Document Intelligence resource + .env with
# DOCUMENT_INTELLIGENCE_ENDPOINT and DOCUMENT_INTELLIGENCE_KEY (same
# variable names this chapter's top-level README already documents).
# Requires: pip install azure-ai-documentintelligence (not in root
# requirements.txt).
# =============================================================================

import os                      # Env vars, clear-screen command
from dotenv import load_dotenv  # Loads .env file variables into the environment

from azure.ai.documentintelligence import DocumentIntelligenceClient        # The dedicated Document Intelligence SDK client
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest      # Wraps the document source (a URL, here) for the analyze call
from azure.core.credentials import AzureKeyCredential                        # Document Intelligence authenticates with a subscription key


def main():
    # Clear the console
    os.system('cls' if os.name == 'nt' else 'clear')

    try:
        load_dotenv()
        endpoint = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
        key = os.getenv("DOCUMENT_INTELLIGENCE_KEY")
        # A publicly-readable sample document URL, so this lab runs with
        # zero setup - swap in your own document's URL to analyze something
        # real. (Document Intelligence needs a fetchable URL or raw bytes,
        # not a bare local file path.)
        document_url = os.getenv(
            "SAMPLE_DOCUMENT_URL",
            "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-layout.pdf",
        )

        client = DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key),
            api_version="2024-11-30",   # Document Intelligence requires pinning an explicit API version
        )

        print(f"Analyzing document: {document_url}")
        print("Using the 'prebuilt-layout' model (extracts text, tables, and structure "
              "without needing a custom-trained model)...\n")

        # begin_analyze_document() starts a LONG-RUNNING OPERATION - document
        # analysis can take several seconds for larger files, so the SDK
        # returns a "poller" you call .result() on, which blocks until the
        # job finishes (rather than making you write your own polling loop).
        poller = client.begin_analyze_document(
            model_id="prebuilt-layout",
            body=AnalyzeDocumentRequest(url_source=document_url),
            output_content_format="markdown",   # asks for the extracted text back as Markdown (preserves headings/tables readably)
        )
        result = poller.result()

        print("=" * 60)
        print("EXTRACTED CONTENT (markdown)")
        print("=" * 60)
        print(result.content)

        # Beyond plain text, `prebuilt-layout` also detects TABLES as
        # structured data (rows/columns/cell text) - useful when you need
        # to reconstruct a table programmatically rather than just reading
        # its markdown rendering.
        if result.tables:
            print(f"\nDetected {len(result.tables)} table(s):")
            for i, table in enumerate(result.tables):
                print(f"  Table {i + 1}: {table.row_count} rows x {table.column_count} columns")

    except Exception as ex:
        # Catch-all: print any error instead of crashing
        print(ex)


if __name__ == "__main__":
    main()
