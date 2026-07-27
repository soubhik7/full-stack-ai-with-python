# =============================================================================
# LAB 24 (NEW — added to fill an AI-103 syllabus gap, not an original course
# file).
#
# ⚠️ PREVIEW / UNVERIFIED SDK ⚠️
# Azure AI Content Understanding IS a real Azure service (see
# `links.txt`'s Microsoft Learn links in this chapter), and this repo's own
# `07. Section Code/01_invoice_analysis.py` and
# `08. Section Code/03_cloudxeus_invoice_agent.py` both already use a
# `ContentUnderstandingClient` from `azure.ai.contentunderstanding` - BUT:
#   - that package is NOT installed in this repo's environment and NOT
#     listed in the root `requirements.txt` (confirmed: `import
#     azure.ai.contentunderstanding` raises ModuleNotFoundError here),
#   - and the two existing Section Code files call it with TWO DIFFERENT
#     method names for what should be the same operation
#     (`begin_analyze_binary` vs `begin_analyze`), which is a strong sign
#     this SDK's surface was still moving/preview when those files were
#     written.
# This lab reproduces the MORE RECENT-looking of the two shapes
# (`begin_analyze` + `AnalysisInput`, from `08. Section Code/`), but treat
# it as a best-effort snapshot, not a guaranteed-stable contract - `pip
# install` the package fresh and check its current docs/changelog before
# depending on this in real work.
#
# Needs a `sample_invoice.pdf` file in this folder (not included).
#
# Prerequisites: an Azure AI Content Understanding resource + .env with
# CONTENT_UNDERSTANDING_ENDPOINT and CONTENT_UNDERSTANDING_KEY.
# Requires: pip install azure-ai-contentunderstanding (preview package,
# name/version may have changed since this lab was written).
# =============================================================================

import os
import json
from pathlib import Path
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential


def main():
    os.system('cls' if os.name == 'nt' else 'clear')

    load_dotenv()
    endpoint = os.getenv("CONTENT_UNDERSTANDING_ENDPOINT")
    key = os.getenv("CONTENT_UNDERSTANDING_KEY")
    invoice_path = Path(__file__).parent / "sample_invoice.pdf"

    try:
        # ⚠️ Best-effort import — see the file header. Wrapped in try/except
        # so this lab fails informatively rather than with a bare traceback
        # if the package isn't installed (which is the expected state in
        # this repo, per the header note above).
        from azure.ai.contentunderstanding import ContentUnderstandingClient
        from azure.ai.contentunderstanding.models import AnalysisInput
    except ImportError as ex:
        print(f"⚠️  Could not import azure.ai.contentunderstanding: {ex}")
        print("This is expected in this repo (see this file's header comment) -")
        print("install the current package yourself to actually run this lab:")
        print("  pip install azure-ai-contentunderstanding")
        print("\nConceptually, this lab would:")
        print("  1. Connect a ContentUnderstandingClient to your resource.")
        print("  2. Submit an invoice PDF to the prebuilt-invoice analyzer.")
        print("  3. Poll the returned operation until it completes.")
        print("  4. Read structured fields (vendor, total, line items, ...) back")
        print("     out of the result, ready to hand to an agent as context.")
        return

    if not invoice_path.exists():
        print(f"No sample_invoice.pdf found at {invoice_path} - add one to try this lab.")
        return

    client = ContentUnderstandingClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )

    print(f"Analyzing {invoice_path.name} with the 'prebuilt-invoice' analyzer...")
    # begin_* / poller.result() is the same "long-running operation" shape
    # used by Document Intelligence (Lab 25) - submit the job, then block
    # until it finishes and grab the final result.
    poller = client.begin_analyze(
        analyzer_id="prebuilt-invoice",
        inputs=[AnalysisInput(url=str(invoice_path))],  # some SDK versions expect a blob URL rather than a local path - adjust if this errors
    )
    result = poller.result()

    # `.as_dict()` converts the typed result object into a plain
    # dict/JSON-serializable structure, handy for printing or handing to
    # an LLM as context.
    print(json.dumps(result.as_dict(), indent=2))


if __name__ == "__main__":
    main()
