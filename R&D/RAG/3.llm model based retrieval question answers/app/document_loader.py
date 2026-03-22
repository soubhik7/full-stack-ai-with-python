"""
Document Loader Module
----------------------

Responsibility:
Load text documents and split into smaller chunks.

Why splitting?
- Improves retrieval accuracy
- Prevents large context hallucination
- Makes vector search efficient
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_docs(path: str):
    """
    Load document and split into chunks.
    """

    # Load raw text file
    loader = TextLoader(path)
    documents = loader.load()

    # Split into smaller overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    return splitter.split_documents(documents)
