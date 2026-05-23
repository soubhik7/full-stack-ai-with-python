"""
Load and split documents into chunks.

Small chunks improve semantic search accuracy.
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_docs(path: str):

    loader = TextLoader(path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30
    )

    return splitter.split_documents(documents)
