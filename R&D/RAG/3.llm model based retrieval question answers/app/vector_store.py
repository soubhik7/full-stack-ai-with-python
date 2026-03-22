"""
Vector Store Module
-------------------

Responsibility:
Convert text chunks into embeddings
Store them inside FAISS (vector database)

What are embeddings?
- Numerical representation of meaning
- Allows semantic search instead of keyword search
"""

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def create_vector_store(chunks):
    """
    Create FAISS vector store from document chunks.
    """

    # Create embedding model
    embeddings = OpenAIEmbeddings()

    # Convert documents to vector store
    vector_db = FAISS.from_documents(chunks, embeddings)

    return vector_db
