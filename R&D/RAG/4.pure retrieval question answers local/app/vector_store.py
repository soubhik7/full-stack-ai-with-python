"""
Create local embedding-based vector store using FAISS.
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def create_vector_store(chunks):

    # Local embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create FAISS index
    vector_db = FAISS.from_documents(chunks, embeddings)

    return vector_db
