"""
Main Application Entry Point
----------------------------

Flow:
1. Load & split documents
2. Create vector store
3. Initialize RAG chain
4. Start interactive Q&A loop
"""

import os
from dotenv import load_dotenv

from app.document_loader import load_and_split_docs
from app.vector_store import create_vector_store
from app.rag_chain import create_rag_chain


def main():
    # Load environment variables
    load_dotenv()

    print("📄 Loading documents...")
    chunks = load_and_split_docs("data/company_policy.txt")

    print("🧠 Creating vector store...")
    vector_db = create_vector_store(chunks)

    print("🤖 Initializing RAG...")
    rag = create_rag_chain(vector_db)

    print("\n✅ RAG is ready! Type 'exit' to quit.\n")

    while True:
        question = input("❓ Question: ")

        if question.lower() == "exit":
            break

        answer = rag(question)

        print(f"\n🤖 Answer: {answer}\n")


if __name__ == "__main__":
    main()
