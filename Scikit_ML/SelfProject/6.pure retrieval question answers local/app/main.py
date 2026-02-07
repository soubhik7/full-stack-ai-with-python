"""
Main Application - No LLM RAG
Pure semantic search system.
"""

from app.document_loader import load_and_split_docs
from app.vector_store import create_vector_store
from app.retriever import retrieve_answer


def main():

    print("📄 Loading documents...")
    chunks = load_and_split_docs("data/company_policy.txt")

    print("🧠 Creating vector store...")
    vector_db = create_vector_store(chunks)

    print("\n✅ Retrieval system ready! Type 'exit' to quit.\n")

    while True:
        question = input("❓ Question: ")

        if question.lower() == "exit":
            break

        answer = retrieve_answer(vector_db, question)

        print(f"\n📘 Answer: {answer}\n")


if __name__ == "__main__":
    main()
