# ============================================================
# RAG SYSTEM (Retrieval-Augmented Generation)
# Beginner-Friendly Complete Implementation
# ============================================================

"""
WHAT IS RAG?
------------

RAG = Retrieval + Generation

Instead of asking an LLM to answer from memory,
we:

1. STORE knowledge in documents
2. CONVERT documents into embeddings (vectors)
3. SEARCH most relevant documents
4. GIVE retrieved context to LLM
5. LLM generates better answer

This improves:
- Accuracy
- Context awareness
- Domain-specific answers

Pipeline:
---------
User Question
     ↓
Convert to Embedding
     ↓
Search Similar Documents
     ↓
Retrieve Top Results
     ↓
Send Context + Question to LLM
     ↓
Generated Answer
"""

# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import numpy as np
import faiss  # Facebook AI Similarity Search (vector search engine)

from sentence_transformers import SentenceTransformer
from transformers import pipeline


# ============================================================
# 2. SAMPLE DOCUMENT DATABASE
# ============================================================

"""
In real-world applications:
- Documents come from PDFs
- Databases
- Websites
- Knowledge bases

Here we create a small sample knowledge base.
"""

documents = [
    "Artificial Intelligence is the simulation of human intelligence in machines.",
    "Machine Learning is a subset of AI that allows systems to learn from data.",
    "Deep Learning uses neural networks with many layers.",
    "RAG stands for Retrieval Augmented Generation.",
    "Transformers are powerful models used in modern NLP systems."
]


# ============================================================
# 3. CREATE EMBEDDING MODEL
# ============================================================

"""
Embeddings convert text into numerical vectors.

Why?
Because computers understand numbers, not text.

We use sentence-transformers model:
- Lightweight
- Good quality
"""

print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


# ============================================================
# 4. CONVERT DOCUMENTS INTO VECTORS
# ============================================================

"""
Each document becomes a high-dimensional vector.

Similar meaning → similar vectors
"""

print("Creating document embeddings...")
doc_embeddings = embedding_model.encode(documents)

# Convert to float32 (required by FAISS)
doc_embeddings = np.array(doc_embeddings).astype("float32")


# ============================================================
# 5. CREATE VECTOR DATABASE USING FAISS
# ============================================================

"""
FAISS is used for fast similarity search.

IndexFlatL2 = Euclidean distance search
"""

dimension = doc_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

print("Vector database created successfully.")


# ============================================================
# 6. LOAD TEXT GENERATION MODEL
# ============================================================

"""
This is the "Generation" part of RAG.

We use a simple HuggingFace pipeline.
"""

print("Loading text generation model...")
generator = pipeline("text-generation", model="gpt2")


# ============================================================
# 7. RETRIEVAL FUNCTION
# ============================================================

def retrieve_documents(query, top_k=2):
    """
    Retrieve top_k most relevant documents.
    """

    # Convert query to embedding
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # Search in FAISS index
    distances, indices = index.search(query_embedding, top_k)

    retrieved_docs = [documents[i] for i in indices[0]]

    return retrieved_docs


# ============================================================
# 8. GENERATION FUNCTION
# ============================================================

def generate_answer(query, retrieved_docs):
    """
    Generate answer using:
    Retrieved context + User question
    """

    # Combine context
    context = "\n".join(retrieved_docs)

    # Create prompt
    prompt = f"""
    Use the following context to answer the question.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    # Generate response
    response = generator(prompt, max_length=200, num_return_sequences=1)

    return response[0]["generated_text"]


# ============================================================
# 9. COMPLETE RAG PIPELINE
# ============================================================

def rag_pipeline(query):
    """
    Complete RAG workflow:
    1. Retrieve
    2. Generate
    """

    print("\nRetrieving relevant documents...")
    retrieved_docs = retrieve_documents(query)

    print("\nRetrieved Documents:")
    for doc in retrieved_docs:
        print("-", doc)

    print("\nGenerating answer...")
    answer = generate_answer(query, retrieved_docs)

    return answer


# ============================================================
# 10. TEST THE SYSTEM
# ============================================================

if __name__ == "__main__":

    user_query = "What is RAG?"

    final_answer = rag_pipeline(user_query)

    print("\n==============================")
    print("FINAL ANSWER:")
    print("==============================")
    print(final_answer)
