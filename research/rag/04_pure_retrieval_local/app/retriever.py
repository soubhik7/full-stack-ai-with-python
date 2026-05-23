"""
Pure Retrieval System (No LLM)

Returns the most relevant chunk directly.
"""

def retrieve_answer(vector_db, question: str, threshold=0.3):
    """
    Retrieve best matching document.

    threshold:
    - If similarity score too low,
      we return "I don't know"
    """

    # Search top 1 most similar chunk
    results = vector_db.similarity_search_with_score(question, k=1)

    if not results:
        return "I don't know."

    document, score = results[0]

    # FAISS score is distance (lower = better)
    # Threshold tuned based on testing:
    # - Good matches: 0.92-1.67
    # - Bad matches: 1.85+
    if score > 1.75:
        return "I don't know."

    return document.page_content
