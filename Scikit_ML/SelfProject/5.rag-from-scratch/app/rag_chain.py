"""
RAG Chain Module
----------------

Responsibility:
1. Retrieve relevant chunks
2. Generate grounded answer

This is where hallucination prevention happens.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


SYSTEM_PROMPT = """
You are a helpful AI assistant.

Answer the question ONLY using the provided context.
If the answer is not in the context, say "I don't know."
"""


def create_rag_chain(vector_db):
    """
    Create a callable RAG function.
    """

    # LLM setup
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0  # Deterministic output
    )

    # Prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}")
        ]
    )

    def rag_answer(question: str):
        """
        Full RAG pipeline:
        1. Retrieve relevant chunks
        2. Inject into prompt
        3. Generate answer
        """

        # Retrieve top 3 relevant documents
        docs = vector_db.similarity_search(question, k=3)

        # Combine retrieved context
        context = "\n\n".join(doc.page_content for doc in docs)

        # Format prompt
        messages = prompt.format_messages(
            context=context,
            question=question
        )

        # Generate answer
        response = llm.invoke(messages)

        return response.content

    return rag_answer
