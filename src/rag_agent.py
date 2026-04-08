"""RAG agent combining vector retrieval + LLM generation."""
from src.llm import LLMChat
# Phase 1: Use mock vector store (no Pinecone needed)
# Phase 2: Switch to real Pinecone with: from src.vector_db import VectorStore
from src.vector_db_mock import MockVectorStore as VectorStore


class RAGAgent:
    """End-to-end RAG pipeline."""

    def __init__(self, vector_store: VectorStore, llm: LLMChat):
        self.vector_store = vector_store
        self.llm = llm
        self.conversation_history = []

    def answer(self, user_query: str) -> dict:
        """
        Process user query with RAG.

        Args:
            user_query: User's question

        Returns:
            {
                "response": str,
                "sources": list[str],
                "confidence": float
            }
        """
        # Step 1: Get embedding for user query
        query_embedding = self.llm.generate_embedding(user_query)

        # Step 2: Retrieve relevant documents
        # top_k=8 retrieves all docs with mock embeddings (no semantic order)
        # With real Pinecone embeddings, reduce to 3-5 for precision
        retrieved = self.vector_store.retrieve(query_embedding, top_k=8)
        context_docs = [doc["text"] for doc in retrieved]

        # Step 3: Generate response with context
        # The generate_response() function handles:
        # - Structuring prompt with documents
        # - Telling Claude to use only these documents
        # - Formatting for clarity
        response = self.llm.generate_response(
            user_message=user_query,
            context_docs=context_docs
        )

        # Step 4: Calculate confidence score
        # Use AVERAGE similarity score across retrieved docs
        # Why average? More stable than min (which gets worse with more docs)
        # and more representative than max (which ignores weaker docs)
        if retrieved:
            scores = [doc["score"] for doc in retrieved]
            confidence = sum(scores) / len(scores)
        else:
            confidence = 0.0

        # Step 5: Track conversation for multi-turn support
        # Stores all turns so follow-up questions have context
        self.conversation_history.append({
            "user": user_query,
            "assistant": response,
            "sources": context_docs,
            "confidence": confidence
        })

        # Step 6: Return complete result to user
        # Include retrieval_scores so evaluator can use real scores
        return {
            "response": response,
            "sources": context_docs,
            "confidence": confidence,
            "retrieval_scores": [doc["score"] for doc in retrieved]
        }

    def reset_conversation(self):
        """Clear chat history for new conversation."""
        self.conversation_history = []
