"""Vector database handler for RAG retrieval."""
from pinecone import Pinecone


class VectorStore:
    """Manages vector embeddings and similarity search."""

    def __init__(self, api_key: str, index_name: str = "chatbot"):
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)

    def upsert_documents(self, docs: list[dict]):
        """
        Store documents with embeddings.

        Args:
            docs: List of {"id": str, "text": str, "embedding": list[float]}
        """
        vectors = [
            (doc["id"], doc["embedding"], {"text": doc["text"]})
            for doc in docs
        ]
        self.index.upsert(vectors=vectors)

    def retrieve(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """
        Find most similar documents.

        Args:
            query_embedding: Vector representation of query
            top_k: Number of results to return

        Returns:
            List of {"text": str, "score": float}
        """
        results = self.index.query(vector=query_embedding, top_k=top_k)
        return [
            {
                "text": match["metadata"]["text"],
                "score": match["score"]
            }
            for match in results["matches"]
        ]
