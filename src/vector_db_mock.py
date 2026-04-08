"""
Mock Vector Database for Phase 1 Testing

This is a simple in-memory vector store.
In Phase 2, we'll replace this with real Pinecone.

Why mock?
- Understand RAG without Pinecone complexity
- Test locally without API keys
- Fast iteration during learning
"""


class MockVectorStore:
    """Simple in-memory vector database for testing."""

    def __init__(self):
        """Initialize empty vector store."""
        self.documents = []  # Stores: {"id": str, "text": str, "embedding": list}

    def upsert_documents(self, docs: list[dict]):
        """
        Store documents with embeddings.

        Args:
            docs: List of {"id": str, "text": str, "embedding": list[float]}
        """
        for doc in docs:
            # Check if document exists, update it; otherwise add new
            existing = next(
                (d for d in self.documents if d["id"] == doc["id"]),
                None
            )
            if existing:
                existing["text"] = doc["text"]
                existing["embedding"] = doc["embedding"]
            else:
                self.documents.append(doc)

    def retrieve(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """
        Find most similar documents using cosine similarity.

        Args:
            query_embedding: Vector representation of query
            top_k: Number of results to return

        Returns:
            List of {"text": str, "score": float} sorted by similarity
        """
        if not self.documents:
            return []

        # Calculate cosine similarity for each document
        similarities = []
        for doc in self.documents:
            similarity = self._cosine_similarity(
                query_embedding,
                doc["embedding"]
            )
            similarities.append({
                "text": doc["text"],
                "score": similarity,
                "id": doc["id"]
            })

        # Sort by score (highest first)
        similarities.sort(key=lambda x: x["score"], reverse=True)

        # Return top_k
        return similarities[:top_k]

    def _cosine_similarity(
        self,
        vec1: list[float],
        vec2: list[float]
    ) -> float:
        """
        Calculate cosine similarity between two vectors.

        Formula: (A·B) / (||A|| × ||B||)

        Where:
        - A·B is dot product
        - ||A|| is magnitude of A
        - ||B|| is magnitude of B

        Result ranges 0-1 (higher = more similar)
        """
        # Ensure vectors are same length
        if len(vec1) != len(vec2):
            # Pad shorter vector with zeros
            max_len = max(len(vec1), len(vec2))
            vec1 = vec1 + [0] * (max_len - len(vec1))
            vec2 = vec2 + [0] * (max_len - len(vec2))

        # Calculate dot product: sum(a*b for each element)
        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        # Calculate magnitudes: sqrt(sum(x^2 for each element))
        magnitude1 = sum(x * x for x in vec1) ** 0.5
        magnitude2 = sum(x * x for x in vec2) ** 0.5

        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        # Raw cosine similarity (-1 to +1)
        raw = dot_product / (magnitude1 * magnitude2)

        # Normalize to 0-1 range so confidence is always positive
        # Formula: (raw + 1) / 2  maps  -1→0,  0→0.5,  +1→1
        return (raw + 1) / 2
