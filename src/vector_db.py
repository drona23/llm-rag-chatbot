"""Vector database handler using Pinecone cloud storage."""
import os
import time
from pinecone import Pinecone, ServerlessSpec


class VectorStore:
    """Manages vector embeddings and similarity search via Pinecone."""

    def __init__(self, index_name: str = "student-loans"):
        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.index_name = index_name
        self._ensure_index_exists()
        self.index = self.pc.Index(index_name)

    def _ensure_index_exists(self):
        """Create the Pinecone index if it doesn't already exist."""
        existing = [i.name for i in self.pc.list_indexes()]
        if self.index_name not in existing:
            print(f"  Creating Pinecone index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=1024,          # voyage-3-large output size
                metric="cosine",         # best for text similarity
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # Wait for index to be ready (takes 10-60 seconds)
            print("  Waiting for index to be ready...", end="", flush=True)
            while not self.pc.describe_index(self.index_name).status["ready"]:
                print(".", end="", flush=True)
                time.sleep(2)
            print(" ready!")
        else:
            print(f"  Using existing Pinecone index '{self.index_name}'")

    def upsert_documents(self, docs: list[dict]):
        """
        Store documents with embeddings in Pinecone.

        Args:
            docs: List of {"id": str, "text": str, "embedding": list[float]}
        """
        # Pinecone v7 upsert format: list of dicts with id, values, metadata
        vectors = [
            {
                "id": doc["id"],
                "values": doc["embedding"],
                "metadata": {"text": doc["text"]}
            }
            for doc in docs
        ]
        self.index.upsert(vectors=vectors)
        print(f"  Uploaded {len(vectors)} documents to Pinecone")

    def retrieve(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """
        Find most similar documents using cosine similarity.

        Args:
            query_embedding: 1024-dim vector from Voyage AI
            top_k: How many documents to return

        Returns:
            List of {"text": str, "score": float}, sorted by relevance
        """
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True    # Needed to get the text back
        )
        return [
            {
                "text": match["metadata"]["text"],
                "score": match["score"]
            }
            for match in results["matches"]
        ]
