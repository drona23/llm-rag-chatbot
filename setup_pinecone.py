"""
PHASE 3 SETUP: Upload documents to Pinecone with real Voyage AI embeddings.

Run this ONCE before using the RAG chatbot with real vector search.
After running this, test_e2e.py will use real semantic retrieval.

Usage:
    python3 setup_pinecone.py
"""

import os
import pathlib
import time
from dotenv import load_dotenv

load_dotenv(override=True)  # override empty shell vars with .env values

from src.llm import LLMChat
from src.vector_db import VectorStore

BASE = pathlib.Path(__file__).parent

print("=" * 60)
print("  PHASE 3 SETUP: Pinecone + Voyage AI")
print("=" * 60)

# ─── STEP 1: Connect to Pinecone (creates index if needed) ────────────────────
print("\n[1/3] Connecting to Pinecone...")
store = VectorStore(index_name="student-loans")

# ─── STEP 2: Embed all documents with Voyage AI ───────────────────────────────
print("\n[2/3] Embedding documents with Voyage AI (voyage-3-large)...")
llm = LLMChat()

doc_paths = {
    "loan_types":              BASE / "data/student_loans/loan_types.txt",
    "interest_rates":          BASE / "data/student_loans/interest_rates.txt",
    "repayment_plans":         BASE / "data/student_loans/repayment_plans.txt",
    "eligibility":             BASE / "data/student_loans/eligibility.txt",
    "faq":                     BASE / "data/student_loans/faq.txt",
    "bad_credit_and_private":  BASE / "data/student_loans/bad_credit_and_private.txt",
    "pslf_and_forgiveness":    BASE / "data/student_loans/pslf_and_forgiveness.txt",
    "income_driven_repayment": BASE / "data/student_loans/income_driven_repayment.txt",
}

# Load all texts first
names, texts = [], []
for name, path in doc_paths.items():
    if not path.exists():
        print(f"  MISSING: {path}")
        continue
    names.append(name)
    texts.append(path.read_text())
    print(f"  Loaded {name} ({len(texts[-1]):,} chars)")

# Embed ALL documents in a single Voyage AI API call (1 request = no rate limit issue)
print(f"\n  Calling Voyage AI embed API with {len(texts)} documents at once...")
result = llm.voyage.embed(texts, model="voyage-3-large")
print(f"  Done. Each embedding: {len(result.embeddings[0])} dims")

docs_to_upload = [
    {"id": names[i], "text": texts[i], "embedding": result.embeddings[i]}
    for i in range(len(names))
]

# ─── STEP 3: Upload to Pinecone ───────────────────────────────────────────────
if not docs_to_upload:
    print("\n[3/3] All documents already in Pinecone, nothing to upload.")
else:
    print(f"\n[3/3] Uploading {len(docs_to_upload)} documents to Pinecone...")
    store.upsert_documents(docs_to_upload)

# Wait for Pinecone to index the vectors
print("  Waiting for index to sync...", end="", flush=True)
time.sleep(5)
print(" done")

# Verify upload
stats = store.index.describe_index_stats()
print(f"\n  Vectors in index: {stats['total_vector_count']}")

print("\n" + "=" * 60)
print("  SETUP COMPLETE")
print("  Now run: python3 test_e2e.py")
print("=" * 60)
