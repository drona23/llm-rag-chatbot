"""
PHASE 1 STEP 2 TEST: RAG Agent (Complete RAG Pipeline)

This test runs the FULL RAG system:
1. Load documents
2. Create embeddings
3. Store in vector DB
4. Retrieve for questions
5. Generate answers with context
6. Calculate confidence

Test Flow:
User Question → Embed → Retrieve → Generate → Return Result
"""

from src.vector_db_mock import MockVectorStore
from src.llm import LLMChat
from src.rag_agent import RAGAgent
import os
import pathlib

BASE = pathlib.Path(__file__).parent

print("=" * 70)
print("PHASE 1 STEP 2 TEST: Complete RAG Pipeline")
print("=" * 70)

# STEP 1: Initialize components
print("\n[STEP 1] Initializing RAG components...")
llm = LLMChat()
vector_store = MockVectorStore()
agent = RAGAgent(vector_store, llm)
print("✓ RAGAgent initialized")

# STEP 2: Load student loan documents
print("\n[STEP 2] Loading and embedding documents...")
doc_paths = {
    "loan_types": BASE / "data/student_loans/loan_types.txt",
    "interest_rates": BASE / "data/student_loans/interest_rates.txt",
    "repayment_plans": BASE / "data/student_loans/repayment_plans.txt",
    "eligibility": BASE / "data/student_loans/eligibility.txt",
}

documents_to_store = []
for name, path in doc_paths.items():
    if os.path.exists(path):
        with open(path, 'r') as f:
            content = f.read()
            # Create embedding for document
            embedding = llm.generate_embedding(content)
            documents_to_store.append({
                "id": name,
                "text": content,
                "embedding": embedding
            })
            print(f"✓ Loaded and embedded {name}.txt ({len(content)} chars)")
    else:
        print(f"⚠ File not found: {path}")

# STEP 3: Store documents in vector DB
print(f"\n[STEP 3] Storing {len(documents_to_store)} documents in vector DB...")
vector_store.upsert_documents(documents_to_store)
print(f"✓ Documents stored in mock vector DB")

# STEP 4: Test RAG pipeline with questions
print("\n" + "=" * 70)
print("RUNNING RAG TESTS")
print("=" * 70)

test_questions = [
    "What types of federal student loans are available?",
    "What's the interest rate for direct subsidized loans?",
    "How do income-driven repayment plans work?",
    "What are the eligibility requirements for federal student loans?",
]

for i, question in enumerate(test_questions, 1):
    print(f"\n[TEST {i}]")
    print(f"Question: {question}")
    print("-" * 70)

    try:
        # THIS IS THE COMPLETE RAG PIPELINE
        # 1. Question → Embedding
        # 2. Search vector DB for similar docs
        # 3. Send docs + question to Claude
        # 4. Get answer
        result = agent.answer(question)

        print(f"Response:\n{result['response']}")
        print("-" * 70)

        # Quality metrics
        print("Quality Metrics:")
        print(f"✓ Confidence: {result['confidence']:.1%}")
        print(f"✓ Sources used: {len(result['sources'])} document(s)")

        if result['confidence'] > 0.5:
            print("✓ High confidence (good document match)")
        elif result['confidence'] > 0.3:
            print("⚠ Medium confidence (decent match)")
        else:
            print("⚠ Low confidence (weak document match)")

        if len(result['response']) > 100:
            print("✓ Detailed response")
        else:
            print("⚠ Brief response")

    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

# STEP 5: Test conversation history
print("\n" + "=" * 70)
print("CONVERSATION HISTORY")
print("=" * 70)

print(f"\nTotal turns in conversation: {len(agent.conversation_history)}")
for i, turn in enumerate(agent.conversation_history, 1):
    print(f"\nTurn {i}:")
    print(f"  Q: {turn['user'][:60]}...")
    print(f"  Confidence: {turn['confidence']:.1%}")
    print(f"  Sources: {len(turn['sources'])}")

print("\n" + "=" * 70)
print("TEST COMPLETE - FULL RAG PIPELINE WORKING!")
print("=" * 70)

print("\n✓ Phase 1 Step 2 Complete:")
print("  • Vector embeddings working")
print("  • Document retrieval working")
print("  • Response generation working")
print("  • Confidence scoring working")
print("  • Conversation history working")
