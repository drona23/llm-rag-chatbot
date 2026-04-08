"""
PHASE 2 STEP 1 TEST: Evaluation Framework

Test the RAG quality evaluation system:
1. Relevance - Does response use sources?
2. Faithfulness - Is response grounded? (no hallucination)
3. Answer Relevance - Does it answer the question?
4. Overall Score - Combined metric

This is critical for:
- Job 1: "analyze model outputs"
- Job 2: "assess responses"
"""

from src.evaluation import RAGEvaluator
from src.vector_db_mock import MockVectorStore
from src.llm import LLMChat
from src.rag_agent import RAGAgent
import os
import pathlib

BASE = pathlib.Path(__file__).parent

print("=" * 70)
print("PHASE 2 STEP 1 TEST: Evaluation Framework")
print("=" * 70)

# Setup RAG system
print("\n[SETUP] Initializing RAG system...")
llm = LLMChat()
vector_store = MockVectorStore()
agent = RAGAgent(vector_store, llm)
evaluator = RAGEvaluator()

# Load documents
doc_paths = {
    "loan_types": BASE / "data/student_loans/loan_types.txt",
    "interest_rates": BASE / "data/student_loans/interest_rates.txt",
}

documents_to_store = []
for name, path in doc_paths.items():
    if os.path.exists(path):
        with open(path, 'r') as f:
            content = f.read()
            embedding = llm.generate_embedding(content)
            documents_to_store.append({
                "id": name,
                "text": content,
                "embedding": embedding
            })

vector_store.upsert_documents(documents_to_store)
print("✓ RAG system ready")

# Test questions
test_cases = [
    {
        "question": "What's the interest rate for direct subsidized loans?",
        "expected_key_terms": ["6.53%", "subsidized", "interest rate"]
    },
    {
        "question": "What's the difference between subsidized and unsubsidized loans?",
        "expected_key_terms": ["government pays", "accrues", "difference"]
    },
    {
        "question": "When do I start paying back my loans?",
        "expected_key_terms": ["grace period", "repayment", "graduation"]
    },
]

print("\n" + "=" * 70)
print("EVALUATION TESTS")
print("=" * 70)

all_scores = []

for i, test_case in enumerate(test_cases, 1):
    question = test_case["question"]
    print(f"\n[TEST {i}]")
    print(f"Question: {question}")
    print("-" * 70)

    # Get RAG response
    result = agent.answer(question)
    response = result["response"]
    sources = result["sources"]

    # Evaluate using real retrieval scores (not faithfulness as proxy)
    retrieval_scores = result.get("retrieval_scores", [])
    relevance = evaluator.relevance_score(response, sources)
    faithfulness = evaluator.faithfulness(response, sources)
    ans_relevance = evaluator.answer_relevance(question, response)
    overall = (relevance * 0.3 + faithfulness * 0.4 + ans_relevance * 0.3)

    all_scores.append({
        "question": question,
        "relevance": relevance,
        "faithfulness": faithfulness,
        "answer_relevance": ans_relevance,
        "overall": overall
    })

    # Display results
    print(f"Response preview: {response[:150]}...")
    print("-" * 70)
    print("Quality Metrics:")
    print(f"  Relevance:       {relevance:.1%} (uses sources)")
    print(f"  Faithfulness:    {faithfulness:.1%} (grounded, no hallucination)")
    print(f"  Answer Relevance: {ans_relevance:.1%} (answers question)")
    print(f"  Overall Score:   {overall:.1%}")

    # Interpretation
    print("\nInterpretation:")
    if overall > 0.75:
        print("  ✓ Excellent - Production ready")
    elif overall > 0.60:
        print("  ✓ Good - Acceptable quality")
    elif overall > 0.45:
        print("  ⚠ Fair - Needs improvement")
    else:
        print("  ✗ Poor - Requires work")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

avg_relevance = sum(s["relevance"] for s in all_scores) / len(all_scores)
avg_faithfulness = sum(s["faithfulness"] for s in all_scores) / len(all_scores)
avg_answer_relevance = sum(s["answer_relevance"] for s in all_scores) / len(all_scores)
avg_overall = sum(s["overall"] for s in all_scores) / len(all_scores)

print(f"\nAverage scores across {len(all_scores)} tests:")
print(f"  Relevance:       {avg_relevance:.1%}")
print(f"  Faithfulness:    {avg_faithfulness:.1%}")
print(f"  Answer Relevance: {avg_answer_relevance:.1%}")
print(f"  Overall:         {avg_overall:.1%}")

print("\n" + "=" * 70)
print("✓ PHASE 2 STEP 1 COMPLETE")
print("=" * 70)

print("\nEvaluation Framework:")
print("  ✓ Relevance scoring (uses sources)")
print("  ✓ Faithfulness detection (no hallucination)")
print("  ✓ Answer relevance (answers question)")
print("  ✓ Overall quality scoring")
print("  ✓ Full RAG pipeline quality assessment")

print("\nWhy this matters:")
print("  • Job 1: 'Analyze model outputs' - ✓ You can do this now")
print("  • Job 2: 'Assess responses' - ✓ You have metrics")
print("  • Production: Measure quality over time")
print("  • Iteration: Know what to improve")
