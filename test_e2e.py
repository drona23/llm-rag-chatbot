"""
END-TO-END TEST: Complete RAG System

Tests the entire pipeline:
1. Load all student loan documents
2. Embed and store in vector DB
3. Ask questions via RAG agent
4. Evaluate every response
5. Print full quality report
"""

import os
import pathlib
from src.llm import LLMChat
from src.vector_db_mock import MockVectorStore
from src.rag_agent import RAGAgent
from src.evaluation import RAGEvaluator

BASE = pathlib.Path(__file__).parent

# ─── SETUP ────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  END-TO-END TEST: Student Loan RAG Chatbot")
print("=" * 70)

print("\n[1/4] Initializing components...")
llm        = LLMChat()
store      = MockVectorStore()
agent      = RAGAgent(store, llm)
evaluator  = RAGEvaluator()
print("  ✓ LLMChat (claude-sonnet-4-6)")
print("  ✓ MockVectorStore")
print("  ✓ RAGAgent")
print("  ✓ RAGEvaluator")

# ─── LOAD DOCUMENTS ───────────────────────────────────────────────────────────

print("\n[2/4] Loading & embedding documents...")

doc_paths = {
    "loan_types":             BASE / "data/student_loans/loan_types.txt",
    "interest_rates":         BASE / "data/student_loans/interest_rates.txt",
    "repayment_plans":        BASE / "data/student_loans/repayment_plans.txt",
    "eligibility":            BASE / "data/student_loans/eligibility.txt",
    "faq":                    BASE / "data/student_loans/faq.txt",
    "bad_credit_and_private": BASE / "data/student_loans/bad_credit_and_private.txt",
    "pslf_and_forgiveness":   BASE / "data/student_loans/pslf_and_forgiveness.txt",
    "income_driven_repayment":BASE / "data/student_loans/income_driven_repayment.txt",
}

docs_stored = []
for name, path in doc_paths.items():
    if path.exists():
        text      = path.read_text()
        embedding = llm.generate_embedding(text)
        docs_stored.append({"id": name, "text": text, "embedding": embedding})
        print(f"  ✓ {name} ({len(text):,} chars)")
    else:
        print(f"  ✗ MISSING: {path}")

store.upsert_documents(docs_stored)
print(f"\n  Total: {len(docs_stored)} documents in vector DB")

# ─── TEST QUESTIONS ────────────────────────────────────────────────────────────

print("\n[3/4] Running RAG queries + evaluation...")

questions = [
    "What are the main types of federal student loans?",
    "What is the interest rate for direct subsidized loans in 2024-2025?",
    "How do income-driven repayment plans work?",
    "What are the eligibility requirements for federal student loans?",
    "What is the difference between subsidized and unsubsidized loans?",
    "How does Public Service Loan Forgiveness work?",
    "Can I get a student loan if I have bad credit?",
    "When do I start paying back my student loans?",
]

results = []
print()

for i, question in enumerate(questions, 1):
    print(f"  [{i}/{len(questions)}] {question[:60]}...")

    try:
        # Full RAG pipeline
        result = agent.answer(question)

        # Evaluate
        scores = evaluator.full_evaluation(
            query=question,
            response=result["response"],
            sources=result["sources"],
            retrieval_scores=result.get("retrieval_scores", [])
        )

        results.append({
            "question":   question,
            "response":   result["response"],
            "confidence": result["confidence"],
            "scores":     scores
        })

        grade = (
            "✓ Excellent" if scores["overall_score"] > 0.75 else
            "✓ Good"      if scores["overall_score"] > 0.60 else
            "⚠ Fair"      if scores["overall_score"] > 0.45 else
            "✗ Poor"
        )
        print(f"       {grade} | Overall: {scores['overall_score']:.0%} | "
              f"Faith: {scores['faithfulness']:.0%} | "
              f"Relevance: {scores['relevance']:.0%}")

    except Exception as e:
        print(f"       ✗ ERROR: {e}")
        results.append({"question": question, "error": str(e)})

# ─── FULL REPORT ───────────────────────────────────────────────────────────────

print("\n[4/4] Full quality report...")

print("\n" + "=" * 70)
print("  DETAILED RESPONSES")
print("=" * 70)

for i, r in enumerate(results, 1):
    if "error" in r:
        print(f"\n[Q{i}] {r['question']}")
        print(f"  ERROR: {r['error']}")
        continue

    print(f"\n[Q{i}] {r['question']}")
    print("-" * 70)
    # Print first 400 chars of response
    preview = r["response"][:400].replace("\n", " ")
    print(f"  Response: {preview}...")
    print(f"  Retrieval confidence: {r['confidence']:.1%}")
    print(f"  Relevance:            {r['scores']['relevance']:.1%}")
    print(f"  Faithfulness:         {r['scores']['faithfulness']:.1%}")
    print(f"  Answer relevance:     {r['scores']['answer_relevance']:.1%}")
    print(f"  Overall:              {r['scores']['overall_score']:.1%}")

# ─── SUMMARY ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)

successful = [r for r in results if "scores" in r]
if successful:
    avg_overall      = sum(r["scores"]["overall_score"]      for r in successful) / len(successful)
    avg_faith        = sum(r["scores"]["faithfulness"]       for r in successful) / len(successful)
    avg_relevance    = sum(r["scores"]["relevance"]          for r in successful) / len(successful)
    avg_ans_rel      = sum(r["scores"]["answer_relevance"]   for r in successful) / len(successful)
    avg_confidence   = sum(r["confidence"]                   for r in successful) / len(successful)

    print(f"\n  Questions tested:    {len(questions)}")
    print(f"  Successful:          {len(successful)}")
    print(f"  Failed:              {len(results) - len(successful)}")
    print()
    print(f"  Avg Overall Score:   {avg_overall:.1%}")
    print(f"  Avg Faithfulness:    {avg_faith:.1%}")
    print(f"  Avg Relevance:       {avg_relevance:.1%}")
    print(f"  Avg Answer Rel:      {avg_ans_rel:.1%}")
    print(f"  Avg Confidence:      {avg_confidence:.1%}")

    print()
    if avg_overall > 0.75:
        print("  ✓ System quality: EXCELLENT - production ready")
    elif avg_overall > 0.60:
        print("  ✓ System quality: GOOD - acceptable for demo")
    elif avg_overall > 0.45:
        print("  ⚠ System quality: FAIR - needs improvement")
    else:
        print("  ✗ System quality: POOR - requires work")

print("\n" + "=" * 70)
print("  END-TO-END TEST COMPLETE")
print("=" * 70)
