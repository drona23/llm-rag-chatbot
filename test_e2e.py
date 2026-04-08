"""
END-TO-END TEST: Complete RAG System (Phase 3 - Real Pinecone + Voyage AI)

Tests the entire pipeline:
1. Connect to real Pinecone vector DB (documents already uploaded via setup_pinecone.py)
2. Ask questions via RAG agent using real semantic search
3. Evaluate every response
4. Print full quality report

Run setup_pinecone.py first if you haven't already.
"""

import os
import pathlib
from dotenv import load_dotenv

load_dotenv(override=True)  # override empty shell vars with .env values

from src.llm import LLMChat
from src.vector_db import VectorStore
from src.rag_agent import RAGAgent
from src.evaluation import RAGEvaluator

BASE = pathlib.Path(__file__).parent

# ─── SETUP ────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  END-TO-END TEST: Student Loan RAG Chatbot (Phase 3)")
print("=" * 70)

print("\n[1/3] Initializing components...")
llm       = LLMChat()
store     = VectorStore(index_name="student-loans")
agent     = RAGAgent(store, llm)
evaluator = RAGEvaluator()
print("  LLM:       Claude claude-sonnet-4-6")
print("  Embeddings: Voyage AI voyage-3-large (1024 dims)")
print("  Vector DB:  Pinecone (student-loans index)")
print("  Evaluator:  RAGEvaluator")

# Verify documents are in Pinecone
stats = store.index.describe_index_stats()
doc_count = stats["total_vector_count"]
print(f"\n  Documents in Pinecone: {doc_count}")
if doc_count == 0:
    print("\n  ERROR: No documents found. Run setup_pinecone.py first.")
    exit(1)

# ─── TEST QUESTIONS ────────────────────────────────────────────────────────────

print("\n[2/3] Running RAG queries + evaluation...")

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
        result = agent.answer(question)

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
            "Excellent" if scores["overall_score"] > 0.75 else
            "Good"      if scores["overall_score"] > 0.60 else
            "Fair"      if scores["overall_score"] > 0.45 else
            "Poor"
        )
        print(f"       {grade} | Overall: {scores['overall_score']:.0%} | "
              f"Faith: {scores['faithfulness']:.0%} | "
              f"Relevance: {scores['relevance']:.0%} | "
              f"Confidence: {result['confidence']:.0%}")

    except Exception as e:
        print(f"       ERROR: {e}")
        results.append({"question": question, "error": str(e)})

# ─── FULL REPORT ───────────────────────────────────────────────────────────────

print("\n[3/3] Full quality report...")

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
    preview = r["response"][:400].replace("\n", " ")
    print(f"  Response: {preview}...")
    print(f"  Confidence:       {r['confidence']:.1%}")
    print(f"  Faithfulness:     {r['scores']['faithfulness']:.1%}")
    print(f"  Relevance:        {r['scores']['relevance']:.1%}")
    print(f"  Answer relevance: {r['scores']['answer_relevance']:.1%}")
    print(f"  Overall:          {r['scores']['overall_score']:.1%}")

# ─── SUMMARY ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)

successful = [r for r in results if "scores" in r]
if successful:
    avg_overall    = sum(r["scores"]["overall_score"]    for r in successful) / len(successful)
    avg_faith      = sum(r["scores"]["faithfulness"]     for r in successful) / len(successful)
    avg_relevance  = sum(r["scores"]["relevance"]        for r in successful) / len(successful)
    avg_ans_rel    = sum(r["scores"]["answer_relevance"] for r in successful) / len(successful)
    avg_confidence = sum(r["confidence"]                 for r in successful) / len(successful)

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

    verdict = (
        "EXCELLENT - Production Ready" if avg_overall > 0.75 else
        "GOOD - Acceptable for Demo"   if avg_overall > 0.60 else
        "FAIR - Needs Improvement"     if avg_overall > 0.45 else
        "POOR - Requires Work"
    )
    print(f"  System verdict: {verdict}")

print("\n" + "=" * 70)
print("  END-TO-END TEST COMPLETE")
print("=" * 70)
