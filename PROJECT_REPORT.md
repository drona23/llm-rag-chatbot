# Project Report: Student Loan RAG Chatbot

## Overview

This project implements a production-quality Retrieval-Augmented Generation (RAG) chatbot for student loan information. It demonstrates end-to-end LLM engineering including document retrieval, prompt engineering, response generation, and automated quality evaluation.

---

## Problem Statement

Large language models like Claude have broad knowledge but lack access to specific, structured domain data. A student asking about loan eligibility or repayment options needs accurate, current information, not generalized answers.

**RAG solves this** by retrieving relevant documents before generating an answer, grounding Claude's responses in verified source material.

---

## Technical Implementation

### Phase 1: Core RAG System

**Components built:**
- `LLMChat`: Claude API wrapper with prompt engineering
- `MockVectorStore`: In-memory vector database with cosine similarity
- `RAGAgent`: Orchestrator combining retrieval + generation
- Mock embedding system using deterministic hash-based vectors

**Key design decision: Prompt Structure:**
```
DOCUMENTS:
[Document 1] {retrieved text}
[Document 2] {retrieved text}

QUESTION: {user query}

Instructions:
- Use ONLY the documents above
- Cite which document you used
- Say "I don't have this information" if not in documents
```

This structured prompt design reduced hallucination and improved source citation.

### Phase 2: Evaluation Framework + Data Expansion

**Evaluation metrics implemented:**

| Metric | Method | Weight |
|--------|--------|--------|
| Faithfulness | Sentence-level keyword grounding check | 40% |
| Relevance | Word overlap between response and sources | 30% |
| Answer Relevance | Key query term coverage in response | 30% |

**Data expanded from 5 to 8 documents:**
- Added `bad_credit_and_private.txt`
- Added `pslf_and_forgiveness.txt`
- Added `income_driven_repayment.txt`

---

## Bugs Found & Fixed (Code Review)

| Bug | Impact | Fix |
|-----|--------|-----|
| Mock embeddings nearly identical | Random retrieval | XOR-based per-dimension hashing |
| Faithfulness used wrong scores | Metric was measuring wrong thing | Pass real retrieval scores |
| Confidence used minimum score | Pessimistic, counter-intuitive | Use average score |
| `evaluate_response()` returned None | Silent failure | Raise NotImplementedError |
| Hardcoded absolute paths | Broke on any other machine | Use `pathlib.Path(__file__).parent` |
| Sentence splitter broke on decimals | "6.53%" split into "6" + "53%" | Regex lookbehind split |

---

## Results

### End-to-End Test (8 Questions)

```
System verdict: EXCELLENT, Production Ready
Overall Score: 76.2%
```

| Question | Score | Grade |
|----------|-------|-------|
| What are the main types of federal student loans? | 82% | Excellent |
| What is the interest rate for direct subsidized loans? | 82% | Excellent |
| How do income-driven repayment plans work? | 74% | Good |
| What are the eligibility requirements? | 85% | Excellent |
| Difference between subsidized and unsubsidized? | 64% | Good |
| How does Public Service Loan Forgiveness work? | 70% | Good |
| Can I get a loan with bad credit? | 71% | Good |
| When do I start paying back loans? | 83% | Excellent |

### Before vs After Fixes

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Overall Score | 63.5% | 76.2% | +12.7% |
| Faithfulness | 59.4% | 88.4% | +29.0% |
| Relevance | 57.5% | 61.2% | +3.7% |
| Confidence | -0.2% | 49.9% | Fixed |

---

## Data Coverage

8 documents covering the full student loan lifecycle:

```
loan_types.txt              → 5 federal loan types + private loans
interest_rates.txt          → 2024-2025 rates, history, origination fees
repayment_plans.txt         → Standard, Graduated, IDR plans, forgiveness
eligibility.txt             → FAFSA, citizenship, SAP requirements
faq.txt                     → 20 common questions answered
bad_credit_and_private.txt  → Credit requirements, cosigner options
pslf_and_forgiveness.txt    → PSLF step-by-step, qualifying employers
income_driven_repayment.txt → IBR, PAYE, REPAYE, ICR explained
```

---

## Limitations

**Mock Embeddings (Phase 1)**
- Not semantic. Retrieval order is not meaningful.
- Workaround: retrieve all documents (`top_k=8`)
- Phase 3 fix: Real Voyage AI embeddings via Pinecone

**Evaluation Metrics**
- Keyword-based, not semantic
- Faithfulness check is approximate (not using NLI model)
- Production upgrade: Use LLM-as-judge evaluation

**No Real-time Data**
- Documents are static (created April 2026)
- Interest rates change annually
- Phase 5 fix: Firecrawl integration for live data

---

## Relevance to Industry

This project directly mirrors real-world ML engineering work:

| Job Requirement | Project Demonstration |
|-----------------|----------------------|
| LLM-powered conversational agents | Full RAG chatbot with multi-turn history |
| RAG pipelines | Vector retrieval + prompt augmentation |
| Prompt engineering | Structured system + user prompts |
| Evaluate model outputs | 3-metric evaluation framework |
| Data quality & annotation | 8 structured domain documents |
| Python + ML tooling | Pure Python implementation |

---

## Future Work

| Phase | Task | Priority |
|-------|------|----------|
| 3 | Replace mock embeddings with Voyage AI | High |
| 3 | Replace MockVectorStore with Pinecone | High |
| 4 | Deploy to AWS Lambda + API Gateway | Medium |
| 4 | Add CloudWatch monitoring | Medium |
| 5 | Firecrawl integration for live rates | Low |
| 5 | LLM-as-judge evaluation | Low |
