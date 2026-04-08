# Build Steps & Implementation Roadmap

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG CHATBOT SYSTEM                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   User Question                                                 │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────┐    embed     ┌──────────────┐                    │
│   │  Input  │ ──────────▶  │  Embeddings  │                    │
│   └─────────┘              └──────┬───────┘                    │
│                                   │ vector                      │
│                                   ▼                             │
│   ┌─────────────────────────────────────┐                      │
│   │          Vector Database            │                       │
│   │   [doc1] [doc2] [doc3] ... [docN]  │                       │
│   └──────────────┬──────────────────────┘                      │
│                  │ top-k similar docs                           │
│                  ▼                                              │
│   ┌─────────────────────────┐                                  │
│   │      RAG Agent          │                                  │
│   │  docs + question        │                                  │
│   │  → structured prompt    │                                  │
│   └───────────┬─────────────┘                                  │
│               │                                                 │
│               ▼                                                 │
│   ┌─────────────────────────┐   ┌─────────────────────┐       │
│   │      Claude API         │   │     Evaluator        │       │
│   │  (claude-sonnet-4-6)    │──▶│  faithfulness 88%    │       │
│   └─────────────────────────┘   │  relevance    61%    │       │
│               │                  │  overall      76%    │       │
│               ▼                  └─────────────────────┘       │
│        Answer + Sources                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Core RAG System

```
┌─────────────────────────────────────────┐
│  PHASE 1 COMPONENTS                     │
├─────────────────────────────────────────┤
│                                         │
│  src/llm.py                             │
│  ├── LLMChat.__init__()                 │
│  ├── generate_embedding()  ← mock hash  │
│  └── generate_response()   ← CORE       │
│                                         │
│  src/vector_db_mock.py                  │
│  ├── MockVectorStore()                  │
│  ├── upsert_documents()                 │
│  └── retrieve()  ← cosine similarity   │
│                                         │
│  src/rag_agent.py                       │
│  └── RAGAgent.answer()  ← orchestrator │
│                                         │
└─────────────────────────────────────────┘
```

### Step 1: Implement `generate_response()`
**File:** `src/llm.py`

```
Input:  user_message + context_docs
         │
         ├── Set system prompt (Claude's role)
         ├── Label each document [Document 1], [Document 2]...
         ├── Build structured prompt
         │     DOCUMENTS: [labeled docs]
         │     QUESTION: [user question]
         │     Instructions: use only documents, cite sources
         └── Send to Claude API → return answer
```

### Step 2: Implement `RAGAgent.answer()`
**File:** `src/rag_agent.py`

```
User Question
      │
      ▼
generate_embedding(question) → vector
      │
      ▼
vector_store.retrieve(vector, top_k=8) → [doc1, doc2, ...]
      │
      ▼
llm.generate_response(question, docs) → answer
      │
      ▼
Calculate confidence (avg similarity score)
      │
      ▼
Return {response, sources, confidence, retrieval_scores}
```

---

## Phase 2: Evaluation Framework

```
┌──────────────────────────────────────────────────────┐
│  EVALUATION PIPELINE                                  │
├──────────────────────────────────────────────────────┤
│                                                       │
│  RAG Answer                                           │
│       │                                               │
│       ├──▶ relevance_score()                          │
│       │    "Does response use source words?"          │
│       │    Method: word overlap                        │
│       │    Weight: 30%                                 │
│       │                                               │
│       ├──▶ faithfulness()                             │
│       │    "Is response grounded in documents?"        │
│       │    Method: sentence-level keyword check        │
│       │    Weight: 40%                                 │
│       │                                               │
│       └──▶ answer_relevance()                         │
│            "Does response answer the question?"        │
│            Method: query keyword coverage              │
│            Weight: 30%                                 │
│                 │                                      │
│                 ▼                                      │
│          overall_score = (rel×0.3) + (faith×0.4)     │
│                        + (ans_rel×0.3)                │
│                                                        │
└──────────────────────────────────────────────────────┘
```

### Score Interpretation
```
> 75%  → Excellent  (production ready)
60-75% → Good       (acceptable)
45-60% → Fair       (needs improvement)
< 45%  → Poor       (requires work)
```

---

## Phase 3: Real Vector DB (Planned)

```
Current (Phase 1-2):               Planned (Phase 3):
────────────────────               ─────────────────────
MockVectorStore                    Pinecone
   │                                  │
   ├── In-memory                      ├── Cloud-hosted
   ├── Mock embeddings                ├── Voyage AI embeddings
   ├── Not semantic                   ├── True semantic search
   └── top_k=8 (retrieve all)        └── top_k=3-5 (precise)

Switch: Change 1 line in rag_agent.py:
  FROM: from src.vector_db_mock import MockVectorStore
  TO:   from src.vector_db import VectorStore
```

---

## Phase 4: AWS Deployment (Planned)

```
┌──────────────────────────────────────────────────────────┐
│                     AWS Architecture                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   HTTP Request                                           │
│        │                                                 │
│        ▼                                                 │
│  ┌─────────────┐     ┌──────────────────┐              │
│  │ API Gateway │────▶│  Lambda Function  │              │
│  │ (Public URL)│     │  (rag_agent.py)  │              │
│  └─────────────┘     └────────┬─────────┘              │
│                               │                          │
│                    ┌──────────┴──────────┐              │
│                    ▼                     ▼              │
│           ┌──────────────┐    ┌──────────────────┐     │
│           │  Environment │    │   CloudWatch     │     │
│           │  Variables   │    │   (Monitoring)   │     │
│           │  (API Keys)  │    │   Logs + Metrics │     │
│           └──────────────┘    └──────────────────┘     │
│                                                          │
│           External Services:                             │
│           ├── Pinecone (Vector DB)                      │
│           └── Claude API (Anthropic)                    │
└──────────────────────────────────────────────────────────┘
```

---

## Phase 5: Live Data (Planned)

```
Current:                           Planned:
─────────────────                  ──────────────────────
Static .txt files                  Firecrawl → StudentLoans.gov
Created manually                   Automated scraping
Updated manually                   Scheduled updates (weekly)
8 documents                        100+ documents
```

---

## Full Roadmap

```
Phase 1 ──────── Phase 2 ──────── Phase 3 ──────── Phase 4 ──────── Phase 5
   │                 │                 │                 │                │
   ▼                 ▼                 ▼                 ▼                ▼
Core RAG         Evaluation       Real Pinecone      AWS Deploy      Live Data
   │                 │                 │                 │                │
generate_        3 metrics        Voyage AI          Lambda          Firecrawl
response()       faithfulness     embeddings         API Gateway     weekly sync
RAGAgent         relevance        semantic           CloudWatch
answer()         answer_rel       retrieval          monitoring

STATUS:          STATUS:          STATUS:            STATUS:         STATUS:
  ✓ Done           ✓ Done           Planned            Planned         Planned
```

---

## Files Created Per Phase

```
Phase 1:
├── src/llm.py
├── src/vector_db_mock.py
├── src/rag_agent.py
├── data/student_loans/ (5 files)
├── test_phase1_step1.py
└── test_phase1_step2.py

Phase 2:
├── src/evaluation.py
├── data/student_loans/ (3 more files, total 8)
├── test_phase2_step1.py
└── test_e2e.py

Phase 3 (planned):
├── src/vector_db.py (exists, needs Pinecone key)
└── Voyage AI embedding integration

Phase 4 (planned):
├── lambda_handler.py
├── Dockerfile
└── cloudwatch_config.json
```
