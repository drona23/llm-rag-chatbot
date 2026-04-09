# Project Report: Student Loan RAG Chatbot

## Overview

This project implements a production-quality Retrieval-Augmented Generation (RAG) chatbot for student loan information. It demonstrates end-to-end LLM engineering including document retrieval, prompt engineering, response generation, automated quality evaluation, cloud deployment, and a web interface.

---

## Problem Statement

Large language models like Claude have broad knowledge but lack access to specific, structured domain data. A student asking about loan eligibility or repayment options needs accurate, current information, not generalized answers.

**RAG solves this** by retrieving relevant documents before generating an answer, grounding Claude's responses in verified source material.

---

## Technical Implementation

### Phase 1: Core RAG System

**Components built:**
- `LLMChat`: Claude API wrapper with structured prompt engineering
- `MockVectorStore`: In-memory vector database with cosine similarity
- `RAGAgent`: Orchestrator combining retrieval and generation
- Mock embedding system using deterministic hash-based vectors

**Key design decision - Prompt Structure:**
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

### Phase 2: Evaluation Framework and Data Expansion

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

### Phase 3: Real Vector Database and Semantic Embeddings

**Components upgraded:**
- `VectorStore`: Replaced mock store with live Pinecone integration (serverless, AWS us-east-1, cosine similarity metric)
- `LLMChat.generate_embedding()`: Replaced hash-based mock with Voyage AI `voyage-3-large` model (1024-dimensional vectors)

**Impact:**
- Retrieval is now semantically meaningful: similar concepts return similar documents
- Reduced `top_k` from 8 (retrieve all) to 5 (precise semantic matches)
- Added rate limit handling for Voyage AI free tier (3 requests per minute)

### Phase 4: AWS Lambda Deployment and Web Interface

**Infrastructure built:**
- `lambda_handler.py`: API Gateway proxy handler with module-level agent initialization for warm start optimization, CORS headers, and input validation
- `Dockerfile`: Lambda container image based on `public.ecr.aws/lambda/python:3.11` to support large dependencies
- `cloudwatch_config.json`: Three production alarms (error rate, p95 latency, throttles), log metric filters for cold starts and RAG errors, and a monitoring dashboard

**Web interface:**
- `app.py`: Gradio `gr.Blocks` layout with chat panel and live sources sidebar
- Right panel updates in real time with each response, showing retrieved source documents, cosine similarity scores, and confidence level (HIGH / MEDIUM / LOW)
- Example questions preloaded for interview demos
- Dual-mode operation: runs locally against a real agent, or in cloud mode by setting `RAG_API_URL` to forward requests to the Lambda endpoint

**Hugging Face Spaces deployment:**
- `requirements_spaces.txt`: lightweight dependency file containing only `gradio` and `requests` (no ML dependencies)
- `SPACES_README.md`: Space configuration with YAML frontmatter (`sdk: gradio`, `app_file: app.py`)
- Public demo hosted at [huggingface.co/spaces/DronA23/student-loan-rag-chatbot](https://huggingface.co/spaces/DronA23/student-loan-rag-chatbot)
- `RAG_API_URL` injected as a Space secret so the frontend calls the live AWS endpoint

---

## Bugs Found and Fixed (Code Review)

| Bug | Impact | Fix |
|-----|--------|-----|
| Mock embeddings nearly identical | Random retrieval order | XOR-based per-dimension hashing |
| Faithfulness used wrong scores | Metric measured wrong thing | Pass real retrieval scores from vector DB |
| Confidence used minimum score | Pessimistic, counter-intuitive | Use average score across retrieved docs |
| `evaluate_response()` returned None | Silent failure | Raise NotImplementedError |
| Hardcoded absolute paths | Broke on other machines | Use `pathlib.Path(__file__).parent` |
| Sentence splitter broke on decimals | "6.53%" split into "6" + "53%" | Regex lookbehind split |
| `gr.State` with live SSL sockets | Gradio deepcopy crash on startup | Move agent to module-level global |
| Gradio chat history format | TypeError on every message | Update from tuple pairs to messages format |
| HF Spaces installed Gradio 5.x despite `>=5.0.0` | `theme` in `launch()` not valid, build error | Removed Gradio pin; platform manages the version |
| numpy 2.x pulled by voyageai in Lambda | C compiler not found, build failure | Pin `numpy==1.26.4` before other deps in Dockerfile |
| Lambda image built as OCI manifest list | Lambda only accepts single-arch images | Added `--provenance=false --sbom=false` to `docker buildx` |

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

### Before vs After Bug Fixes

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Overall Score | 63.5% | 76.2% | +12.7% |
| Faithfulness | 59.4% | 88.4% | +29.0% |
| Relevance | 57.5% | 61.2% | +3.7% |
| Confidence (avg) | broken | 49.9% | Fixed |

---

## Data Coverage

8 documents covering the full student loan lifecycle:

```
loan_types.txt              -> 5 federal loan types + private loans
interest_rates.txt          -> 2024-2025 rates, history, origination fees
repayment_plans.txt         -> Standard, Graduated, IDR plans, forgiveness
eligibility.txt             -> FAFSA, citizenship, SAP requirements
faq.txt                     -> 20 common questions answered
bad_credit_and_private.txt  -> Credit requirements, cosigner options
pslf_and_forgiveness.txt    -> PSLF step-by-step, qualifying employers
income_driven_repayment.txt -> IBR, PAYE, REPAYE, ICR explained
```

---

## Limitations

**Evaluation Metrics**
- Keyword-based, not semantic
- Faithfulness check is approximate (not using NLI model)
- Planned upgrade: LLM-as-judge evaluation using Claude to score responses

**Static Data**
- Documents created April 2026, updated manually
- Interest rates change annually
- Phase 5 fix: Firecrawl integration for automated live data from StudentLoans.gov

**No Authentication**
- Current Lambda handler accepts all requests
- Production would require API keys or JWT tokens on the API Gateway

---

## Relevance to Industry

This project directly mirrors real-world ML engineering work:

| Job Requirement | Project Demonstration |
|-----------------|----------------------|
| LLM-powered conversational agents | Full RAG chatbot with multi-turn conversation history |
| RAG pipelines | Vector retrieval + prompt augmentation |
| Prompt engineering | Structured system and user prompts |
| Evaluate model outputs | 3-metric evaluation framework |
| Data quality and annotation | 8 structured domain documents |
| Python and ML tooling | Anthropic, Pinecone, Voyage AI SDKs |
| Cloud deployment | AWS Lambda container image + API Gateway |
| Monitoring and observability | CloudWatch alarms, dashboards, log filters |

---

## Future Work

| Phase | Task | Priority | Status |
|-------|------|----------|--------|
| 5 | Firecrawl integration for live StudentLoans.gov data | High | Planned |
| 5 | Scheduled weekly document sync | Medium | Planned |
| 5 | LLM-as-judge evaluation | Medium | Planned |
| 5 | API authentication and rate limiting | Low | Planned |
