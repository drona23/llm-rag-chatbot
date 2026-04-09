# Architecture: Student Loan RAG Chatbot

## High-Level System View

```
+-------------------------------------------------------------+
|                    RAG CHATBOT SYSTEM                       |
+-------------------------------------------------------------+

                      USER ASKS A QUESTION
                               |
                   +-----------v-----------+
                   |   RAG Agent (Brain)   |
                   |     rag_agent.py      |
                   +-----------+-----------+
                        /                \
           +-----------v--------+  +------v-----------+
           |   Vector Database  |  |   Claude API     |
           |   (Pinecone Cloud) |  |   (Anthropic)    |
           +--------------------+  +------------------+
                    |                        |
       +------------v-----------+  +---------v--------+
       | Voyage AI Embeddings   |  |   llm.py wrapper |
       | (voyage-3-large, 1024d)|  |   + prompt logic |
       +------------------------+  +------------------+
                    |
       +------------v-----------+
       | 8 Student Loan Docs    |
       | data/student_loans/    |
       +------------------------+


QUERY FLOW (every request):
1. User: "What is the interest rate?"
2. Voyage AI converts question to 1024-dimensional vector
3. Pinecone finds the 5 most similar document vectors
4. RAG Agent builds structured prompt: docs + question + instructions
5. Claude reads documents and generates a grounded answer
6. Evaluator scores faithfulness, relevance, answer quality
7. Return: {answer, sources, confidence, retrieval_scores}
```

---

## File Interaction Map

```
User Input
    |
    v
+---------------------+
|   rag_agent.py      |  <-- Main orchestrator
+---------------------+
    /         |          \
   /           |           \
  v            v            v
vector_db.py  llm.py    evaluation.py
(Retrieval)  (Generation) (Quality Check)

vector_db.py:
  Input:  query embedding (1024-dim float list)
  Does:   cosine similarity search in Pinecone
  Output: list of {text, score} dicts

llm.py:
  Input:  context documents + user question
  Does:   builds structured prompt, calls Claude API
  Output: answer string

evaluation.py:
  Input:  answer + source documents + query
  Does:   keyword overlap, sentence grounding checks
  Output: {faithfulness, relevance, answer_relevance, overall}
```

---

## Data Flow: Setup vs Runtime

```
SETUP (one time - run setup_pinecone.py):

+---------------------+
| 8 .txt Documents    |
| data/student_loans/ |
+----------+----------+
           |
           v
+----------+----------+
| Voyage AI Embedding |  voyage-3-large -> 1024-dim vector per doc
+----------+----------+
           |
           v
+----------+----------+
| Pinecone Index      |  student-loans, cosine metric, AWS us-east-1
| Store vectors +     |
| metadata (text)     |
+---------------------+


RUNTIME (every user request):

+---------------------+
| User Question       |  "What are income-driven repayment plans?"
+----------+----------+
           |
           v
+----------+----------+
| Voyage AI Embedding |  Question -> 1024-dim vector
+----------+----------+
           |
           v
+----------+----------+
| Pinecone Query      |  top_k=5, cosine similarity
+----------+----------+
           |
           v
+----------+----------+
| Top 5 Documents     |  [{text, score}, ...]
+----------+----------+
           |
           v
+----------------------------------+
| Structured Prompt                |
| "DOCUMENTS: [labeled docs]       |
|  QUESTION: [user query]          |
|  Instructions: cite sources..."  |
+----------+-----------------------+
           |
           v
+----------+----------+
| Claude Sonnet 4.6   |  Generates grounded answer
+----------+----------+
           |
           v
+----------+----------+
| Evaluator           |  Scores faithfulness + relevance
+----------+----------+
           |
           v
+----------+--------------------------------------------+
| Response to User                                      |
| {response, sources, confidence, retrieval_scores}    |
+-------------------------------------------------------+
```

---

## AWS Deployment Architecture (Phase 4)

```
                    Internet
                        |
                        v
           +------------+------------+
           |      API Gateway        |
           |  POST /chat             |
           |  Public HTTPS endpoint  |
           +------------+------------+
                        |
                        v
           +------------+------------+
           |      AWS Lambda         |
           |  lambda_handler.py      |
           |  Container image (ECR)  |
           |  Python 3.11            |
           +------+-----+------------+
                  |     |
         +--------+     +--------+
         |                       |
         v                       v
+--------+--------+   +----------+--------+
| Environment     |   | CloudWatch        |
| Variables       |   | Logs + Metrics    |
| ANTHROPIC_KEY   |   | Alarms:           |
| PINECONE_KEY    |   |  - Error rate     |
| VOYAGE_KEY      |   |  - p95 latency    |
+-----------------+   |  - Throttles      |
                       +-------------------+
                                |
                    +-----------v-----------+
                    | External Services     |
                    |                       |
                    |  Pinecone (Vector DB) |
                    |  Anthropic Claude API |
                    |  Voyage AI Embeddings |
                    +-----------------------+
```

**Warm start optimization**: The RAG agent (Pinecone client, Voyage AI client, Claude client) is initialized once at module load time. Lambda reuses the container for subsequent requests, skipping re-initialization. Only the first request per container (cold start) pays the initialization cost.

---

## Gradio Web UI Architecture

The app supports two modes. The mode is selected at startup based on the `RAG_API_URL` environment variable.

```
+------------------------------------+
|  app.py starts                     |
|                                    |
|  RAG_API_URL set?                  |
|        |                           |
|   YES  |   NO                      |
|   v    |    v                      |
| Cloud  |  Local                    |
| mode   |  mode                     |
|        |                           |
| HTTP   |  _agent = build_agent()   |
| POST   |  Pinecone + Voyage + Claude|
| to AWS |  run in-process           |
+--------+---------------------------+
```

### Local Mode

```
Browser
    |
    v (HTTP localhost:7860)
+----------------------------+
|  Gradio Blocks App         |
|  app.py                    |
|                            |
|  +----------+ +---------+  |
|  | Chat     | | Sources |  |
|  | Panel    | | Panel   |  |
|  | (left)   | | (right) |  |
|  +----------+ +---------+  |
+----------+-----------------+
           |
           v (Python function call)
+----------+----------+
|  _agent (global)    |  Module-level singleton
|  RAGAgent instance  |  Holds live Pinecone + Voyage + Claude clients
+---------------------+  Kept global to avoid serialization issues with live network sockets
```

### Cloud Mode (Hugging Face Spaces)

```
Browser (HF Space)
    |
    v (Gradio HTTP)
+----------------------------+
|  Gradio Blocks App         |
|  app.py (cloud mode)       |
|  Only gradio + requests    |
|  installed                 |
+----------+-----------------+
           |
           v (requests.post)
+----------+-----------------------------------+
|  AWS API Gateway                             |
|  POST /prod/chat                             |
|  https://shzjgfckxe.execute-api...           |
+----------+-----------------------------------+
           |
           v
+----------+----------+
|  AWS Lambda          |  Full RAG pipeline runs here
|  lambda_handler.py  |  Pinecone + Voyage AI + Claude
+---------------------+
```

---

## Component Status by Phase

```
Phase 1: Core RAG              COMPLETE
  vector_db_mock.py            Hash-based mock embeddings, in-memory store
  llm.py                       Claude API wrapper, structured prompt
  rag_agent.py                 Orchestrator, conversation history
  evaluation.py                3-metric quality framework

Phase 2: Evaluation + Data     COMPLETE
  8 student loan documents     Full lifecycle coverage
  test_e2e.py                  76.2% overall quality score

Phase 3: Real Vector DB        COMPLETE
  vector_db.py                 Pinecone serverless, cosine similarity
  llm.py (updated)             Voyage AI voyage-3-large embeddings
  setup_pinecone.py            One-time document ingestion

Phase 4: Deployment + UI       COMPLETE
  lambda_handler.py            AWS Lambda entry point, CORS, warm starts
  Dockerfile                   ECR container image
  cloudwatch_config.json       Alarms, dashboards, log filters
  app.py                       Gradio web interface (dual-mode: local + cloud)
  requirements_spaces.txt      Lightweight deps for HF Spaces
  SPACES_README.md             HF Space config with YAML frontmatter

Phase 5: Live Data             PLANNED
  Firecrawl integration        Scrape StudentLoans.gov
  Scheduled sync               Weekly document updates
  LLM-as-judge evaluation      Claude scoring Claude's answers
```

---

## Key Design Decisions

### Why Cosine Similarity?
Cosine similarity measures the angle between two vectors, ignoring magnitude. This is ideal for text embeddings because it compares semantic direction (meaning) rather than vector length (word count). Two documents with the same topic will have high cosine similarity regardless of length.

### Why Average Confidence Score?
The agent computes confidence as the average retrieval score across all top-5 documents, not the minimum. Using the minimum creates a pessimistic metric that gets worse as you retrieve more documents. The average is more stable and representative of overall retrieval quality.

### Why a Module-Level Agent in Gradio?
Gradio internally copies component state between requests. Live network clients (Pinecone, Voyage AI) hold open connections that cannot be serialized and transferred across this boundary. Keeping the agent at module level avoids this entirely and ensures only one set of API connections exists per process.

### Why Lambda Container Image vs ZIP?
Lambda ZIP packages have a 50MB compressed limit. The combined size of `anthropic`, `pinecone`, `voyageai`, and `gradio` exceeds this. Container images support up to 10GB, with dependencies installed as a cached layer so rebuilds only re-copy application code.
