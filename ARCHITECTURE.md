# Architecture Diagram (How Everything Connects)

## High-Level View

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR CHATBOT SYSTEM                      │
└─────────────────────────────────────────────────────────────┘

                        USER ASKS A QUESTION
                                ↓
                    ┌───────────────────────┐
                    │   RAG Agent (Brain)   │
                    └───────────────────────┘
                         ↙          ↘
            ┌──────────────────┐  ┌──────────────────┐
            │  Vector Database │  │   Claude (AI)    │
            │    (Pinecone)    │  │    (API)         │
            └──────────────────┘  └──────────────────┘
                    ↑                      ↑
                    │                      │
         ┌──────────┴──────────┐  ┌────────┴─────────┐
         │ Your Documents      │  │ LLM Wrapper      │
         │ (PDFs, Text Files)  │  │ (llm.py)         │
         └─────────────────────┘  └──────────────────┘


STEP-BY-STEP:
1. User: "What's the interest rate?"
2. RAG Agent extracts key words
3. RAG Agent converts question to vector
4. Vector DB searches: "Which documents are similar?"
5. Vector DB returns: [doc_auto_loans.txt, doc_rates.txt, doc_policies.txt]
6. RAG Agent sends to Claude: "Using these 3 documents, answer: What's the rate?"
7. Claude reads documents and answers
8. RAG Agent evaluates: "Is this answer good? Did Claude hallucinate?"
9. Return: {answer: "...", sources: [...], confidence: 0.95}
```

---

## File Interaction Map

```
User Input
    ↓
┌─────────────────────┐
│   rag_agent.py      │ ← Main orchestrator (the director)
└─────────────────────┘
    ↙        ↘
   /          \
  /            \
 ↓              ↓
vector_db.py   llm.py          evaluation.py
(Retrieval)    (Generation)    (Quality Check)

vector_db.py:
- Takes: question (text)
- Does: converts to vector, searches Pinecone
- Returns: relevant documents

llm.py:
- Takes: documents + question
- Does: sends to Claude with smart prompt
- Returns: answer

evaluation.py:
- Takes: answer + documents
- Does: checks if answer is faithful
- Returns: score (0-1)
```

---

## Data Flow Diagram

```
SETUP PHASE (One time):
┌─────────────────┐
│ Raw Documents   │  (PDFs, CSVs, text files about your business)
└────────┬────────┘
         ↓
┌─────────────────┐
│  Embedding API  │  (Convert text to vectors)
└────────┬────────┘
         ↓
┌─────────────────┐
│  Pinecone DB    │  (Store vectors + original text)
└─────────────────┘


RUNTIME PHASE (Every time user asks):
┌──────────────────┐
│  User Question   │  "What's the interest rate?"
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Convert to Vec   │
└────────┬─────────┘
         ↓
┌──────────────────┐
│  Search Pinecone │  "Find 5 most similar documents"
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Get Top 5 Docs   │  [doc1.txt, doc2.txt, ...]
└────────┬─────────┘
         ↓
┌──────────────────────────────────────┐
│ Build Smart Prompt                   │
│ "Using these docs, answer: ..."      │
└────────┬─────────────────────────────┘
         ↓
┌──────────────────┐
│ Send to Claude   │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Get Response     │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Evaluate Quality │  "Did Claude use documents? No hallucination?"
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Return to User   │  {answer, sources, score}
└──────────────────┘
```

---

## AWS Deployment Architecture (Phase 3)

```
┌─────────────────────────────────────────────────────────────┐
│                    AWS Cloud                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌────────────────┐                 │
│  │  API Gateway │◄─────┤ Lambda Function│                 │
│  │  (Public URL)│      │ (rag_agent.py) │                 │
│  └──────────────┘      └────────────────┘                 │
│                              │                             │
│                    ┌─────────┴─────────┐                  │
│                    ↓                   ↓                  │
│          ┌──────────────────┐  ┌──────────────────┐      │
│          │  Environment Vars │  │  CloudWatch Logs │      │
│          │  (API Keys)       │  │  (Monitoring)    │      │
│          └──────────────────┘  └──────────────────┘      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                    ↓         ↓
        ┌───────────────────────────────┐
        │  External Services            │
        ├───────────────────────────────┤
        │  • Pinecone (Vector DB)       │
        │  • Claude API (Anthropic)     │
        └───────────────────────────────┘
```

---

## The 3 Phases

```
PHASE 1: Core RAG (Weeks 1-2)
├── vector_db.py ✓ (done)
├── llm.py ✓ (done)
├── rag_agent.py ✓ (done)
├── evaluation.py ✓ (done)
├── YOU IMPLEMENT: generate_response() ← HERE
└── Test locally with sample data

PHASE 2: Evaluation & Data (Weeks 3-4)
├── Build annotation system
├── Gather real auto-loan data
├── Test on real data
└── Measure quality metrics

PHASE 3: AWS Deployment (Weeks 5-6)
├── Create Lambda function
├── Set up API Gateway
├── Add CloudWatch monitoring
├── Deploy to production
└── Document everything

PHASE 4: Resume + Interview (Week 7+)
├── Write portfolio piece
├── Prepare to explain architecture
├── Demo live chatbot
└── Discuss design decisions
```

---

## Key Components Explained

### Pinecone (Vector Database)
**What:** A specialized database that stores and searches vectors (numbers)
**Why:** Regular databases search by exact match. Vector DBs search by SIMILARITY.
```
Regular DB: "Find rows where name = 'John'"
Vector DB: "Find rows similar to [0.2, 0.5, 0.8, ...]"
```

### Claude API (LLM)
**What:** AI model that understands text and generates responses
**Why:** It's smart, but doesn't know YOUR business data (hence RAG)

### RAG Agent (Orchestrator)
**What:** Code that coordinates between Vector DB and Claude
**Why:** Ensures proper flow: question → search → generate → evaluate

### Evaluation Module
**What:** Scores how good answers are
**Why:** Production systems MUST measure quality (both jobs ask for this)

---

## Why This Structure Works for the Job

```
Job 1 Requirements:
✓ LLM applications          → RAG Agent uses Claude API
✓ RAG pipelines            → vector_db.py + llm.py + rag_agent.py
✓ Prompt engineering       → generate_response() (you design this)
✓ Data quality             → evaluation.py measures it
✓ Cloud deployment         → AWS Lambda (Phase 3)
✓ Model evaluation         → evaluation.py provides metrics

Job 2 Requirements:
✓ LLM applications         → Whole project
✓ Prompt refinement        → generate_response() design
✓ Data annotation          → Can extend to build annotation UI
✓ Model evaluation         → evaluation.py
✓ Python + ML concepts     → All built in Python with proper structure
```

---

## Mental Model

Think of RAG like being a teacher's assistant:

**Without RAG:**
- Student: "What was the homework?"
- AI (Claude alone): "I don't know, I wasn't in your class"

**With RAG:**
- Student: "What was the homework?"
- You (RAG Agent): "Let me check the syllabus..."
- You: "Here's the syllabus. Based on it, the homework is..."

**Your job:** Be the assistant that finds the right document and gives Claude context.
