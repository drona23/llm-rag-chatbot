---
title: Student Loan RAG Chatbot
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.23.3"
app_file: app.py
pinned: false
---

# Student Loan RAG Chatbot

A production-quality RAG (Retrieval-Augmented Generation) chatbot that answers student loan questions using verified federal documentation.

## How It Works

Every message goes through a full RAG pipeline running on AWS Lambda:

1. Your question is converted to a 1024-dimensional semantic vector using Voyage AI
2. Pinecone searches 8 verified federal documents for the most relevant content
3. Claude Sonnet 4.6 reads the retrieved documents and generates a grounded answer
4. The response includes the source documents used and a confidence score

## What Makes This Different From a Regular Chatbot

A standard chatbot guesses based on training data. This system retrieves the actual source documents before answering. If the information is not in the documents, the bot says so rather than making something up.

## Quality Results

Tested on 8 real student loan questions:

| Metric | Score |
|--------|-------|
| Overall quality | 76.2% |
| Faithfulness (low hallucination) | 88.4% |
| Source relevance | 61.2% |
| Answer relevance | 75.0% |

## Architecture

```
Browser (this Space)
  |
  v  HTTP POST
AWS API Gateway (us-east-1)
  |
  v
AWS Lambda (Python 3.11, container image)
  |-- Voyage AI: question -> 1024-dim vector
  |-- Pinecone: cosine similarity search -> top 5 docs
  |-- Claude Sonnet 4.6: docs + question -> grounded answer
  |-- Evaluator: faithfulness + relevance scores
  |
  v
{response, sources, confidence, retrieval_scores}
```

This Space runs in cloud mode. Only `gradio` and `requests` are installed here. All ML dependencies run on Lambda.

## Source Code

[github.com/drona23/llm-rag-chatbot](https://github.com/drona23/llm-rag-chatbot)
