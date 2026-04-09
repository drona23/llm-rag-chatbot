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

A production-quality RAG chatbot that answers student loan questions using verified federal documentation.

**Architecture:** Voyage AI embeddings -> Pinecone vector search -> Claude Sonnet 4.6 generation

**Quality score:** 76.2% on 8-question end-to-end evaluation

This Space runs in cloud mode -- the Gradio frontend calls a live AWS Lambda backend.
No ML dependencies are installed here; only `gradio` and `requests`.
