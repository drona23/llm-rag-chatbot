"""
AWS Lambda handler for the Student Loan RAG Chatbot.

Deployment: Lambda container image via ECR.
Trigger: API Gateway HTTP POST /chat

Request body (JSON):
    {"message": "What types of student loans are available?"}

Response body (JSON):
    {
        "response": "...",
        "sources": ["...", "..."],
        "confidence": 0.82,
        "retrieval_scores": [0.91, 0.88, ...]
    }
"""
import json
import os
import logging

# Module-level initialization: runs once per container lifecycle (warm starts reuse this).
# Avoids re-connecting to Pinecone + Voyage AI on every request.
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_agent = None


def _get_agent():
    """Lazy-initialize the RAG agent (once per container)."""
    global _agent
    if _agent is None:
        logger.info("Cold start: initializing RAG agent")
        from src.llm import LLMChat
        from src.vector_db import VectorStore
        from src.rag_agent import RAGAgent

        llm = LLMChat()
        store = VectorStore(index_name=os.environ.get("PINECONE_INDEX_NAME", "student-loans"))
        _agent = RAGAgent(vector_store=store, llm=llm)
        logger.info("RAG agent initialized successfully")
    return _agent


def handler(event, context):
    """
    AWS Lambda entry point.

    Args:
        event: API Gateway proxy event dict
        context: Lambda context object (request ID, timeout, etc.)

    Returns:
        API Gateway proxy response dict
    """
    logger.info("Request received: %s", json.dumps({
        "requestId": context.aws_request_id,
        "path": event.get("path", "/"),
        "httpMethod": event.get("httpMethod", "UNKNOWN"),
    }))

    # Handle CORS preflight
    if event.get("httpMethod") == "OPTIONS":
        return _cors_response(200, "")

    # Parse request body
    try:
        body = json.loads(event.get("body") or "{}")
        message = body.get("message", "").strip()
    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning("Bad request body: %s", e)
        return _cors_response(400, json.dumps({"error": "Invalid JSON body"}))

    if not message:
        return _cors_response(400, json.dumps({"error": "Missing 'message' field"}))

    if len(message) > 2000:
        return _cors_response(400, json.dumps({"error": "Message too long (max 2000 chars)"}))

    # Run RAG pipeline
    try:
        agent = _get_agent()
        result = agent.answer(message)
        logger.info("Response generated. confidence=%.3f sources=%d",
                    result["confidence"], len(result["sources"]))
        return _cors_response(200, json.dumps(result))

    except Exception as e:
        logger.error("RAG pipeline error: %s", str(e), exc_info=True)
        return _cors_response(500, json.dumps({"error": "Internal server error"}))


def _cors_response(status_code: int, body: str) -> dict:
    """Wrap response with CORS headers for browser access."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
        },
        "body": body,
    }
