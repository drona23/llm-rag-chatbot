"""LLM interface using Claude API."""
import os
import anthropic
import voyageai


class LLMChat:
    """Wrapper around Claude for conversational AI."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic()
        self.model = model
        # Phase 3: Real Voyage AI embeddings
        # voyage-3-large understands semantic meaning, not just character patterns
        self.voyage = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    def generate_embedding(self, text: str) -> list[float]:
        """
        Convert text to embedding vector using Voyage AI.

        Phase 3 upgrade from mock hash embeddings.
        Voyage AI understands meaning, so similar concepts get similar vectors.
        Auto-retries on rate limit (3 RPM on free tier) with 65s wait.

        Returns:
            List of 1024 floats representing semantic meaning of text
        """
        import time
        import voyageai

        for attempt in range(3):
            try:
                result = self.voyage.embed([text], model="voyage-3-large")
                return result.embeddings[0]
            except voyageai.error.RateLimitError:
                if attempt < 2:
                    print(f"\n  [Rate limit] Waiting 65s before retry {attempt + 2}/3...", flush=True)
                    time.sleep(65)
                else:
                    raise

    def generate_response(
        self,
        user_message: str,
        context_docs: list[str],
        system_prompt: str = None
    ) -> str:
        """
        Generate response using Claude with document context.

        Flow:
        1. Set up system instructions (Claude's role)
        2. Format documents nicely (label each)
        3. Build complete prompt (documents + question)
        4. Send to Claude API
        5. Extract and return response

        Args:
            user_message: User's question
            context_docs: Retrieved documents to include
            system_prompt: Custom system instructions

        Returns:
            Claude's response text
        """

        # STEP 1: Define system prompt (Claude's role)
        if system_prompt is None:
            system_prompt = """You are a helpful student loan advisor AI.

Your job:
- Answer questions using ONLY the provided documents
- If answer isn't in documents, say "I don't have this information"
- Always cite which document you're using
- Be accurate and clear about loan details

Never:
- Make up information about loans
- Go outside the documents
- Provide financial advice beyond what documents say"""

        # STEP 2: Format documents with labels
        # Each document gets a number label for clarity
        formatted_docs = []
        for i, doc in enumerate(context_docs, 1):
            formatted_docs.append(f"[Document {i}]\n{doc}")

        # Join documents with clear separation (---)
        documents_section = "\n\n---\n\n".join(formatted_docs)

        # STEP 3: Build the complete prompt
        # Structure: DOCUMENTS section, QUESTION section, INSTRUCTIONS
        full_prompt = f"""I have some documents about student loans to help answer your question.

DOCUMENTS:
{documents_section}

QUESTION:
{user_message}

Instructions:
- Use ONLY the documents above to answer
- If you can't find the answer in documents, say so
- Cite which document(s) you used
- Be specific and accurate

Answer:"""

        # STEP 4: Send to Claude API
        response = self.client.messages.create(
            model=self.model,                    # Claude model
            max_tokens=1024,                     # Max response length
            system=system_prompt,                # Tell Claude its role
            messages=[
                {
                    "role": "user",              # This is from user
                    "content": full_prompt       # Our structured prompt
                }
            ]
        )

        # STEP 5: Extract text from response
        # Response is wrapped in structure, extract just the text
        answer_text = response.content[0].text

        return answer_text

    def evaluate_response(
        self,
        response: str,
        reference: str = None,
        criteria: str = None
    ) -> dict:
        """
        Score response quality.

        Args:
            response: Generated response
            reference: Ground truth (optional)
            criteria: What to evaluate for

        Returns:
            {"score": float, "feedback": str}
        """
        raise NotImplementedError(
            "evaluate_response() not yet implemented. "
            "Use src/evaluation.py RAGEvaluator for quality scoring."
        )
