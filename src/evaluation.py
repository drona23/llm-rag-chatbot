"""Evaluation metrics for RAG chatbot quality."""
import re
import math
from typing import Optional


class RAGEvaluator:
    """Assess response quality using multiple metrics."""

    @staticmethod
    def relevance_score(response: str, sources: list[str]) -> float:
        """
        How well response uses retrieved context.

        Score 0-1: Does response align with sources?

        Strategy:
        - Extract key phrases from response
        - Check if they appear in sources
        - Score = (phrases in sources) / (total phrases)
        """
        if not sources or not response:
            return 0.0

        # Simple approach: check for overlap of content words
        response_words = set(response.lower().split())
        source_words = set(' '.join(sources).lower().split())

        # Find common words (excluding common words)
        common_words = response_words & source_words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'are', 'was', 'be', 'been'}
        meaningful_common = len(common_words - stop_words)
        meaningful_response = len(response_words - stop_words)

        if meaningful_response == 0:
            return 0.0

        # Score: how many meaningful words from response are in sources
        score = meaningful_common / meaningful_response

        return min(score, 1.0)  # Cap at 1.0

    @staticmethod
    def faithfulness(response: str, sources: list[str]) -> float:
        """
        Does response stay grounded in sources?

        Score 0-1: No hallucination = 1.0

        Strategy:
        - Extract sentences from response
        - Check if each sentence is "supported" by sources
        - Supported = key words from sentence appear in sources
        """
        if not sources or not response:
            return 0.0

        # Strip markdown formatting before splitting
        # Removes headers (##), bold (**), bullets (-), tables (|)
        clean = re.sub(r'[#*_`|>]', '', response)
        clean = re.sub(r'\[.*?\]\(.*?\)', '', clean)  # Remove links

        # Split on sentence boundaries (not on decimals like 6.53%)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean) if s.strip() and len(s.strip()) > 10]
        if not sentences:
            return 0.0

        source_text = ' '.join(sources).lower()
        supported_sentences = 0

        for sentence in sentences:
            # Extract key words (non-stop words)
            words = sentence.lower().split()
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'are', 'was', 'be', 'been', 'this', 'that', 'from', 'with'}
            key_words = [w for w in words if w not in stop_words and len(w) > 3]

            # Check if most key words appear in sources
            if key_words:
                found_words = sum(1 for w in key_words if w in source_text)
                # If 70%+ of key words found, sentence is supported
                if found_words / len(key_words) >= 0.7:
                    supported_sentences += 1

        # Score: percentage of supported sentences
        return supported_sentences / len(sentences)

    @staticmethod
    def answer_relevance(query: str, response: str) -> float:
        """
        Does response actually answer the question?

        Score 0-1: Fully answers = 1.0

        Strategy:
        - Extract key words from query
        - Check if response addresses those concepts
        - Score = how much of question is answered in response
        """
        if not query or not response:
            return 0.0

        # Extract key words from query
        query_words = query.lower().split()
        stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'for', 'of', 'to', 'in', 'on', 'at'}
        key_query_words = [w for w in query_words if w not in stop_words and len(w) > 3]

        response_lower = response.lower()

        # Check how many key query words appear in response
        if key_query_words:
            found_words = sum(1 for w in key_query_words if w in response_lower)
            score = found_words / len(key_query_words)
        else:
            # If no key words found, check basic relevance
            score = 0.5 if len(response) > 100 else 0.2

        return min(score, 1.0)

    @staticmethod
    def retrieval_quality(
        sources: list[str],
        relevance_scores: list[float]
    ) -> dict:
        """
        Was the right context retrieved?

        Returns:
            {
                "avg_score": float,
                "top_score": float,
                "useful_docs": int
            }
        """
        if not relevance_scores:
            return {"avg_score": 0, "top_score": 0, "useful_docs": 0}

        avg = sum(relevance_scores) / len(relevance_scores)
        useful = sum(1 for s in relevance_scores if s > 0.7)

        return {
            "avg_score": round(avg, 3),
            "top_score": round(max(relevance_scores), 3),
            "useful_docs": useful
        }

    def full_evaluation(
        self,
        query: str,
        response: str,
        sources: list[str],
        retrieval_scores: Optional[list[float]] = None,
        reference_answer: Optional[str] = None
    ) -> dict:
        """
        Comprehensive quality assessment.

        Args:
            query: User's question
            response: Claude's answer
            sources: Retrieved document texts
            retrieval_scores: Per-document similarity scores from vector DB
            reference_answer: Ground truth answer (optional)

        Returns:
            {
                "relevance": 0-1,
                "faithfulness": 0-1,
                "answer_relevance": 0-1,
                "retrieval_quality": dict,
                "overall_score": 0-1
            }
        """
        rel = self.relevance_score(response, sources)
        faith = self.faithfulness(response, sources)
        ans_rel = self.answer_relevance(query, response)

        # Fix: use actual retrieval scores from vector DB
        # not faithfulness as a proxy (which was wrong)
        actual_scores = retrieval_scores if retrieval_scores else []
        ret_qual = self.retrieval_quality(sources, actual_scores)

        # Weighted average (faithfulness weighted highest → most important)
        overall = (rel * 0.3 + faith * 0.4 + ans_rel * 0.3)

        return {
            "relevance": round(rel, 3),
            "faithfulness": round(faith, 3),
            "answer_relevance": round(ans_rel, 3),
            "retrieval_quality": ret_qual,
            "overall_score": round(overall, 3)
        }
