"""Faithfulness and Hallucination Grader verifying answer grounding."""

import math
import re
from typing import List, Optional
from loguru import logger

from veritas_crag.config.settings import settings
from veritas_crag.models.pipeline import Citation, GroundingEvaluation
from veritas_crag.retrieval.reranker import CrossEncoderReranker


class GroundingGrader:
    """Verifies that generated answers are faithful to the source context and free of hallucinations."""

    def __init__(
        self,
        reranker: Optional[CrossEncoderReranker] = None,
        threshold: Optional[float] = None,
    ):
        self.reranker = reranker or CrossEncoderReranker()
        self.threshold = threshold or settings.GROUNDING_THRESHOLD

    @staticmethod
    def _sigmoid(x: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        # Split on sentences while ignoring decimal dots
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def evaluate(
        self,
        query: str,
        answer: str,
        citations: List[Citation],
    ) -> GroundingEvaluation:
        """Evaluates whether the answer is strictly grounded in the provided citations.

        Args:
            query: User's original prompt.
            answer: Generated candidate answer.
            citations: Active citations supporting the answer.

        Returns:
            GroundingEvaluation containing grounding score, hallucination flags, and rationale.
        """
        if not citations:
            logger.warning("No citations provided for answer. Marking as ungrounded.")
            return GroundingEvaluation(
                is_grounded=False,
                grounding_score=0.0,
                hallucination_detected=True,
                hallucinated_claims=[answer[:100]],
                answers_query=False,
                rationale="Answer lacks supporting citation context.",
            )

        # Compile all citation texts as reference ground truth
        full_context = " ".join([c.snippet for c in citations])
        sentences = self._split_into_sentences(answer)

        if not sentences:
            sentences = [answer.strip()]

        # Score each answer sentence against reference context using cross-encoder
        pairs = [(sent, full_context) for sent in sentences]
        raw_scores = self.reranker._model.predict(pairs)

        sentence_scores: List[float] = [self._sigmoid(score) for score in raw_scores]
        hallucinated_claims: List[str] = []

        for sent, score in zip(sentences, sentence_scores):
            # If cross-encoder confidence is below threshold, flag as potential hallucination
            if score < 0.40:
                hallucinated_claims.append(sent)

        # Average grounding score
        avg_score = round(sum(sentence_scores) / max(len(sentence_scores), 1), 3)
        is_grounded = avg_score >= self.threshold and len(hallucinated_claims) == 0

        rationale = (
            f"Grounding score: {avg_score:.3f}. All {len(sentences)} claims verified against citations."
            if is_grounded
            else f"Grounding score: {avg_score:.3f}. {len(hallucinated_claims)} claim(s) require verification."
        )

        logger.info(
            f"Grounding Verification: is_grounded={is_grounded}, score={avg_score:.3f}, "
            f"hallucinations={len(hallucinated_claims)}"
        )

        return GroundingEvaluation(
            is_grounded=is_grounded,
            grounding_score=avg_score,
            hallucination_detected=len(hallucinated_claims) > 0,
            hallucinated_claims=hallucinated_claims,
            answers_query=True,
            rationale=rationale,
        )
