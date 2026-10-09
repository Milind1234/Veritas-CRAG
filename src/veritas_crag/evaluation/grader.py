"""Document Grader / Retrieval Evaluator for Corrective RAG."""

import math
from typing import List, Optional
from loguru import logger

from veritas_crag.config.settings import settings
from veritas_crag.models.crag import CRAGAction, DocumentGrade, GradingResult
from veritas_crag.models.document import DocumentChunk
from veritas_crag.retrieval.reranker import CrossEncoderReranker


class DocumentGrader:
    """Evaluates relevance of retrieved chunks and determines corrective action."""

    def __init__(
        self,
        reranker: Optional[CrossEncoderReranker] = None,
        upper_threshold: Optional[float] = None,
        lower_threshold: Optional[float] = None,
    ):
        self.reranker = reranker or CrossEncoderReranker()
        self.upper_threshold = upper_threshold or settings.CRAG_UPPER_THRESHOLD
        self.lower_threshold = lower_threshold or settings.CRAG_LOWER_THRESHOLD

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Converts raw logits / cross-encoder scores to normalized probability [0.0, 1.0]."""
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def grade_chunks(
        self,
        query: str,
        chunks: List[DocumentChunk],
        precomputed_scores: Optional[List[float]] = None,
    ) -> GradingResult:
        """Evaluates candidate chunks against query and returns aggregated grading verdict.

        Args:
            query: User input query.
            chunks: Candidate retrieved document chunks.
            precomputed_scores: Optional raw cross-encoder scores if already calculated.

        Returns:
            GradingResult with chunk-level grades and overall CRAGAction (CORRECT, AMBIGUOUS, INCORRECT).
        """
        if not chunks:
            logger.warning("No chunks provided to grade. Emitting INCORRECT action.")
            return GradingResult(
                grades=[],
                overall_action=CRAGAction.INCORRECT,
                confidence=0.0,
                relevant_chunks=[],
                irrelevant_chunks=[],
            )

        # 1. Obtain raw relevance scores
        if precomputed_scores is not None and len(precomputed_scores) == len(chunks):
            raw_scores = precomputed_scores
        else:
            scored_pairs = self.reranker.rerank(query=query, chunks=chunks, top_k=len(chunks))
            chunk_to_score = {c.chunk_id: score for c, score in scored_pairs}
            raw_scores = [chunk_to_score.get(c.chunk_id, -10.0) for c in chunks]

        # 2. Score and categorize each chunk
        grades: List[DocumentGrade] = []
        relevant_chunks: List[DocumentChunk] = []
        irrelevant_chunks: List[DocumentChunk] = []

        for chunk, raw_score in zip(chunks, raw_scores):
            norm_score = round(self._sigmoid(raw_score), 4)
            is_relevant = norm_score >= self.lower_threshold

            if is_relevant:
                relevant_chunks.append(chunk)
                rationale = f"Relevant (score: {norm_score:.3f} >= {self.lower_threshold})"
            else:
                irrelevant_chunks.append(chunk)
                rationale = f"Irrelevant (score: {norm_score:.3f} < {self.lower_threshold})"

            grades.append(
                DocumentGrade(
                    chunk_id=chunk.chunk_id,
                    score=norm_score,
                    is_relevant=is_relevant,
                    rationale=rationale,
                )
            )

        # 3. Compute overall confidence and action
        # The primary metric is the maximum confidence across candidate documents
        max_score = max((g.score for g in grades), default=0.0)

        if max_score >= self.upper_threshold:
            action = CRAGAction.CORRECT
        elif max_score >= self.lower_threshold:
            action = CRAGAction.AMBIGUOUS
        else:
            action = CRAGAction.INCORRECT

        logger.info(
            f"Evaluator Verdict: {action.value} (Confidence: {max_score:.3f}, "
            f"Relevant Chunks: {len(relevant_chunks)}/{len(chunks)})"
        )

        return GradingResult(
            grades=grades,
            overall_action=action,
            confidence=max_score,
            relevant_chunks=relevant_chunks,
            irrelevant_chunks=irrelevant_chunks,
        )
