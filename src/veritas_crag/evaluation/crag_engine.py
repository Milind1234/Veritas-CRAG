"""Master Corrective RAG (CRAG) Decision and Fallback Orchestration Engine."""

import time
from typing import List, Optional
from loguru import logger

from veritas_crag.config.settings import settings
from veritas_crag.evaluation.grader import DocumentGrader
from veritas_crag.evaluation.query_rewriter import QueryRewriter
from veritas_crag.evaluation.refiner import KnowledgeRefiner
from veritas_crag.evaluation.web_search import WebSearchEngine
from veritas_crag.models.crag import (
    CRAGAction,
    CRAGContext,
    GradingResult,
    RefinedKnowledge,
)
from veritas_crag.models.document import DocumentChunk
from veritas_crag.retrieval.hybrid import HybridRetriever
from veritas_crag.retrieval.reranker import CrossEncoderReranker


class CRAGEngine:
    """Production Corrective RAG Decision Engine.

    Evaluates retrieved document quality, triggers active web search fallback
    when internal knowledge is ambiguous or incorrect, and refines knowledge
    into clean context strips for answer generation.
    """

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        grader: Optional[DocumentGrader] = None,
        query_rewriter: Optional[QueryRewriter] = None,
        web_search: Optional[WebSearchEngine] = None,
        refiner: Optional[KnowledgeRefiner] = None,
    ):
        # Share a single CrossEncoder reranker instance to avoid redundant model loads
        shared_reranker = (
            (retriever.reranker if retriever else None)
            or (grader.reranker if grader else None)
            or (refiner.reranker if refiner else None)
            or CrossEncoderReranker()
        )
        self.retriever = retriever or HybridRetriever(reranker=shared_reranker)
        self.grader = grader or DocumentGrader(reranker=shared_reranker)
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.web_search = web_search or WebSearchEngine()
        self.refiner = refiner or KnowledgeRefiner(reranker=shared_reranker)

    def run(
        self,
        query: str,
        top_k: int = 5,
        force_action: Optional[CRAGAction] = None,
    ) -> CRAGContext:
        """Executes the complete Corrective RAG workflow.

        Args:
            query: User prompt or question.
            top_k: Number of hybrid retrieval candidates.
            force_action: Optional override for testing specific branches.

        Returns:
            CRAGContext containing action, confidence, grading result, and verified knowledge.
        """
        start_time = time.perf_counter()
        logger.info(f"=== Starting CRAG Pipeline for query: '{query}' ===")

        # 1. Retrieve candidates via Hybrid Search
        search_result = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            use_reranker=True,
        )
        candidate_chunks: List[DocumentChunk] = [r.chunk for r in search_result.results]
        precomputed_scores: List[float] = [r.score for r in search_result.results]

        # 2. Evaluate candidates with DocumentGrader
        grading_result: GradingResult = self.grader.grade_chunks(
            query=query,
            chunks=candidate_chunks,
            precomputed_scores=precomputed_scores,
        )

        action = force_action or grading_result.overall_action
        confidence = grading_result.confidence
        logger.info(f"CRAG Action Selected: [{action.value}] (Confidence: {confidence:.3f})")

        internal_knowledge: List[RefinedKnowledge] = []
        web_knowledge: List[RefinedKnowledge] = []
        transformed_query: Optional[str] = None

        # 3. Branching Decision Logic
        if action == CRAGAction.CORRECT:
            # Internal retrieval is sufficient: refine internal chunks, no external search needed
            internal_knowledge = self.refiner.refine_internal_chunks(
                query=query,
                chunks=grading_result.relevant_chunks,
            )

        elif action == CRAGAction.AMBIGUOUS:
            # Ambiguous/incomplete retrieval: keep relevant internal chunks AND fetch external web info
            internal_knowledge = self.refiner.refine_internal_chunks(
                query=query,
                chunks=grading_result.relevant_chunks,
            )
            transformed_query = self.query_rewriter.rewrite(query)
            web_results = self.web_search.search(transformed_query)
            web_knowledge = self.refiner.refine_web_results(web_results)

        elif action == CRAGAction.INCORRECT:
            # Low confidence/irrelevant retrieval: discard internal docs to prevent hallucination!
            logger.warning("Internal retrieval deemed INCORRECT. Triggering full web fallback.")
            transformed_query = self.query_rewriter.rewrite(query)
            web_results = self.web_search.search(transformed_query)
            web_knowledge = self.refiner.refine_web_results(web_results)

        # 4. Synthesize final consolidated context text
        final_context_blocks: List[str] = []

        if internal_knowledge:
            final_context_blocks.append("### Verified Internal Knowledge:")
            for idx, item in enumerate(internal_knowledge, start=1):
                provenance = f" [{item.provenance}]" if item.provenance else ""
                final_context_blocks.append(f"{idx}.{provenance} {item.content}")

        if web_knowledge:
            final_context_blocks.append("\n### External Web Search Ground Truth:")
            for idx, item in enumerate(web_knowledge, start=1):
                provenance = f" [{item.provenance}]" if item.provenance else ""
                final_context_blocks.append(f"{idx}.{provenance} {item.content}")

        final_context_text = "\n".join(final_context_blocks)
        duration = time.perf_counter() - start_time
        logger.info(f"=== CRAG Pipeline finished in {duration:.3f}s (Action: {action.value}) ===")

        return CRAGContext(
            query=query,
            transformed_query=transformed_query,
            action=action,
            confidence=confidence,
            grading_result=grading_result,
            internal_knowledge=internal_knowledge,
            web_knowledge=web_knowledge,
            final_context_text=final_context_text,
        )
