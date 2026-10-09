"""LangGraph State Graph orchestrating the complete Corrective RAG pipeline."""

import time
from typing import Any, Dict, List, Optional, TypedDict
from langgraph.graph import END, START, StateGraph
from loguru import logger

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
from veritas_crag.models.pipeline import Citation, GeneratedResponse, GroundingEvaluation
from veritas_crag.pipeline.generator import ResponseGenerator
from veritas_crag.pipeline.grounding import GroundingGrader
from veritas_crag.retrieval.hybrid import HybridRetriever
from veritas_crag.retrieval.reranker import CrossEncoderReranker


class CRAGGraphState(TypedDict):
    """The central state dictionary passed between LangGraph nodes."""

    query: str
    top_k: int
    candidate_chunks: List[DocumentChunk]
    grading_result: Optional[GradingResult]
    action: Optional[CRAGAction]
    transformed_query: Optional[str]
    internal_knowledge: List[RefinedKnowledge]
    web_knowledge: List[RefinedKnowledge]
    answer: Optional[str]
    citations: List[Citation]
    grounding: Optional[GroundingEvaluation]
    start_time: float
    total_latency: float


class VeritasCRAGWorkflow:
    """LangGraph State Graph orchestrating the Corrective RAG state machine."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        grader: Optional[DocumentGrader] = None,
        query_rewriter: Optional[QueryRewriter] = None,
        web_search: Optional[WebSearchEngine] = None,
        refiner: Optional[KnowledgeRefiner] = None,
        generator: Optional[ResponseGenerator] = None,
        grounding_grader: Optional[GroundingGrader] = None,
    ):
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
        self.generator = generator or ResponseGenerator()
        self.grounding_grader = grounding_grader or GroundingGrader(reranker=shared_reranker)

        # Build and compile LangGraph
        self.graph = self._build_graph()

    def _retrieve_node(self, state: CRAGGraphState) -> Dict[str, Any]:
        """Node 1: Hybrid retrieval (Dense Qdrant + Sparse BM25 + Cross-Encoder Reranker)."""
        logger.info(f"[LangGraph Node: retrieve] Query: '{state['query']}'")
        res = self.retriever.retrieve(query=state["query"], top_k=state["top_k"], use_reranker=True)
        chunks = [r.chunk for r in res.results]
        return {"candidate_chunks": chunks}

    def _grade_documents_node(self, state: CRAGGraphState) -> Dict[str, Any]:
        """Node 2: Retrieval Evaluator checking candidate document relevance."""
        logger.info(f"[LangGraph Node: grade_documents] Candidates: {len(state['candidate_chunks'])}")
        grading = self.grader.grade_chunks(query=state["query"], chunks=state["candidate_chunks"])
        logger.info(f"[LangGraph Action Decision]: {grading.overall_action.value}")
        return {
            "grading_result": grading,
            "action": grading.overall_action,
        }

    def _refine_internal_node(self, state: CRAGGraphState) -> Dict[str, Any]:
        """Node 3: Knowledge refinement on relevant internal chunks."""
        logger.info("[LangGraph Node: refine_internal]")
        relevant = state["grading_result"].relevant_chunks if state["grading_result"] else []
        refined = self.refiner.refine_internal_chunks(query=state["query"], chunks=relevant)
        return {"internal_knowledge": refined}

    def _web_fallback_node(self, state: CRAGGraphState) -> Dict[str, Any]:
        """Node 4: Query transformation and external web search fallback."""
        logger.info("[LangGraph Node: web_fallback]")
        transformed = self.query_rewriter.rewrite(state["query"])
        web_res = self.web_search.search(transformed)
        refined_web = self.refiner.refine_web_results(web_res)
        return {
            "transformed_query": transformed,
            "web_knowledge": refined_web,
        }

    def _generate_node(self, state: CRAGGraphState) -> Dict[str, Any]:
        """Node 5: Grounded answer synthesis with inline citations."""
        logger.info("[LangGraph Node: generate]")
        # Construct temporary CRAGContext for generator
        context = CRAGContext(
            query=state["query"],
            transformed_query=state.get("transformed_query"),
            action=state["action"] or CRAGAction.CORRECT,
            confidence=state["grading_result"].confidence if state["grading_result"] else 1.0,
            grading_result=state["grading_result"],
            internal_knowledge=state.get("internal_knowledge", []),
            web_knowledge=state.get("web_knowledge", []),
            final_context_text="",
        )
        answer, citations = self.generator.generate(context)
        return {"answer": answer, "citations": citations}

    def _verify_grounding_node(self, state: CRAGGraphState) -> Dict[str, Any]:
        """Node 6: Faithfulness & Hallucination verification."""
        logger.info("[LangGraph Node: verify_grounding]")
        grounding = self.grounding_grader.evaluate(
            query=state["query"],
            answer=state["answer"] or "",
            citations=state["citations"],
        )
        total_time = round(time.perf_counter() - state["start_time"], 3)
        return {"grounding": grounding, "total_latency": total_time}

    @staticmethod
    def _route_after_grading(state: CRAGGraphState) -> str:
        """Conditional routing based on the evaluator action."""
        action = state.get("action")
        if action == CRAGAction.CORRECT:
            return "refine_internal"
        elif action == CRAGAction.AMBIGUOUS:
            return "web_fallback_and_refine_both"
        else:  # INCORRECT
            return "web_fallback_only"

    def _build_graph(self):
        """Constructs the LangGraph StateGraph topology."""
        builder = StateGraph(CRAGGraphState)

        # Register nodes
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("grade_documents", self._grade_documents_node)
        builder.add_node("refine_internal", self._refine_internal_node)
        builder.add_node("web_fallback", self._web_fallback_node)
        builder.add_node("generate", self._generate_node)
        builder.add_node("verify_grounding", self._verify_grounding_node)

        # Define edge routing
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "grade_documents")

        # Conditional branching from grade_documents
        builder.add_conditional_edges(
            "grade_documents",
            self._route_after_grading,
            {
                "refine_internal": "refine_internal",
                "web_fallback_and_refine_both": "web_fallback",
                "web_fallback_only": "web_fallback",
            },
        )

        # In AMBIGUOUS case, web_fallback continues to refine_internal to merge both
        # In INCORRECT case, web_fallback goes straight to generate
        def _route_after_web_fallback(state: CRAGGraphState) -> str:
            if state.get("action") == CRAGAction.AMBIGUOUS:
                return "refine_internal"
            return "generate"

        builder.add_conditional_edges(
            "web_fallback",
            _route_after_web_fallback,
            {
                "refine_internal": "refine_internal",
                "generate": "generate",
            },
        )

        builder.add_edge("refine_internal", "generate")
        builder.add_edge("generate", "verify_grounding")
        builder.add_edge("verify_grounding", END)

        return builder.compile()

    def run(self, query: str, top_k: int = 5) -> GeneratedResponse:
        """Executes the full LangGraph workflow and returns a typed GeneratedResponse."""
        initial_state: CRAGGraphState = {
            "query": query,
            "top_k": top_k,
            "candidate_chunks": [],
            "grading_result": None,
            "action": None,
            "transformed_query": None,
            "internal_knowledge": [],
            "web_knowledge": [],
            "answer": None,
            "citations": [],
            "grounding": None,
            "start_time": time.perf_counter(),
            "total_latency": 0.0,
        }

        final_state = self.graph.invoke(initial_state)

        return GeneratedResponse(
            query=query,
            answer=final_state["answer"] or "",
            action=final_state["action"] or CRAGAction.CORRECT,
            confidence=final_state["grading_result"].confidence if final_state["grading_result"] else 1.0,
            citations=final_state["citations"],
            grounding=final_state["grounding"],
            total_latency_seconds=final_state["total_latency"],
        )
