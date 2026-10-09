"""Unit and integration tests for Phase 3: Corrective RAG (CRAG) decision engine."""

from typing import List
from unittest.mock import MagicMock, patch
import pytest

from veritas_crag.evaluation.crag_engine import CRAGEngine
from veritas_crag.evaluation.grader import DocumentGrader
from veritas_crag.evaluation.query_rewriter import QueryRewriter
from veritas_crag.evaluation.refiner import KnowledgeRefiner
from veritas_crag.evaluation.web_search import WebSearchEngine
from veritas_crag.models.crag import CRAGAction, WebSearchResult
from veritas_crag.models.document import DocumentChunk
from veritas_crag.retrieval.reranker import CrossEncoderReranker


@pytest.fixture(scope="session")
def shared_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()


@pytest.fixture
def test_chunks() -> List[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="chunk_crag_01",
            document_source="system_arch.pdf",
            text="The Docling parser achieves 180ms latency on Apple Silicon M4 with MPS hardware acceleration.",
            headings=["Performance", "Latency"],
            page_numbers=[1],
            chunk_type="text",
        ),
        DocumentChunk(
            chunk_id="chunk_crag_02",
            document_source="system_arch.pdf",
            text="Component, Latency, Accuracy. Docling, 180ms, 98.5%. CRAG Grader, 250ms, 94.2%.",
            headings=["Performance", "Benchmark Table"],
            page_numbers=[1],
            chunk_type="table",
        ),
    ]


def test_query_rewriter():
    rewriter = QueryRewriter()
    # Conversational question strip
    res1 = rewriter.rewrite("Can you please tell me what is the Docling latency on M4?")
    assert "can you please tell me" not in res1.lower()
    assert "docling latency on m4" in res1.lower()

    # Context hint addition
    res2 = rewriter.rewrite("performance benchmarks", context_hint="Apple M4")
    assert "Apple M4" in res2


def test_document_grader_correct_action(shared_reranker: CrossEncoderReranker, test_chunks: List[DocumentChunk]):
    grader = DocumentGrader(reranker=shared_reranker)
    query = "What is the Docling latency on Apple Silicon M4?"
    grading = grader.grade_chunks(query=query, chunks=test_chunks)

    assert grading.overall_action == CRAGAction.CORRECT
    assert grading.confidence >= 0.65
    assert len(grading.relevant_chunks) >= 1


def test_document_grader_incorrect_action(shared_reranker: CrossEncoderReranker, test_chunks: List[DocumentChunk]):
    grader = DocumentGrader(reranker=shared_reranker)
    query = "Who was the Roman emperor during the eruption of Mount Vesuvius in 79 AD?"
    grading = grader.grade_chunks(query=query, chunks=test_chunks)

    assert grading.overall_action == CRAGAction.INCORRECT
    assert grading.confidence < 0.30
    assert len(grading.relevant_chunks) == 0


def test_knowledge_refiner(shared_reranker: CrossEncoderReranker, test_chunks: List[DocumentChunk]):
    refiner = KnowledgeRefiner(reranker=shared_reranker)
    query = "Docling parser latency"
    refined_internal = refiner.refine_internal_chunks(query=query, chunks=test_chunks)

    assert len(refined_internal) == 2
    # Table chunk structure must be preserved
    table_item = next(item for item in refined_internal if "Benchmark Table" in (item.provenance or ""))
    assert "180ms" in table_item.content

    # Web results refinement
    web_res = [
        WebSearchResult(
            title="Docling IBM Docs",
            url="https://docling.dev",
            snippet="IBM Docling parses complex layout formats into markdown.",
        )
    ]
    refined_web = refiner.refine_web_results(web_res)
    assert len(refined_web) == 1
    assert refined_web[0].source_type == "web"
    assert "https://docling.dev" in (refined_web[0].provenance or "")


def test_web_search_engine_mock():
    engine = WebSearchEngine()
    fake_results = [
        WebSearchResult(
            title="Apple M4 Details",
            url="https://apple.com/m4",
            snippet="Apple M4 features high performance neural engine cores.",
        )
    ]
    with patch.object(engine, "_search_duckduckgo", return_value=fake_results):
        results = engine.search("Apple M4")
        assert len(results) == 1
        assert results[0].title == "Apple M4 Details"


def test_crag_engine_correct_flow(shared_reranker: CrossEncoderReranker):
    engine = CRAGEngine(grader=DocumentGrader(reranker=shared_reranker))
    context = engine.run(query="What is the Docling latency on M4?", top_k=2)

    assert context.action == CRAGAction.CORRECT
    assert context.confidence >= 0.65
    assert len(context.internal_knowledge) >= 1
    assert len(context.web_knowledge) == 0  # No web search needed for CORRECT!
    assert "Verified Internal Knowledge" in context.final_context_text


def test_crag_engine_incorrect_flow(shared_reranker: CrossEncoderReranker):
    engine = CRAGEngine(grader=DocumentGrader(reranker=shared_reranker))
    # Query completely outside the sample report
    context = engine.run(query="Who won the UEFA Champions League in 2024?", top_k=2)

    assert context.action == CRAGAction.INCORRECT
    assert len(context.internal_knowledge) == 0  # Internal discarded to prevent hallucination!
    assert context.transformed_query is not None
