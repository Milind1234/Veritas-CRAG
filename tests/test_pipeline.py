"""Unit and integration tests for Phase 4: Generation, Grounding & LangGraph Workflow."""

import pytest

from veritas_crag.models.crag import CRAGAction, CRAGContext, RefinedKnowledge
from veritas_crag.models.pipeline import Citation
from veritas_crag.pipeline.generator import ResponseGenerator
from veritas_crag.pipeline.grounding import GroundingGrader
from veritas_crag.pipeline.workflow import VeritasCRAGWorkflow
from veritas_crag.retrieval.reranker import CrossEncoderReranker


@pytest.fixture(scope="session")
def shared_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()


def test_response_generator():
    gen = ResponseGenerator(provider="local")
    context = CRAGContext(
        query="What is the Docling latency?",
        action=CRAGAction.CORRECT,
        confidence=0.95,
        grading_result=None,
        internal_knowledge=[
            RefinedKnowledge(
                source_type="internal",
                content="Docling parser latency is 180ms on Apple Silicon.",
                provenance="test_doc.pdf (p. 1)",
            )
        ],
        web_knowledge=[],
        final_context_text="",
    )
    answer, citations = gen.generate(context)
    assert len(citations) == 1
    assert "180ms" in answer
    assert citations[0].index == 1
    assert "test_doc.pdf" in citations[0].provenance


def test_grounding_grader_faithful(shared_reranker: CrossEncoderReranker):
    grader = GroundingGrader(reranker=shared_reranker)
    query = "What is the Docling latency?"
    answer = "Docling parser latency is 180ms on Apple Silicon. [1]"
    citations = [
        Citation(
            index=1,
            source_type="internal",
            provenance="report.pdf (p. 1)",
            snippet="Docling parser latency is 180ms on Apple Silicon.",
        )
    ]
    eval_res = grader.evaluate(query, answer, citations)
    assert eval_res.is_grounded is True
    assert eval_res.grounding_score >= 0.70
    assert eval_res.hallucination_detected is False


def test_grounding_grader_catches_hallucination(shared_reranker: CrossEncoderReranker):
    grader = GroundingGrader(reranker=shared_reranker)
    query = "What is the Docling latency?"
    # Fabricated answer containing fake facts completely disconnected from citation
    answer = "The Roman Empire collapsed in 476 AD under emperor Romulus Augustulus in Ravenna."
    citations = [
        Citation(
            index=1,
            source_type="internal",
            provenance="report.pdf (p. 1)",
            snippet="Docling parser latency is 180ms on Apple Silicon M4.",
        )
    ]
    eval_res = grader.evaluate(query, answer, citations)
    assert eval_res.is_grounded is False
    assert eval_res.hallucination_detected is True
    assert len(eval_res.hallucinated_claims) >= 1


def test_langgraph_workflow_end_to_end(shared_reranker: CrossEncoderReranker):
    workflow = VeritasCRAGWorkflow()
    response = workflow.run(query="What is the Docling latency on M4?", top_k=2)

    assert response.query == "What is the Docling latency on M4?"
    assert response.action == CRAGAction.CORRECT
    assert response.confidence >= 0.65
    assert len(response.citations) >= 1
    assert response.grounding.is_grounded is True
    assert response.total_latency_seconds > 0.0
