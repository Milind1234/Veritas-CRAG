"""Pydantic data models for Corrective RAG (CRAG) decision engine."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

from veritas_crag.models.document import DocumentChunk


class CRAGAction(str, Enum):
    """The corrective action determined by the retrieval evaluator."""

    CORRECT = "CORRECT"        # High confidence -> Use internal retrieved documents directly
    AMBIGUOUS = "AMBIGUOUS"    # Medium confidence -> Combine internal docs + web search fallback
    INCORRECT = "INCORRECT"    # Low confidence -> Discard internal docs, trigger full web search


class DocumentGrade(BaseModel):
    """Evaluation score and verdict for a single retrieved document chunk."""

    chunk_id: str
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized relevance score [0.0 - 1.0]")
    is_relevant: bool = Field(..., description="Whether this chunk is deemed relevant to the query")
    rationale: Optional[str] = Field(default=None, description="Explanation or reasoning for the grade")


class GradingResult(BaseModel):
    """Aggregate evaluation verdict across all candidate retrieved chunks."""

    grades: List[DocumentGrade] = Field(default_factory=list)
    overall_action: CRAGAction
    confidence: float = Field(..., ge=0.0, le=1.0, description="Aggregate confidence score")
    relevant_chunks: List[DocumentChunk] = Field(default_factory=list)
    irrelevant_chunks: List[DocumentChunk] = Field(default_factory=list)


class WebSearchResult(BaseModel):
    """An external knowledge item returned from web search fallback."""

    title: str
    url: str
    snippet: str


class RefinedKnowledge(BaseModel):
    """A granular, decomposed knowledge strip stripped of irrelevant context."""

    source_type: str = Field(..., description="'internal' document chunk or 'web' search snippet")
    content: str
    provenance: Optional[str] = Field(default=None, description="Document source or URL citation")


class CRAGContext(BaseModel):
    """Comprehensive state output of the CRAG decision & refinement engine."""

    query: str
    transformed_query: Optional[str] = None
    action: CRAGAction
    confidence: float
    grading_result: Optional[GradingResult] = None
    internal_knowledge: List[RefinedKnowledge] = Field(default_factory=list)
    web_knowledge: List[RefinedKnowledge] = Field(default_factory=list)
    final_context_text: str = Field(
        ...,
        description="Consolidated, verified knowledge ready for LLM generation in Phase 4"
    )
