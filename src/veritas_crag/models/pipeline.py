"""Pydantic data models for Generation, Grounding Verification, and API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from veritas_crag.models.crag import CRAGAction
from veritas_crag.models.document import DocumentChunk, DocumentTable


class Citation(BaseModel):
    """An exact source citation referenced in the generated answer."""

    index: int = Field(..., description="Citation marker e.g. [1], [2]")
    source_type: str = Field(..., description="'internal' or 'web'")
    provenance: str = Field(..., description="Document source + breadcrumb or website URL")
    snippet: str = Field(..., description="Exact context passage supporting this citation")


class GroundingEvaluation(BaseModel):
    """Safety and hallucination evaluation of the generated response."""

    is_grounded: bool = Field(..., description="True if answer is fully supported by context")
    grounding_score: float = Field(
        ..., ge=0.0, le=1.0, description="Faithfulness / entailment score [0.0 - 1.0]"
    )
    hallucination_detected: bool = Field(
        default=False, description="True if any unsupported claims are detected"
    )
    hallucinated_claims: List[str] = Field(
        default_factory=list, description="List of unverified claims"
    )
    answers_query: bool = Field(
        default=True, description="Whether the response directly addresses user query"
    )
    rationale: Optional[str] = Field(
        default=None, description="Detailed explanation of the grounding evaluation"
    )


class GeneratedResponse(BaseModel):
    """Final, fully verified response output of the VeritasCRAG pipeline."""

    query: str
    answer: str
    action: CRAGAction
    confidence: float
    citations: List[Citation] = Field(default_factory=list)
    grounding: GroundingEvaluation
    total_latency_seconds: float = 0.0


# FastAPI Request & Response Models
class QueryRequest(BaseModel):
    """Incoming search/question query."""

    query: str = Field(..., min_length=2, description="User search query or question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of candidate chunks")


class QueryResponse(BaseModel):
    """JSON response returned by the VeritasCRAG API."""

    query: str
    answer: str
    action: str
    confidence: float
    is_grounded: bool
    grounding_score: float
    citations: List[Citation]
    latency_seconds: float


class IngestResponse(BaseModel):
    """JSON response returned by document upload/ingestion endpoint."""

    file_name: str
    file_size_kb: float
    num_pages: int
    total_tables: int
    total_chunks: int
    parsing_duration_seconds: float
    markdown: Optional[str] = Field(default="", description="Full extracted document Markdown")
    tables: Optional[List[DocumentTable]] = Field(default_factory=list, description="Extracted tables")
    chunks: Optional[List[DocumentChunk]] = Field(default_factory=list, description="Generated hybrid chunks")

