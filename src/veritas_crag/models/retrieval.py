"""Pydantic data models for Retrieval, Hybrid Search, and Reranking."""

from typing import List, Optional
from pydantic import BaseModel, Field

from veritas_crag.models.document import DocumentChunk


class RetrievalResult(BaseModel):
    """A scored retrieval result from dense, sparse, hybrid, or reranked search."""

    chunk: DocumentChunk
    score: float = Field(..., description="Final combined or primary retrieval score")
    dense_score: Optional[float] = Field(default=None, description="Raw cosine similarity score")
    dense_rank: Optional[int] = Field(default=None, description="1-indexed rank from dense search")
    sparse_score: Optional[float] = Field(default=None, description="Raw BM25 score")
    sparse_rank: Optional[int] = Field(default=None, description="1-indexed rank from sparse search")
    rrf_score: Optional[float] = Field(default=None, description="Reciprocal Rank Fusion score")
    rerank_score: Optional[float] = Field(
        default=None, description="Cross-encoder semantic relevance score"
    )

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def headings(self) -> List[str]:
        return self.chunk.headings

    @property
    def page_numbers(self) -> List[int]:
        return self.chunk.page_numbers


class HybridSearchResult(BaseModel):
    """Encapsulates results from a hybrid retrieval query."""

    query: str
    results: List[RetrievalResult] = Field(default_factory=list)
    total_candidates: int = 0
    duration_seconds: float = 0.0
