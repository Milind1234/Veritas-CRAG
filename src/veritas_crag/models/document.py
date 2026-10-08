"""Pydantic data models for Document Intelligence & RAG."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentTable(BaseModel):
    """Structured representation of an extracted table."""

    table_index: int
    page_numbers: List[int] = Field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0
    markdown: str
    html: Optional[str] = None
    data: Optional[List[List[Any]]] = None


class DocumentMetadata(BaseModel):
    """Metadata of an ingested document."""

    source_path: str
    file_name: str
    file_size_bytes: int = 0
    num_pages: int = 0
    title: Optional[str] = None
    content_hash: Optional[str] = None
    parsed_at: datetime = Field(default_factory=datetime.utcnow)
    parsing_duration_seconds: Optional[float] = None
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """A granular chunk of a document ready for vector embedding & retrieval."""

    chunk_id: str
    document_source: str
    text: str
    page_numbers: List[int] = Field(default_factory=list)
    headings: List[str] = Field(
        default_factory=list,
        description="Hierarchy of headings under which this chunk falls (breadcrumbs)."
    )
    token_count: Optional[int] = None
    chunk_type: str = Field(
        default="text",
        description="Type of chunk: text, table, code, or mixed."
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    """Complete parsed document output from Docling."""

    metadata: DocumentMetadata
    full_text: str
    markdown: str
    tables: List[DocumentTable] = Field(default_factory=list)
    chunks: List[DocumentChunk] = Field(default_factory=list)

    @property
    def total_tables(self) -> int:
        return len(self.tables)

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)
