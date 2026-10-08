"""Document Ingestion package."""

from veritas_crag.ingestion.chunker import DoclingChunker
from veritas_crag.ingestion.parser import SUPPORTED_EXTENSIONS, DoclingParser

__all__ = [
    "DoclingParser",
    "DoclingChunker",
    "SUPPORTED_EXTENSIONS",
]
