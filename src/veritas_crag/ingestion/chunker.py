"""Document chunking strategies for layout-aware RAG."""

import hashlib
from typing import List, Optional
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DoclingDocument
from loguru import logger
from transformers import AutoTokenizer

from veritas_crag.config.settings import settings
from veritas_crag.core.exceptions import ChunkingError
from veritas_crag.models.document import DocumentChunk


class DoclingChunker:
    """Production-grade layout-aware chunker utilizing Docling's HybridChunker."""

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        merge_peers: Optional[bool] = None,
        repeat_table_header: Optional[bool] = None,
        tokenizer_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.max_tokens = max_tokens or settings.MAX_CHUNK_TOKENS
        self.merge_peers = (
            merge_peers if merge_peers is not None else settings.CHUNK_MERGE_PEERS
        )
        self.repeat_table_header = (
            repeat_table_header
            if repeat_table_header is not None
            else settings.REPEAT_TABLE_HEADER
        )

        try:
            raw_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
            hf_tokenizer = HuggingFaceTokenizer(
                tokenizer=raw_tokenizer,
                max_tokens=self.max_tokens,
            )
            self._chunker = HybridChunker(
                tokenizer=hf_tokenizer,
                merge_peers=self.merge_peers,
                repeat_table_header=self.repeat_table_header,
            )
        except Exception as e:
            logger.error(f"Failed to initialize HybridChunker: {e}")
            raise ChunkingError(f"Failed to initialize HybridChunker: {e}") from e

    def chunk_document(
        self,
        docling_doc: DoclingDocument,
        doc_source: str,
        doc_hash: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Splits a parsed DoclingDocument into structured DocumentChunks."""
        try:
            raw_chunks = list(self._chunker.chunk(docling_doc))
            chunks: List[DocumentChunk] = []

            for idx, raw_chunk in enumerate(raw_chunks):
                # Extract page numbers from provenance items
                page_numbers: List[int] = []
                if hasattr(raw_chunk.meta, "doc_items") and raw_chunk.meta.doc_items:
                    for item in raw_chunk.meta.doc_items:
                        if hasattr(item, "prov") and item.prov:
                            for prov in item.prov:
                                if hasattr(prov, "page_no") and prov.page_no not in page_numbers:
                                    page_numbers.append(prov.page_no)

                # Extract heading breadcrumbs
                headings = list(getattr(raw_chunk.meta, "headings", []) or [])

                # Determine chunk type (table vs text)
                chunk_type = "text"
                text_content = raw_chunk.text
                if any(getattr(item, "label", None) == "table" for item in getattr(raw_chunk.meta, "doc_items", [])):
                    chunk_type = "table"

                # Generate a stable unique chunk ID
                hash_base = doc_hash or hashlib.sha256(doc_source.encode()).hexdigest()[:12]
                chunk_id = f"{hash_base}_c{idx:04d}"

                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    document_source=doc_source,
                    text=text_content,
                    page_numbers=page_numbers,
                    headings=headings,
                    token_count=len(text_content.split()),  # approximate or tokenizer based
                    chunk_type=chunk_type,
                    metadata={
                        "chunk_index": idx,
                        "total_chunks": len(raw_chunks),
                    },
                )
                chunks.append(chunk)

            logger.info(
                f"Generated {len(chunks)} layout-aware chunks for document: {doc_source}"
            )
            return chunks

        except Exception as e:
            logger.error(f"Error while chunking document '{doc_source}': {e}")
            raise ChunkingError(f"Failed to chunk document '{doc_source}': {e}") from e
