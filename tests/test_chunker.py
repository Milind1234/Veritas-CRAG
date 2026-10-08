"""Unit tests for layout-aware chunking."""

from pathlib import Path
from veritas_crag.ingestion.chunker import DoclingChunker
from veritas_crag.ingestion.parser import DoclingParser


def test_chunker_structure_and_breadcrumbs(parser: DoclingParser, sample_pdf_path: Path):
    conversion_res = parser._converter.convert(sample_pdf_path)
    dl_doc = conversion_res.document

    chunker = DoclingChunker(max_tokens=256)
    chunks = chunker.chunk_document(
        docling_doc=dl_doc,
        doc_source=str(sample_pdf_path),
        doc_hash="test_hash_123",
    )

    assert len(chunks) > 0

    # Ensure deterministic ID formatting
    for chunk in chunks:
        assert chunk.chunk_id.startswith("test_hash_123_c")
        assert len(chunk.text) > 0
        assert chunk.document_source == str(sample_pdf_path)

    # Check for table chunk detection
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) >= 1

    # Check that headings breadcrumbs are captured
    chunks_with_headings = [c for c in chunks if len(c.headings) > 0]
    assert len(chunks_with_headings) >= 1
