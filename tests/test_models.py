"""Unit tests for document data models."""

from veritas_crag.models.document import (
    DocumentChunk,
    DocumentMetadata,
    DocumentTable,
    ParsedDocument,
)


def test_document_metadata_instantiation():
    meta = DocumentMetadata(
        source_path="/path/to/test.pdf",
        file_name="test.pdf",
        file_size_bytes=1024,
        num_pages=3,
        title="Test Document",
        content_hash="abc123hash",
    )
    assert meta.file_name == "test.pdf"
    assert meta.num_pages == 3
    assert meta.content_hash == "abc123hash"
    assert meta.parsed_at is not None


def test_document_table_model():
    table = DocumentTable(
        table_index=0,
        page_numbers=[1],
        num_rows=3,
        num_cols=2,
        markdown="| A | B |\n|---|---|\n| 1 | 2 |",
        data=[["A", "B"], ["1", "2"]],
    )
    assert table.num_rows == 3
    assert table.num_cols == 2
    assert "| A | B |" in table.markdown


def test_document_chunk_model():
    chunk = DocumentChunk(
        chunk_id="doc1_c0001",
        document_source="test.pdf",
        text="Sample text content for retrieval testing.",
        page_numbers=[1],
        headings=["Section 1", "Subsection 1.1"],
        chunk_type="text",
    )
    assert chunk.chunk_id == "doc1_c0001"
    assert "Section 1" in chunk.headings
    assert chunk.chunk_type == "text"


def test_parsed_document_properties():
    meta = DocumentMetadata(
        source_path="/path/to/test.pdf",
        file_name="test.pdf",
        file_size_bytes=1024,
        num_pages=1,
    )
    table = DocumentTable(
        table_index=0,
        markdown="| Col1 | Col2 |",
    )
    chunk = DocumentChunk(
        chunk_id="c0",
        document_source="test.pdf",
        text="text",
    )
    doc = ParsedDocument(
        metadata=meta,
        full_text="text",
        markdown="# Title\ntext",
        tables=[table],
        chunks=[chunk],
    )
    assert doc.total_tables == 1
    assert doc.total_chunks == 1
    assert doc.metadata.file_name == "test.pdf"
