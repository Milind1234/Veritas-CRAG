"""Unit and integration tests for DoclingParser."""

from pathlib import Path
import pytest

from veritas_crag.core.exceptions import (
    DocumentNotFoundError,
    UnsupportedDocumentFormatError,
)
from veritas_crag.ingestion.parser import DoclingParser


def test_parser_file_not_found(parser: DoclingParser):
    with pytest.raises(DocumentNotFoundError):
        parser.parse("non_existent_file.pdf")


def test_parser_unsupported_format(parser: DoclingParser, tmp_path: Path):
    invalid_file = tmp_path / "test.xyz"
    invalid_file.write_text("dummy content")

    with pytest.raises(UnsupportedDocumentFormatError):
        parser.parse(invalid_file)


def test_parser_successful_pdf_parse(parser: DoclingParser, sample_pdf_path: Path, tmp_path: Path):
    parsed_doc = parser.parse(sample_pdf_path, chunk=True)

    assert parsed_doc is not None
    assert parsed_doc.metadata.num_pages >= 1
    assert parsed_doc.metadata.file_name == sample_pdf_path.name
    assert parsed_doc.metadata.content_hash is not None
    assert len(parsed_doc.markdown) > 0
    assert len(parsed_doc.chunks) > 0

    # Verify Table Extraction
    assert parsed_doc.total_tables >= 1
    table = parsed_doc.tables[0]
    assert "Revenue" in table.markdown or "EBITDA" in table.markdown

    # Verify Export to Disk
    export_dir = parser.export_to_disk(
        parsed_doc=parsed_doc,
        output_dir=tmp_path / "export_test",
        export_markdown=True,
        export_json=True,
        export_chunks=True,
    )
    assert (export_dir / f"{sample_pdf_path.stem}.md").exists()
    assert (export_dir / f"{sample_pdf_path.stem}_parsed.json").exists()
    assert (export_dir / f"{sample_pdf_path.stem}_chunks.json").exists()
