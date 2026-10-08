"""Pytest fixtures for VeritasCRAG tests."""

from pathlib import Path
import pytest
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

from veritas_crag.ingestion.parser import DoclingParser


@pytest.fixture(scope="session")
def sample_pdf_path(tmp_path_factory) -> Path:
    """Creates a deterministic multi-page sample PDF with headers and tables for testing."""
    fn = tmp_path_factory.mktemp("test_docs") / "test_report.pdf"

    doc = SimpleDocTemplate(str(fn), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("VeritasCRAG Automated Test Document", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("1. Executive Summary", styles["Heading1"]))
    story.append(
        Paragraph(
            "This document tests document intelligence layout analysis, table extraction, and chunking.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 12))
    story.append(Paragraph("2. Financial Performance Table", styles["Heading1"]))

    table_data = [
        ["Quarter", "Revenue ($M)", "EBITDA ($M)", "Growth (%)"],
        ["Q1", "120.5", "35.2", "+12%"],
        ["Q2", "145.0", "42.1", "+20%"],
        ["Q3", "160.8", "48.5", "+11%"],
    ]
    t = Table(table_data)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 1, colors.grey),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 12))
    story.append(Paragraph("3. Recommendations", styles["Heading1"]))
    story.append(
        Paragraph(
            "Adopt Corrective RAG with dynamic web search fallback when document confidence score drops below 0.7.",
            styles["Normal"],
        )
    )

    doc.build(story)
    return fn


@pytest.fixture(scope="session")
def parser() -> DoclingParser:
    """Provides a configured DoclingParser instance."""
    return DoclingParser(enable_ocr=False, enable_table_structure=True)
