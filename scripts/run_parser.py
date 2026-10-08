"""CLI utility for running document parsing and inspection."""

from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import torch
import docling

from veritas_crag.config.settings import AcceleratorDevice, settings
from veritas_crag.ingestion.parser import DoclingParser
from veritas_crag.models.document import ParsedDocument

app = typer.Typer(
    name="veritas-crag",
    help="VeritasCRAG Document Ingestion & Parsing CLI",
    add_completion=False,
)
console = Console()


@app.command()
def info():
    """Display environment, hardware acceleration, and Docling versions."""
    table = Table(title="VeritasCRAG System & Hardware Info", show_header=True)
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    table.add_row("Docling Version", docling.__version__)
    table.add_row("PyTorch Version", torch.__version__)
    table.add_row("Apple Silicon MPS Available", str(torch.backends.mps.is_available()))
    table.add_row("Apple Silicon MPS Built", str(torch.backends.mps.is_built()))
    table.add_row("Configured Device", settings.ACCELERATOR_DEVICE.value)
    table.add_row("Default Sample Docs Path", str(settings.SAMPLE_DOCS_DIR))
    table.add_row("Default Output Path", str(settings.OUTPUT_DIR))

    console.print(table)


@app.command()
def parse(
    file_path: Path = typer.Argument(
        ...,
        help="Path to the document file (.pdf, .docx, .pptx, etc.)",
        exists=True,
    ),
    ocr: bool = typer.Option(
        False,
        "--ocr/--no-ocr",
        help="Enable OCR (recommended for scanned PDFs)",
    ),
    tables: bool = typer.Option(
        True,
        "--tables/--no-tables",
        help="Extract detailed table structures",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        "-d",
        help="Hardware accelerator: auto, mps, cuda, cpu",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to export parsed markdown, json, and chunks",
    ),
    chunk: bool = typer.Option(
        True,
        "--chunk/--no-chunk",
        help="Generate layout-aware chunks",
    ),
):
    """Parse a document using Docling and export structured representations."""
    target_output_dir = output_dir or settings.OUTPUT_DIR
    target_device = AcceleratorDevice(device.lower())

    console.print(
        Panel.fit(
            f"[bold blue]VeritasCRAG Document Ingestion[/bold blue]\n"
            f"File: [cyan]{file_path}[/cyan]\n"
            f"Device: [magenta]{target_device.value}[/magenta] | "
            f"OCR: [yellow]{ocr}[/yellow] | Tables: [green]{tables}[/green]",
            title="Ingestion Job Started",
        )
    )

    parser = DoclingParser(
        enable_ocr=ocr,
        enable_table_structure=tables,
        accelerator_device=target_device,
    )

    with console.status("[bold green]Parsing document with Docling..."):
        parsed_doc: ParsedDocument = parser.parse(file_path=file_path, chunk=chunk)

    with console.status("[bold blue]Exporting parsed artifacts..."):
        out_path = parser.export_to_disk(
            parsed_doc=parsed_doc,
            output_dir=target_output_dir,
            export_markdown=True,
            export_json=True,
            export_chunks=chunk,
        )

    # Print Summary Table
    summary_table = Table(title="Parsing Results Summary", show_header=True)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="bold green")

    meta = parsed_doc.metadata
    summary_table.add_row("Document Title", meta.title or meta.file_name)
    summary_table.add_row("File Size", f"{meta.file_size_bytes / 1024:.1f} KB")
    summary_table.add_row("Pages Count", str(meta.num_pages))
    summary_table.add_row("Duration", f"{meta.parsing_duration_seconds:.2f} s")
    summary_table.add_row("Extracted Tables", str(parsed_doc.total_tables))
    summary_table.add_row("Generated Chunks", str(parsed_doc.total_chunks))
    summary_table.add_row("Export Directory", str(out_path))

    console.print(summary_table)

    # Print Table Preview if any
    if parsed_doc.tables:
        console.print("\n[bold]Table Preview (Table 0):[/bold]")
        console.print(parsed_doc.tables[0].markdown)

    # Print First Chunk Preview if any
    if parsed_doc.chunks:
        first_chunk = parsed_doc.chunks[0]
        chunk_panel = Panel(
            f"[bold]ID:[/bold] {first_chunk.chunk_id}\n"
            f"[bold]Headings:[/bold] {' > '.join(first_chunk.headings) if first_chunk.headings else 'Root'}\n"
            f"[bold]Pages:[/bold] {first_chunk.page_numbers}\n"
            f"[bold]Type:[/bold] {first_chunk.chunk_type}\n\n"
            f"[italic]{first_chunk.text}[/italic]",
            title="Sample Chunk [0]",
            border_style="cyan",
        )
        console.print(chunk_panel)

    console.print("[bold green]✓ Ingestion job complete successfully![/bold green]\n")


if __name__ == "__main__":
    app()
