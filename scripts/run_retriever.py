"""CLI utility for indexing and querying VeritasCRAG hybrid retrieval."""

import json
from pathlib import Path
from typing import List, Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from veritas_crag.config.settings import settings
from veritas_crag.models.document import DocumentChunk
from veritas_crag.retrieval.bm25 import BM25Index
from veritas_crag.retrieval.hybrid import HybridRetriever
from veritas_crag.retrieval.vector_store import QdrantVectorStore

app = typer.Typer(
    name="veritas-retriever",
    help="VeritasCRAG Hybrid Indexing & Retrieval CLI",
    add_completion=False,
)
console = Console()


@app.command()
def index(
    chunks_file: Optional[Path] = typer.Option(
        None,
        "--file",
        "-f",
        help="Path to specific chunks JSON file (e.g. data/output/sample_report_chunks.json)",
    ),
    input_dir: Optional[Path] = typer.Option(
        None,
        "--dir",
        "-d",
        help="Directory containing *_chunks.json files",
    ),
):
    """Index document chunks into Dense Qdrant and Sparse BM25 stores."""
    chunks_to_index: List[DocumentChunk] = []

    target_files: List[Path] = []
    if chunks_file and chunks_file.exists():
        target_files.append(chunks_file)
    else:
        search_dir = input_dir or settings.OUTPUT_DIR
        target_files = list(search_dir.glob("*_chunks.json"))

    if not target_files:
        console.print(
            f"[bold red]No chunks files found in {search_dir}. "
            f"Run 'python scripts/run_parser.py parse <pdf>' first![/bold red]"
        )
        raise typer.Exit(1)

    for cf in target_files:
        with open(cf, "r", encoding="utf-8") as f:
            data = json.load(f)
            chunks = [DocumentChunk.model_validate(c) for c in data]
            chunks_to_index.extend(chunks)

    console.print(
        Panel.fit(
            f"[bold cyan]VeritasCRAG Indexing[/bold cyan]\n"
            f"Found [bold green]{len(chunks_to_index)}[/bold green] chunks across {len(target_files)} file(s).\n"
            f"Dense Vector Store: [magenta]Qdrant ({settings.QDRANT_STORAGE_PATH})[/magenta]\n"
            f"Sparse Lexical Store: [yellow]BM25 ({settings.BM25_STORAGE_PATH})[/yellow]",
            title="Indexing Pipeline",
        )
    )

    retriever = HybridRetriever()
    with console.status("[bold green]Generating embeddings and building hybrid indices..."):
        indexed_count = retriever.index_chunks(chunks_to_index)

    console.print(f"[bold green]✓ Successfully indexed {indexed_count} chunks![/bold green]\n")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query string"),
    mode: str = typer.Option(
        "hybrid",
        "--mode",
        "-m",
        help="Retrieval mode: hybrid, dense, or sparse",
    ),
    rerank: bool = typer.Option(
        True,
        "--rerank/--no-rerank",
        help="Apply Cross-Encoder reranking on retrieved candidates",
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        "-k",
        help="Number of results to return",
    ),
):
    """Query the hybrid index using dense, sparse, or combined retrieval."""
    console.print(
        Panel.fit(
            f"Query: [bold cyan]\"{query}\"[/bold cyan]\n"
            f"Mode: [yellow]{mode.upper()}[/yellow] | Rerank: [magenta]{rerank}[/magenta] | Top-K: [green]{top_k}[/green]",
            title="Search Execution",
        )
    )

    if mode.lower() == "dense":
        vstore = QdrantVectorStore()
        with console.status("Querying Dense Vector Store..."):
            raw_results = vstore.search(query=query, top_k=top_k)

        table = Table(title=f"Dense Vector Search Results for '{query}'", show_header=True)
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Cosine Score", style="green", width=14)
        table.add_column("Headings", style="magenta", width=25)
        table.add_column("Content Snippet", style="white")

        for rank, (chunk, score) in enumerate(raw_results, start=1):
            table.add_row(
                str(rank),
                f"{score:.4f}",
                " > ".join(chunk.headings) if chunk.headings else "Root",
                chunk.text[:140] + ("..." if len(chunk.text) > 140 else ""),
            )
        console.print(table)

    elif mode.lower() == "sparse":
        bm25 = BM25Index()
        loaded = bm25.load()
        if not loaded:
            console.print("[bold red]BM25 index not found. Run 'index' command first![/bold red]")
            raise typer.Exit(1)

        with console.status("Querying BM25 Index..."):
            raw_results = bm25.search(query=query, top_k=top_k)

        table = Table(title=f"BM25 Lexical Search Results for '{query}'", show_header=True)
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("BM25 Score", style="yellow", width=12)
        table.add_column("Headings", style="magenta", width=25)
        table.add_column("Content Snippet", style="white")

        for rank, (chunk, score) in enumerate(raw_results, start=1):
            table.add_row(
                str(rank),
                f"{score:.4f}",
                " > ".join(chunk.headings) if chunk.headings else "Root",
                chunk.text[:140] + ("..." if len(chunk.text) > 140 else ""),
            )
        console.print(table)

    else:  # hybrid
        retriever = HybridRetriever()
        bm25_loaded = retriever.bm25_index.load()
        if not bm25_loaded:
            console.print("[bold yellow]Warning: BM25 index not loaded from disk. Did you run 'index'?[/bold yellow]")

        with console.status("Executing Hybrid Retrieval (Dense + Sparse + RRF + Reranker)..."):
            search_result = retriever.retrieve(
                query=query,
                top_k=top_k,
                use_reranker=rerank,
            )

        table = Table(
            title=f"Hybrid Retrieval Results (Duration: {search_result.duration_seconds:.3f}s)",
            show_header=True,
        )
        table.add_column("Rank", style="cyan", width=5)
        table.add_column("Final Score", style="bold green", width=12)
        table.add_column("Dense (R/S)", style="blue", width=14)
        table.add_column("BM25 (R/S)", style="yellow", width=14)
        table.add_column("Headings", style="magenta", width=25)
        table.add_column("Content Snippet", style="white")

        for rank, res in enumerate(search_result.results, start=1):
            dense_info = (
                f"#{res.dense_rank} ({res.dense_score:.3f})"
                if res.dense_rank is not None
                else "-"
            )
            bm25_info = (
                f"#{res.sparse_rank} ({res.sparse_score:.3f})"
                if res.sparse_rank is not None
                else "-"
            )
            table.add_row(
                str(rank),
                f"{res.score:.4f}",
                dense_info,
                bm25_info,
                " > ".join(res.headings) if res.headings else "Root",
                res.text[:140] + ("..." if len(res.text) > 140 else ""),
            )

        console.print(table)


if __name__ == "__main__":
    app()
