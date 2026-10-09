"""CLI utility for running the end-to-end LangGraph CRAG pipeline and API server."""

import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from veritas_crag.config.settings import settings
from veritas_crag.models.pipeline import GeneratedResponse
from veritas_crag.pipeline.workflow import VeritasCRAGWorkflow

app = typer.Typer(
    name="veritas-pipeline",
    help="VeritasCRAG End-to-End LangGraph Pipeline & Serving CLI",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.command()
def query(
    query_str: str = typer.Argument(..., help="Search query or question to answer"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Candidate retrieval chunks"),
):
    """Run full LangGraph Corrective RAG pipeline: Retrieve -> Grade -> Fallback -> Generate -> Verify."""
    console.print(
        Panel.fit(
            f"Question: [bold cyan]\"{query_str}\"[/bold cyan]\n"
            f"Retrieval Pool: [yellow]{top_k}[/yellow] | Engine: [magenta]LangGraph State Graph[/magenta]",
            title="LangGraph CRAG Execution",
        )
    )

    workflow = VeritasCRAGWorkflow()
    with console.status("[bold green]Executing LangGraph State Graph..."):
        response: GeneratedResponse = workflow.run(query=query_str, top_k=top_k)

    # 1. Action & Grounding Banner
    action_color = "green" if response.action == "CORRECT" else ("yellow" if response.action == "AMBIGUOUS" else "red")
    grounding_status = "[bold green]VERIFIED GROUNDED[/bold green]" if response.grounding.is_grounded else "[bold red]UNVERIFIED / HALLUCINATION RISK[/bold red]"

    console.print(
        Panel(
            f"[bold]Action Taken:[/bold] [{action_color}]{response.action.value}[/{action_color}] "
            f"(Confidence: {response.confidence:.3f})\n"
            f"[bold]Grounding Status:[/bold] {grounding_status} "
            f"(Score: {response.grounding.grounding_score:.3f})\n"
            f"[bold]Total Latency:[/bold] {response.total_latency_seconds:.3f} s",
            title="Pipeline Decision & Safety Verification",
            border_style=action_color,
        )
    )

    # 2. Answer Panel
    console.print(
        Panel(
            Text(response.answer),
            title="Generated Grounded Answer",
            border_style="cyan",
        )
    )

    # 3. Citations Table
    if response.citations:
        table = Table(title="Supporting Citations", show_header=True)
        table.add_column("Ref", style="bold cyan", width=5)
        table.add_column("Type", style="yellow", width=10)
        table.add_column("Source / Provenance", style="magenta", width=35)
        table.add_column("Verified Passage", style="white")

        for c in response.citations:
            table.add_row(
                f"[{c.index}]",
                c.source_type.upper(),
                c.provenance[:35] + ("..." if len(c.provenance) > 35 else ""),
                c.snippet[:120] + ("..." if len(c.snippet) > 120 else ""),
            )
        console.print(table)


@app.command()
def serve(
    host: str = typer.Option(settings.API_HOST, "--host", "-h", help="Bind host address"),
    port: int = typer.Option(settings.API_PORT, "--port", "-p", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Enable dev auto-reload"),
):
    """Launch the production FastAPI REST API service with Swagger UI."""
    console.print(
        Panel.fit(
            f"[bold green]Starting VeritasCRAG FastAPI Service[/bold green]\n"
            f"Listening on: [bold cyan]http://{host}:{port}[/bold cyan]\n"
            f"Interactive Swagger Docs: [bold blue]http://{host}:{port}/docs[/bold blue]",
            title="API Server",
        )
    )
    uvicorn.run("veritas_crag.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
