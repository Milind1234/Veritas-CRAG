"""CLI tool for testing the Corrective RAG (CRAG) decision and fallback loop."""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from veritas_crag.config.settings import settings
from veritas_crag.evaluation.crag_engine import CRAGEngine
from veritas_crag.models.crag import CRAGAction, CRAGContext

app = typer.Typer(
    name="veritas-crag",
    help="VeritasCRAG Corrective Decision & Fallback Engine CLI",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.command()
def info():
    """Display CRAG thresholds and search providers."""
    table = Table(title="CRAG Configuration & Thresholds", show_header=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Upper Confidence Threshold (τ_U)", str(settings.CRAG_UPPER_THRESHOLD))
    table.add_row("Lower Confidence Threshold (τ_L)", str(settings.CRAG_LOWER_THRESHOLD))
    table.add_row("Web Search Provider", settings.WEB_SEARCH_PROVIDER)
    table.add_row("Max Web Results", str(settings.MAX_WEB_RESULTS))
    table.add_row("Knowledge Refinement", str(settings.ENABLE_KNOWLEDGE_REFINEMENT))
    console.print(table)


@app.command()
def evaluate(
    query: str = typer.Argument(..., help="Search query or question to evaluate through CRAG"),
    top_k: int = typer.Option(3, "--top-k", "-k", help="Candidate chunks to evaluate"),
):
    """Execute the CRAG decision loop: Retrieval -> Evaluator -> Corrective Action -> Refinement."""
    console.print(
        Panel.fit(
            f"Query: [bold cyan]\"{query}\"[/bold cyan]\n"
            f"Candidate Pool: [yellow]{top_k}[/yellow] | "
            f"Upper Threshold: [green]{settings.CRAG_UPPER_THRESHOLD}[/green] | "
            f"Lower Threshold: [red]{settings.CRAG_LOWER_THRESHOLD}[/red]",
            title="CRAG Evaluation Pipeline",
        )
    )

    engine = CRAGEngine()
    with console.status("[bold green]Executing CRAG pipeline (Retrieval -> Grading -> Action)..."):
        context: CRAGContext = engine.run(query=query, top_k=top_k)

    # 1. Document Grades Table
    grade_table = Table(title="Internal Document Grades", show_header=True)
    grade_table.add_column("Chunk ID", style="cyan", width=22)
    grade_table.add_column("Score", style="magenta", width=8)
    grade_table.add_column("Relevant?", style="bold", width=10)
    grade_table.add_column("Rationale", style="white")

    for grade in context.grading_result.grades:
        verdict_str = "[green]YES[/green]" if grade.is_relevant else "[red]NO[/red]"
        grade_table.add_row(
            grade.chunk_id[:20] + "...",
            f"{grade.score:.3f}",
            verdict_str,
            grade.rationale or "-",
        )

    console.print(grade_table)

    # 2. Action Banner
    action_color = {
        CRAGAction.CORRECT: "bold green",
        CRAGAction.AMBIGUOUS: "bold yellow",
        CRAGAction.INCORRECT: "bold red",
    }[context.action]

    action_details = {
        CRAGAction.CORRECT: "High confidence in internal documents. Using internal refined knowledge directly.",
        CRAGAction.AMBIGUOUS: "Moderate confidence. Combining internal knowledge with active web search fallback.",
        CRAGAction.INCORRECT: "Low/No confidence in internal documents. Discarding internal chunks & triggering full web fallback.",
    }[context.action]

    console.print(
        Panel(
            f"[bold]Action:[/bold] [{action_color}]{context.action.value}[/{action_color}]\n"
            f"[bold]Confidence Score:[/bold] {context.confidence:.3f}\n"
            f"[bold]Transformed Query:[/bold] {context.transformed_query or 'N/A'}\n"
            f"[italic]{action_details}[/italic]",
            title="Corrective Decision Verdict",
            border_style=action_color,
        )
    )

    # 3. Final Consolidated Context Preview
    from rich.text import Text
    rendered_context = (
        Text(context.final_context_text)
        if context.final_context_text
        else Text("No knowledge strips retained.", style="italic yellow")
    )
    console.print(
        Panel(
            rendered_context,
            title="Final Verified Knowledge Context (Ready for Generator)",
            border_style="cyan",
        )
    )


if __name__ == "__main__":
    app()
