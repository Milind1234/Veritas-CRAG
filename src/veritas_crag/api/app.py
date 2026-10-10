"""FastAPI production REST API for VeritasCRAG."""

import json
import shutil
import tempfile
import time
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
import torch
import docling

from veritas_crag.config.settings import settings
from veritas_crag.ingestion.parser import DoclingParser
from veritas_crag.models.pipeline import (
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from veritas_crag.pipeline.workflow import VeritasCRAGWorkflow

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Create FastAPI app
app = FastAPI(
    title="VeritasCRAG API",
    description="Production Corrective RAG & Document Intelligence System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files if directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def serve_dashboard():
    """Serve modern interactive VeritasCRAG UI dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "VeritasCRAG API is running. Visit /docs for OpenAPI specs."}


# Global workflow, parser, and document cache
_workflow: VeritasCRAGWorkflow | None = None
_parser: DoclingParser | None = None
_latest_document: IngestResponse | None = None


def _load_latest_from_disk() -> IngestResponse | None:
    """Load latest parsed document from memory or output artifacts directory."""
    global _latest_document
    if _latest_document is not None:
        return _latest_document

    output_dir = Path(settings.OUTPUT_DIR)
    if not output_dir.exists():
        return None

    parsed_files = sorted(output_dir.glob("*_parsed.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not parsed_files:
        return None

    try:
        latest_file = parsed_files[0]
        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        md_file = latest_file.with_name(latest_file.name.replace("_parsed.json", ".md"))
        md_content = md_file.read_text(encoding="utf-8") if md_file.exists() else data.get("markdown", "")

        metadata = data.get("metadata", {})
        tables = data.get("tables", [])
        chunks = data.get("chunks", [])

        _latest_document = IngestResponse(
            file_name=metadata.get("file_name", latest_file.stem),
            file_size_kb=round(metadata.get("file_size_bytes", 0) / 1024, 2),
            num_pages=metadata.get("num_pages", 1),
            total_tables=len(tables),
            total_chunks=len(chunks),
            parsing_duration_seconds=round(metadata.get("parsing_duration_seconds", 0.0), 3),
            markdown=md_content,
            tables=tables,
            chunks=chunks,
        )
        return _latest_document
    except Exception as e:
        logger.warning(f"Could not load latest document from disk: {e}")
        return None


def get_workflow() -> VeritasCRAGWorkflow:
    global _workflow
    if _workflow is None:
        logger.info("Initializing VeritasCRAGWorkflow for API server...")
        _workflow = VeritasCRAGWorkflow()
    return _workflow


def get_parser() -> DoclingParser:
    global _parser
    if _parser is None:
        logger.info("Initializing DoclingParser for API server...")
        _parser = DoclingParser()
    return _parser


@app.get("/api/v1/health", tags=["System"])
def health_check():
    """System health check and Apple Silicon MPS accelerator status."""
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "environment": settings.ENVIRONMENT,
        "docling_version": docling.__version__,
        "torch_version": torch.__version__,
        "apple_silicon_mps_available": torch.backends.mps.is_available(),
        "accelerator_device": settings.ACCELERATOR_DEVICE.value,
        "qdrant_collection": settings.QDRANT_COLLECTION_NAME,
    }


@app.get("/api/v1/document/latest", response_model=IngestResponse, tags=["Document Ingestion"])
def get_latest_document():
    """Retrieve details, markdown, tables, and chunks of the most recently parsed document."""
    doc = _load_latest_from_disk()
    if not doc:
        raise HTTPException(status_code=404, detail="No parsed documents available yet.")
    return doc


@app.post("/api/v1/query", response_model=QueryResponse, tags=["Retrieval & CRAG"])
def query_pipeline(req: QueryRequest):
    """Execute end-to-end Corrective RAG query with LangGraph orchestration."""
    try:
        workflow = get_workflow()
        response = workflow.run(query=req.query, top_k=req.top_k)

        return QueryResponse(
            query=response.query,
            answer=response.answer,
            action=response.action.value,
            confidence=response.confidence,
            is_grounded=response.grounding.is_grounded,
            grounding_score=response.grounding.grounding_score,
            citations=response.citations,
            latency_seconds=response.total_latency_seconds,
        )
    except Exception as e:
        logger.error(f"Error executing query '{req.query}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ingest", response_model=IngestResponse, tags=["Document Ingestion"])
async def ingest_document(file: UploadFile = File(...)):
    """Upload and ingest a PDF document with Docling layout analysis and hybrid indexing."""
    global _latest_document
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in [".pdf", ".docx", ".pptx", ".html", ".md"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{suffix}'. Supported: .pdf, .docx, .pptx, .html, .md",
        )

    start_time = time.perf_counter()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_path = Path(tmp_file.name)
        shutil.copyfileobj(file.file, tmp_file)

    try:
        parser = get_parser()
        workflow = get_workflow()

        # 1. Parse document with Docling
        parsed_doc = parser.parse(tmp_path, chunk=True)

        # 2. Export artifacts to disk
        parser.export_to_disk(parsed_doc, output_dir=settings.OUTPUT_DIR)

        # 3. Index chunks into Hybrid Retriever (Qdrant + BM25)
        if parsed_doc.chunks:
            workflow.retriever.index_chunks(parsed_doc.chunks)

        total_duration = time.perf_counter() - start_time
        file_size_kb = tmp_path.stat().st_size / 1024

        _latest_document = IngestResponse(
            file_name=file.filename or "uploaded_doc",
            file_size_kb=round(file_size_kb, 2),
            num_pages=parsed_doc.metadata.num_pages,
            total_tables=parsed_doc.total_tables,
            total_chunks=parsed_doc.total_chunks,
            parsing_duration_seconds=round(total_duration, 3),
            markdown=parsed_doc.markdown,
            tables=parsed_doc.tables,
            chunks=parsed_doc.chunks,
        )
        return _latest_document

    except Exception as e:
        logger.error(f"Failed to ingest document '{file.filename}': {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

