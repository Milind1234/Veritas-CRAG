# VeritasCRAG 🛡️
> **Production-Grade Corrective RAG & Document Intelligence System**  
> Optimized for Apple Silicon (M4 / MPS) & Enterprise Workloads.

VeritasCRAG combines deep document layout analysis, structured table parsing, hybrid retrieval (Dense Qdrant + Sparse BM25 + Cross-Encoder reranking), self-correcting evaluation loops with active web search fallback, and LangGraph state orchestration with hallucination/grounding verification.

---

## 🏗️ Architecture Overview

```mermaid
flowchart TD
    UserQuery[User Question] --> Retrieve[Node 1: Hybrid Retrieval<br/>Dense Qdrant + Sparse BM25 + RRF]
    Retrieve --> Grade[Node 2: Retrieval Evaluator / Grader<br/>Sigmoid Cross-Encoder Confidence]
    
    Grade -- "Confidence >= 0.65 (CORRECT)" --> RefineInternal[Node 3: Knowledge Refiner<br/>Decompose & Filter Sentence Strips]
    
    Grade -- "0.30 <= Conf < 0.65 (AMBIGUOUS)" --> WebFallback1[Node 4: Active Web Fallback<br/>Query Rewriter + DuckDuckGo/Tavily]
    WebFallback1 --> RefineBoth[Node 3: Refine Internal + Web Knowledge]
    
    Grade -- "Conf < 0.30 (INCORRECT)" --> WebFallback2[Node 4: Active Web Fallback<br/>Discard Internal Chunks to Stop Hallucinations]
    WebFallback2 --> RefineWebOnly[Refine External Knowledge Only]
    
    RefineInternal --> Generate[Node 5: Grounded Answer Synthesis<br/>Strict Inline Citations [1], [2]]
    RefineBoth --> Generate
    RefineWebOnly --> Generate
    
    Generate --> Verify[Node 6: Faithfulness & Grounding Grader<br/>Cross-Attention Sentence Entailment]
    Verify --> FinalResponse[Final Verified Response + Provenance Citations]
```

---

## 🚀 Key Features by Phase

### Phase 1: High-Fidelity Document Intelligence
- **Docling 2.x Integration**: IBM Docling document converter engine preserving reading order, headers, lists, and complex multi-column layouts.
- **Table Structure Extraction**: Extracts tabular data cleanly into Markdown, HTML, and structured tabular arrays.
- **Apple Silicon M4 Acceleration**: Native PyTorch `MPS` (Metal Performance Shaders) support for layout analysis and embedding inference.
- **Layout-Aware Hybrid Chunking**: Preserves heading hierarchy breadcrumbs (`headings`), table structure continuity, and page provenance for every chunk.

### Phase 2: Hybrid Indexing & Storage Engine
- **Dense Vector Store (Qdrant)**: Embedded zero-dependency vector database running locally at `data/qdrant_db/` with client caching and full payload preservation.
- **Sparse Lexical Search (BM25)**: BM25Okapi index resolving acronyms, exact IDs, and rare technical keywords, persisted to `data/bm25_index.pkl`.
- **Reciprocal Rank Fusion (RRF)**: Fuses dense and sparse rankings: $RRF(d) = \sum \frac{1}{60 + \text{rank}(d)}$.
- **Cross-Encoder Reranking**: Re-ranks top candidates using `cross-encoder/ms-marco-MiniLM-L-6-v2` with deep cross-attention on MPS.

### Phase 3: Corrective RAG (CRAG) Decision Engine
- **Confidence Scoring & Thresholds**: Sigmoid-normalized probabilities determining `CORRECT` ($\ge 0.65$), `AMBIGUOUS` ($[0.30, 0.65)$), or `INCORRECT` ($< 0.30$).
- **Anti-Hallucination Discard**: Automatically discards internal documents when irrelevant to prevent hallucinating on internal context.
- **Active Web Search Fallback**: Zero-config DuckDuckGo integration (`ddgs`) + optional enterprise Tavily API support.
- **Query Rewriter**: Strips conversational baggage and produces entity-dense search engine queries.
- **Knowledge Refinement**: Decomposes paragraphs into atomic sentence strips, eliminates noisy text, and preserves table grids.

### Phase 4: Synthesis, Grounding, LangGraph & REST API
- **LangGraph State Machine**: Full `StateGraph` linking all 6 nodes with conditional branching and state management.
- **Grounded Answer Generator**: Strictly conditions output on numbered citations `[1]`, `[2]` with support for Gemini, OpenAI, or deterministic local offline synthesis.
- **Faithfulness & Hallucination Grader**: Evaluates sentence-level cross-attention entailment against reference context to detect ungrounded claims.
- **Production FastAPI Server**: REST API with endpoints for document upload, hybrid query execution, health checks, and interactive Swagger documentation.

---

## 📂 Project Structure

```
Veritas-CRAG/
├── pyproject.toml                             # Packaging, Ruff, and Pytest configuration
├── requirements.txt                           # Core dependencies
├── README.md                                  # Complete architecture & documentation
├── src/veritas_crag/
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py                             # FastAPI REST endpoints
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py                        # Pydantic BaseSettings (M4/MPS options, paths)
│   ├── core/
│   │   ├── logger.py                          # Loguru structured logger
│   │   └── exceptions.py                      # Custom domain exceptions
│   ├── models/
│   │   ├── document.py                        # DocumentMetadata, DocumentTable, DocumentChunk
│   │   ├── retrieval.py                       # RetrievalResult, HybridSearchResult
│   │   ├── crag.py                            # CRAGAction, DocumentGrade, WebSearchResult
│   │   └── pipeline.py                        # Citation, GroundingEvaluation, GeneratedResponse
│   ├── ingestion/
│   │   ├── parser.py                          # DoclingParser (layout analysis, table extraction)
│   │   └── chunker.py                         # Layout-aware DoclingChunker with breadcrumbs
│   ├── retrieval/
│   │   ├── embeddings.py                      # SentenceTransformerEmbeddings (MPS accelerated)
│   │   ├── vector_store.py                    # QdrantVectorStore (embedded persistent storage)
│   │   ├── bm25.py                            # BM25Index (sparse lexical engine)
│   │   ├── reranker.py                        # CrossEncoderReranker (deep cross-attention)
│   │   └── hybrid.py                          # HybridRetriever (RRF + Reranker)
│   ├── evaluation/
│   │   ├── grader.py                          # DocumentGrader (sigmoid confidence evaluator)
│   │   ├── query_rewriter.py                  # Query transformation engine
│   │   ├── web_search.py                      # Active web search fallback (DuckDuckGo/Tavily)
│   │   ├── refiner.py                         # Knowledge decomposition & strip filter
│   │   └── crag_engine.py                     # Master Corrective Decision Loop
│   └── pipeline/
│       ├── generator.py                       # Grounded response generator with citations
│       ├── grounding.py                       # Faithfulness & hallucination verification
│       └── workflow.py                        # LangGraph StateGraph orchestrator
├── tests/                                     # 27 automated unit & integration tests
├── scripts/
│   ├── run_parser.py                          # CLI: Ingest & parse PDFs
│   ├── run_retriever.py                       # CLI: Index & test hybrid search
│   ├── run_crag.py                            # CLI: Evaluate corrective action decisions
│   └── run_pipeline.py                       # CLI: Run LangGraph pipeline & launch FastAPI server
└── data/
    ├── sample_docs/                           # Input PDFs
    ├── output/                                # Exported Markdown, JSON, and Chunks
    ├── qdrant_db/                             # Local persistent vector database
    └── bm25_index.pkl                         # Serialized lexical index
```

---

## ⚡ Quickstart

### 1. Environment Setup
```bash
# Clone and enter directory
cd /Users/milind/Veritas-CRAG

# Activate virtual environment
source .venv/bin/activate

# Install package in editable mode
pip install -e ".[dev]"
```

### 2. Verify Hardware Status
Check Apple Silicon Metal (`MPS`) and Docling status:
```bash
python scripts/run_parser.py info
```

---

## 🛠️ CLI Workflows

### 1. Ingest & Parse a PDF
```bash
python scripts/run_parser.py parse data/sample_docs/sample_report.pdf
```

### 2. Index Chunks into Hybrid Stores
```bash
python scripts/run_retriever.py index
```

### 3. Test Hybrid Retrieval & Live Scoring
```bash
python scripts/run_retriever.py search "What is the Docling parser latency on M4?"
```

### 4. Test the Corrective RAG Decision Loop
```bash
# Internal Query -> Triggers [CORRECT] Action (No web request needed)
python scripts/run_crag.py evaluate "What is the Docling latency on M4?"

# External Unknown Query -> Triggers [INCORRECT] Action & Web Fallback
python scripts/run_crag.py evaluate "What is the latest Apple M4 Max chip benchmark?"
```

### 5. Execute Full LangGraph Pipeline (End-to-End)
```bash
python scripts/run_pipeline.py query "What is the Docling latency on M4?"
```

### 6. Launch Production FastAPI REST Server
```bash
python scripts/run_pipeline.py serve --port 8000
```
Interactive Swagger documentation will be available at: **http://localhost:8000/docs**

---

## 🌐 API Endpoints

- `GET /api/v1/health`: Checks system health, Docling version, PyTorch version, and Apple Silicon MPS hardware status.
- `POST /api/v1/query`: Accepts `{"query": "...", "top_k": 5}` and returns the complete LangGraph response with answer, action, confidence, grounding score, and citations.
- `POST /api/v1/ingest`: Accepts PDF upload (`multipart/form-data`), runs Docling layout analysis, and indexes chunks into the hybrid stores.

---

## 🧪 Running the Test Suite

Run all **27 unit and integration tests**:

```bash
pytest -v
```
```
tests/test_api.py::test_health_check_endpoint PASSED
tests/test_api.py::test_query_endpoint PASSED
tests/test_api.py::test_ingest_endpoint_unsupported_format PASSED
tests/test_chunker.py::test_chunker_structure_and_breadcrumbs PASSED
tests/test_crag.py::test_query_rewriter PASSED
tests/test_crag.py::test_document_grader_correct_action PASSED
tests/test_crag.py::test_document_grader_incorrect_action PASSED
tests/test_crag.py::test_knowledge_refiner PASSED
tests/test_crag.py::test_web_search_engine_mock PASSED
tests/test_crag.py::test_crag_engine_correct_flow PASSED
tests/test_crag.py::test_crag_engine_incorrect_flow PASSED
tests/test_models.py::test_document_metadata_instantiation PASSED
tests/test_models.py::test_document_table_model PASSED
tests/test_models.py::test_document_chunk_model PASSED
tests/test_models.py::test_parsed_document_properties PASSED
tests/test_parser.py::test_parser_file_not_found PASSED
tests/test_parser.py::test_parser_unsupported_format PASSED
tests/test_parser.py::test_parser_successful_pdf_parse PASSED
tests/test_pipeline.py::test_response_generator PASSED
tests/test_pipeline.py::test_grounding_grader_faithful PASSED
tests/test_pipeline.py::test_grounding_grader_catches_hallucination PASSED
tests/test_pipeline.py::test_langgraph_workflow_end_to_end PASSED
tests/test_retrieval.py::test_embeddings_generation PASSED
tests/test_retrieval.py::test_bm25_lexical_matching PASSED
tests/test_retrieval.py::test_qdrant_vector_store_in_memory PASSED
tests/test_retrieval.py::test_cross_encoder_reranker PASSED
tests/test_retrieval.py::test_hybrid_retriever_end_to_end PASSED
======================= 27 passed in 49.67s ========================
```
