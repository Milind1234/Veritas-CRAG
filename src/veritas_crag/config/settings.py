"""Configuration management for VeritasCRAG."""

from enum import Enum
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AcceleratorDevice(str, Enum):
    AUTO = "auto"
    MPS = "mps"
    CUDA = "cuda"
    CPU = "cpu"


class Settings(BaseSettings):
    """Global configuration settings loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Base Application
    PROJECT_NAME: str = "VeritasCRAG"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # Base Paths
    BASE_DIR: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent.parent
    )
    DATA_DIR: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent.parent / "data"
    )
    SAMPLE_DOCS_DIR: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent.parent / "data" / "sample_docs"
    )
    OUTPUT_DIR: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent.parent / "data" / "output"
    )

    # Document Ingestion / Docling
    ACCELERATOR_DEVICE: AcceleratorDevice = AcceleratorDevice.AUTO
    NUM_THREADS: int = 4
    ENABLE_OCR: bool = False
    ENABLE_TABLE_STRUCTURE: bool = True
    MAX_FILE_SIZE_MB: int = 50
    GENERATE_PAGE_IMAGES: bool = False

    # Chunking Configuration
    MAX_CHUNK_TOKENS: int = 512
    CHUNK_MERGE_PEERS: bool = True
    REPEAT_TABLE_HEADER: bool = True

    # Phase 2: Retrieval & Vector Store Configuration
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 32
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    QDRANT_STORAGE_PATH: Optional[Path] = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent.parent / "data" / "qdrant_db"
    )
    QDRANT_COLLECTION_NAME: str = "veritas_chunks"
    QDRANT_URL: Optional[str] = None  # Set if using external Qdrant server
    QDRANT_API_KEY: Optional[str] = None
    
    BM25_STORAGE_PATH: Optional[Path] = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent.parent / "data" / "bm25_index.pkl"
    )

    DENSE_TOP_K: int = 10
    SPARSE_TOP_K: int = 10
    RRF_K: int = 60
    RERANKER_TOP_K: int = 5

    # Phase 3: Corrective RAG (CRAG) Configuration
    CRAG_UPPER_THRESHOLD: float = 0.65  # Upper threshold (CORRECT if >= threshold)
    CRAG_LOWER_THRESHOLD: float = 0.30  # Lower threshold (INCORRECT if < threshold; AMBIGUOUS if between)
    MAX_WEB_RESULTS: int = 3
    WEB_SEARCH_PROVIDER: str = "duckduckgo"  # "duckduckgo" or "tavily"
    ENABLE_KNOWLEDGE_REFINEMENT: bool = True
    MAX_REFINED_STRIPS_PER_DOC: int = 3

    # API Keys for LLMs & Web Fallbacks
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None


settings = Settings()
