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

    # Future Phases: Vector Store & API keys (Optional for Phase 1)
    QDRANT_URL: Optional[str] = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None


settings = Settings()
