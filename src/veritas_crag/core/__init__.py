"""Core package containing logger and exceptions."""

from veritas_crag.core.exceptions import (
    ChunkingError,
    ConfigurationError,
    DocumentNotFoundError,
    DocumentParsingError,
    UnsupportedDocumentFormatError,
    VeritasCRAGError,
)
from veritas_crag.core.logger import logger, setup_logger

__all__ = [
    "logger",
    "setup_logger",
    "VeritasCRAGError",
    "DocumentParsingError",
    "DocumentNotFoundError",
    "UnsupportedDocumentFormatError",
    "ChunkingError",
    "ConfigurationError",
]
