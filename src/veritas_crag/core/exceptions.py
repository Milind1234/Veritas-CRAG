"""Custom domain exceptions for VeritasCRAG."""


class VeritasCRAGError(Exception):
    """Base exception for all VeritasCRAG errors."""


class DocumentParsingError(VeritasCRAGError):
    """Raised when parsing a document fails."""


class DocumentNotFoundError(VeritasCRAGError):
    """Raised when an input document file is not found."""


class UnsupportedDocumentFormatError(VeritasCRAGError):
    """Raised when an unsupported document format is provided."""


class ChunkingError(VeritasCRAGError):
    """Raised when chunking fails."""


class ConfigurationError(VeritasCRAGError):
    """Raised when application configuration is invalid."""
