"""Embedding models interface and implementation using SentenceTransformers & PyTorch MPS."""

from abc import ABC, abstractmethod
from typing import List, Optional
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from veritas_crag.config.settings import AcceleratorDevice, settings


class BaseEmbeddingModel(ABC):
    """Abstract interface for dense embedding generation."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute dense embeddings for a list of document strings."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Compute dense embedding for a single query string."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the output vector dimension."""
        pass


class SentenceTransformerEmbeddings(BaseEmbeddingModel):
    """Local SentenceTransformer embedding model with Apple Silicon MPS acceleration."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self.normalize_embeddings = normalize_embeddings

        # Determine optimal device (prefer MPS on Apple Silicon)
        self.device = self._resolve_device(device)
        logger.info(
            f"Loading embedding model '{self.model_name}' on device '{self.device}'..."
        )

        self._model = SentenceTransformer(self.model_name, device=self.device)
        if hasattr(self._model, "get_embedding_dimension"):
            self._dimension = self._model.get_embedding_dimension()
        else:
            self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(
            f"Initialized SentenceTransformerEmbeddings (dim: {self._dimension}, device: {self.device})"
        )

    def _resolve_device(self, requested_device: Optional[str]) -> str:
        """Resolves target device, prioritizing Apple Silicon MPS if available."""
        if requested_device:
            return requested_device

        config_device = settings.ACCELERATOR_DEVICE
        if config_device == AcceleratorDevice.MPS:
            return "mps" if torch.backends.mps.is_available() else "cpu"
        elif config_device == AcceleratorDevice.CUDA:
            return "cuda" if torch.cuda.is_available() else "cpu"
        elif config_device == AcceleratorDevice.AUTO:
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return "cpu"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates dense vector embeddings for documents."""
        if not texts:
            return []

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Generates dense vector embedding for a single search query."""
        embedding = self._model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embedding.tolist()
