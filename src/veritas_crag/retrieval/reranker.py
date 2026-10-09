"""Cross-Encoder Reranker for high-precision semantic re-ranking."""

from typing import List, Optional, Tuple
import torch
from loguru import logger
from sentence_transformers import CrossEncoder

from veritas_crag.config.settings import AcceleratorDevice, settings
from veritas_crag.models.document import DocumentChunk


class CrossEncoderReranker:
    """Re-ranks candidate passages using a deep Cross-Encoder model on Apple Silicon MPS."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name or settings.RERANKER_MODEL_NAME
        self.device = self._resolve_device(device)

        logger.info(
            f"Loading CrossEncoder reranker '{self.model_name}' on device '{self.device}'..."
        )
        self._model = CrossEncoder(self.model_name, device=self.device)
        logger.info("Initialized CrossEncoderReranker successfully.")

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

    def rerank(
        self,
        query: str,
        chunks: List[DocumentChunk],
        top_k: Optional[int] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Scores (query, chunk) pairs using cross-attention and sorts by relevance descending.

        Returns:
            List of (DocumentChunk, rerank_score) tuples.
        """
        if not chunks:
            return []

        k = top_k or settings.RERANKER_TOP_K
        pairs = [(query, chunk.text) for chunk in chunks]

        # Predict cross-encoder relevance scores
        scores = self._model.predict(pairs, show_progress_bar=len(pairs) > 50)

        # Pair chunks with scores
        scored_chunks: List[Tuple[DocumentChunk, float]] = [
            (chunk, float(score)) for chunk, score in zip(chunks, scores)
        ]

        # Sort descending by cross-encoder score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        return scored_chunks[:k]
