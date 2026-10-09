"""Sparse Lexical Retrieval Engine using BM25Okapi."""

import pickle
import re
from pathlib import Path
from typing import List, Optional, Tuple
from loguru import logger
from rank_bm25 import BM25Okapi

from veritas_crag.config.settings import settings
from veritas_crag.models.document import DocumentChunk


class BM25Index:
    """Sparse lexical search engine for precise keyword, acronym, and identifier matching."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or settings.BM25_STORAGE_PATH
        self.chunks: List[DocumentChunk] = []
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: List[List[str]] = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple, robust lexical tokenizer that strips punctuation and lowercases terms."""
        return re.findall(r"\b\w+\b", text.lower())

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Adds and indexes DocumentChunk objects into the BM25 model."""
        if not chunks:
            return

        self.chunks.extend(chunks)
        new_tokenized = [self.tokenize(c.text) for c in chunks]
        self._tokenized_corpus.extend(new_tokenized)

        # Re-build BM25 index over full corpus
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(f"Built BM25 index over {len(self.chunks)} total chunks.")

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Performs lexical BM25 search.

        Returns:
            List of (DocumentChunk, bm25_score) tuples, ordered by score descending.
        """
        if not self._bm25 or not self.chunks:
            logger.warning("BM25 index is empty. Returning 0 results.")
            return []

        k = top_k or settings.SPARSE_TOP_K
        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)
        
        # Rank by score descending
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results: List[Tuple[DocumentChunk, float]] = []
        for idx in ranked_indices[:k]:
            score = float(scores[idx])
            if score > 0.0:  # Only return chunks with non-zero lexical overlap
                results.append((self.chunks[idx], score))

        return results

    def save(self, path: Optional[Path] = None) -> Path:
        """Serializes the BM25 index and chunk objects to disk."""
        target_path = Path(path or self.storage_path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "chunks": [c.model_dump() for c in self.chunks],
            "tokenized_corpus": self._tokenized_corpus,
        }
        with open(target_path, "wb") as f:
            pickle.dump(payload, f)

        logger.info(f"Persisted BM25 index ({len(self.chunks)} chunks) to: {target_path}")
        return target_path

    def load(self, path: Optional[Path] = None) -> bool:
        """Deserializes a previously saved BM25 index from disk."""
        target_path = Path(path or self.storage_path).resolve()
        if not target_path.exists():
            logger.debug(f"BM25 index file not found at: {target_path}")
            return False

        try:
            with open(target_path, "rb") as f:
                payload = pickle.load(f)

            self.chunks = [DocumentChunk.model_validate(c) for c in payload["chunks"]]
            self._tokenized_corpus = payload["tokenized_corpus"]
            self._bm25 = BM25Okapi(self._tokenized_corpus)
            logger.info(
                f"Successfully loaded BM25 index with {len(self.chunks)} chunks from: {target_path}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25 index from {target_path}: {e}")
            return False
