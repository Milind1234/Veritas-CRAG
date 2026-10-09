"""Dense Vector Store integration using Qdrant (local persistent / in-memory / remote)."""

import uuid
from pathlib import Path
from typing import List, Optional, Tuple, Union
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from veritas_crag.config.settings import settings
from veritas_crag.models.document import DocumentChunk
from veritas_crag.retrieval.embeddings import BaseEmbeddingModel, SentenceTransformerEmbeddings


class QdrantVectorStore:
    """Production-grade vector database interface using Qdrant."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        storage_path: Optional[Union[str, Path]] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        in_memory: bool = False,
    ):
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.embedding_model = embedding_model or SentenceTransformerEmbeddings()

        # Initialize Qdrant Client based on configuration
        if in_memory:
            logger.info("Initializing QdrantClient in-memory mode (:memory:)")
            self.client = QdrantClient(":memory:")
        elif url or settings.QDRANT_URL:
            target_url = url or settings.QDRANT_URL
            target_key = api_key or settings.QDRANT_API_KEY
            logger.info(f"Connecting to remote Qdrant instance at: {target_url}")
            self.client = QdrantClient(url=target_url, api_key=target_key)
        else:
            local_path = Path(storage_path or settings.QDRANT_STORAGE_PATH).resolve()
            local_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initializing local persistent Qdrant at: {local_path}")
            self.client = QdrantClient(path=str(local_path))

        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """Ensures the collection exists with correct vector dimensions and distance metric."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                logger.info(
                    f"Creating Qdrant collection '{self.collection_name}' (dim: {self.embedding_model.dimension}, metric: Cosine)..."
                )
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.embedding_model.dimension,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
        except Exception as e:
            logger.error(f"Error checking or creating collection '{self.collection_name}': {e}")
            raise

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Encodes and indexes DocumentChunk objects into Qdrant."""
        if not chunks:
            return 0

        logger.info(f"Generating dense embeddings for {len(chunks)} chunks...")
        texts = [c.text for c in chunks]
        vectors = self.embedding_model.embed_documents(texts)

        points: List[qmodels.PointStruct] = []
        for chunk, vector in zip(chunks, vectors):
            # Deterministic UUID from chunk_id
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            payload = {
                "chunk_id": chunk.chunk_id,
                "document_source": chunk.document_source,
                "text": chunk.text,
                "page_numbers": chunk.page_numbers,
                "headings": chunk.headings,
                "token_count": chunk.token_count,
                "chunk_type": chunk.chunk_type,
                "metadata": chunk.metadata,
            }
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        logger.info(f"Upserted {len(points)} chunks into collection '{self.collection_name}'.")
        return len(points)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Searches collection using dense vector cosine similarity.

        Returns:
            List of (DocumentChunk, score) tuples, sorted by cosine similarity descending.
        """
        k = top_k or settings.DENSE_TOP_K
        query_vector = self.embedding_model.embed_query(query)

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=k,
            score_threshold=score_threshold,
            with_payload=True,
        )

        results: List[Tuple[DocumentChunk, float]] = []
        for point in search_result.points:
            payload = point.payload or {}
            chunk = DocumentChunk(
                chunk_id=payload.get("chunk_id", str(point.id)),
                document_source=payload.get("document_source", ""),
                text=payload.get("text", ""),
                page_numbers=payload.get("page_numbers", []),
                headings=payload.get("headings", []),
                token_count=payload.get("token_count"),
                chunk_type=payload.get("chunk_type", "text"),
                metadata=payload.get("metadata", {}),
            )
            results.append((chunk, float(point.score)))

        return results

    def count(self) -> int:
        """Returns total number of points stored in the collection."""
        res = self.client.count(collection_name=self.collection_name)
        return res.count
