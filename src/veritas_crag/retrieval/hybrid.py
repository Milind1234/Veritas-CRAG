"""Hybrid Retriever combining Dense Vector, Sparse BM25, and Reciprocal Rank Fusion."""

import time
from typing import Dict, List, Optional
from loguru import logger

from veritas_crag.config.settings import settings
from veritas_crag.models.document import DocumentChunk
from veritas_crag.models.retrieval import HybridSearchResult, RetrievalResult
from veritas_crag.retrieval.bm25 import BM25Index
from veritas_crag.retrieval.reranker import CrossEncoderReranker
from veritas_crag.retrieval.vector_store import QdrantVectorStore


class HybridRetriever:
    """Production Hybrid Retriever implementing Reciprocal Rank Fusion (RRF) and Reranking."""

    def __init__(
        self,
        vector_store: Optional[QdrantVectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        rrf_k: Optional[int] = None,
    ):
        self.vector_store = vector_store or QdrantVectorStore()
        self.bm25_index = bm25_index or BM25Index()
        self.reranker = reranker or CrossEncoderReranker()
        self.rrf_k = rrf_k or settings.RRF_K

    def index_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Indexes document chunks into both dense vector store and sparse BM25 index."""
        if not chunks:
            return 0

        logger.info(f"Indexing {len(chunks)} chunks across Dense (Qdrant) and Sparse (BM25)...")
        # 1. Index into Qdrant
        added_dense = self.vector_store.add_chunks(chunks)

        # 2. Index into BM25 and save to disk
        self.bm25_index.add_chunks(chunks)
        self.bm25_index.save()

        logger.info(f"Successfully indexed {added_dense} chunks into hybrid stores.")
        return added_dense

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        dense_top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
        use_reranker: bool = True,
    ) -> HybridSearchResult:
        """Executes hybrid retrieval using Reciprocal Rank Fusion and optional cross-encoder reranking.

        Args:
            query: User search query.
            top_k: Number of final results to return.
            dense_top_k: Number of candidates to fetch from dense store.
            sparse_top_k: Number of candidates to fetch from sparse BM25.
            use_reranker: Whether to apply the cross-encoder reranker on fused results.

        Returns:
            HybridSearchResult containing scored and ranked RetrievalResult items.
        """
        start_time = time.perf_counter()
        target_k = top_k or (settings.RERANKER_TOP_K if use_reranker else settings.DENSE_TOP_K)
        d_k = dense_top_k or settings.DENSE_TOP_K
        s_k = sparse_top_k or settings.SPARSE_TOP_K

        logger.info(f"Hybrid retrieval for query: '{query}' (dense_k={d_k}, sparse_k={s_k})")

        # 1. Retrieve Dense Candidates
        dense_results = self.vector_store.search(query=query, top_k=d_k)

        # 2. Retrieve Sparse BM25 Candidates
        sparse_results = self.bm25_index.search(query=query, top_k=s_k)

        # Track candidate chunks by ID
        chunk_map: Dict[str, DocumentChunk] = {}
        dense_ranks: Dict[str, int] = {}
        dense_scores: Dict[str, float] = {}
        sparse_ranks: Dict[str, int] = {}
        sparse_scores: Dict[str, float] = {}

        for rank, (chunk, score) in enumerate(dense_results, start=1):
            chunk_map[chunk.chunk_id] = chunk
            dense_ranks[chunk.chunk_id] = rank
            dense_scores[chunk.chunk_id] = score

        for rank, (chunk, score) in enumerate(sparse_results, start=1):
            chunk_map[chunk.chunk_id] = chunk
            sparse_ranks[chunk.chunk_id] = rank
            sparse_scores[chunk.chunk_id] = score

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF_Score(d) = sum(1 / (k + rank))
        rrf_scores: Dict[str, float] = {}
        for chunk_id in chunk_map:
            score = 0.0
            if chunk_id in dense_ranks:
                score += 1.0 / (self.rrf_k + dense_ranks[chunk_id])
            if chunk_id in sparse_ranks:
                score += 1.0 / (self.rrf_k + sparse_ranks[chunk_id])
            rrf_scores[chunk_id] = score

        # Sort candidates by RRF score descending
        fused_candidate_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: rrf_scores[cid],
            reverse=True,
        )

        total_candidates = len(fused_candidate_ids)
        candidate_chunks = [chunk_map[cid] for cid in fused_candidate_ids]

        # 4. Optional Cross-Encoder Reranking
        final_results: List[RetrievalResult] = []

        if use_reranker and candidate_chunks:
            logger.info(f"Applying Cross-Encoder reranking to top {len(candidate_chunks)} candidates...")
            reranked_pairs = self.reranker.rerank(
                query=query,
                chunks=candidate_chunks[:max(target_k * 2, 10)],  # rerank candidate pool
                top_k=target_k,
            )
            for chunk, rerank_score in reranked_pairs:
                cid = chunk.chunk_id
                final_results.append(
                    RetrievalResult(
                        chunk=chunk,
                        score=rerank_score,
                        dense_score=dense_scores.get(cid),
                        dense_rank=dense_ranks.get(cid),
                        sparse_score=sparse_scores.get(cid),
                        sparse_rank=sparse_ranks.get(cid),
                        rrf_score=rrf_scores.get(cid),
                        rerank_score=rerank_score,
                    )
                )
        else:
            for cid in fused_candidate_ids[:target_k]:
                chunk = chunk_map[cid]
                rrf_score = rrf_scores[cid]
                final_results.append(
                    RetrievalResult(
                        chunk=chunk,
                        score=rrf_score,
                        dense_score=dense_scores.get(cid),
                        dense_rank=dense_ranks.get(cid),
                        sparse_score=sparse_scores.get(cid),
                        sparse_rank=sparse_ranks.get(cid),
                        rrf_score=rrf_score,
                        rerank_score=None,
                    )
                )

        duration = time.perf_counter() - start_time
        logger.info(
            f"Hybrid retrieval finished in {duration:.3f}s. Returned {len(final_results)} results."
        )

        return HybridSearchResult(
            query=query,
            results=final_results,
            total_candidates=total_candidates,
            duration_seconds=round(duration, 4),
        )
