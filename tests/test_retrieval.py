"""Unit and integration tests for Phase 2: Hybrid Indexing & Retrieval."""

from pathlib import Path
import pytest

from veritas_crag.models.document import DocumentChunk
from veritas_crag.retrieval.bm25 import BM25Index
from veritas_crag.retrieval.embeddings import SentenceTransformerEmbeddings
from veritas_crag.retrieval.hybrid import HybridRetriever
from veritas_crag.retrieval.reranker import CrossEncoderReranker
from veritas_crag.retrieval.vector_store import QdrantVectorStore


@pytest.fixture(scope="session")
def embedding_model() -> SentenceTransformerEmbeddings:
    return SentenceTransformerEmbeddings()


@pytest.fixture(scope="session")
def reranker_model() -> CrossEncoderReranker:
    return CrossEncoderReranker()


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="chunk_001",
            document_source="test_doc.pdf",
            text="Docling parser features layout analysis, table structure extraction, and MPS acceleration on M4.",
            headings=["1. Ingestion Engine", "1.1 Docling"],
            page_numbers=[1],
            chunk_type="text",
        ),
        DocumentChunk(
            chunk_id="chunk_002",
            document_source="test_doc.pdf",
            text="The financial performance table shows Q1 revenue was 120.5 million and EBITDA was 35.2 million.",
            headings=["2. Financial Performance"],
            page_numbers=[2],
            chunk_type="table",
        ),
        DocumentChunk(
            chunk_id="chunk_003",
            document_source="test_doc.pdf",
            text="Corrective RAG activates dynamic web search fallback when document retrieval confidence is low.",
            headings=["3. Corrective Strategy"],
            page_numbers=[3],
            chunk_type="text",
        ),
    ]


def test_embeddings_generation(embedding_model: SentenceTransformerEmbeddings):
    texts = ["Test document embedding.", "Another sample chunk."]
    vectors = embedding_model.embed_documents(texts)

    assert len(vectors) == 2
    assert len(vectors[0]) == embedding_model.dimension
    assert embedding_model.dimension == 384

    query_vec = embedding_model.embed_query("Sample query")
    assert len(query_vec) == embedding_model.dimension


def test_bm25_lexical_matching(sample_chunks: list[DocumentChunk], tmp_path: Path):
    index_file = tmp_path / "bm25_test.pkl"
    bm25 = BM25Index(storage_path=index_file)
    bm25.add_chunks(sample_chunks)

    # Search for specific acronym "EBITDA"
    results = bm25.search("EBITDA", top_k=2)
    assert len(results) >= 1
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == "chunk_002"
    assert "EBITDA" in top_chunk.text
    assert score > 0.0

    # Test persistence (save and load)
    saved_path = bm25.save()
    assert saved_path.exists()

    new_bm25 = BM25Index(storage_path=saved_path)
    loaded = new_bm25.load()
    assert loaded is True
    assert len(new_bm25.chunks) == 3

    reloaded_results = new_bm25.search("Docling", top_k=1)
    assert len(reloaded_results) == 1
    assert reloaded_results[0][0].chunk_id == "chunk_001"


def test_qdrant_vector_store_in_memory(
    embedding_model: SentenceTransformerEmbeddings,
    sample_chunks: list[DocumentChunk],
):
    vstore = QdrantVectorStore(
        collection_name="test_in_memory_chunks",
        embedding_model=embedding_model,
        in_memory=True,
    )

    upserted = vstore.add_chunks(sample_chunks)
    assert upserted == 3
    assert vstore.count() == 3

    # Semantic search query
    results = vstore.search("What is the revenue in the financial statement?", top_k=2)
    assert len(results) == 2
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == "chunk_002"
    assert "revenue was 120.5 million" in top_chunk.text
    assert 0.0 <= score <= 1.0


def test_cross_encoder_reranker(
    reranker_model: CrossEncoderReranker,
    sample_chunks: list[DocumentChunk],
):
    query = "How does web search fallback work in Corrective RAG?"
    reranked = reranker_model.rerank(query, sample_chunks, top_k=2)

    assert len(reranked) == 2
    top_chunk, top_score = reranked[0]
    assert top_chunk.chunk_id == "chunk_003"
    assert "Corrective RAG activates dynamic web search" in top_chunk.text


def test_hybrid_retriever_end_to_end(
    embedding_model: SentenceTransformerEmbeddings,
    reranker_model: CrossEncoderReranker,
    sample_chunks: list[DocumentChunk],
    tmp_path: Path,
):
    vstore = QdrantVectorStore(
        collection_name="test_hybrid_collection",
        embedding_model=embedding_model,
        in_memory=True,
    )
    bm25 = BM25Index(storage_path=tmp_path / "bm25_hybrid.pkl")
    retriever = HybridRetriever(
        vector_store=vstore,
        bm25_index=bm25,
        reranker=reranker_model,
    )

    retriever.index_chunks(sample_chunks)

    # Hybrid retrieve with reranking
    search_res = retriever.retrieve(
        query="Tell me about Apple M4 hardware acceleration in Docling",
        top_k=2,
        use_reranker=True,
    )

    assert len(search_res.results) == 2
    assert search_res.total_candidates >= 1
    assert search_res.duration_seconds > 0.0

    top_res = search_res.results[0]
    assert top_res.chunk.chunk_id == "chunk_001"
    assert "M4" in top_res.chunk.text
    assert top_res.rerank_score is not None
