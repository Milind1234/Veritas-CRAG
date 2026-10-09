"""Knowledge Refinement engine: decomposing chunks into atomic strips and filtering noise."""

import re
from typing import List, Optional
from loguru import logger

from veritas_crag.config.settings import settings
from veritas_crag.models.crag import RefinedKnowledge, WebSearchResult
from veritas_crag.models.document import DocumentChunk
from veritas_crag.retrieval.reranker import CrossEncoderReranker


class KnowledgeRefiner:
    """Decomposes retrieved documents into atomic knowledge strips and filters internal noise."""

    def __init__(
        self,
        reranker: Optional[CrossEncoderReranker] = None,
        max_strips_per_doc: Optional[int] = None,
    ):
        self.reranker = reranker or CrossEncoderReranker()
        self.max_strips = max_strips_per_doc or settings.MAX_REFINED_STRIPS_PER_DOC

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """Splits paragraph text into clean sentences/atomic knowledge units."""
        # Split on sentence boundaries while preserving decimal numbers like 120.5
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
        cleaned = [s.strip() for s in sentences if len(s.strip()) > 15]
        return cleaned or [text.strip()]

    @staticmethod
    def _format_provenance(chunk: DocumentChunk) -> str:
        """Constructs human-readable citation with heading breadcrumbs and page numbers."""
        parts = [chunk.document_source]
        if chunk.headings:
            parts.append(" > ".join(chunk.headings))
        base = " | ".join(parts)
        if chunk.page_numbers:
            return f"{base} (p. {', '.join(map(str, chunk.page_numbers))})"
        return base

    def refine_internal_chunks(
        self,
        query: str,
        chunks: List[DocumentChunk],
    ) -> List[RefinedKnowledge]:
        """Filters noise within internal document chunks and keeps the most relevant sentences."""
        if not chunks:
            return []

        refined_items: List[RefinedKnowledge] = []

        for chunk in chunks:
            provenance = self._format_provenance(chunk)

            # If the chunk is a table, keep table structure intact
            if chunk.chunk_type == "table":
                refined_items.append(
                    RefinedKnowledge(
                        source_type="internal",
                        content=chunk.text,
                        provenance=provenance,
                    )
                )
                continue

            # For regular text, decompose into sentences and rank them
            sentences = self._split_into_sentences(chunk.text)
            if len(sentences) <= 1:
                refined_items.append(
                    RefinedKnowledge(
                        source_type="internal",
                        content=chunk.text,
                        provenance=provenance,
                    )
                )
                continue

            # Score each sentence against the query
            pairs = [(query, sent) for sent in sentences]
            scores = self.reranker._model.predict(pairs)
            ranked_sentences = sorted(
                zip(sentences, scores),
                key=lambda x: x[1],
                reverse=True,
            )

            # Pick top sentences above a relevance baseline
            top_sentences = [
                sent for sent, score in ranked_sentences[:self.max_strips] if score > -8.0
            ]
            if not top_sentences and ranked_sentences:
                top_sentences = [ranked_sentences[0][0]]

            refined_content = " ".join(top_sentences)
            refined_items.append(
                RefinedKnowledge(
                    source_type="internal",
                    content=refined_content,
                    provenance=provenance,
                )
            )

        logger.info(f"Refined {len(chunks)} internal chunks into {len(refined_items)} knowledge units.")
        return refined_items

    def refine_web_results(self, web_results: List[WebSearchResult]) -> List[RefinedKnowledge]:
        """Converts raw web snippets into structured external knowledge strips."""
        refined_items: List[RefinedKnowledge] = []
        for res in web_results:
            clean_snippet = res.snippet.strip()
            provenance = f"{res.title} ({res.url})"
            refined_items.append(
                RefinedKnowledge(
                    source_type="web",
                    content=clean_snippet,
                    provenance=provenance,
                )
            )
        return refined_items
