"""Synthesis and Response Generation Engine for VeritasCRAG."""

import os
import re
from typing import List, Optional, Tuple
from loguru import logger

from veritas_crag.config.settings import settings
from veritas_crag.models.crag import CRAGContext, RefinedKnowledge
from veritas_crag.models.pipeline import Citation


class ResponseGenerator:
    """Generates strictly grounded answers with inline citation attribution."""

    SYSTEM_PROMPT = (
        "You are VeritasCRAG, an authoritative, factual document intelligence assistant.\n"
        "Generate a clear, direct, and concise answer to the user's question based STRICTLY "
        "on the provided verified knowledge strips.\n"
        "RULES:\n"
        "1. Every factual statement MUST cite its source using bracketed numbers e.g. [1], [2].\n"
        "2. Do NOT extrapolate, speculate, or introduce external facts outside the provided context.\n"
        "3. If the context does not contain sufficient facts to answer the question, state that clearly."
    )

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        self.provider = provider or settings.LLM_PROVIDER
        self.model_name = model_name or settings.LLM_MODEL_NAME
        self.gemini_api_key = gemini_api_key or settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
        self.openai_api_key = openai_api_key or settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")

    def _build_context_and_citations(
        self,
        context: CRAGContext,
    ) -> Tuple[str, List[Citation]]:
        """Compiles numbered knowledge strips and creates Citation objects."""
        all_knowledge: List[RefinedKnowledge] = (
            context.internal_knowledge + context.web_knowledge
        )
        citations: List[Citation] = []
        context_lines: List[str] = []

        for idx, item in enumerate(all_knowledge, start=1):
            citation = Citation(
                index=idx,
                source_type=item.source_type,
                provenance=item.provenance or "Unknown Source",
                snippet=item.content,
            )
            citations.append(citation)
            context_lines.append(f"[{idx}] {item.content} (Source: {citation.provenance})")

        compiled_context = "\n".join(context_lines)
        return compiled_context, citations

    def generate(
        self,
        context: CRAGContext,
    ) -> Tuple[str, List[Citation]]:
        """Generates an answer with citations from verified CRAG context.

        Returns:
            Tuple of (answer_text, active_citations).
        """
        compiled_context, all_citations = self._build_context_and_citations(context)

        if not all_citations:
            logger.warning("No verified knowledge available to generate answer.")
            return (
                "Unable to locate verified information to answer your query. "
                "Both internal document retrieval and external verification yielded insufficient facts.",
                [],
            )

        # 1. Try Gemini API if key is present
        if (self.provider in ["auto", "gemini"]) and self.gemini_api_key:
            try:
                answer = self._generate_gemini(context.query, compiled_context)
                active_citations = self._filter_active_citations(answer, all_citations)
                return answer, active_citations
            except Exception as e:
                logger.error(f"Gemini generation error: {e}. Falling back to local synthesizer.")

        # 2. Try OpenAI API if key is present
        if (self.provider in ["auto", "openai"]) and self.openai_api_key:
            try:
                answer = self._generate_openai(context.query, compiled_context)
                active_citations = self._filter_active_citations(answer, all_citations)
                return answer, active_citations
            except Exception as e:
                logger.error(f"OpenAI generation error: {e}. Falling back to local synthesizer.")

        # 3. Deterministic Local Grounded Synthesizer (Zero API keys required)
        logger.info("Using local grounded synthesis engine (offline / deterministic).")
        answer = self._generate_local_synthesis(context.query, all_citations)
        active_citations = self._filter_active_citations(answer, all_citations)
        return answer, active_citations

    def _generate_gemini(self, query: str, context_text: str) -> str:
        """Call Google Gemini API."""
        import google.generativeai as genai

        genai.configure(api_key=self.gemini_api_key)
        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.SYSTEM_PROMPT,
        )
        prompt = f"VERIFIED CONTEXT:\n{context_text}\n\nUSER QUESTION: {query}"
        resp = model.generate_content(prompt)
        return resp.text.strip()

    def _generate_openai(self, query: str, context_text: str) -> str:
        """Call OpenAI API."""
        from openai import OpenAI

        client = OpenAI(api_key=self.openai_api_key)
        resp = client.chat.completions.create(
            model=self.model_name or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"VERIFIED CONTEXT:\n{context_text}\n\nUSER QUESTION: {query}"},
            ],
            temperature=settings.TEMPERATURE,
        )
        return resp.choices[0].message.content.strip()

    @staticmethod
    def _generate_local_synthesis(query: str, citations: List[Citation]) -> str:
        """Deterministic local answer synthesis strictly grounded in citation strips."""
        if not citations:
            return "No verified context available to answer this question."

        points: List[str] = []
        for c in citations:
            # Clean snippet for answer body
            clean_snippet = c.snippet.strip()
            # If table format, present clearly
            if "Latency (ms)" in clean_snippet or "|" in clean_snippet:
                points.append(f"Based on benchmark data: {clean_snippet} [{c.index}]")
            else:
                points.append(f"{clean_snippet} [{c.index}]")

        answer_body = "\n\n".join(points)
        return answer_body

    @staticmethod
    def _filter_active_citations(answer: str, all_citations: List[Citation]) -> List[Citation]:
        """Finds all citations [n] actually referenced in the answer."""
        referenced_indices = set(map(int, re.findall(r"\[(\d+)\]", answer)))
        if not referenced_indices:
            return all_citations  # fallback to all provided if format omitted
        return [c for c in all_citations if c.index in referenced_indices]
