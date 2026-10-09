"""Query Transformation and Rewriter for Web Search Fallback."""

import re
from typing import Optional
from loguru import logger

from veritas_crag.config.settings import settings


class QueryRewriter:
    """Transforms raw user queries into concise, high-yield web search queries."""

    # Common conversational prefixes to strip for search engine optimization
    CONVERSATIONAL_PREFIXES = [
        r"^can you (please )?(tell|explain|show) me\b",
        r"^what (is|are|was|were) the\b",
        r"^tell me (about)?\b",
        r"^i want to know\b",
        r"^please find\b",
        r"^could you clarify\b",
        r"^how (does|do|can|did)\b",
    ]

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm

    def rewrite(self, query: str, context_hint: Optional[str] = None) -> str:
        """Transforms a query for external search engines.

        Args:
            query: Original user query.
            context_hint: Optional additional context or document title.

        Returns:
            Optimized query string.
        """
        cleaned = query.strip()

        # Remove trailing question marks and punctuation
        cleaned = re.sub(r"[?!.,]+$", "", cleaned).strip()

        # Strip common conversational question prefixes
        for pattern in self.CONVERSATIONAL_PREFIXES:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

        # If context hint is provided, append relevant keyword
        if context_hint and context_hint.lower() not in cleaned.lower():
            transformed = f"{cleaned} {context_hint}".strip()
        else:
            transformed = cleaned

        logger.info(f"Query rewritten: '{query}' -> '{transformed}'")
        return transformed
