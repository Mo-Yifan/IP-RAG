# src/generation/prompts/__init__.py

from .base_prompt import BasePrompt
from .patent_prompt import PatentQAPrompt
from .citation_utils import format_citations, extract_unique_sources

__all__ = [
    "BasePrompt",
    "PatentQAPrompt",
    "format_citations",
    "extract_unique_sources"
]