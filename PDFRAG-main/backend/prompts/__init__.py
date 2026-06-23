"""
Centralized prompt library.

All LLM prompt text for the backend lives here, separated from the logic that calls
the models. Import the builder/constant you need from this package; never embed prompt
text inline in generator/classifier/summarizer code.

    from backend.prompts import get_system_prompt, build_map_system_prompt
"""

from backend.prompts.generation import (
    BASE_FORMAT,
    INTENT_PROMPTS,
    get_system_prompt,
)
from backend.prompts.map_reduce import (
    build_map_system_prompt,
    build_reduce_system_prompt,
)
from backend.prompts.classification import (
    build_classification_context,
    build_subquery_classification_prompt,
)
from backend.prompts.summarization import (
    SUMMARIZER_SYSTEM_PROMPT,
    build_summary_user_prompt,
)

__all__ = [
    "BASE_FORMAT",
    "INTENT_PROMPTS",
    "get_system_prompt",
    "build_map_system_prompt",
    "build_reduce_system_prompt",
    "build_classification_context",
    "build_subquery_classification_prompt",
    "SUMMARIZER_SYSTEM_PROMPT",
    "build_summary_user_prompt",
]
