"""
Summarization prompts — system + user prompts for the background conversation
summarizer that maintains a running per-group summary.
"""

from typing import Optional

SUMMARIZER_SYSTEM_PROMPT = (
    "You are a precise summarization expert. Produce brief, "
    "information-dense summaries that retain every concrete fact, "
    "claim, value, and conclusion. Never invent content; use only "
    "what the turns provide."
)


def build_summary_user_prompt(
    topic: str,
    turns_text: str,
    existing_summary: Optional[str] = None,
) -> str:
    """Build the user prompt that merges the previous summary with the new turns."""
    prompt_parts = [
        f"You are summarizing a conversation about: {topic}",
        "",
        "Recent conversation turns:",
        turns_text,
    ]

    if existing_summary:
        prompt_parts.insert(2, f"Previous summary: {existing_summary}")

    prompt = "\n".join(prompt_parts)
    prompt += (
        "\n\nWrite a running summary of this conversation by MERGING the previous "
        "summary with the new turns above. Requirements:\n"
        "- Keep it SHORT and BRIEF — no filler, no preamble, every line carries real "
        "information. Length should follow the content: only as long as needed.\n"
        "- Do NOT omit any specific claim, fact, numeric value, formula, definition, "
        "name, unit, or conclusion that was established. Brevity must NOT cost facts.\n"
        "- Preserve exact figures, units, formulas, and terminology verbatim.\n"
        "- Prefer compact bullet points over prose; de-duplicate repeated points.\n"
        "- Keep it self-contained so a later question can be answered from it alone."
    )
    return prompt
