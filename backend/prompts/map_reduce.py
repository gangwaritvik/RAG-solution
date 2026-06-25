"""
Map-reduce prompts — instructions appended to the intent system prompt during the
parallel MAP (per-batch extraction) and REDUCE (merge partials) steps of generation.

Each builder takes the intent-specific ``base_system`` prompt and returns the full
system prompt for that step, so the generator never embeds this wording inline.
"""


def build_map_system_prompt(base_system: str, index: int, total: int) -> str:
    """System prompt for a single MAP batch (extract relevant content from a chunk slice)."""
    return (
        base_system
        + f"\n\nMAP STEP (batch {index}/{total}): You are seeing only PART of the "
          "document chunks. Evaluate EACH chunk in this batch INDIVIDUALLY — do NOT "
          "judge the batch as a whole. For every chunk, one at a time:\n"
          "  • If that chunk contains anything relevant to the question, extract its "
          "relevant facts/points and keep its [Source | Page | Chunk] label.\n"
          "  • If that chunk is not relevant, simply skip it and move to the next.\n"
          "A single relevant chunk is enough to keep — never discard a relevant chunk "
          "just because other chunks in this batch are irrelevant. "
          "Extract EVERY matching item present in the relevant chunks, IN FULL and "
          "VERBATIM (every row, entry, or list item with its identifier) — never a "
          "representative subset, sample, or summary, and never collapse multiple items "
          "into one. "
          "If the request targets items the source demarcates with a specific label, "
          "heading, or marker, include ONLY passages that actually carry that marker; "
          "do NOT treat ordinary body text, formulas, or general statements that merely "
          "resemble them as matches. "
          "Do NOT write a preamble, do NOT say information is missing or incomplete "
          "(other batches cover the rest), and do NOT add a MEMORY_SUMMARY. "
          "Reply with exactly: NONE — only if, after checking EVERY chunk individually, "
          "not a single chunk was relevant."
    )


def build_reduce_system_prompt(base_system: str) -> str:
    """System prompt for the REDUCE step (merge all MAP partials into one final answer)."""
    return (
        base_system
        + "\n\nREDUCE STEP: The context below contains partial findings extracted IN "
          "PARALLEL from different sections of the source document(s). Combine "
          "them into ONE unified answer that includes EVERY distinct item from ALL partials "
          "— do NOT omit, skip, shorten, or summarize away any item. The ONLY thing you may "
          "remove is an EXACT duplicate of the same item appearing in more than one partial "
          "(keep a single copy, merging the fullest detail). Preserve each item's "
          "identifier and the source's original order; if items are numbered, keep the "
          "sequence continuous with no missing entries. Use ONLY information that actually "
          "appears in these partial findings — never introduce topics, sections, themes, or "
          "facts that are not present in them. Follow your formatting rules. "
          "Do NOT mention this merging, the partials, or the chunks, and do NOT add any "
          "preamble — output ONLY the final answer content."
    )
