"""
Vision prompts — system prompts used when a page image is sent to the vision model.

Kept here (not inline in the captioner) so all model-facing prompt text lives in the
backend.prompts package.
"""

# Faithful page transcription. Used when a PDF page's math/symbols did not extract as
# real text (embedded symbol fonts → garbled/empty text layer). The model re-reads the
# whole page from an image and returns clean text + LaTeX, which REPLACES the garbled
# extraction for that page.
MATH_TRANSCRIPTION_PROMPT = (
    "You transcribe a single document page from an image into clean, faithful text.\n\n"
    "RULES:\n"
    "- Reproduce ALL text on the page EXACTLY as written, in natural reading order "
    "(top to bottom; for a multi-column layout, finish the left column before the right).\n"
    "- Render every mathematical expression in LaTeX: inline math as \\( ... \\) and a "
    "standalone/displayed equation as \\[ ... \\]. Use proper LaTeX for roots "
    "(\\sqrt{...}), fractions (\\frac{...}{...}), exponents (x^{2}), subscripts (a_{n}), "
    "Greek letters (\\alpha, \\beta), and symbols (\\pm, \\times, \\leq, \\geq, \\neq).\n"
    "- Transcribe ONLY what is actually on the page. Do NOT solve problems, compute or "
    "fill in answers, add explanations, correct the content, or omit anything.\n"
    "- Preserve all numbering, labels, and option letters exactly as written "
    "(e.g. '1.', '(a)', '(i)', 'Q.3').\n"
    "- Ignore non-content page furniture: running headers/footers, page numbers, and "
    "watermark or margin noise.\n"
    "- For a figure, diagram, or graph that is not text, do NOT attempt to draw it — write "
    "a short bracketed note of what it depicts instead, e.g. "
    "[Figure: downward-opening parabola crossing the x-axis at two points].\n"
    "- Output ONLY the transcribed page content: no preamble, no commentary, no code fences."
)


# Figure/diagram caption. Used when a cropped figure (raster image or vector diagram) is
# sent to the vision model. The returned text is injected into the page as a [FIGURE n]
# marker so the figure is searchable by ordinary text queries, and is faithful enough that
# a later answer can rely on it.
FIGURE_CAPTION_PROMPT = (
    "You describe a single figure cropped from a document so it can be found by text search "
    "and reasoned about.\n\n"
    "RULES:\n"
    "- State what the figure IS (e.g. line graph, bar chart, parabola, circuit diagram, "
    "flowchart, photograph, table-as-image) and what it SHOWS — the key shapes, trends, "
    "relationships, or quantities.\n"
    "- Transcribe any text inside the figure EXACTLY: titles, axis labels, legends, data "
    "labels, and annotations. Render any mathematics in LaTeX (\\( ... \\)).\n"
    "- Be concise but specific (2–5 sentences). Describe only what is actually visible; do "
    "NOT invent values, causes, or context that the figure does not show.\n"
    "- Do NOT add a preamble or commentary — output only the description."
)

