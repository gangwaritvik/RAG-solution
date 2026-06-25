"""
Vision captioner — sends a page/figure image to the Azure vision model.

Used during ingestion when a PDF page's math/symbols did not extract as real text
(embedded symbol fonts produce a garbled or empty text layer). The page is rendered to
an image and re-read here, returning a faithful text + LaTeX transcription that replaces
the broken extraction.

Mirrors the AzureOpenAI client setup used by Embedder/Generator. A module-level
singleton avoids re-creating the HTTP client per file.
"""

import base64
from openai import AzureOpenAI

from backend.config import AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION, VISION_MODEL
from backend.prompts import MATH_TRANSCRIPTION_PROMPT, FIGURE_CAPTION_PROMPT
from backend.utils.logger import get_logger

log = get_logger("vision_captioner")

# A page render + vision pass is heavier than a text call; give it room but still bound it.
VISION_REQUEST_TIMEOUT = 90   # seconds per vision call
VISION_MAX_RETRIES = 2        # SDK-level retries on transient failures/timeouts


class VisionCaptioner:
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION,
            timeout=VISION_REQUEST_TIMEOUT,
            max_retries=VISION_MAX_RETRIES,
        )
        self.model = VISION_MODEL
        log.info(f"VisionCaptioner ✅ initialized | model: {self.model}")

    def transcribe_page(self, png_bytes: bytes) -> str:
        """
        Transcribe a full page image to faithful text + LaTeX.

        Returns the transcription string, or "" on any failure (caller then keeps the
        original extracted text so a vision error never drops the page).
        """
        if not png_bytes:
            return ""

        b64 = base64.b64encode(png_bytes).decode("ascii")
        messages = [
            {"role": "system", "content": MATH_TRANSCRIPTION_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Transcribe this document page faithfully, rendering all "
                            "mathematics in LaTeX. Do not solve or alter anything."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            },
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            log.error(f"[VISION] ❌ Page transcription failed — {type(e).__name__}: {e}")
            return ""

    def caption_figure(self, png_bytes: bytes, mime: str = "image/png") -> str:
        """
        Describe a single figure (raster image or vector diagram) for retrieval.

        ``mime`` is the image's content type (e.g. "image/png", "image/jpeg") so the data
        URI is labelled correctly — PDF crops are always PNG, but DOCX images may be JPEG.
        Returns the caption string, or "" on any failure (caller then skips this figure
        rather than letting a vision error abort ingestion).
        """
        if not png_bytes:
            return ""

        b64 = base64.b64encode(png_bytes).decode("ascii")
        messages = [
            {"role": "system", "content": FIGURE_CAPTION_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this figure for search and reasoning."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            },
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            log.error(f"[VISION] ❌ Figure caption failed — {type(e).__name__}: {e}")
            return ""


# ── Module-level singleton ──
_captioner_singleton = None


def get_vision_captioner() -> VisionCaptioner:
    """Return a shared VisionCaptioner, constructing it on first use (lazy so the
    AzureOpenAI client is only created when a page actually needs vision)."""
    global _captioner_singleton
    if _captioner_singleton is None:
        _captioner_singleton = VisionCaptioner()
    return _captioner_singleton
