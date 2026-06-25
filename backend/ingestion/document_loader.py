import os  
import re
import uuid  
from typing import List, Dict, Any

from backend.ingestion.loaders.pdf_loader import PDFLoader  
from backend.ingestion.loaders.docx_loader import DOCXLoader  
from backend.utils.logger import get_logger

log = get_logger("document_loader")

# Non-printable PDF-extraction artifacts (C0/C1 control chars like \x01, \x04, \x1f,
# \x88 and the Unicode replacement / box-drawing glyphs) that garble extracted text.
# Stripping them at ingestion keeps stored chunks AND their embeddings clean — not just
# the UI. Tab/newline/carriage-return are intentionally preserved.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufffd\u25a0\u2588]")

# Margin/security watermark artifact: some standards PDFs (e.g. ISO) print a vertical
# string of backticks, commas and hyphens down the page edge
# (e.g. --``,,,,,``````,,,,,`,`,`,`,,`,-`-`,,`,,`,`,,`---). PyMuPDF extracts it as text
# and it leaks into chunks/cells. This matches a run of backtick/comma/hyphen/space
# containing AT LEAST TWO backticks, so it never touches a markdown table separator
# ('| --- |', no backticks) or ordinary punctuation.
_MARGIN_NOISE_RE = re.compile(r"[`,\-\s]*`[`,\-\s]*`[`,\-\s]*")

# Supported extensions mapped to loader classes  
SUPPORTED_LOADERS = {  
    ".pdf":  PDFLoader,  
    ".docx": DOCXLoader
}


class DocumentLoader:  
    def __init__(self, upload_dir: str = "uploads"):  
        self.upload_dir = upload_dir  
        os.makedirs(upload_dir, exist_ok=True)  
        log.info(f"DocumentLoader initialized — upload_dir: {upload_dir}")  
        log.info(f"Supported formats: {list(SUPPORTED_LOADERS.keys())}")

    def save_file(self, filename: str, content: bytes) -> str:  
        ext = os.path.splitext(filename)[1].lower()  
        safe_name = f"{uuid.uuid4().hex}{ext}"  
        file_path = os.path.join(self.upload_dir, safe_name)

        with open(file_path, "wb") as f:  
            f.write(content)

        log.info(f"[SAVE] {filename} → {file_path} ({len(content)} bytes)")  
        return file_path

    def load_documents(self, files: List[tuple]) -> List[Dict[str, Any]]:  
        log.info(f"[LOAD] Processing {len(files)} file(s)")  
        results = []

        for filename, content in files:  
            file_path = self.save_file(filename, content)  
            ext = os.path.splitext(filename)[1].lower()  
            doc_id = uuid.uuid4().hex

            loader_class = SUPPORTED_LOADERS.get(ext)

            if loader_class is None:  
                log.warning(f"[LOAD] Unsupported file type: {filename}")  
                results.append({  
                    "doc_id": doc_id,  
                    "filename": filename,  
                    "file_type": ext.replace(".", ""),  
                    "total_pages": 0,  
                    "total_chars": 0,  
                    "documents": [],  
                    "errors": [f"Unsupported file type: {filename}"],  
                })  
                continue

            log.info(f"[LOAD] Using {loader_class.__name__} for {filename}")

            documents = loader_class.load(  
                file_path=file_path,  
                filename=filename,  
                doc_id=doc_id,  
                clean_fn=self._clean,  
            )

            results.append({  
                "doc_id": doc_id,  
                "filename": filename,  
                "file_type": ext.replace(".", ""),  
                "total_pages": len(documents),  
                "total_chars": sum(len(d.page_content) for d in documents),  
                "documents": documents,  
                "errors": [],  
            })

        return results

    @staticmethod  
    def _clean(text: str) -> str:  
        # Strip non-printable control-char artifacts so garbled PDF glyphs never reach
        # the stored chunk text or its embedding, then remove the margin/security
        # watermark run (replace with a space so adjacent words don't fuse), then
        # collapse whitespace onto single lines.
        text = _CONTROL_CHARS_RE.sub("", text)
        text = _MARGIN_NOISE_RE.sub(" ", text)
        lines = [line.strip() for line in text.splitlines() if line.strip()]  
        return " ".join(lines).strip()  
