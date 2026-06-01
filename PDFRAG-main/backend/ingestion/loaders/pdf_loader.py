from typing import List  
from langchain_community.document_loaders import PyMuPDFLoader  
from langchain_core.documents import Document  
from backend.utils.logger import get_logger

log = get_logger("pdf_loader")


class PDFLoader:

    @staticmethod  
    def load(file_path: str, filename: str, doc_id: str, clean_fn=None) -> List[Document]:  
        log.info(f"[LOAD PDF] {filename}")

        try:  
            loader = PyMuPDFLoader(file_path)  
            pages: List[Document] = loader.load()

            for page in pages:  
                page.metadata["doc_id"] = doc_id  
                page.metadata["filename"] = filename  
                page.metadata["source"] = filename  
                page.metadata["file_type"] = "pdf"  
                page.metadata["page"] = int(page.metadata.get("page", 0)) + 1

                if clean_fn:  
                    page.page_content = clean_fn(page.page_content)

            pages = [p for p in pages if p.page_content.strip()]

            log.info(f"[LOAD PDF] ✅ {filename} — {len(pages)} non-empty pages")  
            return pages

        except Exception as e:  
            log.error(f"[LOAD PDF] ❌ FAILED for {filename} — {type(e).__name__}: {e}", exc_info=True)  
            return []  
