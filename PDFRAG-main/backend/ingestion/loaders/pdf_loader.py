import fitz  # PyMuPDF
from typing import List, Optional  
from langchain_core.documents import Document  
from backend.utils.logger import get_logger

log = get_logger("pdf_loader")


class PDFLoader:

    @staticmethod  
    def load(file_path: str, filename: str, doc_id: str, clean_fn=None, extract_tables: bool = False) -> List[Document]:  
        """
        Load PDF using PyMuPDF (fitz) for simple, fast text extraction.
        Table extraction parameter is kept for API compatibility but ignored.
        
        Args:
            file_path: Path to PDF file
            filename: Name of the file
            doc_id: Document ID
            clean_fn: Optional cleaning function
            extract_tables: Ignored (kept for API compatibility)
            
        Returns:
            List of Document objects with text content
        """
        log.info(f"[LOAD PDF] {filename} — using PyMuPDF (simple text extraction)")
        documents: List[Document] = []

        try:  
            pdf = fitz.open(file_path)
            log.info(f"[LOAD PDF] {len(pdf)} pages found")

            for page_num in range(len(pdf)):  
                page = pdf[page_num]
                
                # Extract text from page
                text = page.get_text()
                
                if clean_fn:  
                    text = clean_fn(text)

                if text.strip():  
                    documents.append(  
                        Document(  
                            page_content=text.strip(),  
                            metadata={  
                                "doc_id": doc_id,  
                                "filename": filename,  
                                "source": filename,  
                                "file_type": "pdf",  
                                "page": int(page_num + 1),  
                            },  
                        )  
                    )

            pdf.close()
            log.info(f"[LOAD PDF] ✅ {filename} — {len(documents)} non-empty pages extracted")

        except Exception as e:  
            log.error(f"[LOAD PDF] ❌ FAILED for {filename} — {type(e).__name__}: {e}", exc_info=True)

        return documents  
