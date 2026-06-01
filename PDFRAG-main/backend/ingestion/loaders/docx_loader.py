from typing import List  
from langchain_core.documents import Document  
from backend.utils.logger import get_logger

log = get_logger("docx_loader")


class DOCXLoader:

    @staticmethod  
    def load(file_path: str, filename: str, doc_id: str, clean_fn=None) -> List[Document]:  
        log.info(f"[LOAD DOCX] {filename}")

        try:  
            from docx import Document as DocxDocument

            doc = DocxDocument(file_path)

            pages: List[Document] = []  
            page_num = 1  
            logical_page_size = 500

            buffer = []  
            buffer_len = 0

            for para in doc.paragraphs:  
                text = para.text.strip()  
                if not text:  
                    continue

                buffer.append(text)  
                buffer_len += len(text)

                if buffer_len >= logical_page_size:  
                    content = " ".join(buffer)  
                    if clean_fn:  
                        content = clean_fn(content)

                    if content:  
                        pages.append(  
                            Document(  
                                page_content=content,  
                                metadata={  
                                    "doc_id": doc_id,  
                                    "filename": filename,  
                                    "source": filename,  
                                    "file_type": "docx",  
                                    "page": int(page_num),  
                                },  
                            )  
                        )

                    page_num += 1  
                    buffer = []  
                    buffer_len = 0

            # Remaining text  
            if buffer:  
                content = " ".join(buffer)  
                if clean_fn:  
                    content = clean_fn(content)

                if content:  
                    pages.append(  
                        Document(  
                            page_content=content,  
                            metadata={  
                                "doc_id": doc_id,  
                                "filename": filename,  
                                "source": filename,  
                                "file_type": "docx",  
                                "page": int(page_num),  
                            },  
                        )  
                    )

            log.info(f"[LOAD DOCX] ✅ {filename} — {len(pages)} logical pages extracted")  
            return [p for p in pages if p.page_content.strip()]

        except Exception as e:  
            log.error(f"[LOAD DOCX] ❌ FAILED for {filename} — {type(e).__name__}: {e}", exc_info=True)  
            return []  
