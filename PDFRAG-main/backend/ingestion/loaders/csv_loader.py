import csv  
from typing import List  
from langchain_core.documents import Document  
from backend.utils.logger import get_logger

log = get_logger("csv_loader")


class CSVLoader:

    @staticmethod  
    def load(file_path: str, filename: str, doc_id: str, clean_fn=None) -> List[Document]:  
        log.info(f"[LOAD CSV] {filename}")

        documents: List[Document] = []  
        page_num = 1  
        logical_page_size = 500

        try:  
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:  
                reader = csv.DictReader(f)  
                headers = reader.fieldnames or []

                log.info(f"[LOAD CSV] Headers: {headers}")

                buffer = []  
                buffer_len = 0

                for row_num, row in enumerate(reader, start=1):  
                    row_text = " | ".join([  
                        f"{key}: {value}"  
                        for key, value in row.items()  
                        if value and str(value).strip()  
                    ])

                    if clean_fn:  
                        row_text = clean_fn(row_text)

                    if not row_text:  
                        continue

                    buffer.append(row_text)  
                    buffer_len += len(row_text)

                    if buffer_len >= logical_page_size:  
                        content = "\n".join(buffer)

                        documents.append(  
                            Document(  
                                page_content=content,  
                                metadata={  
                                    "doc_id": doc_id,  
                                    "filename": filename,  
                                    "source": filename,  
                                    "file_type": "csv",  
                                    "page": int(page_num),  
                                },  
                            )  
                        )

                        page_num += 1  
                        buffer = []  
                        buffer_len = 0

                # Remaining rows  
                if buffer:  
                    content = "\n".join(buffer)

                    documents.append(  
                        Document(  
                            page_content=content,  
                            metadata={  
                                "doc_id": doc_id,  
                                "filename": filename,  
                                "source": filename,  
                                "file_type": "csv",  
                                "page": int(page_num),  
                            },  
                        )  
                    )

            log.info(f"[LOAD CSV] ✅ {filename} — {len(documents)} logical pages extracted")

        except Exception as e:  
            log.error(f"[LOAD CSV] ❌ FAILED for {filename} — {type(e).__name__}: {e}", exc_info=True)

        return [d for d in documents if d.page_content.strip()]  
