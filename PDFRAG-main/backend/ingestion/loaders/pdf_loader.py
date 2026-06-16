import pdfplumber  
from typing import List, Optional  
from langchain_core.documents import Document  
from backend.utils.logger import get_logger

log = get_logger("pdf_loader")


class PDFLoader:

    @staticmethod  
    def load(file_path: str, filename: str, doc_id: str, clean_fn=None) -> List[Document]:  
        log.info(f"[LOAD PDF] {filename}")

        documents: List[Document] = []

        try:  
            with pdfplumber.open(file_path) as pdf:  
                log.info(f"[LOAD PDF] {len(pdf.pages)} pages found")

                for page_num, page in enumerate(pdf.pages, start=1):  
                    page_parts = []

                    # ── Extract tables ──  
                    tables = page.extract_tables()  
                    table_count = len(tables) if tables else 0

                    # ── Extract non-table text ──  
                    non_table_text = PDFLoader._get_non_table_text(page, tables)  
                    if non_table_text:  
                        page_parts.append(non_table_text)

                    # ── Convert each table to readable text ──  
                    if tables:  
                        log.info(f"[LOAD PDF] Page {page_num} — {table_count} table(s) found")  
                        for t_idx, table in enumerate(tables):  
                            table_text = PDFLoader._table_to_text(table, t_idx + 1)  
                            if table_text:  
                                page_parts.append(table_text)

                    # ── Combine all parts ──  
                    combined = "\n\n".join(page_parts)

                    if clean_fn:  
                        combined = clean_fn(combined)

                    if combined.strip():  
                        documents.append(  
                            Document(  
                                page_content=combined,  
                                metadata={  
                                    "doc_id": doc_id,  
                                    "filename": filename,  
                                    "source": filename,  
                                    "file_type": "pdf",  
                                    "page": int(page_num),  
                                    "has_tables": table_count > 0,  
                                    "table_count": table_count,  
                                },  
                            )  
                        )

            log.info(f"[LOAD PDF] ✅ {filename} — {len(documents)} non-empty pages")

        except Exception as e:  
            log.error(f"[LOAD PDF] ❌ FAILED for {filename} — {type(e).__name__}: {e}", exc_info=True)

        return documents

    @staticmethod  
    def _table_to_text(table: list, table_num: int) -> str:  
        """Convert a pdfplumber table into readable key:value text."""  
        if not table or len(table) < 2:  
            return ""

        # First row = headers  
        headers = [  
            str(cell).strip() if cell is not None else f"Column {i+1}"  
            for i, cell in enumerate(table[0])  
        ]

        rows = []  
        for row in table[1:]:  
            cells = [  
                str(cell).strip() if cell is not None else ""  
                for cell in row  
            ]

            parts = []  
            for i, cell in enumerate(cells):  
                if cell:  
                    header = headers[i] if i < len(headers) else f"Column {i+1}"  
                    parts.append(f"{header}: {cell}")

            if parts:  
                rows.append(" | ".join(parts))

        if not rows:  
            return ""

        return f"[Table {table_num}]\n" + "\n".join(rows)

    @staticmethod  
    def _get_non_table_text(page, tables) -> str:  
        """Extract text from page areas outside tables."""  
        if not tables:  
            return (page.extract_text() or "").strip()

        try:  
            # Get bounding boxes of all tables  
            found_tables = page.find_tables()  
            table_bboxes = [t.bbox for t in found_tables]

            if not table_bboxes:  
                return (page.extract_text() or "").strip()

            # Get all words and filter out those inside table regions  
            words = page.extract_words()  
            outside_words = []

            for word in words:  
                inside = False  
                for bbox in table_bboxes:  
                    # bbox = (x0, top, x1, bottom)  
                    if (bbox[0] <= word["x0"] <= bbox[2] and  
                            bbox[1] <= word["top"] <= bbox[3]):  
                        inside = True  
                        break

                if not inside:  
                    outside_words.append(word["text"])

            return " ".join(outside_words).strip()

        except Exception:  
            return (page.extract_text() or "").strip()  
