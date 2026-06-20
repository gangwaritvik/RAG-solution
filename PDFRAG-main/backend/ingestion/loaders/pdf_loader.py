import fitz  # PyMuPDF
from typing import List
from langchain_core.documents import Document  
from backend.utils.logger import get_logger

log = get_logger("pdf_loader")


class PDFLoader:

    @staticmethod
    def _clean_cell(value) -> str:
        """Normalize table cell text for stable downstream retrieval."""
        if value is None:
            return ""
        return " ".join(str(value).replace("\n", " ").split()).strip()

    @staticmethod
    def _looks_like_header(row: List[str]) -> bool:
        """Heuristic: a header row usually contains alphabetic labels."""
        if not row:
            return False
        non_empty = [c for c in row if c]
        if not non_empty:
            return False
        alpha_cells = sum(any(ch.isalpha() for ch in cell) for cell in non_empty)
        return alpha_cells >= max(1, len(non_empty) // 2)

    @staticmethod
    def _unique_headers(headers: List[str]) -> List[str]:
        """Ensure deterministic, unique, non-empty header names."""
        out = []
        counts = {}
        for i, h in enumerate(headers):
            base = h or f"Column_{i + 1}"
            counts[base] = counts.get(base, 0) + 1
            out.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
        return out

    @staticmethod
    def _normalize_table(rows: List[List[str]]) -> tuple[List[str], List[List[str]]]:
        """
        Normalize raw table rows into stable schema + data rows.

        - pads/truncates rows to fixed width
        - infers header row when possible
        - propagates carry-down values for merged/blank cells
        """
        cleaned = []
        for row in rows:
            norm_row = [PDFLoader._clean_cell(c) for c in row]
            if any(norm_row):
                cleaned.append(norm_row)

        if not cleaned:
            return [], []

        col_count = max(len(r) for r in cleaned)
        cleaned = [r + [""] * (col_count - len(r)) for r in cleaned]

        first = cleaned[0]
        if PDFLoader._looks_like_header(first):
            headers = PDFLoader._unique_headers(first)
            data_rows = [r[:] for r in cleaned[1:]]
        else:
            headers = [f"Column_{i + 1}" for i in range(col_count)]
            data_rows = [r[:] for r in cleaned]

        # Carry-down propagation: when a cell is blank, inherit previous non-empty
        # value in that column (common in merged-cell tables).
        carry = [""] * col_count
        for row in data_rows:
            for c in range(col_count):
                if row[c]:
                    carry[c] = row[c]
                elif carry[c]:
                    row[c] = carry[c]

        return headers, data_rows

    @staticmethod
    def _table_to_markdown(headers: List[str], data_rows: List[List[str]]) -> str:
        """Render normalized table into markdown for robust semantic retrieval."""
        if not headers:
            return ""
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in data_rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines).strip()

    @staticmethod
    def _extract_page_text(page) -> str:
        """
        Extract page text using PyMuPDF block geometry with dynamic N-column ordering.

        Strategy:
        - Separate full-width (spanning) text blocks from column blocks
        - Cluster column blocks by x-center (left-to-right columns)
        - Read blocks top-to-bottom within each column
        - Preserve top spanning blocks (titles/headings) before columns
        """
        blocks = page.get_text("blocks") or []
        text_blocks = [b for b in blocks if len(b) >= 7 and int(b[6]) == 0 and str(b[4]).strip()]

        if not text_blocks:
            return ""

        page_width = max(1.0, float(page.rect.width))
        spanning = []
        column_blocks = []

        for b in text_blocks:
            x0, y0, x1, y1, txt = float(b[0]), float(b[1]), float(b[2]), float(b[3]), str(b[4]).strip()
            width = max(1.0, x1 - x0)
            width_ratio = width / page_width
            center_x = (x0 + x1) / 2.0
            rec = (x0, y0, x1, y1, txt, center_x, width)

            if width_ratio >= 0.75:
                spanning.append(rec)
            else:
                column_blocks.append(rec)

        if not column_blocks:
            ordered = sorted(spanning, key=lambda r: (r[1], r[0]))
            return "\n".join(r[4] for r in ordered if r[4]).strip()

        # Cluster by x-center using a gap threshold derived from page width.
        centers_sorted = sorted(column_blocks, key=lambda r: r[5])
        gap_threshold = page_width * 0.12
        clusters = [[centers_sorted[0]]]
        for rec in centers_sorted[1:]:
            prev = clusters[-1][-1]
            if (rec[5] - prev[5]) > gap_threshold:
                clusters.append([rec])
            else:
                clusters[-1].append(rec)

        # Merge tiny noise clusters into nearest major cluster.
        if len(clusters) > 1:
            major = [c for c in clusters if len(c) >= 2]
            if major:
                major_centers = [sum(r[5] for r in c) / len(c) for c in major]
                merged = [list(c) for c in major]
                for c in clusters:
                    if len(c) >= 2:
                        continue
                    c_center = sum(r[5] for r in c) / len(c)
                    nearest_idx = min(range(len(major_centers)), key=lambda i: abs(major_centers[i] - c_center))
                    merged[nearest_idx].extend(c)
                clusters = merged

        # Sort clusters left->right; blocks in each cluster top->bottom.
        clusters = sorted(clusters, key=lambda c: sum(r[5] for r in c) / len(c))
        ordered_columns = [sorted(c, key=lambda r: (r[1], r[0])) for c in clusters]

        first_col_y = min(col[0][1] for col in ordered_columns if col) if ordered_columns else 0.0
        top_spanning = sorted([r for r in spanning if r[3] <= (first_col_y + 15)], key=lambda r: (r[1], r[0]))
        rest_spanning = sorted([r for r in spanning if r[3] > (first_col_y + 15)], key=lambda r: (r[1], r[0]))

        ordered = top_spanning
        for col in ordered_columns:
            ordered.extend(col)
        ordered.extend(rest_spanning)

        return "\n".join(r[4] for r in ordered if r[4]).strip()

    @staticmethod
    def _table_to_text(table) -> str:
        """Convert a detected PyMuPDF table to normalized markdown text."""
        try:
            rows = table.extract() or []
        except Exception:
            rows = []

        headers, data_rows = PDFLoader._normalize_table(rows)
        return PDFLoader._table_to_markdown(headers, data_rows)

    @staticmethod  
    def load(file_path: str, filename: str, doc_id: str, clean_fn=None) -> List[Document]:  
        """
        Load PDF using PyMuPDF (fitz) text + table extraction.
        
        Args:
            file_path: Path to PDF file
            filename: Name of the file
            doc_id: Document ID
            clean_fn: Optional cleaning function
        Returns:
            List of Document objects with text content
        """
        log.info(f"[LOAD PDF] {filename} — using PyMuPDF (text + tables)")
        documents: List[Document] = []

        try:  
            pdf = fitz.open(file_path)
            log.info(f"[LOAD PDF] {len(pdf)} pages found")

            for page_num in range(len(pdf)):  
                page = pdf[page_num]
                
                # Extract text from page via strict single-path PyMuPDF extraction.
                text = PDFLoader._extract_page_text(page)

                # Optionally include table content detected by PyMuPDF.
                # We append table rows to page text so retrieval can surface tabular facts.
                if hasattr(page, "find_tables"):
                    try:
                        found = page.find_tables()
                        tables = getattr(found, "tables", found)
                        table_count = 0
                        table_blocks = []
                        for table in tables:
                            table_text = PDFLoader._table_to_text(table)
                            if table_text:
                                table_count += 1
                                table_blocks.append(
                                    f"[TABLE {table_count}]\n{table_text}\n[/TABLE {table_count}]"
                                )

                        if table_blocks:
                            text = f"{text}\n\n" + "\n\n".join(table_blocks)
                            log.info(
                                f"[LOAD PDF] Page {page_num + 1}: extracted {table_count} table(s)"
                            )
                    except Exception as table_err:
                        log.warning(
                            f"[LOAD PDF] Page {page_num + 1}: table extraction skipped "
                            f"({type(table_err).__name__}: {table_err})"
                        )
                
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
