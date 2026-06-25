import fitz  # PyMuPDF
from typing import List
from langchain_core.documents import Document  
from backend.config import (
    VISION_ENABLED, VISION_DPI, VISION_CAPTION_FIGURES, VISION_MAX_WORKERS,
    FIGURE_MIN_WIDTH, FIGURE_MIN_HEIGHT, FIGURE_MAX_PER_PAGE,
)
from backend.processing.parallel_executor import ParallelExecutor
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

    # Math-symbol font families that, embedded as Type0/Identity-H subsets, carry
    # radicals, large brackets, fractions and operators which PyMuPDF cannot map back to
    # real characters — so their formulas drop out of the extracted text layer.
    _MATH_SYMBOL_FONTS = ("Euclid", "Symbol", "CMSY", "CMEX", "MSAM", "MSBM", "Math")

    @staticmethod
    def _page_has_broken_math(page) -> bool:
        """
        Detect whether a page's math/symbols failed to extract as real text.

        Two independent signatures of broken math extraction trigger a vision re-read:

          1. Private-Use-Area glyphs (U+E000–U+F8FF): embedded symbol glyphs that mapped
             to no real character (e.g. a big bracket coming through as ``\uf0e6``).
          2. A Type0/Identity-H math-symbol font on the page (EuclidSymbol, an Identity-H
             Symbol subset, the TeX CM/MS families, …). These layout fonts carry
             roots/fractions/large operators and routinely fail to map, dropping whole
             formulas from the text layer. Plain Type1 ``Symbol`` used only for α/β is NOT
             Type0, so ordinary Greek-letter pages do not trigger.

        Pure-text pages have neither signature, so they never incur a (paid) vision call.
        """
        # Signal 1 — unmapped Private-Use-Area glyphs.
        try:
            raw = page.get_text("rawdict")
            for block in raw.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        for ch in span.get("chars", []):
                            c = ch.get("c", "")
                            if c and 0xE000 <= ord(c[0]) <= 0xF8FF:
                                return True
        except Exception:
            pass

        # Signal 2 — a Type0/Identity-H math-symbol font is present.
        try:
            for font in page.get_fonts(full=True):
                ftype = font[2]
                basefont = font[3] or ""
                base_short = basefont.split("+")[-1]
                if ftype == "Type0" and any(k in base_short for k in PDFLoader._MATH_SYMBOL_FONTS):
                    return True
        except Exception:
            pass

        return False

    @staticmethod
    def _render_page_png(page, page_num: int) -> bytes:
        """
        Render a whole page to PNG bytes (for vision transcription). Done on the MAIN
        thread because PyMuPDF page access is not thread-safe; the resulting bytes are then
        safe to hand to parallel vision workers. Returns b"" on failure.
        """
        try:
            zoom = max(1.0, VISION_DPI / 72.0)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            return pix.tobytes("png")
        except Exception as e:
            log.warning(f"[LOAD PDF] Page {page_num + 1}: failed to render for vision "
                        f"({type(e).__name__}: {e})")
            return b""

    @staticmethod
    def _vision_task(task: dict, index: int, total: int) -> dict:
        """
        Parallel vision worker (runs in a thread). Operates ONLY on plain PNG bytes — never
        touches the PyMuPDF document — so it is thread-safe. Dispatches on task['kind']:
        'math' → faithful page transcription; 'figure' → figure caption.
        """
        from backend.ingestion.vision_captioner import get_vision_captioner
        captioner = get_vision_captioner()
        kind = task.get("kind")
        if kind == "math":
            text = captioner.transcribe_page(task["png"])
            return {"kind": "math", "page_idx": task["page_idx"], "text": text}
        # figure
        caption = captioner.caption_figure(task["png"])
        return {
            "kind": "figure",
            "page_idx": task["page_idx"],
            "fig_seq": task["fig_seq"],
            "png": task["png"],
            "caption": caption,
        }

    # ──────────────────────────────────────────────────────────────────
    #  FIGURE / DIAGRAM DETECTION + CAPTIONING
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _cluster_rects(rects: List["fitz.Rect"], gap: float) -> List[tuple]:
        """
        Greedily group nearby rectangles into clusters.

        Vector diagrams (graphs, charts) are drawn as MANY small path primitives; a single
        primitive is meaningless, but their union is the figure. Rectangles whose expanded
        bounds touch are merged. Returns a list of (bounding_rect, member_count).
        """
        remaining = list(rects)
        clusters = []
        while remaining:
            base = remaining.pop(0)
            bbox = fitz.Rect(base)
            count = 1
            changed = True
            while changed:
                changed = False
                still = []
                for r in remaining:
                    expanded = fitz.Rect(bbox.x0 - gap, bbox.y0 - gap, bbox.x1 + gap, bbox.y1 + gap)
                    if expanded.intersects(r):
                        bbox |= r  # union
                        count += 1
                        changed = True
                    else:
                        still.append(r)
                remaining = still
            clusters.append((bbox, count))
        return clusters

    @staticmethod
    def _detect_figures(page) -> List["fitz.Rect"]:
        """
        Detect figure regions on a page: embedded raster images plus clusters of vector
        drawings large enough to be a real diagram. Size-gated so rules, underlines and
        stray glyphs (e.g. a √ sign) are never treated as figures.
        """
        page_area = max(1.0, page.rect.width * page.rect.height)
        found: List[fitz.Rect] = []

        # 1) Raster images (photos, embedded figures).
        try:
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    for r in page.get_image_rects(xref):
                        rr = fitz.Rect(r)
                        if rr.width >= FIGURE_MIN_WIDTH and rr.height >= FIGURE_MIN_HEIGHT:
                            found.append(rr)
                except Exception:
                    continue
        except Exception:
            pass

        # 2) Vector diagram clusters (graphs/charts built from many path primitives).
        try:
            draw_rects = []
            for d in page.get_drawings():
                r = d.get("rect")
                if r is not None and r.width > 1 and r.height > 1:
                    draw_rects.append(fitz.Rect(r))
            for bbox, count in PDFLoader._cluster_rects(draw_rects, gap=20.0):
                area = bbox.width * bbox.height
                # A genuine diagram: several primitives, both dimensions sizable, and not
                # the whole-page frame (area < 90% of page).
                if (count >= 4
                        and bbox.width >= FIGURE_MIN_WIDTH
                        and bbox.height >= FIGURE_MIN_HEIGHT
                        and area < page_area * 0.9):
                    found.append(bbox)
        except Exception:
            pass

        return PDFLoader._merge_overlapping(found)

    @staticmethod
    def _merge_overlapping(rects: List["fitz.Rect"]) -> List["fitz.Rect"]:
        """Merge rectangles that overlap substantially so a raster sitting inside a vector
        cluster (or two detections of the same figure) isn't captioned twice."""
        merged: List[fitz.Rect] = []
        for r in sorted(rects, key=lambda x: (-(x.width * x.height))):
            absorbed = False
            for m in merged:
                inter = fitz.Rect(m) & r
                if inter.width > 0 and inter.height > 0:
                    smaller = min(r.width * r.height, m.width * m.height)
                    if smaller > 0 and (inter.width * inter.height) / smaller > 0.5:
                        m |= r
                        absorbed = True
                        break
            if not absorbed:
                merged.append(fitz.Rect(r))
        return merged

    @staticmethod
    def _crop_figures(page, page_num: int) -> List[bytes]:
        """
        Detect figures on a page and crop each to PNG bytes. Done on the MAIN thread (it
        accesses the PyMuPDF page); the returned bytes are then captioned in parallel. The
        list order is stable so figures get deterministic p{page}_f{n} numbering later.
        Returns [] when the page has no figures.
        """
        rects = PDFLoader._detect_figures(page)
        if not rects:
            return []

        zoom = max(1.0, VISION_DPI / 72.0)
        crops: List[bytes] = []
        for rect in rects[:FIGURE_MAX_PER_PAGE]:
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect)
                crops.append(pix.tobytes("png"))
            except Exception as e:
                log.warning(f"[LOAD PDF] Page {page_num + 1}: figure crop failed "
                            f"({type(e).__name__}: {e})")
        return crops

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

            do_figures = VISION_ENABLED and VISION_CAPTION_FIGURES

            # ── PASS 1 (main thread): all PyMuPDF access (NOT thread-safe). Extract text,
            # detect tables, and render/crop the images that need a vision call into plain
            # PNG bytes. Collect every vision call (math + figures) as a task so they can be
            # run together in parallel afterwards. ──
            page_records = []   # per page: {text, table_blocks, needs_math, n_figs}
            vision_tasks = []   # flat list of {kind, page_idx, ...png...}
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                text = PDFLoader._extract_page_text(page)
                rec = {"text": text, "table_blocks": "", "needs_math": False, "n_figs": 0}

                # MATH FALLBACK: broken-math page → queue a whole-page transcription that
                # will REPLACE the garbled extraction. Skip the table pass for such pages
                # (the transcription already covers the whole page).
                if VISION_ENABLED and PDFLoader._page_has_broken_math(page):
                    png = PDFLoader._render_page_png(page, page_num)
                    if png:
                        rec["needs_math"] = True
                        vision_tasks.append({"kind": "math", "page_idx": page_num, "png": png})
                        log.info(f"[LOAD PDF] Page {page_num + 1}: broken math detected — queued for vision transcription")

                # TABLES: only when the page isn't being fully transcribed by vision.
                if not rec["needs_math"] and hasattr(page, "find_tables"):
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
                            rec["table_blocks"] = "\n\n".join(table_blocks)
                            log.info(f"[LOAD PDF] Page {page_num + 1}: extracted {table_count} table(s)")
                    except Exception as table_err:
                        log.warning(
                            f"[LOAD PDF] Page {page_num + 1}: table extraction skipped "
                            f"({type(table_err).__name__}: {table_err})"
                        )

                # FIGURES: crop each detected figure to bytes; queue one caption task each.
                if do_figures:
                    try:
                        crops = PDFLoader._crop_figures(page, page_num)
                        for k, crop in enumerate(crops):
                            vision_tasks.append({
                                "kind": "figure", "page_idx": page_num, "fig_seq": k, "png": crop,
                            })
                        rec["n_figs"] = len(crops)
                    except Exception as fig_err:
                        log.warning(
                            f"[LOAD PDF] Page {page_num + 1}: figure detection skipped "
                            f"({type(fig_err).__name__}: {fig_err})"
                        )

                page_records.append(rec)

            pdf.close()  # all PyMuPDF access done; bytes are now self-contained

            # ── PASS 2 (parallel): run EVERY vision call (math transcriptions + figure
            # captions) concurrently in moderate waves, instead of a long sequential chain.
            # Workers touch only PNG bytes, so this is thread-safe. ──
            math_text_by_page = {}
            figs_by_page = {}   # page_idx → list of (fig_seq, png, caption)
            if vision_tasks:
                log.info(f"[LOAD PDF] {filename}: running {len(vision_tasks)} vision call(s) in parallel "
                         f"(max_workers={VISION_MAX_WORKERS})")
                results = ParallelExecutor.execute_parallel(
                    tasks=vision_tasks,
                    task_func=PDFLoader._vision_task,
                    max_workers=VISION_MAX_WORKERS,
                    operation_name="vision-ingest",
                )
                for r in results:
                    if not r or r.get("error"):
                        continue
                    if r.get("kind") == "math":
                        if r.get("text"):
                            math_text_by_page[r["page_idx"]] = r["text"]
                    elif r.get("kind") == "figure":
                        if r.get("caption"):
                            figs_by_page.setdefault(r["page_idx"], []).append(
                                (r["fig_seq"], r["png"], r["caption"])
                            )

            # ── PASS 3 (main thread): stitch results in page order. Math transcription
            # REPLACES page text; tables stay inline. Each captioned figure becomes its OWN
            # text chunk (the caption), so the figure's information is searchable on its own
            # merits. Only the caption DATA is kept — the cropped image is not stored. ──
            figure_documents: List[Document] = []
            for page_num, rec in enumerate(page_records):
                text = rec["text"]

                # Math transcription REPLACES the page text when it succeeded; otherwise the
                # original extraction (plus any tables) is kept.
                if rec["needs_math"] and page_num in math_text_by_page:
                    text = math_text_by_page[page_num]
                    log.info(f"[LOAD PDF] Page {page_num + 1}: ✅ vision transcription applied "
                             f"({len(text)} chars)")
                elif rec["table_blocks"]:
                    text = f"{text}\n\n{rec['table_blocks']}" if text.strip() else rec["table_blocks"]

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

                # Each captioned figure on this page → its OWN text chunk (caption only).
                page_figs = sorted(figs_by_page.get(page_num, []), key=lambda x: x[0])
                if page_figs:
                    for _seq, _png, caption in page_figs:
                        body = f"Figure (page {page_num + 1}): {caption.strip()}"
                        figure_documents.append(
                            Document(
                                page_content=body,
                                metadata={
                                    "doc_id": doc_id,
                                    "filename": filename,
                                    "source": filename,
                                    "file_type": "pdf",
                                    "page": int(page_num + 1),
                                },
                            )
                        )
                    log.info(f"[LOAD PDF] Page {page_num + 1}: captioned {len(page_figs)} figure(s) as text chunk(s)")

            # Append figure Documents AFTER the page Documents so page order is preserved for
            # positional intents; each figure still carries its own page in metadata.
            documents.extend(figure_documents)

            log.info(f"[LOAD PDF] ✅ {filename} — {len(documents)} document(s) "
                     f"({len(figure_documents)} figure caption chunk(s))")

        except Exception as e:  
            log.error(f"[LOAD PDF] ❌ FAILED for {filename} — {type(e).__name__}: {e}", exc_info=True)

        return documents  
