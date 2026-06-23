# PDF-RAG — Architecture & Flow

A Retrieval-Augmented Generation system over PDF/DOCX documents. FastAPI backend
(with token streaming) + ChromaDB vector store + Azure OpenAI (gpt-4.1 + embeddings),
served to a vanilla JS frontend.

---

## 1. System Architecture (high level)

```mermaid
flowchart TB
    subgraph FE["Frontend (port 3000)"]
        UI[index.html / app.js / styles.css]
    end
    subgraph BE["FastAPI Backend (port 8000) — backend/app.py"]
        EP["Endpoints: /ingest /query /query/stream<br/>/status /delete /clear /group/*"]
    end
    subgraph PIPE["Pipeline singletons (backend/main.py)"]
        LOAD[DocumentLoader]
        CHUNK[Chunkers]
        EMB[Embedder]
        VS[(ChromaDB<br/>vector_store)]
        RET[Retriever]
        GEN[Generator]
        MEM[MemoryManager]
        CR[ContextResolver]
        BG[BackgroundSummarizer]
    end
    AZ[(Azure OpenAI<br/>gpt-4.1 + embeddings)]

    UI -->|HTTP / SSE| EP
    EP --> PIPE
    EMB <--> AZ
    GEN <--> AZ
    CR <--> AZ
    VS -.persist.- DISK[(storage/chroma_db)]
```

---

## 2. Ingestion flow (POST /ingest)

```mermaid
flowchart TD
    A[Upload PDF/DOCX + chunk_mode] --> B[/ingest: validate files/]
    B -->|invalid| E1[400 error]
    B -->|valid| C[Mark ingestion_status = processing]
    C --> D[Return 202 Accepted immediately]
    C --> BGT[Background thread]
    BGT --> L[Load document text]
    L --> CK{chunk_mode}
    CK -->|recursive| R1[RecursiveCharacterSplitter]
    CK -->|semantic| R2[SemanticChunker]
    CK -->|sliding| R3[SlidingWindow]
    CK -->|fixed| R4[FixedChunker]
    R1 & R2 & R3 & R4 --> EMB[Embed chunks - batch 500]
    EMB --> ADD[vector_store.add + filename cache]
    ADD --> DONE[ingestion_status = completed N chunks]
    DONE --> POLL[Frontend polls /status until done]
```

---

## 3. Query flow — the core pipeline (POST /query and /query/stream)

```mermaid
flowchart TD
    Q[User query] --> S1[STEP 1: ContextResolver.resolve]
    S1 --> CLS["LLM classify:<br/>dependency_type, retrieval_intent,<br/>answer_source, retrieval_scope,<br/>standalone_query"]
    CLS --> SCOPE{available_documents<br/>passed in?}
    SCOPE -->|whole-doc request + docs loaded| OK[resolvable]
    SCOPE -->|vague + no docs| AMB

    OK --> DEP{dependency_type}
    DEP -->|ambiguous| AMB[Skip retrieval →<br/>generate clarification]
    DEP -->|independent / dependent / multi_group| AS

    AS{answer_source}
    AS -->|previous_answer<br/>+ stored answer| PREV[Use previous answer as<br/>SOLE context, skip retrieval]
    AS -->|document| S2

    S2[STEP 2: Decide retrieval breadth] --> OV{Override priority}
    OV -->|manual Top-K on| UK[user K wins]
    OV -->|whole_document scope| ALL[retrieve ALL chunks auto]
    OV -->|summary/extraction/analysis| GAR[get_all_relevant]
    OV -->|else| DEF[per-intent default]

    UK & ALL & GAR & DEF --> RET[Retriever.retrieve]
    RET --> MODE{intent mode}
    MODE -->|top_k| M1[best N above threshold]
    MODE -->|comprehensive| M2[all above threshold, cap 15]
    MODE -->|exhaustive| M3[entire document]

    M1 & M2 & M3 --> HITS[context chunks]
    PREV --> HITS
    HITS --> S3[STEP 3: Generator.generate]
    AMB --> S3
    S3 --> GROUTE{>12 chunks &<br/>summary/extraction/analysis?}
    GROUTE -->|yes| MAPRED[Map-Reduce]
    GROUTE -->|no| SINGLE[Single LLM call]
    MAPRED --> ANS[Answer + memory_summary]
    SINGLE --> ANS
    ANS --> S4[STEP 4: Store turn in group]
    S4 --> RESP[Return answer + sources + group_id]
```

---

## 4. Map-Reduce generation (broad intents)

Used for `summary` / `extraction` / `analysis` when more than 12 chunks are retrieved,
so the whole document is covered without one oversized LLM call.

```mermaid
flowchart TD
    C[N retrieved chunks] --> B[Split into batches of 10]
    B --> P[MAP: one LLM call per batch, in parallel<br/>max 30 workers]
    P --> EVAL[Each batch evaluates chunks INDIVIDUALLY<br/>extract relevant / skip irrelevant / NONE]
    EVAL --> FILT[Drop NONE batches → partials]
    FILT --> EMPTY{any partials?}
    EMPTY -->|no| NF[Grounded 'not contained' answer]
    EMPTY -->|yes| RED[REDUCE: 1 LLM call<br/>merge + de-duplicate]
    RED --> OUT["Final answer + MEMORY_SUMMARY<br/>(streamed if /query/stream)"]
```

**LLM calls per generation:** `N batches + 1 reduce` (plus 1 classification call upstream).

---

## 5. Streaming flow (POST /query/stream — Server-Sent Events)

```mermaid
flowchart TD
    REQ[Frontend fetch /query/stream] --> CTX[Resolve context + retrieve<br/>in worker thread]
    CTX --> META[SSE event: meta<br/>sources + group_id + intent]
    META --> GEN[generate_stream]
    GEN --> MR{map-reduce intent?}
    MR -->|yes| MAP[Run MAP phase first<br/>not streamable]
    MAP --> STREAMR[Stream the REDUCE call]
    MR -->|no| STREAMS[Stream the single call]
    STREAMR & STREAMS --> TOK[SSE events: token deltas<br/>ANSWER section only]
    TOK --> UI[Live render in bubble + caret]
    TOK --> DONE[SSE event: done<br/>full answer + memory_summary]
    DONE --> STORE[Persist turn to memory]
    DONE --> FINAL[Re-render full markdown + MathJax + sources]
```

Only the `ANSWER:` section is streamed; the `MEMORY_SUMMARY:` is buffered (never leaked
to the client) and used for conversation memory.

---

## 6. Conversation memory lifecycle

```mermaid
flowchart TD
    T[Each turn stored] --> F[ConversationTurn:<br/>query + memory_summary + full_answer]
    F --> G[Added to group all_turns + recent_turns]
    G --> CHK{≥5 unsummarized turns?}
    CHK -->|no| WAIT[wait]
    CHK -->|yes| BGT[Background thread]
    BGT --> SRC[Summarize from FULL answers<br/>brief but lossless]
    SRC --> EMB[Embed summary]
    EMB --> SAVE[group.summary + summary_ready=true<br/>clear recent_turns]
    SAVE --> USE{Future dependent query}
    USE -->|summary ready| CS[Use group summary - compact]
    USE -->|not ready| RT[Use recent_turns - memory_summary]
```

**Two summary tiers:**
- `memory_summary` (per turn, 1–2 lines) — compact; used to show many recent turns cheaply.
- group `summary` (rolling, from full answers) — richer; used as conversational context once ready.

---

## 7. Intent handling matrix

| Intent | Retrieval mode | top_k | threshold | Generation |
|---|---|---|---|---|
| **factual** | top_k | 5 | 0.20 | single call |
| **summary** | comprehensive | 15 | 0.10 | map-reduce (>12 chunks) |
| **analysis** | comprehensive | 15 | 0.10 | map-reduce (>12 chunks), adaptive prompt |
| **comparison** | comprehensive | 15 | 0.12 | single call, comparison table |
| **extraction** | exhaustive | all | 0.0 | map-reduce (>12 chunks), label-fidelity |
| **ambiguous** | (skip) | 3 | 0.20 | clarification request |

**Retrieval modes:**
- `top_k` — best N chunks by similarity above threshold (precise answers).
- `comprehensive` — all chunks above threshold, capped at top_k (synthesis).
- `exhaustive` — entire document, no threshold (enumeration / "list all X").

---

## 8. LLM-driven routing (no hardcoded keywords)

Every routing decision is made by the LLM classifier, not by keyword matching, so it
generalizes across documents and phrasings:

| Field | Decides | Values |
|---|---|---|
| `dependency_type` | conversation relationship | independent / dependent / multi_group / ambiguous |
| `retrieval_intent` | retrieval mode + prompt + map-reduce routing | factual / summary / comparison / extraction / analysis / ambiguous |
| `answer_source` | operate on previous answer vs. retrieve fresh | document / previous_answer |
| `retrieval_scope` | whole document vs. specific topic | whole_document / specific |

**Retrieval breadth precedence** (highest first):
1. Previous-answer follow-up → skip document retrieval, use prior answer as sole context.
2. Manual Top-K toggle ON → user's K wins.
3. Whole-document scope (auto) → retrieve all chunks, no cap.
4. Broad intent default → all relevant chunks above threshold (capped at 15).
5. Otherwise → per-intent default.

---

## Key files

| Area | File |
|---|---|
| FastAPI app + streaming | `PDFRAG-main/backend/app.py` |
| Pipeline singletons + legacy server | `PDFRAG-main/backend/main.py` |
| Classification | `PDFRAG-main/backend/memory/classifiers/llm_classifier.py` |
| Context resolution | `PDFRAG-main/backend/memory/resolution/context_resolver.py` |
| Retrieval modes + intent config | `PDFRAG-main/backend/retrieval/retriever.py` |
| Generation + map-reduce + streaming | `PDFRAG-main/backend/generation/generator.py` |
| All LLM prompts (centralized) | `PDFRAG-main/backend/prompts/` |
| Background group summaries | `PDFRAG-main/backend/summarization/background_summarizer.py` |
| Conversation memory | `PDFRAG-main/backend/memory/management/conversation_manager.py` |
| Frontend | `PDFRAG-main/frontend/{index.html,app.js,styles.css}` |
| Launcher | `run.py` |
