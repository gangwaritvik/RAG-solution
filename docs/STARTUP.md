# Quick Start Guide — PDFRAG

## Prerequisites
- Python 3.10+
- Azure OpenAI API key (set in `backend/config.py`)

## Running the System

### 🚀 Option 1: Normal Run (Keep Previous Data)
```bash
python run.py
```

Starts both servers with all previously saved data:
- **Backend**: http://localhost:8000
- **Frontend**: http://localhost:3000

### 🆕 Option 2: Fresh Start (Clear Everything)
```bash
python run.py --fresh
```

Clears the vector store and all conversation memory before starting. Use this for:
- ✅ Testing with clean slate
- ✅ Resetting conversations
- ✅ Removing old document uploads
- ✅ Fresh testing session

**What gets deleted with `--fresh`:**
```
Vector store (chroma_db):
  ❌ 680 existing vectors → cleared
  ❌ 4 conversation groups → cleared

Uploads:
  ❌ All uploaded files → cleared
```

### 🔧 Option 3: Backend Only
```bash
python run.py --backend
```

Start only the backend (port 8000). Useful for:
- Testing API directly with curl/Postman
- Running with external frontend
- Backend development

### 🌐 Option 4: Frontend Only
```bash
python run.py --frontend
```

Start only the frontend (port 3000). Useful for:
- UI testing with separate backend
- Frontend development
- Using production backend

### 📟 Option 5: Manual Terminal Control

**Terminal 1 - Backend:**
```bash
cd PDFRAG-main
python backend/main.py
```

**Terminal 2 - Frontend:**
```bash
cd PDFRAG-main/frontend
python -m http.server 3000
```

## 📊 Understanding Startup Messages

### What You'll See:
```
VectorStore ✅ Collection loaded — 680 existing vectors
[MEMORY_STORE] ✅ Collections loaded — metadata: 4, summaries: 0
[MEMORY_MGR] ✅ Retrieved 4 groups
```

### What It Means:

**680 existing vectors** = Embeddings from all previously uploaded documents
- Each document chunk gets converted to a vector (1536 dimensions)
- From previous uploads: test_valid.pdf, sample_rag.pdf, etc.
- Stored in: `backend/storage/chroma_db/`

**metadata: 4** = 4 conversation session groups
- Each group stores a conversation thread
- Contains: queries, answers, memory summaries, context

**retrieved 4 groups** = System loaded all 4 previous conversations
- Your chat history is preserved!
- You can continue conversations from before

### To Start Fresh:
```bash
python run.py --fresh
```

This will:
1. Delete `backend/storage/chroma_db/` (all vectors)
2. Clear `backend/storage/uploads/` (all files)
3. Start with clean state (0 vectors, 0 groups)

---

## System Architecture

```
PDFRAG-main/
├── run.py                          ← START HERE
│
├── PDFRAG-main/
│   ├── backend/
│   │   ├── main.py                 ← HTTP server (port 8000)
│   │   ├── config.py               ← Configuration
│   │   ├── generation/             ← LLM generation
│   │   ├── ingestion/              ← Document processing
│   │   ├── memory/                 ← Conversation memory
│   │   ├── processing/             ← Query processing
│   │   ├── retrieval/              ← Semantic search
│   │   ├── storage/
│   │   │   ├── chroma_db/          ← Vector store (persisted)
│   │   │   └── uploads/            ← Uploaded files
│   │   └── utils/                  ← Logging, timeouts
│   │
│   └── frontend/
│       ├── index.html
│       ├── app.js
│       └── styles.css
│
├── pyproject.toml
├── requirements.txt
├── README.md
└── STARTUP.md
```

---

## Features

### ✅ Document Processing
- PDF & DOCX support
- 4 chunking modes (Recursive, Semantic, Sliding Window, Fixed)
- Azure OpenAI embeddings (1536 dimensions)
- ChromaDB HNSW indexing

### ✅ Query Processing
- Intent-based classification
- Parallel multi-group queries
- Semantic search with top-K
- LLM answer generation (GPT-4.1)

### ✅ Conversation Management
- Multi-turn support
- Automatic context memory
- Query dependency tracking
- Persistent session groups

### ✅ User Interface
- Document upload & status
- Per-file chunk visualization
- Interactive query interface
- Source citations
- Response timing

---

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/ingest` | POST | Upload & process documents |
| `/query` | POST | Submit query & get answer |
| `/status` | GET | System status & file chunks |
| `/clear` | POST | Delete all documents |

---

## Configuration

Edit `backend/config.py`:

```python
# Azure OpenAI Credentials
AZURE_OPENAI_KEY = "your-api-key-here"
AZURE_OPENAI_ENDPOINT = "https://your-region.openai.azure.com/"

# Models
CHAT_MODEL = "gpt-4.1"
EMBEDDING_MODEL = "text-embedding-3-small"

# Processing
CHUNK_SIZE = 300              # Characters per chunk
CHUNK_OVERLAP = 50            # Overlap for context
TOP_K = 5                     # Default chunks to retrieve
```

---

## Troubleshooting

### Port Already in Use

**Windows PowerShell:**
```powershell
# Kill process on port 8000
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process

# Kill process on port 3000
Get-Process -Id (Get-NetTCPConnection -LocalPort 3000).OwningProcess | Stop-Process
```

**Linux/Mac:**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Kill process on port 3000
lsof -ti:3000 | xargs kill -9
```

### Want Fresh Data

```bash
python run.py --fresh
```

### Slow Queries

- Reduce Top-K slider (default 5)
- Check Azure API rate limits
- Verify internet connection

### No Chunks Displaying

- Hard refresh browser (Ctrl+Shift+R)
- Check backend logs for errors
- Verify file processing completed

### API Connection Issues

- Verify backend is running: `http://localhost:8000/status`
- Check backend logs for errors
- Verify Azure credentials in `config.py`

---

## Performance Reference

| Operation | Time |
|-----------|------|
| Single query | ~5-7s |
| 2-group comparison | ~6.5s |
| 3-group comparison | ~8.5s |
| PDF upload & embed | ~8-10s |

---

## Next Steps

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Azure API key:**
   - Edit `PDFRAG-main/backend/config.py`
   - Add your API key and endpoint

3. **Start the system:**
   ```bash
   python run.py
   ```

4. **Open in browser:**
   - http://localhost:3000

5. **Upload a PDF and start chatting!**

---

See [README.md](README.md) for full documentation.
