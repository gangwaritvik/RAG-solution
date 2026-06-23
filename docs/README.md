# ⬡ PDFRAG — Enterprise Retrieval-Augmented Generation System

A production-ready RAG system combining document ingestion, semantic search, and LLM-powered answer generation with conversation memory and multi-turn query support.

## 🚀 Quick Start

```bash
python run.py
```

This starts **both servers automatically**:
- 🌐 **Frontend**: http://localhost:3000 (Web UI)
- 🔧 **Backend**: http://localhost:8000 (API)

See [STARTUP.md](STARTUP.md) for detailed setup guide.

## ✨ Key Features

### Document Processing
- ✅ PDF & DOCX support
- ✅ 4 chunking strategies (Recursive, Semantic, Sliding Window, Fixed)
- ✅ Automatic vector embedding (Azure OpenAI text-embedding-3-small)
- ✅ ChromaDB persistent storage with HNSW indexing

### Query Processing
- ✅ Intent-based classification (Factual, Summary, Extraction, Analysis, Comparison)
- ✅ Parallel multi-group query processing (2.5x faster)
- ✅ Semantic search with top-K retrieval
- ✅ LLM-powered answer generation (GPT-4.1)

### Conversation Management
- ✅ Multi-turn conversation support
- ✅ Automatic context memory
- ✅ Query dependency tracking
- ✅ Session-based group management

### User Interface
- ✅ Document upload & processing status
- ✅ Per-file chunk visualization
- ✅ Interactive query builder
- ✅ Source citations & similarity scores
- ✅ Response time tracking

## 📋 System Architecture

```
Backend (Python HTTP Server - Port 8000)
├── Document Ingestion Pipeline
│   ├── PDF/DOCX Loading
│   ├── Chunking (4 modes)
│   └── Vector Embedding
├── Vector Store (ChromaDB)
├── Memory Management (Conversation Groups)
├── Query Processing
│   ├── Intent Classification (LLM)
│   ├── Context Retrieval
│   └── Answer Generation (LLM)
└── Parallel Processing (ThreadPoolExecutor)

Frontend (JavaScript Web UI - Port 3000)
├── Document Management
├── File Upload & Status
├── Chunk Visualization
├── Query Interface
└── Results Display
```

## 🔌 API Endpoints

### Document Management
```bash
# Upload & process documents
POST /ingest
Content-Type: multipart/form-data
Body: file (PDF/DOCX)
Response: {status: "ok", message: "Processing...", file_id: "..."}

# Get status & chunks
GET /status
Response: {
  status: "ok",
  total_vectors: 676,
  files: [{filename, chunk_count, chunks}]
}

# Delete all documents
POST /clear
Response: {status: "ok", message: "All vectors deleted"}
```

### Query & Retrieval
```bash
# Submit query
POST /query
Content-Type: application/json
Body: {
  query: "What is RAG?",
  top_k: 5,
  temperature: 0.2,
  group_id: "optional-session-id"
}
Response: {
  answer: "...",
  retrieved_chunks: [...],
  retrieval_intent: "factual",
  memory_summary: "...",
  is_multi_group: false
}
```

## 📁 Project Structure

```
PDFRAG-main/
├── backend/
│   ├── main.py                     # HTTP server & request handler
│   ├── config.py                   # Configuration & secrets
│   ├── generation/                 # Answer generation
│   │   └── generator.py
│   ├── ingestion/                  # Document processing
│   │   ├── document_loader.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── vector_store.py
│   │   └── loaders/
│   ├── memory/                     # Conversation management
│   │   ├── classifiers/
│   │   ├── management/
│   │   ├── resolution/
│   │   └── storage/
│   ├── processing/                 # Query processing
│   │   └── multi_group_processor.py
│   ├── retrieval/                  # Semantic search
│   │   └── retriever.py
│   ├── summarization/              # Background summarization
│   └── utils/                      # Logging, timeouts
│
├── frontend/
│   ├── index.html                  # Web UI
│   ├── app.js                      # JavaScript logic
│   └── styles.css                  # Styling
│
├── run.py                          # Unified entry point
├── STARTUP.md                      # Detailed setup guide
└── README.md                       # This file
```

## ⚙️ Configuration

Edit `backend/config.py`:
```python
# Azure OpenAI
AZURE_OPENAI_KEY = "your-api-key"
AZURE_OPENAI_ENDPOINT = "https://region.openai.azure.com/"

# Models
CHAT_MODEL = "gpt-4.1"              # Answer generation
EMBEDDING_MODEL = "text-embedding-3-small"  # Vector embeddings

# Document Processing
CHUNK_SIZE = 300                    # Characters per chunk
CHUNK_OVERLAP = 50                  # Overlap for context
```

## 🔍 Example Workflow

1. **Upload Document**
   ```
   POST /ingest (sample_rag.pdf)
   → Document split into 4 chunks
   → Embedded to vectors (1536 dimensions)
   → Stored in ChromaDB
   ```

2. **Query System**
   ```
   POST /query
   Query: "What are the benefits of RAG?"
   → Intent classified as SUMMARY
   → Top-5 most relevant chunks retrieved
   → LLM generates comprehensive answer
   ```

3. **View Results**
   ```
   Response includes:
   - Answer text
   - Source chunks with citations
   - Similarity scores
   - Response time
   ```

## 📊 Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Single query | ~5-7s | Includes LLM latency |
| Multi-group (2 sub-queries) | ~6.5s | Parallel execution |
| Multi-group (3 sub-queries) | ~8.5s | 2.2x faster than sequential |
| File processing (PDF) | ~8-10s | Includes embedding |

## 🛠️ Technology Stack

- **Backend**: Python 3.10+, BaseHTTPServer
- **Vector Store**: ChromaDB with HNSW indexing
- **LLM**: Azure OpenAI (GPT-4.1)
- **Embeddings**: Azure OpenAI (text-embedding-3-small, 1536-dim)
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Concurrency**: ThreadPoolExecutor (Python threads)

## 📝 Testing

Run a test query:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "top_k": 5, "temperature": 0.2}'
```

## 🐛 Troubleshooting

**Backend won't start:**
- Check Azure API key in `backend/config.py`
- Verify port 8000 is available
- Check Python version (3.10+)

**Frontend won't load:**
- Verify port 3000 is available
- Check browser console for errors
- Hard refresh: Ctrl+Shift+R

**Slow queries:**
- Reduce top_k value (default 5)
- Check network latency to Azure
- Review ChromaDB logs

See [STARTUP.md](STARTUP.md) for more solutions.

## 📄 License

MIT License