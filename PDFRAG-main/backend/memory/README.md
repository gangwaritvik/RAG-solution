# Group-Based Conversational Memory Architecture

## Overview

This module implements the core infrastructure for **Intent-Driven Group-Based Conversational Memory** as described in the architecture specification. It provides:

- **Group Memory**: Organizes conversations into semantic topics instead of linear chat history
- **Persistent Storage**: Uses Chroma DB to store and retrieve conversation groups
- **Scalable Context**: Maintains constant prompt size regardless of conversation length
- **Semantic Search**: Find relevant groups by embedding similarity

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│         ConversationMemoryManager                       │
│  (High-level API for memory operations)                │
└──────────────┬──────────────────────────────────────────┘
               │
      ┌────────┴────────┐
      │                 │
┌─────▼────────┐  ┌────▼──────────────┐
│ GroupMemory  │  │ MemoryStore       │
│ (in-memory)  │  │ (Chroma DB backed)│
└──────────────┘  └───────────────────┘
      │                 │
      └────────┬────────┘
               │
        ┌──────▼─────────┐
        │  Chroma DB     │
        │ (Persistent)   │
        └────────────────┘
```

## Core Components

### 1. **Group Memory** (`group_memory.py`)

Data structures for managing conversation state:

- **`ConversationTurn`**: Single query-response pair
  ```python
  turn = ConversationTurn(
      turn_id="turn_abc123",
      query="What is the EMD amount?",
      memory_summary="EMD = 2% of contract value"
  )
  ```

- **`Group`**: Semantic topic with conversation history
  ```python
  group = Group(
      group_id="group_xyz",
      topic="EMD Requirements",
      summary="...",
      recent_turns=[...],
      all_turns=[...],
      summary_ready=True
  )
  ```

- **`GroupMemory`**: In-memory manager for fast access
  ```python
  gm = GroupMemory()
  g = gm.create_group("g1", "Eligibility")
  gm.add_turn_to_group("g1", turn)
  ```

### 2. **Memory Store** (`memory_store.py`)

Chroma DB backed persistent storage:

```python
store = MemoryStore(chroma_db_path="./storage/chroma_db")

# Save/update group
store.save_group(group)

# Retrieve group
retrieved_group = store.get_group("group_xyz")

# Search by embedding
results = store.search_groups_by_embedding(query_embedding, top_k=5)

# List all groups
all_groups = store.list_groups()
```

**Collections Used:**
- `conversation_memory`: Stores full group metadata (JSON)
- `group_summaries`: Stores summary embeddings for semantic search

### 3. **Conversation Manager** (`conversation_manager.py`)

High-level API coordinating in-memory and persistent storage:

```python
manager = ConversationMemoryManager(chroma_db_path="./storage/chroma_db")

# Create group
group = manager.create_conversation_group("Eligibility Requirements")

# Add turn
turn = manager.add_conversation_turn(
    group_id=group.group_id,
    query="What is the minimum turnover?",
    memory_summary="Minimum ₹50 Cr annual turnover"
)

# Update summary
manager.update_group_summary(
    group_id=group.group_id,
    summary="...",
    summary_embedding=embedding_vector
)

# Search groups
results = manager.search_groups_by_embedding(
    query_embedding=vector,
    similarity_threshold=0.5,
    top_k=5
)

# Get group context for LLM
context = manager.get_group_context(group_id)
```

## Key Features

### 1. **Topic-Based Organization**

Instead of linear chat:
```
Q1 → A1 → Q2 → A2 → Q3 → A3
```

Organized by semantic topics:
```
Group 1 (Eligibility)     → Q1 → A1, Q4 → A4, Q7 → A7
Group 2 (EMD)             → Q2 → A2, Q5 → A5
Group 3 (Timeline)        → Q3 → A3, Q6 → A6
```

### 2. **Bounded Prompt Size**

Memory summaries stay small:
```python
# Instead of storing full answers:
# "Answer: Minimum turnover is ₹50 Cr. This applies to..."

# Store compressed summary:
# "Minimum turnover ₹50 Cr"
```

### 3. **Automatic Summarization**

Trigger conditions:
- `unsummarized_turns >= 5` (configurable threshold)
- Context switch detected
- Conversation ends

```python
if manager.should_summarize_group(group_id):
    # Trigger LLM summarization
    summary = llm_summarize(group.recent_turns)
    manager.update_group_summary(group_id, summary, embedding)
```

### 4. **Semantic Group Search**

Find relevant groups by embedding:
```python
results = manager.search_groups_by_embedding(
    query_embedding=[...],
    similarity_threshold=0.5,
    top_k=5
)
# Results: [Group1(similarity=0.87), Group2(similarity=0.72), ...]
```

### 5. **Race Condition Safe**

No blocking on slow summarization:
```python
# If summary is being generated:
group.summary_ready = False

# New query uses recent_turns instead:
context = group.recent_turns  # Non-blocking!

# When summary completes:
group.summary_ready = True
group.recent_turns.clear()
# Future queries use summary
```

## Usage Example

```python
from backend.memory import ConversationMemoryManager

# Initialize
manager = ConversationMemoryManager(chroma_db_path="./storage/chroma_db")

# Create groups for topics
eligibility_group = manager.create_conversation_group("Eligibility Requirements")
emd_group = manager.create_conversation_group("EMD Requirements")

# Add turns to appropriate groups
manager.add_conversation_turn(
    eligibility_group.group_id,
    query="What is the minimum turnover?",
    memory_summary="Minimum ₹50 Cr annual turnover"
)

manager.add_conversation_turn(
    emd_group.group_id,
    query="What is the EMD amount?",
    memory_summary="EMD = 2% of contract value"
)

# Update summaries when needed
if manager.should_summarize_group(eligibility_group.group_id):
    summary = "Eligibility: ₹50 Cr turnover, startups exempted, ..."
    embedding = embedder.embed(summary)  # Get from Azure OpenAI
    manager.update_group_summary(
        eligibility_group.group_id,
        summary,
        embedding
    )

# Get context for answer generation
context = manager.get_group_context(eligibility_group.group_id)
# Use context.summary + context.recent_turns in RAG pipeline
```

## Integration with Existing RAG Pipeline

The memory system integrates seamlessly:

```python
# In main.py
from backend.memory import ConversationMemoryManager

# Initialize alongside existing components
vector_store = VectorStore(persist_dir=CHROMA_DIR)
retriever = Retriever(embedder, vector_store)
generator = Generator()
memory_manager = ConversationMemoryManager(chroma_db_path=CHROMA_DIR)

# In query handling:
# 1. Determine which group this query belongs to (using embeddings)
# 2. Get group context from memory_manager
# 3. Retrieve relevant documents
# 4. Generate answer with group context + document chunks
# 5. Store query and memory summary in group
```

## Chroma DB Collections

The memory system creates 2 new Chroma DB collections (in addition to existing `pdf_rag`):

### `conversation_memory`
Stores full group metadata as JSON:
```json
{
  "group_id": "group_xyz",
  "topic": "EMD Requirements",
  "summary": "...",
  "recent_turns": [...],
  "all_turns": [...],
  "created_at": "2024-12-01T10:30:00",
  "updated_at": "2024-12-01T10:45:00",
  "summary_ready": true
}
```

### `group_summaries`
Stores summary embeddings for semantic search:
```
ID: group_xyz
Embedding: [0.1, 0.2, ..., 0.5]  (1536-dim Azure OpenAI)
Document: "EMD = 2% of contract value..."
Metadata: {"group_id": "group_xyz", "topic": "EMD Requirements"}
```

## Testing

Run the demo to see the system in action:

```bash
cd backend
python -m memory.demo
```

Output:
```
======================================================================
DEMO: Group-Based Conversational Memory Architecture
======================================================================

1️⃣  Creating conversation groups...
   ✅ Created group: Eligibility Requirements (ID: group_abc)
   ✅ Created group: EMD (Earnest Money Deposit) (ID: group_def)
   ✅ Created group: Timeline & Deadlines (ID: group_ghi)

2️⃣  Adding conversation turns...
   ✅ Added 3 turns to Eligibility group
   ✅ Added 2 turns to EMD group
   ✅ Added 2 turns to Timeline group

[... more demo output ...]
```

## Configuration

Key thresholds (configurable):

```python
# In conversation_manager.py
SUMMARIZATION_THRESHOLD = 5  # Summarize after 5 unsummarized turns
SEARCH_THRESHOLD = 0.5       # Group similarity threshold
TOP_K = 5                    # Max groups to return in search
```

## Next Steps

1. **Context Resolution Layer**: Classify query dependencies (INDEPENDENT, DEPENDENT, MULTI_GROUP, AMBIGUOUS)
2. **Standalone Query Generation**: Convert ambiguous follow-ups into retrieval-ready queries
3. **Intent Classification**: Determine retrieval strategy (FACTUAL, SUMMARY, COMPARISON, EXTRACTION, ANALYSIS)
4. **API Integration**: Create endpoints for conversation management
5. **Async Summarization**: Background task to summarize groups without blocking

## Performance Characteristics

- **Group Creation**: O(1)
- **Add Turn**: O(1) (in-memory) + O(1) (Chroma insert)
- **Get Group**: O(1)
- **Search Groups**: O(log n) (Chroma HNSW search)
- **List Groups**: O(n)
- **Memory per Group**: ~1-5 KB (compressed summaries)
- **Max Conversation Length**: Unlimited (bounded by summaries)

## Architecture Advantages

✅ Topic-based memory scales better than linear chat history
✅ Constant prompt size regardless of conversation length
✅ Supports natural context switches between topics
✅ Allows revisiting old topics instantly
✅ Better retrieval through standalone query generation
✅ Low token usage via compressed summaries
✅ Fully compatible with existing RAG pipeline
✅ Maintains auditability through full turn storage
✅ Non-blocking summarization with race condition safety
