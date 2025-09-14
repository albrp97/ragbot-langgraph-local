# Implementation Summary RAGBot - LangGraph

- **A chatbot UI**
  - Implemented a **Gradio** interface with a chat panel and status area.
  - File management panel to upload/list PDFs and control which files are used.
  - Path: `app/ui/gradio_app.py`.

- **Ingest several long PDFs as a RAG knowledge base using a vector DB**
  - Vector store: **Chroma** (persistent at `./data/chroma`).
  - Text embeddings: **Qwen3 Embedding** (configurable via `.env`).
  - Paths: `app/rag/ingest.py`, `app/rag/retriever.py`.

- **Index images from PDFs**
  - Per-page **image extraction** with PyMuPDF (skips tiny assets by byte threshold).
  - **Image captioning** (generic *image-to-text* pipeline) → **rich description** (expanded by the LLM) for better retrieval.
  - Each image becomes a **document** in Chroma with metadata
  - Sidecar **JSON** metadata saved per image
  - Paths: `app/rag/ingest.py` (captioning + JSON sidecars in `data/processed/images`).

- **Structured extraction from a document into JSON**
  - A dedicated **“Structured extraction (CVs)”** flow:
    - CV **detector** (skip non-CVs)
    - Writes one **JSON per PDF**
  - Paths: `app/extract/cv_detect.py`, `app/extract/structured.py`, wiring in `app/graph/rag_graph.py`.

- **Dynamic memory that summarizes when the conversation exceeds X tokens**
  - Conversation **history compression** triggered by token limits from `.env`.
  - Summary injected into the prompt as **Conversation memory**.
  - Path: `app/llm/memory.py` and node `prepare_memory` in the graph.

- **LangChain/LangGraph-based implementation**
  - **LangGraph** state machine with nodes:
    - prepare_memory → route → ((retrieve | run_python) → generate | structured_extract) → END
  - Path: `app/graph/rag_graph.py`.

- **Execute Python when a question needs it**
  - A **Python tool** node (`run_python`) for computation (statistics, random, lists, arithmetic).
  - Router supports:
    - **Explicit command**: `python …`
    - **Heuristics** for math-like queries (e.g., “random number between 0 and 100”, dice simulations, list ops).
  - Path: `app/tools/plan_and_run.py` + routing in `app/graph/rag_graph.py`.

---

## Extra features

- **File selection UX**
  - Checkbox with **`ALL`** and per-PDF selection; chat input is only enabled when at least one doc is selected.

- **Streaming ingestion with live progress**
  - **Two levels of progress**:
    1) Per-PDF progress bar and “currently indexing” label,
    2) **Per-image progress** within each PDF (**phase: images**) showing the **caption/description snippet** being created.
  - Console version prints the **image’s description** as it’s generated.
  - Paths: `app/rag/ingest.py` (stream events), `scripts/ingest_pdfs.py` (console), `app/ui/gradio_app.py` (UI bars).

- **Simple router commands**
  - `python …` to force computation via tool.
  - `extract …` to trigger the CV extractor directly.
  - Otherwise, **heuristics** decide RAG vs Python.

- **“Extract to JSON” button in the UI**
  - Streams progress over the selected PDFs.
  - Posts a human-readable **summary** into the chat once finished.

- **Upload PDFs from the UI**
  - **Drag & drop** PDFs, auto-reindex, progress shown, and selection reset to `ALL` after completion.

- **Manifest-aware ingestion**
  - Tracks per-file hash and embedding/caption model id; **auto-rebuilds** when models change.

---

## Architecture

- **Ingestion**
  1) Load PDF → split text → embed → **Chroma**  
  2) Extract images → caption (short) → **expand to rich description** → save **JSON sidecar** → index **description** in Chroma
- **Serving**
  - **LangGraph**: router → ((retrieve | run_python) → generate | structured_extract) → END
  - **Retriever** supports filtering by selected PDFs
- **UI**
  - Gradio chat, **file manager**, **upload**, **progress bars**, **CV extraction panel**.

---

## Known limitations

1) Local-first design  
This prototype is wired for **local inference**. To switch to a hosted API you’ll need minor changes.

2) Small non-reasoning model (0.6B)  
We currently run a **0.6B non-reasoning** model, which limits retrieval quality, routing accuracy, and long-context synthesis.  
- **Reasoning models:** if you choose a reasoning model that emits `<think>`/reason traces, **do not surface them to users**. Strip tags that only removes known metadata blocks

3) OCR for scanned PDFs  
Right now, image-only PDFs can be misclassified as “no text.” Add an **OCR fallback** so scans are treated as text.

4) Routing quality (heuristics)  
The current `route` node uses **keyword/heuristic rules**. This is brittle for ambiguous queries.  
- **LLM router:** replace with a small classifier or an LLM-based router that returns one of `{rag, python, extract}` as a **structured JSON** field.
