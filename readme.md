# RAGBot — LangGraph

This repository contains my solution for a 3-part challenge:

* **Part 1:** A local RAG chatbot with PDF + image indexing, dynamic memory, and a Python tool.
* **Part 2:** Brief theoretical answers about LLMs, RAG, agents, and evaluation.
* **Part 3:** A local object-detection microservice (cars & people) with a Dockerized FastAPI + YOLO11.

For detailed write-ups, see:
`docs/part1.md`, `docs/part2.md`, `docs/part3.md`.

---

## Repo layout (high level)

```
.
├─ app/                     # Application code (RAG)
│  ├─ config/               # Pydantic settings (.env)
│  ├─ graph/                # LangGraph state machine & routing
│  ├─ llm/                  # LLM wrappers (chat, memory, embeddings)
│  ├─ rag/                  # Ingestion + retriever (text + image captions)
│  └─ ui/                   # Gradio app (chat UI, upload, progress bars)
├─ data/
│  ├─ raw_pdfs/             # Drop PDFs here (or upload from UI)
│  ├─ processed/images/     # Extracted PDF images + JSON sidecars (captions)
│  ├─ structured/cv/        # CV extraction output (JSON)
│  ├─ chroma/               # Persistent Chroma vector store
│  └─ car_people_samples/   # Sample images for the detector (Part 3)
├─ detector/                # Part 3: FastAPI + YOLO11 service (+ Dockerfile.gpu)
├─ docs/                    # Part 1–3 explanations and instructions
├─ scripts/                 # CLI helpers (e.g., scripts/ingest_pdfs.py)
├─ run.py                   # Launch the Gradio RAG app
├─ .env                     # Configuration
└─ yolo11n.pt               # Default detector weights (optional, auto-downloaded)
```

---

### Part 1 — RAGBot features

* **Gradio chat UI** with a file manager (upload/list/select PDFs).
* **PDF ingestion** into **Chroma** using **Qwen3 embeddings**.
* **Image indexing:** extracts images from PDFs, captions them (image-to-text), saves **JSON metadata**, and indexes **rich descriptions** for retrieval.
* **Structured extraction (CVs):** button in the UI; writes one JSON per CV.
* **Dynamic memory:** conversation is summarized automatically beyond token limits.
* **Python tool:** router sends math/computation queries to a sandboxed Python executor.
* Code highlights: `app/ui/gradio_app.py`, `app/rag/ingest.py`, `app/graph/rag_graph.py`.

See **`docs/part1.md`** for details.

### Part 2 — Theory

* Concise answers covering completion vs chat, reasoning vs generalists, enforcing formats, RAG vs fine-tuning, what an agent is, and how to evaluate Q\&A/RAG apps.

See **`docs/part2.md`**.

### Part 3 — Local Object Detection

* **FastAPI** service using **Ultralytics YOLO11** (pretrained) to detect **persons** and **cars**.
* Endpoints: `/health`, `/detect` (multipart `file`), returns JSON detections.
* **Dockerized** (GPU): `detector/Dockerfile.gpu`.
* Sample images under `data/car_people_samples/`.

See **`docs/part3.md`** for run and Docker instructions.

---

## Notes

* Everything runs **locally** by default; switch models via `.env`.
* Image captions are saved to `data/processed/images/*.json` and indexed for multimodal retrieval.
* More guidance, trade-offs, and limitations are documented in the **docs** folder.
