# NAAC Accreditation RAG Agent

An end-to-end RAG pipeline that answers questions about the **NAAC Revised Manual for Autonomous Colleges (2020)** using **IBM Granite** via **watsonx.ai**, with a web UI served by **FastAPI** and a **FAISS** vector index for grounded retrieval.

---

## Project Structure

```
.
├── .env                          # Credentials (never committed)
├── pyproject.toml                # Package + uv script entry point
├── Dockerfile                    # Single-container deployment
├── docker-compose.yml            # Convenience wrapper
├── Revised-Autonomous-Manual-as-on-24-2-2020_2.pdf  # Source document
├── agents/
│   └── naac_rag_agent.yaml       # Orchestrate agent spec
├── tools/
│   └── naac_rag_tool.py          # Orchestrate tool definition
└── naac_rag/
    ├── __init__.py
    ├── parser.py                 # PDF → MetricChunk (by KI/Metric)
    ├── vector_store.py           # FAISS index build + retrieval
    ├── rag.py                    # Granite inference pipeline
    ├── app.py                    # FastAPI backend + single-page UI
    ├── cli.py                    # CLI entry point
    └── requirements.txt
```

---

## Prerequisites

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) (or `pip`)
- Docker (for containerized deployment)
- IBM Cloud API key and watsonx.ai project (already in `.env`)

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r naac_rag/requirements.txt
# OR with uv:
uv sync
```

### 2. Build the FAISS index (one-time)

```bash
python naac_rag/vector_store.py
```

### 3. Run the 5 automated test queries

```bash
python naac_rag/cli.py --test
# OR with uv:
uv run orchestrate --test
```

### 4. Start the web server

```bash
python naac_rag/cli.py --serve
# OR:
uv run orchestrate --serve
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### 5. Interactive CLI

```bash
uv run orchestrate
```

---

## Docker Deployment (single container)

```bash
# Build and run
docker build -t naac-rag-agent .
docker run -p 8000:8000 --env-file .env naac-rag-agent

# OR with docker-compose
docker-compose up --build
```

Open [http://localhost:8000](http://localhost:8000).

---

## watsonx Orchestrate Integration

### 1. Activate your Orchestrate environment

```bash
orchestrate env activate <your-env-name>
```

### 2. Import the tool

```bash
orchestrate tools import -k python -p tools/naac_rag_tool.py \
  --package-root naac_rag
```

### 3. Import the agent

```bash
orchestrate agents import -f agents/naac_rag_agent.yaml
```

---

## How It Works

1. **Parsing** (`parser.py`): `PyMuPDF` extracts text from the NAAC manual PDF. The parser walks line by line detecting `Criterion`, `Key Indicator`, and `Metric` headers using regex, building a `MetricChunk` per metric with full metadata.

2. **Indexing** (`vector_store.py`): Each chunk's `metadata_header + text` is encoded with `all-MiniLM-L6-v2` (sentence-transformers) and stored in a FAISS `IndexFlatIP` (cosine similarity). The index is persisted to `naac_index.faiss`.

3. **Retrieval**: Top-3 most similar chunks are retrieved for every user query.

4. **Generation** (`rag.py`): Retrieved chunks are injected into a strict prompt sent to `ibm/granite-3-8b-instruct` via the watsonx.ai chat API. The model is instructed to cite Criterion/KI/Metric in every answer and reply "Not found in the manual" if no chunk is relevant.

5. **Frontend** (`app.py`): FastAPI serves a self-contained HTML page. `POST /ask` returns `{answer, sources}` as JSON.

---

## Environment Variables (`.env`)

| Variable | Description |
|---|---|
| `WATSONX_URL` | watsonx.ai chat endpoint |
| `WATSONX_PROJECT_ID` | watsonx.ai project ID |
| `WATSONX_MODEL_ID` | Model ID (default: `ibm/granite-3-8b-instruct`) |
| `WATSONX_API_KEY` | IBM Cloud API key (`ApiKey-…`) |
| `PDF_PATH` | Path to the NAAC manual PDF |

---

## Test Queries (run with `--test`)

1. What is required under Key Indicator 1.1?
2. What are the qualitative metrics for Curriculum Enrichment?
3. What does Academic Flexibility measure?
4. What is the weightage of Metric 1.1.2?
5. What is required for Institutional Distinctiveness?
