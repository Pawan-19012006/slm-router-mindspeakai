# SLM Router

> Local AI Request Classification & Intelligent 3-Way Dispatch Engine.

**SLM Router** classifies incoming natural language requests on-device using an open-weight Small Language Model (SLM) and automatically routes each query to the optimal execution path:
1. **`LOCAL`**: Answering bounded factual and explanatory questions directly on-device.
2. **`COMMAND`**: Identifying automation intents and generating safe, simulated action confirmations on-device.
3. **`CLOUD`**: Offloading deep research, complex reasoning, and long-form writing to Google Gemini.

---

## High-Level Architecture

```
HTTP Client / External App
           ↓
FastAPI (src/slm_router/api.py) [Transport Layer]
           ↓
Router (src/slm_router/router.py) [Orchestration Layer]
           ↓
ClassifierV3 (src/slm_router/classifier_v3.py) [Classification Triage]
           ↓
BaseModel (src/slm_router/models/base.py) [Generic Model Contract]
           ↓
TransformersRuntime (src/slm_router/models/transformers.py) [Hugging Face Runtime]
           ↓
Configured Open-Weight Model (Default: Qwen/Qwen2.5-1.5B-Instruct)
```

---

## Requirements

- **Python**: `>= 3.11`
- **Package Manager**: [`uv`](https://docs.astral.sh/uv/)
- **Hardware Acceleration**:
  - Apple Silicon (`mps`) — recommended for macOS
  - NVIDIA GPU (`cuda`) — recommended for Linux / Windows
  - CPU (`cpu`) — supported as fallback
- **Memory**: ~4 GB available RAM / VRAM for the default 1.5B model.

---

## Quick Start & Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd SLM-Router

# 2. Sync/install dependencies using uv
uv sync

# 3. Create .env from template
cp .env.example .env

# 4. (Optional) Set your Google Gemini API key in .env for live cloud routing
# GEMINI_API_KEY=your_key_here

# 5. Launch the FastAPI server
uv run uvicorn --app-dir src slm_router.api:app --host 127.0.0.1 --port 8000
```

---

## Environment Configuration

Configuration is managed via `.env` or system environment variables:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `LOCAL_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Hugging Face repository identifier for the local causal language model. |
| `GEMINI_API_KEY` | `""` | Google Gemini API key. If empty, cloud routing runs in simulated mock mode. |
| `CLOUD_MODEL` | `gemini-3.6-flash` | Gemini model endpoint for cloud offloading. |
| `CLOUD_MODE` | `live` | Mode for cloud requests (`live` or `mock`). |

---

## FastAPI Endpoints

When running `uv run uvicorn --app-dir src slm_router.api:app --host 127.0.0.1 --port 8000`, the service exposes three endpoints on `http://127.0.0.1:8000`:

| Method | Path | Request Body | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | None | Health check & active local model reporting |
| `POST` | `/classify` | `{"query": "..."}` | 3-way request classification triage only |
| `POST` | `/route` | `{"query": "..."}` | Complete 3-way classification + handler execution |

### 1. Health Probe (`GET /health`)
```bash
curl -s http://127.0.0.1:8000/health
```
```json
{
  "status": "healthy",
  "model": "Qwen/Qwen2.5-1.5B-Instruct"
}
```

### 2. Classify Query (`POST /classify`)
```bash
curl -s -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"query": "Turn off the office fan."}'
```
```json
{
  "query": "Turn off the office fan.",
  "label": "COMMAND",
  "raw_output": "COMMAND"
}
```

### 3. Route Query (`POST /route`)
```bash
curl -s -X POST http://127.0.0.1:8000/route \
  -H "Content-Type: application/json" \
  -d '{"query": "What is 2 + 2?"}'
```
```json
{
  "query": "What is 2 + 2?",
  "route": "LOCAL",
  "handler": "Local SLM",
  "response": "The sum of 2 plus 2 is 4.",
  "success": true,
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "timings": {
    "classification": 1.12,
    "handler": 0.68,
    "total": 1.80
  }
}
```

---

## Direct Python Usage (In-Process)

You can use the router directly inside Python without starting the FastAPI server:

```python
from slm_router.router import Router

# Loads model into memory once and reuses across calls
router = Router()

# Route queries directly
result = router.route("What is 2 + 2?")
print("Assigned Route:", result["route"])      # LOCAL
print("Response:      ", result["response"])   # "The sum of 2 plus 2 is 4."
print("Execution Time:", result["timings"]["total"], "seconds")
```

See [docs/INTEGRATION.md](docs/INTEGRATION.md) and [examples/python_client.py](examples/python_client.py) for more integration patterns.

---

## Changing the Local Model (`LOCAL_MODEL`)

You can swap the local model without editing any code in the router or classifier:

```bash
# Set in shell or in .env
LOCAL_MODEL="HuggingFaceTB/SmolLM2-360M-Instruct" uv run uvicorn --app-dir src slm_router.api:app --host 127.0.0.1 --port 8000
```

The runtime supports causal language models compatible with `AutoTokenizer` and `AutoModelForCausalLM`. See [docs/MODELS.md](docs/MODELS.md) for compatibility details and empirical benchmarks.

---

## Running Tests

```bash
# 1. Run all unit tests (includes API, model abstraction, router, and handlers)
uv run python -m unittest discover -s src/slm_router/tests -p "test_*.py"

# 2. Run the 15-query classifier evaluation benchmark
uv run python tests/test_classifier.py
```

---

## Limitations

- **Model Compatibility**: Targets decoder-only causal language models. Encoder-only architectures (e.g. BERT) or multimodal models without standard text generation heads are not supported.
- **Safety Simulation for Commands**: The `COMMAND` route produces dynamic simulated action confirmations on-device. It deliberately does not execute raw operating system shell commands to prevent untrusted execution.
- **Model Capability Dependence**: Classification accuracy depends on the instruction-following capabilities of the local model. While `Qwen/Qwen2.5-1.5B-Instruct` achieves 100% on the standard benchmark, smaller models (e.g. 360M parameters) may exhibit prompt drift or label hallucination.
- **Cloud API Key**: The `CLOUD` route requires a valid `GEMINI_API_KEY` for live model inference; otherwise, it falls back to simulated mock mode.

---

## Documentation Links

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): System architecture, layer responsibilities, and request lifecycles.
- [docs/API.md](docs/API.md): Comprehensive HTTP API endpoint reference.
- [docs/MODELS.md](docs/MODELS.md): Model abstraction, lifecycle, device management, and benchmark results.
- [docs/INTEGRATION.md](docs/INTEGRATION.md): In-process Python and HTTP integration guides.
