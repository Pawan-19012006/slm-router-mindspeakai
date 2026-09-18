# SLM Router

A lightweight API service that classifies user queries into `LOCAL`, `COMMAND`, or `CLOUD` and routes them to the appropriate handler. The local model is accessed through a generic model abstraction, allowing compatible Hugging Face models to be selected through configuration without changing router or classifier source code.

---

## What It Does

Incoming user requests flow through a decoupled classification and execution pipeline:

```
Client (HTTP / Python)
  ↓
FastAPI
  ↓
Router
  ↓
ClassifierV3
  ↓
Routing Decision
  ├── LOCAL   → Local Hugging Face model (direct on-device response)
  ├── COMMAND → Command handler (safe dynamic simulated confirmation)
  └── CLOUD   → Configured cloud handler (Google Gemini API / mock fallback)
```

- **Router**: Evaluates query metadata, coordinates triage, measures timings, and decides where the request goes. The router itself is an orchestrator, not a language model.
- **Local SLM**: An on-device causal language model that generates responses for `LOCAL` queries and dynamic action confirmations for `COMMAND` intents.
- **Classification Labels**: The active classifier (`ClassifierV3`) categorizes requests into `LOCAL`, `COMMAND`, or `CLOUD`. An `UNKNOWN` label is returned as a safe fallback if model output is unclassifiable or unparseable.

---

## Current Architecture

The codebase is organized into five decoupled layers:

1. **FastAPI Transport Layer** ([src/slm_router/api.py](src/slm_router/api.py)): Exposes HTTP endpoints (`/health`, `/classify`, `/route`), enforces Pydantic request validation, serializes responses, and manages singleton application lifecycle. Contains zero routing or classification logic.
2. **Router / Orchestration Layer** ([src/slm_router/router.py](src/slm_router/router.py)): Coordinates input preprocessing ([preprocessor.py](src/slm_router/preprocessor.py)), timing telemetry, and handler dispatch ([commands.py](src/slm_router/commands.py), [cloud.py](src/slm_router/cloud.py)).
3. **Classification Triage Layer** ([src/slm_router/classifier_v3.py](src/slm_router/classifier_v3.py)): Performs prompt-constrained 3-way triage via `ClassifierV3`, depending strictly on the abstract model interface.
4. **Generic Model Abstraction** ([src/slm_router/models/base.py](src/slm_router/models/base.py)): Defines the `BaseModel` abstract interface (`generate`, `model_name`, `device`) that decouples routing logic from concrete machine learning libraries.
5. **Transformers Model Runtime** ([src/slm_router/models/transformers.py](src/slm_router/models/transformers.py)): Concrete `BaseModel` implementation using Hugging Face `transformers` and `torch`. Handles tokenization, chat templates, hardware acceleration (`mps`, `cuda`, `cpu`), and token generation.

---

## Requirements

- **Python**: `>= 3.11` (specified in `pyproject.toml` and `.python-version`)
- **Package Manager**: [`uv`](https://docs.astral.sh/uv/)
- **Local Model**: An open-weight causal language model supported by Hugging Face `AutoModelForCausalLM` and `AutoTokenizer` (defaults to `Qwen/Qwen2.5-1.5B-Instruct`)
- **Hardware Acceleration**: Apple Silicon (`mps`), NVIDIA GPU (`cuda`), or `cpu` fallback
- **Cloud API Key**: A Google Gemini API key is required only when `CLOUD_MODE=live` is used

---

## Installation

Clone the repository and install dependencies using `uv`:

```bash
git clone <repository-url>
cd SLM-Router
uv sync
```

---

## Configuration

Environment variables are managed through a `.env` file in the repository root. Create one from the provided template:

```bash
cp .env.example .env
```

Example `.env` configuration:

```env
LOCAL_MODEL=Qwen/Qwen2.5-1.5B-Instruct
GEMINI_API_KEY=your-api-key-here
CLOUD_MODEL=gemini-3.6-flash
CLOUD_MODE=live
```

### Configuration Rules:
- **`LOCAL_MODEL`**: Specifies the Hugging Face repository identifier. Model IDs must be complete repository identifiers (e.g. `Qwen/Qwen2.5-1.5B-Instruct`, not `Qwen2.5-1.5B-Instruct`).
- **Default Model**: If `LOCAL_MODEL` is omitted or left empty, the runtime automatically falls back to `Qwen/Qwen2.5-1.5B-Instruct`.
- **Precedence**: Explicit model parameters passed in Python code take precedence over environment variables.
- **`GEMINI_API_KEY`**: Only required for live cloud calls. If omitted or empty, cloud routing automatically operates in simulated `mock` mode.
- **`CLOUD_MODE`**: Set to `live` to execute live Google Gemini API requests via the `google-genai` SDK, or `mock` to return simulated cloud payloads locally without external network transit.
- **Security**: Never commit `.env` or sensitive API keys to version control.

---

## Start the API

Start the FastAPI application with Uvicorn using the project's source directory layout:

```bash
uv run uvicorn --app-dir src slm_router.api:app --host 127.0.0.1 --port 8000
```

The service initializes the configured local model into memory during startup and binds to `http://127.0.0.1:8000`.

---

## Swagger / OpenAPI

FastAPI automatically generates an interactive documentation and testing interface. Once the server is running, open:

**[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

From the Swagger UI, you can inspect endpoint schemas and execute live test requests directly in your browser without writing curl commands.

The raw OpenAPI specification is available at:
**[http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)**

*(Note: Swagger UI is the standard interactive API testing interface generated by FastAPI, not a custom application dashboard.)*

---

## API Usage

### 1. `GET /health`

**Purpose**: Check service health and report the currently active local model.

**Example Request**:
```bash
curl -s http://127.0.0.1:8000/health
```

**Response (`200 OK`)**:
```json
{
  "status": "healthy",
  "model": "Qwen/Qwen2.5-1.5B-Instruct"
}
```

---

### 2. `POST /classify`

**Purpose**: Run on-device 3-way classification triage only without executing downstream handlers.

**Request Schema**:
```json
{
  "query": "Turn off the office fan."
}
```

**Example Request**:
```bash
curl -s -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"query": "Turn off the office fan."}'
```

**Response (`200 OK`)**:
```json
{
  "query": "Turn off the office fan.",
  "label": "COMMAND",
  "raw_output": "COMMAND"
}
```

**Supported Classification Labels**:
- `LOCAL`: Informational, factual, or explanatory queries suited for local resolution.
- `COMMAND`: Action intents targeting device state changes or system automation.
- `CLOUD`: Complex research, deep analysis, or long-form writing exceeding local budget.
- `UNKNOWN`: Fallback label returned when model generation does not match a valid category.

---

### 3. `POST /route`

**Purpose**: Execute the complete classification and dispatch pipeline.

**Request Schema**:
```json
{
  "query": "What is 2 + 2?"
}
```

**Response Fields**:
- `query`: The sanitized input query.
- `route`: Assigned category (`LOCAL`, `COMMAND`, `CLOUD`, or `UNKNOWN`).
- `handler`: Name of the dispatch handler that executed the request.
- `processing_type`: Processing category (`local`, `command`, `cloud`, `fallback`, or `none`).
- `response` / `result`: Primary generated text answer or action confirmation.
- `success`: Boolean indicating successful processing.
- `mode`: Execution mode (`LOCAL`, `MOCK`, `LIVE`, or `FALLBACK`).
- `model`: Model identifier responsible for the response.
- `details`: Metadata including classification token and handler status.
- `timings`: Latency telemetry dictionary (`classification`, `handler`, `total` in seconds).

#### Representative Responses:

**A. LOCAL Route** (`"What is 2 + 2?"`):
```json
{
  "query": "What is 2 + 2?",
  "route": "LOCAL",
  "handler": "Local SLM",
  "processing_type": "local",
  "response": "The sum of 2 plus 2 is 4.",
  "result": "The sum of 2 plus 2 is 4.",
  "success": true,
  "mode": "LOCAL",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "details": {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "classification_token": "LOCAL",
    "status": "COMPLETED_LOCALLY"
  },
  "timings": {
    "classification": 1.121,
    "handler": 0.684,
    "total": 1.805
  }
}
```

**B. COMMAND Route** (`"Turn on the light."`):
```json
{
  "query": "Turn on the light.",
  "route": "COMMAND",
  "handler": "Local SLM",
  "processing_type": "command",
  "action": "Action Confirmation",
  "status": "COMMAND EXECUTED",
  "response": "Sure thing! The light has been turned on for you.",
  "result": "Sure thing! The light has been turned on for you.",
  "success": true,
  "mode": "LOCAL",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "details": {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "classification_token": "COMMAND",
    "status": "EXECUTED_DYNAMICALLY",
    "success": true
  },
  "timings": {
    "classification": 1.130,
    "handler": 1.023,
    "total": 2.152
  }
}
```

**C. CLOUD Route** (`"Write a 3000-word research essay on artificial intelligence."`):
```json
{
  "query": "Write a 3000-word research essay on artificial intelligence.",
  "route": "CLOUD",
  "handler": "Cloud LLM",
  "processing_type": "cloud",
  "status": "READY FOR CLOUD LLM (MOCK MODE)",
  "response": "Cloud LLM routing selected.\nThis request has been identified as exceeding local SLM resource budget.\nPrepared payload for target endpoint: gemini-3.6-flash via Google Gemini.\n(Mock mode active — no external API calls made).",
  "result": "Cloud LLM routing selected.\nThis request has been identified as exceeding local SLM resource budget.\nPrepared payload for target endpoint: gemini-3.6-flash via Google Gemini.\n(Mock mode active — no external API calls made).",
  "success": true,
  "mode": "MOCK",
  "model": "gemini-3.6-flash",
  "details": {
    "provider": "Cloud LLM (Google Gemini Provider)",
    "target_model": "gemini-3.6-flash",
    "complexity": "Moderate / High",
    "classification_token": "CLOUD",
    "mode": "MOCK"
  },
  "timings": {
    "classification": 1.202,
    "handler": 0.0,
    "total": 1.202
  }
}
```
*(Note: In live mode, cloud execution depends on external network connectivity and Google Gemini API quota; requests may fail if credentials or quotas are invalid.)*

---

## Quick API Test

The **primary and recommended** method for manual exploration is the interactive Swagger documentation at **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**.

Alternatively, test via `curl`:

```bash
# 1. Health check
curl -s http://127.0.0.1:8000/health

# 2. Classify intent
curl -s -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"query": "Turn on the light."}'

# 3. Full routing pipeline
curl -s -X POST http://127.0.0.1:8000/route \
  -H "Content-Type: application/json" \
  -d '{"query": "What is 2 + 2?"}'
```

---

## Python Integration

An executable, zero-dependency client script is provided in [examples/python_client.py](examples/python_client.py):

```bash
python3 examples/python_client.py
```

### Calling via Python HTTP Requests:
```python
import urllib.request
import json

payload = json.dumps({"query": "What is 2 + 2?"}).encode("utf-8")
req = urllib.request.Request(
    "http://127.0.0.1:8000/route",
    data=payload,
    headers={"Content-Type": "application/json"}
)

with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read().decode("utf-8"))
    print("Route:", result["route"])
    print("Response:", result["response"])
```

### In-Process Python Usage (Without Server):
You can also import and use the router directly in Python code:

```python
from slm_router.router import Router

# Initialize model once into memory
router = Router()

# Route queries directly
result = router.route("What is 2 + 2?")
print("Route:", result["route"])
print("Response:", result["response"])
```

See [docs/INTEGRATION.md](docs/INTEGRATION.md) for full integration details.

---

## Model Integration & Switching Models

The router does **not** rely on model-specific adapters (such as `QwenAdapter` or `SmolLMAdapter`). Instead, it uses a generic `BaseModel` abstraction and a unified `TransformersRuntime`.

Any compatible causal language model available on Hugging Face can be loaded through configuration:

```env
LOCAL_MODEL=Qwen/Qwen2.5-1.5B-Instruct
```
or:
```env
LOCAL_MODEL=HuggingFaceTB/SmolLM2-360M-Instruct
```

### How to Switch Models:
1. Update `LOCAL_MODEL` in `.env` (or pass it as an environment variable).
2. Restart the API server:
   ```bash
   LOCAL_MODEL="HuggingFaceTB/SmolLM2-360M-Instruct" uv run uvicorn --app-dir src slm_router.api:app --host 127.0.0.1 --port 8000
   ```
3. Verify the loaded model via `GET /health`.
4. The exact same API, Router, and ClassifierV3 pipelines remain active.

### Compatibility vs. Classification Quality:
- **Model Compatibility**: Determines whether weights can be downloaded, loaded, and executed via Hugging Face `AutoModelForCausalLM` and `AutoTokenizer` with chat template support.
- **Classification Quality**: Determines how reliably a given model adheres to the system prompt. For example, while `Qwen/Qwen2.5-1.5B-Instruct` achieves 100% on the benchmark dataset, smaller models (such as `SmolLM2-360M-Instruct`) may experience instruction drift and output conversational text instead of strict categorical labels.

See [docs/MODELS.md](docs/MODELS.md) for comprehensive empirical measurements and validation details.

---

## Example Workflow

A quick walkthrough for evaluating the project:

1. **Clone**: `git clone <repository-url> && cd SLM-Router`
2. **Install**: `uv sync`
3. **Configure**: `cp .env.example .env`
4. **Start Service**: `uv run uvicorn --app-dir src slm_router.api:app --host 127.0.0.1 --port 8000`
5. **Open Interactive Docs**: Navigate to `http://127.0.0.1:8000/docs`
6. **Test Health**: Execute `GET /health` to confirm active model (`Qwen/Qwen2.5-1.5B-Instruct`)
7. **Test Classification**: Execute `POST /classify` with `"What is 2 + 2?"` (returns `LOCAL`)
8. **Test Routing**: Execute `POST /route` with `"Turn on the light."` (returns `COMMAND`)
9. **Test Model Swapping**: Set `LOCAL_MODEL=HuggingFaceTB/SmolLM2-360M-Instruct` in `.env`
10. **Verify Alternate Model**: Restart server and confirm `GET /health` reports `HuggingFaceTB/SmolLM2-360M-Instruct`

---

## Testing

Run all unit tests and evaluation benchmarks:

```bash
# 1. Run all 34 unit tests (API endpoints, model abstraction, router, handlers)
uv run python -m unittest discover -s src/slm_router/tests -p "test_*.py"

# 2. Run the 15-query classifier evaluation benchmark
uv run python tests/test_classifier.py

# 3. Verify Python bytecode compilation across packages
uv run python -m compileall -q src examples
```

*Note on Benchmarks: The included 15-query test suite verified 15/15 (100.00% accuracy) for `Qwen/Qwen2.5-1.5B-Instruct`. This is a development baseline evaluation dataset and does not represent all possible inputs or other model architectures.*

---

## Project Structure

```
SLM-Router/
├── src/
│   └── slm_router/
│       ├── api.py                     # FastAPI transport layer & HTTP endpoints
│       ├── router.py                  # Core 3-way routing orchestrator
│       ├── classifier_v3.py           # V3 prompt-based 3-way classifier
│       ├── classifier.py              # Legacy V1 classifier prompt (retained for benchmark baseline)
│       ├── preprocessor.py            # Input validation and string normalization
│       ├── commands.py                # Command sandbox and action confirmation helpers
│       ├── cloud.py                   # Google Gemini integration & mock mode fallback
│       ├── model.py                   # Backward-compatible SLM wrapper
│       ├── models/
│       │   ├── __init__.py            # Model package exports
│       │   ├── base.py                # BaseModel abstract interface
│       │   └── transformers.py        # Generic Transformers runtime & LOCAL_MODEL resolver
│       └── tests/
│           ├── test_api.py            # FastAPI endpoint unit tests
│           ├── test_model_abstraction.py # Generic model abstraction unit tests
│           ├── test_classifier_v3.py  # ClassifierV3 unit tests
│           ├── test_router.py         # Router orchestrator unit tests
│           ├── test_preprocessor.py   # Preprocessor unit tests
│           ├── test_commands.py       # Command handling unit tests
│           └── test_cloud.py          # Cloud handler unit tests
├── tests/
│   └── test_classifier.py             # 15-query classifier evaluation benchmark
├── docs/
│   ├── ARCHITECTURE.md                # Detailed system architecture and sequence diagrams
│   ├── API.md                         # Complete HTTP API endpoint specification
│   ├── MODELS.md                      # Model runtime, lifecycle, and benchmark data
│   └── INTEGRATION.md                 # In-process Python and microservice integration guide
├── examples/
│   └── python_client.py               # Minimal Python HTTP API client example
├── pyproject.toml                     # Project dependencies and build configuration
└── uv.lock                            # Deterministic dependency lockfile
```

---

## Documentation

For in-depth architectural and technical references, see the `docs/` directory:
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): Comprehensive system architecture, layer separation, request lifecycles, and memory management.
- [docs/API.md](docs/API.md): Detailed REST API contracts, request/response schemas, validation rules, and curl examples.
- [docs/MODELS.md](docs/MODELS.md): Guide to `BaseModel`, `TransformersRuntime`, device selection, and empirical benchmark findings.
- [docs/INTEGRATION.md](docs/INTEGRATION.md): Integration guide covering both direct in-process Python usage and HTTP service patterns.

---

## Important Notes & Limitations

- **Simulated Command Execution**: The `COMMAND` route produces dynamic simulated action confirmations on-device. It deliberately does not execute arbitrary shell commands or operating system binaries to ensure host security.
- **Cloud Execution Dependency**: The `CLOUD` route requires a valid `GEMINI_API_KEY` for live remote execution; without an API key, it operates in simulated mock mode.
- **Model Capability Dependence**: Classification accuracy directly reflects the instruction-following capabilities of the configured model. Smaller models may exhibit label hallucination or prompt drift.
- **Supported Categories**: The active classifier classifies queries strictly into `LOCAL`, `COMMAND`, and `CLOUD`.

---

## Security

- **API Keys**: Store API keys exclusively in `.env`. Never commit `.env` or credential files to version control.
- **Credential Redaction**: Cloud error logs automatically redact API credentials before formatting error strings.
- **Local Development Scope**: The FastAPI service is configured for local development and internal network integration. If exposing over public networks, configure appropriate reverse proxy, authentication, and rate limiting safeguards.
