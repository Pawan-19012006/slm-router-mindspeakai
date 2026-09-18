# SLM Router — HTTP API Reference

The SLM Router exposes a lightweight FastAPI HTTP interface that wraps the underlying `Router` and `ClassifierV3` pipeline.

The API runs by default on `http://127.0.0.1:8000`.

---

## Server Lifecycle

- The active Hugging Face model and `Router` singleton are initialized once during FastAPI application startup (`lifespan`) and kept in memory.
- Subsequent requests reuse the resident model weights with zero cold-start overhead.
- No model reloading occurs per request.

To start the server:
```bash
uv run uvicorn --app-dir src slm_router.api:app --host 127.0.0.1 --port 8000
```

---

## Endpoints Overview

| Method | Path | Request Body | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | None | Service liveness probe & active model identifier |
| `POST` | `/classify` | `{"query": "..."}` | 3-way request classification triage only (`LOCAL`, `COMMAND`, `CLOUD`) |
| `POST` | `/route` | `{"query": "..."}` | Complete 3-way execution pipeline (classification + downstream dispatch) |

---

## 1. Health Probe (`GET /health`)

### Purpose
Confirms that the FastAPI service is active and dynamically returns the identifier of the configured local model currently loaded in memory.

### Request
- **Method**: `GET`
- **Path**: `/health`
- **Headers**: None required
- **Body**: None

### Response (`200 OK`)
```json
{
  "status": "healthy",
  "model": "Qwen/Qwen2.5-1.5B-Instruct"
}
```

### Validation Behavior
- No query parameters or request body are accepted.

### Example Curl
```bash
curl -s http://127.0.0.1:8000/health
```

---

## 2. Request Classification (`POST /classify`)

### Purpose
Executes on-device 3-way classification triage using `ClassifierV3` without executing any downstream handlers (does not answer questions, execute commands, or call Google Gemini).

### Request
- **Method**: `POST`
- **Path**: `/classify`
- **Headers**: `Content-Type: application/json`
- **Body Schema**:
  ```json
  {
    "query": "string (min length: 1)"
  }
  ```

### Response (`200 OK`)
```json
{
  "query": "What is 2 + 2?",
  "label": "LOCAL",
  "raw_output": "LOCAL"
}
```

#### Possible Labels:
- `LOCAL`: Lightweight question, calculation, or explanation suitable for local resolution.
- `COMMAND`: Physical or operating system action intent.
- `CLOUD`: Deep analytical, long-form authoring, or complex research task.
- `UNKNOWN`: Unresolved or unclassifiable input.

### Validation Behavior
- Missing `query` field: Returns `422 Unprocessable Entity`.
- Empty string (`""`): Returns `422 Unprocessable Entity`.

### Example Curl
```bash
curl -s -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"query": "Turn off the office fan."}'
```

**Response:**
```json
{
  "query": "Turn off the office fan.",
  "label": "COMMAND",
  "raw_output": "COMMAND"
}
```

---

## 3. Full Pipeline Dispatch (`POST /route`)

### Purpose
Runs the complete end-to-end `Router.route()` pipeline:
1. Normalizes and validates query via `Preprocessor`.
2. Classifies intent via `ClassifierV3`.
3. Dispatches to the corresponding handler:
   - `LOCAL`: Answers directly on-device using the local causal language model.
   - `COMMAND`: Safely produces a dynamic simulated action confirmation on-device.
   - `CLOUD`: Dispatches complex workload to Google Gemini (or mock mode if API key not set).
   - `UNKNOWN`: Returns a safe fallback notice.
4. Gathers execution timings and telemetry into a structured JSON response.

### Request
- **Method**: `POST`
- **Path**: `/route`
- **Headers**: `Content-Type: application/json`
- **Body Schema**:
  ```json
  {
    "query": "string (min length: 1)"
  }
  ```

### Responses (`200 OK`)

#### A. LOCAL Route Example
**Request:**
```bash
curl -s -X POST http://127.0.0.1:8000/route \
  -H "Content-Type: application/json" \
  -d '{"query": "What is 2 + 2?"}'
```

**Response:**
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

#### B. COMMAND Route Example
**Request:**
```bash
curl -s -X POST http://127.0.0.1:8000/route \
  -H "Content-Type: application/json" \
  -d '{"query": "Turn on the light."}'
```

**Response:**
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

#### C. CLOUD Route Example
**Request:**
```bash
curl -s -X POST http://127.0.0.1:8000/route \
  -H "Content-Type: application/json" \
  -d '{"query": "Write a 3000-word research essay on artificial intelligence."}'
```

**Response (Mock Mode / Live Mode depending on GEMINI_API_KEY):**
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

### Validation Behavior
- Missing `query` key: Returns `422 Unprocessable Entity`.
- Empty string (`""`): Returns `422 Unprocessable Entity`.
- Whitespace-only string (`"   "`): Returns `200 OK` with `route: "UNKNOWN"` and `success: false` via internal preprocessor triage.
