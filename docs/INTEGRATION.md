# SLM Router — Developer Integration Guide

This guide explains how external services, edge applications, and Python codebases can integrate with SLM Router.

---

## 1. High-Level Integration Flow

External applications submit raw natural language requests to SLM Router over HTTP or directly via the Python package:

```
External Application / Client
        ↓
Transport Layer (HTTP API or Python In-Process)
        ↓
    SLM Router
        ↓
┌─────────────────┬────────────────────────┬──────────────────────┐
│     LOCAL       │        COMMAND         │        CLOUD         │
│  (Direct SLM)   │ (Simulated Local Conf) │ (Google Gemini LLM)  │
└─────────────────┴────────────────────────┴──────────────────────┘
```

---

## 2. Integration Option A: Python Direct Usage (In-Process)

For Python applications (e.g. desktop tools, CLI utilities, background workers) running on the same host, the router can be imported directly with zero network overhead.

### Basic In-Process Routing
```python
from slm_router.router import Router

# Initialize the router (loads model weights into memory once)
router = Router()

# Route a query through the full 3-way pipeline
result = router.route("What is 2 + 2?")

print(f"Assigned Route: {result['route']}")
print(f"Handler:        {result['handler']}")
print(f"Success:        {result['success']}")
print(f"Response:       {result['response']}")
print(f"Total Duration: {result['timings']['total']}s")
```

### Direct Classification Only
If your application only needs intent classification and handles execution independently:
```python
from slm_router.router import Router

router = Router()
label, raw_output = router.classifier.classify_with_raw("Turn off the living room lights.")

print(f"Category Label: {label}")  # 'COMMAND'
```

### Dependency Injection
Custom runtimes conforming to `BaseModel` can be explicitly injected:
```python
from slm_router.models.transformers import TransformersRuntime
from slm_router.router import Router

custom_runtime = TransformersRuntime(model_name="Qwen/Qwen2.5-1.5B-Instruct")
router = Router(model=custom_runtime)
```

---

## 3. Integration Option B: HTTP API

For microservices, web apps, or applications written in other languages (Node.js, Go, Rust, Swift, Kotlin), use the FastAPI transport layer.

### Starting the Service
```bash
uv run uvicorn slm_router.api:app --host 0.0.0.0 --port 8000
```

### Classify Only (`POST /classify`)
Lightweight intent triage returning the classification label:
```bash
curl -s -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"query": "Turn on the porch light."}'
```

```json
{
  "query": "Turn on the porch light.",
  "label": "COMMAND",
  "raw_output": "COMMAND"
}
```

### Full Route Pipeline (`POST /route`)
Full execution dispatch returning response text, execution status, and timing breakdown:
```bash
curl -s -X POST http://127.0.0.1:8000/route \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain photosynthesis in simple terms."}'
```

```json
{
  "query": "Explain photosynthesis in simple terms.",
  "route": "LOCAL",
  "handler": "Local SLM",
  "response": "Photosynthesis is the process by which green plants use sunlight to synthesize nutrients...",
  "success": true,
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "timings": {
    "classification": 1.15,
    "handler": 1.20,
    "total": 2.35
  }
}
```

---

## 4. Configuration Reference (`LOCAL_MODEL`)

You can change the active local causal language model without modifying application code:

1. In your `.env` file:
   ```env
   LOCAL_MODEL=Qwen/Qwen2.5-1.5B-Instruct
   ```
2. Or via environment variable before launching the server or Python script:
   ```bash
   LOCAL_MODEL=HuggingFaceTB/SmolLM2-360M-Instruct uv run uvicorn slm_router.api:app
   ```

If unset, the system defaults to `Qwen/Qwen2.5-1.5B-Instruct`.
