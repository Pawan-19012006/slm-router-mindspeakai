# SLM Router — Model Architecture & Runtime Guide

This document describes the model abstraction layer, runtime configuration, device acceleration, model lifecycle, and empirical validation results across supported open-weight models.

---

## 1. Model Abstraction Architecture

The local intelligence layer decouples high-level classification and routing logic from low-level machine learning frameworks via an abstract interface:

```
Router / ClassifierV3
        ↓
    BaseModel (Abstract Interface)
        ↓
TransformersRuntime (Hugging Face PyTorch Implementation)
        ↓
Configured Causal Language Model (Qwen2.5, SmolLM2, etc.)
```

### `BaseModel` Interface (`src/slm_router/models/base.py`)
`BaseModel` defines the standard contract required by all routing and classification consumers:

- **`model_name: str`**: Public identifier of the active model.
- **`device: torch.device`**: Hardware accelerator currently hosting the model weights.
- **`generate(prompt, messages, max_new_tokens, do_sample, **kwargs) -> str`**: Token generation method supporting either raw text prompts or chat-templated message arrays (`[{"role": "...", "content": "..."}]`).

### `TransformersRuntime` (`src/slm_router/models/transformers.py`)
`TransformersRuntime` is the default concrete implementation of `BaseModel`. It:
- Manages weight downloading and caching via Hugging Face Hub.
- Applies chat templates via `AutoTokenizer.apply_chat_template(..., add_generation_prompt=True)`.
- Moves input tensors to the target hardware device.
- Slices prompt tokens out of output sequences to return clean, generated response strings.

---

## 2. Model Configuration & Swapping

### Precedence Hierarchy
The local model identifier is resolved via a three-tier hierarchy:

1. **Explicit Code Argument**:
   ```python
   runtime = TransformersRuntime(model_name="HuggingFaceTB/SmolLM2-360M-Instruct")
   ```
2. **Environment Configuration**:
   ```bash
   export LOCAL_MODEL="HuggingFaceTB/SmolLM2-360M-Instruct"
   ```
   or in `.env`:
   ```env
   LOCAL_MODEL=HuggingFaceTB/SmolLM2-360M-Instruct
   ```
3. **Default Fallback**:
   If unset or empty, the runtime automatically defaults to:
   ```
   Qwen/Qwen2.5-1.5B-Instruct
   ```

### Changing the Active Model
No code modifications in `router.py`, `classifier_v3.py`, or `api.py` are required to test or swap models. Simply update `LOCAL_MODEL` in `.env` or in your execution environment and start the service:

```bash
LOCAL_MODEL="HuggingFaceTB/SmolLM2-360M-Instruct" uv run uvicorn slm_router.api:app
```

---

## 3. Supported Model Categories & Compatibility

> [!IMPORTANT]
> Not all Hugging Face models are supported. The runtime specifically targets **decoder-only causal language models** that meet the following criteria:

- Compatible with `AutoTokenizer` and `AutoModelForCausalLM`.
- Provide a valid chat template (e.g. ChatML or standard instruction format).
- Fit within host memory (RAM or VRAM).

Unsupported models include:
- Encoder-only models (BERT, RoBERTa).
- Sequence-to-sequence encoder-decoder models (T5, BART).
- Vision/diffusion/audio architectures without compatible causal LM text heads.

---

## 4. Hardware Acceleration & Device Selection

The runtime automatically detects and selects the most performant available hardware accelerator:

1. **Apple Silicon (`mps`)**: Metal Performance Shaders on macOS devices (M1/M2/M3/M4).
2. **NVIDIA GPU (`cuda`)**: CUDA acceleration on Linux/Windows hosts.
3. **CPU (`cpu`)**: Fallback execution mode when no hardware accelerator is present.

---

## 5. Model Lifecycle & Memory Residency

To maintain predictable response latency:
- **FastAPI Transport Layer**: The configured model is loaded once into memory during application startup (`lifespan`) and kept resident for the process duration. No per-request re-instantiation occurs.
- **Python Direct Usage**: Instantiating `router = Router()` loads the model once and shares that single model instance between `ClassifierV3`, `_handle_local`, and `_handle_command`.

---

## 6. Empirical Model Validation (Benchmark Results)

The project includes a standardized 15-query evaluation benchmark (`tests/test_classifier.py`) spanning `LOCAL`, `COMMAND`, and `CLOUD` queries.

> [!NOTE]
> These figures represent empirical measurements taken against the repository's current development benchmark dataset. They are intended as baseline capability indicators for this routing task, not universal model rankings.

### Model A: `Qwen/Qwen2.5-1.5B-Instruct` (Default)
- **Parameters**: ~1.54 Billion
- **Status**: Production default
- **Device**: `mps`
- **Measured Accuracy**: **100.00% (15 / 15 correct)**
  - `LOCAL` accuracy: 5 / 5 (100%)
  - `COMMAND` accuracy: 5 / 5 (100%)
  - `CLOUD` accuracy: 5 / 5 (100%)
  - `UNKNOWN` count: 0
- **Classification Latency**: ~1.1s – 2.4s per query
- **Observations**: Strictly adheres to the classification prompt constraints; generates exact one-token labels (`LOCAL`, `COMMAND`, `CLOUD`) without conversational drift.

### Model B: `HuggingFaceTB/SmolLM2-360M-Instruct` (Validation Test)
- **Parameters**: ~360 Million
- **Status**: Validated alternative model (tested via `LOCAL_MODEL` without code changes or adapters)
- **Device**: `mps`
- **Measured Accuracy**:
  - Under V1 benchmark prompt (`test_classifier.py`): **13.33% (2 / 15 correct)**
  - Under V3 router prompt (`ClassifierV3`): **26.67% (4 / 15 correct)**
  - `LOCAL` accuracy: 1 / 5 (20%)
  - `COMMAND` accuracy: 0 / 5 (0%)
  - `CLOUD` accuracy: 3 / 5 (60%)
  - `UNKNOWN` count: 4 / 15 (queries where output was neither LOCAL, COMMAND, nor CLOUD)
- **Classification Latency**: ~0.40s – 0.52s per query (~2x–5x lower latency due to 360M parameter size)
- **Observations**:
  - Fully compatible with `TransformersRuntime` (weights loaded, tokenized, and generated cleanly).
  - Demonstrates an empirical bias toward generating the `CLOUD` label for simple commands.
  - Frequently begins directly answering or completing the query (e.g. producing `"Photosynthesis is a..."` instead of the required label `"LOCAL"`), resulting in clean fallback to `UNKNOWN`.
