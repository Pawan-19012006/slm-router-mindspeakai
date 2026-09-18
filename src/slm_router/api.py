"""FastAPI transport layer for SLM Router."""

from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from slm_router.router import Router

_router_instance: Optional[Router] = None


def get_router() -> Router:
    """Retrieve or initialize the singleton Router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = Router()
    return _router_instance


def set_router(router: Optional[Router]) -> None:
    """Set or reset the global Router instance (useful for testing/mocking)."""
    global _router_instance
    _router_instance = router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eagerly initialize the model/router on application startup."""
    get_router()
    yield


app = FastAPI(
    title="SLM Router API",
    description="Lightweight HTTP interface for local SLM classification and 3-way routing",
    version="0.1.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query to classify or route")


class ClassifyResponse(BaseModel):
    query: str
    label: str
    raw_output: str


class HealthResponse(BaseModel):
    status: str
    model: str


@app.get("/health", response_model=HealthResponse)
def health(router: Router = Depends(get_router)) -> HealthResponse:
    """Check API health and return the active local model name."""
    model_name = getattr(router.model, "model_name", "unknown")
    return HealthResponse(status="healthy", model=model_name)


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: QueryRequest, router: Router = Depends(get_router)) -> ClassifyResponse:
    """Classify a query using the underlying ClassifierV3 without executing handlers."""
    label, raw_output = router.classifier.classify_with_raw(request.query)
    return ClassifyResponse(query=request.query, label=label, raw_output=raw_output)


@app.post("/route")
def route(request: QueryRequest, router: Router = Depends(get_router)) -> Dict[str, Any]:
    """Execute the full 3-way Router pipeline and return the result."""
    return router.route(request.query)
