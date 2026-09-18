"""Backward-compatibility module for the legacy SLM class.

Delegates to slm_router.models.transformers.TransformersRuntime.
"""

from typing import Optional
import torch

from slm_router.models.transformers import (
    DEFAULT_MODEL_NAME,
    TransformersRuntime,
    get_default_device,
)

MODEL_NAME = DEFAULT_MODEL_NAME


class SLM(TransformersRuntime):
    """Backward-compatible wrapper around TransformersRuntime.

    Preserves the legacy constructor signature SLM(device=None).
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        super().__init__(model_name=model_name, device=device)