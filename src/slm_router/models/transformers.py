import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from slm_router.models.base import BaseModel

load_dotenv()

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def get_default_device() -> torch.device:
    """Auto-detect CUDA, MPS, or fall back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TransformersRuntime(BaseModel):
    """Generic Hugging Face Transformers model runtime.

    Loads and executes causal language models supported by AutoTokenizer and AutoModelForCausalLM.
    Default model is Qwen/Qwen2.5-1.5B-Instruct (configurable via LOCAL_MODEL environment variable).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        load_dotenv()
        env_model = os.getenv("LOCAL_MODEL", "").strip()
        self.model_name = model_name or (env_model if env_model else DEFAULT_MODEL_NAME)
        self.device = torch.device(device) if device is not None else get_default_device()
        print(f"Loading Transformers model '{self.model_name}' on device: {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)

        print(f"Model '{self.model_name}' loaded on {self.device}.")

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 100,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> str:
        if messages is None:
            if prompt is None:
                raise ValueError("Either prompt or messages must be provided.")
            messages = [
                {"role": "user", "content": prompt}
            ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        # Move all input tensors to the model's device
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **kwargs,
        )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return response.strip()
