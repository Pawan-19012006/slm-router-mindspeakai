"""Abstract base interface for model runtimes."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseModel(ABC):
    """Abstract base class defining the minimal interface required by ClassifierV3 and Router."""

    @abstractmethod
    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 100,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> str:
        """Generate text given a prompt or chat messages.

        Args:
            prompt: Optional raw text prompt.
            messages: Optional list of chat message dictionaries with 'role' and 'content'.
            max_new_tokens: Maximum number of tokens to generate.
            do_sample: Whether to use sampling (False for greedy deterministic decoding).
            **kwargs: Additional runtime-specific generation parameters.

        Returns:
            The generated text string stripped of special tokens.
        """
        pass
