"""Focused tests for the generic model abstraction layer."""

import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Ensure src is in sys.path
src_path = Path(__file__).resolve().parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from slm_router.models.base import BaseModel
from slm_router.models.transformers import TransformersRuntime, DEFAULT_MODEL_NAME, get_default_device
from slm_router.models import BaseModel as BaseModelFromInit, TransformersRuntime as TransformersRuntimeFromInit
from slm_router.model import SLM, MODEL_NAME
from slm_router.classifier_v3 import ClassifierV3
from slm_router.router import Router
from slm_router.cloud import CloudHandler


class DummyModel(BaseModel):
    def __init__(self, response="LOCAL"):
        self.response = response
        self.last_generate_kwargs = {}

    def generate(self, prompt=None, messages=None, max_new_tokens=100, do_sample=False, **kwargs):
        self.last_generate_kwargs = {
            "prompt": prompt,
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            **kwargs,
        }
        return self.response


class TestModelAbstraction(unittest.TestCase):

    def test_generic_model_interface_import_and_contract(self):
        """Verify BaseModel can be imported from both module paths and enforces ABC contract."""
        self.assertIs(BaseModel, BaseModelFromInit)

        # Cannot instantiate ABC directly
        with self.assertRaises(TypeError):
            BaseModel()

        # Dummy subclass can be instantiated
        dummy = DummyModel()
        self.assertIsInstance(dummy, BaseModel)
        self.assertEqual(dummy.generate(prompt="test"), "LOCAL")

    def test_transformers_runtime_configuration(self):
        """Verify TransformersRuntime configuration and inheritance without downloading models."""
        self.assertIs(TransformersRuntime, TransformersRuntimeFromInit)
        self.assertTrue(issubclass(TransformersRuntime, BaseModel))
        self.assertEqual(DEFAULT_MODEL_NAME, "Qwen/Qwen2.5-1.5B-Instruct")

        with patch("slm_router.models.transformers.AutoTokenizer.from_pretrained") as mock_tok, \
             patch("slm_router.models.transformers.AutoModelForCausalLM.from_pretrained") as mock_model:
            mock_model_instance = MagicMock()
            mock_model.return_value = mock_model_instance

            runtime = TransformersRuntime(model_name="custom/test-model")
            self.assertEqual(runtime.model_name, "custom/test-model")
            mock_tok.assert_called_once_with("custom/test-model")
            mock_model.assert_called_once_with("custom/test-model")
            mock_model_instance.to.assert_called_once_with(runtime.device)

    def test_classifier_v3_works_with_generic_base_model(self):
        """Verify ClassifierV3 works seamlessly with any BaseModel implementation."""
        dummy_model = DummyModel(response="COMMAND")
        classifier = ClassifierV3(model=dummy_model)

        self.assertIs(classifier.model, dummy_model)
        self.assertIs(classifier.slm, dummy_model)

        label, raw = classifier.classify_with_raw("Turn off the lights.")
        self.assertEqual(label, "COMMAND")
        self.assertEqual(raw, "COMMAND")

        # Verify arguments passed to generate
        self.assertEqual(dummy_model.last_generate_kwargs["max_new_tokens"], 4)
        self.assertFalse(dummy_model.last_generate_kwargs["do_sample"])
        messages = dummy_model.last_generate_kwargs["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1]["content"], "Turn off the lights.")

    def test_classifier_v3_backward_compatibility_positional_slm(self):
        """Verify ClassifierV3 accepts positional model argument and sets .slm alias."""
        dummy_model = DummyModel(response="CLOUD")
        classifier = ClassifierV3(dummy_model)

        self.assertIs(classifier.model, dummy_model)
        self.assertIs(classifier.slm, dummy_model)
        self.assertEqual(classifier.classify("Write an essay on AI."), "CLOUD")

    def test_router_works_with_generic_base_model(self):
        """Verify Router works seamlessly with any BaseModel implementation."""
        dummy_model = DummyModel(response="Simulated local response")
        mock_classifier = MagicMock()
        mock_classifier.classify_with_raw.return_value = ("LOCAL", "LOCAL")

        router = Router(
            model=dummy_model,
            classifier=mock_classifier,
            cloud_handler=CloudHandler(mode="mock"),
        )

        self.assertIs(router.model, dummy_model)
        self.assertIs(router.slm, dummy_model)

        res = router.route("What is photosynthesis?")
        self.assertEqual(res["route"], "LOCAL")
        self.assertEqual(res["response"], "Simulated local response")
        self.assertTrue(res["success"])

    def test_router_backward_compatibility_slm_kwarg(self):
        """Verify Router accepts slm kwarg for backwards compatibility."""
        dummy_model = DummyModel(response="Simulated command response")
        mock_classifier = MagicMock()
        mock_classifier.classify_with_raw.return_value = ("COMMAND", "COMMAND")

        router = Router(
            slm=dummy_model,
            classifier=mock_classifier,
            cloud_handler=CloudHandler(mode="mock"),
        )

        self.assertIs(router.model, dummy_model)
        self.assertIs(router.slm, dummy_model)

        res = router.route("Turn on the lamp.")
        self.assertEqual(res["route"], "COMMAND")
        self.assertEqual(res["response"], "Simulated command response")

    def test_backward_compatibility_slm_module(self):
        """Verify slm_router.model.SLM preserves inheritance and defaults."""
        self.assertEqual(MODEL_NAME, "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertTrue(issubclass(SLM, TransformersRuntime))
        self.assertTrue(issubclass(SLM, BaseModel))


if __name__ == "__main__":
    unittest.main()
