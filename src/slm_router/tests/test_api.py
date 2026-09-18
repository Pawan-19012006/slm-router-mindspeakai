"""Unit tests for FastAPI transport layer."""

import json
import sys
import unittest
from unittest.mock import MagicMock
from pathlib import Path

# Ensure src is in sys.path
src_path = Path(__file__).resolve().parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from fastapi.testclient import TestClient
from slm_router.api import app, set_router


class TestAPI(unittest.TestCase):
    """Test FastAPI endpoints using mocked Router to prevent model loading."""

    def setUp(self):
        self.mock_router = MagicMock()
        self.mock_model = MagicMock()
        self.mock_model.model_name = "MockedModel/Mock-1B"
        self.mock_router.model = self.mock_model

        self.mock_classifier = MagicMock()
        self.mock_classifier.classify_with_raw.return_value = ("LOCAL", "LOCAL")
        self.mock_router.classifier = self.mock_classifier

        self.mock_router.route.return_value = {
            "query": "What is 2 + 2?",
            "route": "LOCAL",
            "handler": "Local SLM",
            "response": "4",
            "result": "4",
            "success": True,
            "mode": "LOCAL",
            "model": "MockedModel/Mock-1B",
            "processing_type": "local",
            "timings": {
                "classification": 0.05,
                "handler": 0.10,
                "total": 0.15,
            },
            "details": {
                "model": "MockedModel/Mock-1B",
                "classification_token": "LOCAL",
                "status": "COMPLETED_LOCALLY",
            },
        }

        # Inject mock router before TestClient runs lifespan
        set_router(self.mock_router)
        self.client = TestClient(app)

    def tearDown(self):
        set_router(None)

    def test_health_endpoint(self):
        """GET /health returns 200 with healthy status and dynamic model name."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["model"], "MockedModel/Mock-1B")
        # Ensure json serializable
        self.assertIsInstance(json.dumps(data), str)

    def test_classify_valid_query(self):
        """POST /classify accepts valid input and calls router.classifier."""
        response = self.client.post("/classify", json={"query": "What is 2 + 2?"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["query"], "What is 2 + 2?")
        self.assertEqual(data["label"], "LOCAL")
        self.assertEqual(data["raw_output"], "LOCAL")
        self.mock_classifier.classify_with_raw.assert_called_once_with("What is 2 + 2?")

    def test_classify_rejects_missing_query(self):
        """POST /classify returns 422 if query field is missing."""
        response = self.client.post("/classify", json={})
        self.assertEqual(response.status_code, 422)

    def test_classify_rejects_empty_query(self):
        """POST /classify returns 422 if query is empty string."""
        response = self.client.post("/classify", json={"query": ""})
        self.assertEqual(response.status_code, 422)

    def test_route_valid_query(self):
        """POST /route runs router.route and returns complete response structure."""
        response = self.client.post("/route", json={"query": "What is 2 + 2?"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["route"], "LOCAL")
        self.assertEqual(data["handler"], "Local SLM")
        self.assertTrue(data["success"])
        self.mock_router.route.assert_called_once_with("What is 2 + 2?")
        # Ensure json serializable
        self.assertIsInstance(json.dumps(data), str)

    def test_route_rejects_missing_query(self):
        """POST /route returns 422 when query is missing."""
        response = self.client.post("/route", json={"invalid_field": 123})
        self.assertEqual(response.status_code, 422)

    def test_route_rejects_empty_query(self):
        """POST /route returns 422 when query is empty."""
        response = self.client.post("/route", json={"query": ""})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
