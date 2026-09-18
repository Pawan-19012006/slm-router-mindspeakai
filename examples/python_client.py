"""Minimal example client demonstrating HTTP API interaction with SLM Router."""

import json
import sys
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8000"


def make_request(method: str, path: str, payload: dict = None) -> dict:
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"} if payload else {}
    data = json.dumps(payload).encode("utf-8") if payload else None

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"Error connecting to SLM Router API at {url}: {e}")
        print("Please ensure the FastAPI server is running with:")
        print("  uv run uvicorn slm_router.api:app --host 127.0.0.1 --port 8000")
        sys.exit(1)


def main():
    print("1. Checking API Health...")
    health = make_request("GET", "/health")
    print(f"   Status: {health.get('status')} | Active Model: {health.get('model')}\n")

    query = "What is 2 + 2?"
    print(f"2. Classifying Query: {query!r}...")
    classification = make_request("POST", "/classify", {"query": query})
    print(f"   Assigned Label: {classification.get('label')}\n")

    print(f"3. Routing Query: {query!r}...")
    routed = make_request("POST", "/route", {"query": query})
    print(f"   Route:    {routed.get('route')}")
    print(f"   Handler:  {routed.get('handler')}")
    print(f"   Response: {routed.get('response')}")
    print(f"   Timings:  {routed.get('timings')}\n")


if __name__ == "__main__":
    main()
