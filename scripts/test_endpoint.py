#!/usr/bin/env python3
"""Quick smoke test against the Modal vLLM endpoint."""

import argparse
import json
import os
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://ndif--gemma4-e2b-it-gptq-baseline-serve.modal.run/v1"
DEFAULT_MODEL = "kh4dien/gemma4-e2b-it-gptq-baseline"

COLD_START_TIMEOUT = 600
POLL_INTERVAL = 10


def api_key() -> str:
    return os.environ.get("OPENAI_API_KEY", "EMPTY")


def request(url: str, payload: dict | None = None, timeout: int = 300) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key()}",
        "Content-Type": "application/json",
    }
    data = json.dumps(payload).encode() if payload else None
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def wait_for_server(base_url: str) -> list[str]:
    """Poll /v1/models until the server is up, returns list of model ids."""
    url = f"{base_url}/models"
    print(f"Waiting for server at {url} ...")
    start = time.time()
    attempt = 0
    while True:
        attempt += 1
        elapsed = time.time() - start
        if elapsed > COLD_START_TIMEOUT:
            raise SystemExit(f"Server not ready after {COLD_START_TIMEOUT}s — giving up.")
        try:
            result = request(url, timeout=POLL_INTERVAL + 5)
            models = [m["id"] for m in result.get("data", [])]
            print(f"  Server ready after {elapsed:.0f}s — models: {models}")
            return models
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            reason = type(exc).__name__
            print(f"  attempt {attempt} ({elapsed:.0f}s): {reason}, retrying in {POLL_INTERVAL}s ...")
            time.sleep(POLL_INTERVAL)


def chat(base_url: str, model: str, prompt: str, max_tokens: int) -> None:
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    print(f"POST {url}")
    print(f"  model={model}  max_tokens={max_tokens}")
    print(f"  prompt: {prompt!r}")
    print()

    result = request(url, payload)

    choice = result["choices"][0]
    text = choice["message"]["content"]
    usage = result.get("usage", {})

    print("--- response ---")
    print(text)
    print("--- usage ---")
    print(
        f"  input={usage.get('prompt_tokens')}  "
        f"output={usage.get('completion_tokens')}  "
        f"total={usage.get('total_tokens')}"
    )
    finish = choice.get("finish_reason", "?")
    print(f"  finish_reason={finish}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="What is 2+2? Answer in one word.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    models = wait_for_server(args.base_url)

    if args.list_models:
        return

    if args.model not in models:
        print(f"\n  WARNING: requested model {args.model!r} not in {models}")
        print(f"  (will try anyway)\n")

    print()
    chat(args.base_url, args.model, args.prompt, args.max_tokens)


if __name__ == "__main__":
    main()
