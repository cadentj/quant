#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "quantized" / "gemma4-e2b-it-gptq-baseline"
DEFAULT_PROMPT = "Complete the sentence logically: The capital of France is"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a short generation against a local quantized checkpoint or a "
            "Hugging Face repo ID to sanity-check output quality."
        )
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help=(
            "Local quantized checkpoint directory or HF repo ID "
            f"(default: {DEFAULT_MODEL_PATH})"
        ),
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="User prompt to send to the model.",
    )
    parser.add_argument(
        "--system",
        default="You are a concise assistant. Answer directly.",
        help="Optional system instruction.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=12,
        help="Number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling cutoff when temperature > 0.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional HF revision when --model is a repo ID.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom modeling code when loading from HF.",
    )
    return parser.parse_args()


def require_transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency `transformers`. Install it first, for example: "
            "`uv pip install --upgrade 'transformers>=5.5'`"
        ) from exc
    return transformers


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency `torch`. Install it first, for example: "
            "`uv pip install torch`"
        ) from exc
    return torch


def load_manifest(model_path: Path) -> dict[str, Any]:
    manifest_path = model_path / "quantization_manifest.json"
    if not manifest_path.is_file():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def choose_dtype(torch: Any) -> Any:
    if torch.cuda.is_available():
        return "auto"
    if hasattr(torch, "float32"):
        return torch.float32
    return "auto"


def build_messages(system_prompt: str, user_prompt: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt.strip():
        messages.append(
            {"role": "system", "content": [{"type": "text", "text": system_prompt.strip()}]}
        )
    messages.append(
        {"role": "user", "content": [{"type": "text", "text": user_prompt.strip()}]}
    )
    return messages


def main() -> None:
    args = parse_args()
    transformers = require_transformers()
    torch = require_torch()

    model_arg = args.model
    model_path = Path(model_arg).expanduser()
    manifest = load_manifest(model_path) if model_path.exists() else {}

    model_type = manifest.get("model_type")
    loader_name = manifest.get("loader")

    processor = transformers.AutoProcessor.from_pretrained(
        model_arg,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )

    if loader_name == "Gemma4ForConditionalGeneration" or model_type == "gemma4":
        model_cls = getattr(transformers, "Gemma4ForConditionalGeneration", None)
        if model_cls is None:
            raise SystemExit(
                "This checkpoint expects `transformers.Gemma4ForConditionalGeneration`, "
                "but your installed transformers build does not provide it. "
                "Install a recent Gemma 4-capable version first."
            )
    else:
        model_cls = transformers.AutoModelForCausalLM

    model = model_cls.from_pretrained(
        model_arg,
        revision=args.revision,
        torch_dtype=choose_dtype(torch),
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    messages = build_messages(args.system, args.prompt)
    prompt_inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )

    prompt_len = int(prompt_inputs["input_ids"].shape[-1])
    model_device = next(model.parameters()).device
    prompt_inputs = {
        key: value.to(model_device) if hasattr(value, "to") else value
        for key, value in prompt_inputs.items()
    }

    do_sample = args.temperature > 0
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": processor.tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    with torch.inference_mode():
        output = model.generate(**prompt_inputs, **generation_kwargs)

    new_tokens = output[0, prompt_len:]
    text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print(f"model: {model_arg}")
    if args.revision:
        print(f"revision: {args.revision}")
    print(f"prompt: {args.prompt}")
    print(f"generated_text: {text or '<empty>'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
