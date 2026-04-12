#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_MODEL_REVISION = "b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf"
DEFAULT_CALIBRATION_DATASET = "HuggingFaceH4/ultrachat_200k"
DEFAULT_CALIBRATION_SPLIT = "train_sft"
DEFAULT_RECIPE_NAME = "gptq-w4a16"
DEFAULT_GROUP_SIZE = 128
DEFAULT_CALIBRATION_SAMPLES = 512
DEFAULT_MAX_SEQ_LEN = 2048
DEFAULT_SEED = 42
DEFAULT_PRECISION = "auto"
MIN_TRANSFORMERS_FOR_GEMMA4 = (5, 5)
DEFAULT_GEMMA4_IGNORE_PATTERNS = (
    "lm_head",
    "re:.*embed.*",
    "re:.*router",
    "re:.*vision_tower.*",
    "re:.*audio.*",
    "re:.*multi_modal_projector.*",
    "re:.*projector.*",
)
DEFAULT_TEXT_IGNORE_PATTERNS = ("lm_head",)


@dataclass
class RuntimeContext:
    model: Any
    processor: Any | None
    tokenizer: Any | None
    model_type: str | None
    loader: str
    calibration_mode: str
    pipeline: str | None
    ignore_patterns: list[str]


@dataclass
class Manifest:
    base_model_id: str
    base_revision: str
    model_type: str | None
    loader: str
    precision: str
    recipe_name: str
    quantization_scheme: str
    target_modules: list[str]
    weight_bits: int
    weight_strategy: str
    weight_symmetric: bool
    calibration_mode: str
    calibration_dataset: str
    calibration_revision: str | None
    calibration_split: str
    calibration_samples: int
    max_seq_len: int
    seed: int
    group_size: int
    ignore_patterns: list[str]
    upload_repo: str | None
    created_at_utc: str
    package_versions: dict[str, str | None]
    dataset_preparation: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantize google/gemma-4-E2B-it offline with llm-compressor using a "
            "stock GPTQ W4A16 recipe and emit a reproducibility manifest."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--precision", default=DEFAULT_PRECISION)
    parser.add_argument("--calibration-dataset", default=DEFAULT_CALIBRATION_DATASET)
    parser.add_argument("--calibration-revision", default=None)
    parser.add_argument("--calibration-split", default=DEFAULT_CALIBRATION_SPLIT)
    parser.add_argument("--calibration-samples", default=DEFAULT_CALIBRATION_SAMPLES, type=int)
    parser.add_argument("--max-seq-len", default=DEFAULT_MAX_SEQ_LEN, type=int)
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int)
    parser.add_argument("--group-size", default=DEFAULT_GROUP_SIZE, type=int)
    parser.add_argument("--upload-repo", default=None)
    parser.add_argument("--upload-private", action="store_true")
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=None,
        help=(
            "Module pattern to exclude from quantization. Can be repeated. "
            "Gemma 4 defaults exclude embeddings, routers, and multimodal towers."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom modeling code when loading the base model or processor.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete an existing output directory before writing the quantized artifact.",
    )
    return parser.parse_args()


def require_dependency(module_name: str, install_hint: str) -> Any:
    try:
        return __import__(module_name, fromlist=["_"])
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency `{module_name}`. Install it on the quantization "
            f"machine first, for example: `{install_hint}`"
        ) from exc


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def parse_version_prefix(raw: str | None) -> tuple[int, ...]:
    if not raw:
        return ()
    prefix: list[int] = []
    for segment in raw.split("."):
        digits = ""
        for char in segment:
            if char.isdigit():
                digits += char
            else:
                break
        if not digits:
            break
        prefix.append(int(digits))
    return tuple(prefix)


def require_transformers_support_for_gemma4() -> None:
    installed = parse_version_prefix(package_version("transformers"))
    if installed and installed < MIN_TRANSFORMERS_FOR_GEMMA4:
        wanted = ".".join(str(piece) for piece in MIN_TRANSFORMERS_FOR_GEMMA4)
        current = package_version("transformers")
        raise SystemExit(
            "Gemma 4 quantization in llm-compressor requires "
            f"`transformers>={wanted}`. Found `{current}`. "
            "Install from source or upgrade first, for example: "
            "`uv pip install --upgrade 'transformers>=5.5'`"
        )


def normalize_role(role: str | None) -> str:
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant"}:
        return "assistant"
    if role == "system":
        return "system"
    return "user"


def flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        flattened: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                flattened.append(str(item.get("text", "")))
            else:
                flattened.append(str(item))
        return "\n".join(part for part in flattened if part)
    if isinstance(content, dict) and content.get("type") == "text":
        return str(content.get("text", ""))
    return str(content or "")


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = normalize_role(message.get("role") or message.get("from"))
        content = flatten_content(message.get("content") or message.get("value") or "")
        normalized.append({"role": role, "content": content})
    return normalized


def wrap_messages_for_gemma4(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "role": message["role"],
            "content": [{"type": "text", "text": message["content"]}],
        }
        for message in messages
    ]


def prompt_to_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def dataset_messages(row: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    if "messages" in row:
        return normalize_messages(row["messages"]), "messages"
    if "conversations" in row:
        return normalize_messages(row["conversations"]), "conversations"
    if "text" in row:
        return prompt_to_messages(str(row["text"])), "text"
    if "prompt" in row:
        return prompt_to_messages(str(row["prompt"])), "prompt"
    raise SystemExit(
        "Unsupported calibration dataset schema. Expected one of the columns "
        "`text`, `messages`, `conversations`, or `prompt`."
    )


def gemma4_preprocess(row: dict[str, Any], processor: Any, max_seq_len: int) -> dict[str, Any]:
    messages, _ = dataset_messages(row)
    rendered = processor.apply_chat_template(
        wrap_messages_for_gemma4(messages),
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
        add_generation_prompt=False,
        processor_kwargs={
            "padding": False,
            "truncation": True,
            "max_length": max_seq_len,
            "add_special_tokens": False,
        },
    )
    return {key: value.tolist() for key, value in rendered.items()}


def text_preprocess(row: dict[str, Any], tokenizer: Any, max_seq_len: int) -> dict[str, Any]:
    messages, _ = dataset_messages(row)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    tokenized = tokenizer(
        text,
        padding=False,
        max_length=max_seq_len,
        truncation=True,
        add_special_tokens=False,
    )
    return dict(tokenized)


def single_sample_collator(torch_module: Any):
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        assert len(batch) == 1
        collated: dict[str, Any] = {}
        for key, value in batch[0].items():
            tensor = torch_module.tensor(value)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            collated[key] = tensor
        return collated

    return collate


def prepare_calibration_dataset(
    args: argparse.Namespace,
    runtime: RuntimeContext,
) -> tuple[Any, dict[str, Any], Any]:
    datasets = require_dependency("datasets", "pip install datasets")
    torch = require_dependency("torch", "pip install torch")

    split_expr = args.calibration_split
    if "[" not in split_expr:
        split_expr = f"{split_expr}[:{args.calibration_samples}]"

    dataset = datasets.load_dataset(
        args.calibration_dataset,
        revision=args.calibration_revision,
        split=split_expr,
    )
    original_columns = list(dataset.column_names)
    dataset = dataset.shuffle(seed=args.seed)
    if len(dataset) > args.calibration_samples:
        dataset = dataset.select(range(args.calibration_samples))

    first_row = dataset[0]
    _, detected_format = dataset_messages(first_row)

    if runtime.calibration_mode == "gemma4_processor":
        processor = runtime.processor
        prepared = dataset.map(
            lambda row: gemma4_preprocess(row, processor, args.max_seq_len),
            remove_columns=original_columns,
            desc="Tokenizing Gemma 4 calibration samples",
        )
    else:
        tokenizer = runtime.tokenizer
        prepared = dataset.map(
            lambda row: text_preprocess(row, tokenizer, args.max_seq_len),
            remove_columns=original_columns,
            desc="Tokenizing calibration samples",
        )

    prepared = prepared.filter(
        lambda example: len(example["input_ids"]) >= args.max_seq_len,
        desc="Filtering for full-length calibration samples",
    )

    info = {
        "input_format": detected_format,
        "pretokenized": True,
        "num_rows_loaded": len(prepared),
    }
    return prepared, info, single_sample_collator(torch)


def build_recipe(args: argparse.Namespace, ignore_patterns: list[str]) -> Any:
    try:
        gptq_module = __import__("llmcompressor.modifiers.gptq", fromlist=["GPTQModifier"])
        GPTQModifier = getattr(gptq_module, "GPTQModifier")
    except (ImportError, AttributeError) as exc:
        raise SystemExit(
            "Could not import `GPTQModifier` from llmcompressor. "
            "Install a recent `llmcompressor` build, preferably from source."
        ) from exc

    if args.group_size == DEFAULT_GROUP_SIZE:
        return GPTQModifier(
            targets="Linear",
            scheme="W4A16",
            ignore=ignore_patterns,
        )

    return GPTQModifier(
        ignore=ignore_patterns,
        config_groups={
            "group_0": {
                "targets": ["Linear"],
                "input_activations": None,
                "output_activations": None,
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": args.group_size,
                },
            }
        },
    )


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise SystemExit(
                f"Output directory `{path}` already exists and is not empty. "
                "Pass `--overwrite-output` to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def is_gemma4_model(model_id: str, model_type: str | None) -> bool:
    return model_type == "gemma4" or "gemma-4" in model_id.lower()


def load_runtime(args: argparse.Namespace) -> RuntimeContext:
    transformers = require_dependency(
        "transformers",
        "uv pip install --upgrade 'transformers>=5.5'",
    )
    torch = require_dependency("torch", "pip install torch")
    require_dependency(
        "llmcompressor",
        "uv pip install -e /path/to/llm-compressor",
    )

    config = transformers.AutoConfig.from_pretrained(
        args.model_id,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    model_type = getattr(config, "model_type", None)

    if is_gemma4_model(args.model_id, model_type):
        require_transformers_support_for_gemma4()
        if not hasattr(transformers, "Gemma4ForConditionalGeneration"):
            raise SystemExit(
                "Installed transformers does not expose `Gemma4ForConditionalGeneration`. "
                "Upgrade to a Gemma 4-capable build, for example: "
                "`uv pip install --upgrade 'transformers>=5.5'`"
            )

        model = transformers.Gemma4ForConditionalGeneration.from_pretrained(
            args.model_id,
            revision=args.revision,
            dtype=args.precision,
            trust_remote_code=args.trust_remote_code,
        )
        processor = transformers.AutoProcessor.from_pretrained(
            args.model_id,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
        return RuntimeContext(
            model=model,
            processor=processor,
            tokenizer=processor.tokenizer,
            model_type=model_type,
            loader="Gemma4ForConditionalGeneration",
            calibration_mode="gemma4_processor",
            pipeline="basic",
            ignore_patterns=list(args.ignore_pattern or DEFAULT_GEMMA4_IGNORE_PATTERNS),
        )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        dtype=args.precision,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    return RuntimeContext(
        model=model,
        processor=None,
        tokenizer=tokenizer,
        model_type=model_type,
        loader="AutoModelForCausalLM",
        calibration_mode="tokenizer_text",
        pipeline=None,
        ignore_patterns=list(args.ignore_pattern or DEFAULT_TEXT_IGNORE_PATTERNS),
    )


def maybe_upload(output_dir: Path, repo_id: str, private: bool) -> None:
    huggingface_hub = require_dependency(
        "huggingface_hub",
        "pip install huggingface_hub",
    )
    api = huggingface_hub.HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload Gemma 4 GPTQ W4A16 baseline",
    )


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir, args.overwrite_output)

    runtime = load_runtime(args)
    if runtime.pipeline == "basic":
        basic_pipeline = __import__("llmcompressor.pipelines.basic.pipeline", fromlist=["dispatch_model"])
        setattr(basic_pipeline, "dispatch_model", lambda model: model)
    calibration_dataset, dataset_info, data_collator = prepare_calibration_dataset(args, runtime)
    recipe = build_recipe(args, runtime.ignore_patterns)

    oneshot_module = __import__("llmcompressor", fromlist=["oneshot"])
    oneshot = getattr(oneshot_module, "oneshot")

    manifest = Manifest(
        base_model_id=args.model_id,
        base_revision=args.revision,
        model_type=runtime.model_type,
        loader=runtime.loader,
        precision=args.precision,
        recipe_name=DEFAULT_RECIPE_NAME,
        quantization_scheme="W4A16",
        target_modules=["Linear"],
        weight_bits=4,
        weight_strategy="group",
        weight_symmetric=True,
        calibration_mode=runtime.calibration_mode,
        calibration_dataset=args.calibration_dataset,
        calibration_revision=args.calibration_revision,
        calibration_split=args.calibration_split,
        calibration_samples=args.calibration_samples,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        group_size=args.group_size,
        ignore_patterns=runtime.ignore_patterns,
        upload_repo=args.upload_repo,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        package_versions={
            "llmcompressor": package_version("llmcompressor"),
            "transformers": package_version("transformers"),
            "datasets": package_version("datasets"),
            "huggingface_hub": package_version("huggingface_hub"),
            "torch": package_version("torch"),
        },
        dataset_preparation=dataset_info,
    )

    print(
        "Prepared Gemma quantization run:",
        json.dumps(
            {
                "model_id": args.model_id,
                "revision": args.revision,
                "loader": runtime.loader,
                "model_type": runtime.model_type,
                "precision": args.precision,
                "output_dir": str(args.output_dir),
                "calibration_dataset": args.calibration_dataset,
                "calibration_split": args.calibration_split,
                "calibration_samples": args.calibration_samples,
                "max_seq_len": args.max_seq_len,
                "group_size": args.group_size,
                "ignore_patterns": runtime.ignore_patterns,
            },
            indent=2,
            sort_keys=True,
        ),
        sep="\n",
    )

    oneshot_kwargs: dict[str, Any] = {
        "model": runtime.model,
        "recipe": recipe,
        "dataset": calibration_dataset,
        "data_collator": data_collator,
        "num_calibration_samples": args.calibration_samples,
        "shuffle_calibration_samples": False,
        "max_seq_length": args.max_seq_len,
        "output_dir": str(args.output_dir),
        "save_compressed": True,
    }
    if runtime.processor is not None:
        oneshot_kwargs["processor"] = runtime.processor
    elif runtime.tokenizer is not None:
        oneshot_kwargs["tokenizer"] = runtime.tokenizer
    if runtime.pipeline is not None:
        oneshot_kwargs["pipeline"] = runtime.pipeline

    oneshot(**oneshot_kwargs)

    if runtime.processor is not None:
        runtime.processor.save_pretrained(str(args.output_dir))
    elif runtime.tokenizer is not None:
        runtime.tokenizer.save_pretrained(str(args.output_dir))

    write_json(args.output_dir / "quantization_manifest.json", asdict(manifest))

    if args.upload_repo:
        maybe_upload(args.output_dir, args.upload_repo, args.upload_private)
        print(f"Uploaded quantized checkpoint to `{args.upload_repo}`")

    print(f"Wrote quantized artifact to `{args.output_dir}`")


if __name__ == "__main__":
    main()
