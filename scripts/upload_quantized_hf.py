#!/usr/bin/env python3
"""Upload a local quantized checkpoint folder to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FOLDER = REPO_ROOT / "quantized" / "gemma4-e2b-it-gptq-baseline"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repo_id",
        help="HF model repo id, e.g. username/gemma4-e2b-it-gptq-baseline",
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=DEFAULT_FOLDER,
        help=f"Local folder to upload (default: {DEFAULT_FOLDER})",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create or use a private repo",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload Gemma 4 GPTQ W4A16 baseline",
        help="Git commit message for this upload",
    )
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.is_dir():
        print(f"error: not a directory: {folder}", file=sys.stderr)
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("error: install huggingface_hub (pip install huggingface_hub)", file=sys.stderr)
        sys.exit(1)

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
    )
    print(f"Uploaded `{folder}` to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
