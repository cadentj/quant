import json
from pathlib import Path

from datasets import load_dataset
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL_ID = "google/gemma-3-4b-it"

SAVE_DIR = MODEL_ID.split("/")[1] + "-W8A8"

# --- Strip "language_model." prefix from weight keys ---
# The multimodal Gemma 3 saves weights with a "language_model." prefix,
# but the text-only Gemma3ForCausalLM config expects keys without it.
PREFIX = "language_model."
save_path = Path(SAVE_DIR)

# Fix safetensors shards
for shard in sorted(save_path.glob("*.safetensors")):
    tensors = load_file(shard)
    renamed = {}
    needs_rename = False
    for key, tensor in tensors.items():
        if key.startswith(PREFIX):
            renamed[key[len(PREFIX):]] = tensor
            needs_rename = True
        else:
            renamed[key] = tensor
    if needs_rename:
        print(f"Stripping '{PREFIX}' prefix from {shard.name}")
        save_file(renamed, shard)

# Fix index file
index_path = save_path / "model.safetensors.index.json"
if index_path.exists():
    with open(index_path) as f:
        index = json.load(f)
    new_weight_map = {}
    for key, shard in index["weight_map"].items():
        new_key = key[len(PREFIX):] if key.startswith(PREFIX) else key
        new_weight_map[new_key] = shard
    index["weight_map"] = new_weight_map
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print("Updated model.safetensors.index.json")