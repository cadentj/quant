# Gemma 4 GPTQ W4A16 Quantization

## What was done

GPTQ W4A16 (group size 128) quantization of `google/gemma-4-E2B-it` using
`llm-compressor` on a single A100 80GB.

### Artifacts produced

- `quantized/gemma4-e2b-it-gptq-smoke/` — 4-sample smoke test
- `quantized/gemma4-e2b-it-gptq-baseline/` — 512-sample baseline

Both contain `model.safetensors` (~6.1G) and `quantization_manifest.json`.

## Environment setup

The main difficulty is that Gemma 4 requires bleeding-edge transformers (>=5.5)
while llm-compressor pins transformers<=4.57.6, and RunPod A100 images ship a
CUDA 12.4 driver that's incompatible with the default torch (CUDA 13). Everything
below works around that.

```bash
# On an A100 80GB RunPod instance (CUDA driver 12.4)
cd /root/quant

# Clone llm-compressor (has Gemma 4 modeling support on main)
git clone https://github.com/vllm-project/llm-compressor.git /tmp/llm-compressor

# Create isolated venv
uv venv .quant-venv --python 3.11

# Install llm-compressor (brings in transformers 4.57.6)
uv pip install --python .quant-venv/bin/python \
  -e /tmp/llm-compressor \
  datasets huggingface_hub accelerate sentencepiece

# Fix 1: Pin CUDA 12.4 compatible torch (default pulls CUDA 13)
uv pip install --python .quant-venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu124 \
  --force-reinstall torch==2.6.0

# Fix 2: Install transformers from source (4.57.6 has no gemma4 module)
uv pip install --python .quant-venv/bin/python \
  --upgrade --no-deps \
  git+https://github.com/huggingface/transformers.git

# Fix 3: Upgrade huggingface_hub from source (transformers dev needs is_offline_mode)
uv pip install --python .quant-venv/bin/python \
  --upgrade --no-deps \
  git+https://github.com/huggingface/huggingface_hub.git

# Fix 4: torchvision (AutoProcessor loads Gemma4VideoProcessor which requires it)
uv pip install --python .quant-venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu124 \
  torchvision==0.21.0

# Remove leftover CUDA 13 packages from initial torch install
uv pip uninstall --python .quant-venv/bin/python \
  cuda-bindings cuda-pathfinder cuda-toolkit \
  nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
  nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand \
  nvidia-cusolver nvidia-cusparse nvidia-cusparselt-cu13 \
  nvidia-nccl-cu13 nvidia-nvshmem-cu13 nvidia-nvtx

# Repair CUDNN after cleanup
uv pip install --python .quant-venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu124 \
  --force-reinstall nvidia-cudnn-cu12==9.1.0.70
```

### Validate

```bash
.quant-venv/bin/python - <<'PY'
import torch, transformers, llmcompressor
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('has gemma4', hasattr(transformers, 'Gemma4ForConditionalGeneration'))
from llmcompressor.modifiers.gptq import GPTQModifier
print('GPTQModifier ok')
PY
```

Expected: torch 2.6.0+cu124, cuda 12.4, transformers 5.6.0.dev0, gemma4 True.

## Gemma 4 specific workarounds in the script

llm-compressor's GPTQ pipeline doesn't work out of the box with
`Gemma4ForConditionalGeneration`. Three issues had to be patched in
`scripts/quantize_gemma4.py`:

1. **Text-only wrapper**: Load `Gemma4ForConditionalGeneration`, then extract the
   language model into a `Gemma4ForCausalLM` shell using
   `accelerate.init_empty_weights()` (avoids expensive random re-init). The
   sequential GPTQ pipeline can't trace the full multimodal model.

2. **Basic pipeline + no dispatch**: Force `pipeline="basic"` in the `oneshot()`
   call and monkey-patch `dispatch_model` to a no-op. The sequential pipeline
   hits a `KeyError: 13` on Gemma 4's shared KV layers (layers with
   `is_kv_shared_layer=True` reference earlier layer indices). The basic pipeline
   avoids the layer-splitting trace, and skipping dispatch keeps the model on GPU
   without accelerate's offload wrappers breaking the shared state.

3. **Pre-move to CUDA**: The wrapped model is explicitly `.to("cuda")` before
   `oneshot()` since the basic pipeline's dispatch is disabled.

## Running quantization

```bash
# Smoke test (~4 min)
.quant-venv/bin/python scripts/quantize_gemma4.py \
  --output-dir quantized/gemma4-e2b-it-gptq-smoke \
  --calibration-samples 4 \
  --max-seq-len 512 \
  --overwrite-output

# Full baseline (~14 min)
.quant-venv/bin/python scripts/quantize_gemma4.py \
  --output-dir quantized/gemma4-e2b-it-gptq-baseline \
  --calibration-samples 512 \
  --max-seq-len 2048 \
  --overwrite-output
```

### Defaults

| Parameter | Value |
|---|---|
| Model | `google/gemma-4-E2B-it` |
| Revision | `b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf` |
| Calibration dataset | `HuggingFaceH4/ultrachat_200k` (split `train_sft`) |
| GPTQ scheme | W4A16, group size 128 |
| Calibration mode | tokenizer text-only (chat template applied) |

## Runtime stack (validated)

| Package | Version |
|---|---|
| torch | 2.6.0+cu124 |
| transformers | 5.6.0.dev0 (source) |
| huggingface_hub | 1.11.0.dev0 (source) |
| llmcompressor | 0.10.1.dev88+g3c9d4fd7 (source) |
| torchvision | 0.21.0+cu124 |
| GPU | NVIDIA A100 80GB PCIe |
| CUDA driver | 12.4 |