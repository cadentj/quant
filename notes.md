# Gemma 4 GPTQ W4A16 Quantization

Use a dedicated Python 3.11 venv for quantization. The key constraints are:

- `llm-compressor` currently wants older `transformers`
- Gemma 4 needs current `transformers` from source
- the box should use the CUDA 12.4 PyTorch wheels, not the default CUDA 13 stack

## Environment setup

```bash
cd /root/quant

git clone https://github.com/vllm-project/llm-compressor.git /tmp/llm-compressor

uv venv .quant-venv --python 3.11

uv pip install --python .quant-venv/bin/python \
  -e /tmp/llm-compressor \
  datasets huggingface_hub accelerate sentencepiece

uv pip install --python .quant-venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu124 \
  --force-reinstall torch==2.6.0

uv pip install --python .quant-venv/bin/python \
  --upgrade --no-deps \
  git+https://github.com/huggingface/transformers.git

uv pip install --python .quant-venv/bin/python \
  --upgrade --no-deps \
  git+https://github.com/huggingface/huggingface_hub.git

uv pip install --python .quant-venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu124 \
  torchvision==0.21.0

uv pip uninstall --python .quant-venv/bin/python \
  cuda-bindings cuda-pathfinder cuda-toolkit \
  nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
  nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand \
  nvidia-cusolver nvidia-cusparse nvidia-cusparselt-cu13 \
  nvidia-nccl-cu13 nvidia-nvshmem-cu13 nvidia-nvtx

uv pip install --python .quant-venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu124 \
  --force-reinstall nvidia-cudnn-cu12==9.1.0.70
```

## Validate

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

Expected: `torch 2.6.0+cu124`, `cuda 12.4`, `transformers 5.6.0.dev0`, `gemma4 True`.

## Run quantization

```bash
cd /root/quant

.quant-venv/bin/python scripts/quantize_gemma4.py \
  --output-dir quantized/gemma4-e2b-it-gptq-baseline \
  --calibration-samples 512 \
  --max-seq-len 2048 \
  --overwrite-output
```
