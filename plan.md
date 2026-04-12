# Gemma 4 E2B INT4 Baseline Plan

## Summary

This project needs a clean baseline for `google/gemma-4-E2B-it` using stock GPTQ-style `W4A16` quantization before testing a modified GPTQ objective that incorporates interpretability. The baseline should be easy to reproduce, easy to compare against later variants, and compatible with the current repo workflow, which already serves Gemma 4 through Modal + vLLM and evaluates it through `inspect eval`.

The recommended workflow is:

1. Quantize Gemma 4 offline with a dedicated script.
2. Save the quantized checkpoint locally and optionally upload it to Hugging Face.
3. Serve the uploaded checkpoint through the existing vLLM Modal path.
4. Run the same eval slice against both the original model and the quantized model.
5. Use that result as the control when comparing against future interpretability-aware quantization runs.

This avoids mixing quantization logic into the serving path and keeps the baseline artifact portable. The GPU machine doing the quantization and the machine doing inference do not need to be the same machine once the artifact is uploaded.

## Why These Decisions

### Use stock `W4A16` GPTQ first

The baseline should answer a narrow question: how does standard GPTQ behave on Gemma 4 E2B under the same evaluation setup? If the first baseline already includes custom loss terms, calibration tricks, or nonstandard module selection, it becomes hard to attribute any later difference to the interpretability-aware objective.

`W4A16` is the right baseline because:

- it matches vLLM's INT4 serving path directly
- it is standard enough to compare against upstream docs and community expectations
- it materially reduces memory while usually preserving enough quality to be a realistic deployment baseline

### Keep quantization offline, not inside the server

Quantization is an artifact-production step, not an inference-time concern. The server should only load a model that already exists. Mixing quantization into the Modal server would make deployment slower, harder to debug, and much less reproducible.

An offline script is better because it:

- gives you a stable baseline artifact you can version and upload
- lets you run quantization on a separate GPU machine
- makes it easier to compare multiple quantized variants later
- avoids accidental differences between deployments caused by recomputing quantization

### Reuse the existing vLLM server, but parameterize it

You do not want one new server script per quantized checkpoint. That does not scale once you have multiple baseline and modified variants uploaded to Hugging Face.

The right split is:

- one quantization script that produces checkpoints
- one configurable serving script that can point at any chosen checkpoint

This matches your instinct about using CLI or env configuration. The practical workflow is usually:

1. quantize locally or on a GPU box
2. upload checkpoint to HF
3. deploy the same server code with a different model repo / revision

That means the server should accept model-selection inputs rather than hard-coding `google/gemma-4-E2B-it`.

### Stay text-only for the first baseline

Your current `modal/vllm_server.py` already serves Gemma 4 with `--language-model-only` and `--skip-mm-profiling`. That is important context: even though Gemma 4 is newer and multimodal, your current repo flow is already narrowed to text inference.

That makes text-only the right first baseline because:

- it matches the current evaluation path
- it removes multimodal calibration complexity from the first experiment
- it reduces the chance that image or audio components create quantization or loading issues unrelated to the research question

If multimodal support matters later, it can be added as a separate branch of work. For the baseline, preserving compatibility with the current text-serving path is more important than covering every Gemma 4 capability immediately.

### Quantize the language-model linear layers only

Gemma 4 is newer than the older generic INT4 docs, so the safe baseline is not "quantize absolutely everything." The baseline should target the transformer language-model path that vLLM actually uses in the current server configuration.

That implies:

- quantize the standard LM linear layers with stock GPTQ
- leave pieces like `lm_head` and multimodal towers unquantized unless the implementation path is explicitly known to be stable

This is a conservatism decision. It keeps the first baseline focused on the part of the model actually exercised by the current eval loop.

### Use a generic chat calibration corpus

The baseline should not be optimized for one narrow benchmark family. If calibration is too task-specific, later quality differences become harder to interpret.

A generic chat corpus such as `HuggingFaceH4/ultrachat_200k` is the right default because it:

- matches the instruction-tuned nature of `gemma-4-E2B-it`
- avoids leaking benchmark-specific structure into the baseline
- gives a reproducible calibration source for all future variants

Start with `512` samples and a fixed seed. If that baseline is unstable, move to `1024` samples before changing anything more complicated.

### Keep A100 as the target hardware assumption

The current server already targets `A100-80GB`, and that is a reasonable baseline deployment target for vLLM INT4 serving. vLLM's INT4 path is supported beyond pre-Hopper, including Hopper, but the important project constraint is not "what is theoretically supported" but "what is the cleanest target for a reproducible baseline in this repo."

For this repo, that target is still `1x A100-80GB` because:

- it matches the current Modal config exactly
- it avoids changing too many variables at once
- it is a known-good inference target for vLLM mixed-precision INT4 serving

If later you want hardware-comparison results between A100 and Hopper, that should be a separate experiment rather than something folded into the baseline plan.

## Implementation Plan

### 1. Add a dedicated quantization script

Create a script such as `scripts/quantize_gemma4.py` that performs offline quantization and writes a reusable artifact.

The script should:

- default to `google/gemma-4-E2B-it`
- default to the same model revision pinned in `modal/vllm_server.py`
- accept an output directory
- optionally push the result to a Hugging Face repo
- expose stable knobs for calibration size, seed, and sequence length
- keep GPTQ recipe defaults close to upstream stock behavior

Suggested CLI surface:

```bash
python scripts/quantize_gemma4.py \
  --model-id google/gemma-4-E2B-it \
  --revision <base_revision> \
  --output-dir /path/to/output \
  --calibration-dataset HuggingFaceH4/ultrachat_200k \
  --calibration-samples 512 \
  --max-seq-len 2048 \
  --seed 42 \
  --group-size 128 \
  --upload-repo <optional_hf_repo>
```

The script should also write a manifest file next to the quantized model containing:

- base model id
- base revision
- recipe name
- calibration dataset
- calibration sample count
- sequence length
- seed
- GPTQ-specific knobs
- date and package versions if easy to capture

That manifest will matter later when you compare custom objective variants against the baseline.

### 2. Use a stock GPTQ `W4A16` recipe

The recipe should be intentionally boring for the baseline.

Defaults:

- weights: 4-bit
- activations: 16-bit compute
- group size: `128`
- calibration samples: `512`
- max seq len: `2048`
- seed: `42`

The exact library API may differ depending on the llm-compressor version installed on the GPU machine, but the implementation goal is stable:

- use upstream stock GPTQ machinery
- target only the LM linear layers used for text generation
- avoid introducing custom objective logic in the baseline branch

If the GPU box needs package setup, that should happen there rather than by changing this repo's existing serving environment.

### 3. Parameterize `modal/vllm_server.py`

The current server file hard-codes:

- model name
- revision
- app name
- context-related settings

Change it so that the same file can serve:

- the original full-precision model
- the baseline quantized model
- future custom quantized variants

The minimal server configuration inputs should be:

- `MODEL_NAME`
- `MODEL_REVISION`
- `SERVED_MODEL_NAME`
- `MAX_MODEL_LEN`
- `GPU_MEMORY_UTILIZATION`

Those can come from env vars with sensible defaults matching the current file.

Gemma-4-specific serving flags should remain in place for now:

- `--reasoning-parser gemma4`
- `--tool-call-parser gemma4`
- `--chat-template examples/tool_chat_template_gemma4.jinja`
- `--language-model-only`
- `--skip-mm-profiling`

The point is to vary the model artifact, not the serving behavior.

### 4. Make `eval.sh` reusable across variants

Right now `eval.sh` points at one fixed Modal URL and one fixed model id. That is fine for one deployment, but not for comparing baseline and future variants.

Update it so the caller can override:

- `VLLM_BASE_URL`
- model name used by `inspect eval`
- possibly the eval task slice if needed later

The default behavior can stay identical to the current file. The important change is to remove the need to edit the script for every quantized repo you want to test.

### 5. Adopt a stable artifact workflow

The expected workflow for the handoff machine should be:

1. pull base model
2. run offline quantization
3. validate the quantized checkpoint locally if possible
4. upload checkpoint to HF
5. point `modal/vllm_server.py` at that uploaded repo
6. deploy and run evals

This is preferable to trying to serve directly from a partially generated local artifact because:

- Hugging Face gives you versioned, shareable artifacts
- Modal can load the same repo later without extra packaging work
- the baseline becomes easy to reproduce on another machine

## Public Interfaces To Hand Off

### New file

- `scripts/quantize_gemma4.py`

### Updated files

- `modal/vllm_server.py`
- `eval.sh`

### Environment / CLI contract

The implementation should support at least these variables or equivalents:

```bash
MODEL_NAME=<hf_repo_or_base_model>
MODEL_REVISION=<optional_revision>
SERVED_MODEL_NAME=<name_exposed_by_vllm>
MAX_MODEL_LEN=32768
GPU_MEMORY_UTILIZATION=0.90
VLLM_BASE_URL=<modal_or_local_vllm_url>
EVAL_MODEL_NAME=<model_name_passed_to_inspect>
```

## Validation And Test Cases

### Quantization smoke test

- Run the new quantization script on a small calibration subset.
- Confirm the output directory contains a loadable compressed checkpoint.
- Confirm the manifest was written and matches the chosen parameters.

### Local or staging load test

- Start vLLM against the quantized checkpoint.
- Send a short text-only generation request.
- Verify that Gemma 4 loads correctly and does not fail during startup due to parser or template mismatches.

### Baseline comparison run

- Evaluate the original `google/gemma-4-E2B-it`.
- Evaluate the baseline quantized checkpoint under the same eval slice.
- Record:
  - task score or pass rate
  - latency
  - memory footprint
  - any notable decoding regressions

### Stability fallback

If the baseline looks abnormally bad, do not immediately change the recipe. First:

1. rerun with the same settings to rule out setup mistakes
2. increase calibration samples from `512` to `1024`
3. only then investigate recipe-level changes

This protects the experiment from confusing setup noise with algorithmic differences.

## Assumptions

- The first target is a baseline control, not the final research recipe.
- The serving path should stay aligned with the repo's current text-only Gemma 4 setup.
- The quantized artifact will likely live on Hugging Face so it can be served and compared repeatedly.
- The GPU machine running quantization can install whatever packages are needed for llm-compressor and transformers support, even if this repo itself does not yet declare them.

## Open Risks To Watch

- Gemma 4 is newer than many generic INT4 examples, so exact quantization support may depend on current llm-compressor and transformers versions on the GPU box.
- Multimodal components may require special handling even if they are unused during text-only serving.
- If a quantized checkpoint loads in transformers but not in vLLM, the issue is likely artifact compatibility or kernel-path support rather than the high-level experimental design.

## Handoff Note

The most important instruction for the implementation model is to keep the baseline narrow and reproducible. Do not mix the interpretability-aware objective into this first artifact. Produce one stock `W4A16` Gemma 4 E2B checkpoint, serve it through the existing configurable vLLM path, and measure it under the same eval slice as the unquantized model.
