"""
Modal vLLM server for Gemma 4 — aligned with:
https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html

Thinking + tool calling require the `gemma4` reasoning/tool parsers and the
bundled `tool_chat_template_gemma4.jinja`. The official Docker image
`vllm/vllm-openai:gemma4` ships both; the nightly pip wheel does not.
"""
import modal

# The official Gemma 4 Docker image has the gemma4 reasoning/tool parsers
# and the jinja template built in. The nightly pip wheel does not.
vllm_image = (
    modal.Image.from_registry("vllm/vllm-openai:gemma4", add_python="3.12")
    .entrypoint([])
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

MODEL_NAME = "kh4dien/gemma4-e2b-it-gptq-baseline-v2"
MODEL_REVISION = "160666628b09a7c2723a69230d6026d501c512a0"

MAX_MODEL_LEN = 32768
GPU_MEMORY_UTILIZATION = 0.90

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

FAST_BOOT = False

app = modal.App("gemma4-e2b-it-gptq-baseline")

N_GPU = 1
MINUTES = 60
VLLM_PORT = 8000


@app.function(
    image=vllm_image,   
    gpu=f"A100-80GB:{N_GPU}",
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    max_containers=1,
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd: list[str] = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--uvicorn-log-level=info",
        "--async-scheduling",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--language-model-only",
        "--skip-mm-profiling",
        "--enable-auto-tool-choice",
        "--reasoning-parser",
        "gemma4",
        "--tool-call-parser",
        "gemma4",
        "--chat-template",
        "examples/tool_chat_template_gemma4.jinja",
        "--default-chat-template-kwargs",
        '{"enable_thinking": true}',
    ]

    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]
    cmd += ["--tensor-parallel-size", str(N_GPU)]

    print(*cmd)

    subprocess.Popen(cmd)
