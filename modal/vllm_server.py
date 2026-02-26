import modal

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.13.0",
        "huggingface-hub==0.36.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507-FP8"
MODEL_REVISION = "953532f942706930ec4bb870569932ef63038fdf"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

FAST_BOOT = False

app = modal.App("Qwen3-4B-Thinking-2507-FP8")

N_GPU = 1
MINUTES = 60
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
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

    cmd = [
        f"vllm serve {MODEL_NAME}",
        "--uvicorn-log-level=info",
        f"--revision {MODEL_REVISION}",
        f"--served-model-name {MODEL_NAME}",
        "--host 0.0.0.0",
        f"--port {VLLM_PORT}",
    ]

    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]
    cmd += ["--tensor-parallel-size", str(N_GPU)]
    cmd += ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"]
    cmd += ["--reasoning-parser", "qwen3"]

    print(*cmd)

    subprocess.Popen(" ".join(cmd), shell=True)
