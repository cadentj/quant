#!/usr/bin/env python
"""
Modal app that runs a W&B sweep over parity.py.

Usage
-----
    modal run sweep.py
    modal run sweep.py --count 100 --method random
    modal run sweep.py --sweep-id <EXISTING_ID>   # resume a sweep
"""

from dataclasses import dataclass, field
from typing import List

import modal

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------

app = modal.App("parity-sweep")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.10.0",
        "wandb>=0.24.1",
        "tqdm>=4.67.2",
        "chz>=0.4.0",
        "numpy>=2.4.2",
    )
    .add_local_file("parity.py", "/root/parity.py")
    .add_local_file("utils.py", "/root/utils.py")
)

vol = modal.Volume.from_name("composition-results", create_if_missing=True)

# ---------------------------------------------------------------------------
# Sweep configuration (dataclass)
# ---------------------------------------------------------------------------


@dataclass
class SweepConfig:
    """All knobs for the sweep – fixed params and search space."""

    # ---- W&B ----
    wandb_project: str = "math-capstone"
    sweep_name: str = "parity-sweep"
    method: str = "bayes"  # grid | random | bayes
    count: int = 50

    # ---- Fixed training params (not swept) ----
    n: int = 64
    steps: int = 200_000
    device: str = "cuda"
    codes: str = "[[0], [1], [2], [3], [0, 1, 2, 3]]"
    seed: int = 0
    save_dir: str = "/vol/sweep"

    # ---- Search space ----
    width_values: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    depth_values: List[int] = field(default_factory=lambda: [2, 3, 4])
    lr_min: float = 1e-4
    lr_max: float = 1e-2
    activation_values: List[str] = field(default_factory=lambda: ["ReLU", "Tanh"])
    layernorm_values: List[bool] = field(default_factory=lambda: [True, False])
    samples_per_task_values: List[int] = field(
        default_factory=lambda: [1000, 2000, 4000]
    )
    dtype_values: List[str] = field(default_factory=lambda: ["bfloat16", "float32"])

    # -----------------------------------------------------------------

    def to_wandb_sweep_config(self) -> dict:
        """Build the dict that ``wandb.sweep()`` expects."""
        return {
            "name": self.sweep_name,
            "method": self.method,
            "metric": {"name": "loss", "goal": "minimize"},
            "parameters": {
                "width": {"values": self.width_values},
                "depth": {"values": self.depth_values},
                "lr": {
                    "min": self.lr_min,
                    "max": self.lr_max,
                    "distribution": "log_uniform_values",
                },
                "activation": {"values": self.activation_values},
                "layernorm": {"values": self.layernorm_values},
                "samples_per_task": {"values": self.samples_per_task_values},
                "dtype": {"values": self.dtype_values},
            },
        }

    def fixed_params(self) -> dict:
        """Params shared by every sweep run."""
        return {
            "n": self.n,
            "steps": self.steps,
            "device": self.device,
            "codes": self.codes,
            "seed": self.seed,
            "save_dir": self.save_dir,
            "verbose": False,
        }


# ---------------------------------------------------------------------------
# Remote function – one sweep trial per Modal container
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="T4",
    timeout=7200,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def run_sweep_agent(sweep_id: str, project: str, fixed: dict):
    """Execute a single W&B sweep trial."""
    import sys

    sys.path.insert(0, "/root")

    import wandb
    from parity import CompositionJobConfig, run_parity

    def _train():
        run = wandb.init(project=project)
        sweep_params = dict(wandb.config)

        # Merge fixed + swept params
        params = {**fixed, **sweep_params}
        params["save_dir"] = f"{fixed['save_dir']}/{run.id}"
        params["wandb_project"] = project

        config = CompositionJobConfig(**params)

        # Prevent run_parity's internal wandb.init from replacing
        # the sweep run that the agent just created.
        _real_init = wandb.init
        wandb.init = lambda *a, **kw: run
        try:
            run_parity(config)
        finally:
            wandb.init = _real_init

    wandb.agent(sweep_id, function=_train, count=1, project=project)


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    count: int = 50,
    method: str = "bayes",
    project: str = "math-capstone",
    sweep_id: str = "",
):
    """Create (or resume) a W&B sweep and fan out Modal containers."""
    import wandb

    cfg = SweepConfig(count=count, method=method, wandb_project=project)

    if not sweep_id:
        sweep_id = wandb.sweep(cfg.to_wandb_sweep_config(), project=project)
        print(f"Created sweep: {sweep_id}")
    else:
        print(f"Resuming sweep: {sweep_id}")

    fixed = cfg.fixed_params()

    # Spawn `count` parallel containers, each running one trial.
    run_sweep_agent.spawn_map(
        [sweep_id] * cfg.count,
        [project] * cfg.count,
        [fixed] * cfg.count,
    )
