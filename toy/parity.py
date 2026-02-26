#!/usr/bin/env python
# coding: utf-8
"""
This script trains MLPs on multiple sparse parity problems at once,
including composite tasks.
"""

from collections import defaultdict
from typing import Literal
import json
import pickle
import os

import chz
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from model import Mlp, MlpConfig
from task import ParityTaskConfig, get_subsets, get_batch, K
import wandb


@chz.chz
class CompositionJobConfig:
    # Model configuration
    model: MlpConfig

    # Task configuration
    task: ParityTaskConfig

    # Number of samples per task
    samples_per_task: int = 2000

    # Number of training steps
    steps: int = 200_000

    # Learning rate
    lr: float = 1e-3

    # Device to use
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Data type
    dtype: Literal["bfloat16", "float32", "float64"] = "float32"

    # Random seed
    seed: int = 0

    # Directory to save results
    save_dir: str

    # Verbose output
    verbose: bool = False

    # Wandb project name
    wandb_project: str


def run_parity(args: CompositionJobConfig):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float64":
        dtype = torch.float64

    device = torch.device(args.device)

    n_tasks = args.task.n_tasks
    assert args.task.n >= n_tasks * K
    Ss = get_subsets(n_tasks)

    try:
        codes = eval(args.task.codes)
        if not isinstance(codes, list) or not all(isinstance(code, list) for code in codes):
            raise ValueError("Codes must be a list of lists")
    except Exception as e:
        raise ValueError(f"Invalid codes format: {e}. Must be a valid Python list of lists.")

    train_sizes = [args.samples_per_task] * len(codes)

    input_dim = n_tasks + args.task.n
    mlp = Mlp(input_dim, args.model).to(dtype).to(device)

    print("Number of parameters:", mlp.ps())
    print("Codes:", codes)

    optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr, eps=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    steps = []
    samples = []
    losses = []
    subtask_losses = defaultdict(list)

    use_wandb = args.wandb_project is not None
    if use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    for step in tqdm(range(args.steps), disable=not args.verbose):
        with torch.no_grad():
            for i, code in enumerate(codes):
                x, y = get_batch(
                    n_tasks,
                    args.task.n,
                    Ss,
                    [code],
                    [train_sizes[i]],
                    dtype=dtype,
                    device=device,
                )
                y_pred = mlp(x)
                subtask_losses[i].append(loss_fn(y_pred, y).item())

        x, y = get_batch(
            n_tasks, args.task.n, Ss, codes, train_sizes, dtype=dtype, device=device
        )
        y_pred = mlp(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        steps.append(step)
        losses.append(loss.item())
        samples.append(x.shape[0])

        if use_wandb and step % 1000 == 0:
            wandb.log({"loss": loss.item()}, step=step)

    os.makedirs(args.save_dir, exist_ok=True)

    # TODO: this should probably be simpler to configure, maybe n is set post init?
    torch.save(
        {"state_dict": mlp.state_dict(), "input_dim": input_dim},
        os.path.join(args.save_dir, "model.pt"),
    )

    results = {
        "steps": steps,
        "losses": losses,
        "subtask_losses": subtask_losses,
        "Ss": Ss,
        "codes": codes,
        "n_parameters": mlp.ps(),
        "samples": samples,
    }

    with open(os.path.join(args.save_dir, "results.pkl"), "wb") as f:
        pickle.dump(results, f)

    config_to_save = {
        "model": chz.asdict(args.model),    
        "task": chz.asdict(args.task),
        "steps": args.steps,
        "lr": args.lr,
        "dtype": args.dtype,
        "seed": args.seed,
        "save_dir": args.save_dir,
        "wandb_project": args.wandb_project,
        "verbose": args.verbose,
        "samples_per_task": args.samples_per_task,
    }
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(config_to_save, f)


if __name__ == "__main__":
    args = chz.entrypoint(CompositionJobConfig)
    run_parity(args)
