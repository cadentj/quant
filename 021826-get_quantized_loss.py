# %%

from model import Mlp
from task import ParityTaskConfig, get_subsets, get_batch
import torch
import torch.nn as nn

DTYPE = torch.bfloat16

mlp = Mlp.from_pretrained("results/").to(DTYPE)

# %%

task_cfg = ParityTaskConfig()
Ss = get_subsets(task_cfg.n_tasks)
codes = eval(task_cfg.codes)

loss_fn = nn.CrossEntropyLoss()

print(f"Evaluating on {len(codes)} tasks with 1000 samples each:")
print()

task_losses = []
for i, code in enumerate(codes):
    x, y = get_batch(task_cfg.n_tasks, task_cfg.n, Ss, [code], [1000], dtype=DTYPE)
    with torch.no_grad():
        preds = mlp(x)
        loss = loss_fn(preds, y).item()
        acc = (preds.argmax(dim=-1) == y).float().mean().item()
    task_losses.append(loss)
    print(f"Task {i} (code={code}): loss={loss:.4f}, accuracy={acc:.4f}")

print()
print(f"Mean loss: {sum(task_losses) / len(task_losses):.4f}")