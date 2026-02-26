# %%

import pickle as pkl

with open("test/results.pkl", "rb") as f:
    results = pkl.load(f)

# %%

results['subtask_losses'].keys()

import numpy as np
import matplotlib.pyplot as plt

# Clip outliers so one spike doesn't dominate the y-axis (e.g. 99th percentile)
clip_percentile = 99  # adjust if needed (100 = no clipping)
all_losses = np.concatenate(list(results['subtask_losses'].values()))
cap = np.nanpercentile(all_losses, clip_percentile)

for task_idx, task_losses in results['subtask_losses'].items():
    clipped = np.minimum(np.asarray(task_losses, dtype=float), cap)
    plt.plot(clipped, label=f"Task {task_idx}")
plt.xscale("log")
plt.xlim(350, len(results['steps']))
plt.legend()
plt.show()