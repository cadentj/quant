# %%
import os
from pathlib import Path
from inspect_ai.log import read_eval_log
from docent import Docent
from docent.loaders.load_inspect import load_inspect_log
from docent.data_models.agent_run import AgentRunView

# %%
DOCENT_API_KEY = "dk_SilWb4NLFwzZRyHp_8UwyptaxYuF9PHVIjdc8Fwaq8tUCxJYIN0OEvJlPKsHUgJ    "
COLLECTION_NAME = "HumanEval Quantization"

EVAL_FILES = [
    "logs/2026-02-09T22-33-58+00-00_humaneval_9HZ4NqRui7jP9VYzeGnQLs.eval",
    "logs/2026-02-09T22-42-30+00-00_humaneval_WCfsuUNC7CGnQzgfdLQwbT.eval",
]

# %%
# Load and convert all eval files
agent_runs = []
for eval_path in EVAL_FILES:
    print(f"Loading {eval_path}...")
    eval_log = read_eval_log(eval_path)
    runs = load_inspect_log(eval_log)
    print(f"  -> {len(runs)} agent runs")
    agent_runs.extend(runs)

print(f"\nTotal agent runs: {len(agent_runs)}")

# %%
# Dump only run with id humaneval/37 to .txt file
OUTPUT_TXT = "logs/humaneval_runs.txt"
TARGET_RUN_ID = "HumanEval/37"
runs_to_dump = [r for r in agent_runs if r.metadata['sample_id'] == TARGET_RUN_ID or r.metadata.get("task_id") == TARGET_RUN_ID]
with open(OUTPUT_TXT, "w") as f:
    for run in runs_to_dump:
        f.write(AgentRunView.from_agent_run(run).to_text())
        f.write("\n\n")
print(f"Dumped {len(runs_to_dump)} run(s) (id={TARGET_RUN_ID}) to {OUTPUT_TXT}")

# %%
# Validate a sample before uploading
for i, run in enumerate(agent_runs[:3]):
    try:
        _ = AgentRunView.from_agent_run(run).to_text()
        print(f"✓ Run {i} valid")
    except Exception as e:
        print(f"✗ Run {i} invalid: {e}")

# %%
# Upload to Docent
client = Docent(api_key=DOCENT_API_KEY)

collection_id = client.create_collection(
    name=COLLECTION_NAME,
    description="HumanEval benchmark results from quantized model evaluation",
)
print(f"Created collection: {collection_id}")

client.add_agent_runs(collection_id, agent_runs)
print(f"Uploaded {len(agent_runs)} runs")
print(f"View at: https://docent.transluce.org/collection/{collection_id}")

# %%
# Verify
try:
    collection_info = client.get_collection(collection_id)
    uploaded_count = collection_info.get("agent_run_count", "unknown")
    print(f"Source runs:   {len(agent_runs)}")
    print(f"Uploaded runs: {uploaded_count}")
    if uploaded_count == len(agent_runs):
        print("✓ Counts match!")
    else:
        print(f"⚠ Count mismatch: expected {len(agent_runs)}, got {uploaded_count}")
except Exception as e:
    print(f"Could not verify via API: {e}")
    print(f"Check manually: https://docent.transluce.org/collection/{collection_id}")
