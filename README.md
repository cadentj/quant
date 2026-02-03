# Math 4025 project


## Launching batch jobs

We use Modal for processing batch jobs.

As of 02-03-2026, Modal's volumes v2 is in beta so we use the `--version` flag to enable it for safe concurrent writes.

```bash
modal volume create composition-results --version 2
```

