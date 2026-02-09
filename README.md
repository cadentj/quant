# Math 4025 project

`inspect eval-retry logs/2026-02-09T17-20-28+00-00_aime2025_NrwYGRS82DVh57rbFf7zCQ.eval`

## Launching batch jobs

We use Modal for processing batch jobs.

As of 02-03-2026, Modal's volumes v2 is in beta so we use the `--version` flag to enable it for safe concurrent writes.

```bash
modal volume create composition-results --version 2
```

## TODOS

Exp 1
- Train at bf16
- does the mlp learn the cmsp at this precision? eric ran things at fp32
- Might need to adjust width

- can also play around with other optimizations to make sure the MLP learns at this precision. see: 
  - https://proceedings.neurips.cc/paper_files/paper/2023/hash/6c0ff499edc529c7d8c9f05c7c0ccb82-Abstract-Conference.html

Exp 2
- train at bf16 for different number of training samples (increasing N/D). adjust the # of samples depending on whether the mlp learns at bf16
- simple PTQ to 8,6,4,2 bit 

Exp 3
- pretrain at lower precision, does this robustify the model to PTQ?

Exp 4
- does the model learn different algorithms when trained at a lower precision? 

Maybe:
- derive a relation between precision and MLP width