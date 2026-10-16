<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 234. The real 27B LoRA and QLoRA steps run on one GH200

Measured 2026-10-16 on the GH200 (95 GiB, CUDA backend, release binaries, brain at
258eaba9). Peak is the per-process footprint sampled from `nvidia-smi` every 50 ms.
Raw logs, commands and the summary are kept with the run record of this session.

**Real Qwen3.8-27B (FP8 HF directory streamed tensor by tensor), `brain qwen35 finetune`, LoRA rank 16:**

| base | rows x batch | steps | wall s | host peak RSS MiB | device peak MiB | loss first -> last |
|---|---|---:|---:|---:|---:|---|
| NF4 | 1024 x 1 | 1 | 460 | 6,924 | 27,989 | 2.99 -> 3.58 |
| NF4 | 128 x 2, lr 2e-4 | 40 | 937 | 6,938 | 24,559 | 2.78 -> 0.0002 |
| bf16 | 1024 x 1 | 1 | 321 | 6,970 | 59,053 | 3.07 -> 3.64 |
| NF4 | 1024 x 32 (old CLI default batch) | 1 | - | 6,923 | 92,571 | out of memory |

- Host memory stays below 7 GiB for the whole 27B. The 40-step run drives the loss on
  eight invented-fact examples to about zero: that is memorisation of the training set,
  not a held-out result. One step at the default learning rate overshoots (the loss rises).
- A batch of 32 rows of 1024 tokens ran out of device memory. A streamed-base run now
  refuses a batch wider than one beyond 8192 tokens per micro-step and names
  `--batch 1 --grad-accum G`; rows of 2048 tokens or more at batch 1 switch on the
  long-sequence settings (segmented Gated-DeltaNet layers, bf16 residual stream, MLP in
  token chunks). Those settings were measured with the probe below; the CLI path with them
  on the real checkpoint has not been run.

**Real width, 64 layers, random weights (`qwen35_train_probe`), rank 16, recompute on,
GDN segment 128, bf16 residuals, MLP chunks of 512:**

| base | tokens | peak MiB | step s | tokens/s |
|---|---:|---:|---:|---:|
| NF4 | 8,192 | 35,947 | 405 | 20 |
| bf16 | 8,192 | 67,011 | 435 | 19 |
| bf16 | 16,384 | 77,879 | 874 | 19 |
| NF4 | 32,768 | 68,565 | 1,724 | 19 (record in 233, not reproduced in this run) |

- Step cost is linear: 0.547 MiB per token for NF4 (matches 233). By that line a bf16
  base should still fit 24,576 tokens (about 88.7 GiB, derived) and not 32,768 (about
  99.6 GiB, derived); an NF4 base reaches about 49K tokens (derived), and the widest
  single buffer caps a pass at 61,680 tokens against a 4 GiB binding (derived).
- Throughput is 19-20 tokens/s for both tiers; it is not tuned.

**Not established:** that a trained qwen35 adapter changes the model's generation on the
real 27B. No path applies a qwen35 adapter to the real checkpoint: `brain qwen35 infer
--adapter` folds into a 108 GB fp32 file on the host (above the job's 102.6 GiB cap), the
INT8 GGUF resident path has no adapter code, and the serving adapter bank is qwen3-only.
That is the next item for the 27B (adapter on the resident engine).
