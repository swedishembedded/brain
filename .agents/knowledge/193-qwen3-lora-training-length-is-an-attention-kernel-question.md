<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 193. Qwen3 LoRA training length was capped by one T² buffer; now attention speed caps it

The materialised training attention (`gqa_scores` -> `attn_softmax` ->
`gqa_apply`, and the `gqa_bwd_*` chain) keeps `n_heads·T²` fp32 scores,
probabilities and score gradients. For Qwen3-0.6B and 1.7B (16 query heads)
one such buffer is `64·T²` bytes. It passes the 2047 MiB storage-binding limit
at `T = 5793`. A 5824-token step failed at `create_bind_group` for
`gqa_scores` (2070 MiB, wgpu). On native Vulkan, 5760 and 5824 tokens made no
progress in 12 and 15 minutes. Well below the limit the chain was already slow:
24.2 s per step at 2048 tokens and 160 s at 4096.

Training now runs the causal GQA flash trio (`flash_attn_causal_gqa`, which
also writes the row log-sum-exp, then `flash_attn_causal_gqa_bwd_{dq,dkv}`).
The backward rebuilds each softmax weight from that log-sum-exp, so no
`[H,T,T]` buffer exists. Three other changes were needed before a long block
fit on the card and completed a step:

- The backward recomputes each layer's forward from the saved residual
  `res[l]`, so all layers share one set of activation buffers.
- The LM head works in row chunks, and `WeightedCe` takes the same chunks.
- The tape is submitted one layer at a time. Submitted whole, a 16k-token
  backward is about 80 s in one batch, and the 30 s `BRAIN_GPU_WAIT_S` bound
  fired on it.

Profiling on the way found the per-row cross-entropy pair (`ce_value_masked`,
`ce_stats`) at 1% of its memory roof. It cost 0.64 s of a 4.2 s step at 2048
tokens. `ce_value_stats_rows` (one workgroup per row) does the same work in
4 ms.

Measured by `qwen_bench lora-train` on one P40 (`--device vulkan`). Setup:
batch 1, rank 8, alpha 16, all 7 projections, AdamW, clip 1.0. Each row is
the mean of 3 timed steps after 2 warm-up steps (1 timed step from 16k up).
0.6B used the real checkpoint. 1.7B used random weights at its published
shape, because its checkpoint is not in the model store. Peak is the
process's own device memory from `nvidia-smi`.

| T | 0.6B s/step | tok/s | peak MiB | 1.7B s/step | tok/s | peak MiB |
|---|---|---|---|---|---|---|
| 512 | 0.88 | 585 | 2920 | 1.76 | 291 | 7626 |
| 1024 | 1.70 | 602 | 3958 | 3.10 | 330 | 8500 |
| 2048 | 3.61 | 567 | 5453 | 6.25 | 328 | 10247 |
| 4096 | 9.72 | 421 | 7806 | 15.0 | 272 | 13074 |
| 8192 | 29.7 | 275 | 9053 | 40.4 | 203 | 15315 |
| 16384 | 101.5 | 161 | 11584 | 122.9 | 133 | 19799 |
| 24576 | 215.7 | 114 | 14115 | 247.1 | 99 | card full |
| 32768 | 373.3 | 88 | card full | out of memory | | |
| 40960 | 574.3 | 71 | card full | | | |

Rows marked "card full" read the whole 24 GB from `nvidia-smi` while another
process shared the card, so the reading is an upper bound and not this run's
own peak.

At 16k tokens the flash forward is 14.4 s of a 19.0 s forward. The three
flash kernels run at 15-20% of the card's fp32 roof. Throughput past a few
thousand tokens is now set by those kernels, not by memory. Recomputing each
layer costs one extra forward, about 0.4 s of the 1.7 s step at 1024 tokens.

Rule: before calling a sequence length a model's limit, find the single
largest buffer and the single longest submission, and check each against the
binding limit and the wait bound.
