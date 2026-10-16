<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 233. Where a 27B LoRA step puts its memory, per token, after recompute

Measured with `qwen35_train_probe` on the GH200 (CUDA backend, one process,
`nvidia-smi` per-process footprint sampled every 50 ms): the first four layers
of Qwen3.8-27B at real width (d_model 5120, ff 17408, vocab 248320; three
Gated-DeltaNet layers and one GQA layer), random weights, rank 16, one LoRA
step per row. "Step" is peak minus the footprint after the build, per token.

| base | tokens | GDN segment | step MiB/token | peak MiB |
|------|-------:|------------:|---------------:|---------:|
| bf16 | 512    | whole       | 1.73           | 10 915   |
| bf16 | 2048   | whole       | 1.68           | 14 739   |
| nf4  | 2048   | 512         | 0.93           | 11 267   |
| nf4  | 4096   | 128         | 0.75           | 12 843   |

Reading it, per token (these are the components, not additional rows):

- **Layer-boundary residuals**: 20 480 B per boundary in fp32, 10 240 B in
  bf16 (`TrainOpts::bf16_residuals`), `n_layers + 1` boundaries. Measured by
  the footprint after the build: 8 layers at 4096 tokens, 755 MB fp32 against
  545 MB packed, which is the formula. At 64 layers this is 1.33 MB/token in
  fp32 and 0.67 MB/token in bf16, the largest single term once the layer
  working set is segmented.
- **One layer's working set** while it is recomputed and differentiated: 1.7
  MB/token with a whole-sequence GDN layer, 0.93 with 512-token segments,
  0.75 with 128. What is left at 0.75 is not the Gated-DeltaNet recurrence but
  the dense MLP (gate, up and hidden activations of 17 408 floats each, and
  their three gradients) and the four whole-sequence GDN projections.
- **Fixed, independent of the sequence**: the chunked LM head, two buffers of
  `rows x 248320` fp32 with `rows = 1024` at the default 1 GiB budget
  (2.0 GB), the bf16 embedding table and head (5.1 GB), the base itself.

## Derived, not measured at depth

Base bytes at 64 layers: 24.3 B linear parameters, so 48.6 GB in bf16 and
15.2 GB in NF4 (0.625 B per weight with the fp32 group scale), plus the 5.1 GB
of tables. On a 95 GiB card with about 25 GB held by an NF4 base, tables, head
chunks, adapters with their Adam state and the context, roughly 70 GB is left
for the per-token terms: 1.55 MB/token with bf16 residuals and 128-token
segments (about 46 000 tokens), 2.2 MB/token with fp32 residuals (about
32 000). A bf16 base leaves about 39 GB: 25 000 tokens with bf16 residuals.
The widest single buffer (the MLP activation, 17 408 floats per token) caps a
pass at 61 680 tokens on a 4 GiB storage binding and 30 840 on a 2 GiB one.
These are estimates from the per-token terms above; a run at depth is what
settles them.
