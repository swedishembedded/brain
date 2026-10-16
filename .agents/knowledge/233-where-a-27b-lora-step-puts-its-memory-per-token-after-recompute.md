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

## Both knobs on, at depth

With NF4 base, bf16 residuals, 128-token GDN segments and a 512-token MLP chunk
(`--bf16-residuals on --gdn-segment 128 --mlp-rows 512`), release build:

| layers | tokens | built MiB | peak MiB | step MiB/token | step time |
|-------:|-------:|----------:|---------:|---------------:|----------:|
| 8      | 4096   | 10 977    | 13 225   | 0.55           | 22 s      |
| 64     | 16 384 | 37 851    | 46 815   | 0.55           | 692 s     |
| 16     | 49 152 | 27 457    | 54 363   | 0.55           | 712 s     |

The step term is the same 0.55 MiB/token at 16K and 49K tokens, so nothing
grows faster than linearly in the sequence (flash attention for the GQA
layers, segmented recurrence for the GDN layers). The full 64-layer model runs
at about 24 tokens/s on this card in this configuration; the segmented
recurrence and the on-the-fly NF4 decode are the slow parts, not the memory
plan.

## Derived from those runs

Fixed at 64 layers with an NF4 base: about 25.7 GiB (base 15.2 GB, tables
5.1 GB, head chunks 2.0 GB, adapters with their Adam state, context). Per
token: 0.63 MiB of bf16 boundaries (65 x 10 240 B) plus about 0.1 MiB of
whole-pass buffers plus the 0.55 step term, about 1.3 MiB. A 95 GiB card has
about 69 000 MiB for that: roughly 53 000 tokens, against the 61 680-token
cap of the widest buffer (the MLP activation) on a 4 GiB storage binding
(30 840 on a 2 GiB one). At 49 152 tokens the 64-layer peak is the 16-layer
measurement plus 48 layers of boundaries (23 000 MiB) and NF4 weights (10 900
MiB): about 88 000 MiB, which fits with little to spare. A bf16 base is
48.6 GB and leaves about 39 GB for the per-token terms: 25 000 to 30 000
tokens. These are arithmetic on measured terms; the 64-layer run at 49 152
tokens was not made.
