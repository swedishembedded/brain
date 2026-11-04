<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Optimizing a model for a memory budget - `brain optimize`

A model stored in one weight format everywhere wastes memory: some linears
lose almost nothing at four bits, others noticeably at eight. `brain optimize`
finds a format for every linear of a chat model that fits a byte budget, and
accepts the result only when it measures better than storing every linear in
one format of the same size.

```
brain optimize Qwen/Qwen3-0.6B --dataset domain.jsonl --out runs/optimize
```

It prints the measurements and the policy to serve, and writes
`runs/optimize/optimize.json`:

```
brain serve --weight-format 'f32,blocks.0.attn.wq.weight=i8,...'
```

[How the model optimizer works](../../concepts/model-optimizer.md) explains
the method with diagrams and a worked result.

The same search is `brain::ModelOptimizer` in the SDK:

```rust
let outcome = brain::ModelOptimizer::from_pretrained("Qwen/Qwen3-0.6B")
    .dataset("domain.jsonl")
    .out_dir("runs/optimize")
    .run()?;
println!("--weight-format {}", outcome.report.policy);
```

## What it does

1. **Splits the data.** A quarter of the dataset's records is held out
   (`--held-out FILE` names a separate file instead). The rest calibrates. The
   two never share a record.
2. **Estimates.** On the calibration records it measures, for every linear and
   candidate format, how much the loss would rise if that linear alone were
   stored in that format: the squared rounding error of the real codec,
   weighted by how sensitive the loss is to each weight (the mean squared
   gradient).
3. **Plans.** It times every linear's matrix product on the device in every
   candidate format, then picks the format of every linear that minimizes the
   summed estimate within the byte budget and, when you set one, a latency
   limit: a format that is small but slow to decode on this device is avoided
   where the limit needs it. Linears the engine fuses into one matrix
   product (Q/K/V, gate/up) always get one format together.
4. **Measures.** On the held-out records it measures the plan, the original
   model, and every single format that fits the budget, first with the weights
   rounded by the deployed codec, then as deployed: through the serving engine
   `brain serve` runs (its kernels, int8's quantized activations included)
   where the architecture has an engine scoring path, else on the
   training build's frozen base. The report says which.
5. **Decides.** The plan is **accepted** when its deployed held-out loss is
   lower than that of every single format that fits the budget.
   `brain optimize` exits 0 when accepted and 3 when not.
6. **Recovers (optional).** With `--recover-steps N`, a LoRA adapter is
   trained over the base held at the plan to match the original model's
   output distributions on the calibration records, and the plan is
   measured without and with it. The adapter is written to
   `DIR/recovery.safetensors` and is served beside the policy like any
   adapter.

The estimate only chooses what to measure. Acceptance always comes from the
held-out measurement.

## Options

| Option | Meaning |
|---|---|
| `--dataset FILE` | chat JSONL (the fine-tuning format) representative of what will be served |
| `--out DIR` | where the prepared data and `optimize.json` go |
| `--held-out FILE` | chat JSONL to measure on; all of `--dataset` then calibrates |
| `--budget FORMAT` | the bytes every linear takes in one format; by default the budget is halfway between the two smallest candidates' uniform sizes |
| `--bits-per-weight B` | mean stored bits per weight, scales included (`int8` is 9) |
| `--max-bytes N` | bytes over every searched linear |
| `--frontier N` | also solve N plans across the candidates' range and measure each, to show the trade-off |
| `--latency-rows N` | the rows each product is timed at (default 1, single-stream decode) |
| `--max-latency-like FORMAT` | no slower, summed over the linears' products, than every linear in FORMAT |
| `--max-latency-ms MS` | the linears' summed product time in milliseconds |
| `--formats LIST` | candidate formats, comma separated |
| `--arch ID` | the model's architecture (default: what its config names) |
| `--seed N` | the split's seed (default random, reported) |
| `--max-block N` | the longest record, in tokens |
| `--no-thinking` | render records without the reasoning block |
| `--recover-steps N` | distill a recovery adapter for N steps (default 0: none) |
| `--recover-lr LR`, `--recover-rank R` | the recovery adapter's learning rate (default 1e-4) and rank (default 16) |

The candidates default to every format that the architecture can (a) hold a
frozen base in, which is what lets each be measured on its deployed kernels,
(b) serve one linear at a time, so the accepted plan can be served as printed,
and (c) hold on this device as itself (int8 needs packed-int8 products). For
qwen3 that is `f32`, `bf16`, `int8` and `nf4`. Formats left out by (c) are listed. A
format named in `--formats` that fails any of the three is refused before the
model loads.

Only the linears are searched. Embeddings, norms and the output head keep the
format they have.

## Serving the result

The printed policy is a `--weight-format` value (see
[running a server](serving.md)), with one format per linear:

```
brain serve --weight-format "$(jq -r .report.policy runs/optimize/optimize.json)"
```

## Removing channels - `brain prune`

`brain prune` makes the model itself smaller: it removes the least important
channels of every dimension the architecture lets it narrow (for qwen3, each
layer's MLP hidden channels) and writes the narrower model as a checkpoint
directory with the base's tokenizer, which every brain command opens like any
other.

```
brain prune Qwen/Qwen3-0.6B --dataset domain.jsonl --out runs/prune --keep 0.75
brain serve --checkpoint runs/prune/checkpoint
```

Each channel is ranked by how much removing it would raise the loss on the
calibration records (the same estimate the format search uses, with the
channel's weights as the error). The original and the pruned model are then
measured on the held-out records, through the serving engine where the
architecture has a scoring path. `--keep F` keeps that share of each
dimension's channels (default 0.75), `--keep-channels N` exactly N;
`--recover-steps N` distills a recovery adapter over the pruned checkpoint
from the original, as `brain optimize` does. `DIR/prune.json` holds the
outcome.

## Models and backends

Neither command has code for a particular architecture. The architecture is
read from the checkpoint (its `config.json`, or a brain checkpoint's own
config; `--arch` names it otherwise), and a model family takes part through
what it declares about itself: the linears whose format can be chosen, the
ones its engine fuses into one product, the dimensions whose channels can be
removed, the formats its serving engine holds per linear, and how to build
that engine at a format plan. Today qwen3 (with the Llama and Qwen2
checkpoints it serves as config variants), qwen35 and qwen35moe declare all
of it; a family that declares nothing to compress is refused by name.

The search runs on whatever backend `--device` selects (wgpu, native Vulkan,
CPU, CUDA). A format the device cannot hold as itself (int8 without
packed-int8 products, bf16 on a backend without bf16 storage) is left out of
the search and listed, and an engine asked to serve one says what it holds it
as instead. The timings come from that device.

## Choosing the data

The estimate and the measurement are only as good as the records. Use
conversations like the ones the model will serve; a plan searched on one
domain is measured on that domain only. Larger held-out sets make the
accept/reject decision more reliable.
