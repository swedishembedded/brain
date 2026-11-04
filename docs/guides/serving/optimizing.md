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
3. **Plans.** It picks the format of every linear that minimizes the summed
   estimate within the budget. Linears the engine fuses into one matrix
   product (Q/K/V, gate/up) always get one format together.
4. **Measures.** On the held-out records it measures the plan, the original
   model, and every single format that fits the budget, first with the weights
   rounded by the deployed codec, then with the model built in each format on
   the kernels serving uses.
5. **Decides.** The plan is **accepted** when its held-out loss on the
   deployed kernels is lower than that of every single format that fits the
   budget. `brain optimize` exits 0 when accepted and 3 when not.

The estimate only chooses what to measure. Acceptance always comes from the
held-out measurement.

## Options

| Option | Meaning |
|---|---|
| `--dataset FILE` | chat JSONL (the fine-tuning format) representative of what will be served |
| `--out DIR` | where the prepared data and `optimize.json` go |
| `--held-out FILE` | chat JSONL to measure on; all of `--dataset` then calibrates |
| `--budget FORMAT` | the bytes every linear takes in one format (default `int8`) |
| `--bits-per-weight B` | mean stored bits per weight, scales included (`int8` is 9) |
| `--max-bytes N` | bytes over every searched linear |
| `--formats LIST` | candidate formats, comma separated |
| `--arch ID` | the model's architecture (default `qwen3`) |
| `--seed N` | the split's seed (default random, reported) |
| `--max-block N` | the longest record, in tokens |
| `--no-thinking` | render records without the reasoning block |

The candidates default to every format that the architecture can (a) hold a
frozen base in, which is what lets each be measured on its deployed kernels,
(b) serve one linear at a time, so the accepted plan can be served as printed,
and (c) hold on this device as itself (int8 needs packed-int8 products). For
qwen3 that is `f32`, `bf16` and `int8`. Formats left out by (c) are listed. A
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

## Choosing the data

The estimate and the measurement are only as good as the records. Use
conversations like the ones the model will serve; a plan searched on one
domain is measured on that domain only. Larger held-out sets make the
accept/reject decision more reliable.
