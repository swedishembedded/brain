# Federated / sharded MoE training

## The idea

A dense model can't be split and recombined, but a Mixture-of-Experts can: each
expert is independent given a frozen shared backbone + router. So workers can
train experts **separately** and a coordinator **assembles** them into one model.

brain uses **vertical expert shards** — expert `E` spans every layer. The
shared backbone is everything else (embeddings, attention, norms, the router,
head).

## Checkpoint directory layout

```
<dir>/
  shared.safetensors               # all non-expert tensors + model config
  experts/expert_NNNN.safetensors  # expert NNNN's tensors across all layers
  manifest.json                    # base-config SHA-256 + per-file SHA-256 + expert list
```

## `brain federated` subcommands

- `split <base.safetensors> <dir>` — vertical split into shared + per-expert shards
  with a hash-verified manifest.
- `verify <dir>` — re-hash every file and confirm the shared config matches the
  manifest's base hash (rejects tampering / wrong base).
- `merge <dir> --out <full.safetensors>` — reassemble a shard dir into one checkpoint.
- `assemble <base_dir> [overlay_dir ...] --out <full.safetensors>` — overlay expert
  (or shared) shards onto a base, **last-wins** per expert id, verifying all
  overlays share the base config hash.
- `train-expert --base <B> --expert E --out <dir>` — train one expert against a
  frozen shared backbone, leaving the backbone and every other expert
  bit-for-bit unchanged, and write an overlay shard dir ready for `assemble`.
  This is the "train experts separately" worker step — run it one expert at a
  time, in separate sessions, only needing the common base.

## Train experts separately, then assemble

```bash
brain train --steps 2000 --out base.safetensors              # common base (resumes if it exists)
brain federated split base.safetensors out/base              # base shard set
brain federated train-expert --base base.safetensors --expert 0 --out out/exp0 --steps 500
brain federated train-expert --base base.safetensors --expert 1 --out out/exp1 --steps 500
brain federated assemble out/base out/exp0 out/exp1 --out out/final.safetensors
```

Each `train-expert` runs independently and only needs `base.safetensors`, so you
can train one shard at a time on a small machine (control per-step memory with
`--batch`/`--block`).

## Current limitation

Today's implementation is single-box: the whole model stays GPU-resident during
a worker's `train-expert` run, so this is not yet a true distributed
multi-node training system. What you get is independent, sequential, auditable
per-expert training — each expert trained and verified on its own, then
assembled — not memory sharding across machines.
