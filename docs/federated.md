# Federated / sharded MoE in brain

Source design: `federated-moe.md` (root) and the Python reference
`scratchpad/reference/sharded_moe_example/`. This doc describes what brain
implements (`crates/federated`) and what remains.

## The idea

A dense model can't be split and recombined, but a Mixture-of-Experts can: each
expert is independent given a frozen shared backbone + router. So workers can
train experts **separately** and a coordinator **assembles** them into one model.

brain uses **vertical expert shards** — expert `E` spans every layer
(`blocks.<L>.moe.experts.<E>.{w_gate,w_up,w_down}.weight`). The shared backbone is
everything else (embeddings, attention, norms, `blocks.<L>.moe.router.weight`,
head).

## Checkpoint directory layout

```
<dir>/
  shared.safetensors               # all non-expert tensors + model config
  experts/expert_NNNN.safetensors  # expert NNNN's tensors across all layers
  manifest.json                # base-config SHA-256 + per-file SHA-256 + expert list
```

## Implemented (`crates/federated` + `crates/moe`, CLI `brain federated`)

- `split <base.safetensors> <dir>` — vertical split into shared + per-expert shards
  with a hash-verified manifest.
- `verify <dir>` — re-hash every file and confirm the shared config matches the
  manifest's base hash (rejects tampering / wrong base).
- `merge <dir> --out <full.safetensors>` — reassemble a shard dir into one checkpoint.
- `assemble <base_dir> [overlay_dir ...] --out <full.safetensors>` — overlay expert
  (or shared) shards onto a base, **last-wins** per expert id, verifying all
  overlays share the base config hash.
- **`train-expert --base <B> --expert E --out <dir>`** — train one expert against
  a frozen shared backbone (`Trainer::freeze_grads_except_expert` + AdamW with
  `wd=0`, leaving the backbone and every other expert **bit-for-bit unchanged**)
  and write an overlay shard dir ready for `assemble`. This is the on-GPU "train
  experts separately" worker step — run it one expert at a time, in separate
  sessions, only needing the common base.
- Dependency-free SHA-256; split→assemble is provably an identity (test); the
  train-scope freeze is verified by a test (backbone + non-target expert
  unchanged, target expert moves).

`make federated-demo` runs train→split→verify→merge on a real MoE checkpoint.

### Train experts separately, then assemble

```bash
brain train --steps 2000 --out base.safetensors              # common base (resumes if it exists)
brain federated split base.safetensors out/base              # base shard set
brain federated train-expert --base base.safetensors --expert 0 --out out/exp0 --steps 500
brain federated train-expert --base base.safetensors --expert 1 --out out/exp1 --steps 500
brain federated assemble out/base out/exp0 out/exp1 --out out/final.safetensors
```

Each `train-expert` runs independently and only needs `base.safetensors`, so you can
train one shard at a time on a small machine (control per-step memory with
`--batch`/`--block`). Note: the whole model is still GPU-resident during a
worker run — true memory sharding (CPU offload / layer-expert shards from
`federated-moe.md`) is not yet implemented; what you get today is independent,
sequential, auditable per-expert training.

## The full lifecycle (target)

1. Train the shared backbone / general expert.
2. Clone the expert initialization.
3. **Train each expert independently** on specialist data (frozen backbone,
   routing forced/biased toward that expert).
4. Assemble the expert shards.
5. **Router-only integration** pass (experts + backbone frozen).
6. Evaluate: routing-specialization matrix `R(e,D)`, ablation damage `A(e,D)`,
   forced-expert loss, and **marginal utility** `U(e,D)`.

## Remaining integration (tracked in docs/TESTING.md)

Steps 3/5 need MoE-engine support that isn't wired yet:
- **Train-scope** (freeze backbone, train only expert N): implementable as
  zeroing non-selected gradients each step + `weight_decay = 0` (AdamW leaves a
  param with zero grad and no decay unchanged), or a trainable-subset in `optim`.
- **Router-only integration** and the **anchor-KL + router-selectivity losses**
  (`L = L_LM + λ_anchor·KL(anchor‖client) + λ_router·BCE(client−anchor)`): new
  WGSL + a frozen anchor copy.
- **Marginal-utility / ablation eval** in `crates/eval`.

The checkpoint-level federation (the part that lets independent training results
be combined and audited) is complete; the above is the on-GPU training side.
