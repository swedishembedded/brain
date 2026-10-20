# Optimizer state, precision and checkpoints for full fine-tuning

Full fine-tuning keeps, per trainable parameter, a value, a gradient, optimizer
state (moments) and optionally a master copy. brain chooses each independently,
per parameter group, and accounts the physical bytes of every one.

## Precision per parameter

`paramstore::precision::ParamPrecision` has four independent fields: value
dtype (`F32`, `Bf16`), gradient dtype, optimizer-state format
(`F32`, `Bf16`, `Int8Block { block }`) and master policy (`None`, `Bf16`,
`F32`), plus a stochastic-rounding switch for low-precision re-encodes.
A `PrecisionPolicy` is an ordered list of matchers (name contains / suffix,
minimum size) over a default; `report` prints, per parameter, which rule
fired and the physical bytes. Nothing is chosen from hardware or memory
pressure silently; `optimizer_state_ladder` gives a memory planner the byte
cost of each candidate state format for a manifest.

## State formats

* `F32`, `Bf16`: plain.
* `Int8Block { block }`: 8-bit codes with one f32 scale (the block maximum) per
  block. First moments and momentum use sign plus a 7-bit square-root map,
  second moments an 8-bit fourth-root map that never decodes a positive value
  to zero. Cost is `1 + 4/block` bytes per element per slot.

Stochastic rounding matters for 8-bit state: an exponential moving average
with decay 0.999 changes a value by less than one code, and nearest rounding
freezes it. The randomness is a function of `(seed, step, tensor, slot,
element)`, so results do not depend on tiling or on resuming.

## Update rules and where they run

`optim::rule::Rule` is the single definition: `AdamW`, `Lion`, `SgdMomentum`.
It runs

* on the host, tile by tile, through `StreamedOptimizer` over an
  `OptimizerStateStorage` (memory or a file, bounded staging independent of the
  parameter count), used by `OffloadAdam` and by a generic trainer through
  `TrainState`;
* on the device, fused decode-update-encode for AdamW with int8 moments
  (`adamw_q8.wgsl`, block 64), reached through `Optim` when a `ParamStore` is
  built with `new_with_policy_src`; the CUDA backend compiles the same source.
  Both are gated against the host reference.

Adafactor and logarithmic 8-bit state are not implemented yet.

## Wiring a trainer

A model supplies only a manifest of `(name, numel)`. A trainer creates a
`optim::trainstate::TrainState` from the manifest, a policy and a rule, and
implements `ParamTiles` (value and gradient tile access). `commit` writes
values, optimizer state, step, seed and the trainer's schedule scalars as one
atomic generation (temporary directory, fsync, rename, length and hash
verification on restore, the two newest valid generations kept); `restore`
returns the newest generation that verifies.
