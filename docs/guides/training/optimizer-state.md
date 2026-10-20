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
  (`adamw_q8.wgsl`, block 64; `adamw_q8_wg.wgsl`, the cooperative form the
  selection policy picks where the device has workgroup reductions), reached through `Optim` when a `ParamStore` is
  built with `new_with_policy_src`; the CUDA backend compiles the same source.
  Both are gated against the host reference.

Adafactor and logarithmic 8-bit state are not implemented yet.

## Values in a quantized encoding, layer streaming, export

`optim::encoded::EncodedValues` stores a parameter's value in any encoding the
checkpoint codec registry can encode and decode (bf16, int8, NF4, GGUF block
formats, block-scaled FP8);
reading decodes a tile, writing re-encodes one. With an optimizer master copy
this is mixed-precision and quantization-aware training: the master accumulates
the updates, the stored value is its quantization, and the gradient taken at
the stored value is applied to the master (the straight-through estimator).
Without a master a low-precision value loses small updates, and the stochastic
rounding request is refused for encoded values rather than quietly rounded
to nearest.

A trainer that finishes one layer at a time calls `begin_step` once, then
`update_tensor` per tensor. `DeferredPublish` keeps the new values aside until
`publish`, so every gradient of a step is taken against one parameter version
(tied or shared weights make an in-place update wrong). The global clip needs
the whole gradient norm, so a layer-streamed trainer passes its own multiplier.

`export_values` streams values - from the live store, or from the masters
through `MasterView` - into any `TensorSink` in a chosen encoding, a tile at a
time.

## Wiring a trainer

A model supplies only a manifest of `(name, numel)`. A trainer creates a
`optim::trainstate::TrainState` from the manifest, a policy and a rule, and
implements `ParamTiles` (value and gradient tile access). `commit` writes
values, optimizer state, step, seed and the trainer's schedule scalars as one
atomic generation (temporary directory, fsync, rename, length and hash
verification on restore, the two newest valid generations kept); `restore`
returns the newest generation that verifies.
