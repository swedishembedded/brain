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
* on the device, fused decode-update-encode for AdamW with bf16 moments
  (`adamw_bf16.wgsl`, two moments to a word, stochastic rounding optional; qwen3 and
  qwen35) or with int8 moments
  (`adamw_q8.wgsl`, block 64; `adamw_q8_wg.wgsl`, the cooperative form the
  selection policy picks where the device has workgroup reductions), reached through `Optim` when a `ParamStore` is
  built with `new_with_policy_src`; the CUDA backend compiles the same source.
  Both are gated against the host reference.

Two further state forms run on the host streamed path:

* `StateFormat::Log8Block`: 8-bit codes on a logarithmic scale per block
  (constant relative step over ~16 octaves below the block maximum; 8 steps per
  octave signed, 16 non-negative), for heavy-tailed second moments. A positive
  second moment never decodes to zero.
* `Rule::Adafactor` (arXiv:1804.04235, no first moment): the second moment of a
  matrix is `rows + columns` floats (`TrainStateConfig::row_len` names the
  matrices; other tensors keep a plain second moment). It runs in three
  streamed passes over a tensor's gradient tiles, which are whole rows; the
  statistics live in host memory and are checkpointed beside the state
  (`factored.bin`).

Neither has a device kernel: a device-optimised parameter asking for them is
refused by name (`resolve_precisions`).

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

## Full fine-tune of a chat model

What a full fine-tune stores each trainable parameter as is ONE typed value,
`FitOpts::precision` (`model::TrainPrecision`; `ChatFullFineTune::train_precision`
in the SDK). It is parsed once from `state=int8:64,master=f32,sr=on,narrow=on,moments=host,activations=recompute`
(`TrainPrecision::parse`), carried to the model's builder, and part of the
identity a saved training state is resumed under. Nothing is read from the
calling thread (the old thread-scoped policy is gone); `OptimizerState` still
says where the moments live.

The builder resolves the precision parameter by parameter
(`TrainPrecision::resolve`) and returns an error naming the parameter for a
choice the trainer does not run: weights and gradients are f32 device buffers
with no master copy on this path, the device AdamW keeps f32, bf16 or int8
blocks of 64, and any state format (also log8) is held by the host optimizer.
`moments=host` asks for the host optimizer with any format; a format only it
runs implies it.
`activations=keep|recompute|bf16|fp8` says what the backward keeps of the
forward: every layer's intermediates, or only the residual between layers (f32,
or bf16/fp8) with each layer replayed before its backward. A narrow residual is
rounded as the forward writes it, so the backward replays exactly the input the
forward used (`model::saved::Residuals`). The fit loop hands the choice to the
model (`Model::set_saved_policy`); deepseek2, glmdsa, gpt2 and lfm2 then share
one set of activation buffers across their layers (gpt2 also holds bf16 or fp8
residuals), qwen3 and qwen35 answer from what they were built with, spectrallm,
ouro and huginn always rebuild each layer from its f32 residual and accept only
`recompute`, and a model that keeps every layer refuses any other choice by
name. A built model only narrows what it keeps.

The memory planner decides the optimizer-state format before anything loads
(`TrainPrecision::plan` over `memplan::training_demands`): the requested format
when the run fits, a narrower one (bf16, then int8 blocks) only when the run
authorized `narrow=on` (the report of the decision is printed), otherwise a
refusal with the arithmetic.

Resume is exact wherever the state lives. The generic fit loop writes the
resume file (weights and moments of the device-optimised parameters, step, batch
generator, read through `Model::optimizer_parts` for any model with a store)
and commits the state kept off the device (the host optimizer of an offloaded
store, through `Optim::host_commit`, for every model; the master copies of a
resident build, through `Model::off_device_state`) as one `CheckpointDir`
generation keyed by the
applied-step count first; a resume restores exactly the generation its file
names and refuses (naming the generation) one that no longer verifies. The
stochastic rounding of the device state is keyed on the run's seed
(`Model::pin_rounding_seed`), so a resumed run replays the draws.

## Resident bf16 / FP8 full fine-tune

`model::resident_ft::ResidentFullFt` trains the linears of a model while they
live on the device in the encoding they are served in (bf16, or block-scaled
FP8 E4M3 with 2-D tile scales) as the same `ops::Weight` tiers the forward and
`dX` products already run. Per step the backward accumulates each tensor's `dW`
with `Ops::weight_grad` (fp32 or bf16 gradient buffer); `step` multiplies by
the clip and accumulation scale, updates the master (f32 or bf16) and the
moments (f32, bf16 or int8 blocks) off the device through `TrainState`, and
re-encodes the stored value in place (`ops::rewrite_tile`). `layered_step` is
the layer-at-a-time form: `update(i)` as each gradient completes, new values
held in a `DeferredPublish` and published together.

A master is required (a resident low-precision value loses sub-spacing
updates), and the planner refuses a configuration before loading
(`resident_ft::check_fits` over the weights, gradients, host-side master and
state and the caller's other demands). qwen3 and qwen35 wire it as
`train_full_resident` / `new_resident`. The vocabulary tables (the embedding
and an untied LM head, `ResidentRole::Table`) live in the same state in bf16,
whatever the linears' encoding, with their master and moments off the device;
the embedding gradient scatters over the distinct looked-up rows only
(`Ops::embedding_grad`). Only the norms train in fp32 on the device, and the
clip spans both groups. The step itself is one
generic function, `resident_ft::adamw_step`: a model supplies its backward and
its device optimizer through `ResidentBackward`, and the global gradient norm
is reduced on the device (`model::gradnorm`: `gradnorm_part`, with a bf16 twin
for bf16 gradients), never read back to the host.

### Layer-streamed full fine-tune

When the weights do not all fit, `ResidentFtConfig::window` keeps only a window
of layers on the device. The stored value of every tensor then lives on the host
in its device encoding; a layer's tensors are copied into one of a bounded set of
device slots when the forward or the backward reaches it
(`ResidentFullFt::load_layer`), and the gradients of a layer go to one
gradient lane per tensor shape that the next layer reuses. The device holds
`window` layers of weights and one layer of gradients, whatever the depth, plus
the bf16 tables and their gradients, which stay resident for the whole step and
are updated once the backward is complete, by the step's final multiplier; the
planner prices exactly that (`resident_ft::demands`).

A layer is updated between the backward segments, as soon as its gradient is
complete (`begin_pass` / `finish_layer` / `end_pass`): nothing later in the
same step reads a finished layer's weights, so every gradient of a step is taken
against one parameter version. With `defer_publication` the new values of a
layer wait in a `DeferredPublish` store and reach the host copy together, so a
layer is never half updated. No gradient outlives its layer, which makes a
step one micro-batch (a second backward before the update is refused).

The global clip needs the whole gradient norm before any update. Two forms:
`ClipPass::TwoPass` (default) runs the backward twice, the first pass only
measuring the norm on the device, so the update applies the exact global
multiplier and the result equals the resident step bit for bit
(`model/tests/layer_streamed_ft.rs`, `qwen3/tests/layer_streamed_full_ft.rs`,
`qwen35/tests/layer_streamed_full_ft.rs`); `ClipPass::Running` runs it once and
clips each layer by the norm accumulated so far (the head first), which never
clips harder than the global clip and equals it when nothing clips. Without a
clip there is one pass.

The run asks for it with one model-agnostic value (the SDK's
`train_precision`, the CLI's `--train-precision`):
`state=int8,master=f32,weights=fp8,window=2,clip=two-pass`. Every trainer
declares what it supports (`ChatTrainer::TRAIN_PRECISION`) and a request outside
it is refused by name before anything loads.

The optimizer-state part of that value reaches every model through one generic
path. A model that keeps a `ParamStore` and the fused device `Optim` exposes them
(`Model::optimizer_state`), and the fit loop applies the run's precision before
the first step or resume (`trainprec::apply_train_precision`): each trainable
tensor gets f32, bf16 or int8-block moments as asked, the update graph is
recorded again, and a model whose build already holds that format is left as it
is. With `moments=host` (or a host-only format) every trainable tensor moves to
the host optimizer instead (`ParamStore::offload_all`): its device moment
buffers are released and `Optim::step` updates it from host memory, so the
device holds each weight and its gradient only. A data-parallel replica or a
pipeline stage, whose updates the trainer's own host optimizer runs, releases
its device moments the same way (`trainprec::release_device_moments`). A model that exposes no optimizer state, a format the device optimizer does
not run (log8, another int8 block size), or a moment kernel the model did not
register (`optim::MOMENT_KERNELS`, which `optim::with_moment_kernels` appends to
a kernel list) is refused by name, never trained in fp32 instead.

Resume is transactional: the model exposes `Model::off_device_state`, and the
generic fit loop commits it (`<state>.offdevice/`, one generation per optimizer
step, keyed by the applied-step count) before it writes the resume file, and
restores exactly that generation on resume. `Qwen::export_resident` streams the
masters into a safetensors file in a chosen encoding
(`checkpoint::packed::SafetensorsSink`), one tile of host staging; the
`PackedReader` and `DeviceWeightSink` reload it byte-identically.
