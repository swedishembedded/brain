# minimaxmusic3 - roadmap

MiniMax Music 3 (`MiniMaxAI/MiniMax-Music3`): lyrics + a structured music
description in, a full song out (up to 5 minutes, 44.1 kHz stereo). No
official inference code exists upstream - only an unmerged `diffusers` PR
(commit `dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d`) implements it; this port
reads that PR (and the checkpoint's own real tensor shapes, read directly
over HTTP range requests) as the reference, without vendoring any of its
files into this tree.

Five chained components, ~19B parameters total:

1. **Global LLM** - a real Qwen3-8B (`hidden=4096, layers=36, heads=32,
   kv_heads=8, head_dim=128, vocab=200000` - the checkpoint's own
   `language_model/config.json`, not the smaller published Qwen3-8B preset),
   reused verbatim from `crates/qwen3`. Autoregressive, CFG-guided: one
   semantic RVQ code per 25 Hz frame. The checkpoint ships a SECOND
   language-model directory, `qwen_7B/qwen_7B/`, matching the same gross
   dims but NOT this architecture (`config.json`: `"architectures":
   ["AbabForCausalLM"]`, `"model_type": "mixtral"`, with per-layer
   LayerNorm alpha/beta residual-scaling MiniMax's native training
   checkpoint carries and plain Qwen3 does not) - `language_model/`, not
   `qwen_7B/qwen_7B/`, is the one this port loads through `crates/qwen3`.
2. **RVQ depth decoder** (0.65B) - a 4-layer causal transformer that
   autoregressively predicts the 7 residual codebooks per frame from the
   Global LLM's hidden state, and owns the residual-code embedding table.
3. **Condition encoder** (25M) - softmax-mixes the 8 per-frame hidden states
   (LLM + 7 depth steps), projects, nearest-resamples from the 25 Hz frame
   rate to the Flow-VAE latent timeline.
4. **Flow-matching DiT** (2.4B) - 36-layer LayerNorm transformer (partial
   RoPE: 32-of-64 head dims rotate), denoises Flow-VAE latents in 200-frame
   chunks with a 100-frame hop, splicing consecutive chunks over a
   172-latent overlap.
5. **Vocoder** (123M) - a DAC-style decoder (SnakeBeta + weight-normalized
   conv/conv-transpose) folding 128 latent channels into 2 (stereo) at
   44.1 kHz.

## Validation policy

Real-weight tensor/layer parity for every component (streamed + int8 where
fp32 won't fit in this machine's ~21 GB usable RAM), plus one real, short
(a few seconds, a single denoise chunk, a short AR run) end-to-end
generation producing an actual WAV on this machine. A full 5-minute
generation (14 overlapping chunks, an AR loop over ~1500 frames on the 8B
LM) is not attempted here - recorded as a hardware-bound gap, not silently
skipped.

## Phase 1: golden dumper

`tools/goldens/minimaxmusic3_dump_reference.py` covers the four
`MiniMaxMusic3*`-prefixed diffusers classes (condition encoder, vocoder,
RVQ depth decoder, DiT) at both random-weight `--tiny` dims (matching
`crates/minimaxmusic3::config`'s `::tiny()`, no checkpoint needed) and
`--real` dims (real weights, `strict=True` state-dict load). The Global
LLM's own golden path is `transformers.Qwen3ForCausalLM` directly - no
`diffusers` PR dependency - and is deferred to the Global LLM milestone.

Reference source: an unmerged `diffusers` PR, installed into a scratch venv
this repo does not track (`pip install
"git+https://github.com/huggingface/diffusers@dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d"`,
alongside `torch`/`transformers`/`safetensors`/`numpy`/`huggingface_hub`) -
see `requirements.txt`'s "NOT pip-installable" block. No file from that PR
is vendored into this tree.

Real weights for the three small components (condition encoder 97 MB,
vocoder 207 MB, RVQ depth decoder 1.3 GB - `resources/minimax-music3/`,
gitignored) were fetched and dumped; every `state_dict.load_state_dict(...,
strict=True)` succeeded on the first try, confirming the tensor names/shapes
recorded from the checkpoint's real safetensors headers were exactly right.
Real weights for the DiT (9.7 GB) and the Global LLM (17.2 GB, from
`language_model/` - see Phase 9's own note on why that directory, not the
repository's other, same-shaped but architecturally different
`qwen_7B/qwen_7B/`) are deferred to their own milestones.

Measured output shapes (real dims, batch=1): condition encoder
`(1,5,32768) -> (1,17,2048)`; vocoder `(1,128,6) -> (1,2,3072)`; RVQ depth
decoder hidden `(1,8,4096)`.

## Phase 2: condition encoder

`crates/minimaxmusic3::condition_encoder` - import (`checkpoint::safetensors`,
now also resolving a bare `diffusion_pytorch_model.safetensors` single-file
dir, not just HF-transformers' `model.safetensors` - a real gap in
`checkpoint::safetensors::read_model_dir` this component's import hit and
fixed for every future diffusers-format import too) + forward. Pure host
math, not a device (WGSL) forward: every op runs once per ~200-frame
denoise chunk on a few-MB tensor, so a device round trip would be pure
overhead with nothing to parallelize across; the conv reuses
`audio::conv::conv1d_ref`, the exact reference oracle the WGSL `conv1d`
kernel is gradient-checked against elsewhere.

One real numeric bug caught by the parity harness before it ever saw real
weights: the reference's latent-length formula chains three Python `/`
divisions, which are FLOAT division at every step (`int()` truncates only
once, at the end) - a first Rust draft using integer division at each step
computed 16 instead of the correct 17 for a 5-frame test case. The `::tiny()`
parity fixture (which needed the dumper to ALSO save the random-weight
state dict, not just the forward's input/output - PyTorch's RNG cannot be
reproduced bit-for-bit from Rust) caught this immediately.

Measured: tiny cosine 1.000000000 (exact), real-weight (4096-dim hidden,
2048-dim output) cosine 1.000000000, max_abs 3e-6 (fp32 rounding only).

## Phase 3: vocoder inference

`crates/minimaxmusic3::vocoder` - a real device (WGSL) forward, unlike the
condition encoder: this component upsamples up to 512x per call
(`upsampling_ratios = [8,8,4,2]`) and is genuinely compute-heavy, so it
belongs on the tape-based device engine every other serving path uses.
`dec_in_proj -> conv_in -> 4x (Snake -> ConvTranspose1d upsample -> 3x
dilated VocoderResidualUnit) -> snake_out -> conv_out -> tanh`.

Two new WGSL kernels, `snake1d`/`snake1d_bwd_{dx,dalpha}` (forward +
FD-gated backward, `crates/audio/src/snake.rs` + `crates/audio/tests/
snake_kernels.rs`): the checkpoint's Snake activation is the DAC original
single-parameter form (`y = x + (alpha+eps)^-1 * sin(alpha*x)^2`, `alpha`
used directly), NOT `kernels::SNAKE_BETA`'s two-parameter log-space BigVGAN
v2 form (`a = exp(alpha)`, separate `beta`) - the existing kernel does not
match this model's math and was not reused, correcting an earlier mislabel
in this crate's own doc comments. Every conv/conv-transpose reuses
`audio::conv`'s existing device kernels unchanged; weight-norm folding
(`weight[i] = g[i] * v[i] / ||v[i]||_2`, generalized over whichever axis is
dim0 of the stored tensor - confirmed against the real checkpoint that
`ConvTranspose1d`'s `weight_g` is one scalar per INPUT channel, not output)
is a one-time host op at import, not a kernel.

Fixed along the way: `checkpoint::safetensors::read_model_dir`'s
single-file fallback only resolved `model.safetensors` (HF-transformers'
name); every diffusers checkpoint here ships `diffusion_pytorch_model.
safetensors` instead, already handled on the sharded path via its own
index filename but missing on the single-file path. Fixed generically, not
locally worked around, since every future diffusers-format import hits the
same gap.

Measured: tiny cosine 1.000000000 (exact), real-weight (a 6-latent-frame,
2-channel input decoding to 3072 samples through all four upsample stages
and every residual unit) cosine 1.000000000, max_abs 1e-6 (fp32 rounding
only) - both on the CPU (Cranelift JIT) backend, no GPU needed.

## Phase 4: vocoder backward + gradcheck

`crates/minimaxmusic3::train::Trainer` - a SEPARATE forward from
`vocoder::forward` (the served path), because training needs persistent
device-resident weight/gradient buffers reused across many steps and every
intermediate activation kept around for backward, neither of which the
one-shot served path should pay for. Two more new kernels:
`bias_grad_ncl` (the per-channel bias gradient over NCL layout -
`bias_grad.wgsl` assumes the feature axis is fastest-varying, the opposite
convention from NCL's channel-then-length layout, so it doesn't fit) and
`tanh_act_bwd` (the vocoder's final activation had a forward kernel but no
backward anywhere in the workspace despite its own doc comment claiming
one existed - a stale claim, not a real gap that had been closed).

Every one of the residual unit's two consumers of its own input (the
Snake-branch and the direct `add2` skip) sums correctly in backward: this
is checked, not assumed - `crates/gradcheck::minimaxmusic3::check_vocoder`
runs `directional_check` (a random-direction two-sided finite difference)
against EVERY one of the vocoder's ~38 named parameters (every conv/
conv-transpose weight and bias across all 4 blocks x 3 residual units,
every Snake alpha), and every one passes with relative error under 8e-4 -
an order of magnitude inside the workspace's `(4e-3, 8e-2)` gate. Wired
into `make gradcheck` via `crates/gradcheck/tests/imaging_models.rs`'s
`check!` macro, same as every other model's gate.

The loss used to gradcheck is a plain MSE reconstruction loss - enough to
prove every gradient is analytically correct, which is what gradcheck
exists to do. It is NOT the loss a real training run would use.

`overfits_a_single_batch` closes the other half of this workspace's
training bar (gradcheck proves the gradients; only actually training
something proves the loop as a whole works): 800 steps of plain gradient
descent (not AdamW - `crates/optim`'s device-resident `ParamStore` is a
separate integration this trainer does not use) on one fixed batch drives
the MSE loss down more than 20x (measured: 0.0525 -> under 0.0026).

## Phase 5: vocoder LoRA

`crates/minimaxmusic3::lora` - fold-then-run, never a separate device
path: each step, `effective_weights` composes `W_eff = W_base +
(alpha/r)*B@A` on the host for every one of the 17-31 conv weights (small -
`rows` at most 1536, `cols` at most `1536*16`, and this runs once per
step, not per element of the tape), then hands the composed
`VocoderWeights` to the ordinary `train::Trainer` unchanged - every
existing conv/backward kernel runs completely unaware an adapter exists.
`train::Trainer::read_grad` already returns `dW_eff` as if the whole
tensor were trainable; `lora::backward` converts that into `(dA, dB)` via
two more small host matmuls, and `W_base` is never written back.

The three gates the workspace holds every LoRA integration to, all
passing: `zero_b_is_an_exact_no_op` (standard zero-init `B` must leave the
forward untouched, exactly), `fold_matches_apply_bit_for_bit` (composing
`W_eff` via `effective_weights`, in place, vs a separately-computed fold
must produce bit-identical fp32 output), and `lora_grads_match_finite_
differences` (`directional`-style FD check on every one of `::tiny()`'s 17
adapters' `(A, B)`). `lora_only_overfits_with_base_frozen` adds the
training-loop half: 1500 steps of gradient descent on `(A, B)` alone,
`W_base` provably untouched, reduces the loss by 40%+ - a real but looser
bar than full fine-tuning's, since a rank-2 adapter has far fewer
trainable parameters.

## Phase 6: adversarial (discriminator) training

`crates/minimaxmusic3::discriminator` - the real new-capability item
`crates/mimi::recon`'s module doc lists as absent workspace-wide (a GAN
discriminator + adversarial + feature-matching training stack), closed
here scoped to what this vocoder needs, not generalized into `crates/mimi`.
A single-resolution STFT-magnitude PatchGAN discriminator (`|STFT| ->
Conv2d -> LeakyReLU -> Conv2d -> LeakyReLU -> Conv2d -> patch logits`),
LSGAN adversarial loss, and an L1 feature-matching loss. Every conv reuses
the workspace's EXISTING `conv2d`/`conv2d_dx`/`conv2d_dw` kernels (2D conv
already had full forward+backward, from an earlier, unrelated port) plus
`leaky_relu`/`leaky_relu_bwd` (its own doc comment already named GAN
vocoder discriminators as the anticipated use) and the
`add_chan_inplace`/`bias_grad_ncl` bias pair from Phase 4 - both already
layout-generic over `[rows, C, inner]`, and NCHW's `inner = H*W` fits that
unchanged. No new WGSL kernel was needed for the discriminator itself.

The one new piece is the STFT: a direct DFT-matrix formulation
(`O(n_fft^2)` per frame, not an FFT butterfly) - deliberately, since a
windowed matmul against a fixed, precomputed cos/sin basis is trivially
differentiable (backward is the same matmul against the transposed basis),
where backpropagating through an FFT algorithm's butterfly network would
be real additional work for no benefit at the frame sizes a short training
clip needs.

Every gradient is checked, including through the STFT: a
`full_chain_waveform_gradient_matches_finite_differences` test perturbs
the FAKE waveform directly and confirms the LSGAN generator loss moves the
way `stft_mag_bwd(discriminator_backward(...).d_mag)` predicts - the exact
seam a joint generator+discriminator training loop would use. The
discriminator's own conv backward is FD-gated the same way as the vocoder
itself (Phase 4's `directional`-style check), and
`discriminator_learns_to_separate_real_and_fake` trains a discriminator
from scratch on two different waveform populations and measures its LSGAN
loss more than halve.

Scope, stated honestly: this proves the mechanism and wires every piece a
real adversarial fine-tune of the vocoder would need, but does not run
that joint generator+discriminator loop against the actual vocoder
weights, and is single-resolution (multi-resolution - running the same
discriminator at several `(n_fft, hop)` settings and summing the losses -
is a straightforward parameterization of what exists, not yet exercised).

## Phase 7: RVQ depth decoder

`crates/minimaxmusic3::depth_decoder` - a 4-layer causal transformer
(RMSNorm, plain multi-head causal self-attention with no RoPE/QK-norm/GQA/
bias, SwiGLU MLP). Pure host math, like the condition encoder and for the
same reason: a forward call processes at most `num_codebooks` (8)
positions - the checkpoint's own inference recipe recomputes this whole
short sequence from scratch at every depth step rather than caching it -
so there is nothing here to parallelize a device dispatch across.
`model::hostmath`'s existing `matvec`/`silu`/`softmax` cover the pieces
with a shared host implementation already; attention and every backward
pass are hand-derived, since this un-rotated/ungrouped/unnormalized-QK
causal-attention shape has no existing device or host counterpart to call
into.

One real bug the parity harness caught immediately: the reference class's
own `forward` adds `pos_embedding(arange(s))` INTERNALLY, before its
first layer - a first draft assumed the caller pre-added it (mirroring a
misreading of the pipeline's OWN calling code, which does no such thing
for this particular class) and got cosine 0.63 against real data despite
gradchecking perfectly against itself (a bug in what the function computes
relative to the reference, not in whether its own backward is internally
consistent - exactly the class of defect only a real-reference parity
check catches, not a self-consistency check).

Four independently-verified taps, matching cosine 1.000000000 (exact) at
both tiny and real (`hidden_size=4096, num_attention_heads=16,
num_layers=4`) dims: the transformer stack itself, `projection`, the raw
(un-summed) `audio_embeddings` gather, and each of the 7 `audio_heads`.
Backward is FD-gated the same way as every other component here,
including the newly-added `pos_embedding` gradient.

LoRA (`crates/minimaxmusic3::depth_lora`) reuses `crate::lora`'s adapter
math completely unchanged - it only ever needed a flat `[rows, cols]`
weight and never referenced the vocoder's own types, so the SAME
`LoraW`/`delta`/`apply`/`backward` apply here too. Targets the 7
per-layer linear projections (`attn.{to_q,to_k,to_v,to_out}`,
`{gate,up,down}_proj` - `14` weights at `::tiny()`'s 2 layers); RMSNorm
gains, `pos_embedding`, `audio_embeddings` and `audio_heads` are out of
scope for this adapter (not linear projections). The same three gates as
the vocoder's LoRA, all passing: exact no-op at zero-init `B`, a
directional FD check on every one of the 14 adapters' `(A, B)`, and
`lora_only_overfits_with_base_frozen` (1500 steps, base weights provably
untouched, loss reduced 40%+).

## Phase 8: flow-matching DiT

`crates/minimaxmusic3::dit` - a real device (WGSL) forward, unlike the
condition encoder/depth decoder: 36 layers at `inner_dim=2048` is genuinely
compute-heavy, and unlike the vocoder's hand-rolled conv stack this IS a
standard transformer block shape with existing reusable device primitives -
`model::block`'s `Bidir`/`rope2d_partial`/`LayerNorm`/`swiglu`/`kv_expand`
Step-builders cover the whole block (bidirectional partial-RoPE attention +
fused gated FFN) with zero new WGSL kernels. Per block: `norm1 -> QKV
(3 matmuls) -> pack into a fused qkv buffer (kv_expand, group=1, the same
non-GQA packing `crates/lfm2` already uses) -> partial RoPE (rotary_dim=32
of head_dim=64, theta=10000 - the reference's own `RotaryEmbedding`
default, distinct from the Global LLM's `1e6`) -> bidirectional attention
(no causal mask - a diffusion denoiser sees the whole chunk at once) -> out
proj -> residual -> norm2 -> ff_in (ONE fused `Linear(dim, 2*ff_inner)`,
split into `[gate_states, gate]`) -> `gate_states * silu(gate)` -> ff_out ->
residual`. The top-level glue (`cat([latent, zeros, condition^T]) ->
preprocess_conv residual -> proj_in`, a prepended Fourier-timestep token,
`proj_out -> postprocess_conv residual`) stays host math - infrequent,
tiny relative to the 36-layer stack. Scope: batch=1 only, matching every
real caller in the reference pipeline (the chunked-denoise loop is a Python
`for` loop over windows, never a batched tensor).

Cosine 1.000000000 at both `::tiny()` and the real checkpoint's dims (a
36-layer, 9.7 GB, 2-shard `transformer/` download) - the single largest
component parity-checked in this port so far, first try.

Backward + gradcheck (`crates/minimaxmusic3::dit_train::Trainer`) reuses
the SAME `model::block` builders' backward halves
(`bidir_bwd`/`rope2d_partial_bwd`/`layernorm_dx_bwd`/`swiglu_bwd`/
`kv_expand_bwd`) rather than hand-deriving attention's softmax-jacobian
backward a second time - the real payoff of having chosen device
Step-builders for the forward in the first place. The top-level glue's
backward is hand-derived host math (`model::hostmath::linear_rows_bwd`,
hoisted from `depth_decoder.rs`'s own private copy of the identical
pattern - the "one implementation" rule this workspace holds itself to).
Every one of `::tiny()`'s ~30 named parameters passes a directional FD
check to within `(4e-3, 8e-2)`; `forward_matches_serving_forward` confirms
the trainer's own forward is bit-identical (within float tolerance) to
`dit::forward`'s served path; 1500 steps of plain SGD collapse a fixed
batch's loss to under 5% of its start.

LoRA (`dit_lora`) is fold-then-run against a FRESH `dit_train::Trainer`
per step - unlike the depth decoder's host-pure LoRA, the DiT's own
Trainer is device-resident, so this follows the vocoder's own `lora.rs`
shape instead. Targets the 6 per-block linear projections
(`attn.{to_q,to_k,to_v,to_out}`, `ff_in.weight`, `ff_out.weight` - `12`
adapters at `::tiny()`'s 2 layers); LayerNorm gains/biases and the
top-level glue are out of scope (not linear projections). The same three
gates as every other LoRA integration here, all passing, plus
`lora_only_overfits_with_base_frozen` (1500 steps, `lr=0.3` - a higher
learning rate than the other two LoRA integrations needed to clear the
same `40%+` reduction bar, tried before loosening the bar itself, per this
workspace's own "tighter fixture before looser assertion" convention).

INT8 storage tier (`dit_int8`, following `ltxv::int8`'s own precedent
closely) - a smaller checkpoint in host RAM/on disk, no compute-path
change, no new kernel. Never-quantized: `proj_in`/`proj_out` (the model's
first/last projections) and both `time_embed` linears (the
timestep-conditioning MLP); every other 2D weight left eligible is the
same 6-per-block-linear set LoRA targets. 12 of 33 tensors int8-eligible
at `::tiny()`'s dims; full-model forward cosine 0.999999+ after a round
trip through int8 storage.

`model::Shardable` pipeline-parallel sharding (`dit_shard::DitStage`),
following `crates/ltxv::LtxDit`'s own precedent closely (the one existing
diffusion-transformer `Shardable` impl in this repo, discovered mid-
implementation to require the FULL `model::Model` trait too, not a thin
adapter - the user's explicit direction on discovering this was "we don't
retrofit things in brain, we implement them properly", so this is the
real trait, done the way the one legitimate precedent for this model
class already does it, not a shortcut): every stage loads only its own
contiguous block range plus its replicated `time_proj`/`time_embed`
weights; embed/head stages additionally own the boundary projections; the
residual stream is the only thing that crosses a stage boundary; `Model::
backward`/`Shardable::run_backward_stage` are honest `unimplemented!()`
gaps, not silently-wrong stubs - this crate's real, single-device DiT
training story stays `dit_train::Trainer`, which `Shardable` does not
build on (the same split `ltxv::LtxDit`'s own module doc documents for the
identical reason: pipeline sharding exists to let a too-big-for-one-card
model run split for INFERENCE, not to add a second training path).
Validated the only way possible on this machine (no discrete GPU at all):
`new_shard` genuinely loads only its block range's weight subset; the
single-shard degenerate case and a real two-stage split with a
host-staged residual handoff both match `dit::forward` bit-for-bit;
`shard_cost`-driven `plan_balanced` produces a well-formed partition at
both `::tiny()` and the real 36-layer config's shape.

## Phase 9: Global LLM

`crates/minimaxmusic3::global_llm` - a real Qwen3-8B, reused VERBATIM from
`crates/qwen3` (no local reimplementation at all, unlike every other
component). This module owns only streamed import and the training
objective this port adds.

A real find mid-milestone: the checkpoint ships TWO differently-shaped
language-model directories. `qwen_7B/qwen_7B/`'s own `config.json` reads
`"architectures": ["AbabForCausalLM"]`, `"model_type": "mixtral"`, with
per-layer LayerNorm alpha/beta residual-scaling constants - MiniMax's
native training-checkpoint format, NOT reusable through `crates/qwen3`
despite matching `hidden=4096, layers=36, heads=32, kv_heads=8,
head_dim=128, vocab=200000` on the surface. `language_model/`'s own
`config.json` reads `"architectures": ["Qwen3ForCausalLM"]`,
`"model_type": "qwen3"`, standard fields throughout - a genuine Qwen3
re-export, the one this port actually loads. An earlier working
assumption (recorded in this ledger's own Phase 1 entry before this was
checked) had the two backwards; corrected in place rather than left
stale. `crates/qwen3::import::hf_source` + `Qwen::new_shard_i8` stream it
one tensor at a time, quantized to int8 (DP4A) as it goes - int8 is not
merely smaller-and-nice-to-have for an 8B model on a machine with no
discrete GPU and ~21 GB usable RAM; it is what makes the model resident
at all (fp32 would need ~2x the checkpoint's own bf16 size).

Real-weight parity: a single REAL decoder layer (layer 0), streamed via a
1-layer `model::Shard` (never the whole 36-layer stack), compared against
`transformers.Qwen3DecoderLayer` loaded with the SAME real weights
(`tools/goldens/minimaxmusic3_global_llm_dump_reference.py` - no
`diffusers` PR dependency, plain `transformers>=4.51`, already in
`requirements.txt`) - the same "real weights, too big to load whole"
discipline `qwen35_dump_real_layer_reference.py` established for an
unrelated model in this repo. Cosine 1.000000000, first try, confirming
both that `language_model/` really is what its `config.json` claims and
that the streaming import is correct.

The training objective this milestone adds: ordinary next-token
cross-entropy restricted to audio-code target positions
(`global_llm::audio_code_batch`, `model::Batch::LmWeighted` - weight 0 on
every position whose target is still inside the prompt, 1 once the
target is the first audio-code token and onward), reusing
`crates/qwen3`'s own already-gradchecked weighted-CE gradient rather than
a new loss kernel. Proven trainable at `QwenConfig::tiny()` scale (plain
AdamW, 300 steps, loss collapses to under 10% of its start) - the real
8B checkpoint is inference-only here (`new_shard_i8` has no backward),
matching every other real-scale exercise this port records as a
hardware-bound gap rather than attempting.

Prompt assembly text, the CFG-guided AR sampling loop, and the
depth-decoder feedback loop are M7 (pipeline glue) scope - this module
owns only the special-token/offset constants both milestones read from
one place (`AUDIO_CODE_OFFSET`, `AUDIO_END_TOKEN_ID`, `AUDIO_CFG_TOKEN_ID`,
the prompt template's structure tokens), confirmed against the reference
`diffusers` PR's own `MiniMaxMusic3TextEncoderStep`/
`MiniMaxMusic3SemanticGenerationStep` classes.

## Not yet done

- [ ] Joint generator+discriminator training against the real vocoder
      weights (composing `train::Trainer` and `discriminator::` into one
      loop) - the mechanism exists in both directions, the composition
      does not
- [ ] Multi-resolution discriminator (several `(n_fft, hop)` STFT
      settings, summed) - the single-resolution version generalizes
      directly but this has not been exercised
- [ ] Pipeline: prompt assembly, two-axis CFG (AR logits + DiT
      zero-conditioning), chunk windowing/overlap-splice, vocoder
      crop-and-stitch - one real short end-to-end WAV on this machine
- [ ] Serving contract: capability manifest, residency adapter, CLI verb,
      D-Bus, a runnable example

## Recorded gaps (expected, not yet reached)

- No full 5-minute generation on this machine (no discrete GPU, ~21 GB
  usable RAM) - only a short single-chunk generation is exercised here.
- No NPU export path planned in the initial port.
- No real multi-device execution for the DiT's `model::Shardable` slice
  (this machine has no discrete GPU at all) - only single-device
  structural validation (weight-subset loading, single/two-stage
  parity) is possible here; genuine multi-card agreement is unverified.
- No backward pass through the DiT's pipeline-sharded slice
  (`dit_shard::DitStage`) - an explicit, documented scope decision
  (matching `ltxv::LtxDit`'s own precedent), not an oversight: this
  crate's real DiT training path is the single-device `dit_train::Trainer`.
- No real-scale (8B, `language_model/`'s real weights) audio-code CE
  training run - the real checkpoint is imported int8/inference-only
  (`Qwen::new_shard_i8` has no backward); the training objective itself is
  proven correct at `QwenConfig::tiny()` scale only, same hardware-bound
  reasoning as every other real-scale training gap this ledger records.
