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
one tensor at a time, requesting int8 as it goes.

**Correction (Phase 10): int8 does not actually shrink this on either of
this machine's backends** - see Phase 10's own entry for the measured
numbers and root cause. This paragraph originally claimed int8 "is what
makes the model resident at all"; that was an untested assumption,
disproven the first time this port actually tried whole-model residency
(Phase 10). Left here, struck through in spirit rather than deleted, so
the ledger shows the assumption AND its correction rather than quietly
rewriting history.

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

## Phase 10: pipeline glue (M7)

`crates/minimaxmusic3::{pipeline, denoise, stitch}` plus
`global_llm::assemble_prompt` - every piece of orchestration between the
five components, none of it new model math (all of it composes forwards
already landed in Phases 2-9):

- `global_llm::assemble_prompt` - `MiniMaxMusic3PromptStep` ported: builds
  `<|im_start|><|caption_start|>{clean_caption}<|caption_end|>
  <|lyrics_start|>{normalize_lyrics}<|lyrics_end|><|im_end|><|audio_start|>`,
  tokenizes once via `QwenBpe::encode` (already special-token-aware), then
  derives the CFG-unconditional variant by replacing `ids[1:-2]` with
  `AUDIO_CFG_TOKEN_ID`. `clean_caption`/`normalize_lyrics` (markdown-
  stripping and lyrics-tag normalization, ported byte-for-byte from the
  Python reference, no `regex` crate available so hand-written char
  scanning) landed earlier alongside the token contract; both are unit-
  tested against literal Python reference output (19 + 11 assertions).
- `pipeline::generate_frames` - the CFG-guided AR sampling loop: two
  `qwen3::Qwen` instances (conditional/unconditional) stepped in
  lockstep, top-k-restricted CFG over the LLM's own logits (scale 1.5,
  top-k 50 twice - once to threshold against the conditional branch's own
  candidates, once inside `_sample_top_k`'s own re-restriction), and a
  second, independently-discovered CFG axis inside the depth decoder (the
  reference runs `_generate_depth_codes` on BOTH branches too - easy to
  miss on a first read of `denoise.py` alone, since that file only shows
  the DiT's own CFG).
- `denoise::denoise_chunk` - `ChunkConditionStep`/`ChunkPrepareLatentsStep`/
  `ChunkSetTimestepsStep`/`ChunkDenoiseInner`/`ChunkUpdateStep` ported: 200-
  frame chunks, 100-frame hop, a `FlowMatchEulerScheduler` run with
  `invert_sigmas` (the DiT's own 0=noise/1=data convention), CFG via a
  zeroed condition tensor (the DiT's OWN conditioning, not a second full
  model - distinct from the AR stage's axis above), and a 172-latent
  overlap blended into every Euler step via `ChunkState`, not just spliced
  once at the end. Initial noise draws through `data::rng::Rng::
  next_gaussian` (`Lcg` has no Gaussian sampler; `data::rng`'s own doc
  says never hand-roll a fresh Box-Muller copy) - the first use of `Rng`
  in this crate, alongside `Lcg` everywhere else.
- `stitch::Stitcher` - `decoders.py`'s crop-and-stitch: decode each
  chunk's full latent span through the vocoder, crop
  `CROP_LEFT_LATENT`/`CROP_RIGHT_LATENT` off the shared edges (skipped on
  the song's first/last chunk), concatenate.
- `audio::wav::encode_multi`/`write_multi` - the stereo WAV write path
  this crate's own plan named as a gap up front (the existing writer was
  mono-only); planar `[channels, samples]` in, no interleaving required
  from the caller (matches the vocoder's own output layout). Mono
  `encode`/`write` now delegate to this (byte-identical output, pinned by
  a regression test).

All of the above is unit/structurally tested (no numerical reference
exists for a multi-stage composition like this - same reasoning
`pipeline::generate_frames`'s own tests already used) and passes.
`crates/minimaxmusic3/tests/e2e_short_generation.rs` wires all five real
components together end to end (AR stage -> denoise -> stitch -> WAV),
gated behind all six `BRAIN_MINIMAXMUSIC3_*` env vars, with an explicit
sequential-stage RAM discipline (the AR stage's two Global LLM instances
and the depth decoder are dropped, out of scope, before the DiT loads;
the DiT is dropped before the vocoder loads). It compiles, clippy-clean,
and skips cleanly when the env vars are unset - but **could not be
validated end-to-end on this machine**; see the next entry.

### Why the real short e2e WAV was not reached here

Attempted with all six real checkpoints present (under `resources/
minimax-music3/`, per this port's own resources layout). Two
independent, measured, pre-existing infrastructure limits block
whole-8B-model residency on
this machine, neither a defect in this port:

1. **CPU-JIT backend (`BRAIN_DEVICE=cpu`): int8 silently promotes to
   fp32.** `backend_api::DType::promote` (workspace-wide, not this crate)
   demotes `I8`/`Q4` requests to `F32` whenever
   `NumericSupport::int8_dot` is `false` - `crates/backend-cpu`'s own
   `caps()` reports `int8_dot: false` unconditionally today (no backend
   in this workspace executes real int8 compute yet; that field's own
   doc already said so). So `Qwen::new_shard_i8` on this backend actually
   allocates the FULL fp32 8B model (~32 GB: ~6.4B per-layer-linear
   params at 4 bytes + ~1.64B embedding/lm_head params at 4 bytes),
   not the ~13 GB int8 one its name promises. Measured directly: a
   single Global LLM instance's RSS climbs from near-zero past 30 GB and
   is OOM-killed (confirmed via `/sys/fs/cgroup/memory.events`'s
   `oom_kill` counter incrementing) on this machine's ~26 GB available
   RAM - before even reaching the SECOND (unconditional-branch) instance
   `pipeline::generate_frames` needs.
2. **This machine's real GPU (an Intel integrated Vulkan device, not a
   discrete card) hits a separate, harder limit first if int8 execution
   ever lands there:** `tok.weight`/`lm_head.weight`
   (`[200000, 4096]` fp32, ~3.28 GB each - excluded from the int8 tier
   since embeddings aren't a `Q8::LINEARS` projection) exceed that
   device's own `max_buffer_size` (2047 MiB), so even a WORKING int8
   compute path would still fail to place the embedding/head tables as
   single buffers on this specific GPU.

Both are corrected in `global_llm::import`'s own doc (previously claimed
int8 "is what makes an 8B model resident on this machine's CPU backend
possible at all" - disproven by this measurement, doc fixed in place
rather than left stale). Fixing either root cause (a real backend-cpu
int8/VNNI compute path, or >2 GB-buffer-spanning tensor placement in
`gpu_core`/`qwen3`) is a substantial change to shared, heavily-used
infrastructure well outside this port's own scope - recorded here rather
than attempted. `tests/e2e_short_generation.rs` is real, correct, ready
code for whenever a future session has more RAM, a real int8-capable CPU
path, or a discrete GPU: run it with all six `BRAIN_MINIMAXMUSIC3_*` env
vars set and it either produces a WAV or fails somewhere new worth
investigating.

## Phase 11: serving contract (M8)

`crates/minimaxmusic3::{caps, generate}` + `crates/cli::resident_minimaxmusic3`:

- `generate::generate` is the ONE implementation of "run the whole
  pipeline" - factored out of what `tests/e2e_short_generation.rs`
  previously inlined, and generalized from that test's single-chunk
  shortcut to the real `denoise::chunk_starts`/`ChunkState` multi-chunk
  loop (still respecting the sequential-stage RAM discipline: every
  chunk's latents are produced before the DiT drops, then every chunk
  decodes through the vocoder stage - never interleaved, which would
  need both resident at once). `generate::Paths::from_env` reads the same
  six `BRAIN_MINIMAXMUSIC3_*` vars `crates/arch`'s own `weights_env` table
  names. `duration_seconds` converts to an AR frame cap via the AR
  stage's own 25 Hz frame rate - confirmed by construction (the condition
  encoder's resample and the vocoder's upsample keep 1 AR frame = 1/25 s
  of eventual audio end to end), not just asserted.
- `caps::MinimaxMusic3Provider`/`manifest`/`generate_spec` follow the
  `wan::caps`/`qwen3tts::caps` pattern (generation-only params, paths from
  the environment) but are STATELESS like `qwen3tts::caps::TtsProvider`,
  not `wan`'s hot-DiT cache - nothing here is worth keeping warm, since
  the whole checkpoint does not fit in RAM even once on this machine (see
  Phase 10). The `audio` output blob is a complete WAV
  (`audio::wav::encode_multi`, `meta.format == "wav"`), not headerless
  PCM, since `caps_cli::save_blob`'s raw-PCM arm is mono-only and this
  model is stereo.
- `resident_minimaxmusic3::MinimaxMusic3Resident` mirrors
  `resident_tts.rs`'s load-per-call shape. `estimate()` budgets the
  LARGER of the AR stage (two Global LLM loads, sized at 2x the on-disk
  checkpoint bytes per instance to reflect the fp32-promotion reality
  Phase 10 measured, plus the depth decoder) and the denoise stage (DiT +
  vocoder + condition encoder), never their sum, since the two stages are
  never resident together. `activate()` fails fast on missing
  directories; the returned `Instance::run` dispatches straight through
  `caps::generate_action` - one implementation of param decoding,
  generation, and outcome shaping shared by the direct (`brain do`) and
  resident (D-Bus/scheduler) paths.
- CLI/registry wiring: one `catalog.rs` `ModelEntry` (manifest/provider/
  resident, the `qwen3tts` shape), one `resolve.rs` `ARCH_TO_MODEL` row
  (the generic capability-dispatched path - no dedicated `_cli.rs` needed
  for a single-action model). No `crates/dbus` changes: that crate's
  D-Bus surface is entirely generic over whatever `ResidentModel` the
  CLI's `build_executor` registers (which already merges
  `catalog::residents()`), so the catalog+resident wiring is the model's
  complete D-Bus reachability story. Verified manually: `brain caps
  brain/minimaxmusic3` lists the `generate` action with its full param
  schema; `brain minimaxmusic3 generate --lyrics ... --caption ...` (no
  env vars set) fails with the exact, actionable "set
  BRAIN_MINIMAXMUSIC3_LM" message, not a generic/unknown-model error -
  the whole resolve -> catalog -> provider -> Paths::from_env chain is
  live end to end, short of the RAM-blocked real run itself.
- `examples/musicgen/generate_song.py` + README: the `videogen`/`imagegen`
  example convention (`BrainDBus`, `skip()` when unserved, only send
  params the caller chose), simplified by the complete-WAV blob
  convention above (no client-side encode step). The README states
  plainly that this has NOT been run to completion in this repo's own dev
  environment - consistent with this port's own honesty conventions,
  not a demonstration with invented numbers.

All of `catalog.rs`'s and `resolve.rs`'s own invariant tests pass with
the new entries in place (`catalog_ids_are_unique`,
`every_listed_model_is_constructible_by_name`,
`catalog_and_residency_do_not_overlap`,
`every_arch_to_model_id_is_a_real_registry_entry`,
`arch_handlers_and_arch_to_model_do_not_overlap`); the whole-workspace
`clippy-gate.sh`, `check-arch-names.sh`, and `check-linear-history.sh`
gates are all green with this milestone's changes in place.

## Phase 12: validation on real hardware - what Phases 10/11 could not reach

Every blocker Phases 10 and 11 record belongs to the machine this port was
BUILT on, not to this port. The validation machine is a different box: **two
Tesla P40s** (GP102, 24 GB each, native DP4A), a 48-thread dual E5-2690 v3,
and **184 GB RAM**. Read every "on this machine" claim above as historical.

Corrections, each measured or read from source rather than assumed:

1. **"No backend in this workspace executes real int8 compute yet"
   (Phase 10, and copied into `global_llm::import`'s own doc) is FALSE, and
   was false when written.** It holds only for `backend-cpu`.
   `backend-wgpu` reports `int8_dot: true` unconditionally
   (`crates/backend-wgpu/src/lib.rs`), and `backend-vulkan` queries the real
   `VK_KHR_shader_integer_dot_product`
   `4x8BitPackedSignedAccelerated` property, which GP102 reports as
   accelerated. `backend-api`'s own test asserts
   `DType::I8.promote(&wgpu_like) == DType::I8`. So `Qwen::new_shard_i8`
   genuinely runs int8 on a P40, and the 8B LM is ~9 GB there, not 32 GB.
   The same false claim also sits in `backend-api`'s `promote` doc comment
   and in `ltxv::int8`'s module doc - all three want retiring together.
2. **The 2047 MiB single-buffer limit is REAL and survives the hardware
   change.** It is not an Intel-iGPU quirk: wgpu clamps
   `max_storage_buffer_binding_size` to `i32::MAX` on every card, so the
   `[200000, 4096]` fp32 `tok.weight`/`lm_head.weight` (3.28 GB) still
   cannot be bound as one buffer on a P40. `model::block::vocab_tiles_on` +
   `Gpu::step_sliced` is the existing answer and `qwen3`'s own embed path
   already uses it; the gap is a decode-path head that does.
3. **There IS now an official upstream implementation**, contradicting this
   ledger's own opening paragraph and `requirements.txt`'s "NOT
   pip-installable" block. MiniMax-Music3 shipped in **released diffusers
   0.40.0** - the four `MiniMaxMusic3*` model classes AND a
   `diffusers.modular_pipelines.minimax_music3` modular pipeline - so the
   reference is a stable pip install, not a git commit. There is also an
   official repo (`MiniMax-AI/MiniMax-Music3`) carrying an official
   SGLang-Omni serving path, a documented structured-caption format, and a
   reference output WAV generated with this checkpoint (`assets/
   minimax_ttm.wav`) - the end-to-end oracle this port never had.
   The official README's "32 kHz" is loose: `vocoder/config.json` says
   `sampling_rate: 44100`, which is what this port implements.

### Hardware facts that bound every optimisation here

Measured or read from vendor documentation, not extrapolated:

- **AVX2 int8 on this CPU is worth nothing.** `VPMADDUBSW` and `VPMADDWD`
  are both port-0-only on Haswell: 16 int8 MAC/cycle against fp32 FMA's 16.
  Benchmarked on this box at a **1.01x** int8/fp32 ratio. The widely-quoted
  1.3-2x is a cache effect from 4x smaller weights, not arithmetic, and the
  real 4x needs VNNI, which this Xeon does not have (nor AVX-512 at all).
  With the CPU at 7.7% of the machine's fp32 and 2.1% of its int8, an AVX2
  int8 GEMM path is not worth building - the CPU's role is to overlap GPU
  work, not to do GEMMs. `backend-cpu`'s `int8_dot: false` is therefore
  HONEST on this hardware, not a gap.
- **DP4A on a P40 is ~1.5-1.8x over fp32, not 4x.** NVIDIA's own Pascal
  tuning guide says DP4A has "throughput equal to that of FP32
  arithmetic"; the 4x is op-accounting (8 ops per DP4A vs 2 per FMA).
  Real DP4A kernels measure 29-34% of the nominal 47 TOPS. Worth doing,
  but a kernel below ~25% of peak merely ties a good fp32 SGEMM.
- **fp16 is not merely slow on Pascal - `SHADER_F16` is not exposed at
  all** under Vulkan on GP102. f16 can be a storage tier here, never a
  compute tier. Same for q4: there is no q4 dot instruction anywhere in
  this workspace (`matmul_q4_*` unpack nibbles by hand), so q4 rides
  int8's activations. Both are recorded as PLANNED tiers rather than
  optimised ones.

### Where the time actually goes (budget, not a hypothesis)

For a 180 s song (4500 AR frames, 45 denoise windows) the **flow DiT is
65-80% of wall-clock**: 45 windows x 30 Euler steps x 2 CFG branches, and
its GEMMs sit at ~85 FLOP/byte, well past a P40's ~34 ridge point, so it is
compute-bound. The 8B Global LM is bandwidth-bound and roughly a fifth of
the budget; the 123M vocoder is negligible. **Optimise the DiT first** - an
earlier working assumption in this session that the AR stage's host LM head
dominated was wrong, and is recorded here because `.agents/rules/kernels.md`
§F exists precisely for this failure mode.

Two structural facts, read from this crate's own code, that bound what is
possible:

- **The Global LM consumes all 8 codebooks of a frame**
  (`pipeline::embed_audio_frame` sums every codebook's embedding before
  `Qwen::step_embed`), so the AR stage is strictly serial and the depth
  decoder is NOT batchable across frames.
- **The denoise windows are sequentially dependent**
  (`denoise::ChunkState::previous_latent` chains each chunk to the last),
  so the two cards can PIPELINE stages but cannot data-parallelise windows.

### Landed in this phase

- `gpu_core::Gpu::open` - one device-token to handle mapping, replacing 17
  private copies that all mangled an indexed card into the ambient
  selection.
- `dit::Resident` - DiT weights and RoPE tables upload once per chunk, not
  once per forward. The denoise loop evaluated the DiT `2 * steps` times
  per chunk and each call re-sent the whole 36-block stack (~9.7 GB at real
  dims) for byte-identical weights. Gated bit-for-bit.
- The DiT and vocoder honour `--device` instead of `Gpu::new_cpu`. They
  were pinned to the one device that cannot use a GPU, silently making
  `--device gpu` a no-op for most of this pipeline's cost.
- `crate::ProgressSink` - `generate` actually emits progress. It had
  declared `.streaming()` since it landed while every layer dropped the
  callback, which also made the model unbenchmarkable, since `brain perf`
  builds its whole timeline from those callbacks.

## Phase 13: the first real song, and the five defects between here and it

The port produced audio for the first time: 44.1 kHz stereo, all five real
components, two P40s. The parity ladder closed first - every component at
cosine 1.000000000 against **released diffusers 0.40.0** (and
`transformers` for the Global LLM), with the DiT and vocoder additionally
checked on a real GPU rather than only the CPU JIT:

| component | params | backend | cosine | max_abs |
|---|---|---|---|---|
| condition encoder | 25M | CPU JIT | 1.000000000 | 3e-6 |
| RVQ depth decoder | 0.65B | CPU JIT | 1.000000000 | 1.6e-5 |
| vocoder (DAV) | 123M | CPU JIT + P40 | 1.000000000 | 1e-6 |
| flow-matching DiT | 2.4B | P40 | 1.000000000 | 1.3e-5 |
| Global LLM (Qwen3-8B) | 8B | P40 | 1.000000000 | 1e-6 |

**None of those tests had ever run in this checkout.** `testdata/golden/
minimaxmusic3/` did not exist, so all nine parity tests skipped - and cargo
reports a skipped test as a pass, so the suite was green while proving
nothing. Every "cosine 1.0" claim in this ledger was, locally,
unreproducible. That is the single most important lesson here and it
generalizes: a gate whose fixture is absent is indistinguishable from a
gate that passed.

### The five defects, and why the existing tests could not see any of them

1. **The e2e gate tested a path no caller uses.** It re-composed the five
   stages inline instead of calling `generate::generate`, with its own LM
   instances and its own `Gpu::new_cpu` handles - so it exercised neither
   the cross-card placement nor the device selection, and could neither
   reproduce a failure the shipped path avoids nor catch one it hits.
   `generate`'s own module doc had claimed the test called it directly.
2. **The AR stage was built in the wrong SHAPE.** `new_shard_dt` passed
   `decode_only: false`, so an inference-only build allocated the batched
   forward: activations at `n = b*t`, `n_heads*ctx^2` scores/probs, an
   `n*vocab` logits buffer, and backward scratch (the `bwd` closure is
   dummied by `decode_only`, not by `train`). The AR loop only ever calls
   `prefill`/`step_embed`. int8 got the linears to a MEASURED 6.95 GB and
   the unused scratch spent the saving straight back, taking one instance
   past 24 GB. `Qwen::new_shard_dt_decode` now gives the two axes - weight
   tier and activation shape - together.
3. **The decode path bound a 3.28 GB buffer whole.** The batched forward
   vocab-tiles `tok.weight` via `step_sliced`; the decode path dispatched
   untiled `EMBED` against the entire `[200000, 4096]` table
   (`range 3276800000 exceeds limit 2147483644`). That limit is wgpu's
   `i32::MAX` clamp, present on EVERY card - **not** the integrated-GPU
   quirk Phase 10 assumed.
4. **The DiT ran every projection through the REFERENCE matmul.** It
   registered only `("matmul", kernels::MATMUL)` - `@opt 2`, one thread per
   output element - while `ltxv`/`wan` dispatch the register-tiled `@opt 5`
   `matmul_reg3` through `model::block::gemm_variant`. Thirty-six blocks,
   six projections each, at a fraction of a percent of the card's fp32
   peak. Exactly the class `.agents/rules/kernels.md` names as this repo's
   most expensive.
5. **Four `Gpu::storage` calls asked for 4x the memory they needed**
   (`elements * 4` where the API takes WORDS), plus 23 more in the training
   paths. Invisible at `::tiny()` dims, fatal at a real 689-latent chunk.

### Measured

    denoise, per Euler step   37.4 s  ->  1.21 s   (~31x, defect 4)
    end to end, 1 s of audio  447.6 s ->  153.5 s
    depth decoder, per frame  3.77 s  ->  1.01 s   (~3.75x, Phase 12 KV cache)

Correctness held throughout: `dit_parity` stayed at cosine 1.000000000 and
the rendered audio is **bit-identical** between the naive and fast GEMM
paths (waveform cosine 1.000000000, max abs diff 0.000000).

### Correction: the stated reason for the vocoder OOM was WRONG

`stage_devices`'s doc comment and commit ee32a5f1 both claim "wgpu does not
return a device's VRAM to the driver when the handle drops". **That is
false.** It was an inference reached for to explain an out-of-memory, never
measured, and it is contradicted by this session's own numbers as well as
by the source:

- `wgpu-core/src/resource.rs:473` - `Buffer::drop` calls `destroy_buffer`
  IMMEDIATELY; there is no deferral to a poll.
- `wgpu-core/src/device/resource.rs:307` - `Device::drop` tears its
  resources down, and each buffer frees itself as its refcount falls.
- `wgpu-hal/src/vulkan/device.rs:1045` - allocations are
  `AllocationScheme::GpuAllocatorManaged`, i.e. SUB-allocated from larger
  blocks. A freed buffer returns its sub-allocation to the allocator's
  pool, reusable within that device; the blocks themselves go back when the
  allocator dies with the Device.
- Measured here: two vocoder chunks back to back peak at 12.26 GB, not
  24.5 - so buffers are plainly reclaimed within one device. And in the
  split-device run **gpu0 falls to 15 MiB** while the vocoder runs on
  gpu1 - so the DiT's device released its VRAM on drop.

Both measurements were in hand when the wrong explanation was written; they
simply were not joined up. The FIX (separate cards) is real and the run is
green, but the mechanism recorded for it is not, and a wrong durable
finding is worse than none: the next reader would design around a
constraint that does not exist.

**The actual cause of the same-card OOM is therefore still unknown.** The
open candidates are allocator-pool fragmentation across the stage boundary,
or something retaining the DiT `Gpu` past its block scope so the two
devices genuinely overlap. The probe that settles it: allocate ~9 GB on a
device, drop it, then immediately build a second device on the SAME card
and allocate ~12 GB, watching `nvidia-smi` across the boundary. Not yet run
- both cards are busy with a generation - and until it is, `stage_devices`
should be read as "a fix whose mechanism is unconfirmed", not as evidence
about wgpu.

### Two hypotheses tested and rejected

Recorded because the rejections cost as much as the fixes, and because
`.agents/rules/kernels.md` §E exists for exactly this:

- **Submitting the vocoder's tape per upsample stage** to release
  intermediates earlier: 12264 MiB vs 12261 MiB. No improvement - wgpu does
  not return the memory on submit either. Reverted rather than kept.
- **A waveform crossfade at the chunk seam** (CosyVoice 2 and F5-TTS both
  use 150-160 ms). Not needed: at the 4.0 s boundary the max
  sample-to-sample step is 0.0031 against a 99.9th percentile of 0.0635
  elsewhere - the seam is SMOOTHER than ordinary audio, because the
  172-latent overlap is replaced in the LATENT domain before the vocoder
  sees it.

An earlier assumption in this session that the AR stage's host LM head
dominated wall-clock was also wrong: the DiT is 65-80% of it, as the
budget arithmetic predicted and the profile confirmed.

### Where the time goes now, and what is left

For a 4-minute track (6000 AR frames, ~59 chunks): AR ~1.9 h, denoise
~1.3 h at 8 steps. **The AR stage is now the bottleneck, and ~90% of it is
the depth decoder running as HOST math** - its "nothing to parallelize"
justification reasons about sequence length (<=8 positions), not work; at
real dims it is 4096x6144 GEMMs 16 times per frame. Moving it to the device
is the same fix as defect 4 and is the next lever, followed by the
front-loaded sigma schedule the literature reports taking flow-matching
models from 30 steps to 7-16 without measured quality loss.

## Not yet done

- [ ] Joint generator+discriminator training against the real vocoder
      weights (composing `train::Trainer` and `discriminator::` into one
      loop) - the mechanism exists in both directions, the composition
      does not
- [ ] Multi-resolution discriminator (several `(n_fft, hop)` STFT
      settings, summed) - the single-resolution version generalizes
      directly but this has not been exercised
- [ ] The real short end-to-end WAV gate itself (`tests/
      e2e_short_generation.rs` exists and is correct; blocked on THIS
      machine by the int8-promotion/buffer-size limits Phase 10 records)
- [ ] A CLI verb for per-component training (LoRA/full fine-tune) - the
      library functions are real and gradchecked (Phases 3-9), but
      `caps::MinimaxMusic3Provider` only exposes `generate`; training
      stays a Rust-API-only surface for this model, matching how the
      "Not yet done"/Support table entries elsewhere in this repo
      distinguish library-level from CLI-reachable capability.

## Recorded gaps (expected, not yet reached)

- No full 5-minute generation on this machine - superseded by the more
  precise Phase 10/11 diagnosis below (this isn't just "no discrete GPU":
  even a SHORT generation cannot complete here). `generate::generate`
  itself is NOT single-chunk-only (it loops over the real
  `denoise::chunk_starts`, so a long song is the general case, not a
  special one) - what's actually unexercised is any real run long enough
  to invoke that loop more than once.
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
- No real short end-to-end WAV on this machine (Phase 10's own gate,
  `tests/e2e_short_generation.rs`, is implemented, correct, and skips
  cleanly when unset - it is BLOCKED, not unwritten): whole-8B-model
  residency fails on both of this machine's backends - CPU-JIT's `int8`
  request silently promotes to fp32 (`backend-cpu`'s `caps().numeric.
  int8_dot` is `false`, workspace-wide, so `DType::promote` demotes to
  `F32` - no backend here executes real int8 compute yet), and this
  machine's own Vulkan device (an Intel integrated GPU, not discrete)
  caps single buffers at 2047 MiB, smaller than the ~3.28 GB
  `tok.weight`/`lm_head.weight` tensors regardless of dtype. Measured via
  `/sys/fs/cgroup/memory.events`'s `oom_kill` counter incrementing on a
  single Global LLM instance import alone. See Phase 10's own writeup for
  the full diagnosis; fixing either backend limit is out of this port's
  scope.
