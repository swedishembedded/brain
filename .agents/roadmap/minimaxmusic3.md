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
~1.2 h at the reference's 30 steps (59 x 30 x 2 forwards x 1.21 s), or
~19 min at 8. An earlier revision of this line said "~1.3 h at 8 steps",
which was the 30-step number mislabelled. **The AR stage is now the bottleneck, and ~90% of it is
the depth decoder running as HOST math** - its "nothing to parallelize"
justification reasons about sequence length (<=8 positions), not work; at
real dims it is 4096x6144 GEMMs 16 times per frame. Moving it to the device
is the same fix as defect 4 and is the next lever, followed by the
front-loaded sigma schedule the literature reports taking flow-matching
models from 30 steps to 7-16 without measured quality loss.

## Phase 15: the DiT's attention was 75% of its device time at 0.5% of roof

`mm3_bench dit 689` - the instrument Phase 14 added - put three quarters of
this DiT's device time in three kernels that were all flagged DEFECT against
its 35%-of-roof floor. Measured on a free Tesla P40 at `DitConfig::real()`,
689 latents / 690 rows, against that card's own measured roofline (10517
GFLOP/s, 287.5 GB/s DRAM, ridge 36.6 FLOP/byte):

```
attn_scores_bidir     1817.80 ms   36 calls   50.9%   50.494 ms/call    0.5% of its memory roof
matmul_reg3            753.65 ms  252 calls   21.1%    2.991 ms/call   42.1% of fp32 peak
attn_softmax_bidir     686.35 ms   36 calls   19.2%   19.065 ms/call    2.2% of its memory roof
attn_apply_bidir       186.44 ms   36 calls    5.2%    5.179 ms/call    4.9% of its memory roof
WHOLE PASS            4383.49 ms  832 dispatches      device-time sum 3583.6 ms
```

The GEMMs beside them were already at 42% of this card's fp32 peak, i.e. at
the ceiling `matmul_reg3` reaches anywhere in this workspace - §F.2's test says
that is structural and attention is not.

The fix was not a kernel. `model::block` has carried the four-rung
`flash_attn_bidir{,_split,_reg,_reg2}` family and its `flash_bidir_variant`
selector for some time; `dit::PIPELINES` registered NONE of them. This is
verbatim the defect class §A opens with and §F.3 asks about first - "a fast
kernel nobody knew about" - and it is the SECOND instance of it in this one
file, after the `matmul_reg3` registration recorded in `PIPELINES`' own
comment.

The shape contract was already met, and was verified rather than assumed
before the switch: `block_fwd` already packs q/k/v into one `[rows, 3*inner]`
slab through three `kv_expand` copies, so `stride = 3*inner` and
`q_off/k_off/v_off = 0/inner/2*inner` are exactly what `flash_bidir_step`
documents; `d_model = inner = heads*head_dim = 2048`; `head_dim = 64`, inside
the family's 128 limit; `bsz = 1`; and both arms apply the same
`inverseSqrt(head_dim)` scale and write the same `[rows, inner]` row-major
context, so `to_out` and the residual cannot tell them apart. Nothing about
the dispatch sequence above the attention changed.

After, same binary, same card, same shape:

```
matmul_reg3            752.26 ms  252 calls   77.5%    2.985 ms/call   42.1% of fp32 peak
flash_attn_bidir_reg2   93.66 ms   36 calls    9.6%    2.602 ms/call   14.6% of fp32 peak
layernorm               49.00 ms   72 calls    5.0%    0.681 ms/call    5.8% of its memory roof
conv1d                  35.42 ms    2 calls    3.6%
bias_add                17.97 ms  108 calls    1.9%
WHOLE PASS            1537.21 ms  760 dispatches      device-time sum 970.8 ms
```

* The attention itself: **2690.6 ms -> 93.7 ms, 28.7x**, and the three
  dispatches per block became one.
* Whole forward: **4383 -> 1537 ms wall clock, 2.85x**; device-time sum
  **3583.6 -> 970.8 ms, 3.69x**. A 30-step chunk of DiT alone drops from
  ~263 s to ~92 s per CFG branch pair.
* The selector picked `reg2` on this card (`max_workgroup_size >= 256`,
  `workgroup_mem_bytes >= 49152`), with no shape term in the decision.
* The pass is now GEMM-dominated at 77.5%, and those GEMMs are at the card's
  ceiling - §F.9's "the bottleneck moves" has moved it somewhere there is no
  obvious next lever inside this kernel set.

`flash_attn_bidir_reg2` at 14.6% of fp32 peak still trips the bench's own 30%
compute floor, and that is worth stating rather than hiding: `block.rs`'s own
measured ladder reaches 38.0% at Wan's T=14040, and this model runs at T=690
with `head_dim` 64 against that kernel's 128-wide compile-time tile, so half
of every tile is zero-fill. A `head_dim`-specialised template is the honest
next step for it - but it can return at most 93.7 ms of a 1537 ms pass, which
is §E's "check what fraction of the PASS it can possibly return" saying not
yet.

**Not bit-identical, unlike everything else landed on this model so far.**
Flash attention's online softmax reassociates the sum, so the output moves in
the last bits by construction. What is gated is `dit_parity` against the
real-weight diffusers golden, on BOTH backends (§F.4):

| rung | backend | cosine | max_abs | rel_l2 |
|---|---|---|---|---|
| real | P40 (flash) | 1.000000000 | 9.537e-6 | 2.550e-6 (was 2.558e-6) |
| tiny | P40 (flash) | 1.000000000 | 1.565e-7 | 2.533e-7 (was 2.553e-7) |
| real | `BRAIN_DEVICE=cpu` (trio) | 1.000000000 | 9.537e-6 | 2.514e-6 |
| tiny | `BRAIN_DEVICE=cpu` (trio) | 1.000000000 | 1.453e-7 | 2.254e-7 |

Both moved rungs moved TOWARDS the reference, which is the expected direction
for an online-softmax reassociation and not something cosine alone would have
shown - the `rel_l2` assertion beside it is what makes the table readable as
evidence.

The fallback is the branch the CPU JIT takes, gated on the same QUERIED
`workgroup_reductions` capability `linear_step` already reads, so
`BRAIN_DEVICE=cpu`, `make gradcheck` and the whole backward (`bidir_bwd`,
`dit_train`) keep the materialized trio unchanged - the flash family is
forward-only and the backward reads the `probs` slab the flash path never
writes.

VRAM, measured with `nvidia-smi` sampled once a second across a whole run on
an otherwise idle card: peak **9759 MiB -> 9553 MiB**. Block weights are 9430
MiB of that in both, so the per-forward transient scratch is **329 MiB -> 123
MiB, a 206 MiB / 63% cut** - the `[32, 690, 690]` `scores` and `probs` slabs
(60.9 MB each) are never materialised.

## Phase 16: the second card stops being idle during the denoise stage

`denoise::denoise_chunk` evaluated the DiT twice per Euler step - once
against this chunk's condition, once against a zeroed copy of it
(`denoise.py`'s `zeros_like(condition)`) - one after the other on ONE card,
while `generate::stage_devices` handed the vocoder stage the other card and
the two stages never overlap. The second card was therefore at 0% for the
whole denoise stage, which Phase 13 measures as ~1.2 h of a four-minute
track. The two forwards share `latents`, the timestep and `length`, share no
intermediate value, and are read only by the host-side fold
`u + (c - u) * GUIDANCE_SCALE` after both return - which is exactly the shape
two cards want.

Ported from `crates/ltxv`'s own solution rather than invented.
`minimaxmusic3::devplan` is `ltxv::devplan` minus the text-encoder placement:
`DevicePlan::{Single,Split,Auto}`, `Placement::cfg_is_parallel`, `on_gpu`,
the base card read from `current_gpu()` so a run scoped by the residency
executor keeps its assigned card as `cond` and borrows only the other one,
and a `BRAIN_MINIMAXMUSIC3_CFG_PARALLEL=0` opt-out matching
`BRAIN_LTXV_CFG_PARALLEL`'s spelling.

### Landed in this phase

- `denoise::CfgDevices` - one `Gpu` per card the placement names, opened
  ONCE per generation (not per chunk, never per step) and dropped before the
  vocoder stage loads, so the vocoder still gets a card with nothing of the
  denoise stage on it.
- `denoise::ChunkResidents` - one `dit::Resident` PER CARD, built once per
  chunk and reused across all `2 * steps` evaluations. The two ~9.7 GB
  uploads run concurrently for the same reason the forwards do; serialising
  them would put a second full weight upload on every chunk's critical path
  and hand back a slice of what the concurrent forwards just won.
- The progress callback stays on the orchestrating thread, called after the
  join. `crate::ProgressSink` is a `&mut dyn FnMut`, is not `Sync`, and the
  tempting fix (cloning or locking it) would reorder the per-step reports -
  which `every_euler_step_reports_progress` asserts the order of.
- Falls back to the previous single-card path, byte for byte, on fewer than
  two schedulable cards, on the CPU backend, when the caller pinned an
  explicit `GenOpts::device`, and under the env opt-out.

### Bit-identity is the prediction, not a hope

Nothing is split across the cards and recombined, so no sum is reassociated
and no accumulator partitioned; each card runs the identical dispatch
sequence over identical bytes it would have run alone. Both gates are
therefore bit-pattern comparisons rather than tolerances - a disagreement
would be a real defect (a nondeterministic kernel, an uninitialised read, one
`Resident` shared across cards, or the two cards' autotuners selecting
different kernel variants) and is worth failing on.

- `denoise::tests::the_concurrent_cfg_pair_is_bit_identical_to_the_
  sequential_one` - `DitConfig::tiny()`, milliseconds, no checkpoint, real
  kernels on real devices. On a two-card box it really does dispatch across
  both (it prints the placement it resolved); on one card or none it
  degenerates to the same thing twice and still gates the fold and the
  plumbing.
- `crates/minimaxmusic3/tests/cfg_parallel.rs` (`#[ignore]`d - it is the only
  test here that wants BOTH cards to itself) - the real-weight half: a full
  200-frame (689-latent) chunk of the real 2.4B DiT, same seed and weights,
  differing only in the placement, compared bit for bit, plus an assertion
  that the chunk is not constant (any two runs of a frozen loop agree). It is
  also where the per-Euler-step wall clock of each arm is reported.
- 4 new gates in `minimaxmusic3::devplan` pin the placement rules themselves:
  an explicit device never splits, two branches on one card do not read as
  parallel, and `Auto` never names a card outside
  `ambient_compute_set()`.

### Measured

Real `transformer/` checkpoint, one full 200-AR-frame chunk (689 latents x
128 channels), 5 Euler steps with the first discarded (it carries both
cards' `Resident` upload and every first-touch cost), two Tesla P40s. The
per-step numbers come from the progress callback, which is the one point per
step that runs on the orchestrating thread after both branches have joined.
The absolute times are against Phase 15's attention; the RATIO is what this
phase claims, and it does not depend on which kernels a forward dispatches -
both arms run the identical ones.

| run | box | sequential, one card (s/step) | concurrent, two cards (s/step) | best-of ratio |
|---|---|---|---|---|
| A | shared | 2.96 3.12 3.04 3.07 | 1.88 1.76 (then 11.44 11.24) | 1.68x |
| B | shared | (9.83) 2.88 3.13 3.03 | 2.93 2.26 2.15 2.12 | 1.36x |
| **C** | **idle** | **3.09 3.10 3.07 3.11** | **1.81 1.88 1.82 2.00** | **1.69x** |

**1.69x per Euler step, and bit-identical over all 88192 latents** in every
one of the three runs. C is the number to quote: both cards otherwise idle,
and both arms steady to within 1% and 10% respectively rather than drifting.
`nvidia-smi` during C's concurrent arm reads 9.55 GiB and ~80% busy on BOTH
cards at the same time, against gpu1 at 1 MiB and 0% through the whole
sequential arm - the idle card this phase exists to remove.

A and B are kept because they show which way contention biases the
measurement: every contaminated sample lands on the CONCURRENT arm, because
that is the arm needing both cards (A's last two steps jumped to 11 s when a
neighbouring process took gpu1 mid-run; B's whole concurrent arm ran against
a neighbour holding ~4 GB there). The sequential arm needs only gpu0 and is
steady at ~3.1 s/step in all three. A shared box therefore UNDER-reports this
change, never over-reports it.

2x is the ceiling and this does not reach it, for a reason visible in the
profile rather than guessed at: `dit::forward_resident` reads a buffer back
to the HOST in the middle of every one of the 36 blocks, so a real fraction
of each forward is host work and host<->device sync that two threads on this
Xeon do not scale perfectly through. Closing that gap is a different piece
of work (it is the same readback `mm3_bench`'s "total minus summed kernel
time" gap already reports) and is not attempted here.

For a 4-minute track at the reference's 30 steps, 1.69x turns Phase 13's
~1.2 h denoise stage into ~43 min, and the AR stage - already the
bottleneck - stays untouched.

The per-chunk `Resident` upload is the one cost the split ADDS, and it is
paid concurrently: C's concurrent arm spent 58.2 s on "two 9.7 GB uploads
plus the first Euler step" against 47.6 s for one upload plus one step, i.e.
the second card's upload costs ~10 s per chunk rather than the ~40 s a
serialised second upload would have. At 30 steps per chunk that is ~1% of
the chunk; at the 8-step schedule it is ~5%, and worth revisiting only if
the step count drops further.

### The honest gap

`Auto` borrows the second card without asking `crates/residency` whether
another model is resident on it; there is no seam to ask through today. This
is not a new claim on the box - `generate::ar_branch_devices` already puts
the AR stage's two Global LLM instances on both cards unconditionally - but
it is the same gap, and `devplan`'s module doc states it rather than taking
the card quietly. What bounds it is `memauth`'s `--limit-vram-total`, which
is a process-wide TOTAL across all cards rather than a per-card ceiling, so a
concurrent pair charges it twice: a value sized for one card should be raised
or paired with `BRAIN_MINIMAXMUSIC3_CFG_PARALLEL=0`.

## Phase 17: the depth decoder was 77% of a generation, on the host, at 0.5 FLOP/byte

After Phase 15 (flash attention, 28.7x on the DiT's attention trio) and Phase
16 (the two CFG branches on two cards, 1.69x), the denoise stage was ~10% of a
generation and the AR stage ~77%. Roughly 90% of that AR stage was
`depth_decoder`, which ran as pure host math and issued no dispatch at all -
both P40s read 0% utilisation for the entire AR phase while holding 15 GB of
Global LLM weights.

`mm3_bench depth` (real checkpoint, 48-core Xeon, two Tesla P40s) at
`DepthDecoderConfig::real()` - hidden 4096, inter 6144, 4 layers, 16 heads, 8
codebooks:

```
call class                           ms/frame        % calls/frame
projection (seed x2)                     3.16     0.3%          2
step (transformer block stack)         915.20    98.1%         16
audio_head                               5.15     0.6%         14
embedding lookup + projection            9.27     1.0%          6
WHOLE FRAME                            932.78   (18.38 GFLOP, 36.74 GB streamed
                                                 -> 19.7 GFLOP/s, 39.4 GB/s)
```

Every number in this phase comes from one paired `mm3_bench depth 4 3
--device gpu0` run on the real checkpoint, so the three arms saw the same
machine. This box is shared, and the HOST arms are the ones contention moves:
across runs the b=1 arm landed between 933 and 1055 ms/frame and the b=2 arm
between 489 and 534, while the DEVICE arm sat at 216-220 ms in every run. The
run quoted here is the least contaminated one (§F: the minimum is the least
contaminated sample), and a contended box always UNDER-reports the device
speedup, never over-reports it.

**0.50 FLOP/byte.** That single number decided everything below. The host code
was not slow - `hostmath::matvec` is already AVX2+FMA over rayon and was
running at this box's DRAM speed. There was nothing to make faster; there were
only bytes to stop moving.

### Step 1: batch the two CFG branches on the HOST - 1.91x, bit-identical

`pipeline::generate_depth_codes` drove two independent `KvCache`s with the
SAME `row` at every position past 0, meeting only at `cfg_blend` after both
`audio_head` calls. Two `m=1` GEMVs against one weight matrix stream that
matrix from DRAM twice for arithmetic that could share one pass.

The fix went into the shared primitive, not into this model:
`hostmath::linear_rows` became ONE `backend_cpu::fast_ops::matmul_abt` call at
`n = rows` instead of `rows` calls at `n = 1`. `depth_decoder::step_batch` is
the new b-row step; `step` is a one-line wrapper on it, so there is no second
copy of the block stack.

It is **bit-identical**, and that was verified rather than reasoned about:
`row_abt_avx2` gives every output its own 8-wide accumulator and reduces `k`
in the same order whatever its 4-column register blocking does, so `c[o,r]`
does not depend on `rows`. `hostmath::tests::linear_rows_is_bit_identical_to_
a_per_row_matvec_loop` `assert_eq!`s that at rows 1,2,3,4,5,8 (straddling the
4-column block) and at an `inn` that is not a multiple of 8 (exercising the
scalar k-tail).

```
WHOLE FRAME  489.43 ms  (18.38 GFLOP, 18.49 GB streamed -> 37.6 GFLOP/s, 37.8 GB/s)
b=1 vs b=2 over every hidden state of one frame: max|delta| = 0e0
```

**932.78 -> 489.43 ms/frame, 1.91x, and the achieved bandwidth did not move
(39.4 -> 37.8 GB/s).** That is the whole proof of the diagnosis: the win came
entirely from halving the bytes, not from going faster. It also helps machines
with no card at all, which is the other half of why this step went in first.

Nothing else in this loop is batchable, and both alternatives are blocked
rather than unimplemented: the 8 depth steps within a frame are strictly
dependent (step `i+1`'s input is the embedding of the code sampled from step
`i`'s logits), and consecutive frames are too (`embed_audio_frame` sums all 8
codebooks before the Global LLM advances).

### Step 2: the device port - 4.28x total, and ZERO new WGSL

`depth_decoder::Resident` runs the same graph on one card at the same b=2.
Every op mapped to a kernel that already existed:

| op | kernel | via |
|---|---|---|
| RMSNorm | `rmsnorm_eps` / `rmsnorm_rows` | `block::rms_variant` |
| q/k/v/o, gate/up/down | `matmul_gemv` (m=2) / `matmul` | `block::gemm_variant` |
| KV append + attention | `paged_kv_append_batched`, `paged_decode_scores_batched`, `decode_softmax_batched`, `paged_decode_apply_batched` | direct, `group = 1` |
| `silu(gate)*up` | `silu_mul` | `block::swiglu_fwd` |
| both residual adds | `add2` | direct |

Paging is not a complication here, it is the mechanism that makes attention
BATCHED over the two branches with no per-row slicing at all: one page per
branch, `block_size = cap = max_position_embeddings`, `max_bt = 1`, so branch
`b`'s position `j` is pool row `b*cap + j`. `block::gqa_attn_sublayer_decode_
step` was deliberately NOT used - it dispatches RoPE unconditionally (this
architecture has none) and is b=1.

`pos_embedding` is added host-side into the buffer being uploaded rather than
through a `bias_add` dispatch: the input row comes from the host anyway (it is
a projection of a freshly sampled code), so the add is free there and costs a
dispatch on the device.

Measured on one Tesla P40 (measured roof 10517 GFLOP/s, 287.5 GB/s DRAM),
real checkpoint:

```
WHOLE FRAME  218.07 ms  (18.38 GFLOP, 18.49 GB streamed -> 84.3 GFLOP/s, 84.8 GB/s)
Decoder::device upload (ONCE per generation): 6.28 s for 2.28 GB
host b=2 vs device b=2 over every hidden state of one frame:
    cosine = 1.000000000, max|delta| = 1.335144e-5
```

**489.43 -> 218.07 ms/frame, 2.24x on top of the host batching, 4.28x against
the b=1 baseline** (2.24-2.47x and 4.28-4.88x across runs). A 200-frame
denoise chunk's worth of depth decoding drops from 187 s to 44 s; a 4-minute
track's AR depth stage from ~1.6 h to ~0.36 h.

The weights upload ONCE per generation. Re-uploading 2.28 GB per frame at the
measured 0.36 GB/s would cost 6.3 s per frame against 0.22 s of compute -
nearly thirty times the computation. This is `dit::Resident`'s lesson at a
much higher call rate.

### The per-kernel table, and the hypothesis it kills

`mm3_bench depth` now also profiles ONE `Resident::step` from device
timestamps (§F.1), b=2, 69 dispatches:

```
kernel                            ms      n      %   ms/call     GB/s   %roof
matmul_gemv                    21.78     28  97.2%     0.778    104.8   36.5%
rmsnorm_rows                    0.31      9   1.4%     0.034      2.4    0.8%
paged_decode_scores_batched     0.11      4   0.5%     0.026     21.2    7.4%
paged_kv_append_batched         0.06      8   0.2%     0.007      9.5    3.3%
add2                            0.05      8   0.2%     0.007     15.1    5.2%
decode_softmax_batched          0.03      4   0.2%     0.009      0.5    0.2%
paged_decode_apply_batched      0.03      4   0.2%     0.009     64.2   22.3%
silu_mul                        0.03      4   0.1%     0.007     21.3    7.4%
WHOLE PASS                     24.21     69          (sum of kernel time 22.41)
```

The plan for this port assumed ~320 genuinely tiny dispatches per frame would
be "pure launch overhead (3-10 ms)" and prescribed fusing `wq|wk|wv` into one
`[3d, d]` buffer and `gate|up` into `[2*inter, d]` to cut the dispatch count.
**The profile kills that.** Whole pass minus summed kernel time is 0.8-1.8 ms
of 23-24 (3-7%), and that gap is host math, the readback and launch gaps
TOGETHER, not launch alone. Fusing would have removed 2 of 17 dispatches per
layer and could not have returned more than a fraction of it.

It would also have cost something real. Slicing a fused `[b, 3d]` output back
into q/k/v needs `step_sliced` offsets of `r*3d + {0,d,2d}` floats, and
`step_sliced` offsets must clear `min_storage_buffer_offset_alignment` (256 B
= 64 floats on this card). At `real()` dims 4096 and 6144 both clear it; at
`tiny()` dims 8 and 16 do not - so the fused form would have been unable to
run the tiny parity gate on a real GPU at all, which is where the port was
actually debugged. Not fusing was free; fusing would have bought ~0 and cost
the small-shape gate. §E, again.

**The one row that matters is `matmul_gemv` at 36.5% of this card's measured
memory roof**, holding 97.2% of the step. That is where the next lever is, and
it is not a depth-decoder question: it is the shared decode-regime GEMM every
model in this workspace dispatches at small `m`, so a fix there lands in
`block::gemm_variant`'s tier and reaches all of them (§F.7). Not attempted
here.

### Parity, and what does NOT survive

The host `forward`/`backward` pair is untouched and stays host math on
purpose - it is the training path, `gradcheck`'s reference, `depth_lora`'s
substrate, and the oracle everything else is gated against, including
`tests/depth_decoder_parity.rs` at cosine 0.9999 vs diffusers. The module doc
says so, because "finish the job by porting `forward` too" would delete the
reference that proves the device path right.

Gates added:

* `hostmath::tests::linear_rows_is_bit_identical_to_a_per_row_matvec_loop` -
  `assert_eq!`, the batching's exactness.
* `depth_decoder::tests::the_device_step_matches_the_host_forward_on_the_{cpu_
  backend,default_device}` - walks every position a frame can reach with
  INDEPENDENT random inputs in the two batch rows (so a batch-row mix-up or a
  KV page bleeding into its neighbour is visible), comparing each row against
  that row's own sequence through `forward`. Floors: **cosine >= 0.999999 AND
  rel_l2 <= 1e-4**, both, because cosine alone is scale-invariant. Measured
  worst at `tiny()`: cosine 1.000000000, rel_l2 1.52e-7 (CPU backend) and
  1.25e-7 (P40).
* `..._at_real_dims`, the same walk at `real()` dims with the REAL checkpoint,
  behind `BRAIN_MINIMAXMUSIC3_DEPTH` (it uploads 2.28 GB and runs the host
  `forward` over every prefix). Measured worst: cosine 1.000000000, rel_l2
  **2.84e-6** on the P40 and **9.50e-7** under `BRAIN_DEVICE=cpu`. Real dims
  are not a formality here - `tiny()` is 8 wide, which is where neither the
  cooperative `rmsnorm_rows` nor the workgroup-per-column `matmul_gemv` has
  anything to split, and real weights are the value distribution the
  tolerance has to hold at.
* `pipeline::tests::the_device_depth_decoder_matches_the_host_one_at_the_cfg_
  blended_logits` - worst cosine 1.000000000.
* `depth_decoder::tests::a_reset_decoder_reproduces_its_own_first_frame` - the
  per-frame `reset` lifecycle, both implementations.

The gate was mutation-verified (§F.8) - four mutations, each run on BOTH
backends, cosine floor / rel_l2 ceiling:

| mutation | cosine | rel_l2 | caught by |
|---|---|---|---|
| `pos_embedding` add dropped | 0.9027 | 0.441 | both |
| `wk`/`wv` swapped | 0.7414 | 0.719 | both |
| the two batch rows swapped on the way out | -0.0803 | 1.470 | both |
| the final RMSNorm's eps 1e-6 -> 1e-2 | **1.000000** | **0.0467** | **rel_l2 only** |

The last row is why the gate asserts both. A wrong epsilon scales the whole
output vector uniformly; cosine is scale-invariant and reports a perfect 1,
and only relative L2 sees it. A cosine-only gate would have shipped it.

**End-to-end bit-identity does not survive the device port, and is not
claimed.** The host b=2 step is bit-identical (max|delta| exactly 0), but the
device step is not - a different GEMM reduction order and a different `rsqrt`
put it 1.34e-5 away at real dims. `sample_top_k` turns those logits into a
discrete draw, so a wobble that small is enough to reorder two near-tied
candidates or move a cumulative-probability boundary across the RNG's draw,
and one different code changes every later frame. The gate is therefore at the
CFG-blended LOGITS per frame, never at the waveform: a device run produces a
different, equally valid sample of the same distribution, not the host run's
audio.

### Placement, and why the depth decoder does NOT get its own card

It goes on the SAME card as the conditional Global LLM branch. The entire
point of b=2 is that both branches share one pass over the weights, which
requires them in one dispatch and therefore on one device; a second card would
buy back the 2x that batching already took, at 2.28 GB more VRAM.
`generate::ar_branch_devices` puts the two LM instances on gpu0/gpu1 and the
`Resident` sits beside the first of them (~2.28 GB fp32 next to the ~7 GB int8
LM).

`BRAIN_DEVICE=cpu` keeps `Decoder::Host`. That is a selection, not a missing
port, and it is measured rather than assumed - §F.4 says to profile the branch
your hardware does not take, so `mm3_bench depth --device cpu` runs the device
row anyway and labels it as not-what-ships:

```
DEVICE b=2 on the Cranelift JIT   2358.87 ms/frame   (7.8 GB/s)
    -> 0.23x the host b=2 path, 0.51x even the b=1 host baseline
```

**Four times slower than the host implementation it would replace.** Two
reasons, both structural: the JIT reports `workgroup_reductions: false`, so
`block::gemm_variant` and `block::rms_variant` fall back to the reference
`matmul` (one thread per output element, serial inner reduction) and the
per-element `rmsnorm_eps` rather than `matmul_gemv`/`rmsnorm_rows`; and even
the fast kernels would be competing with `matmul_abt`'s AVX2+FMA over the same
48 cores. A branch nobody measures is how a slow path survives, so this one is
in the harness permanently.

## Phase 17: residency, caching and batching stop being placeholders

`crates/cli/src/resident_minimaxmusic3.rs` was a load-per-call adapter whose
`estimate()` reported `MemCost::new(0, ram)` - and `vram == 0` is not a
conservative choice, it is a disabling one: `residency::place::pick_device`
`continue`s past every GPU when `cost.vram == 0`, so on this 2x24 GB box the
scheduler could never place this model on a card at all. Its RAM figure was
the on-disk checkpoint size times four, a multiplier justified by an
int8-promotes-to-fp32 claim Phase 12 already disproved for GPU backends.

### Honest per-stage VRAM (`crates/minimaxmusic3/src/memory.rs`)

Derived from this model's own `::real()` configs where a closed form exists,
measured where one does not, and every figure pinned by a test in that
module - a number nothing checks is a number that goes stale.

| stage | charged | where it comes from |
|---|---|---|
| AR | **16.09 GB** | one Global LLM instance (6.95 GB of int8 linears, MEASURED Phase 14, + `tok.weight` and `lm_head` at 3.28 GB each, fp32 and vocab-tiled) + the depth decoder's 2.58 GB, which `depth_decoder_device` puts on the SAME card |
| denoise | **10.05 GB** | 9.664 GB closed form over `DitConfig::real()` (36 x 64 Mi params) + a 384 MiB margin covering the 9553 MiB whole-device peak Phase 15 measured |
| vocode | **12.86 GB** | MEASURED, 12264 MiB on one 689-latent chunk (Phase 13); `CHUNK_FRAMES` caps every chunk at that, however long the song |

The three are never co-resident, so the charge is their MAX, not their sum.
Two closed forms reproduce a measurement exactly rather than restating it:
the DiT's 9.664 GB and the depth decoder's 2.584 GB - the latter is also
what pins its MLP as a SwiGLU triple (a plain up/down pair gives 2.18 GB).

**The second card is still charged to no budget.** A generation occupies
both (`ar_branch_devices`, `devplan`), and `estimate` is single-device. The
seam is `residency::MultiDeviceResidentModel`; it is not taken because
`multi::pick_devices` is all-or-nothing over a fixed device set (naming two
cards makes this model unplaceable on a one-card box, where it degrades
correctly today), multi-device residents are not auto-evicted, and
`crates/stats`' `ModelStat` schema is single-device so such a resident
vanishes from `braintop`. Same gap `resident_ltxv.rs` has.

### Warm weights (`crates/minimaxmusic3/src/weightcache.rs`)

Four components import into a plain tree of host `Vec<f32>` (DiT, depth
decoder, vocoder, condition encoder) and are now held in a process-wide
store keyed on the checkpoint directory's recursive `(summed length, newest
mtime)` - `ltxv::weightcache`'s shape. That removes a ~10.7 GB re-read per
`generate` call. Safe to drop at any moment: an entry is a pure function of
immutable checkpoint bytes, so an eviction costs time and never a number.

Governed by residency first (`estimate_at`/`demote`/`promote` on the
adapter, with `metrics` reporting hits/misses/evictions/bytes) and by a
local `memauth::limits().ram_total` share second, evicted with
`residency::place::CostAware` - the same GDSF policy the manager scores whole
instances with, reused rather than transcribed.

The Global LLM is deliberately NOT warm: it owns its own `Gpu`, is `Send`
but not `Sync`, and its KV capacity `t = prompt_len + max_frames + 8` is a
function of the request, not the checkpoint.

### `run_batch` is serial, and the reason is in the file

Not the default fallthrough - an explicit override carrying three
independently sufficient reasons. (1) One request already occupies every
card here: two AR-stage Global LLM instances do not fit one 24 GB card
(`memory.rs` asserts exactly that), so `residency::DevicePool`'s
one-generation-per-card trick has no spare card to offer. (2) The DiT is
`batch=1` by construction with five silent-corruption sites at `b > 1` (the
RoPE `tmod`, the row-0 timestep slice, the `preprocess_hidden_lc`
transpose, the timestep row assembly, the `Bidir { b: 1, .. }` slabs) and
unmasked bidirectional attention, so a batched slab leaks across requests
without erroring. (3) `global_llm::import` asserts `b == 1` because
`qwen3`'s KV decode path sizes `kcache`/`vcache` as `t * kv_dim` with no
batch axis. What a batch does get is one instance against a warm cache.

## Phase 18: `matmul_gemv` was 97% of a depth-decoder step at 36% of the memory roof

Phase 17's table ended with one row and one sentence: *"the one row that matters
is `matmul_gemv` at 36.5% of this card's measured memory roof, holding 97.2% of
the step... a fix there lands in the shared decode-regime GEMM every model in
this workspace dispatches at small `m`."* This is that fix.

Hardware: one idle Tesla P40 (GP102, cc 6.1), `gpu_core::roof` measured
**10517 GFLOP/s / 287.5 GB/s**. Harness: `mm3_bench depth 8 3`, real
`DepthDecoderConfig::real()` dims (`hidden 4096`, `inter 6144`, 4 layers →
7 GEMVs/layer × 4 = the 28 calls), `b = 2` (the two CFG branches), device
timestamps inside the production pass.

### The two limiters, each measured, not argued

`matmul_gemv` accumulates in `var<workgroup> partial: array<f32, 2048>` and
does a read-modify-**write** into it per `(k, m)`.

**1. The array is sized for the worst case (`m = 32`) - 8 KB per workgroup at
every `m`.** Sweeping ONLY that literal (`partial` resized, body untouched,
bit-identical) at the depth decoder's own shapes, `m = 2`, `k = n = 4096`:

```
partial     bytes/wg    ms      GB/s    predicted resident wgs/SM
2048 f32      8192    0.844      79.5    12  (96 KB / 8 KB)     -> 768/2048 thr, 37.5%
1024 f32      4096    0.474     141.5    24                     -> 1536/2048 thr, 75%
 512 f32      2048    0.402     166.8    32 (block cap binds)   -> 2048/2048 thr, 100%
 256 f32      1024    0.406     165.4    32 (capped)            -> 100%
 128 f32       512    0.410     163.8    32 (capped)            -> 100%
```

The measurement **confirms the GP102 numbers rather than assuming them**: the
curve rises at 8 KB→4 KB, rises again at 4 KB→2 KB, and then goes *flat* - which
is only explicable if 96 KB of shared memory per SM and a 32-block-per-SM cap
are both real, since below 2 KB/block the block cap binds and further shrinking
buys nothing. 2.10x from one literal.

**2. `partial[i] = partial[i] + x*wv` is a serial dependency chain through
shared-memory latency**, one link per k-step per row. Register accumulators
remove it - and, at `m = 2`, they matter MORE than the occupancy:

```
m = 2, k = n = 4096       ms      GB/s   %roof
matmul_gemv (8 KB)      0.844      79.5   27.6%
  partial shrunk to m*64 only
                        0.410     163.8   57.0%
register acc, 8 KB partial (37.5% occupancy!)
                        0.318     210.7   73.3%
register acc + m*64 partial
                        0.314     213.9   74.4%
```

Registers at **37.5% occupancy beat shared-memory accumulators at 100%**. So
limiter 1 is real but SUBORDINATE: shrinking the array only helps because it
buys enough occupancy to hide the chain; removing the chain makes occupancy
nearly irrelevant. Both fixes fall out of one constant anyway (see below), so
this is a finding about the mechanism, not a choice that had to be made.

(`%roof` in these micro-A/B tables counts weight bytes only, `4*n*k`; the
`mm3_bench` table counts `x` and `out` too, so its baseline reads 35.9%, not
27.6%. Same kernel, two byte conventions - compare within a table.)

### Killed: a runtime-`m` register kernel

The obvious way to avoid per-`m` specialisation is named scalar accumulators
with uniform `if (p.m > 1u) { ... }` guards inside the k-loop - runtime `m`, no
array, no template. **Measured, and it is worse than doing nothing structural:**

```
m = 2, k = n = 4096                            ms      GB/s
matmul_gemv, 8 KB partial                    0.844      79.5
4 named scalars + uniform guards, 8 KB       0.774      86.7
4 named scalars + uniform guards, 1 KB       0.460     145.8
partial shrunk to m*64, NO register change   0.410     163.8   <- plain shrink beats it
```

Uniform, perfectly-predicted branches in the innermost loop still cost more than
they save. That is why the register variant needs a **compile-time** row count,
and therefore a `kernels::template` knob and a bucket ladder.

### What shipped

**ONE new kernel file**, `crates/kernels/wgsl/matmul_gemv_reg.wgsl`, plus one
row in `gpu_core::upgrade`. No model crate changed at all.

* The body is `matmul_gemv`'s with the accumulators moved to a function-local
  `array<f32, MREG>` whose every index is a compile-time-bounded loop
  (checklist §C1), and `partial` sized `MREG * 64` - **both limiters from the
  one `MREG` constant**, which is exactly the `kernels::template` shape
  (`interned(stem, src, &[("MREG", b)])`), so there is no hand-written per-`m`
  file.
* It is a genuine second FILE, not a template variant of `matmul_gemv`, because
  a function-local accumulator array is a different body - and because the CPU
  JIT rejects one outright. That claim was **re-verified, not inherited**:
  `wgsl_cpu::Jit::new` returns `array local in a work-group kernel is
  unsupported`. So `matmul_gemv` keeps its workgroup accumulators and stays
  `@cpu yes`; `matmul_gemv_reg` is `@cpu no`. Each header points at the other
  and says "edit the two together".
* Selection is `gpu_core::upgrade` - the seam `.agents/rules/kernels.md` §A.4
  prescribes for a *drop-in*: same `Params{m,k,n}`, same bindings, same
  `n * 64` thread count, bit-identical results. It meets all four bars, so
  every crate that registers `matmul_gemv` (qwen3 incl. `serve` and the
  `decode_logits` head, qwen35, qwen35moe, flux1, wan, ltxv, pulid, instantid,
  minimaxmusic3's DiT **and** depth decoder, and anything routed through
  `model::rowemit`/`model::dispatch`) inherits it with **zero** edits. The
  capability gate is still `backend_api::select`: the row activates only where
  `select::candidates(MatMul, m = 1, ...)` heads with `WorkgroupPerOutput`.

**New in `upgrade`: a shape-specialised row.** The register kernel needs `m` at
compile time, so the row carries a knob + a bucket ladder and `apply` picks the
smallest bucket covering the caller's own `params[0]`. Six pipelines are
appended (`MREG ∈ {1,2,4,8,16,32}`); their compile cost is **unmeasurable**
against device init (301 ms vs 308 ms with the table disabled, 5 runs each).

**The ladder is a measurement (§F.6), not a guess.** A single worst-case
`MREG = 32` specialisation would have been a REGRESSION at the most common
shape of all - `m = 1`, plain single-sequence decode:

```
actual m -> variant       1      2      3      5      9     13     17     25
matmul_gemv (ms)       0.580  0.741  0.952  1.005  1.556  2.022  2.504  3.580
MREG=32 for everything 1.322  1.405  1.439  1.428  1.399  1.443  1.456  1.551
  -> 0.44x   0.53x  0.66x  0.70x  1.11x  1.40x  1.72x  2.31x
power-of-two bucket    0.311  0.316  0.408  0.535  0.890  1.046  1.456  1.551
  -> 1.87x   2.34x  2.33x  1.88x  1.75x  1.93x  1.72x  2.31x
```

Cost is a function of `MREG`, essentially not of `m`, so "wins at every shape"
(bar 3) forces the ladder. Every `m` in 1..=32 wins by >= 1.7x.

### Bit-identity: claimed, and gated on the BITS

Same k-stride (`k = t; k += 64`), one accumulator per output, the same 64
partials folded in the same ascending order. Nothing reassociates, so
`crates/gpu-core/tests/gemv_reg_upgrade.rs` asserts `to_bits()` equality - not
cosine, not rel_l2 - for **every `m` in 1..=32 across four shapes**
(`k,n` = 512,384 / 384,512 / 517,129 (ragged `k`) / 64,71), against
`kernels::MATMUL_GEMV` registered under a second name so one handle runs both
forms in one submit. Both are also checked against an independent f64 host
oracle, because a bit-identical pair can still be identically wrong (§F.5).

Mutation-verified - four mutations, each caught by the right assertion:

| mutation | bit-identity | f64 oracle | ladder |
|---|---|---|---|
| fold order reversed (pure reassociation, same maths) | **FAIL** | pass | pass |
| `xoff` clamp broken (every row reads row 0) | **FAIL** | **FAIL** | pass |
| off-by-one on the k axis | **FAIL** | **FAIL** | pass |
| `MREG=1` removed from the ladder (a SELECTION bug) | pass | pass | **FAIL** |

The first row is the point: a reassociated reduction is invisible to the oracle
and to any tolerance-based comparison, and is exactly the change these two
kernels must never make independently.

### Measured, before and after (`mm3_bench depth 8 3`, same binary, one idle P40)

The A/B is `BRAIN_NO_KERNEL_UPGRADE=1` versus not, so it is the same build.

```
ONE `Resident::step` (b = 2, position 0), 69 dispatches

                          before                        after
kernel                 ms    n   ms/call  GB/s %roof |    ms  ms/call  GB/s %roof
matmul_gemv         22.06   28    0.788  103.5 36.0% |  8.23   0.294  277.6 96.6%
rmsnorm_rows         0.31    9    0.034    2.4  0.8% |  0.31   0.034    2.4  0.8%
paged_decode_scores  0.11    4    0.027   20.6  7.2% |  0.11   0.027   20.6  7.2%
paged_kv_append      0.06    8    0.007    9.3  3.2% |  0.06   0.007    9.3  3.2%
add2                 0.06    8    0.007   14.2  4.9% |  0.06   0.007   14.0  4.9%
paged_decode_apply   0.04    4    0.009   62.4 21.7% |  0.04   0.009   62.4 21.7%
decode_softmax       0.03    4    0.008    0.5  0.2% |  0.03   0.009    0.5  0.2%
silu_mul             0.03    4    0.007   22.2  7.7% |  0.03   0.007   21.3  7.4%
WHOLE PASS          24.29   69                       | 11.03
sum of kernel time  22.75                            |  8.86
```

Run-to-run spread over repeated `best-of-3` runs on the idle card:
`matmul_gemv` 8.2-8.4 ms, whole pass 10.6-11.0 ms, whole frame 106.1-106.5 ms.
The numbers above are one such run, not a best pick across them.

* **`matmul_gemv`: 22.06 -> 8.23 ms, 2.68x. 0.788 -> 0.294 ms/call. 103.5 ->
  277.6 GB/s. 36.0% -> 96.6% of this card's measured memory roof.**
* One `Resident::step`: 24.29 -> 11.03 ms (2.20x).
* Block stack (8 steps): 195.4 -> 85.8 ms/frame.
* **Whole frame: 216.96 -> 106.07 ms, 2.05x** (against a predicted 1.67x).
* Device vs the host b=2 path: 2.14x -> 4.09x (8.94x vs the b=1 host baseline).
* Extrapolated: a 200-frame denoise chunk's depth decoding 43.1 s -> 21.3 s; a
  4-minute track 0.36 h -> 0.18 h.
* `BRAIN_PROFILE`'s physical-pipeline line names `matmul_gemv_reg#MREG=2`, so
  the right bucket is visible in a profile, not inferred.

**The top row is now at 96.6% of the memory roof, so §F.2 says stop**: no
further kernel change can help this shape. The only lever left is moving fewer
bytes - the weights are the traffic (2.28 GB per frame at fp32), so the next
real step for this component is a narrower weight tier (bf16/int8), not a
kernel.

### What this did NOT cover

* **The `#w=bf16` / `#w=f16` storage tiers of `matmul_gemv`**
  (`kernels::template::dtype_variant`) register under their own names, so the
  upgrade table does not reach them and they still carry the 8 KB `partial` and
  the shared-memory chain. `qwen3` registers both. Adding their register
  siblings is additive - a second knob on the same file - and is the obvious
  follow-up.
* **`Gpu::step_buf`** cannot be upgraded by a shape-specialised row at all: its
  shape lives in a caller-owned uniform buffer the seam cannot read, so such a
  call keeps the registered kernel. No fp32 GEMV site in this tree uses it
  today (every one passes `&[m, k, n]` to `step`/`step_sliced`), but a future
  one would silently stay on the slow path.
* **`matmul_gemv` itself was left at `array<f32, 2048>` on purpose.** Shrinking
  it via a template knob was measured (2.10x, table above) and deliberately not
  shipped: after this change that kernel only runs on the CPU JIT, on the
  `@npu` path, and under `BRAIN_NO_KERNEL_UPGRADE=1` - none of which pays a
  shared-memory occupancy cost - so the knob would have been machinery with no
  live caller.

## Phase 19: the DiT TRAINER never got the fast GEMM, and every unused kernel slot pointed at kernel 0

Two defects in `crates/minimaxmusic3/src/dit_train.rs`, one file, one commit.

### Defect 1 - `dit_train` dispatched the naive `matmul`/`matmul_dx`/`matmul_dw`

Phase 15 fixed exactly this for `dit.rs` (the served forward) and the trainer
never inherited it: its `PIPELINES` registered only the `@opt 2` reference trio
and hardcoded them at all ten of its GEMM sites. The fix routes the forward
through the SAME `model::block::gemm_variant` call `dit::linear_step` makes
(the selector is now literally the same function, shared, with each module
passing its own indices), and the backward through `model::block::pick_gemm`
over `matmul_dx_reg` / `matmul_dw_reg` - the shared rule `t5encoder::train`,
`clip::model` and `codeformer::train` already use.

Per-kernel A/B, `mm3_bench gemm-bwd 7` / `gemm gemm 7`, one P40 (measured
roofline 10517 GFLOP/s, 287.5 GB/s), `DitConfig::real()`, 690 rows, best of 7,
warm-up excluded:

| site | shape `[m,k,n]` | naive ms | tiled ms | speedup | tiled % of fp32 peak |
|---|---|---|---|---|---|
| fwd attn q/k/v/out | 690, 2048, 2048 | 183.5 | 1.65 | 111x | 33.3% |
| fwd ff_in (fused) | 690, 2048, 16384 | 1214.0 | 9.32 | 130x | 47.3% |
| fwd ff_out | 690, 8192, 2048 | 1972.0 | 5.77 | **342x** | 38.2% |
| dX attn q/k/v/out | 690, 2048, 2048 | 25.41 | 1.98 | 12.8x | 27.7% |
| dX ff_in | 690, 2048, 16384 | 234.6 | 12.48 | 18.8x | 35.3% |
| dX ff_out | 690, 8192, 2048 | 146.3 | 5.54 | 26.4x | 39.7% |
| dW attn q/k/v/out | 690, 2048, 2048 | 17.72 | 1.75 | 10.1x | 31.4% |
| dW ff_in | 690, 2048, 16384 | 173.0 | 10.47 | 16.5x | 42.1% |
| dW ff_out | 690, 8192, 2048 | 142.2 | 5.47 | 26.0x | 40.3% |

Whole pass, `mm3_bench dit-train <layers> 689 3`, same card, best of 3:

| layers | naive ms | fast ms |
|---|---|---|
| 1 | 10583.5 | 5609.7 |
| 2 | 15660.3 | 5942.7 |
| 3 | 20795.5 | 6795.2 |

The MARGINAL cost of one block's training step (forward + backward) is
therefore **5106.0 ms -> 592.8 ms, 8.6x**; the whole 3-layer step is 3.06x.
The gap between 8.6x and the per-kernel table's 10-342x is the ~5.0 s of
LAYER-INDEPENDENT host glue every step pays either way (`proj_in`/`proj_out`
and the two `k=1` convs are host `matvec` loops over 689 positions - 6.5 GFLOP
single-threaded), which is unchanged by this work and is now the top row.
Extrapolated to the real 36 layers: ~189 s -> ~26 s.

### Split-K was measured and REJECTED, both families

`matmul_dw_reg_splitk` (+ `dw_splitk_reduce`) and `matmul_reg3_splitk` exist for
a tile grid too small to fill the card. This model's grids are not small - the
dW grids are 256 / 1024 / 2048 workgroups on a 30-SM P40 - and every slice count
tried was slower:

| dW site | unsplit ms | s=2 | s=4 | s=8 |
|---|---|---|---|---|
| attn (256 wgs) | 1.753 | 2.061 | 2.432 | 3.180 |
| ff_in (2048 wgs) | 10.465 | 12.821 | 16.586 | 21.129 |
| ff_out (1024 wgs) | 5.466 | 6.695 | 8.292 | 10.528 |

The forward sibling is the only place split-K was ever even close, and it is
shape-dependent rather than a win: at `ff_out` (96 wgs) `matmul_reg3` 5.693 ms
vs `s=8` 5.039 ms (1.13x), at `attn` (96 wgs) 1.663 vs 1.640 (1.01x), and at
`ff_in` (768 wgs) 9.269 vs 10.041 - a 1.08x LOSS. Not worth a second selection
rule that would also diverge from `dit.rs`'s served-path reduction order, and
both split-K kernels are GPU-only with no `backend-cpu` native path, so shipping
them would have put a silently-all-zero-gradient failure mode on the backend
every test in this crate uses. `mm3_bench gemm-bwd` is the harness that produced
all of the above and is checked in, so the decision is re-measurable rather than
remembered.

### Defect 2 - `0` as an "unregistered" marker in `model::block::KernelIds`

`dit.rs` and `dit_train.rs` both filled the thirteen RMSNorm/RoPE/GQA slots they
never dispatch with `0`. Index 0 is `conv1d` in `dit::PIPELINES` and `matmul` in
`dit_train::PIPELINES` - both live, both dispatched on every pass. A builder
reaching such a slot runs a real kernel against another kernel's bindings and
uniform: on a GPU backend the binding check panics, but `backend-cpu` has no
buffer-count or uniform-size check at dispatch, so it is an out-of-bounds read
that no unit test on that backend can see.

`model::vit::UNREGISTERED = usize::MAX` was already the idiom; `model::block`
now has the twin, plus `KernelIds::slots()` so a gate cannot silently check
15 of 16 fields. **13 sites across the workspace** carried the defect class
(7 with a literal `0`, 6 with a live index for a DIFFERENT kernel used as a
"valid placeholder"): `minimaxmusic3::{dit, dit_train, depth_decoder}`,
`mimi::model`, `qwen3::serve`, `qwen3omnimoe::{talker, thinker}`,
`qwen3tts::{gen, mtp}`, `qwen35::{model, stream}`,
`qwen35::bin::qwen35_bench`, `qwen35moe::model`.

The gate is `dit::slot_gate` + one test per module: run the real pass, read the
device's own per-kernel dispatch counters back, and fail if any unused slot's
index names a kernel that pass dispatched. Mutation-verified - putting a single
slot back to `0` fails both tests by name ("unused KernelIds slot `rmsnorm`
holds pipeline index 0 = `matmul`, which this pass really dispatches").

### What else changed, and why

* `dit_train::Trainer::new_on(gpu, ..)` - the trainer hardcoded
  `Gpu::new_cpu`, so the capability-gated fast tier it now registers was
  unreachable from any caller and `check_dit` could never have exercised it.
  `Trainer::new` still builds on `backend-cpu`, unchanged.
* `gradcheck::minimaxmusic3::check_dit_tiled` - `check_dit` on the pooled test
  device at dims past `block::pick_gemm`'s `m >= 8` / `n >= 128` crossover, so
  the register-tiled backward is finite-difference-checked rather than merely
  compiled. The `check_t5` / `check_t5_tiled` pair is the precedent. Green on
  the P40 and under `BRAIN_DEVICE=cpu`.
* `dit_train::tests::gradients_match_between_the_cpu_and_the_pooled_test_device`
  compares every gradient across the two backends at both shapes, and asserts
  from the dispatch counters that the fast kernels really ran - otherwise the
  "above the crossover" case could quietly become a second copy of the naive one.

### The next row (F.9)

With the GEMMs fixed, one block's marginal 592.8 ms is no longer GEMM-bound
(the nine GEMMs above sum to ~48 ms of it). The remaining cost is the
materialized bidirectional attention trio at 690 rows (`probs` alone is
32x690x690 = 61 MB) and the HOST round trips `block_fwd`/`block_bwd` make per
block - a 45 MB `gpu.read` of `ff_raw` on the forward and two more on the
backward, to split and recombine the fused gate/up halves. `dit.rs` already
removed its own copy of that split (Phase 15's two-half upload); the trainer has
not. Neither is touched here.

## Phase 20: the vocoder's two 1D convolutions were 99.4% of the stage, at 2.2% and 0.4% of roof

`mm3_bench vocoder 689` measured one 8 s chunk at **16780.6 ms, 0.48x
realtime** (best of 5, one P40, real checkpoint, box idle). Two kernels were
essentially all of it:

| kernel | ms | calls | % of pass | % of compute roof |
|---|---|---|---|---|
| `conv1d` | 8317.9 | 27 | 51.4% | **2.2%** |
| `convtr1d` | 7762.2 | 4 | 48.0% | **0.4%** |
| everything else | 97.4 | 72 | 0.6% | at 93-98% of the MEMORY roof |

Both are `@opt 2` one-thread-per-output kernels with a serial `Cin*K`
reduction. A kernel at a few percent of peak is not a kernel that needs tuning
- it is the wrong kernel (§F.2). There is no fast 1D conv sibling anywhere in
the tree: `conv1d.wgsl`/`convtr1d.wgsl` (+ `_dx`/`_dw`) are the only 1D
convolutions there are, `im2col.wgsl` is 2D-only, and `audio::conv` had no
selector at all - so roughly a dozen crates that convolve in 1D were all
running the naive kernel.

### What was done

The 2D side already had the whole pattern (`vae::blocks::conv_s` = `im2col_at`
+ `matmul_reg3` + `nlc_bias_nchw`, chunked, gated on `Cout`). It was ported one
spatial axis down, into a NEW additive seam in `audio::conv`
(`conv1d_bias_fwd` / `convtr1d_bias_fwd` + `ConvGemmKernels` + `ConvScratch`),
with the choice made by `backend_api::select` (`Op::Conv1d`,
`Op::ConvTranspose1d`). `ConvKernels`, `conv1d_fwd`/`convtr1d_fwd` and both
backward builders are untouched, so no existing caller had to change and
`train.rs` keeps the direct path bit-for-bit.

Two new kernels, both barrier-free and both `@cpu yes`: `im2col1d_at.wgsl` and
`col2im1d_bias.wgsl`.

**The algebra needs no transposes and no weight permutes.** That is the whole
reason this is cheap, and it is worth writing down because it is not obvious:

* `conv1d`, `K > 1`: the native `[Cout, Cin/G, K]` weight IS `[Cout, Cin*K]`
  row-major at `G = 1`, so `matmul_reg3` eats the checkpoint tensor as-is:
  `y[Lo, Cout] = col[Lo, Cin*K] . W^T`, epilogue `nlc_bias_nchw` (reused, not
  copied).
* `conv1d`, `K == 1` (stride 1, no pad): `matmul_dx_reg`'s NN form gives
  `y[Cout, Lo] = W[Cout, Cin] . x[Cin, L]` straight over the native NCL
  operands - no im2col at all and no epilogue transpose, since the result is
  already NCL. This covers `dec_in_proj` and all twelve residual `conv2`s.
* `convtr1d`: the TN form contracts over the LEADING axis of both operands,
  which is exactly how `[Cin, Cout/G, K]` and `[Cin, L]` are already laid out.
  `col[Cout*K, L] = W^T.x`, then `col2im1d_bias` gathers the taps.

`matmul_dw_reg_splitk` with `s = 1`, not `matmul_dw_reg`, because the former
ASSIGNS and the latter ACCUMULATES - with a scratch buffer reused across all
eight transposed-conv dispatches an accumulating GEMM would fold the previous
stage in, and zeroing it instead would cost a full extra pass over the largest
buffer in the pipeline.

### Result

`mm3_bench vocoder 689`, best of 5, same binary, same box, A/B by
`BRAIN_CONV1D_GEMM=0`:

| | before (direct) | after (lowered) |
|---|---|---|
| whole pass | 16780.6 ms | **1488.9 ms** (11.3x) |
| sum of kernel device time | 16177.5 ms | **798.9 ms** (20.3x) |
| realtime factor | 0.48x | **5.37x** |
| whole pass vs roof | 1.3% | 14.2% |

Per kernel-kind, after:

| kernel | ms | calls | % | % of roof |
|---|---|---|---|---|
| `matmul_reg3` | 368.7 | 128 | 46.2% | 43.0% compute |
| `im2col1d_at` | 142.7 | 128 | 17.9% | 73.0% memory |
| `matmul_dw_reg_splitk` | 73.4 | 8 | 9.2% | 40.4% compute |
| `matmul_dx_reg` | 70.3 | 26 | 8.8% | 31.7% compute |
| `nlc_bias_nchw` | 41.3 | 26 | 5.2% | 36.1% memory |
| `snake1d` | 37.4 | 29 | 4.7% | 92.9% memory |
| `add2` | 22.6 | 12 | 2.8% | 98.4% memory |
| `col2im1d_bias` | 22.5 | 8 | 2.8% | 33.0% memory |
| `add_chan_inplace` | 16.1 | 14 | 2.0% | 92.7% memory |
| `conv1d` | 3.8 | 1 | 0.5% | (the `Cout = 1` output conv, correctly still direct) |

So `conv1d`'s 8317.9 ms became 626.8 ms across five kinds (**13.3x**) and
`convtr1d`'s 7762.2 ms became 95.9 ms across two (**80.9x**). No row is flagged
as a defect any more: the top one is a GEMM at 43% of the compute roof.

### The threshold was NOT the 2D one, and copying it would have cost a win

`vae::blocks`'s `GEMM_CONV_MIN_COUT` is 32 for the same `matmul_reg3`. Swept
here (`crates/audio/tests/bench_conv1d_lowering.rs`, `--ignored`, best of 5,
`BRAIN_CONV1D_GEMM=force` to reach the sub-threshold side - without it the
selector answers "direct" for both columns and the ratio reads a meaningless
1.0x):

| Cout | 4 | 8 | 12 | 16 | 24 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| `conv1d` k=7 pad=3, L=44096 | 0.27x | 0.54x | 0.85x | **1.06x** | 1.88x | 2.52x | 5.68x | 10.3x | 15.4x |
| `conv1d` k=1, L=44096 | 0.71x | 0.74x | 0.89x | **1.53x** | 2.66x | 3.36x | 5.13x | 14.2x | 20.6x |
| `convtr1d` stride=4 K=8, L=11024 | **1.35x** | 2.47x | 3.18x | 3.69x | 6.01x | 8.15x | 20.1x | 35.3x | 64.7x |

The 1D crossover is between 12 and 16, not 32, because the *baseline* is
different: the 2D sweep's "direct" side is `conv_bias_reg`, an `@opt 5`
register-tiled conv at ~700 GFLOP/s, while the 1D one is a naive kernel at 2.2%
of roof. A much weaker baseline crosses over much earlier. And the transposed
pair needs its OWN number (§F.6): its direct kernel throws away all but
`K/stride` of the taps every thread walks, so the lowering wins at every width
measured. Hence `GEMM_CONV1D_MIN_COUT = 16` and `GEMM_CONVTR1D_MIN_COUT = 4`.

### Correctness

`tests/vocoder_parity.rs` (which now asserts cosine >= 0.999 AND rel_l2 <=
1e-4), both stages actually run:

    vocoder[tiny]: cosine=1.000000000 rel_l2=5.499e-7 max_abs=0.000000
    vocoder[real]: cosine=1.000000000 rel_l2=1.676e-6 max_abs=0.000001

An unexpected measured property, recorded because the next reader will wonder:
the lowered `conv1d` came out **bit-identical** to the direct kernel at every
shape in the sweep (`max|delta|` exactly 0.0). The `matmul_reg*` family
accumulates strictly in increasing `k`, one FMA at a time, and the col operand
is laid out `ci*K + kw` - the same order `conv1d.wgsl`'s nested loops sum in.
`convtr1d` genuinely does reassociate (the GEMM sums over `Cin` and
`col2im1d_bias` adds the taps afterwards, where the direct kernel nests the
other way), so the 1.676e-6 above is entirely its doing. This is a measured
property of this driver, not a contract - do not build a bit-exact gate on it.

### What did NOT work, so it is not re-proposed

* **Chunking the transposed lowering.** Its TN GEMM's output rows index
  `Cout*K`, so a range of `L` is a strided slice of both `col` and the input,
  not a sub-range - and `step_sliced` binds ranges, not strides. Every
  reorientation that makes `L` the row axis needs one of the two operands
  transposed. So that lowering is BOUNDED instead (`Cout*K*L` floats, 271 MB at
  this decoder's worst stage) and falls back to the direct kernel where it
  would not bind. The plain conv IS chunked, because there positions are the
  row axis of both operands - which is the whole reason `im2col_at`'s
  orientation was chosen in 2D too.
* **A bigger im2col budget.** Swept 32/48/64/96/128/256 MiB: device time is
  flat from 96 MiB up (794-800 ms) and climbs below it (+6% at 64, +11% at 32),
  so 128 MiB is the smallest budget that costs nothing. 512 MiB (the VAE
  default) buys nothing and costs a third of a gigabyte of resident scratch.
* **Reading the whole-pass number to tune the budget.** It moved 1484-1740 ms
  across every configuration including repeats of the same one - the host gap
  below swamps a 2% effect. The device-time sum is the only instrument fine
  enough for that decision.

### Where the time is now

The pass is 1488.9 ms of which only 798.9 ms is device kernel time. The
remaining **690 ms is host: allocation, weight upload and readback**, and it is
NOT new (it was 603 ms before this change, on a 16.8 s pass where nobody could
see it). Measured against length - 183 ms at 64 latents, 260 ms at 172, 641 ms
at 689 - it is ~155 ms constant plus ~0.78 ms per latent frame. The constant is
the ~216 MB of conv weights `vocoder::forward` re-uploads on EVERY call
(`gpu.storage_init` per conv, per forward); the linear part is that the forward
allocates a fresh `gpu.storage` for every intermediate and holds all of them,
because it records one tape and submits at the end - several GB of buffer
creation per chunk. The fix is an activation pool and a weight cache, exactly
what `vae::blocks::Builder` already has; it is now 46% of this stage and the
obvious next target.

## Not yet done

- [ ] Charge the SECOND card. Needs `residency::multi` to grow a
      degrade-to-fewer-devices placement and auto-eviction, and
      `crates/stats`' `ModelStat` to grow a `devices: Vec<(Device, u64)>`
      shape, before `MultiDeviceResidentModel` is the right answer here.
- [ ] A card-PAIR device pool, so a 4+-card box runs `floor(N/2)`
      generations concurrently. `generate` takes no pair argument today;
      `ar_branch_devices` and `devplan::auto` both read the whole ambient
      set unconditionally.
- [ ] Cross-request AR batching through `model::serve::PagedDecoder` - the
      seam `qwen3::serve::Engine` implements for continuous batching.
      Assessed, deliberately not started: it is a `crates/qwen3` change,
      not an adapter change.
- [ ] A warm Global LLM (a capacity-max policy plus a KV-reset seam
      `qwen3` does not expose), which is the last per-call load left.
- [ ] Per-stage achieved-vs-roof in `StatsSnapshot` (design in the Phase 17
      handover): `gpu_core::roof` already measures the ceiling and
      `mm3_bench` already computes % of roof per kernel; what is missing is
      a live counter on the serving path, emitted through `Instance::
      metrics` into the executor's `extra` map, never a hardcoded count.
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
