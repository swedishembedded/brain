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
   `qwen_7B/qwen_7B/config.json`, not the smaller published Qwen3-8B preset),
   reused verbatim from `crates/qwen3`. Autoregressive, CFG-guided: one
   semantic RVQ code per 25 Hz frame.
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
Real weights for the DiT (9.7 GB) and the Global LLM (17.2 GB via the
pre-split `language_model/`, or 18.5 GB via `qwen_7B/` - the pre-split dir
is simpler and is what those milestones will use, not the manual key-split
this roadmap's Phase 0 draft assumed) are deferred to their own milestones.

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

## Not yet done

- [ ] Vocoder LoRA fine-tune (the backward this phase landed makes it
      straightforward - not yet wired)
- [ ] The multi-scale STFT/mel discriminator + adversarial + feature-
      matching training loop this model's training scope actually needs -
      the real new-capability item (`crates/mimi::recon`'s module doc
      lists this stack as absent workspace-wide; this closes it, scoped
      to what this vocoder needs, not generalized into `crates/mimi`)
- [ ] RVQ depth decoder: import + forward/backward + gradcheck + LoRA
- [ ] Flow-matching DiT: import + forward/backward + gradcheck + LoRA +
      int8 storage tier + pipeline sharding
- [ ] Global LLM: streamed import via `crates/qwen3` + an audio-code
      cross-entropy training objective
- [ ] Pipeline: prompt assembly, two-axis CFG (AR logits + DiT
      zero-conditioning), chunk windowing/overlap-splice, vocoder
      crop-and-stitch - one real short end-to-end WAV on this machine
- [ ] Serving contract: capability manifest, residency adapter, CLI verb,
      D-Bus, a runnable example

## Recorded gaps (expected, not yet reached)

- No full 5-minute generation on this machine (no discrete GPU, ~21 GB
  usable RAM) - only a short single-chunk generation is exercised here.
- No NPU export path planned in the initial port.
- No multi-GPU shard parity unless a second GPU is available when that
  milestone lands.
