# minimaxh3 - roadmap

MiniMax-H3: a 33B joint video+audio rectified-flow diffusion transformer -
**one packed self-attention stack**, not a two-stream architecture like
`ltxv`. A single 50-block transformer denoises video and audio latent rows
in the same sequence, under **two independent shifted-sigma Euler schedules**
(`shift=12` video, `shift=3` audio) driven from one forward per step, with
no CFG pass (guidance-distilled). Conditioning comes from a Qwen3-VL text
encoder truncated to an intermediate hidden layer (exact depth: TBD, see
"Convention questions" below), refined by 2 token-refiner blocks before
packing. Video: a causal 3D VAE (24 latent channels) + `1x2x2` patchify.
Audio: a DAC-lineage encoder + BigVGAN decoder (32 latent channels, 32kHz
stereo). Released tasks: `t2va` (text-to-video+audio), `fl2va`
(first/last-frame-conditioned), `ref2va` (up to 12 mixed image/video/audio
references).

The port follows `.agents/rules/porting.md` in order: facts before code,
reference goldens before Rust, two-way import coverage, tiny-config smoke,
then the five-rung parity ladder, never skipped. The phases below are the
approved implementation plan for this port.

**Reference material.** Unlike every prior port in this repo, the math
authority here is **not** MiniMax's own repository: `model_index.json`
sources every component (`MiniMaxH3DiTModel`, `MiniMaxH3VideoVAE`,
`MiniMaxH3AudioVAE`, `MiniMaxH3Qwen3VLHFEncoder`) from **`diffusers`**/
**`transformers`**, both Apache-2.0. The audio VAE additionally ships its
own Apache-2.0/MIT Python source directly inside the checkpoint
(`audio_vae/{minimax_h3_audio_vae,dac_*}.py`) - an exact, redistributable
oracle needing no separate install. MiniMax's own repository (community-
licensed) is used, if at all, only to cross-check facts already verified
against the Apache-licensed sources - never translated into brain directly.
Real checkpoint: `MiniMaxAI/MiniMax-H3` on HuggingFace, `FL2VA/` (tasks
`t2va`+`fl2va`) and `Ref2VA/` (task `ref2va`) partitions.

**Licensing.** MiniMax H3 Community License Agreement: territorial carve-out
(excludes EU/UK/South Korea/US from the ordinary grant), a >$20M/yr revenue
registration clause, mandatory "MiniMax H3" UI attribution. Brain's Rust
*implementation* is Apache-2.0 (ported from the Apache-2.0 diffusers/
transformers reference, never from MiniMax's own community-licensed repo).
The *weights* are never vendored, never auto-fetched (`default_ref: None`,
the SUPIR/FinCast precedent) and gated at runtime by
`minimaxh3::caps::check_license` (`BRAIN_MINIMAXH3_ALLOW_COMMUNITY=1`,
modeled byte-for-byte on `flux2::caps::check_license`). See
`docs/models/minimaxh3.md`.

**Hardware.** 2xTesla P40 (24GB each) + 184GB RAM. Pascal has no bf16
compute; brain's loader demotes BF16->F32 on read, so the 66GB bf16
checkpoint is 132GB fp32-equivalent - too large to hold resident at fp32.
`adaln_proj` is 260M params/block x 50 blocks = **13.0B of the 33B total**
(confirmed from the real shard-0 safetensors header) - AdaLN precompute
(Phase 8) is load-bearing for fitting this model, not an optimization:
folding it into small per-(step,modality) modulation tables turns the DiT
into a ~19.3B backbone that fits fully resident at int8 (~19GB) across both
P40s via `model::shard::plan_fewest_devices`, with no per-step PCIe
streaming required on the critical path. The Qwen3-VL text encoder (a
separate ~33B-class model) cannot be co-resident with the DiT - it runs in
its own sequential phase, wan's proven pattern (encode -> free -> denoise ->
free -> decode).

## Verified ground truth (read directly from the checkpoint, not from prose)

- `model_index.json`: `sigma_shift_scales: {video: 12.0, audio: 3.0}`;
  `tasks: [t2va, fl2va]` for the FL2VA partition (one transformer serves
  both); `scheduler: null` (no bundled CFG scheduler).
- `transformer/config.json`: `hidden_size=5376`, `num_layers=50`,
  `token_refiner_num_layers=2`, `num_attention_heads=56`,
  `attention_head_dim=128` (QKV width `56*128=7168 != 5376` - do not derive
  attention width from `hidden_size`), `ffn_hidden_size=14336`,
  `latents_dim=24` (video), `audio_latents_dim=32`, `patch_size=[1,2,2]`,
  `text_dim=5120`, `timestep_input_dim=256`, `time_embed_hidden_size=5376`,
  `time_embed_dim=2688`, `adaln_out_features=96768` (=18x5376 - which 18
  modulation sites is UNSETTLED, see below), `final_adaln_out_features=10752`
  (=2x5376, shift+scale only, no gate), `rope_inv_freq_len=16`.
- Real shard-0 tensor names (`transformer/model-00001-of-00013.safetensors`
  header; BF16 except norms/embed/rope which are F32):
  `blocks.N.qkv_proj.weight [21504,5376]` (fused QKV, 21504=3x7168),
  `attn.{q,k}_norm.weight [128]` (one 128-vector, shared across the 56
  heads), `attn.out_proj.weight [5376,7168]`, `mlp.fc1.weight [28672,5376]`
  (SwiGLU, 28672=2x14336), `mlp.fc2.weight [5376,14336]`,
  `norm{1,2}.weight [5376]`, `adaln_proj.linear.{weight [96768,2688],bias}`
  per block, `condition_proj.{weight [5376,5120],bias}` (once, Qwen hidden ->
  H3 hidden), `time_embedder.proj_{in,out}` (256->5376->2688, once, shared),
  `token_refiner.blocks.{0,1}.*` (same block shape) +
  `token_refiner.final_norm.weight`, `video_patch_proj.weight [5376,96]`
  (96=24x1x2x2), `audio_patch_proj.weight [5376,32]` (no patchify),
  `rope.inv_freq [16]`.
- `audio_vae/`: fully downloaded (578MB). `latent_channels=32`,
  `sample_rate=32000`, `output_channel=2` (stereo, L/R independently through
  a shared mono VAE), real per-channel `latents_mean`/`latents_std` (32
  values each). Ships its own reference source, see above.
- `text_encoder/`: Qwen3-VL. `hidden_size=5120`, `num_hidden_layers=64`,
  `num_attention_heads=64`, `num_key_value_heads=8`, `head_dim=128`,
  `intermediate_size=25600`, `rope_theta=5000000`, M-RoPE
  `mrope_section=[24,20,20]` interleaved. Vision tower: `depth=27`,
  `hidden_size=1152`, `num_heads=16`, `deepstack_visual_indexes=[8,16,24]`,
  `out_hidden_size=5120`, `patch_size=16`, `spatial_merge_size=2`.
- `video_vae/` not yet downloaded (folder does not exist on disk yet).

## Convention questions - settle from source, not experiment

Per porting.md, none of these are to be assumed from the pasted-doc numbers
that started this port, nor from MiniMax's own (community-licensed) repo
prose - settle each from the Apache-2.0 diffusers source once installed
(Phase 1), and record the answer + citation here as it lands.

- [ ] What are the 18 modulation vectors in `adaln_proj`'s
      96768=18x5376 output? (hypothesis: 2 modalities x 2 sub-layers
      (attn,ffn) x 3 (shift,scale,gate) = 12, leaving 6 unexplained - do not
      guess, read `MiniMaxH3DiTModel`'s block forward.)
- [ ] Is H3's timestep conditioning per-clip-scalar or per-row/per-token
      (diffusion forcing)? This gates whether porting.md's SS7 modulation-fold
      shortcut is even legal - ltxv's precedent is per-token; do not assume
      H3 matches without checking.
- [ ] Packed-sequence row order: token-refiner/condition rows vs video rows
      vs audio rows vs (for fl2va/ref2va) reference rows - which comes
      first, how are boundaries computed and communicated to attention.
- [ ] `rope.inv_freq [16]`: one shared 16-dim table sliced per axis (t,h,w),
      or three independent 16-dim tables summing differently? Confirm
      against the real RoPE application code, not assumed from the LTX/Wan
      precedent (both differ from each other already).
- [ ] `MiniMaxH3Qwen3VLHFEncoder`'s exact truncation layer and whether the
      returned hidden state is pre- or post-final-norm (the pasted doc's
      "layer 50" claim is UNVERIFIED against a 64-layer text config; do not
      cite it until confirmed from the actual encoder class).
- [ ] fl2va's keyframe conditioning convention (packed as extra rows? as a
      `keyframes_mask` analogous to ltxv's? something H3-specific?) - do not
      assume LTX's convention transfers.
- [ ] Whether `text_encoder`/`audio_vae`/`video_vae` are identical across
      the FL2VA/Ref2VA partitions (share loading) or partition-specific,
      once `Ref2VA/model_index.json` is available.
- [ ] video_vae's exact causal-3D-encoder / non-causal-ViT-decoder
      conventions (chunking, cross-chunk cache, padding, norm axis) - per
      the wan/ltxv precedent, these two "causal 3D VAEs" differ from each
      other in nearly every convention despite both being causal 3D VAEs;
      H3's must be independently re-derived, not assumed to match either.

## Phases (see the approved plan for full detail)

- [x] Phase 0 - ledger, arch registration, crate skeleton, license gate
- [ ] Phase 1 - reference oracle (torch+diffusers) + golden dumper
- [x] Phase 2 - fetch recipe for the two-level partitioned checkpoint (`brain
      pull` only - `default_ref: None` still blocks auto-fetch; the
      `crates/cli/src/supply.rs::convert`-side manifest write for this
      recipe id is a recorded gap below)
- [x] Phase 3 - Qwen3-VL `encode_hidden`/`encode_hiddens` extension (text-only
      and `_with_image` twins; found and documented a real landmine along
      the way - `enable_mm_splice`'s row range is positional and
      unconditional per forward once baked in at construction, so the
      text-only path is only safe on an `n_visual=0` instance)
- [~] Phase 4 - audio VAE **decoder** port: structurally complete and
      real-checkpoint-validated, numeric parity still open (see below)
- [ ] Phase 5 - H3 DiT core, tiny-config -> real-weight parity ladder
- [ ] Phase 6 - video VAE (gated on `video_vae/` download)
- [ ] Phase 7 - dual rectified-flow schedulers
- [ ] Phase 8 - AdaLN precompute checkpoint transform
- [ ] Phase 9 - t2va / fl2va / ref2va pipelines
- [ ] Phase 10 - training: gradcheck -> LoRA -> device trainer
- [ ] Phase 11 - capability / residency / D-Bus serving contract
- [ ] Phase 12 - streaming-overlap engine (separate, measured perf phase)

## Phase 4 detail: audio VAE decoder

`crates/minimaxh3/src/vocoder.rs` + `import.rs` - the BigVGAN-topology
decoder half of the audio VAE only (`DacAudioVAE.decode`: `dec_in_proj` then
the BigVGAN stack). **Encode is out of scope** for this milestone - the
checkpoint's own shipped `dac_audio_vae.py` only implements `.decode()` at
all; `encoder`/`mean_proj`/`logs_proj`/`pre_block` (173 of the checkpoint's
1087 tensors) are present for training-time compatibility but dead in the
shipped inference-only forward, and neither reads them.

This reparameterizes an existing in-workspace BigVGAN v2 / AMP1 / anti-
aliased-SnakeBeta vocoder port (same topology, different config numbers) -
genuinely new work was the H3-specific config/manifest and three real,
checked naming differences from that precedent: `decoder.ups.{i}.0.*` (not
flat `ups.{i}.*` - `BigVGAN.__init__` wraps each stage's transposed conv in
its own one-element `ModuleList`), `decoder.resblocks.{idx}.activations.{0..5}`
flat (not split `acts1.{d}`/`acts2.{d}` - `AMPBlock1` stores one 6-element
list and slices it in its own forward), and a top-level `dec_in_proj` that
is a PLAIN (non-weight-normalized) 1x1 conv, unlike every other conv in the
file.

**Verified without any Python oracle** (real checkpoint, no torch needed for
any of this):
- `tensor_manifest()`'s 779 post-fold tensor count matches an independent
  Python count from the real safetensors header exactly (914 raw tensors
  under `decoder.*`+`dec_in_proj.*`, folding each `weight_g`/`weight_v` pair
  to one `.weight`).
- `import_audio_vae_decoder` achieves real two-way coverage against the
  actual 578MB checkpoint: every claimed tensor present and correctly
  shaped, the 173 unclaimed tensors are exactly the expected out-of-scope
  `encoder`/`mean_proj`/`logs_proj`/`pre_block` prefixes (asserted, not
  assumed).
- `decode()` runs the FULL real-scale graph (1024-wide, 7 stages, 21
  resblocks) against the real imported weights end to end on this
  hardware's CPU backend: finite, correctly `[-1,1]`-clamped, non-trivial
  output, ~6s including import.
- A weight-free tiny-config smoke test also exercises the same graph
  end-to-end; its first version had a real bug (channel width `24` right-
  shifted to 0 after 7 halvings, `24 >> 7 == 0`), caught by its own
  "output must not be trivially all-zero" assertion - fixed by widening to
  `128`. Worth recording: the assertion did its job.

**Still open, explicitly not claimed:** numeric parity against the real
reference implementation. No torch/diffusers oracle is installed yet (Phase
1), so "runs and stays finite at real scale" is the honest ceiling on what
is verified so far - it proves the graph is wired and the checkpoint loads
correctly, not that the waveform is right. This gate is what closes Phase 4
for real.

## Recorded gaps (kept current)

- `H3Recipe` (in `modelstore::recipe`) makes `brain pull MiniMaxAI/MiniMax-H3`
  fetch the right nested per-partition files, but `crates/cli/src/
  supply.rs::convert` has no case for recipe id `"minimaxh3"` yet, so a
  completed pull is not yet turned into a servable manifest this way -
  `minimaxh3::import` reading `BRAIN_MINIMAXH3_DIR` directly (a plain local
  checkout, exactly what this session's own download already is) is the
  supported path until that finish-side wiring lands.
- `video_vae/` has not finished downloading; Phase 6 real-weight parity is
  blocked on it.
- `Ref2VA/` partition has not started downloading; Phase 9's `ref2va` task
  real-weight parity is blocked on it.
- No torch/diffusers oracle installed yet (Phase 1 not started) - every
  "convention question" above is genuinely open.
