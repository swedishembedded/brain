# FLUX.2 Klein in brain

Black Forest Labs' FLUX.2 Klein text-to-image + image-editing family (4B
Apache-2.0, 9B non-commercial), implemented from scratch on brain's WGSL
engine: an MMDiT-style rectified-flow transformer (double-stream img/txt
blocks with joint attention, then single-stream parallel blocks) over the
FLUX.2 autoencoder's 128-channel latent, conditioned on three concatenated
Qwen3 hidden states. Klein variants are step+guidance-distilled (4 Euler
steps, no CFG); the `base` variants are undistilled (50 steps + real CFG)
with byte-identical tensor layout.

## Quick start

```bash
# weights: the ungated klein-4B diffusers layout (transformer/, vae/,
# text_encoder/, tokenizer/) — see resources/flux/README.md for sources
export BRAIN_FLUX2_DIT=…/FLUX.2-klein-4B/transformer
export BRAIN_FLUX2_VAE=…/FLUX.2-klein-4B/vae
export BRAIN_FLUX2_TE=…/FLUX.2-klein-4B/text_encoder
export BRAIN_FLUX2_TOKENIZER=…/FLUX.2-klein-4B/tokenizer/tokenizer.json

brain flux2 generate --prompt "a red fox on a mossy rock" --out fox.ppm \
    --width 512 --height 512 --seed 7            # 4 steps, no CFG (klein)
brain flux2 generate --prompt "make it snow" --ref fox.ppm --out snow.ppm  # editing
```

## Architecture (as brain sees it)

| | Klein 4B | Klein 9B |
|---|---|---|
| hidden / heads / head_dim | 3072 / 24 / 128 | 4096 / 32 / 128 |
| double / single blocks | 5 / 20 | 8 / 24 |
| text conditioning | Qwen3-4B layers [9,18,27] concat → 7680 | Qwen3-8B → 12288 |
| RoPE | 4 axes × 32 dims (t,h,w,l), theta **2000**, interleaved pairs | same |
| modulation | **3 global** linears (not per-block), chunk (shift, scale, gate) | same |
| MLP | SwiGLU ratio 3.0, gate = first half; single blocks fuse qkv+mlp in `linear1` | same |
| norms | LayerNorm affine-free eps 1e-6; QK-RMSNorm eps 1e-6 | same |
| latent | VAE 32 ch × 2×2 unshuffle = 128 ch, 16× downscale, patch_size 1 | same |
| latent norm | per-channel eval BatchNorm (`bn.running_*`), eps 1e-4 | same |
| schedule | Euler over `exp(mu)/(exp(mu)+(1/t−1))`, `mu = empirical(seq_len, steps)` | same |
| editing | refs VAE-encoded at own size, RoPE t = 10·(i+1), tokens appended, pred truncated | same |

Everything is bias-free in the DiT. Text pads (right-padded to 512 with
`<|endoftext|>`) participate in DiT attention **un-masked** but are computed by
the text encoder **with** key masking — brain reproduces this via
`Qwen::encode_hiddens_padded` (the `gqa_scores_kmask` kernel).

## How it is built (brain mapping)

- `crates/flux2` — `config.rs` (presets + tensor manifest), `import.rs`
  (BFL-canonical names; diffusers layout re-fused/remapped incl. the
  `norm_out` shift/scale half-swap; GGUF via `checkpoint::gguf`), `model.rs`
  (the GPU forward: one joint residual slab, per-stream row ranges via
  `step_sliced`, global modulation folded into LayerNorm gamma/beta — 6 LN
  sites + 5 gates per forward), `pipeline.rs` (text→DiT→VAE, editing, CFG for
  base variants), plus training (`grad.rs`/`modelgrad.rs`/`lora.rs`/
  `finetune.rs`).
- Shared crates: `dit::rope` (4-axis interleaved tables),
  `diffusion::scheduler` (`empirical_mu`, `klein_sigmas`), `vae`
  (FLUX.2 config: quant convs + `latent::pack/unpack` boundary), `qwen`
  (multi-layer masked-pad encoder), `checkpoint` (safetensors + GGUF).
- Kernels: composed entirely from pre-existing WGSL except one addition,
  `gqa_scores_kmask` (additive key mask for padded encoder batches).

## Parity (the gate)

Stage-by-stage vs the diffusers/BFL reference on dumped goldens
(`tools/flux2_dump_reference.py`, `testdata/flux2/klein-4b/`) — see
`status.md` for the measured table. Gradient checking: in-crate FD tests +
`gradcheck::check_flux2`.

## Licensing

Klein 4B + the FLUX.2 VAE + Qwen3 encoders: Apache 2.0 (commercial OK).
Klein 9B / base-9B: FLUX Non-Commercial License v2.1 — research only,
content-filter-or-review obligation, attribution required when redistributing
derivatives (including converted checkpoints). Keep 9B usage behind explicit
opt-in and ship the license text alongside any converted 9B artifact.
