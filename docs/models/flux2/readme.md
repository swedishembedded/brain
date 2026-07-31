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

# int8 (DP4A) DiT — ~4x smaller weights, GPU only; see the int8 section
brain flux2 generate --prompt "…" --out x.ppm --precision int8
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

## Int8 (DP4A) inference

`--precision int8` (CLI), the `precision` capability param, or
`Pipeline::build_with(…, Precision::Int8)` builds the DiT with every block
linear quantized to int8: per-channel symmetric weights
(`model::int8::quantize_weight` — the ONE engine-wide implementation, shared
with zimage and qwen), dynamic per-token activation quant on-device
(`max_abs_row` → `quant_pack`), and the DP4A GEMM (`matmul_i8_dyn`).
Norms/RoPE/attention/SwiGLU stay f32. One activation quant feeds every linear
reading it (row-range slicing included — sliced `step_sliced` views work for
the packed buffers because every row offset is 0 or `txt_len` rows, 256-byte
aligned for all variants).

Three-and-a-half things stay fp32, each justified by a measured bisection on
the parity fixture (`docs/models/flux2/status.md` has the table): `txt_in`
(its input is raw Qwen3 hidden states whose channel outliers crush a per-token
int8 scale — quantizing it alone costs cosine 0.995 → 0.984), `img_in` +
`final_layer.linear` (boundary insurance, 3 MB), and the double-block mlp-down
(`*_mlp.2`, SwiGLU-activation outliers early in the stack: 0.9965 → 0.9989 for
~850 MB). Everything else — double-block qkv/proj/mlp-up, the whole
single-block `linear1`/`linear2` — is int8. Weights drop ~15.5 GiB → ~4.8 GiB
(int8 DiT ~5.1 GiB resident on the P40 vs 14.7 GiB fp32).

The text encoder has its own tier: `BRAIN_FLUX2_TE_DEVICE=gpu<i>` places the
truncated fp32 shard (layers 0..=27, ~16 GiB resident on a non-ReBAR P40);
`gpu<i>:i8` uses `Qwen::new_shard_i8` instead so **int8 DiT + int8 TE fit one
24 GB card**. Parity + timing tables: `status.md` §P8.

## Prompting the edit path (measured)

Klein is an *instruction*-trained editor. Prompt format materially changes the
result — measured on a B&W photo at 768×1056, `luma-corr` = structural fidelity
to the source, `sat` = mean saturation (did it actually add colour):

| prompt / mode | luma-corr | sat |
|---|---:|---:|
| long scene description, from noise | 0.657 | 0.511 |
| **`"Colorize this photograph."` (instruction), from noise** | **0.853** | **0.414** |
| instruction + `--strength 0.1` (img2img init) | 0.999 | 0.012 |
| instruction + `--strength 0.5` | 0.975 | 0.011 |
| instruction + `--strength 0.9` | 0.768 | 0.020 |

Two rules follow:

1. **Use short imperative instructions referring to the image**
   (`"Colorize this photograph."`, `"Make it snow."`), not a description of the
   scene you want. Same model, same seed: fidelity 0.657 → 0.853 with colour
   retained. A description is a text-to-image prompt and the model treats it
   as one.
2. **`--strength` trades fidelity for freedom, monotonically** (0.1 → 0.999,
   0.5 → 0.975, 0.9 → 0.768), but it does **not** buy colour: saturation stays
   ≈0.01–0.02 across the whole range while fidelity collapses. The init latent
   carries the source's greyness as content, and Klein is guidance-distilled,
   so there is no CFG to weight the prompt against that evidence. Use
   `--strength` for structure-preserving *tonal/texture* edits; use
   reference-only (from noise) when the edit must change colour, and accept
   ~0.85 structural fidelity.

   **Implementation note (a bug worth not repeating):** the init image must NOT
   also be passed as a conditioning reference token. Doing so pins the output
   at ~0.999 for *every* strength — the dial looks inert because the reference
   tokens override the noise level. `Pipeline` therefore consumes the first
   reference as the init and skips it in both the token builder and
   `position_ids`; extra references still ride along as edit context. Side
   benefit: the joint sequence halves (6848 → 3680 tokens here), so img2img is
   also 2.4× faster (96 s → 40 s at 8 steps). Faithful colorization needs CFG (the undistilled
   `base-4b`), a colorization LoRA (`brain flux2` can train one), or a
   purpose-built model.

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
