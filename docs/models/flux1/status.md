# FLUX.1 / Kontext (`crates/flux1`) — status ledger

Chronological, measured-only. Reference material lives in
`/…/resources/flux1-kontext/` (outside the repo); goldens in
`testdata/flux1/kontext-dev/` (gitignored).

**What this crate is today: one transformer evaluation.** There is no sampler
loop, no VAE glue, no text-encoder call, no CLI subcommand and no serving
surface. "FLUX.1 works" is not a claim this ledger supports.

## P0–P1 (2026-08) — goldens, import, forward parity

Goldens: `tools/goldens/flux1_dump_reference.py`, forward hooks during a real
`FluxKontextPipeline` run, every block boundary captured, for both a t2i and a
one-reference edit run.

- **Import**: diffusers **1160 tensors → 780 BFL-canonical** fused tensors
  (20 + 19·24 + 38·8 = 780; diffusers 20 + 19·32 + 38·14 = 1160), two-way
  covered — every manifest name present with the exact shape, `map.len() ==
  manifest.len()` so extras are rejected, and any unrecognized source tensor is
  an error by name. No zero-fill path exists.
- Reduced-depth truncation drops 1048 of the 1160 as out-of-depth.
- **Kontext ids**: reference images get axis-0 = 1, verified against
  `FluxKontextPipeline.prepare_latents`, which does exactly
  `image_ids[..., 0] = 1` — *not* FLUX.2's `10·(i+1)`.
- **Modulation order**: the checkpoint emits `(shift, scale, gate)` triples and
  `film_row` reads `(scale, shift)`; the rows are permuted **at build time**, so
  no runtime shuffle exists. BFL and diffusers `AdaLayerNormZero{,Single}` both
  emit `(shift, scale, gate)`; only `norm_out` differs (diffusers
  `AdaLayerNormContinuous` chunks `(scale, shift)`), which import swaps to BFL
  order and `swap_pair` swaps back — a double swap, pinned by a unit test with
  distinct per-slot fills.

### Forward parity — measured (Tesla P40, `BRAIN_DEVICE=gpu0`, `--release`)

**Two gates, because the fp32 model does not fit one card.**

**Gate 1 — reduced depth (2 double + 2 single), fp32, real Kontext-dev weights.**

| stage | t2i 1−cos | edit 1−cos |
|---|---|---|
| `temb` | 1.819e-13 | 1.670e-13 |
| `db0_txt` / `db0_img` | 9.012e-12 / 1.064e-11 | 7.580e-12 / 7.162e-12 |
| `db1_txt` / `db1_img` | 4.534e-12 / 2.381e-12 | 4.192e-12 / 2.759e-12 |
| `sg0_txt` / `sg0_img` | 1.001e-11 / 1.462e-11 | 9.121e-12 / 5.384e-12 |
| `sg1_txt` / `sg1_img` | 1.511e-11 / 6.458e-12 | 1.296e-11 / 7.244e-12 |
| `pre_final` | 6.458e-12 | 7.244e-12 |
| **`out`** | **0.999999999996** (max_abs 3e-5) | **0.999999999993** (max_abs 8e-5) |

Worst stage anywhere: `sg1_txt`, 1−cos **1.511e-11**.

**Gate 2 — full depth (19 + 38), int8, one P40** (`BRAIN_FLUX1_FULL=1`):

| | t2i | edit |
|---|---|---|
| `out` | **0.998544477641** (max_abs 0.40522) | **0.999137355865** (max_abs 0.61119) |
| `pre_final` | 0.998047889316 | 0.997514888243 |
| worst stage | `db18_img` **0.994796217798** | `sg36_txt` **0.989315144936** |

**The full-depth fp32 number is NOT measured and is not claimed** — 47.6 GiB of
weights against a 24 GiB card. int8 brings the 12 B model to ~12 GiB (measured
peak 13 557 MiB during upload, 12 561 MiB steady).

## Shared code — what this port did NOT copy

`dit::rope`, `model::block::flash_bidir_step`, `model::block::rms_variant`,
`model::block::gemm_variant`, `model::int8::{quantize_weight,
quant_rows_steps}`, `model::hostmath::timestep_embedding`, and the kernels
`layernorm(_rows)`, `film_row`, `gate_row`, `bias_add`, `gelu`, `rmsnorm_rows`,
`rope_interleave_table`, `pack_qkv`, `matmul_reg3`, `matmul_gemv`,
`matmul_i8_dyn`, `matmul_i8_gemv`. **No kernel is added.**

`model::block::gemm_variant` is the skinny-M GEMM selection rule, hoisted here
and shared with flux2. flux1 is the model that needs it: its modulation is
**per-block** (77 linears, ~3.2 B of 11.9 B parameters), so it issues 77 `m = 1`
GEMVs per forward — measured 14 of them at reduced depth 2+2, all taking
`matmul_gemv`. flux2's modulation is global and host-folded, and its measured
minimum M is 512, so it registers no GEMV kernel.

## Deferred — explicitly NOT done

* **Full-depth fp32 parity.** Needs >24 GiB or sharding.
* **Backward / gradcheck** (`check_flux1`). No new kernel is needed. The FD
  oracle must **not** call `hostmath::timestep_embedding` — copy
  `flux2::modelgrad`'s generic-`T` pattern (the AGENTS.md exception-1 twin).
  `bias_add` is in-place and `film_row`→`gate_row` reuse `x0`/`x1`, so a
  training forward needs fresh buffers per stage.
* **The `!fast` attention path** (`attn_scores_bidir` / `_softmax_` / `_apply_`)
  has **no permanent gate** — it is dispatched only where
  `workgroup_reductions == false` and no test exercises it numerically. Checked
  by hand once (CPU vs GPU tiny forward agreed to ~1.5e-5 relative); nothing
  keeps it correct. flux1 is also not in `make parity`.
* **`SMOKE_STEPS`** is read from the environment on *every* forward and
  truncates the dispatch list — a debug backdoor producing plausible-looking
  wrong output with no error. flux2 has the identical line; both should be
  `#[cfg(debug_assertions)]` or read once.
* **RoPE tables are rebuilt every forward**: 3×4096×(d/2) = 262 144
  `powf`+`sin`+`cos` per call, of which ~49 152 are used at 768 tokens. Host
  only, no correctness impact, same in flux2. Profile before touching
  (`docs/kernel-checklist.md` §E).
* **Serving contract**: no `Provider`, no residency adapter, no `run_batch`, no
  D-Bus, no example, no CLI. Budget from the **measured 13 557 MiB**, not the
  weight-only figure; `run_batch` needs the batched forward first, and per-block
  modulation makes `modout` `[B · sites · 3D]` — `film_row`/`gate_row`'s
  `rows_per_cond` groups already express that, so still no new kernel.
* **Perf**: not profiled. No speed claim is made.
* Untested knob worth one experiment if the int8 tier ever needs to be better:
  `BRAIN_FLUX1_I8_KEEP_F32=_mod.lin,modulation.lin,adaLN` — the txt stream is
  the weak one (cosine 0.989) and modulation multiplies every token.
