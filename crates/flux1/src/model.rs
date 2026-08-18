// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The FLUX.1 transformer forward: 19 double-stream blocks (separate img/txt
//! weights, joint attention over the concatenated `[txt | img]` sequence) then
//! 38 single-stream parallel blocks, with **per-block** modulation.
//!
//! ## What differs from `crates/flux2`, and what is shared
//!
//! Shared verbatim (no second copy): `dit::rope` (the multi-axis interleaved
//! RoPE table build — 3 axes at theta 10000 here instead of 4 at 2000),
//! `model::block::flash_bidir_step` (joint bidirectional attention),
//! `model::int8` (per-channel symmetric weight packing + DP4A GEMM),
//! `model::hostmath` (the conditioning mat-vecs), and the kernels
//! `layernorm(_rows)`, `film_row`, `gate_row`, `bias_add`, `gelu`,
//! `rmsnorm_rows`, `rope_interleave_table`, `pack_qkv`, `matmul_reg3`,
//! `matmul_gemv`, `matmul_i8_dyn`, `matmul_i8_gemv`. **No kernel is added.**
//!
//! Different, and why the code cannot simply be flux2's:
//!
//! * **Per-block modulation.** FLUX.2 has three *global* modulation linears, so
//!   it folds `(1+scale)·LN(x)+shift` into six LayerNorm `(gamma, beta)` pairs
//!   computed once on the host. FLUX.1 has 2 per double block + 1 per single
//!   block + 1 final = 77 modulation linears totalling ~3.2 B of the model's
//!   11.9 B parameters — 13 GiB at fp32, far too much to mat-vec on the host
//!   every forward. They stay on the **device** and run as 77 `m = 1` GEMVs
//!   into one `[Σ sites · 3D]` output buffer; each modulated LayerNorm is then
//!   `layernorm` (no affine) → `film_row` (`x·(1+s)+b`), which is also the
//!   *unfolded* form the reference evaluates.
//!
//!   `film_row` reads `(scale, shift)` in that order from one packed buffer,
//!   while the checkpoint emits `(shift, scale, gate)` triples — so the
//!   modulation **weight rows are permuted at build time** (host side, exactly
//!   like the qkv split), and no runtime shuffle exists.
//! * **Biases everywhere.** Every FLUX.1 linear is biased (FLUX.2's are not);
//!   each matmul is followed by `bias_add` over the same row range.
//! * **GELU(tanh) MLPs**, not SwiGLU: `mlp_hidden = 4D`, the single block's
//!   `linear1` emits `3D + mlp` (not `3D + 2·mlp`), and the activation is one
//!   `gelu` dispatch instead of `silu_mul`.
//! * **T5 + CLIP conditioning**: the modulation vector is
//!   `time_in(t) + guidance_in(g) + vector_in(clip_pooled)`, and `txt_in` reads
//!   a T5-XXL `[txt, 4096]` sequence whose length is a *runtime* argument.
//!
//! ## Layout
//!
//! One joint residual slab `[n, D]`, **text rows first** (`0..n_txt`) then
//! image (and, for Kontext, reference) rows — the reference's own `cat(txt,
//! img)` order, which is what makes joint attention a single full-slab
//! dispatch and the per-stream ops `step_sliced` row ranges. Fused checkpoint
//! weights (`qkv`, `linear1`, `linear2`) are split at build time so every
//! matmul is a plain full-buffer dispatch.
//!
//! ## Kontext edit path
//!
//! Reference images are VAE-encoded, packed, and appended to the image tokens;
//! their position ids carry axis-0 = 1 (verified against
//! `FluxKontextPipeline.prepare_latents`, which does exactly
//! `image_ids[..., 0] = 1` — *not* FLUX.2's `10·(i+1)`). Attention is full and
//! bidirectional over the whole concatenation; the caller truncates the
//! prediction to the noise span via `n_pred`.

use gpu_core::{DeviceBuffer, Gpu, Step};

use crate::config::Flux1Config;
use crate::import::Tensors;

pub const KERNELS: &[(&str, &str)] = &[
    ("layernorm", kernels::LAYERNORM),
    ("layernorm_rows", kernels::LAYERNORM_ROWS),
    ("matmul", kernels::MATMUL),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("matmul_gemv", kernels::MATMUL_GEMV),
    ("rmsnorm_eps", kernels::RMSNORM_EPS),
    ("rmsnorm_rows", kernels::RMSNORM_ROWS),
    ("rope_interleave_table", kernels::ROPE_INTERLEAVE_TABLE),
    ("pack_qkv", kernels::PACK_QKV),
    ("attn_scores_bidir", kernels::ATTN_SCORES_BIDIR),
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
    ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR),
    ("flash_attn_bidir", kernels::FLASH_ATTN_BIDIR),
    ("flash_attn_bidir_split", kernels::FLASH_ATTN_BIDIR_SPLIT),
    ("gelu", kernels::GELU),
    ("film_row", kernels::FILM_ROW),
    ("gate_row", kernels::GATE_ROW),
    ("bias_add", kernels::BIAS_ADD),
    // int8 DP4A path (GPU only): per-token activation quant + DP4A GEMM.
    ("max_abs_row", kernels::MAX_ABS_ROW),
    ("quant_pack", kernels::QUANT_PACK),
    ("matmul_i8_dyn", kernels::MATMUL_I8_DYN),
    ("matmul_i8_gemv", kernels::MATMUL_I8_GEMV),
    ("flash_attn_bidir_reg", kernels::FLASH_ATTN_BIDIR_REG),
    ("flash_attn_bidir_reg2", kernels::FLASH_ATTN_BIDIR_REG2),
];
const K_LN: usize = 0;
const K_LN_ROWS: usize = 1;
const K_MATMUL: usize = 2;
const K_MATMUL_REG3: usize = 3;
const K_MATMUL_GEMV: usize = 4;
const K_RMSNORM: usize = 5;
const K_RMSNORM_ROWS: usize = 6;
const K_ROPE: usize = 7;
const K_PACK: usize = 8;
const K_SCORES: usize = 9;
const K_SOFTMAX: usize = 10;
const K_APPLY: usize = 11;
const K_FLASH: usize = 12;
const K_FLASH_SPLIT: usize = 13;
const K_GELU: usize = 14;
const K_FILM: usize = 15;
const K_GATE: usize = 16;
const K_BIAS: usize = 17;
const K_MAXABS: usize = 18;
const K_QUANT: usize = 19;
const K_MATMUL_I8: usize = 20;
const K_MATMUL_I8_GEMV: usize = 21;
/// The register-tiled flash-attention pair, appended so every index above is
/// unchanged. `model::block::flash_bidir_variant` picks between all four from
/// the device's queried caps.
const K_FLASH_REG: usize = 22;
const K_FLASH_REG2: usize = 23;

fn f(x: f32) -> u32 {
    x.to_bits()
}

// DiT numeric tier — `model::dispatch::Precision`, shared with flux2 (the
// ONE name→tier map). For THIS model: fp32 is the parity reference (47.6 GiB
// of weights at full depth — it does NOT fit one 24 GiB card, so the fp32
// gate runs at reduced depth); int8 quantizes every linear except the ones
// named in [`Flux1Model::new_with`] and brings the full 12 B model to
// ~12 GiB. Norms / RoPE / attention / GELU always stay f32.
pub use model::dispatch::Precision;

// The resident-weight representation (fp32 | packed int8 + per-channel
// scale) is shared with flux2 via `model::dispatch`.
use model::dispatch::LinW;

/// One biased linear. Every FLUX.1 linear has a bias, so it lives here rather
/// than at each call site.
struct Lin {
    w: LinW,
    b: DeviceBuffer,
}

impl Lin {
    fn is_i8(&self) -> bool {
        self.w.is_i8()
    }
}

/// One stream's weights inside a double block (a block holds img and txt).
struct StreamW {
    wq: Lin,
    wk: Lin,
    wv: Lin,
    nq: DeviceBuffer,
    nk: DeviceBuffer,
    wo: Lin,
    /// GELU MLP: `[mlp, D]` then `[D, mlp]`
    w1: Lin,
    w2: Lin,
}

struct DoubleW {
    img: StreamW,
    txt: StreamW,
    /// modulation linears; their output lands in the shared `modout` buffer
    mod_img: Lin,
    mod_txt: Lin,
    /// float offsets into `modout` of this block's img / txt 6D triples
    off_img: u64,
    off_txt: u64,
}

struct SingleW {
    wq: Lin,
    wk: Lin,
    wv: Lin,
    nq: DeviceBuffer,
    nk: DeviceBuffer,
    /// `linear1`'s mlp quarter `[mlp, D]`
    wm: Lin,
    /// `linear2` column split: `out = wo_a @ attn ⧺ wo_b @ gelu(mlp)`; the
    /// single `[D]` bias rides on `wo_a` and the sum picks it up exactly once.
    wo_a: Lin,
    wo_b: LinW,
    modl: Lin,
    off: u64,
}

struct Scratch {
    x0: DeviceBuffer,
    x1: DeviceBuffer,
    n1: DeviceBuffer,
    n2: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    qn: DeviceBuffer,
    kn: DeviceBuffer,
    qr: DeviceBuffer,
    kr: DeviceBuffer,
    qkv: DeviceBuffer,
    ctx: DeviceBuffer,
    proj: DeviceBuffer,
    mlpo: DeviceBuffer,
    h1: DeviceBuffer,
    hs: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    out: DeviceBuffer,
    cos: DeviceBuffer,
    sin: DeviceBuffer,
    tok_in: DeviceBuffer,
    ctx_in: DeviceBuffer,
    /// `silu(vec)` — the modulation input, one row of `[D]`
    vecb: DeviceBuffer,
    /// every modulation site's `(scale, shift, gate)` triples, back to back
    modout: DeviceBuffer,
    /// constant `1` / `0` affine params: the LayerNorms are affine-free and the
    /// modulation is applied by `film_row`, the reference's own unfolded form.
    ln_gamma: DeviceBuffer,
    ln_beta: DeviceBuffer,
}

// Int8 activation scratch — `model::dispatch::I8Scratch`, keyed by the two
// in-block contraction widths (hidden, mlp), shared with flux2.
use model::dispatch::I8Scratch;

/// A per-stage tap of the residual slab, for the parity ladder.
pub struct Trace {
    /// `(name, [n, D])` slab snapshots in dispatch order: `db{n}`, `sg{n}`.
    pub stages: Vec<(String, Vec<f32>)>,
    /// image rows entering `final_layer` — the reference's `pre_final`.
    pub pre_final: Vec<f32>,
    /// the conditioning vector `vec` before the final `silu` — `temb`.
    pub temb: Vec<f32>,
}

pub struct Flux1Model {
    pub cfg: Flux1Config,
    gpu: Gpu,
    n_max: u32,
    fast: bool,
    precision: Precision,
    i8scr: Option<I8Scratch>,
    dbl: Vec<DoubleW>,
    sgl: Vec<SingleW>,
    img_in: Lin,
    txt_in: Lin,
    final_w: Lin,
    final_mod: Lin,
    final_off: u64,
    scr: Scratch,
    // host-side conditioning weights (small: ~30 M MACs per forward)
    time_a: (Vec<f32>, Vec<f32>),
    time_b: (Vec<f32>, Vec<f32>),
    vec_a: (Vec<f32>, Vec<f32>),
    vec_b: (Vec<f32>, Vec<f32>),
    guid_a: Option<(Vec<f32>, Vec<f32>)>,
    guid_b: Option<(Vec<f32>, Vec<f32>)>,
}

/// Rows of a `(shift, scale, gate)`-triple modulation weight/bias reordered to
/// `(scale, shift, gate)` — the order `film_row` reads `(s, b)` in.
///
/// `rows` is the row stride (`cols` for a weight, 1 for a bias) and `triples`
/// the number of triples (2 for a double-stream `[6D, D]`, 1 for a single
/// block, and a *pair* for the final layer, handled by [`swap_pair`]).
fn permute_triples(w: &[f32], d: usize, cols: usize, triples: usize) -> Vec<f32> {
    let blk = d * cols;
    let mut out = Vec::with_capacity(w.len());
    for t in 0..triples {
        let base = t * 3 * blk;
        out.extend_from_slice(&w[base + blk..base + 2 * blk]); // scale
        out.extend_from_slice(&w[base..base + blk]); // shift
        out.extend_from_slice(&w[base + 2 * blk..base + 3 * blk]); // gate
    }
    out
}

/// `final_layer.adaLN_modulation` is a `(shift, scale)` PAIR, not a triple.
fn swap_pair(w: &[f32], d: usize, cols: usize) -> Vec<f32> {
    let blk = d * cols;
    let mut out = Vec::with_capacity(w.len());
    out.extend_from_slice(&w[blk..2 * blk]);
    out.extend_from_slice(&w[..blk]);
    out
}

impl Flux1Model {
    /// Build device state from imported (BFL-named) tensors, sized for at most
    /// `n_max` joint tokens (`txt + image + reference`). fp32 — the parity
    /// reference.
    pub fn new(cfg: &Flux1Config, ts: &Tensors, gpu: Gpu, n_max: u32) -> Flux1Model {
        Flux1Model::new_with(cfg, ts, gpu, n_max, Precision::F32)
    }

    /// [`Flux1Model::new`] at a numeric tier.
    ///
    /// Under [`Precision::Int8`] every linear is packed int8 + per-channel
    /// scales EXCEPT the three boundary linears (`img_in`, `txt_in`,
    /// `final_layer.linear`, ~51 MiB), whose inputs are raw conditioning or the
    /// model's own output — the exemptions FLUX.2 measured as the ones that
    /// matter. `BRAIN_FLUX1_I8_KEEP_F32` (comma-separated substrings) keeps
    /// further linears at fp32 for bisection; `=_mlp.2` restores FLUX.2's
    /// full policy at +5.7 GiB.
    pub fn new_with(
        cfg: &Flux1Config,
        ts: &Tensors,
        gpu: Gpu,
        n_max: u32,
        precision: Precision,
    ) -> Flux1Model {
        let d = cfg.hidden;
        let mlp = cfg.mlp_hidden();
        let hd = cfg.head_dim();
        let nh = cfg.n_heads as u32;
        let fast = gpu.caps().workgroup_reductions;
        if precision == Precision::Int8 {
            assert!(
                fast,
                "flux1 int8 needs a GPU backend (DP4A + workgroup barriers); use fp32 on the {} backend",
                gpu.kind()
            );
        }
        // Every sliced storage binding must respect the 256-byte
        // min_storage_buffer_offset_alignment = 64 floats. Every offset this
        // model takes is a multiple of one of these widths.
        for (what, w) in [("hidden", d), ("mlp_hidden", mlp), ("in_channels", cfg.in_channels)] {
            assert!(w.is_multiple_of(64), "flux1: {what} = {w} floats breaks the 64-float storage-binding alignment");
        }
        let get = |name: &str| -> &(Vec<usize>, Vec<f32>) {
            ts.get(name).unwrap_or_else(|| panic!("flux1: missing tensor {name}"))
        };
        // Periodic flush during the multi-GB upload: wgpu holds a staging copy
        // per `write` until a blocking poll reclaims them, and a non-ReBAR card
        // OOMs long before the weights do (flux2's `new_with` documents the
        // 22 GiB-for-15.5 GiB observation on a P40).
        let uploaded = std::cell::Cell::new(0u64);
        let flush = |b: &DeviceBuffer, words: usize| {
            uploaded.set(uploaded.get() + 4 * words as u64);
            if uploaded.get() > (1 << 30) {
                let _ = gpu.read(b, 1);
                uploaded.set(0);
            }
        };
        let upv = |w: &[f32]| -> DeviceBuffer {
            let b = gpu.storage(w.len() as u64);
            gpu.write(&b, bytemuck::cast_slice(w));
            flush(&b, w.len());
            b
        };
        let up = |name: &str| -> DeviceBuffer { upv(&get(name).1) };

        let keep_env = std::env::var("BRAIN_FLUX1_I8_KEEP_F32").unwrap_or_default();
        let keeps: Vec<&str> = keep_env.split(',').filter(|s| !s.is_empty()).collect();
        // fp32 exemptions: only the three BOUNDARY linears (~51 MiB), whose
        // inputs are raw conditioning or the model's output. FLUX.2 also keeps
        // its double-block `mlp.2` fp32, but FLUX.1 has 19 double blocks with a
        // 4x MLP: those 38 tensors are 5.7 GiB at fp32 and OOM'd a 24 GiB P40
        // (measured). They are quantized here; `BRAIN_FLUX1_I8_KEEP_F32=_mlp.2`
        // restores the FLUX.2 policy on a card that has the room.
        let always_f32 = ["img_in", "txt_in", "final_layer.linear"];
        let linw = |name: &str, w: &[f32], n_out: usize, k: usize| -> LinW {
            let exempt = always_f32.iter().any(|s| name.contains(s))
                || keeps.iter().any(|s| name.contains(s));
            match precision {
                Precision::F32 => LinW::F32(upv(w)),
                Precision::Int8 if exempt => LinW::F32(upv(w)),
                Precision::Int8 => {
                    let (packed, sw) = model::int8::quantize_weight(w, n_out, k);
                    let pb = gpu.storage(packed.len() as u64);
                    gpu.write(&pb, &packed);
                    flush(&pb, packed.len());
                    let sb = gpu.storage(sw.len() as u64);
                    gpu.write(&sb, bytemuck::cast_slice(&sw));
                    LinW::I8(pb, sb)
                }
            }
        };
        // One whole checkpoint linear.
        let lin = |name: &str, n_out: usize, k: usize| -> Lin {
            Lin { w: linw(name, &get(&format!("{name}.weight")).1, n_out, k), b: up(&format!("{name}.bias")) }
        };
        // One row-slice of a FUSED checkpoint linear (`qkv`, `linear1`).
        let lin_slice =
            |name: &str, r0: usize, r1: usize, k: usize| -> Lin {
                let (_, w) = get(&format!("{name}.weight"));
                let (_, b) = get(&format!("{name}.bias"));
                Lin {
                    w: linw(name, &w[r0 * k..r1 * k], r1 - r0, k),
                    b: upv(&b[r0..r1]),
                }
            };
        // A modulation linear, rows permuted into film_row's (scale, shift) order.
        let lin_mod = |name: &str, triples: usize| -> Lin {
            let (_, w) = get(&format!("{name}.weight"));
            let (_, b) = get(&format!("{name}.bias"));
            Lin {
                w: linw(name, &permute_triples(w, d, d, triples), 3 * triples * d, d),
                b: upv(&permute_triples(b, d, 1, triples)),
            }
        };

        let stream = |p: &str| -> StreamW {
            StreamW {
                wq: lin_slice(&format!("{p}_attn.qkv"), 0, d, d),
                wk: lin_slice(&format!("{p}_attn.qkv"), d, 2 * d, d),
                wv: lin_slice(&format!("{p}_attn.qkv"), 2 * d, 3 * d, d),
                nq: up(&format!("{p}_attn.norm.query_norm.scale")),
                nk: up(&format!("{p}_attn.norm.key_norm.scale")),
                wo: lin(&format!("{p}_attn.proj"), d, d),
                w1: lin(&format!("{p}_mlp.0"), mlp, d),
                w2: lin(&format!("{p}_mlp.2"), d, mlp),
            }
        };

        // Modulation-output layout: per double block 12D (img 6D then txt 6D),
        // then 3D per single block, then the final layer's 2D. Every offset is
        // a multiple of D, hence of 64 floats.
        let dbl: Vec<DoubleW> = (0..cfg.depth_double)
            .map(|b| DoubleW {
                img: stream(&format!("double_blocks.{b}.img")),
                txt: stream(&format!("double_blocks.{b}.txt")),
                mod_img: lin_mod(&format!("double_blocks.{b}.img_mod.lin"), 2),
                mod_txt: lin_mod(&format!("double_blocks.{b}.txt_mod.lin"), 2),
                off_img: (b * 12 * d) as u64,
                off_txt: (b * 12 * d + 6 * d) as u64,
            })
            .collect();
        let base_single = (cfg.depth_double * 12 * d) as u64;
        let sgl: Vec<SingleW> = (0..cfg.depth_single)
            .map(|b| {
                let p = format!("single_blocks.{b}");
                let (_, l2) = get(&format!("{p}.linear2.weight"));
                // linear2 is [D, D+mlp]; split its CONTRACTION (column) dim
                let mut wo_a = Vec::with_capacity(d * d);
                let mut wo_b = Vec::with_capacity(d * mlp);
                for r in 0..d {
                    wo_a.extend_from_slice(&l2[r * (d + mlp)..r * (d + mlp) + d]);
                    wo_b.extend_from_slice(&l2[r * (d + mlp) + d..(r + 1) * (d + mlp)]);
                }
                SingleW {
                    wq: lin_slice(&format!("{p}.linear1"), 0, d, d),
                    wk: lin_slice(&format!("{p}.linear1"), d, 2 * d, d),
                    wv: lin_slice(&format!("{p}.linear1"), 2 * d, 3 * d, d),
                    nq: up(&format!("{p}.norm.query_norm.scale")),
                    nk: up(&format!("{p}.norm.key_norm.scale")),
                    wm: lin_slice(&format!("{p}.linear1"), 3 * d, 3 * d + mlp, d),
                    wo_a: Lin {
                        w: linw(&format!("{p}.linear2"), &wo_a, d, d),
                        b: up(&format!("{p}.linear2.bias")),
                    },
                    wo_b: linw(&format!("{p}.linear2"), &wo_b, d, mlp),
                    modl: lin_mod(&format!("{p}.modulation.lin"), 1),
                    off: base_single + (b * 3 * d) as u64,
                }
            })
            .collect();
        let final_off = base_single + (cfg.depth_single * 3 * d) as u64;
        let final_mod = {
            let (_, w) = get("final_layer.adaLN_modulation.1.weight");
            let (_, b) = get("final_layer.adaLN_modulation.1.bias");
            Lin {
                w: linw("final_layer.adaLN_modulation.1", &swap_pair(w, d, d), 2 * d, d),
                b: upv(&swap_pair(b, d, 1)),
            }
        };

        let n = n_max as u64;
        let du = d as u64;
        let mlpu = mlp as u64;
        let attn_mat = if fast { 1 } else { nh as u64 * n * n };
        let a = |len: u64| gpu.storage(len);
        let scr = Scratch {
            x0: a(n * du),
            x1: a(n * du),
            n1: a(n * du),
            n2: a(n * du),
            q: a(n * du),
            k: a(n * du),
            v: a(n * du),
            qn: a(n * du),
            kn: a(n * du),
            qr: a(n * du),
            kr: a(n * du),
            qkv: a(n * 3 * du),
            ctx: a(n * du),
            proj: a(n * du),
            mlpo: a(n * du),
            h1: a(n * mlpu),
            hs: a(n * mlpu),
            scores: a(attn_mat),
            probs: a(attn_mat),
            out: a(n * cfg.in_channels as u64),
            cos: a(n * (hd as u64 / 2)),
            sin: a(n * (hd as u64 / 2)),
            tok_in: a(n * cfg.in_channels as u64),
            ctx_in: a(n * cfg.context_in_dim as u64),
            vecb: a(du.max(64)),
            modout: a(final_off + 2 * du),
            ln_gamma: upv(&vec![1.0f32; d]),
            ln_beta: upv(&vec![0.0f32; d]),
        };
        let i8scr = (precision == Precision::Int8).then(|| I8Scratch::new(&gpu, n.max(64), n, &[cfg.hidden as u32, cfg.mlp_hidden() as u32]));

        let host = |name: &str| -> (Vec<f32>, Vec<f32>) {
            (get(&format!("{name}.weight")).1.clone(), get(&format!("{name}.bias")).1.clone())
        };
        Flux1Model {
            cfg: cfg.clone(),
            n_max,
            fast,
            precision,
            i8scr,
            dbl,
            sgl,
            img_in: lin("img_in", d, cfg.in_channels),
            txt_in: lin("txt_in", d, cfg.context_in_dim),
            final_w: lin("final_layer.linear", cfg.in_channels, d),
            final_mod,
            final_off,
            scr,
            time_a: host("time_in.in_layer"),
            time_b: host("time_in.out_layer"),
            vec_a: host("vector_in.in_layer"),
            vec_b: host("vector_in.out_layer"),
            guid_a: cfg.guidance_embed.then(|| host("guidance_in.in_layer")),
            guid_b: cfg.guidance_embed.then(|| host("guidance_in.out_layer")),
            gpu,
        }
    }

    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// Max joint tokens (txt + image + reference) this model is sized for.
    pub fn max_tokens(&self) -> u32 {
        self.n_max
    }

    /// One `MLPEmbedder`: `out_layer(silu(in_layer(x)))`, both biased.
    fn mlp_embed(a: &(Vec<f32>, Vec<f32>), b: &(Vec<f32>, Vec<f32>), x: &[f32], d: usize) -> Vec<f32> {
        use model::hostmath::{matvec_par, silu_slice};
        let mut h = matvec_par(&a.0, x, d, x.len());
        for (v, bb) in h.iter_mut().zip(&a.1) {
            *v += bb;
        }
        let h = silu_slice(&h);
        let mut o = matvec_par(&b.0, &h, d, d);
        for (v, bb) in o.iter_mut().zip(&b.1) {
            *v += bb;
        }
        o
    }

    /// The conditioning vector `vec = time_in(t) [+ guidance_in(g)] +
    /// vector_in(pooled)` — the reference's `temb`. Host math: ~30 M MACs.
    fn conditioning(&self, t: f32, guidance: f32, pooled: &[f32]) -> Vec<f32> {
        let d = self.cfg.hidden;
        // `time_factor = 1000` is the reference's, applied here (not in
        // hostmath) because it is a FLUX pipeline convention, not embedding math.
        let temb = |x: f32| model::hostmath::timestep_embedding(x * 1000.0, 256, true, 0.0, 10000.0);
        let mut vec_ = Self::mlp_embed(&self.time_a, &self.time_b, &temb(t), d);
        if let (Some(ga), Some(gb)) = (&self.guid_a, &self.guid_b) {
            let g = Self::mlp_embed(ga, gb, &temb(guidance), d);
            for (v, x) in vec_.iter_mut().zip(&g) {
                *v += x;
            }
        }
        let p = Self::mlp_embed(&self.vec_a, &self.vec_b, pooled, d);
        for (v, x) in vec_.iter_mut().zip(&p) {
            *v += x;
        }
        vec_
    }

    fn mm_kernel(&self) -> usize {
        K_MATMUL_REG3
    }

    /// The fp32 GEMM tier this model dispatches, for `model::block::gemm_variant`.
    /// At `m <= 32` that routes to the GEMV kernel: a register-tiled GEMM at M=1
    /// wastes 127/128 of every tile, which the 77 modulation mat-vecs would pay
    /// 77 times per forward.
    fn gemm_tier(&self) -> model::block::GemmVariants {
        if self.fast {
            model::block::GemmVariants::Fast { gemv: Some(K_MATMUL_GEMV), tiled: self.mm_kernel() }
        } else {
            model::block::GemmVariants::Reference(K_MATMUL)
        }
    }

    /// Sliced fp32 matmul: rows `xr0..xr0+m` of `x` `[.., k]` → the `m*n` floats
    /// of `o` at float offset `ooff` — the shared `model::dispatch::mm_rows_off`.
    #[allow(clippy::too_many_arguments)]
    fn mm_rows_at(&self, x: &DeviceBuffer, w: &DeviceBuffer, o: &DeviceBuffer, xr0: u32, ooff: u64, m: u32, k: u32, n: u32) -> Step {
        model::dispatch::mm_rows_off(&self.gpu, self.gemm_tier(), x, w, o, xr0, ooff, m, k, n)
    }

    /// Int8 only (a no-op under fp32): quantize rows `r0..r1` of `x` `[.., k]`
    /// into the K-matched packed scratch with fresh per-token scales. ONE quant
    /// feeds every linear reading that activation.
    fn quant_rows(&self, s: &mut Vec<Step>, x: &DeviceBuffer, r0: u32, r1: u32, k: u32) {
        let Some(i8s) = self.i8scr.as_ref() else { return };
        i8s.quant_rows(&self.gpu, [K_MAXABS, K_QUANT], s, x, r0, r1, k);
    }

    /// Int8 DP4A matmul over pre-quantized rows (tiled, or the GEMV at `m ≤ 32`)
    /// — the shared `model::dispatch::mm8_rows_off`. Same selection rule as the
    /// fp32 tier: the DP4A family has identical dispatch geometry and is
    /// GPU-only, hence always the `Fast` arm.
    #[allow(clippy::too_many_arguments)]
    fn mm8(&self, wq: &DeviceBuffer, sw: &DeviceBuffer, o: &DeviceBuffer, xr0: u32, ooff: u64, m: u32, k: u32, n: u32) -> Step {
        let i8s = self.i8scr.as_ref().expect("int8 scratch");
        let tier = model::block::GemmVariants::Fast { gemv: Some(K_MATMUL_I8_GEMV), tiled: K_MATMUL_I8 };
        model::dispatch::mm8_rows_off(&self.gpu, tier, i8s, wq, sw, o, xr0, ooff, m, k, n)
    }

    /// One linear over rows `r0..r1` at the model's tier, bias included.
    #[allow(clippy::too_many_arguments)]
    fn lin_rows(&self, s: &mut Vec<Step>, x: &DeviceBuffer, w: &Lin, o: &DeviceBuffer, r0: u32, r1: u32, k: u32, n: u32) {
        self.lin_rows_at(s, x, w, o, r0, r0 as u64 * n as u64, r1 - r0, k, n);
    }

    /// [`Self::lin_rows`] with an independent input row base and an explicit
    /// FLOAT output offset — the modulation sites are addressed by offset, not
    /// by row, because their widths (6D / 3D / 2D) differ per site.
    #[allow(clippy::too_many_arguments)]
    fn lin_rows_at(&self, s: &mut Vec<Step>, x: &DeviceBuffer, w: &Lin, o: &DeviceBuffer, xr0: u32, ooff: u64, m: u32, k: u32, n: u32) {
        self.matw_rows_at(s, x, &w.w, o, xr0, ooff, m, k, n);
        self.bias_rows(s, o, &w.b, ooff, m, n);
    }

    /// The matmul half of a linear (no bias) — the single block's `wo_b`, whose
    /// bias is added once on the `wo_a` half.
    #[allow(clippy::too_many_arguments)]
    fn matw_rows_at(&self, s: &mut Vec<Step>, x: &DeviceBuffer, w: &LinW, o: &DeviceBuffer, xr0: u32, ooff: u64, m: u32, k: u32, n: u32) {
        s.push(match w {
            LinW::F32(wb) => self.mm_rows_at(x, wb, o, xr0, ooff, m, k, n),
            LinW::I8(wq, sw) => self.mm8(wq, sw, o, xr0, ooff, m, k, n),
        });
    }

    /// `o[ooff .. ooff + m*n] += bias` (the bias buffer is exactly `[n]`).
    fn bias_rows(&self, s: &mut Vec<Step>, o: &DeviceBuffer, bias: &DeviceBuffer, ooff: u64, m: u32, n: u32) {
        let oo = (ooff, m as u64 * n as u64);
        s.push(self.gpu.step_sliced(K_BIAS, &[o, bias], &[oo, (0, 0)], &[m, n], m * n));
    }

    /// Affine-free LayerNorm over rows `r0..r1`, coalesced where the device
    /// supports it (`Op::LayerNorm` selection — the per-element kernel gives
    /// one thread a whole 3072-float row and reads at 1/8 of bandwidth).
    fn ln_rows(&self, x: &DeviceBuffer, o: &DeviceBuffer, r0: u32, r1: u32) -> Step {
        let d = self.cfg.hidden as u32;
        let m = r1 - r0;
        let off = (r0 as u64 * d as u64, m as u64 * d as u64);
        let (kind, threads) = model::block::ln_variant(&self.gpu, K_LN, Some(K_LN_ROWS), m, d);
        self.gpu.step_sliced(
            kind,
            &[x, &self.scr.ln_gamma, &self.scr.ln_beta, o],
            &[off, (0, 0), (0, 0), off],
            &[d, m, f(self.cfg.norm_eps)],
            threads,
        )
    }

    /// `y = x·(1 + scale) + shift` over rows `r0..r1`, reading the `(scale,
    /// shift)` pair at float offset `moff` of `modout`.
    fn film_rows(&self, x: &DeviceBuffer, moff: u64, y: &DeviceBuffer, r0: u32, r1: u32) -> Step {
        let d = self.cfg.hidden as u32;
        let m = r1 - r0;
        let off = (r0 as u64 * d as u64, m as u64 * d as u64);
        self.gpu.step_sliced(
            K_FILM,
            &[x, &self.scr.modout, y],
            &[off, (moff, 2 * d as u64), off],
            &[m, d, m],
            m * d,
        )
    }

    /// `y = x + gate ⊙ h` over rows `r0..r1`, gate at float offset `goff`.
    fn gate_rows(&self, x: &DeviceBuffer, goff: u64, h: &DeviceBuffer, y: &DeviceBuffer, r0: u32, r1: u32) -> Step {
        let d = self.cfg.hidden as u32;
        let m = r1 - r0;
        let off = (r0 as u64 * d as u64, m as u64 * d as u64);
        self.gpu.step_sliced(
            K_GATE,
            &[x, &self.scr.modout, h, y],
            &[off, (goff, d as u64), off, off],
            &[m, d, m],
            m * d,
        )
    }

    /// QK-RMSNorm over rows `r0..r1` (per-head rows of `head_dim`), coalesced
    /// where the device supports it (measured 19× at exactly this shape).
    fn qknorm_rows(&self, x: &DeviceBuffer, scale: &DeviceBuffer, o: &DeviceBuffer, r0: u32, r1: u32) -> Step {
        let d = self.cfg.hidden as u32;
        let hd = self.cfg.head_dim() as u32;
        let nh = self.cfg.n_heads as u32;
        let m = r1 - r0;
        let off = (r0 as u64 * d as u64, m as u64 * d as u64);
        let rows = m * nh;
        let (kind, threads) =
            model::block::rms_variant(&self.gpu, K_RMSNORM, Some(K_RMSNORM_ROWS), rows, hd);
        self.gpu.step_sliced(kind, &[x, scale, o], &[off, (0, 0), off], &[hd, rows, f(self.cfg.norm_eps)], threads)
    }

    /// RoPE + pack + joint bidirectional attention over the whole `n`-row slab.
    fn push_attn_core(&self, s: &mut Vec<Step>, n: u32) {
        let d = self.cfg.hidden as u32;
        let hd = self.cfg.head_dim() as u32;
        let nh = self.cfg.n_heads as u32;
        let half = hd / 2;
        let scr = &self.scr;
        s.push(self.gpu.step(K_ROPE, &[&scr.qn, &scr.cos, &scr.sin, &scr.qr], &[n, nh, hd, half], n * nh * half));
        s.push(self.gpu.step(K_ROPE, &[&scr.kn, &scr.cos, &scr.sin, &scr.kr], &[n, nh, hd, half], n * nh * half));
        s.push(self.gpu.step(K_PACK, &[&scr.qr, &scr.kr, &scr.v, &scr.qkv], &[n, d], n * 3 * d));
        if self.fast {
            s.push(model::block::flash_bidir_step(
                &self.gpu,
                model::block::FlashIds {
                    bidir: K_FLASH,
                    split: Some(K_FLASH_SPLIT),
                    reg: Some(K_FLASH_REG),
                    reg2: Some(K_FLASH_REG2),
                },
                1,
                nh,
                n,
                hd,
                d,
                &scr.qkv,
                &scr.ctx,
            ));
        } else {
            s.push(self.gpu.step(K_SCORES, &[&scr.qkv, &scr.scores], &[1, nh, n, hd, 3 * d, 0, d], nh * n * n));
            s.push(self.gpu.step(K_SOFTMAX, &[&scr.scores, &scr.probs], &[1, nh, n], nh * n));
            s.push(self.gpu.step(K_APPLY, &[&scr.probs, &scr.qkv, &scr.ctx], &[1, nh, n, hd, 3 * d, 2 * d, d], nh * n * hd));
        }
    }

    /// Forward one denoising evaluation.
    ///
    /// * `img_tokens` — packed latent tokens `[n_img, in_channels]`; for the
    ///   Kontext edit path the noise tokens come first, then the VAE-encoded
    ///   reference tokens.
    /// * `ctx` — the T5-XXL sequence `[n_txt, context_in_dim]`.
    /// * `pooled` — the CLIP-L pooled vector `[vec_in_dim]`.
    /// * `t` — sigma in `[0, 1]` (the pipeline's `timestep / 1000`).
    /// * `guidance` — the raw guidance scale (3.5 for dev/Kontext); ignored
    ///   when the variant has no `guidance_in`.
    /// * `ids` — joint 3-axis position ids, **text rows first**
    ///   (`(n_txt + n_img) * 3`).
    ///
    /// Returns the prediction for the first `n_pred` image tokens
    /// `[n_pred, in_channels]` — the noise span for an edit run.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, img_tokens: &[f32], ctx: &[f32], pooled: &[f32], t: f32, guidance: f32, ids: &[u32], n_pred: usize) -> Vec<f32> {
        self.run(img_tokens, ctx, pooled, t, guidance, ids, n_pred, None, None)
    }

    /// [`Self::forward`] with an adapter contributing dispatches after each
    /// block — see [`crate::inject`]. The adapter's steps join the model's own
    /// list, so a conditioned forward is still one submit.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_injected(
        &self,
        img_tokens: &[f32],
        ctx: &[f32],
        pooled: &[f32],
        t: f32,
        guidance: f32,
        ids: &[u32],
        n_pred: usize,
        inject: &dyn crate::inject::BlockInject,
    ) -> Vec<f32> {
        self.run(img_tokens, ctx, pooled, t, guidance, ids, n_pred, None, Some(inject))
    }

    /// [`Self::forward`] plus a per-stage tap of the residual slab — the parity
    /// ladder's rung 2/3 replay. Tracing submits once per block and reads the
    /// slab back, so it is far slower than [`Self::forward`]; it exists to
    /// localize a divergence to the block that introduced it.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_traced(&self, img_tokens: &[f32], ctx: &[f32], pooled: &[f32], t: f32, guidance: f32, ids: &[u32], n_pred: usize) -> (Vec<f32>, Trace) {
        let mut tr = Trace { stages: Vec::new(), pre_final: Vec::new(), temb: Vec::new() };
        let out = self.run(img_tokens, ctx, pooled, t, guidance, ids, n_pred, Some(&mut tr), None);
        (out, tr)
    }

    /// [`Self::forward_traced`] with an adapter — the conditioned parity replay.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_traced_injected(
        &self,
        img_tokens: &[f32],
        ctx: &[f32],
        pooled: &[f32],
        t: f32,
        guidance: f32,
        ids: &[u32],
        n_pred: usize,
        inject: &dyn crate::inject::BlockInject,
    ) -> (Vec<f32>, Trace) {
        let mut tr = Trace { stages: Vec::new(), pre_final: Vec::new(), temb: Vec::new() };
        let out =
            self.run(img_tokens, ctx, pooled, t, guidance, ids, n_pred, Some(&mut tr), Some(inject));
        (out, tr)
    }

    #[allow(clippy::too_many_arguments)]
    fn run(&self, img_tokens: &[f32], ctx: &[f32], pooled: &[f32], t: f32, guidance: f32, ids: &[u32], n_pred: usize, mut trace: Option<&mut Trace>, inject: Option<&dyn crate::inject::BlockInject>) -> Vec<f32> {
        let cfg = &self.cfg;
        let d = cfg.hidden as u32;
        let mlp = cfg.mlp_hidden() as u32;
        let cin = cfg.in_channels as u32;
        let ctxd = cfg.context_in_dim as u32;
        assert_eq!(pooled.len(), cfg.vec_in_dim, "pooled vector width");
        assert_eq!(ctx.len() % cfg.context_in_dim, 0, "ctx not a multiple of context_in_dim");
        assert_eq!(img_tokens.len() % cfg.in_channels, 0, "latents not a multiple of in_channels");
        let nt = (ctx.len() / cfg.context_in_dim) as u32;
        let ni = (img_tokens.len() / cfg.in_channels) as u32;
        let n = nt + ni;
        assert!(n <= self.n_max, "sized for {} joint tokens, got {n}", self.n_max);
        assert_eq!(ids.len() as u32, n * 3, "ids must be 3-axis, text rows first");
        assert!(n_pred as u32 <= ni);
        // Every per-stream binding starts at row `nt` and must clear the
        // 256-byte `min_storage_buffer_offset_alignment` = 64 floats. For the
        // `[.., D]` slabs the offset is nt·D, which `D % 64 == 0` (checked in
        // `new_with`) already satisfies for ANY nt — so that is NOT the binding
        // constraint. The binding that actually constrains `nt` is the int8
        // per-token scale `sx`, offset by `nt` ROWS, hence `nt % 64 == 0`.
        // `model::int8::quant_rows_steps` re-asserts it at the dispatch; this
        // one fails at the top of the forward instead of 40 layers in.
        // nt is the T5 length (256 / 512 in every released pipeline).
        assert!(
            self.precision != Precision::Int8 || nt.is_multiple_of(64),
            "int8: txt rows {nt} must be a multiple of 64 (the per-token scale buffer is bound at row {nt})"
        );

        // ---- host conditioning + device modulation ---------------------------
        let temb = self.conditioning(t, guidance, pooled);
        let sv = model::hostmath::silu_slice(&temb);
        self.gpu.write(&self.scr.vecb, bytemuck::cast_slice(&sv));
        if let Some(tr) = trace.as_deref_mut() {
            tr.temb = temb.clone();
        }

        // RoPE tables from the joint ids (t, h, w), interleaved pairs.
        let rc = dit::rope::RopeConfig {
            axes_dims: cfg.axes_dim.iter().map(|&a| a as u32).collect(),
            axes_lens: vec![4096, 4096, 4096],
            theta: cfg.rope_theta,
        };
        let tables = dit::rope::tables_for_ids(&rc, ids, 3);
        self.gpu.write(&self.scr.cos, bytemuck::cast_slice(&tables.cos));
        self.gpu.write(&self.scr.sin, bytemuck::cast_slice(&tables.sin));
        self.gpu.write(&self.scr.tok_in, bytemuck::cast_slice(img_tokens));
        self.gpu.write(&self.scr.ctx_in, bytemuck::cast_slice(ctx));

        let scr = &self.scr;
        let mut s: Vec<Step> = Vec::new();
        // Every modulation linear reads the SAME `silu(vec)` row, so one
        // quantization feeds all 77 of them.
        self.quant_rows(&mut s, &scr.vecb, 0, 1, d);
        for b in &self.dbl {
            self.lin_rows_at(&mut s, &scr.vecb, &b.mod_img, &scr.modout, 0, b.off_img, 1, d, 6 * d);
            self.lin_rows_at(&mut s, &scr.vecb, &b.mod_txt, &scr.modout, 0, b.off_txt, 1, d, 6 * d);
        }
        for b in &self.sgl {
            self.lin_rows_at(&mut s, &scr.vecb, &b.modl, &scr.modout, 0, b.off, 1, d, 3 * d);
        }
        self.lin_rows_at(&mut s, &scr.vecb, &self.final_mod, &scr.modout, 0, self.final_off, 1, d, 2 * d);

        // ---- embed both streams into the joint slab, text rows first ---------
        self.lin_rows_at(&mut s, &scr.ctx_in, &self.txt_in, &scr.x0, 0, 0, nt, ctxd, d);
        self.lin_rows_at(&mut s, &scr.tok_in, &self.img_in, &scr.x0, 0, nt as u64 * d as u64, ni, cin, d);

        let (mut xa, mut xb) = (&scr.x0, &scr.x1);
        let flush = |s: &mut Vec<Step>| {
            self.gpu.submit(&[], s);
            s.clear();
        };

        for (bi, b) in self.dbl.iter().enumerate() {
            for (w, moff, r0, r1) in [
                (&b.txt, b.off_txt, 0, nt),
                (&b.img, b.off_img, nt, n),
            ] {
                s.push(self.ln_rows(xa, &scr.n1, r0, r1));
                s.push(self.film_rows(&scr.n1, moff, &scr.n2, r0, r1));
                if w.wq.is_i8() {
                    self.quant_rows(&mut s, &scr.n2, r0, r1, d);
                }
                self.lin_rows(&mut s, &scr.n2, &w.wq, &scr.q, r0, r1, d, d);
                self.lin_rows(&mut s, &scr.n2, &w.wk, &scr.k, r0, r1, d, d);
                self.lin_rows(&mut s, &scr.n2, &w.wv, &scr.v, r0, r1, d, d);
                s.push(self.qknorm_rows(&scr.q, &w.nq, &scr.qn, r0, r1));
                s.push(self.qknorm_rows(&scr.k, &w.nk, &scr.kn, r0, r1));
            }
            self.push_attn_core(&mut s, n);
            if b.txt.wo.is_i8() || b.img.wo.is_i8() {
                self.quant_rows(&mut s, &scr.ctx, 0, n, d);
            }
            for (w, moff, r0, r1) in [(&b.txt, b.off_txt, 0, nt), (&b.img, b.off_img, nt, n)] {
                self.lin_rows(&mut s, &scr.ctx, &w.wo, &scr.proj, r0, r1, d, d);
                s.push(self.gate_rows(xa, moff + 2 * d as u64, &scr.proj, xb, r0, r1));
            }
            std::mem::swap(&mut xa, &mut xb);
            // MLP halves (second modulation triple, at +3D within the site)
            for (w, moff, r0, r1) in [(&b.txt, b.off_txt, 0, nt), (&b.img, b.off_img, nt, n)] {
                let m2 = moff + 3 * d as u64;
                s.push(self.ln_rows(xa, &scr.n1, r0, r1));
                s.push(self.film_rows(&scr.n1, m2, &scr.n2, r0, r1));
                if w.w1.is_i8() {
                    self.quant_rows(&mut s, &scr.n2, r0, r1, d);
                }
                self.lin_rows(&mut s, &scr.n2, &w.w1, &scr.h1, r0, r1, d, mlp);
            }
            s.push(self.gpu.step(K_GELU, &[&scr.h1, &scr.hs], &[n * mlp], n * mlp));
            if b.txt.w2.is_i8() || b.img.w2.is_i8() {
                self.quant_rows(&mut s, &scr.hs, 0, n, mlp);
            }
            for (w, moff, r0, r1) in [(&b.txt, b.off_txt, 0, nt), (&b.img, b.off_img, nt, n)] {
                self.lin_rows(&mut s, &scr.hs, &w.w2, &scr.mlpo, r0, r1, mlp, d);
                s.push(self.gate_rows(xa, moff + 5 * d as u64, &scr.mlpo, xb, r0, r1));
            }
            std::mem::swap(&mut xa, &mut xb);
            if let Some(inj) = inject {
                inj.after_double(bi, crate::inject::InjectSite { x: xa, n_txt: nt, n, d, n_pred: n_pred as u32 }, &mut s);
            }
            if let Some(tr) = trace.as_deref_mut() {
                flush(&mut s);
                tr.stages.push((format!("db{bi}"), self.gpu.read(xa, (n * d) as usize)));
            }
        }

        for (bi, w) in self.sgl.iter().enumerate() {
            // parallel attn ‖ MLP over ONE shared modulated norm — every op is
            // stream-agnostic, so the whole slab goes through in one dispatch.
            s.push(self.ln_rows(xa, &scr.n1, 0, n));
            s.push(self.film_rows(&scr.n1, w.off, &scr.n2, 0, n));
            if w.wq.is_i8() || w.wm.is_i8() {
                self.quant_rows(&mut s, &scr.n2, 0, n, d);
            }
            self.lin_rows(&mut s, &scr.n2, &w.wq, &scr.q, 0, n, d, d);
            self.lin_rows(&mut s, &scr.n2, &w.wk, &scr.k, 0, n, d, d);
            self.lin_rows(&mut s, &scr.n2, &w.wv, &scr.v, 0, n, d, d);
            self.lin_rows(&mut s, &scr.n2, &w.wm, &scr.h1, 0, n, d, mlp);
            s.push(self.qknorm_rows(&scr.q, &w.nq, &scr.qn, 0, n));
            s.push(self.qknorm_rows(&scr.k, &w.nk, &scr.kn, 0, n));
            self.push_attn_core(&mut s, n);
            s.push(self.gpu.step(K_GELU, &[&scr.h1, &scr.hs], &[n * mlp], n * mlp));
            // linear2 over cat(attn, gelu(mlp)): two column-split matmuls whose
            // gated adds sum. The [D] bias rides on the attn half only.
            if w.wo_a.is_i8() {
                self.quant_rows(&mut s, &scr.ctx, 0, n, d);
            }
            self.lin_rows(&mut s, &scr.ctx, &w.wo_a, &scr.proj, 0, n, d, d);
            if matches!(w.wo_b, LinW::I8(..)) {
                self.quant_rows(&mut s, &scr.hs, 0, n, mlp);
            }
            self.matw_rows_at(&mut s, &scr.hs, &w.wo_b, &scr.mlpo, 0, 0, n, mlp, d);
            let g = w.off + 2 * d as u64;
            s.push(self.gate_rows(xa, g, &scr.proj, xb, 0, n));
            std::mem::swap(&mut xa, &mut xb);
            s.push(self.gate_rows(xa, g, &scr.mlpo, xb, 0, n));
            std::mem::swap(&mut xa, &mut xb);
            if let Some(inj) = inject {
                inj.after_single(bi, crate::inject::InjectSite { x: xa, n_txt: nt, n, d, n_pred: n_pred as u32 }, &mut s);
            }
            if let Some(tr) = trace.as_deref_mut() {
                flush(&mut s);
                tr.stages.push((format!("sg{bi}"), self.gpu.read(xa, (n * d) as usize)));
            }
        }

        // ---- final layer on the predicted image rows only --------------------
        // last use of `trace`, so it moves rather than reborrows
        if let Some(tr) = trace {
            flush(&mut s);
            let slab = self.gpu.read(xa, (n * d) as usize);
            tr.pre_final = slab[(nt * d) as usize..].to_vec();
        }
        let p0 = nt;
        let p1 = nt + n_pred as u32;
        s.push(self.ln_rows(xa, &scr.n1, p0, p1));
        s.push(self.film_rows(&scr.n1, self.final_off, &scr.n2, p0, p1));
        if self.final_w.is_i8() {
            self.quant_rows(&mut s, &scr.n2, p0, p1, d);
        }
        self.lin_rows_at(&mut s, &scr.n2, &self.final_w, &scr.out, p0, 0, n_pred as u32, d, cin);


        // debug aid: SMOKE_STEPS=k submits only the first k steps
        let take = std::env::var("SMOKE_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(s.len());
        self.gpu.submit(&[], &s[..take.min(s.len())]);
        self.gpu.read(&self.scr.out, n_pred * cfg.in_channels)
    }
}

/// Joint 3-axis position ids in the reference layout, **text rows first**.
///
/// Text tokens: `(0,0,0)`. Generated image: `(0,h,w)` raster-major. Each
/// Kontext reference image: `(1,h,w)` — `FluxKontextPipeline.prepare_latents`
/// sets `image_ids[..., 0] = 1` for **every** reference, unlike FLUX.2's
/// per-reference `10·(i+1)` offset. `refs` are (height, width) in latent-token
/// units.
pub fn position_ids(txt_len: usize, lh: usize, lw: usize, refs: &[(usize, usize)]) -> Vec<u32> {
    let extra: usize = refs.iter().map(|&(h, w)| h * w).sum();
    let mut ids = Vec::with_capacity((txt_len + lh * lw + extra) * 3);
    for _ in 0..txt_len {
        ids.extend([0, 0, 0]);
    }
    for h in 0..lh {
        for w in 0..lw {
            ids.extend([0, h as u32, w as u32]);
        }
    }
    for &(rh, rw) in refs {
        for h in 0..rh {
            for w in 0..rw {
                ids.extend([1, h as u32, w as u32]);
            }
        }
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triple_permutation_puts_scale_first() {
        // one triple of 1-wide "vectors": shift=1, scale=2, gate=3
        let d = 1;
        let w = vec![1.0, 2.0, 3.0];
        assert_eq!(permute_triples(&w, d, 1, 1), vec![2.0, 1.0, 3.0]);
        // two triples (a double-stream 6D modulation)
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(permute_triples(&w, d, 1, 2), vec![2.0, 1.0, 3.0, 5.0, 4.0, 6.0]);
        // the final layer is a (shift, scale) PAIR
        assert_eq!(swap_pair(&[1.0, 2.0], d, 1), vec![2.0, 1.0]);
    }

    #[test]
    fn kontext_ids_offset_references_on_axis_zero() {
        let ids = position_ids(2, 2, 2, &[(1, 2)]);
        assert_eq!(ids.len(), (2 + 4 + 2) * 3);
        assert_eq!(&ids[0..6], &[0, 0, 0, 0, 0, 0]); // text
        assert_eq!(&ids[6..18], &[0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1]); // image
        assert_eq!(&ids[18..24], &[1, 0, 0, 1, 0, 1]); // reference: axis0 = 1
    }
}
