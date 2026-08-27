// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The FLUX.2 transformer forward: double-stream blocks (separate img/txt
//! weights, joint attention) then single-stream parallel blocks, with the three
//! **global** modulation linears folded into LayerNorm affine params.
//!
//! Klein's standard path modulates every token identically (the modulation
//! vector depends only on timestep — and guidance on dev), so
//! `(1 + scale)·LN(x) + shift` is exactly `LayerNorm` with `gamma = 1+scale`,
//! `beta = shift`. The whole model therefore carries six modulated-LN
//! (gamma, beta) pairs and five gate vectors, recomputed on the host and
//! re-uploaded once per forward — no per-block modulation work at all. (The
//! Klein-9B-KV blended per-token modulation path would break this fold; it is
//! deliberately out of scope here.)
//!
//! Layout: one joint residual slab `[B·n, D]`, **sample-major** — sample `b`
//! owns rows `[b·n, (b+1)·n)`, text rows first (`0..nt`) then image (+
//! reference) rows, the reference's `[txt, img, refs]` order. Stream-specific
//! ops run on row ranges via `step_sliced`; joint attention reads the whole
//! slab with `bsz = B` (the bidirectional kernels index `qkv[(b·T + j)·stride]`,
//! so samples cannot attend across each other by construction). Fused
//! checkpoint weights (`qkv`, `mlp.0`, `linear1`, `linear2`) are split at build
//! time into per-projection device buffers so every matmul is a plain
//! full-buffer dispatch.
//!
//! **Batching** (`forward_batch`): B samples with independent timesteps, text
//! conditioning and latents share one forward. The single-stream blocks (20 of
//! klein-4B's 25) become one GEMM at `M = B·n`; the double-stream blocks stay
//! per-sample because a stream's rows are `n`-strided in a sample-major slab
//! (the layout joint attention requires). Modulation is per-sample: the six
//! (gamma, beta) pairs and five gates are uploaded as `[B, D]` and indexed by
//! the sample — `gate_row`'s `rows_per_cond` condition groups do this in ONE
//! dispatch for the single blocks, and the per-sample LayerNorm sites bind
//! their own `[D]` slice. **No new WGSL is needed for any of it.** Because
//! every kernel's per-output reduction order is independent of `M`/`bsz`, a
//! batch-of-N forward is *bit-identical* to N single forwards
//! (`tests/batch_parity.rs`), and B=1 records exactly the dispatch sequence it
//! did before batching existed.

use gpu_core::{DeviceBuffer, Gpu, Step};

use crate::config::Flux2Config;
use crate::import::Tensors;

pub const KERNELS: &[(&str, &str)] = &[
    ("layernorm", kernels::LAYERNORM),
    ("matmul", kernels::MATMUL),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("rmsnorm_eps", kernels::RMSNORM_EPS),
    ("rmsnorm_rows", kernels::RMSNORM_ROWS),
    ("rope_interleave_table", kernels::ROPE_INTERLEAVE_TABLE),
    ("pack_qkv", kernels::PACK_QKV),
    ("attn_scores_bidir", kernels::ATTN_SCORES_BIDIR),
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
    ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR),
    ("flash_attn_bidir", kernels::FLASH_ATTN_BIDIR),
    ("flash_attn_bidir_split", kernels::FLASH_ATTN_BIDIR_SPLIT),
    ("silu_mul", kernels::SILU_MUL),
    ("gate_row", kernels::GATE_ROW),
    // int8 DP4A path (GPU only): per-token activation quant + DP4A GEMM.
    // Listing them is harmless off-GPU — only dispatched under Precision::Int8.
    ("max_abs_row", kernels::MAX_ABS_ROW),
    ("quant_pack", kernels::QUANT_PACK),
    ("matmul_i8_dyn", kernels::MATMUL_I8_DYN),
    ("flash_attn_bidir_reg", kernels::FLASH_ATTN_BIDIR_REG),
    ("flash_attn_bidir_reg2", kernels::FLASH_ATTN_BIDIR_REG2),
];
const K_LN: usize = 0;
const K_MATMUL: usize = 1;
const K_MATMUL_REG3: usize = 2;
const K_RMSNORM: usize = 3;
const K_RMSNORM_ROWS: usize = 4;
const K_ROPE: usize = 5;
const K_PACK: usize = 6;
const K_SCORES: usize = 7;
const K_SOFTMAX: usize = 8;
const K_APPLY: usize = 9;
const K_FLASH: usize = 10;
const K_FLASH_SPLIT: usize = 11;
const K_SILU_MUL: usize = 12;
const K_GATE: usize = 13;
const K_MAXABS: usize = 14;
const K_QUANT: usize = 15;
const K_MATMUL_I8: usize = 16;
/// The register-tiled flash-attention pair, appended so every index above is
/// unchanged. `model::block::flash_bidir_variant` picks between all four from
/// the device's queried caps.
const K_FLASH_REG: usize = 17;
const K_FLASH_REG2: usize = 18;

const EPS: f32 = 1e-6;

fn f(x: f32) -> u32 {
    x.to_bits()
}


// The numeric-tier machinery (Precision map, packed-int8 resident weight,
// K-keyed activation scratch, DP4A dispatch) is shared with flux1 via
// `model::dispatch` — this file keeps only the FLUX.2-specific graph.
pub use model::dispatch::Precision;
use model::dispatch::LinW as Lin;

/// One attention/MLP weight set (a double block holds two: img and txt).
struct StreamW {
    wq: Lin,
    wk: Lin,
    wv: Lin,
    nq: DeviceBuffer,
    nk: DeviceBuffer,
    wo: Lin,
    w1: Lin,
    w3: Lin,
    w2: Lin,
}

struct SingleW {
    wq: Lin,
    wk: Lin,
    wv: Lin,
    nq: DeviceBuffer,
    nk: DeviceBuffer,
    w1: Lin,
    w3: Lin,
    /// linear2 column-split: `out = wo_a @ attn_ctx + wo_b @ mlp_act`.
    wo_a: Lin,
    wo_b: Lin,
}

/// The six modulated-LN sites and five gates, in upload order. Each buffer is
/// `[b_max, D]` — sample `b`'s vector at row `b`, which is what makes different
/// timesteps per batch member free (`gate_row`'s `rows_per_cond` groups for the
/// single blocks; a `(b·D, D)` binding slice for the per-sample LN sites).
struct ModBufs {
    gamma: Vec<DeviceBuffer>, // [img1, img2, txt1, txt2, single, final]
    beta: Vec<DeviceBuffer>,
    gate: Vec<DeviceBuffer>, // [img1, img2, txt1, txt2, single]
}

/// One sample's folded modulation (host side), in [`ModBufs`] site order.
struct ModVals {
    gamma: [Vec<f32>; 6],
    beta: [Vec<f32>; 6],
    gate: [Vec<f32>; 5],
}

struct Scratch {
    x0: DeviceBuffer,
    x1: DeviceBuffer,
    n1: DeviceBuffer,
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
    h1: DeviceBuffer,
    h2: DeviceBuffer,
    hs: DeviceBuffer,
    mlp: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    out: DeviceBuffer,
    cos: DeviceBuffer,
    sin: DeviceBuffer,
    tok_in: DeviceBuffer,
    ctx_in: DeviceBuffer,
}

// Int8 activation scratch (allocated only under [`Precision::Int8`]):
// `model::dispatch::I8Scratch`, keyed by the two in-block contraction widths
// (hidden, mlp). One quant feeds every linear reading that activation
// (n1 → q/k/v and w1/w3, ctx → wo, hs → w2/wo_b); the boundary linears
// (img_in/txt_in/final_layer) stay fp32.
use model::dispatch::I8Scratch;

pub struct Flux2Model {
    pub cfg: Flux2Config,
    gpu: Gpu,
    /// max joint rows (txt + img + refs) PER SAMPLE the scratch is sized for
    n_max: u32,
    /// max samples per [`Flux2Model::forward_batch`] the scratch is sized for
    b_max: u32,
    fast: bool,
    precision: Precision,
    i8scr: Option<I8Scratch>,
    dbl: Vec<(StreamW, StreamW)>,
    sgl: Vec<SingleW>,
    img_in: Lin,
    txt_in: Lin,
    final_w: Lin,
    modb: ModBufs,
    scr: Scratch,
    // host-side conditioning weights
    time_in_a: Vec<f32>,  // [D,256]
    time_in_b: Vec<f32>,  // [D,D]
    mod_img: Vec<f32>,    // [6D,D]
    mod_txt: Vec<f32>,    // [6D,D]
    mod_single: Vec<f32>, // [3D,D]
    final_adaln: Vec<f32>, // [2D,D]
}

impl Flux2Model {
    /// Build device state from imported (BFL-named) tensors, sized for at most
    /// `n_max` joint tokens (txt_len + image + reference tokens). fp32 — the
    /// parity reference.
    pub fn new(cfg: &Flux2Config, ts: &Tensors, gpu: Gpu, n_max: u32) -> Flux2Model {
        Flux2Model::new_with(cfg, ts, gpu, n_max, Precision::F32)
    }

    /// [`Flux2Model::new`] with a numeric tier. [`Precision::Int8`] uploads
    /// every linear as packed int8 + per-channel scales (~4× smaller — the 4B
    /// DiT drops from ~15.5 GiB to ~3.9 GiB resident) and the forward runs the
    /// quant→DP4A sequence per linear. GPU only (DP4A + workgroup barriers).
    pub fn new_with(cfg: &Flux2Config, ts: &Tensors, gpu: Gpu, n_max: u32, precision: Precision) -> Flux2Model {
        Flux2Model::new_batched(cfg, ts, gpu, n_max, 1, precision)
    }

    /// [`Flux2Model::new_with`] sized for up to `b_max` samples per forward
    /// ([`Flux2Model::forward_batch`]). Only the activation scratch grows —
    /// weights are shared — so the extra cost is `(b_max − 1) × ` the per-sample
    /// working set (≈ 0.5 GiB at 512² for klein-4B). `b_max = 1` allocates
    /// exactly what the unbatched model always did.
    pub fn new_batched(cfg: &Flux2Config, ts: &Tensors, gpu: Gpu, n_max: u32, b_max: u32, precision: Precision) -> Flux2Model {
        Flux2Model::new_from(cfg, &crate::weights::DitWeights::Map(ts), gpu, n_max, b_max, precision)
    }

    /// [`Flux2Model::new_batched`] over an arbitrary weight source.
    ///
    /// `DitWeights::Map` is the fp32 path every other constructor takes.
    /// `DitWeights::Gguf` decodes a Q8_0 checkpoint one weight matrix at a
    /// time and, at `Precision::Int8`, requantizes each straight to packed
    /// int8 without ever building the fp32 model - bit-identical to the round
    /// trip, see `crate::weights`.
    pub fn new_from(cfg: &Flux2Config, src: &crate::weights::DitWeights, gpu: Gpu, n_max: u32, b_max: u32, precision: Precision) -> Flux2Model {
        assert!(b_max >= 1, "b_max must be >= 1");
        assert!(!cfg.guidance_embed, "guidance-embedded variants not supported");
        let d = cfg.hidden;
        let mlp = cfg.mlp_hidden();
        let hd = cfg.head_dim();
        let nh = cfg.n_heads as u32;
        let fast = gpu.caps().workgroup_reductions;
        if precision == Precision::Int8 {
            assert!(
                fast,
                "flux2 int8 needs a GPU backend (DP4A + workgroup barriers); use fp32 on the {} backend",
                gpu.kind()
            );
            // step_sliced binds sub-ranges at BYTE offset elem*4; storage
            // bindings must be 256-byte aligned. Every sliced row offset here is
            // 0 or txt_len rows, so widths and txt_len must be multiples of 64.
            assert!(cfg.txt_len.is_multiple_of(64) && d.is_multiple_of(64) && mlp.is_multiple_of(64), "int8 slicing alignment");
        }
        let getv = |name: &str| -> Vec<f32> { src.with_f32(name, <[f32]>::to_vec) };
        // Periodic poll_wait during the multi-GB weight upload: wgpu holds a
        // staging copy per `write` until a blocking poll reclaims them; on a
        // non-ReBAR card the un-reclaimed staging OOMs the device (observed
        // 22 GiB for 15.5 GiB of weights on a P40 — zimage's dev.rs documents
        // the same). Flush roughly every GiB.
        // Build-time spans, printed under `BRAIN_PROFILE`. Device timestamps
        // cannot see any of this: the load is host work plus queue writes,
        // with no kernel running, so a per-kernel table reports a near-zero
        // total for a phase that on a real 9B checkpoint outweighs several
        // denoise steps. `t_build` is the whole constructor; `quant_ns` and
        // `write_ns` split the per-linear cost into its two halves, because
        // they have completely different fixes.
        let t_build = std::time::Instant::now();
        let quant_ns = std::cell::Cell::new(0u128);
        let write_ns = std::cell::Cell::new(0u128);
        let flush_ns = std::cell::Cell::new(0u128);
        let flush_n = std::cell::Cell::new(0u32);
        let split_ns = std::cell::Cell::new(0u128);
        let bytes_up = std::cell::Cell::new(0u64);
        let uploaded = std::cell::Cell::new(0u64);
        let flush = |b: &DeviceBuffer, words: usize| {
            uploaded.set(uploaded.get() + 4 * words as u64);
            bytes_up.set(bytes_up.get() + 4 * words as u64);
            if uploaded.get() > (1 << 30) {
                // force a real flush: a readback drains the queue (an empty
                // submit records nothing) and the poll reclaims the staging
                // wgpu holds per write — without this a non-ReBAR card OOMs
                // at ~22 GiB for 15.5 GiB of weights
                let t = std::time::Instant::now();
                let _ = gpu.read(b, 1);
                flush_ns.set(flush_ns.get() + t.elapsed().as_nanos());
                flush_n.set(flush_n.get() + 1);
                uploaded.set(0);
            }
        };
        let upv = |w: &[f32]| -> DeviceBuffer {
            let t = std::time::Instant::now();
            let b = gpu.storage(w.len() as u64);
            gpu.write(&b, bytemuck::cast_slice(w));
            write_ns.set(write_ns.get() + t.elapsed().as_nanos());
            flush(&b, w.len());
            b
        };
        let up = |name: &str| -> DeviceBuffer { src.with_f32(name, |w| upv(w)) };
        // One linear `[n_out, k]`, uploaded at the requested tier.
        let lin_v = |w: &[f32], n_out: usize, k: usize| -> Lin {
            match precision {
                Precision::F32 => Lin::F32(upv(w)),
                Precision::Int8 => {
                    let tq = std::time::Instant::now();
                    let (packed, sw) = model::int8::quantize_weight(w, n_out, k);
                    quant_ns.set(quant_ns.get() + tq.elapsed().as_nanos());
                    let tw = std::time::Instant::now();
                    let pb = gpu.storage(packed.len() as u64);
                    gpu.write(&pb, &packed);
                    write_ns.set(write_ns.get() + tw.elapsed().as_nanos());
                    flush(&pb, packed.len());
                    let tw = std::time::Instant::now();
                    let sb = gpu.storage(sw.len() as u64);
                    gpu.write(&sb, bytemuck::cast_slice(&sw));
                    write_ns.set(write_ns.get() + tw.elapsed().as_nanos());
                    Lin::I8(pb, sb)
                }
            }
        };
        // Debug aid (parity bisection): BRAIN_FLUX2_I8_KEEP_F32=sub1,sub2 keeps
        // every linear whose name contains a listed substring at fp32.
        let keep_f32 = std::env::var("BRAIN_FLUX2_I8_KEEP_F32").unwrap_or_default();
        let keeps: Vec<String> = keep_f32.split(',').filter(|s| !s.is_empty()).map(str::to_string).collect();
        let lin_n = |name: &str, w: &[f32], n_out: usize, k: usize| -> Lin {
            if keeps.iter().any(|s| name.contains(s.as_str())) {
                Lin::F32(upv(w))
            } else {
                lin_v(w, n_out, k)
            }
        };
        // Upload an already-packed int8 linear (the direct-from-Q8_0 route);
        // the fp32 tier never reaches here.
        let up_i8 = |packed: Vec<u32>, sw: Vec<f32>| -> Lin {
            let tw = std::time::Instant::now();
            let pb = gpu.storage(packed.len() as u64);
            gpu.write(&pb, &packed);
            write_ns.set(write_ns.get() + tw.elapsed().as_nanos());
            flush(&pb, packed.len());
            let tw = std::time::Instant::now();
            let sb = gpu.storage(sw.len() as u64);
            gpu.write(&sb, bytemuck::cast_slice(&sw));
            write_ns.set(write_ns.get() + tw.elapsed().as_nanos());
            Lin::I8(pb, sb)
        };
        // ONE rectangle of a stored tensor -> one `Lin`. `store` is the
        // checkpoint tensor, `label` what BRAIN_FLUX2_I8_KEEP_F32 matches.
        // Tries the direct Q8_0 -> int8 route first; `try_i8_rect` declines
        // (returning None) for anything it cannot serve exactly, and the fp32
        // route below is then the same code the map path has always run.
        let lin_rect = |store: &str, label: &str, stride: usize, r0: usize, n_out: usize, c0: usize, k: usize| -> Lin {
            if precision == Precision::Int8 && !keeps.iter().any(|s| label.contains(s.as_str())) {
                let tq = std::time::Instant::now();
                let direct = src.try_i8_rect(store, stride, r0, n_out, c0, k);
                quant_ns.set(quant_ns.get() + tq.elapsed().as_nanos());
                if let Some((packed, sw)) = direct {
                    return up_i8(packed, sw);
                }
            }
            src.with_f32(store, |w| {
                if c0 == 0 && k == stride {
                    lin_n(label, &w[r0 * stride..(r0 + n_out) * stride], n_out, k)
                } else {
                    let tsp = std::time::Instant::now();
                    let mut blk = vec![0f32; n_out * k];
                    backend_cpu::par::rows_mut(&mut blk, k, |i, dst| {
                        let e0 = (r0 + i) * stride + c0;
                        dst.copy_from_slice(&w[e0..e0 + k]);
                    });
                    split_ns.set(split_ns.get() + tsp.elapsed().as_nanos());
                    lin_n(label, &blk, n_out, k)
                }
            })
        };
        // A whole stored tensor as one linear.
        let lin = |name: &str, n_out: usize, k: usize| -> Lin { lin_rect(name, name, k, 0, n_out, 0, k) };

        let stream = |p: &str| -> StreamW {
            let qkv_n = format!("{p}_attn.qkv.weight");
            let qkv_l = format!("{p}_attn.qkv");
            let m0_n = format!("{p}_mlp.0.weight");
            let m0_l = format!("{p}_mlp.0");
            StreamW {
                wq: lin_rect(&qkv_n, &qkv_l, d, 0, d, 0, d),
                wk: lin_rect(&qkv_n, &qkv_l, d, d, d, 0, d),
                wv: lin_rect(&qkv_n, &qkv_l, d, 2 * d, d, 0, d),
                nq: up(&format!("{p}_attn.norm.query_norm.scale")),
                nk: up(&format!("{p}_attn.norm.key_norm.scale")),
                wo: lin(&format!("{p}_attn.proj.weight"), d, d),
                // SwiGLU chunk order: x1 (silu-gated) is the FIRST half
                w1: lin_rect(&m0_n, &m0_l, d, 0, mlp, 0, d),
                w3: lin_rect(&m0_n, &m0_l, d, mlp, mlp, 0, d),
                // The double-block mlp-down stays fp32 (~850 MB over the 10
                // streams): its input is the SwiGLU activation, whose per-token
                // outliers early in the stack cost the most int8 parity —
                // measured cosine 0.9965 (int8 w2) → 0.9989 (fp32 w2) on the
                // parity fixture. The single-block hs consumer (wo_b) measured
                // insensitive (+0.0002 for 3 GB) and stays int8.
                w2: Lin::F32(up(&format!("{p}_mlp.2.weight"))),
            }
        };
        let dbl: Vec<(StreamW, StreamW)> = (0..cfg.depth_double)
            .map(|b| {
                (stream(&format!("double_blocks.{b}.img")), stream(&format!("double_blocks.{b}.txt")))
            })
            .collect();

        let sgl: Vec<SingleW> = (0..cfg.depth_single)
            .map(|b| {
                let p = format!("single_blocks.{b}");
                let l1_n = format!("{p}.linear1.weight");
                let l1_l = format!("{p}.linear1");
                // linear2 is [D, D+mlp]; wo_a/wo_b are its two COLUMN blocks,
                // which is why `lin_rect` takes a column range at all.
                let l2_n = format!("{p}.linear2.weight");
                let l2_l = format!("{p}.linear2");
                SingleW {
                    wq: lin_rect(&l1_n, &l1_l, d, 0, d, 0, d),
                    wk: lin_rect(&l1_n, &l1_l, d, d, d, 0, d),
                    wv: lin_rect(&l1_n, &l1_l, d, 2 * d, d, 0, d),
                    nq: up(&format!("{p}.norm.query_norm.scale")),
                    nk: up(&format!("{p}.norm.key_norm.scale")),
                    w1: lin_rect(&l1_n, &l1_l, d, 3 * d, mlp, 0, d),
                    w3: lin_rect(&l1_n, &l1_l, d, 3 * d + mlp, mlp, 0, d),
                    wo_a: lin_rect(&l2_n, &l2_l, d + mlp, 0, d, 0, d),
                    wo_b: lin_rect(&l2_n, &l2_l, d + mlp, 0, d, d, mlp),
                }
            })
            .collect();

        // Scratch spans the whole batch slab: b_max samples of n_max joint rows.
        let n = n_max as u64 * b_max as u64;
        let du = d as u64;
        let mlpu = mlp as u64;
        // Attention stays flash under `fast` at BOTH tiers: the materialised
        // scores→softmax→apply trio was measured SLOWER here (int8 @1536
        // joint tokens: the untiled scores/apply kernels are bandwidth-bound at
        // hd=128), unlike zimage's dims where the trio wins.
        // [B, H, T, T] — per SAMPLE T, not the whole slab (samples never mix).
        let attn_mat = if fast { 1 } else { b_max as u64 * nh as u64 * n_max as u64 * n_max as u64 };
        let a = |len: u64| gpu.storage(len);
        let scr = Scratch {
            x0: a(n * du),
            x1: a(n * du),
            n1: a(n * du),
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
            h1: a(n * mlpu),
            h2: a(n * mlpu),
            hs: a(n * mlpu),
            mlp: a(n * du),
            scores: a(attn_mat),
            probs: a(attn_mat),
            out: a(n * cfg.in_channels as u64),
            cos: a(n * (hd as u64 / 2)),
            sin: a(n * (hd as u64 / 2)),
            tok_in: a(n * cfg.in_channels as u64),
            ctx_in: a(b_max as u64 * cfg.txt_len as u64 * cfg.context_in_dim as u64),
        };
        let bd = b_max as u64 * du;
        let modb = ModBufs {
            gamma: (0..6).map(|_| a(bd)).collect(),
            beta: (0..6).map(|_| a(bd)).collect(),
            gate: (0..5).map(|_| a(bd)).collect(),
        };
        let i8scr = (precision == Precision::Int8).then(|| I8Scratch::new(&gpu, n, n, &[cfg.hidden as u32, cfg.mlp_hidden() as u32]));

        if gpu_core::profile::enabled() {
            let ms = |n: u128| n as f64 / 1e6;
            let w = ms(write_ns.get());
            let gb = bytes_up.get() as f64 / 1e9;
            eprintln!(
                "flux2 build: {:.0} ms total | quantize {:.0} ms | linear2 split {:.0} ms | write {:.0} ms ({gb:.1} GB, {:.2} GB/s) | staging flush {:.0} ms x{}",
                ms(t_build.elapsed().as_nanos()),
                ms(quant_ns.get()),
                ms(split_ns.get()),
                w,
                gb / (w / 1e3),
                ms(flush_ns.get()),
                flush_n.get(),
            );
        }

        Flux2Model {
            cfg: cfg.clone(),
            n_max,
            b_max,
            fast,
            precision,
            i8scr,
            dbl,
            sgl,
            // The three boundary linears stay fp32 at every tier (~97 MB —
            // negligible). Measured on the parity fixture (t2i, 1536 tokens):
            // quantizing txt_in costs cosine 0.9946 → 0.9843 alone — its input
            // is raw concatenated Qwen3 hidden states whose channel outliers
            // (the ~6e3 magnitudes the masking experiment measured) crush a
            // per-token int8 scale. img_in/final_layer are cheap insurance at
            // the in/out boundaries (0.9946 → 0.9955 together).
            img_in: Lin::F32(up("img_in.weight")),
            txt_in: Lin::F32(up("txt_in.weight")),
            final_w: Lin::F32(up("final_layer.linear.weight")),
            modb,
            scr,
            time_in_a: getv("time_in.in_layer.weight"),
            time_in_b: getv("time_in.out_layer.weight"),
            mod_img: getv("double_stream_modulation_img.lin.weight"),
            mod_txt: getv("double_stream_modulation_txt.lin.weight"),
            mod_single: getv("single_stream_modulation.lin.weight"),
            final_adaln: getv("final_layer.adaLN_modulation.1.weight"),
            gpu,
        }
    }

    /// The numeric tier this model was built at.
    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// The largest batch [`Flux2Model::forward_batch`] accepts (the scratch was
    /// sized for it at build time).
    pub fn max_batch(&self) -> u32 {
        self.b_max
    }

    /// Max joint tokens (txt + img + refs) per sample.
    pub fn max_tokens(&self) -> u32 {
        self.n_max
    }

    /// Host conditioning for ONE timestep: the timestep MLP + the three global
    /// modulation linears, folded into the six (gamma, beta) LN pairs and five
    /// gate vectors. Pure — the device write is [`Self::upload_modulation`].
    fn modulation_for(&self, t: f32) -> ModVals {
        let d = self.cfg.hidden;
        use model::hostmath::{matvec_par, silu_slice};
        // `time_factor = 1000` is the FLUX pipeline convention, applied here;
        // the embedding itself is `hostmath::timestep_embedding` (cos block
        // first, angles in f64) — shared with `flux1`, byte-for-byte the
        // local copy this replaced.
        let emb = model::hostmath::timestep_embedding(t * 1000.0, 256, true, 0.0, 10000.0);
        let h = silu_slice(&matvec_par(&self.time_in_a, &emb, d, 256));
        let vec_ = matvec_par(&self.time_in_b, &h, d, d);
        let sv = silu_slice(&vec_);

        let m_img = matvec_par(&self.mod_img, &sv, 6 * d, d);
        let m_txt = matvec_par(&self.mod_txt, &sv, 6 * d, d);
        let m_sgl = matvec_par(&self.mod_single, &sv, 3 * d, d);
        let m_fin = matvec_par(&self.final_adaln, &sv, 2 * d, d);

        // chunk order per triple: (shift, scale, gate); final layer: (shift, scale)
        let gamma = |m: &[f32], c: usize| -> Vec<f32> {
            m[(3 * c + 1) * d..(3 * c + 2) * d].iter().map(|s| 1.0 + s).collect()
        };
        let beta = |m: &[f32], c: usize| m[3 * c * d..(3 * c + 1) * d].to_vec();
        let gate = |m: &[f32], c: usize| m[(3 * c + 2) * d..(3 * c + 3) * d].to_vec();

        // sites: 0=img1 1=img2 2=txt1 3=txt2 4=single 5=final; gates likewise
        ModVals {
            gamma: [gamma(&m_img, 0), gamma(&m_img, 1), gamma(&m_txt, 0), gamma(&m_txt, 1), gamma(&m_sgl, 0), m_fin[d..2 * d].iter().map(|s| 1.0 + s).collect()],
            beta: [beta(&m_img, 0), beta(&m_img, 1), beta(&m_txt, 0), beta(&m_txt, 1), beta(&m_sgl, 0), m_fin[..d].to_vec()],
            gate: [gate(&m_img, 0), gate(&m_img, 1), gate(&m_txt, 0), gate(&m_txt, 1), gate(&m_sgl, 0)],
        }
    }

    /// Upload the per-sample modulation for a whole batch: each site's buffer
    /// becomes `[B, D]` with sample `b`'s vector at row `b`. Timesteps repeat
    /// whenever a batch steps in lockstep, so identical `t` values are computed
    /// once (the four host mat-vecs are ~132 MFLOP each at klein-4B).
    fn upload_modulation(&self, ts: &[f32]) {
        let d = self.cfg.hidden;
        let mut uniq: Vec<(u32, ModVals)> = Vec::new();
        let mut order: Vec<usize> = Vec::with_capacity(ts.len());
        for &t in ts {
            let key = t.to_bits();
            let at = match uniq.iter().position(|(k, _)| *k == key) {
                Some(i) => i,
                None => {
                    uniq.push((key, self.modulation_for(t)));
                    uniq.len() - 1
                }
            };
            order.push(at);
        }
        let mut buf = Vec::with_capacity(ts.len() * d);
        let mut wf = |dst: &DeviceBuffer, pick: &dyn Fn(&ModVals) -> &[f32]| {
            buf.clear();
            for &i in &order {
                buf.extend_from_slice(pick(&uniq[i].1));
            }
            self.gpu.write(dst, bytemuck::cast_slice(&buf));
        };
        for site in 0..6 {
            wf(&self.modb.gamma[site], &|m: &ModVals| m.gamma[site].as_slice());
            wf(&self.modb.beta[site], &|m: &ModVals| m.beta[site].as_slice());
            if site < 5 {
                wf(&self.modb.gate[site], &|m: &ModVals| m.gate[site].as_slice());
            }
        }
    }

    /// The register-tiled GEMM this model dispatches. `matmul_reg3` is
    /// `matmul_reg2` with the shared-memory bank conflicts removed: same tiling,
    /// same K accumulation order, therefore BIT-IDENTICAL output (verified
    /// max_abs 0.0 across all 12 of this graph's shapes), and measured faster
    /// on the klein-4B mix - by the widest margin on the narrow-K boundary
    /// linears, and never meaningfully slower on any shape in it (re-measure
    /// with `crates/flux2/src/bin/flux2_bench.rs`). `matmul_reg2` stays the
    /// default everywhere else in the
    /// workspace until each model measures its own shapes.
    fn mm_kernel(&self) -> usize {
        K_MATMUL_REG3
    }

    /// The fp32 GEMM tier this model dispatches, for `model::block::gemm_variant`
    /// — the rule shared with flux1.
    fn gemm_tier(&self) -> model::block::GemmVariants {
        if self.fast {
            // `gemv: None` deliberately, and this is MEASURED, not assumed: the
            // smallest M this model dispatches on the real klein-4B forward is
            // **512** (fp32: M in {512, 1024, 1536, 2048, 2560}; int8: {512,
            // 1024, 1536}), so not one GEMM would ever reach the GEMV kernel's
            // `m <= 32` arm. That is a consequence of the design in this file's
            // header — FLUX.2's modulation is GLOBAL and folded on the host, so
            // unlike flux1 (whose per-block modulation issues 77 `m = 1`
            // mat-vecs per forward) there is no skinny-M work on the device at
            // all. Registering the kernels would add two dead pipelines, and an
            // M-dependent kernel choice would put a hazard under
            // `tests/batch_parity.rs`, whose bit-identity claim rests on every
            // dispatch being independent of M.
            model::block::GemmVariants::Fast { gemv: None, tiled: self.mm_kernel() }
        } else {
            model::block::GemmVariants::Reference(K_MATMUL)
        }
    }

    fn mm(&self, x: &DeviceBuffer, w: &DeviceBuffer, o: &DeviceBuffer, m: u32, k: u32, n: u32) -> Step {
        let (kind, threads) = model::block::gemm_variant(self.gemm_tier(), m, n);
        self.gpu.step(kind, &[x, w, o], &[m, k, n], threads)
    }

    /// Sliced matmul: read rows `xr0..xr0+m` of `x`, write rows `or0..or0+m` of
    /// `o` (both row-major, `[.., k]` / `[.., n]`). Independent input/output row
    /// bases are what a sample-major batch slab needs — an embedding reads its
    /// sample's block of a compact `[B·rows, k]` input and writes into the
    /// sample's window of the joint slab.
    #[allow(clippy::too_many_arguments)]
    fn mm_rows_at(&self, x: &DeviceBuffer, w: &DeviceBuffer, o: &DeviceBuffer, xr0: u32, or0: u32, m: u32, k: u32, n: u32) -> Step {
        model::dispatch::mm_rows_off(&self.gpu, self.gemm_tier(), x, w, o, xr0, or0 as u64 * n as u64, m, k, n)
    }

    /// Int8 only (no-op under fp32): quantize rows `r0..r1` of `x` `[.., k]`
    /// into the K-matched packed scratch with fresh per-token scales
    /// (`max_abs_row` → `quant_pack`). ONE quant feeds every linear reading
    /// that activation (n1 → q/k/v and w1/w3, ctx → wo, hs → w2).
    fn quant_rows(&self, s: &mut Vec<Step>, x: &DeviceBuffer, r0: u32, r1: u32, k: u32) {
        let Some(i8s) = self.i8scr.as_ref() else { return };
        i8s.quant_rows(&self.gpu, [K_MAXABS, K_QUANT], s, x, r0, r1, k);
    }

    /// Int8 DP4A matmul over pre-quantized rows `xr0..xr0+m` of the K-matched
    /// packed scratch, writing rows `or0..or0+m` of `o` `[.., n]` — the shared
    /// `model::dispatch::mm8_rows_off`. Same selection rule and the same
    /// measured `gemv: None` as the fp32 tier — see `gemm_tier`.
    #[allow(clippy::too_many_arguments)]
    fn mm8(&self, wq: &DeviceBuffer, sw: &DeviceBuffer, o: &DeviceBuffer, xr0: u32, or0: u32, m: u32, k: u32, n: u32) -> Step {
        let i8s = self.i8scr.as_ref().expect("int8 scratch");
        let tier = model::block::GemmVariants::Fast { gemv: None, tiled: K_MATMUL_I8 };
        model::dispatch::mm8_rows_off(&self.gpu, tier, i8s, wq, sw, o, xr0, or0 as u64 * n as u64, m, k, n)
    }

    /// One linear over rows `r0..r1` at the model's tier: fp32 [`Self::mm_rows`]
    /// or the DP4A GEMM over the activation [`Self::quant_rows`] pre-packed.
    #[allow(clippy::too_many_arguments)]
    fn lin_rows(&self, x: &DeviceBuffer, w: &Lin, o: &DeviceBuffer, r0: u32, r1: u32, k: u32, n: u32) -> Step {
        self.lin_rows_at(x, w, o, r0, r0, r1 - r0, k, n)
    }

    /// [`Self::lin_rows`] with independent input/output row bases
    /// ([`Self::mm_rows_at`]).
    #[allow(clippy::too_many_arguments)]
    fn lin_rows_at(&self, x: &DeviceBuffer, w: &Lin, o: &DeviceBuffer, xr0: u32, or0: u32, m: u32, k: u32, n: u32) -> Step {
        match w {
            Lin::F32(wb) => self.mm_rows_at(x, wb, o, xr0, or0, m, k, n),
            Lin::I8(wq, sw) => self.mm8(wq, sw, o, xr0, or0, m, k, n),
        }
    }

    /// Whole-slab linear (`m` rows from row 0) at the model's tier — the fp32
    /// arm is the plain unsliced [`Self::mm`] (byte-identical to before).
    fn lin_full(&self, x: &DeviceBuffer, w: &Lin, o: &DeviceBuffer, m: u32, k: u32, n: u32) -> Step {
        match w {
            Lin::F32(wb) => self.mm(x, wb, o, m, k, n),
            Lin::I8(wq, sw) => self.mm8(wq, sw, o, 0, 0, m, k, n),
        }
    }

    /// Modulated LayerNorm over rows `r0..r1` under sample `b`'s modulation:
    /// `LN_noaffine·gamma[b] + beta[b]`. `layernorm` takes ONE `[D]` gamma/beta
    /// pair, so the sample selection is a binding slice — at `b = 0` this is the
    /// same binding the unbatched model made (its buffers were exactly `[D]`).
    fn ln_rows(&self, x: &DeviceBuffer, site: usize, o: &DeviceBuffer, b: u32, r0: u32, r1: u32) -> Step {
        let d = self.cfg.hidden as u32;
        let m = r1 - r0;
        let off = (r0 as u64 * d as u64, m as u64 * d as u64);
        let mo = (b as u64 * d as u64, d as u64);
        self.gpu.step_sliced(
            K_LN,
            &[x, &self.modb.gamma[site], &self.modb.beta[site], o],
            &[off, mo, mo, off],
            &[d, m, f(EPS)],
            m,
        )
    }

    /// Gated residual over rows `r0..r1`: `y = x + gate[b] ⊙ h` (one condition
    /// group — the whole range belongs to sample `b`).
    // Arity is the `gate_row` kernel's binding list + its row window, like the
    // other dispatch helpers in this file.
    #[allow(clippy::too_many_arguments)]
    fn gate_rows(&self, x: &DeviceBuffer, gi: usize, h: &DeviceBuffer, y: &DeviceBuffer, b: u32, r0: u32, r1: u32) -> Step {
        let d = self.cfg.hidden as u32;
        let m = r1 - r0;
        let off = (r0 as u64 * d as u64, m as u64 * d as u64);
        let go = (b as u64 * d as u64, d as u64);
        self.gpu.step_sliced(
            K_GATE,
            &[x, &self.modb.gate[gi], h, y],
            &[off, go, off, off],
            &[m, d, m],
            m * d,
        )
    }

    /// Gated residual over the WHOLE batch slab in ONE dispatch:
    /// `y[r] = x[r] + gate[r / rows_per_cond] ⊙ h[r]`. This is `gate_row`'s
    /// `rows_per_cond` condition-group contract (`NC = B`) doing per-sample
    /// modulation with no new kernel and no extra dispatch — the reason
    /// different timesteps per batch member are free. At `B = 1`,
    /// `rows_per_cond == rows` and the params equal [`Self::gate_rows`]'s.
    fn gate_grouped(&self, x: &DeviceBuffer, gi: usize, h: &DeviceBuffer, y: &DeviceBuffer, rows: u32, rows_per_cond: u32) -> Step {
        let d = self.cfg.hidden as u32;
        let off = (0u64, rows as u64 * d as u64);
        self.gpu.step_sliced(
            K_GATE,
            &[x, &self.modb.gate[gi], h, y],
            &[off, (0, 0), off, off],
            &[rows, d, rows_per_cond],
            rows * d,
        )
    }

    /// QK-RMSNorm over rows `r0..r1` (per-head rows of length `head_dim`).
    ///
    /// `head_dim` is 128, so the per-element kernel's one-thread-per-row layout
    /// makes every warp read 32 rows that are 128 floats apart — one useful
    /// float per 32-byte sector. The workgroup-per-row kernel is coalesced and
    /// measured an order of magnitude faster at exactly this shape (36864 rows
    /// x 128); the selection rule lives in
    /// `backend_api::select` (`Op::RmsNorm`), which now prefers it at EVERY
    /// row count on a device with workgroup barriers.
    fn qknorm_rows(&self, x: &DeviceBuffer, scale: &DeviceBuffer, o: &DeviceBuffer, r0: u32, r1: u32) -> Step {
        let d = self.cfg.hidden as u32;
        let hd = self.cfg.head_dim() as u32;
        let nh = self.cfg.n_heads as u32;
        let m = r1 - r0;
        let off = (r0 as u64 * d as u64, m as u64 * d as u64);
        let rows = m * nh;
        let (kind, threads) =
            model::block::rms_variant(&self.gpu, K_RMSNORM, Some(K_RMSNORM_ROWS), rows, hd);
        self.gpu.step_sliced(kind, &[x, scale, o], &[off, (0, 0), off], &[hd, rows, f(EPS)], threads)
    }

    /// Joint attention over `bsz` samples of `n` rows each. Both the flash and
    /// the materialised trio take **`bsz` as their first Param** and index
    /// `qkv[(b·T + j)·stride]`, so a sample-major slab is batched by raising
    /// `bsz` alone: no sample can attend to another's tokens, by construction.
    fn push_attention(&self, s: &mut Vec<Step>, bsz: u32, n: u32) {
        let nh = self.cfg.n_heads as u32;
        let hd = self.cfg.head_dim() as u32;
        let dim = self.cfg.hidden as u32;
        let scr = &self.scr;
        if self.fast {
            // The lane-split flash kernel where the device's workgroup limit
            // allows it (well over an order of magnitude on the baseline at
            // hd=128 - see `model::block::FlashIds`), else the baseline.
            s.push(model::block::flash_bidir_step(
                &self.gpu,
                model::block::FlashIds {
                    bidir: K_FLASH,
                    split: Some(K_FLASH_SPLIT),
                    reg: Some(K_FLASH_REG),
                    reg2: Some(K_FLASH_REG2),
                },
                bsz,
                nh,
                n,
                hd,
                dim,
                &scr.qkv,
                &scr.ctx,
            ));
        } else {
            s.push(self.gpu.step(K_SCORES, &[&scr.qkv, &scr.scores], &[bsz, nh, n, hd, 3 * dim, 0, dim], bsz * nh * n * n));
            s.push(self.gpu.step(K_SOFTMAX, &[&scr.scores, &scr.probs], &[bsz, nh, n], bsz * nh * n));
            s.push(self.gpu.step(K_APPLY, &[&scr.probs, &scr.qkv, &scr.ctx], &[bsz, nh, n, hd, 3 * dim, 2 * dim, dim], bsz * nh * n * hd));
        }
    }

    /// Attention core shared by both block kinds: qkv is already in
    /// `scr.q/k/v` (rope'd + packed here), result lands in `scr.ctx`. RoPE,
    /// packing and QK-norm are per-row ops, so they run over the whole `B·n`
    /// slab in one dispatch each; the cos/sin tables are the SAME table
    /// replicated per sample (same resolution ⇒ same ids), computed once on the
    /// host in [`Self::forward_batch`].
    fn push_attn_core(&self, s: &mut Vec<Step>, bsz: u32, n: u32) {
        let d = self.cfg.hidden as u32;
        let hd = self.cfg.head_dim() as u32;
        let nh = self.cfg.n_heads as u32;
        let half = hd / 2;
        let rows = bsz * n;
        let scr = &self.scr;
        s.push(self.gpu.step(K_ROPE, &[&scr.qn, &scr.cos, &scr.sin, &scr.qr], &[rows, nh, hd, half], rows * nh * half));
        s.push(self.gpu.step(K_ROPE, &[&scr.kn, &scr.cos, &scr.sin, &scr.kr], &[rows, nh, hd, half], rows * nh * half));
        s.push(self.gpu.step(K_PACK, &[&scr.qr, &scr.kr, &scr.v, &scr.qkv], &[rows, d], rows * 3 * d));
        self.push_attention(s, bsz, n);
    }

    /// Forward one denoising evaluation.
    ///
    /// `img_tokens`: packed latent tokens `[n_img, in_channels]` (noise image
    /// first, then any reference tokens). `ctx`: text conditioning
    /// `[txt_len, context_in_dim]`. `ids`: joint 4-axis position ids, **text
    /// rows first** then image/ref rows (`(txt_len + n_img) * 4`). Returns the
    /// prediction for the first `n_pred` image tokens `[n_pred, in_channels]`.
    ///
    /// This is [`Self::forward_batch`] at B = 1 and records exactly the same
    /// dispatch sequence it always did — the latency path is untouched.
    pub fn forward(&self, img_tokens: &[f32], ctx: &[f32], t: f32, ids: &[u32], n_pred: usize) -> Vec<f32> {
        let mut out = self.forward_batch(&[Sample { img_tokens, ctx, t }], ids, n_pred);
        out.pop().expect("one sample in, one out")
    }

    /// Forward `B = samples.len()` denoising evaluations in ONE device pass.
    ///
    /// Every sample carries its own latents, text conditioning and **timestep**;
    /// they share `ids` (hence the resolution and reference layout), which is
    /// what lets the RoPE tables be computed once and replicated. Returns one
    /// `[n_pred, in_channels]` prediction per sample, in input order.
    ///
    /// Bit-identical to running the samples one at a time: every kernel's
    /// per-output reduction order is independent of `M` (the register-tiled
    /// GEMMs accumulate over K within a fixed 128×128 tile) and of `bsz` (the
    /// attention kernels give each `(b, h, query-tile)` its own workgroup), and
    /// the per-row norms/quantizers are row-local. `tests/batch_parity.rs`
    /// asserts max_abs == 0.
    pub fn forward_batch(&self, samples: &[Sample<'_>], ids: &[u32], n_pred: usize) -> Vec<Vec<f32>> {
        let cfg = &self.cfg;
        let d = cfg.hidden as u32;
        let mlp = cfg.mlp_hidden() as u32;
        let cin = cfg.in_channels as u32;
        let nt = cfg.txt_len as u32;
        let bsz = samples.len() as u32;
        assert!(bsz >= 1, "forward_batch needs at least one sample");
        assert!(bsz <= self.b_max, "sized for batch {}, got {bsz}", self.b_max);
        let ni = (samples[0].img_tokens.len() / cfg.in_channels) as u32;
        let n = nt + ni;
        assert!(n <= self.n_max, "sized for {} joint tokens, got {n}", self.n_max);
        assert_eq!(ids.len() as u32, n * 4);
        assert!(n_pred as u32 <= ni);
        for (i, s) in samples.iter().enumerate() {
            assert_eq!(s.img_tokens.len(), (ni * cin) as usize, "sample {i}: latent length differs from sample 0");
            assert_eq!(s.ctx.len(), cfg.txt_len * cfg.context_in_dim, "sample {i}: ctx length");
        }
        // Every per-sample buffer slice is bound at a byte offset, and storage
        // bindings must respect `min_storage_buffer_offset_alignment` (256 B =
        // 64 floats on the P40). Name the offending stride here rather than let
        // it surface as a wgpu validation error (P9 finding, status.md).
        if bsz > 1 && self.fast {
            let al = |what: &str, v: u64| {
                assert!(v.is_multiple_of(64), "flux2 batched forward: {what} = {v} floats is not a multiple of 64 (256-byte storage-binding alignment); use B=1 at these dims");
            };
            al("hidden", d as u64);
            al("mlp_hidden", mlp as u64);
            al("txt_len * context_in_dim", nt as u64 * cfg.context_in_dim as u64);
            al("n_img * in_channels", ni as u64 * cin as u64);
            al("n_pred * in_channels", n_pred as u64 * cin as u64);
            if self.precision == Precision::Int8 {
                al("joint tokens (int8 per-token scale offset)", n as u64);
            }
        }

        // Debug aid (batch profiling): BRAIN_FLUX2_TIME_FORWARD=1 splits the
        // forward into host conditioning / input upload / step recording /
        // device execution, which is how the B≥4 batch-scaling plateau
        // was attributed to a specific stage.
        let timed = std::env::var("BRAIN_FLUX2_TIME_FORWARD").is_ok();
        let t_start = std::time::Instant::now();

        let ts: Vec<f32> = samples.iter().map(|s| s.t).collect();
        self.upload_modulation(&ts);
        let t_mod = t_start.elapsed();

        // RoPE tables from the joint ids (t, h, w, l), interleaved pairs. The
        // ids are shared, so the (expensive) host table build happens ONCE and
        // the result is replicated per sample — the kernel indexes the table by
        // absolute slab row.
        let rc = dit::rope::RopeConfig {
            axes_dims: cfg.axes_dim.iter().map(|&a| a as u32).collect(),
            axes_lens: vec![4096, 4096, 4096, 4096],
            theta: cfg.rope_theta,
        };
        let tables = dit::rope::tables_for_ids(&rc, ids, 4);
        let tile = |v: &[f32]| -> Vec<f32> {
            let mut out = Vec::with_capacity(v.len() * bsz as usize);
            for _ in 0..bsz {
                out.extend_from_slice(v);
            }
            out
        };
        if bsz == 1 {
            self.gpu.write(&self.scr.cos, bytemuck::cast_slice(&tables.cos));
            self.gpu.write(&self.scr.sin, bytemuck::cast_slice(&tables.sin));
        } else {
            self.gpu.write(&self.scr.cos, bytemuck::cast_slice(&tile(&tables.cos)));
            self.gpu.write(&self.scr.sin, bytemuck::cast_slice(&tile(&tables.sin)));
        }

        // Inputs concatenate sample-major, matching the slab.
        if bsz == 1 {
            self.gpu.write(&self.scr.tok_in, bytemuck::cast_slice(samples[0].img_tokens));
            self.gpu.write(&self.scr.ctx_in, bytemuck::cast_slice(samples[0].ctx));
        } else {
            let mut toks = Vec::with_capacity(samples[0].img_tokens.len() * bsz as usize);
            let mut ctxs = Vec::with_capacity(samples[0].ctx.len() * bsz as usize);
            for s in samples {
                toks.extend_from_slice(s.img_tokens);
                ctxs.extend_from_slice(s.ctx);
            }
            self.gpu.write(&self.scr.tok_in, bytemuck::cast_slice(&toks));
            self.gpu.write(&self.scr.ctx_in, bytemuck::cast_slice(&ctxs));
        }

        let t_up = t_start.elapsed();
        let scr = &self.scr;
        let rows = bsz * n; // whole-slab row count
        let mut s: Vec<Step> = Vec::new();
        // Embed both streams into the joint residual slab x0 = [txt | img] per
        // sample. The embeds are fp32 at every tier (see `new_with`), so no
        // quant here. Both are per-sample because the destination rows are
        // n-strided (sample-major slab).
        for b in 0..bsz {
            let base = b * n;
            let ctx_r0 = b * nt; // ctx_in is [B*txt_len, context_in_dim]
            s.push(self.lin_rows_at(&scr.ctx_in, &self.txt_in, &scr.x0, ctx_r0, base, nt, cfg.context_in_dim as u32, d));
            s.push(self.lin_rows_at(&scr.tok_in, &self.img_in, &scr.x0, b * ni, base + nt, ni, cin, d));
        }

        let (mut xa, mut xb) = (&scr.x0, &scr.x1);
        // sites: 0=img1 1=img2 2=txt1 3=txt2; gates likewise
        for (img_w, txt_w) in &self.dbl {
            // attention halves of both streams into the joint q/k/v
            for b in 0..bsz {
                let (t0, t1) = (b * n, b * n + nt);
                let (i0, i1) = (b * n + nt, (b + 1) * n);
                s.push(self.ln_rows(xa, 2, &scr.n1, b, t0, t1)); // txt norm1
                if txt_w.wq.is_i8() {
                    self.quant_rows(&mut s, &scr.n1, t0, t1, d);
                }
                s.push(self.lin_rows(&scr.n1, &txt_w.wq, &scr.q, t0, t1, d, d));
                s.push(self.lin_rows(&scr.n1, &txt_w.wk, &scr.k, t0, t1, d, d));
                s.push(self.lin_rows(&scr.n1, &txt_w.wv, &scr.v, t0, t1, d, d));
                s.push(self.ln_rows(xa, 0, &scr.n1, b, i0, i1)); // img norm1
                if img_w.wq.is_i8() {
                    self.quant_rows(&mut s, &scr.n1, i0, i1, d);
                }
                s.push(self.lin_rows(&scr.n1, &img_w.wq, &scr.q, i0, i1, d, d));
                s.push(self.lin_rows(&scr.n1, &img_w.wk, &scr.k, i0, i1, d, d));
                s.push(self.lin_rows(&scr.n1, &img_w.wv, &scr.v, i0, i1, d, d));
                s.push(self.qknorm_rows(&scr.q, &txt_w.nq, &scr.qn, t0, t1));
                s.push(self.qknorm_rows(&scr.k, &txt_w.nk, &scr.kn, t0, t1));
                s.push(self.qknorm_rows(&scr.q, &img_w.nq, &scr.qn, i0, i1));
                s.push(self.qknorm_rows(&scr.k, &img_w.nk, &scr.kn, i0, i1));
            }
            self.push_attn_core(&mut s, bsz, n);
            // per-stream projection + gated residual (ONE ctx quant, two GEMMs)
            if txt_w.wo.is_i8() || img_w.wo.is_i8() {
                self.quant_rows(&mut s, &scr.ctx, 0, rows, d);
            }
            for b in 0..bsz {
                let (t0, t1) = (b * n, b * n + nt);
                let (i0, i1) = (b * n + nt, (b + 1) * n);
                s.push(self.lin_rows(&scr.ctx, &txt_w.wo, &scr.proj, t0, t1, d, d));
                s.push(self.lin_rows(&scr.ctx, &img_w.wo, &scr.proj, i0, i1, d, d));
                s.push(self.gate_rows(xa, 2, &scr.proj, xb, b, t0, t1));
                s.push(self.gate_rows(xa, 0, &scr.proj, xb, b, i0, i1));
            }
            std::mem::swap(&mut xa, &mut xb);
            // MLP halves
            for b in 0..bsz {
                let (t0, t1) = (b * n, b * n + nt);
                let (i0, i1) = (b * n + nt, (b + 1) * n);
                s.push(self.ln_rows(xa, 3, &scr.n1, b, t0, t1)); // txt norm2
                if txt_w.w1.is_i8() {
                    self.quant_rows(&mut s, &scr.n1, t0, t1, d);
                }
                s.push(self.lin_rows(&scr.n1, &txt_w.w1, &scr.h1, t0, t1, d, mlp));
                s.push(self.lin_rows(&scr.n1, &txt_w.w3, &scr.h2, t0, t1, d, mlp));
                s.push(self.ln_rows(xa, 1, &scr.n1, b, i0, i1)); // img norm2
                if img_w.w1.is_i8() {
                    self.quant_rows(&mut s, &scr.n1, i0, i1, d);
                }
                s.push(self.lin_rows(&scr.n1, &img_w.w1, &scr.h1, i0, i1, d, mlp));
                s.push(self.lin_rows(&scr.n1, &img_w.w3, &scr.h2, i0, i1, d, mlp));
            }
            s.push(self.gpu.step(K_SILU_MUL, &[&scr.h1, &scr.h2, &scr.hs], &[rows * mlp], rows * mlp));
            if txt_w.w2.is_i8() || img_w.w2.is_i8() {
                self.quant_rows(&mut s, &scr.hs, 0, rows, mlp);
            }
            for b in 0..bsz {
                let (t0, t1) = (b * n, b * n + nt);
                let (i0, i1) = (b * n + nt, (b + 1) * n);
                s.push(self.lin_rows(&scr.hs, &txt_w.w2, &scr.mlp, t0, t1, mlp, d));
                s.push(self.lin_rows(&scr.hs, &img_w.w2, &scr.mlp, i0, i1, mlp, d));
                s.push(self.gate_rows(xa, 3, &scr.mlp, xb, b, t0, t1));
                s.push(self.gate_rows(xa, 1, &scr.mlp, xb, b, i0, i1));
            }
            std::mem::swap(&mut xa, &mut xb);
        }

        for w in &self.sgl {
            // parallel attn ‖ MLP over one shared modulated norm — ONE n1 quant
            // feeds q/k/v AND w1/w3 (attention touches no d-width quant state).
            // Everything but the LN is stream-agnostic, so the whole B·n slab
            // goes through in ONE dispatch each: this is where batching buys the
            // GEMM its extra rows (20 of klein-4B's 25 blocks).
            for b in 0..bsz {
                s.push(self.ln_rows(xa, 4, &scr.n1, b, b * n, (b + 1) * n));
            }
            if w.wq.is_i8() || w.w1.is_i8() {
                self.quant_rows(&mut s, &scr.n1, 0, rows, d);
            }
            s.push(self.lin_full(&scr.n1, &w.wq, &scr.q, rows, d, d));
            s.push(self.lin_full(&scr.n1, &w.wk, &scr.k, rows, d, d));
            s.push(self.lin_full(&scr.n1, &w.wv, &scr.v, rows, d, d));
            s.push(self.qknorm_rows(&scr.q, &w.nq, &scr.qn, 0, rows));
            s.push(self.qknorm_rows(&scr.k, &w.nk, &scr.kn, 0, rows));
            self.push_attn_core(&mut s, bsz, n);
            s.push(self.lin_full(&scr.n1, &w.w1, &scr.h1, rows, d, mlp));
            s.push(self.lin_full(&scr.n1, &w.w3, &scr.h2, rows, d, mlp));
            s.push(self.gpu.step(K_SILU_MUL, &[&scr.h1, &scr.h2, &scr.hs], &[rows * mlp], rows * mlp));
            // linear2 over cat(attn, mlp): two column-split matmuls, summed.
            // ctx is quantized only now — after w1/w3 consumed the n1 packing.
            if w.wo_a.is_i8() {
                self.quant_rows(&mut s, &scr.ctx, 0, rows, d);
            }
            s.push(self.lin_full(&scr.ctx, &w.wo_a, &scr.proj, rows, d, d));
            if w.wo_b.is_i8() {
                self.quant_rows(&mut s, &scr.hs, 0, rows, mlp);
            }
            s.push(self.lin_full(&scr.hs, &w.wo_b, &scr.mlp, rows, mlp, d));
            // y = x + gate ⊙ proj ; then y += gate ⊙ mlp (two gated adds), each
            // one dispatch over the batch via `rows_per_cond = n` groups.
            s.push(self.gate_grouped(xa, 4, &scr.proj, xb, rows, n));
            std::mem::swap(&mut xa, &mut xb);
            s.push(self.gate_grouped(xa, 4, &scr.mlp, xb, rows, n));
            std::mem::swap(&mut xa, &mut xb);
        }

        // final layer on each sample's predicted image rows only
        for b in 0..bsz {
            let p0 = b * n + nt;
            let p1 = p0 + n_pred as u32;
            s.push(self.ln_rows(xa, 5, &scr.n1, b, p0, p1));
            if matches!(self.final_w, Lin::I8(..)) {
                self.quant_rows(&mut s, &scr.n1, p0, p1, d);
            }
            s.push(self.lin_rows_at(&scr.n1, &self.final_w, &scr.out, p0, b * n_pred as u32, n_pred as u32, d, cin));
        }

        // debug aid: SMOKE_STEPS=k submits only the first k steps
        let take = std::env::var("SMOKE_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(s.len());
        let t_rec = t_start.elapsed();
        let nsteps = s.len();
        self.gpu.submit(&[], &s[..take.min(s.len())]);
        let flat = self.gpu.read(&self.scr.out, bsz as usize * n_pred * cfg.in_channels);
        if timed {
            let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
            eprintln!(
                "flux2 forward B={bsz} n={n} steps={nsteps}: modulation {:.1} ms, upload {:.1} ms, record {:.1} ms, device {:.1} ms, total {:.1} ms",
                ms(t_mod),
                ms(t_up - t_mod),
                ms(t_rec - t_up),
                ms(t_start.elapsed() - t_rec),
                ms(t_start.elapsed())
            );
        }
        flat.chunks(n_pred * cfg.in_channels).map(<[f32]>::to_vec).collect()
    }
}

/// One sample of a batched DiT forward ([`Flux2Model::forward_batch`]).
///
/// The batch shares position ids (resolution + reference layout) and therefore
/// the RoPE tables; everything else — latents, text conditioning and the
/// **timestep** — is per sample, which is what lets requests with different
/// seeds, step counts and CFG settings ride in one forward.
pub struct Sample<'a> {
    /// packed latent tokens `[n_img, in_channels]` (noise image, then refs)
    pub img_tokens: &'a [f32],
    /// text conditioning `[txt_len, context_in_dim]`
    pub ctx: &'a [f32],
    /// this sample's sigma / timestep
    pub t: f32,
}

/// Joint 4-axis position ids in the reference layout, text rows first.
///
/// Text tokens: `(0,0,0,l)`; generated image: `(0,h,w,0)` raster-major; each
/// reference image i: `(10·(i+1), h, w, 0)`. `refs` are (height, width) in
/// latent-token units.
pub fn position_ids(txt_len: usize, lh: usize, lw: usize, refs: &[(usize, usize)]) -> Vec<u32> {
    let mut ids = Vec::with_capacity((txt_len + lh * lw) * 4);
    for l in 0..txt_len {
        ids.extend([0, 0, 0, l as u32]);
    }
    for h in 0..lh {
        for w in 0..lw {
            ids.extend([0, h as u32, w as u32, 0]);
        }
    }
    for (i, &(rh, rw)) in refs.iter().enumerate() {
        let t = 10 * (i as u32 + 1);
        for h in 0..rh {
            for w in 0..rw {
                ids.extend([t, h as u32, w as u32, 0]);
            }
        }
    }
    ids
}
