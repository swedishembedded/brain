// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Z-Image single-stream transformer block as a recorded brain kernel graph.
//!
//! Mirrors diffusers `ZImageTransformerBlock.forward` (single-stream, global
//! modulation): a double-RMSNorm sandwich per sub-block with adaLN scale on the
//! pre-norm and adaLN gate on the post-norm, QK-normalized attention with
//! multi-axis interleaved RoPE, and a SwiGLU MLP. adaLN scale/gate are folded
//! into the four RMSNorm weights on the host (`rmsnorm(x,w)·scale =
//! rmsnorm(x,w·scale)`, `gate·rmsnorm(y,w)=rmsnorm(y,w·gate)`), so no scale/gate
//! kernels are needed.
//!
//! The step-builder ([`build_block_steps`]) and resident weights ([`BlockWeights`])
//! are shared: [`ZImageBlock`] owns a device and runs one block (the parity
//! reference), while the device-resident chain (`crate::dev`) uploads every
//! block's weights once and records the whole stack into one graph.

use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu, Step};

/// Z-Image RMSNorm epsilon (diffusers `norm_eps` / attention `eps`).
pub(crate) const EPS: f32 = 1e-5;

// Kernel-table indices (order matches KERNELS).
pub(crate) const K_RMSNORM: usize = 0;
pub(crate) const K_MATMUL: usize = 1;
pub(crate) const K_ROPE: usize = 2;
pub(crate) const K_PACK: usize = 3;
pub(crate) const K_SCORES: usize = 4;
pub(crate) const K_SOFTMAX: usize = 5;
pub(crate) const K_APPLY: usize = 6;
pub(crate) const K_SILU_MUL: usize = 7;
pub(crate) const K_ADD2: usize = 8;
pub(crate) const K_MATMUL_REG3: usize = 9;
pub(crate) const K_QUANT_PACK: usize = 10;
pub(crate) const K_MATMUL_I8: usize = 11;
pub(crate) const K_MAX_ABS_ROW: usize = 12;
pub(crate) const K_FLASH: usize = 13;
pub(crate) const K_FLASH_SPLIT: usize = 14;

pub(crate) const KERNELS: [(&str, &str); 15] = [
    ("rmsnorm_eps", kernels::RMSNORM_EPS),
    ("matmul", kernels::MATMUL),
    ("rope_interleave_table", kernels::ROPE_INTERLEAVE_TABLE),
    ("pack_qkv", kernels::PACK_QKV),
    ("attn_scores_bidir", kernels::ATTN_SCORES_BIDIR),
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
    ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR),
    ("silu_mul", kernels::SILU_MUL),
    ("add2", kernels::ADD2),
    // GPU-only fast GEMM (software-pipelined register tiling). The CPU JIT can't
    // compile its barrier, so CPU uses the naive `matmul` (native AVX2 path).
    ("matmul_reg3", kernels::MATMUL_REG3),
    // int8 DP4A path (GPU only): per-token activation quant (max_abs_row) + DP4A GEMM.
    ("quant_pack", kernels::QUANT_PACK),
    ("matmul_i8_dyn", kernels::MATMUL_I8_DYN),
    ("max_abs_row", kernels::MAX_ABS_ROW),
    // Flash attention: fused scores/softmax/apply with online softmax, O(T·hd)
    // memory (no materialised [nh·T·T]). Enables high-resolution latents.
    ("flash_attn_bidir", kernels::FLASH_ATTN_BIDIR),
    ("flash_attn_bidir_split", kernels::FLASH_ATTN_BIDIR_SPLIT),
];

/// Host tensors by name → `(shape, row-major f32 data)`. Implements
/// `checkpoint::TensorSource` (the impl lives in `checkpoint` itself — the
/// orphan rule blocks it here, see that impl's doc). Production builders no
/// longer construct a whole one of these -- they take a streaming
/// `&dyn checkpoint::TensorSource` (a mmap'd `WeightReader`, typically
/// wrapped in a `checkpoint::remap::RemapSource`) so a build's host peak
/// stays one tensor, never the whole model. `Tensors` survives as the
/// eager, in-memory shape that seam still needs to serve: test fixtures
/// (`ZImageBlock`, the golden-parity tests, `zimage_bench`) that already
/// have a small, wholly-materialized checkpoint in hand and have no reason
/// to build one.
pub type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

/// Shape parameters of one Z-Image block.
#[derive(Clone, Copy, Debug)]
pub struct BlockDims {
    pub dim: u32,
    pub n_heads: u32,
    pub head_dim: u32,
    /// adaLN conditioning width = `min(dim, 256)`.
    pub cdim: u32,
    /// SwiGLU hidden width = `dim*8/3`.
    pub hidden: u32,
}

impl BlockDims {
    pub fn new(dim: u32, n_heads: u32) -> BlockDims {
        BlockDims { dim, n_heads, head_dim: dim / n_heads, cdim: dim.min(256), hidden: dim * 8 / 3 }
    }
}

pub(crate) fn wf(gpu: &Gpu, buf: &DeviceBuffer, data: &[f32]) {
    let bits: Vec<u32> = data.iter().map(|v| v.to_bits()).collect();
    gpu.write(buf, &bits);
}

/// Per-upload chunk size, in elements (4 MiB as f32/u32 words) — mirrors
/// `paramstore::UPLOAD_CHUNK_WORDS`; not shared directly since paramstore's
/// upload path is `ParamStore`-shaped (weight/grad/Adam roles) and this one
/// is a bare device-buffer helper zimage's block builders call directly.
const UPLOAD_CHUNK_WORDS: usize = 1 << 20;

/// Upload tensor `name` from `src` into a fresh device buffer, bounding host
/// scratch to one chunk (never one tensor) whenever the source can say its
/// size without decoding (`numel`) — every real source this crate builds
/// from (`Tensors`, a `WeightReader`/`RemapSource` pair) can. Falls back to
/// one unbounded `with_tensor` call only for a source that can't report
/// `numel` cheaply, reproducing exactly what every caller here did before
/// this existed. Panics if `name` is missing — every caller here already had
/// that panic, just spelled `t.get(..).unwrap_or_else(..)`.
pub(crate) fn upload_named(gpu: &Gpu, src: &dyn checkpoint::TensorSource, name: &str) -> DeviceBuffer {
    if let Some(words) = src.raw_words(name) {
        let b = gpu.storage(words.len() as u64);
        for (i, part) in words.chunks(UPLOAD_CHUNK_WORDS).enumerate() {
            gpu.write_at(&b, (i * UPLOAD_CHUNK_WORDS) as u64, part);
        }
        return b;
    }
    if let Some(numel) = src.numel(name) {
        let b = gpu.storage(numel as u64);
        let mut total = 0usize;
        let found = src.with_tensor_chunks(name, UPLOAD_CHUNK_WORDS, &mut |off, chunk| {
            gpu.write_f32_at(&b, off, chunk);
            total += chunk.len();
        });
        assert!(found, "zimage: missing {name}");
        assert_eq!(total, numel, "size mismatch for {name}");
        return b;
    }
    let mut buf: Option<DeviceBuffer> = None;
    let found = src.with_tensor(name, &mut |data| {
        let b = gpu.storage(data.len() as u64);
        wf(gpu, &b, data);
        buf = Some(b);
    });
    assert!(found, "zimage: missing {name}");
    buf.expect("with_tensor found the tensor, so it must have set buf")
}

/// [`upload_named`], but into an already-allocated buffer instead of a fresh
/// one — the windowed engine's fixed slot pool calls this on every miss to
/// refresh a reused device buffer with a different block's data, so a
/// window's device footprint never grows past its build-time allocation no
/// matter how many distinct blocks rotate through it over a run.
pub(crate) fn upload_named_into(gpu: &Gpu, buf: &DeviceBuffer, src: &dyn checkpoint::TensorSource, name: &str) {
    if let Some(words) = src.raw_words(name) {
        for (i, part) in words.chunks(UPLOAD_CHUNK_WORDS).enumerate() {
            gpu.write_at(buf, (i * UPLOAD_CHUNK_WORDS) as u64, part);
        }
        return;
    }
    let mut total = 0usize;
    let found = src.with_tensor_chunks(name, UPLOAD_CHUNK_WORDS, &mut |off, chunk| {
        gpu.write_f32_at(buf, off, chunk);
        total += chunk.len();
    });
    assert!(found, "zimage: missing {name}");
    assert!(total > 0, "zimage: {name} decoded to zero elements");
}

/// [`upload_named`], returning the host f32 values too (for the small,
/// explicitly-retained tensors — norm weights, adaLN projections — that are
/// read again every forward, not just uploaded once). Bounded the same way;
/// only ever called on tensors small enough that this is a modest, declared
/// retention, never the whole model.
pub(crate) fn read_named(src: &dyn checkpoint::TensorSource, name: &str) -> Vec<f32> {
    let mut out = Vec::new();
    let found = src.with_tensor_chunks(name, UPLOAD_CHUNK_WORDS, &mut |off, chunk| {
        let off = off as usize;
        if out.len() < off + chunk.len() {
            out.resize(off + chunk.len(), 0.0);
        }
        out[off..off + chunk.len()].copy_from_slice(chunk);
    });
    assert!(found, "zimage: missing {name}");
    out
}

/// Resident (upload-once) static weights of one block.
pub(crate) struct BlockWeights {
    pub wq: DeviceBuffer,
    pub wk: DeviceBuffer,
    pub wv: DeviceBuffer,
    pub wo: DeviceBuffer,
    pub nq: DeviceBuffer,
    pub nk: DeviceBuffer,
    pub w1: DeviceBuffer,
    pub w2: DeviceBuffer,
    pub w3: DeviceBuffer,
}

impl BlockWeights {
    pub fn upload(gpu: &Gpu, t: &dyn checkpoint::TensorSource, prefix: &str) -> BlockWeights {
        // Upload via storage()+write() (DEVICE_LOCAL + transient staging), NOT
        // storage_init(): create_buffer_init's mapped-at-creation path forces
        // weight buffers into an inefficient memory type on a non-ReBAR P40,
        // ballooning ~12 GB of weights to ~22 GB (OOM). Plain DEVICE_LOCAL
        // buffers pack tightly (a raw-alloc probe holds 22 GB cleanly).
        // upload_named additionally bounds host scratch to one chunk, never
        // one tensor, whenever the source can report `numel` cheaply.
        let dev = |n: &str| upload_named(gpu, t, &format!("{prefix}.{n}"));
        BlockWeights {
            wq: dev("attention.to_q.weight"),
            wk: dev("attention.to_k.weight"),
            wv: dev("attention.to_v.weight"),
            wo: dev("attention.to_out.0.weight"),
            nq: dev("attention.norm_q.weight"),
            nk: dev("attention.norm_k.weight"),
            w1: dev("feed_forward.w1.weight"),
            w2: dev("feed_forward.w2.weight"),
            w3: dev("feed_forward.w3.weight"),
        }
    }

    /// Allocate one block's device buffers at `d`'s shape, with no tensor
    /// data loaded yet — the windowed engine's fixed slot pool calls this
    /// exactly `budget` times at build, then reuses the same buffers for
    /// every block that slot ever holds via [`Self::load_into`]. Same sizes
    /// [`Self::upload`] would produce for a real checkpoint at this shape
    /// (`quantize_weight`'s row/col convention: `to_*`/`to_out` are
    /// `dim×dim`, `w1`/`w3` are `hidden×dim`, `w2` is `dim×hidden` — see
    /// `Int8Weights::upload`'s identical `(no, k)` sizing).
    pub fn alloc(gpu: &Gpu, d: BlockDims) -> BlockWeights {
        let (dim, hid, hd) = (d.dim as u64, d.hidden as u64, d.head_dim as u64);
        BlockWeights {
            wq: gpu.storage(dim * dim),
            wk: gpu.storage(dim * dim),
            wv: gpu.storage(dim * dim),
            wo: gpu.storage(dim * dim),
            nq: gpu.storage(hd),
            nk: gpu.storage(hd),
            w1: gpu.storage(hid * dim),
            w2: gpu.storage(dim * hid),
            w3: gpu.storage(hid * dim),
        }
    }

    /// Overwrite this slot's buffers with `prefix`'s block weights — reuses
    /// the allocation from [`Self::alloc`]; no new device buffer is ever
    /// created after build, which is what keeps a window's device footprint
    /// fixed regardless of how many distinct blocks rotate through it.
    pub fn load_into(&self, gpu: &Gpu, t: &dyn checkpoint::TensorSource, prefix: &str) {
        let dev = |buf: &DeviceBuffer, n: &str| upload_named_into(gpu, buf, t, &format!("{prefix}.{n}"));
        dev(&self.wq, "attention.to_q.weight");
        dev(&self.wk, "attention.to_k.weight");
        dev(&self.wv, "attention.to_v.weight");
        dev(&self.wo, "attention.to_out.0.weight");
        dev(&self.nq, "attention.norm_q.weight");
        dev(&self.nk, "attention.norm_k.weight");
        dev(&self.w1, "feed_forward.w1.weight");
        dev(&self.w2, "feed_forward.w2.weight");
        dev(&self.w3, "feed_forward.w3.weight");
    }
}

/// The four per-forward folded-norm buffers (rewritten each forward from the
/// timestep conditioning; see [`fold_adaln`]).
pub(crate) struct NormBufs {
    pub an1: DeviceBuffer,
    pub an2: DeviceBuffer,
    pub fn1: DeviceBuffer,
    pub fn2: DeviceBuffer,
    // Host copies of the raw norm weights + adaLN projection (for folding).
    pub raw_an1: Vec<f32>,
    pub raw_an2: Vec<f32>,
    pub raw_fn1: Vec<f32>,
    pub raw_fn2: Vec<f32>,
    pub adaln_w: Vec<f32>,
    pub adaln_b: Vec<f32>,
    pub modulation: bool,
}

impl NormBufs {
    pub fn new(gpu: &Gpu, t: &dyn checkpoint::TensorSource, prefix: &str, dim: u32, modulation: bool) -> NormBufs {
        let get = |n: &str| read_named(t, &format!("{prefix}.{n}"));
        NormBufs {
            an1: gpu.storage(dim as u64),
            an2: gpu.storage(dim as u64),
            fn1: gpu.storage(dim as u64),
            fn2: gpu.storage(dim as u64),
            raw_an1: get("attention_norm1.weight"),
            raw_an2: get("attention_norm2.weight"),
            raw_fn1: get("ffn_norm1.weight"),
            raw_fn2: get("ffn_norm2.weight"),
            adaln_w: if modulation { get("adaLN_modulation.0.weight") } else { Vec::new() },
            adaln_b: if modulation { get("adaLN_modulation.0.bias") } else { Vec::new() },
            modulation,
        }
    }

    /// Fold the timestep conditioning `c` into the four norm weights and upload.
    pub fn upload_folded(&self, gpu: &Gpu, c: &[f32], dim: usize, cdim: usize) {
        let (an1, an2, fn1, fn2) = fold_adaln(self, c, dim, cdim);
        wf(gpu, &self.an1, &an1);
        wf(gpu, &self.an2, &an2);
        wf(gpu, &self.fn1, &fn1);
        wf(gpu, &self.fn2, &fn2);
    }

    /// Allocate one slot's fixed `an1`/`an2`/`fn1`/`fn2` device buffers with
    /// no host norm weights loaded yet — the windowed engine's fixed slot
    /// pool calls this once per slot; [`Self::reload_host`] refreshes the
    /// host arrays on a miss, reusing these same device buffers (they are
    /// rewritten every forward anyway, via [`Self::upload_folded`], so they
    /// need no reload themselves — only the host raw/adaLN arrays do).
    pub fn alloc(gpu: &Gpu, dim: u32, modulation: bool) -> NormBufs {
        NormBufs {
            an1: gpu.storage(dim as u64),
            an2: gpu.storage(dim as u64),
            fn1: gpu.storage(dim as u64),
            fn2: gpu.storage(dim as u64),
            raw_an1: Vec::new(),
            raw_an2: Vec::new(),
            raw_fn1: Vec::new(),
            raw_fn2: Vec::new(),
            adaln_w: Vec::new(),
            adaln_b: Vec::new(),
            modulation,
        }
    }

    /// Refresh the host-side norm/adaLN arrays for a new block identity —
    /// the counterpart to [`BlockWeights::load_into`] for the tiny host
    /// tensors [`fold_adaln`] reads every forward. The four device buffers
    /// are untouched here; they get rewritten by the next
    /// [`Self::upload_folded`] regardless of which block occupies the slot.
    pub fn reload_host(&mut self, t: &dyn checkpoint::TensorSource, prefix: &str) {
        let get = |n: &str| read_named(t, &format!("{prefix}.{n}"));
        self.raw_an1 = get("attention_norm1.weight");
        self.raw_an2 = get("attention_norm2.weight");
        self.raw_fn1 = get("ffn_norm1.weight");
        self.raw_fn2 = get("ffn_norm2.weight");
        if self.modulation {
            self.adaln_w = get("adaLN_modulation.0.weight");
            self.adaln_b = get("adaLN_modulation.0.bias");
        }
    }
}

/// adaLN fold: `mod = adaLN_w·c + adaLN_b` → `(scale_msa, gate_msa, scale_mlp,
/// gate_mlp)`; norms become `raw·(1+scale)` / `raw·tanh(gate)`. When
/// `modulation=false` the raw norm weights pass through unchanged.
pub(crate) fn fold_adaln(nb: &NormBufs, c: &[f32], dim: usize, cdim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    if !nb.modulation {
        return (nb.raw_an1.clone(), nb.raw_an2.clone(), nb.raw_fn1.clone(), nb.raw_fn2.clone());
    }
    let mut m = vec![0f32; 4 * dim];
    for (i, mi) in m.iter_mut().enumerate() {
        let mut acc = nb.adaln_b[i];
        for (wj, &cj) in nb.adaln_w[i * cdim..i * cdim + cdim].iter().zip(c) {
            acc += wj * cj;
        }
        *mi = acc;
    }
    let fold_scale = |raw: &[f32], s: &[f32]| -> Vec<f32> { raw.iter().zip(s).map(|(&w, &sc)| w * (1.0 + sc)).collect() };
    let fold_gate = |raw: &[f32], g: &[f32]| -> Vec<f32> { raw.iter().zip(g).map(|(&w, &g)| w * g.tanh()).collect() };
    (
        fold_scale(&nb.raw_an1, &m[0..dim]),
        fold_gate(&nb.raw_an2, &m[dim..2 * dim]),
        fold_scale(&nb.raw_fn1, &m[2 * dim..3 * dim]),
        fold_gate(&nb.raw_fn2, &m[3 * dim..4 * dim]),
    )
}

/// Append one block's forward steps to `s`, reading `x_in` and the shared
/// `cos`/`sin` RoPE tables, and return the fresh output buffer (for chaining).
/// Reusable per-block intermediate buffers, sized for a stage's token count.
/// Allocated ONCE per stage and reused across its blocks (a forward needs no
/// per-block SSA), cutting a 30-layer stack from ~660 buffers to ~24 — wgpu's
/// block allocator otherwise wastes ~1.6× rounding each small buffer up.
pub(crate) struct Scratch {
    n1: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    qn: DeviceBuffer,
    kn: DeviceBuffer,
    qr: DeviceBuffer,
    kr: DeviceBuffer,
    qkv: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
    attn_out: DeviceBuffer,
    n2: DeviceBuffer,
    x1: DeviceBuffer,
    f1: DeviceBuffer,
    g: DeviceBuffer,
    u: DeviceBuffer,
    hsw: DeviceBuffer,
    ff: DeviceBuffer,
    f2: DeviceBuffer,
}

impl Scratch {
    pub fn new(gpu: &Gpu, d: BlockDims, t: u32) -> Scratch {
        Scratch::new_maybe_flash(gpu, d, t, use_flash(gpu, d.n_heads, t))
    }

    /// Allocate block scratch. Under `flash`, the materialised `scores`/`probs`
    /// `[nh·t·t]` buffers are NOT allocated (a single dummy element each), which is
    /// the whole memory win — at high `t` those are gigabytes and hit the 2 GiB
    /// per-binding limit.
    pub fn new_maybe_flash(gpu: &Gpu, d: BlockDims, t: u32, flash: bool) -> Scratch {
        let td = (t * d.dim) as u64;
        let th = (t * d.hidden) as u64;
        let a = |n: u64| gpu.storage(n);
        let attn_mat = if flash { 1 } else { (d.n_heads * t * t) as u64 };
        Scratch {
            n1: a(td), q: a(td), k: a(td), v: a(td), qn: a(td), kn: a(td), qr: a(td), kr: a(td),
            qkv: a((t * 3 * d.dim) as u64),
            scores: a(attn_mat), probs: a(attn_mat),
            ctx: a(td), attn_out: a(td), n2: a(td), x1: a(td), f1: a(td),
            g: a(th), u: a(th), hsw: a(th), ff: a(td), f2: a(td),
        }
    }
}

/// Whether to use flash attention (fused, O(t·hd) memory) instead of the
/// materialised scores→softmax→apply trio, for `nh` heads and `t` tokens on this
/// `gpu`.
///
/// The materialised path uses a tuned register-tiled GEMM, so where it FITS it is
/// at least as fast as the fused flash loops. So flash is a MEMORY escape hatch,
/// not a blanket speedup: auto-enable only once the `[nh·t·t]` scores buffer would
/// approach **this device's** per-binding limit (queried, not hard-coded — so it
/// scales from a 2 GiB-binding card up to a large-binding one). Below that,
/// materialised stays; above it, flash is the only thing that runs at all.
/// `BRAIN_ZIMAGE_FLASH=1|0` forces it (1 to benchmark/verify flash at any size;
/// 0 to prove the OOM).
pub(crate) fn use_flash(gpu: &Gpu, nh: u32, t: u32) -> bool {
    match std::env::var("BRAIN_ZIMAGE_FLASH").ok().as_deref() {
        Some("1") => return true,
        Some("0") => return false,
        _ => {}
    }
    // Switch before the scores buffer reaches the binding ceiling (90% margin
    // leaves room for `probs` + allocator overhead).
    let scores = (nh as u64) * (t as u64) * (t as u64) * 4;
    scores > gpu.max_storage_binding_bytes() * 9 / 10
}

/// Append the self-attention (scores→softmax→apply) for one block, from the packed
/// `qkv` into `ctx`. `flash` fuses it into one tiled online-softmax kernel with
/// O(t·hd) memory; otherwise the materialised trio.
pub(crate) fn push_attention(gpu: &Gpu, s: &mut Vec<Step>, scr: &Scratch, nh: u32, t: u32, hd: u32, dim: u32, flash: bool) {
    if flash {
        s.push(model::block::flash_bidir_step(
            gpu,
            model::block::FlashIds { bidir: K_FLASH, split: Some(K_FLASH_SPLIT) },
            1, // Z-Image's S³-DiT graph is built for a single joint sequence
            nh,
            t,
            hd,
            dim,
            &scr.qkv,
            &scr.ctx,
        ));
    } else {
        s.push(gpu.step(K_SCORES, &[&scr.qkv, &scr.scores], &[1, nh, t, hd, 3 * dim, 0, dim], nh * t * t));
        s.push(gpu.step(K_SOFTMAX, &[&scr.scores, &scr.probs], &[1, nh, t], nh * t));
        s.push(gpu.step(K_APPLY, &[&scr.probs, &scr.qkv, &scr.ctx], &[1, nh, t, hd, 3 * dim, 2 * dim, dim], nh * t * hd));
    }
}

/// Append one block's forward steps to `s`, reading `x_in` + shared `cos`/`sin`,
/// reusing `scr`, and writing the result into `out` (which the caller must keep
/// distinct from `x_in` — a stage double-buffers the residual across blocks).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_block_steps(
    gpu: &Gpu,
    s: &mut Vec<Step>,
    w: &BlockWeights,
    nb: &NormBufs,
    x_in: &DeviceBuffer,
    out: &DeviceBuffer,
    scr: &Scratch,
    cos: &DeviceBuffer,
    sin: &DeviceBuffer,
    d: BlockDims,
    t: u32,
    reg_gemm: bool,
) {
    let (dim, nh, hd, hidden) = (d.dim, d.n_heads, d.head_dim, d.hidden);
    let half = hd / 2;
    // GPU: register-tiled matmul_reg3 (128×128 tile, 256 threads). CPU: naive
    // matmul (native AVX2 fast path; the JIT can't compile reg_gemm's barrier).
    let mm = |x: &DeviceBuffer, wt: &DeviceBuffer, o: &DeviceBuffer, m: u32, kk: u32, n: u32| {
        if reg_gemm {
            gpu.step(K_MATMUL_REG3, &[x, wt, o], &[m, kk, n], m.div_ceil(128) * n.div_ceil(128) * 256)
        } else {
            gpu.step(K_MATMUL, &[x, wt, o], &[m, kk, n], m * n)
        }
    };
    // attention
    s.push(gpu.step(K_RMSNORM, &[x_in, &nb.an1, &scr.n1], &[dim, t, f(EPS)], t));
    s.push(mm(&scr.n1, &w.wq, &scr.q, t, dim, dim));
    s.push(mm(&scr.n1, &w.wk, &scr.k, t, dim, dim));
    s.push(mm(&scr.n1, &w.wv, &scr.v, t, dim, dim));
    s.push(gpu.step(K_RMSNORM, &[&scr.q, &w.nq, &scr.qn], &[hd, t * nh, f(EPS)], t * nh));
    s.push(gpu.step(K_RMSNORM, &[&scr.k, &w.nk, &scr.kn], &[hd, t * nh, f(EPS)], t * nh));
    s.push(gpu.step(K_ROPE, &[&scr.qn, cos, sin, &scr.qr], &[t, nh, hd, half], t * nh * half));
    s.push(gpu.step(K_ROPE, &[&scr.kn, cos, sin, &scr.kr], &[t, nh, hd, half], t * nh * half));
    s.push(gpu.step(K_PACK, &[&scr.qr, &scr.kr, &scr.v, &scr.qkv], &[t, dim], t * 3 * dim));
    push_attention(gpu, s, scr, nh, t, hd, dim, reg_gemm && use_flash(gpu, nh, t));
    s.push(mm(&scr.ctx, &w.wo, &scr.attn_out, t, dim, dim));
    s.push(gpu.step(K_RMSNORM, &[&scr.attn_out, &nb.an2, &scr.n2], &[dim, t, f(EPS)], t));
    s.push(gpu.step(K_ADD2, &[x_in, &scr.n2, &scr.x1], &[t * dim], t * dim));
    // MLP
    s.push(gpu.step(K_RMSNORM, &[&scr.x1, &nb.fn1, &scr.f1], &[dim, t, f(EPS)], t));
    s.push(mm(&scr.f1, &w.w1, &scr.g, t, dim, hidden));
    s.push(mm(&scr.f1, &w.w3, &scr.u, t, dim, hidden));
    s.push(gpu.step(K_SILU_MUL, &[&scr.g, &scr.u, &scr.hsw], &[t * hidden], t * hidden));
    s.push(mm(&scr.hsw, &w.w2, &scr.ff, t, hidden, dim));
    s.push(gpu.step(K_RMSNORM, &[&scr.ff, &nb.fn2, &scr.f2], &[dim, t, f(EPS)], t));
    s.push(gpu.step(K_ADD2, &[&scr.x1, &scr.f2, out], &[t * dim], t * dim));
}

// ---------------- int8 (DP4A) block path ----------------

/// Resident int8-quantized weights for one block: each linear as packed int8
/// (`u32`, 4-per-word) + its per-tensor scale. Norms stay f32 (not matmuls).
pub(crate) struct Int8Weights {
    // (packed int8 weight, per-channel scale buffer [N]).
    wq: (DeviceBuffer, DeviceBuffer),
    wk: (DeviceBuffer, DeviceBuffer),
    wv: (DeviceBuffer, DeviceBuffer),
    wo: (DeviceBuffer, DeviceBuffer),
    w1: (DeviceBuffer, DeviceBuffer),
    w2: (DeviceBuffer, DeviceBuffer),
    w3: (DeviceBuffer, DeviceBuffer),
    nq: DeviceBuffer, // QK-norm weights stay f32 (RMSNorm, not a matmul)
    nk: DeviceBuffer,
}

impl Int8Weights {
    pub fn upload(gpu: &Gpu, t: &dyn checkpoint::TensorSource, prefix: &str, d: BlockDims) -> Int8Weights {
        let (dim, hid) = (d.dim as usize, d.hidden as usize);
        // quantize_weight computes a per-row scale over the WHOLE matrix, so
        // this pulls one tensor at a time (never the whole model) but cannot
        // itself be chunk-bounded further without a row-block quantizer —
        // that refinement belongs to the windowed-execution phase, which
        // needs it anyway for the fp32 stress case; see docs/lessons.md.
        let q = |n: &str, no: usize, k: usize| {
            let name = format!("{prefix}.{n}");
            let mut result: Option<(Vec<u32>, Vec<f32>)> = None;
            let found = t.with_tensor(&name, &mut |data| result = Some(crate::int8::quantize_weight(data, no, k)));
            assert!(found, "zimage: missing {name}");
            let (packed, sw) = result.expect("with_tensor found the tensor, so it must have set result");
            let pb = gpu.storage(packed.len() as u64);
            gpu.write(&pb, &packed);
            let sb = gpu.storage(sw.len() as u64);
            wf(gpu, &sb, &sw);
            (pb, sb)
        };
        let f32buf = |n: &str| upload_named(gpu, t, &format!("{prefix}.{n}"));
        Int8Weights {
            wq: q("attention.to_q.weight", dim, dim),
            wk: q("attention.to_k.weight", dim, dim),
            wv: q("attention.to_v.weight", dim, dim),
            wo: q("attention.to_out.0.weight", dim, dim),
            w1: q("feed_forward.w1.weight", hid, dim),
            w2: q("feed_forward.w2.weight", dim, hid),
            w3: q("feed_forward.w3.weight", hid, dim),
            nq: f32buf("attention.norm_q.weight"),
            nk: f32buf("attention.norm_k.weight"),
        }
    }
}

/// Per-stage int8 activation-quantization scratch (reused across blocks): the
/// max-abs partials, the dynamic scale, and packed-activation buffers for the
/// dim-width (q/k/v/out, w1/w3) and hidden-width (w2) inputs.
pub(crate) struct Int8Scratch {
    sx: DeviceBuffer, // [t] per-token activation scale
    xq_dim: DeviceBuffer,
    xq_hid: DeviceBuffer,
}

impl Int8Scratch {
    pub fn new(gpu: &Gpu, d: BlockDims, t: u32) -> Int8Scratch {
        Int8Scratch {
            sx: gpu.storage(t as u64),
            xq_dim: gpu.storage((t * d.dim / 4) as u64),
            xq_hid: gpu.storage((t * d.hidden / 4) as u64),
        }
    }
}

/// Append one int8 DP4A block. Same math/graph as [`build_block_steps`] but the
/// 7 linears run in int8: each activation is quantized once (shared by the
/// linears reading it — n1→q/k/v, f1→w1/w3), then `matmul_i8_dyn` dequantizes.
/// Norm/RoPE/attention/SwiGLU stay f32. GPU only (DP4A + barriers).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_block_steps_i8(
    gpu: &Gpu,
    s: &mut Vec<Step>,
    w: &Int8Weights,
    nb: &NormBufs,
    x_in: &DeviceBuffer,
    out: &DeviceBuffer,
    scr: &Scratch,
    i8: &Int8Scratch,
    cos: &DeviceBuffer,
    sin: &DeviceBuffer,
    d: BlockDims,
    t: u32,
) {
    let (dim, nh, hd, hidden) = (d.dim, d.n_heads, d.head_dim, d.hidden);
    let half = hd / 2;
    // Quantize activation `x` [t·k] → `xq` with fresh per-token scales i8.sx[t].
    let quant = |s: &mut Vec<Step>, x: &DeviceBuffer, xq: &DeviceBuffer, k: u32| {
        s.push(gpu.step(K_MAX_ABS_ROW, &[x, &i8.sx], &[t, k], t));
        s.push(gpu.step(K_QUANT_PACK, &[x, &i8.sx, xq], &[t, k], t * k / 4));
    };
    // out = dequant(xq @ wᵀ): dynamic activation scale i8.sx × per-channel wp.1.
    let mm8 = |s: &mut Vec<Step>, xq: &DeviceBuffer, wp: &(DeviceBuffer, DeviceBuffer), o: &DeviceBuffer, k: u32, n: u32| {
        s.push(gpu.step(K_MATMUL_I8, &[xq, &wp.0, &i8.sx, &wp.1, o], &[t, k / 4, n], t.div_ceil(128) * n.div_ceil(128) * 256));
    };

    // attention
    s.push(gpu.step(K_RMSNORM, &[x_in, &nb.an1, &scr.n1], &[dim, t, f(EPS)], t));
    quant(s, &scr.n1, &i8.xq_dim, dim);
    mm8(s, &i8.xq_dim, &w.wq, &scr.q, dim, dim);
    mm8(s, &i8.xq_dim, &w.wk, &scr.k, dim, dim);
    mm8(s, &i8.xq_dim, &w.wv, &scr.v, dim, dim);
    s.push(gpu.step(K_RMSNORM, &[&scr.q, &w.nq, &scr.qn], &[hd, t * nh, f(EPS)], t * nh));
    s.push(gpu.step(K_RMSNORM, &[&scr.k, &w.nk, &scr.kn], &[hd, t * nh, f(EPS)], t * nh));
    s.push(gpu.step(K_ROPE, &[&scr.qn, cos, sin, &scr.qr], &[t, nh, hd, half], t * nh * half));
    s.push(gpu.step(K_ROPE, &[&scr.kn, cos, sin, &scr.kr], &[t, nh, hd, half], t * nh * half));
    s.push(gpu.step(K_PACK, &[&scr.qr, &scr.kr, &scr.v, &scr.qkv], &[t, dim], t * 3 * dim));
    push_attention(gpu, s, scr, nh, t, hd, dim, use_flash(gpu, nh, t)); // int8 path is GPU-only
    quant(s, &scr.ctx, &i8.xq_dim, dim);
    mm8(s, &i8.xq_dim, &w.wo, &scr.attn_out, dim, dim);
    s.push(gpu.step(K_RMSNORM, &[&scr.attn_out, &nb.an2, &scr.n2], &[dim, t, f(EPS)], t));
    s.push(gpu.step(K_ADD2, &[x_in, &scr.n2, &scr.x1], &[t * dim], t * dim));
    // MLP
    s.push(gpu.step(K_RMSNORM, &[&scr.x1, &nb.fn1, &scr.f1], &[dim, t, f(EPS)], t));
    quant(s, &scr.f1, &i8.xq_dim, dim);
    mm8(s, &i8.xq_dim, &w.w1, &scr.g, dim, hidden);
    mm8(s, &i8.xq_dim, &w.w3, &scr.u, dim, hidden);
    s.push(gpu.step(K_SILU_MUL, &[&scr.g, &scr.u, &scr.hsw], &[t * hidden], t * hidden));
    quant(s, &scr.hsw, &i8.xq_hid, hidden);
    mm8(s, &i8.xq_hid, &w.w2, &scr.ff, hidden, dim);
    s.push(gpu.step(K_RMSNORM, &[&scr.ff, &nb.fn2, &scr.f2], &[dim, t, f(EPS)], t));
    s.push(gpu.step(K_ADD2, &[&scr.x1, &scr.f2, out], &[t * dim], t * dim));
}

/// A single-block forward graph with weights resident, for a fixed token count.
/// This is the parity reference; the device-resident chain lives in `crate::dev`.
pub struct ZImageBlock {
    gpu: Gpu,
    d: BlockDims,
    t: u32,
    steps: Vec<Step>,
    x_in: DeviceBuffer,
    cos: DeviceBuffer,
    sin: DeviceBuffer,
    nb: NormBufs,
    out: DeviceBuffer,
    _scr: Scratch,
}

impl ZImageBlock {
    pub fn new(tensors: &Tensors, prefix: &str, d: BlockDims, t: u32, modulation: bool, device: Option<&str>) -> ZImageBlock {
        let reg_gemm = device != Some("cpu");
        let gpu = match device {
            Some("cpu") => Gpu::new_cpu(&KERNELS),
            Some("gpu") | Some("wgpu") => Gpu::new_wgpu(&KERNELS),
            _ => Gpu::new(&KERNELS),
        };
        let w = BlockWeights::upload(&gpu, tensors, prefix);
        let nb = NormBufs::new(&gpu, tensors, prefix, d.dim, modulation);
        let half = d.head_dim / 2;
        let x_in = gpu.storage((t * d.dim) as u64);
        let cos = gpu.storage((t * half) as u64);
        let sin = gpu.storage((t * half) as u64);
        let mut steps = Vec::new();
        let scr = Scratch::new(&gpu, d, t);
        let out = gpu.storage((t * d.dim) as u64);
        build_block_steps(&gpu, &mut steps, &w, &nb, &x_in, &out, &scr, &cos, &sin, d, t, reg_gemm);
        ZImageBlock { gpu, d, t, steps, x_in, cos, sin, nb, out, _scr: scr }
    }

    /// Forward one block. `x`: `[t·dim]`; `c`: `[cdim]` adaLN conditioning
    /// (ignored when `modulation=false`); `cos`/`sin`: `[t·head_dim/2]`.
    pub fn forward(&self, x: &[f32], c: &[f32], cos: &[f32], sin: &[f32]) -> Vec<f32> {
        self.nb.upload_folded(&self.gpu, c, self.d.dim as usize, self.d.cdim as usize);
        wf(&self.gpu, &self.x_in, x);
        wf(&self.gpu, &self.cos, cos);
        wf(&self.gpu, &self.sin, sin);
        self.gpu.submit(&[], &self.steps);
        self.gpu.read(&self.out, (self.t * self.d.dim) as usize)
    }
}
