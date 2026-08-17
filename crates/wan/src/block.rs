// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One `WanAttentionBlock` as a recorded brain kernel graph.
//!
//! Mirrors `wan/modules/model.py`'s block: an affine-free LayerNorm carrying the
//! timestep modulation, QK-normalised self-attention with three-axis RoPE, a
//! gated residual, a **separate** cross-attention into the text encoding (SDXL's
//! topology, not FLUX's joint sequence), and a GELU(tanh) FFN behind a second
//! modulated LayerNorm.
//!
//! ## The modulation fold
//!
//! `e0 = time_projection(time_embedding(t))` is `[1, 6, dim]` - a function of
//! the **timestep alone**, with no token axis - so `modulation + e0` is
//! token-independent and
//!
//! ```text
//! LayerNorm_noaffine(x)·(1 + scale) + shift  ==  LayerNorm(x, gamma = 1+scale, beta = shift)
//! ```
//!
//! is an identity, not an approximation. The six vectors therefore become two
//! `(gamma, beta)` pairs plus two gates per block, computed once per forward on
//! the host, and no separate modulation kernel is dispatched. The `modulation +
//! e0` add itself is `dit::adaln::add_table` (PixArt's shared-table trick,
//! generalised there to also cover a future per-token modulation vector) at
//! `rows=1` - the ROW COUNT is what encodes token-independence, not a
//! different function. Wan 2.2's TI2V variant passes a **per-token** `temb`
//! (`temb.ndim == 4` in diffusers), which breaks token-independence and is
//! out of scope here; [`ModBufs::upload`] takes a single `[6·dim]` vector, so
//! that variant cannot be fed to it by accident.
//!
//! ## Attention
//!
//! 81 frames at 480p is 32,760 tokens: a materialised score matrix is 51 GB
//! across 12 heads, which is not slow but unallocatable (the P40 this was
//! written on reports a 2047 MiB per-binding ceiling). Self-attention therefore
//! runs as the fused flash kernel on any device with workgroup reductions, and
//! as query-chunked `[heads, chunk, t]` slabs on the CPU JIT, which cannot run
//! the flash kernel's barriers. `BRAIN_WAN_ATTN=flash|chunked` forces either,
//! which is what lets one parity test prove both paths agree.

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::block::{chunked_bidir_fwd, flash_bidir_fwd, CrossIds, FlashIds, GemmVariants, LayerNormIds};

// Kernel-table indices (order matches KERNELS).
pub(crate) const K_LAYERNORM: usize = 0;
pub(crate) const K_RMSNORM_EPS: usize = 1;
pub(crate) const K_MATMUL: usize = 2;
pub(crate) const K_BIAS_ADD: usize = 3;
pub(crate) const K_ROPE: usize = 4;
pub(crate) const K_PACK_QKV: usize = 5;
pub(crate) const K_XSCORES: usize = 6;
pub(crate) const K_XSOFTMAX: usize = 7;
pub(crate) const K_XAPPLY: usize = 8;
pub(crate) const K_GATE_ROW: usize = 9;
pub(crate) const K_ADD2: usize = 10;
pub(crate) const K_GELU: usize = 11;
pub(crate) const K_MATMUL_REG3: usize = 12;
pub(crate) const K_MATMUL_GEMV: usize = 13;
pub(crate) const K_FLASH: usize = 14;
pub(crate) const K_FLASH_SPLIT: usize = 15;
pub(crate) const K_LAYERNORM_ROWS: usize = 16;
pub(crate) const K_RMSNORM_ROWS: usize = 17;
pub(crate) const K_SOFTMAX_ROWS: usize = 18;
/// The coalesced cross-attention scores pair, appended so every index above is
/// unchanged: a one-off K transpose plus the scores kernel that reads it.
pub(crate) const K_KV_K_HEADT: usize = 19;
pub(crate) const K_XSCORES_KT: usize = 20;

/// Every kernel the DiT dispatches. Nothing here is new: the whole model is
/// existing kernels at Wan's shapes.
pub const KERNELS: [(&str, &str); 21] = [
    ("layernorm", kernels::LAYERNORM),
    ("rmsnorm_eps", kernels::RMSNORM_EPS),
    ("matmul", kernels::MATMUL),
    ("bias_add", kernels::BIAS_ADD),
    ("rope_interleave_table", kernels::ROPE_INTERLEAVE_TABLE),
    ("pack_qkv", kernels::PACK_QKV),
    // The cross trio serves BOTH attentions: self-attention's chunked fallback
    // needs two independent lengths (query chunk vs full key span) exactly as
    // cross-attention does, so `model::block::chunked_bidir_fwd` is written
    // against these and not against the `*_bidir` pair.
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
    ("gate_row", kernels::GATE_ROW),
    ("add2", kernels::ADD2),
    ("gelu", kernels::GELU),
    // Cooperative variants, selected through the shared seams in
    // `model::block` (never by backend name) so a device that cannot run a
    // workgroup barrier keeps the reference kernel above.
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("matmul_gemv", kernels::MATMUL_GEMV),
    ("flash_attn_bidir", kernels::FLASH_ATTN_BIDIR),
    ("flash_attn_bidir_split", kernels::FLASH_ATTN_BIDIR_SPLIT),
    ("layernorm_rows", kernels::LAYERNORM_ROWS),
    ("rmsnorm_rows", kernels::RMSNORM_ROWS),
    ("softmax_rows", kernels::SOFTMAX_ROWS),
    // Cross-attention scores against a key-minor K. `attn_scores_cross` reads
    // the natural `[text_len, dim]` K with the KEY index as the fastest thread
    // index, so every lane of a warp lands on its own cache line; transposing K
    // once a block (`kv_k_headt`, ~3 MB) makes the same 44 GB-a-block sweep
    // coalesce. See `attn_scores_cross_kt.wgsl` for the measured numbers.
    ("kv_k_headt", kernels::KV_K_HEADT),
    ("attn_scores_cross_kt", kernels::ATTN_SCORES_CROSS_KT),
];

/// Shape parameters of one Wan block.
#[derive(Clone, Copy, Debug)]
pub struct BlockDims {
    pub dim: u32,
    pub n_heads: u32,
    pub head_dim: u32,
    pub ffn_dim: u32,
    /// Text tokens the cross-attention reads (always `text_len`, zero-padded).
    pub text_len: u32,
    pub eps: f32,
}

impl BlockDims {
    pub fn new(cfg: &crate::WanConfig) -> BlockDims {
        BlockDims {
            dim: cfg.dim as u32,
            n_heads: cfg.num_heads as u32,
            head_dim: cfg.head_dim() as u32,
            ffn_dim: cfg.ffn_dim as u32,
            text_len: cfg.text_len as u32,
            eps: cfg.eps,
        }
    }
}

/// Which self-attention implementation a graph records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnMode {
    /// One fused online-softmax dispatch per span, O(t·head_dim) memory.
    Flash,
    /// Query-chunked materialised `[heads, chunk, t]` slabs.
    Chunked,
}

/// The self-attention path for this device.
///
/// Flash is the default wherever the device can run it, because it is the only
/// form that fits at Wan's real token counts - the materialised slab is 51 GB
/// at 32,760 tokens and 12 heads, against a 2047 MiB per-binding ceiling. The
/// chunked trio exists for the CPU JIT (no workgroup barriers) and as the A/B
/// partner a parity test can force.
pub fn attn_mode(gpu: &Gpu) -> AttnMode {
    match std::env::var("BRAIN_WAN_ATTN").ok().as_deref() {
        Some("flash") => return AttnMode::Flash,
        Some("chunked") => return AttnMode::Chunked,
        _ => {}
    }
    if gpu.caps().workgroup_reductions {
        AttnMode::Flash
    } else {
        AttnMode::Chunked
    }
}

/// Query rows per materialised score chunk, so `[heads, chunk, keys]` stays
/// inside a fraction of this device's per-binding ceiling. Never below 64 (a
/// smaller chunk buys nothing and multiplies dispatches) and never above the
/// query count.
pub fn score_chunk(gpu: &Gpu, heads: u32, keys: u32, queries: u32) -> u32 {
    // Bytes on both sides: a slab of `chunk` query rows costs
    // `heads · keys · 4` bytes per row, and two of them (scores and probs) must
    // fit alongside everything else, hence half the ceiling.
    let budget = gpu.max_storage_binding_bytes() / 2;
    let per_row = (heads as u64) * (keys as u64) * 4;
    let c = (budget / per_row.max(1)).max(64).min(u32::MAX as u64) as u32;
    c.min(queries).max(1)
}

fn wf(gpu: &Gpu, buf: &DeviceBuffer, data: &[f32]) {
    let bits: Vec<u32> = data.iter().map(|v| v.to_bits()).collect();
    gpu.write(buf, &bits);
}

/// Upload tensor `name` into a fresh device buffer, bounding host scratch to
/// one chunk where the source can report a size (every source this crate
/// builds from can).
pub(crate) fn upload_named(gpu: &Gpu, src: &dyn checkpoint::TensorSource, name: &str) -> DeviceBuffer {
    let numel = src
        .raw_words(name)
        .map(|w| w.len())
        .or_else(|| src.numel(name))
        .unwrap_or_else(|| panic!("wan: missing {name}"));
    paramstore::upload::Uploader::new(gpu).tensor(src, name, numel).unwrap_or_else(|e| panic!("wan: {e}"))
}

/// Read a (small) tensor to the host.
pub(crate) fn read_named(src: &dyn checkpoint::TensorSource, name: &str) -> Vec<f32> {
    let mut out = Vec::new();
    let found = src.with_tensor(name, &mut |data| out = data.to_vec());
    assert!(found, "wan: missing {name}");
    out
}

/// A biased linear's device weight + bias.
pub(crate) struct Linear {
    pub w: DeviceBuffer,
    pub b: DeviceBuffer,
}

impl Linear {
    fn upload(gpu: &Gpu, t: &dyn checkpoint::TensorSource, prefix: &str) -> Linear {
        Linear {
            w: upload_named(gpu, t, &format!("{prefix}.weight")),
            b: upload_named(gpu, t, &format!("{prefix}.bias")),
        }
    }
}

/// Resident (upload-once) weights of one block.
pub(crate) struct BlockWeights {
    pub sq: Linear,
    pub sk: Linear,
    pub sv: Linear,
    pub so: Linear,
    pub snq: DeviceBuffer,
    pub snk: DeviceBuffer,
    pub cq: Linear,
    pub ck: Linear,
    pub cv: Linear,
    pub co: Linear,
    pub cnq: DeviceBuffer,
    pub cnk: DeviceBuffer,
    /// `norm3` upstream: the affine LayerNorm on the cross-attention input.
    /// The only norm in the block with learned affine params.
    pub xnorm_w: DeviceBuffer,
    pub xnorm_b: DeviceBuffer,
    pub ff1: Linear,
    pub ff2: Linear,
}

impl BlockWeights {
    pub fn upload(gpu: &Gpu, t: &dyn checkpoint::TensorSource, prefix: &str) -> BlockWeights {
        let lin = |n: &str| Linear::upload(gpu, t, &format!("{prefix}.{n}"));
        let dev = |n: &str| upload_named(gpu, t, &format!("{prefix}.{n}"));
        BlockWeights {
            sq: lin("self_attn.q"),
            sk: lin("self_attn.k"),
            sv: lin("self_attn.v"),
            so: lin("self_attn.o"),
            snq: dev("self_attn.norm_q.weight"),
            snk: dev("self_attn.norm_k.weight"),
            cq: lin("cross_attn.q"),
            ck: lin("cross_attn.k"),
            cv: lin("cross_attn.v"),
            co: lin("cross_attn.o"),
            cnq: dev("cross_attn.norm_q.weight"),
            cnk: dev("cross_attn.norm_k.weight"),
            xnorm_w: dev("norm3.weight"),
            xnorm_b: dev("norm3.bias"),
            ff1: lin("ffn.0"),
            ff2: lin("ffn.2"),
        }
    }
}

/// The six per-forward modulation vectors of one block, folded into two
/// LayerNorm `(gamma, beta)` pairs and two residual gates.
pub(crate) struct ModBufs {
    pub ln1_g: DeviceBuffer,
    pub ln1_b: DeviceBuffer,
    pub gate1: DeviceBuffer,
    pub ln2_g: DeviceBuffer,
    pub ln2_b: DeviceBuffer,
    pub gate2: DeviceBuffer,
    /// This block's `modulation` parameter, `[6·dim]`, kept on the host: it is
    /// added to `e0` every forward and never read by a kernel on its own.
    pub modulation: Vec<f32>,
}

impl ModBufs {
    pub fn new(gpu: &Gpu, t: &dyn checkpoint::TensorSource, prefix: &str, dim: u32) -> ModBufs {
        let modulation = read_named(t, &format!("{prefix}.modulation"));
        assert_eq!(modulation.len(), 6 * dim as usize, "{prefix}.modulation must be [1, 6, dim]");
        let a = || gpu.storage(dim as u64);
        ModBufs { ln1_g: a(), ln1_b: a(), gate1: a(), ln2_g: a(), ln2_b: a(), gate2: a(), modulation }
    }

    /// Fold `modulation + e0` into the norm affines and gates and upload.
    /// `e0` is `[6·dim]`, chunk order `(shift, scale, gate)` for self-attention
    /// then the same three for the FFN - upstream's `chunk(6, dim=1)`. The add
    /// itself is `dit::adaln::add_table` at `rows=1` - `e0` is one modulation
    /// vector shared by every token in this forward, not a per-token one.
    pub fn upload(&self, gpu: &Gpu, e0: &[f32], dim: usize) {
        assert_eq!(e0.len(), 6 * dim, "e0 must be [6, dim]");
        let m = dit::adaln::add_table(e0, &self.modulation, 1, 6 * dim);
        let part = |i: usize| -> Vec<f32> { m[i * dim..(i + 1) * dim].to_vec() };
        let shift1 = part(0);
        let scale1 = part(1);
        let gate1 = part(2);
        let shift2 = part(3);
        let scale2 = part(4);
        let gate2 = part(5);
        let one_plus = |v: &[f32]| -> Vec<f32> { v.iter().map(|x| 1.0 + x).collect() };
        wf(gpu, &self.ln1_g, &one_plus(&scale1));
        wf(gpu, &self.ln1_b, &shift1);
        wf(gpu, &self.gate1, &gate1);
        wf(gpu, &self.ln2_g, &one_plus(&scale2));
        wf(gpu, &self.ln2_b, &shift2);
        wf(gpu, &self.gate2, &gate2);
    }
}

/// Per-block intermediates, allocated once and reused across the stack (a
/// forward needs no per-block SSA).
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
    ao: DeviceBuffer,
    x1: DeviceBuffer,
    n3: DeviceBuffer,
    xq: DeviceBuffer,
    xqn: DeviceBuffer,
    xk: DeviceBuffer,
    xkn: DeviceBuffer,
    /// `xkn` transposed to key-minor `[dim, text_len]`, rebuilt once a block.
    xkt: DeviceBuffer,
    xv: DeviceBuffer,
    xscores: DeviceBuffer,
    xprobs: DeviceBuffer,
    xctx: DeviceBuffer,
    xo: DeviceBuffer,
    x2: DeviceBuffer,
    n2: DeviceBuffer,
    h1: DeviceBuffer,
    hg: DeviceBuffer,
    ff: DeviceBuffer,
    /// Query rows per materialised chunk (self-attention, then cross).
    self_chunk: u32,
    cross_chunk: u32,
    mode: AttnMode,
}

impl Scratch {
    pub fn new(gpu: &Gpu, d: BlockDims, t: u32) -> Scratch {
        let mode = attn_mode(gpu);
        let (dim, te, nh) = (d.dim, d.text_len, d.n_heads);
        let self_chunk = score_chunk(gpu, nh, t, t);
        let cross_chunk = score_chunk(gpu, nh, te, t);
        let a = |n: u64| gpu.storage(n.max(1));
        let td = (t as u64) * (dim as u64);
        let ted = (te as u64) * (dim as u64);
        // Flash allocates no score slab at all - that is the whole memory win.
        let self_slab = match mode {
            AttnMode::Flash => 1,
            AttnMode::Chunked => (nh as u64) * (self_chunk as u64) * (t as u64),
        };
        let cross_slab = (nh as u64) * (cross_chunk as u64) * (te as u64);
        Scratch {
            n1: a(td),
            q: a(td),
            k: a(td),
            v: a(td),
            qn: a(td),
            kn: a(td),
            qr: a(td),
            kr: a(td),
            qkv: a(3 * td),
            scores: a(self_slab),
            probs: a(self_slab),
            ctx: a(td),
            ao: a(td),
            x1: a(td),
            n3: a(td),
            xq: a(td),
            xqn: a(td),
            xk: a(ted),
            xkn: a(ted),
            xkt: a(ted),
            xv: a(ted),
            xscores: a(cross_slab),
            xprobs: a(cross_slab),
            xctx: a(td),
            xo: a(td),
            x2: a(td),
            n2: a(td),
            h1: a((t as u64) * (d.ffn_dim as u64)),
            hg: a((t as u64) * (d.ffn_dim as u64)),
            ff: a(td),
            self_chunk,
            cross_chunk,
            mode,
        }
    }
}

/// The kernel-selection handles a Wan graph needs, resolved once per device.
#[derive(Clone, Copy)]
pub(crate) struct Sel {
    pub gemm: GemmVariants,
    pub ln: LayerNormIds,
    pub rms_rows: Option<usize>,
    pub softmax_rows: Option<usize>,
    pub cross: CrossIds,
    pub flash: FlashIds,
}

impl Sel {
    pub fn new(gpu: &Gpu) -> Sel {
        let fast = gpu.caps().workgroup_reductions;
        Sel {
            gemm: if fast {
                GemmVariants::Fast { gemv: Some(K_MATMUL_GEMV), tiled: K_MATMUL_REG3 }
            } else {
                GemmVariants::Reference(K_MATMUL)
            },
            // Built literally, not by name: this crate owns its kernel table,
            // so the indices stay greppable and no runtime lookup is needed.
            // Both `*_rows` variants are gated inside `model::block`'s
            // selection rules on `DeviceCaps`, never here.
            ln: LayerNormIds {
                layernorm: K_LAYERNORM,
                layernorm_rows: Some(K_LAYERNORM_ROWS),
                ln_stats: K_LAYERNORM,
                ln_stats_rows: None,
                layernorm_dx: K_LAYERNORM,
                layernorm_dx_rows: None,
            },
            rms_rows: Some(K_RMSNORM_ROWS),
            softmax_rows: fast.then_some(K_SOFTMAX_ROWS),
            cross: CrossIds { scores: K_XSCORES, softmax: K_XSOFTMAX, apply: K_XAPPLY },
            flash: FlashIds { bidir: K_FLASH, split: Some(K_FLASH_SPLIT) },
        }
    }
}

/// `out = x·Wᵀ + b`, through the shared GEMM selection rule.
fn linear(gpu: &Gpu, s: &mut Vec<Step>, sel: &Sel, x: &DeviceBuffer, w: &Linear, out: &DeviceBuffer, m: u32, k: u32, n: u32) {
    let (kind, threads) = model::block::gemm_variant(sel.gemm, m, n);
    s.push(gpu.step(kind, &[x, &w.w, out], &[m, k, n], threads));
    s.push(gpu.step(K_BIAS_ADD, &[out, &w.b], &[m, n], m * n));
}

/// RMSNorm over the FULL model width, not per head.
///
/// This is the trap in Wan's QK normalisation: `WanRMSNorm(dim)` runs before
/// the `view(b, s, n, d)` that splits the heads, and diffusers spells the same
/// thing `RMSNorm(dim_head * heads)` under the config name
/// `"rms_norm_across_heads"`. Normalising per head instead would divide by a
/// different scalar for every head and still produce a plausible-looking video.
fn qk_norm(gpu: &Gpu, s: &mut Vec<Step>, sel: &Sel, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, rows: u32, dim: u32, eps: f32) {
    let (kind, threads) = model::block::rms_variant(gpu, K_RMSNORM_EPS, sel.rms_rows, rows, dim);
    s.push(gpu.step(kind, &[x, w, out], &[dim, rows, f(eps)], threads));
}

/// Cross-attention from `t` query rows into `te` text keys/values, query-chunked
/// so the materialised `[heads, chunk, te]` slab stays bounded.
#[allow(clippy::too_many_arguments)]
fn push_cross(
    gpu: &Gpu,
    s: &mut Vec<Step>,
    sel: &Sel,
    scr: &Scratch,
    d: BlockDims,
    t: u32,
    q: &DeviceBuffer,
    k: &DeviceBuffer,
    v: &DeviceBuffer,
    out: &DeviceBuffer,
) {
    let (dim, nh, hd, te) = (d.dim, d.n_heads, d.head_dim, d.text_len);
    // K transposed to key-minor ONCE for the whole block: it does not vary with
    // the query chunk, and the scores sweep re-reads it `t` times. See
    // `attn_scores_cross_kt.wgsl` - this is what turns that sweep's per-lane
    // memory transactions into coalesced ones.
    s.push(gpu.step(K_KV_K_HEADT, &[k, &scr.xkt], &[te, dim, dim, 0], dim * te));
    let mut q0 = 0u32;
    while q0 < t {
        let qn = scr.cross_chunk.min(t - q0);
        let qoff = (q0 as u64) * (dim as u64);
        let qlen = (qn as u64) * (dim as u64);
        s.push(gpu.step_sliced(
            K_XSCORES_KT,
            &[q, &scr.xkt, &scr.xscores],
            &[(qoff, qlen), (0, 0), (0, 0)],
            &[1, nh, qn, te, hd, dim, 0],
            nh * qn * te,
        ));
        match sel.softmax_rows {
            Some(i) => s.push(gpu.step(i, &[&scr.xscores, &scr.xprobs], &[nh * qn, te], nh * qn * 64)),
            None => s.push(gpu.step(K_XSOFTMAX, &[&scr.xscores, &scr.xprobs], &[1, nh, qn, te], nh * qn)),
        }
        s.push(gpu.step_sliced(
            K_XAPPLY,
            &[&scr.xprobs, v, out],
            &[(0, 0), (0, 0), (qoff, qlen)],
            &[1, nh, qn, te, hd, dim, 0, dim],
            nh * qn * hd,
        ));
        q0 += qn;
    }
}

/// Append one block's forward steps, reading `x_in` plus the shared RoPE tables
/// and the embedded text context, and writing into `out` (which must be
/// distinct from `x_in`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_block_steps(
    gpu: &Gpu,
    s: &mut Vec<Step>,
    sel: &Sel,
    w: &BlockWeights,
    m: &ModBufs,
    x_in: &DeviceBuffer,
    out: &DeviceBuffer,
    scr: &Scratch,
    cos: &DeviceBuffer,
    sin: &DeviceBuffer,
    ctx: &DeviceBuffer,
    d: BlockDims,
    t: u32,
) {
    let (dim, nh, hd, ff, te) = (d.dim, d.n_heads, d.head_dim, d.ffn_dim, d.text_len);
    let half = hd / 2;

    // --- self-attention -------------------------------------------------
    s.push(model::block::layernorm_fwd(gpu, &sel.ln, x_in, &m.ln1_g, &m.ln1_b, &scr.n1, dim, t, d.eps));
    linear(gpu, s, sel, &scr.n1, &w.sq, &scr.q, t, dim, dim);
    linear(gpu, s, sel, &scr.n1, &w.sk, &scr.k, t, dim, dim);
    linear(gpu, s, sel, &scr.n1, &w.sv, &scr.v, t, dim, dim);
    qk_norm(gpu, s, sel, &scr.q, &w.snq, &scr.qn, t, dim, d.eps);
    qk_norm(gpu, s, sel, &scr.k, &w.snk, &scr.kn, t, dim, d.eps);
    s.push(gpu.step(K_ROPE, &[&scr.qn, cos, sin, &scr.qr], &[t, nh, hd, half], t * nh * half));
    s.push(gpu.step(K_ROPE, &[&scr.kn, cos, sin, &scr.kr], &[t, nh, hd, half], t * nh * half));
    s.push(gpu.step(K_PACK_QKV, &[&scr.qr, &scr.kr, &scr.v, &scr.qkv], &[t, dim], t * 3 * dim));
    match scr.mode {
        AttnMode::Flash => {
            flash_bidir_fwd(gpu, sel.flash, nh, hd, dim, &scr.qkv, 3 * dim, 0, dim, 2 * dim, &scr.ctx, &[(0, t)], s)
        }
        // The self-attention span is the WHOLE video latent (33k rows at
        // dim 1536), so a key-minor K would be a 200 MB scratch on the
        // fallback path of a card that already cannot run flash attention.
        // Cross-attention, whose K is the 512-row text memory, is where this
        // model transposes (see `push_cross`).
        AttnMode::Chunked => chunked_bidir_fwd(
            gpu,
            &sel.cross,
            None,
            nh,
            hd,
            dim,
            &scr.qkv,
            3 * dim,
            0,
            dim,
            2 * dim,
            &scr.ctx,
            &scr.scores,
            &scr.probs,
            &[(0, t)],
            scr.self_chunk,
            None,
            s,
        ),
    }
    linear(gpu, s, sel, &scr.ctx, &w.so, &scr.ao, t, dim, dim);
    // x1 = x_in + gate_msa · attn_out
    s.push(gpu.step(K_GATE_ROW, &[x_in, &m.gate1, &scr.ao, &scr.x1], &[t, dim, t], t * dim));

    // --- cross-attention into the text encoding --------------------------
    s.push(model::block::layernorm_fwd(gpu, &sel.ln, &scr.x1, &w.xnorm_w, &w.xnorm_b, &scr.n3, dim, t, d.eps));
    linear(gpu, s, sel, &scr.n3, &w.cq, &scr.xq, t, dim, dim);
    linear(gpu, s, sel, ctx, &w.ck, &scr.xk, te, dim, dim);
    linear(gpu, s, sel, ctx, &w.cv, &scr.xv, te, dim, dim);
    qk_norm(gpu, s, sel, &scr.xq, &w.cnq, &scr.xqn, t, dim, d.eps);
    qk_norm(gpu, s, sel, &scr.xk, &w.cnk, &scr.xkn, te, dim, d.eps);
    push_cross(gpu, s, sel, scr, d, t, &scr.xqn, &scr.xkn, &scr.xv, &scr.xctx);
    linear(gpu, s, sel, &scr.xctx, &w.co, &scr.xo, t, dim, dim);
    // The cross-attention residual is UNGATED - the only one in the block.
    s.push(gpu.step(K_ADD2, &[&scr.x1, &scr.xo, &scr.x2], &[t * dim], t * dim));

    // --- FFN -------------------------------------------------------------
    s.push(model::block::layernorm_fwd(gpu, &sel.ln, &scr.x2, &m.ln2_g, &m.ln2_b, &scr.n2, dim, t, d.eps));
    linear(gpu, s, sel, &scr.n2, &w.ff1, &scr.h1, t, dim, ff);
    s.push(gpu.step(K_GELU, &[&scr.h1, &scr.hg], &[t * ff], t * ff));
    linear(gpu, s, sel, &scr.hg, &w.ff2, &scr.ff, t, ff, dim);
    s.push(gpu.step(K_GATE_ROW, &[&scr.x2, &m.gate2, &scr.ff, out], &[t, dim, t], t * dim));
}

/// One block with its weights resident, for a fixed token count - the parity
/// unit `crate::model` composes and `crate::dev` chains.
pub struct WanBlock {
    gpu: Gpu,
    d: BlockDims,
    t: u32,
    steps: Vec<Step>,
    x_in: DeviceBuffer,
    cos: DeviceBuffer,
    sin: DeviceBuffer,
    ctx: DeviceBuffer,
    m: ModBufs,
    out: DeviceBuffer,
    _w: BlockWeights,
    _scr: Scratch,
}

impl WanBlock {
    pub fn new(cfg: &crate::WanConfig, t: &dyn checkpoint::TensorSource, prefix: &str, tokens: u32, device: Option<&str>) -> WanBlock {
        let gpu = open_device(device);
        Self::on(gpu, cfg, t, prefix, tokens)
    }

    /// [`WanBlock::new`] on an already-open device (so a caller running many
    /// blocks does not open one per block).
    pub fn on(gpu: Gpu, cfg: &crate::WanConfig, t: &dyn checkpoint::TensorSource, prefix: &str, tokens: u32) -> WanBlock {
        let d = BlockDims::new(cfg);
        let sel = Sel::new(&gpu);
        let w = BlockWeights::upload(&gpu, t, prefix);
        let m = ModBufs::new(&gpu, t, prefix, d.dim);
        let scr = Scratch::new(&gpu, d, tokens);
        let x_in = gpu.storage((tokens as u64) * (d.dim as u64));
        let out = gpu.storage((tokens as u64) * (d.dim as u64));
        let cos = gpu.storage((tokens as u64) * (d.head_dim as u64) / 2);
        let sin = gpu.storage((tokens as u64) * (d.head_dim as u64) / 2);
        let ctx = gpu.storage((d.text_len as u64) * (d.dim as u64));
        let mut steps = Vec::new();
        build_block_steps(&gpu, &mut steps, &sel, &w, &m, &x_in, &out, &scr, &cos, &sin, &ctx, d, tokens);
        WanBlock { gpu, d, t: tokens, steps, x_in, cos, sin, ctx, m, out, _w: w, _scr: scr }
    }

    /// Forward one block. `x`: `[t·dim]`, `e0`: `[6·dim]`, `cos`/`sin`:
    /// `[t·head_dim/2]`, `ctx`: the embedded text `[text_len·dim]`.
    pub fn forward(&self, x: &[f32], e0: &[f32], cos: &[f32], sin: &[f32], ctx: &[f32]) -> Vec<f32> {
        self.m.upload(&self.gpu, e0, self.d.dim as usize);
        wf(&self.gpu, &self.x_in, x);
        wf(&self.gpu, &self.cos, cos);
        wf(&self.gpu, &self.sin, sin);
        wf(&self.gpu, &self.ctx, ctx);
        self.gpu.submit(&[], &self.steps);
        self.gpu.read(&self.out, (self.t * self.d.dim) as usize)
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }
}

/// Open a device for the DiT's kernel table. `None` takes brain's default.
pub fn open_device(device: Option<&str>) -> Gpu {
    match device {
        Some("cpu") => Gpu::new_cpu(&KERNELS),
        Some("gpu") | Some("wgpu") => Gpu::new_wgpu(&KERNELS),
        _ => Gpu::new(&KERNELS),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_chunk_is_bounded_and_never_zero() {
        let gpu = Gpu::new_cpu(&KERNELS);
        // A real 480p clip: 32,760 queries over 32,760 keys, 12 heads.
        let c = score_chunk(&gpu, 12, 32_760, 32_760);
        assert!(c >= 64, "chunk {c} below the floor");
        let bytes = 12u64 * c as u64 * 32_760 * 4;
        assert!(bytes <= gpu.max_storage_binding_bytes(), "slab {bytes} exceeds the binding ceiling");
        // Fewer queries than a chunk: never over-report.
        assert_eq!(score_chunk(&gpu, 12, 512, 40), 40);
    }
}
