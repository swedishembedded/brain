// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One `BasicAVTransformerBlock`'s video-only path, as a recorded brain
//! kernel graph - `ltx_core.model.transformer.transformer.
//! BasicAVTransformerBlock.forward`'s `run_vx` branch (the audio branch and
//! the audio<->video cross-attention are entirely absent from this
//! milestone; see `crates/ltxv/src/lib.rs`'s module doc).
//!
//! ## Block order (pinned against source + the golden's real taps, not
//! ## assumed from the class name)
//!
//! 1. **adaLN-single self-attn modulation**: `norm_vx = rms_norm(x, eps) *
//!    (1+scale_msa) + shift_msa` - an UNWEIGHTED RMSNorm (`ada_zero_function`
//!    passes `norm_weights=None`), not the learnable QK-norm below.
//! 2. **Self-attention** (`attn1`): QKV linear (biased) -> QK-RMSNorm
//!    (learnable, full `inner_dim`, NOT per-head - `Attention.q_norm`/
//!    `k_norm` run before the head split) -> RoPE (split/rotate-half, see
//!    `crate::rope`) -> attention -> output projection (biased).
//! 3. **Gated residual + fused re-norm**: `x = x + attn1_out * gate_msa`,
//!    then `x_normed = rms_norm(x, eps)` - again UNWEIGHTED. This is the
//!    ONLY norm between self-attention and text cross-attention (`post_sa_
//!    function` returns both the updated `x` AND this norm in one call) -
//!    there is no separate `norm2` the way a standard PixArt block has one.
//! 4. **Text cross-attention with AdaLN modulation** (`cross_attention_
//!    adaln=true`): the query side is modulated by PER-TOKEN `shift_q`/
//!    `scale_q` (rows 6-7 of the 9-row table) exactly like step 1; the KEY/
//!    VALUE side is modulated by this BLOCK's own static
//!    `prompt_scale_shift_table` (`[2, dim]`, `[shift_kv, scale_kv]`) -
//!    NOT the per-token table, and NOT per-token at all
//!    (`use_prompt_adaln_single=false`, so there is no timestep MLP on this
//!    path; see `crate::config`'s doc). `attn2` gets NO RoPE (`pe=None` at
//!    this call site in the reference - only self-attention rotates).
//!    The cross-attention output is scaled by `gate_q` (row 8) before the
//!    residual add.
//! 5. **MLP sublayer**: `vx_scaled = rms_norm(x, eps) * (1+scale_mlp) +
//!    shift_mlp` (rows 3-4), `GELU(tanh)` FFN (mult=4, `ff_bias=false` for
//!    the real LTX-2.5 config), gated by `gate_mlp` (row 5).
//!
//! Every RMSNorm above that is NOT the learnable QK-norm is the plain
//! `torch.nn.functional.rms_norm(x, weight=None, eps)` form - dispatched
//! here as `rmsnorm_eps` with an all-ones weight buffer, since no kernel in
//! this repo has a "no learnable gain" RMSNorm variant and folding a
//! constant-1 weight into the existing kernel is exact, not an
//! approximation.
//!
//! ## Attention: which kernels, and why
//!
//! Both `attn1` (self, `t_dec == t_enc == T`) and `attn2` (cross, `t_dec ==
//! T`, `t_enc == context_len`) are expressed against the SAME generic
//! scores/softmax/apply trio (`attn_scores_cross_kt` / `softmax_rows` /
//! `attn_apply_cross`, `crates/model/src/block.rs`'s `push_cross`/Wan's
//! non-flash path uses the same trio for cross-attention) rather than a
//! self-attention-specific kernel: the trio's `q_stride`/`kv_stride`/
//! `q_off`/`k_off`/`v_off` are plain parameters, not hardcoded to a
//! fused-QKV layout, so passing separate `[T, dim]` Q/K/V buffers at
//! `stride=dim, off=0` for BOTH self- and cross-attention is exactly what
//! the kernel already supports. That trio is the reference definition of the
//! math and the branch every device without workgroup reductions takes;
//! [`attn_context`] is the one place a capable device swaps it for a FUSED
//! online-softmax dispatch instead - `flash_attn_bidir_reg2` for `attn1`,
//! `flash_attn_cross_reg2` for `attn2` - because the materialized
//! `[heads, T, T]` and `[heads, T, context_len]` slabs are what dominate the
//! pass once T is a real generation's token count, not a test config's.
//! Query-chunking (`model::block::chunked_bidir_fwd`'s reason to exist at
//! Wan's 30k-token scale) has no call site here, because the arm that would
//! need it - the trio - is only ever taken by devices this crate does not run
//! a real generation on, and the fused arm has no slab to chunk.
//!
//! ## Which kernel for RoPE, and why
//!
//! See `crate::rope::apply_rope_step`'s doc - `rope_neox` is refuted
//! (analytic angle, no table input); `rope2d`, dispatched once per head,
//! is the match.


use gpu_core::{f, DeviceBuffer, Gpu, Step};
use vae::blocks::Tensors;

use crate::config::{LtxAudioDitConfig, LtxDitConfig};
use crate::dit::upload_rope_tables;
use crate::rope::{apply_rope_step, ltx_rope_tables};

// ---------------------------------------------------------------------------
// Audio<->video extension: `LtxAvBlock` below adds the
// audio stream (own self-/text-cross-attention, own FFN, own per-block
// tables - structurally identical to the video-only path, just narrower
// dims) and the bidirectional audio<->video cross-attention -
// `BasicAVTransformerBlock.forward`'s `run_ax` branch and the `run_a2v`/
// `run_v2a` block that follows both streams' self-attn+text-CA. See
// [`LtxAvBlock`]'s own doc for the exact op order and adaLN table layout;
// [`self_attn_and_text_ca`]/[`mlp_sublayer`] below are the two sub-sequences
// [`LtxBlock::forward`] and [`LtxAvBlock::forward`] share verbatim (both
// streams run the identical sequence, only dims/weights differ).
// ---------------------------------------------------------------------------

// Kernel-table indices (order matches KERNELS below).
const K_MATMUL: usize = 0;
const K_BIAS_ADD: usize = 1;
const K_RMSNORM_EPS: usize = 2;
const K_GELU: usize = 3;
const K_MUL: usize = 4;
const K_ADD2: usize = 5;
const K_GATE_ROW: usize = 6;
/// `attn_scores_cross` (the non-transposed reference) is deliberately ABSENT:
/// both attention call sites in this crate now go exclusively through
/// [`attn_scores_kt`]'s coalesced `attn_scores_cross_kt` path (see
/// [`K_KV_K_HEADT`]'s doc) - keeping the slow kernel registered here would
/// build a pipeline this crate never dispatches.
const K_ATTN_SOFTMAX: usize = 7;
const K_ATTN_APPLY: usize = 8;
const K_ROPE2D: usize = 9;
/// Gated attention's `sigmoid(gate_logits)` (`crate::config::LtxDitConfig::
/// apply_gated_attention`'s doc, `2*sigmoid` - the `2*` is folded into the
/// per-head "expand to `inner_dim`" matmul's constant matrix in [`attention`]
/// rather than a separate elementwise scale step, see that function's doc).
/// Pre-existing generic elementwise kernel (`crates/kernels/wgsl/
/// sigmoid.wgsl`), not previously in THIS crate's table - `gate_row` was
/// already reused (§F.3: grepped for a broadcast-multiply-with-residual
/// kernel matching `gate_row.wgsl`'s exact shape before this milestone;
/// nothing existing matches the per-HEAD, per-TOKEN, no-residual broadcast
/// gated attention needs, so it composes from `matmul`+`sigmoid`+`mul`
/// instead - see [`attention`]'s doc).
const K_SIGMOID: usize = 10;
/// The quantized-compute tier (int8/int4 - see [`QTier`]): dynamic per-token
/// activation quantization (`max_abs_row` -> `quant_pack`, the same recipe
/// `crates/wan/src/block.rs`'s `QTier` path and `crates/model/src/int8.rs`'s
/// `QuantRows` use) plus the two DP4A-family GEMMs. Appended so every index
/// above is unchanged; unused (and therefore free) on a plain-f32 [`LtxBlock`].
const K_MAX_ABS_ROW: usize = 11;
const K_QUANT_PACK: usize = 12;
const K_MATMUL_I8_DYN: usize = 13;
const K_MATMUL_Q4_DYN: usize = 14;
/// The fp32 fast-tier GEMM pair `linear()` now selects between via
/// `model::block::gemm_variant` (Phase 8 finding: `linear()` used to
/// hardcode `K_MATMUL`, the one-thread-per-output reference kernel,
/// unconditionally - measured at roughly a thousandth of this device's
/// roofline on every fp32 projection in the block. `matmul_reg3`/`matmul_gemv` are
/// the same pipelines `crates/wan/src/block.rs::Sel::new` registers for the
/// identical 10-linear block shape - see that module's `Sel`/`GemmVariants`
/// precedent, which this mirrors). Only the fp32 reference path
/// ([`LtxBlock`]/[`LtxAvBlock`], used by `LtxDit::forward`/gradcheck/
/// `ltxv_bench dit`) needed this - the quantized path ([`LtxBlockQ`]'s
/// `qlinear`) already dispatches `matmul_i8_dyn`/`matmul_q4_dyn`
/// unconditionally, which for DP4A has no naive sibling to fall back to.
const K_MATMUL_REG3: usize = 15;
const K_MATMUL_GEMV: usize = 16;
/// Phase 8 finding #2: `attn_scores_cross` parallelises over the KEY index
/// and reduces over `head_dim`, so at THIS crate's real width (heads=32,
/// head_dim=128, 1024-token self-attention) it measured a fraction of one
/// percent of this device's roofline and was the single largest kernel in a
/// whole-pass profile (more than half of `ltxv_bench dit`'s real-width
/// total), ahead of even the pre-fix naive `matmul`. `attn_scores_cross.wgsl`'s OWN doc already names
/// the fix: transpose K to key-minor once (`kv_k_headt`) so the score sweep's
/// reads coalesce (`attn_scores_cross_kt`) - the same defect and the same fix
/// `crates/wan/src/block.rs` already carries (`K_XSCORES_KT`/`kv_k_headt`).
/// Since [`attention`]/[`attention_q`] share their score/softmax/apply trio
/// with the int8 tier (only the four projections are quantized - see
/// [`attention_q`]'s doc), this fix reaches the real 22B checkpoint's
/// generation path too, not just the fp32 bench.
const K_KV_K_HEADT: usize = 17;
const K_ATTN_SCORES_CROSS_KT: usize = 18;
/// Phase 8 finding #3: with #2 fixed, `attn_softmax_cross` (one THREAD per
/// row, 3 serial `t_enc`-wide passes) became the new #2 kernel by share
/// (about a quarter of a real-width whole pass) at only a couple of percent
/// of the measured roof.
/// `softmax_rows.wgsl`'s own doc names `attn_softmax_cross` as exactly the
/// kernel it exists to replace (one WORKGROUP per row instead of one
/// thread), gated behind `DeviceCaps::workgroup_reductions` the same way
/// `crates/wan/src/block.rs::Sel.softmax_rows` already gates it - mirrored
/// here rather than re-derived.
const K_SOFTMAX_ROWS: usize = 19;
/// Phase 9 finding: with #2 and #3 fixed, `attn_apply_cross` +
/// `attn_scores_cross_kt` + `softmax_rows` were **the overwhelming majority
/// of GPU kernel time** at the real 720p token count (T=3520, heads=32,
/// head_dim=128, measured by `ltxv_bench streamed 8 3520 1024 1` against the
/// real Q8_0 22B checkpoint) - because SELF-attention is O(T²) in the video token count
/// where every other kernel in the block is O(T), so it overtakes the whole
/// block once T passes a few thousand. The materialized `[32, 3520, 3520]`
/// score AND probability slabs are 1.59 GiB *each*.
///
/// [`K_PACK_QKV`] + the flash ladder replace that whole scores/softmax/apply
/// chain with ONE fused online-softmax dispatch for `attn1`; `attn2` gets the
/// same treatment from [`K_FLASH_CROSS_REG2`] (see [`attn_context`] for both
/// gates). Appended so every index above is unchanged.
///
/// All four flash kernels are pre-existing, already gated and already
/// measured on this hardware (`model::block::flash_bidir_variant`'s own
/// per-rung GFLOP/s table) - §F.3: this is a kernel-SELECTION change in one
/// model, not a new kernel. `pack_qkv` is likewise the same pre-existing
/// kernel `crates/flux2/src/model.rs` packs its DiT self-attention operands
/// with.
const K_PACK_QKV: usize = 20;
const K_FLASH: usize = 21;
const K_FLASH_SPLIT: usize = 22;
const K_FLASH_REG: usize = 23;
const K_FLASH_REG2: usize = 24;
/// The ONE kernel this crate contributes rather than reuses (§F.3: the tree
/// was grepped first - `bias_add` broadcasts a row but only IN PLACE,
/// `region_copy` preserves the source layout rather than compacting a strided
/// row out of it, and `add_chan_bcast`/`broadcast_add_hw` are NCHW spatial
/// ops; none can produce a dense `[t, dim]` plane out of a `[t, 9, dim]`
/// table). It replaces the per-block HOST `add_table` + `slice_mod` +
/// nine-vector upload that a real 48-layer 720p forward spent about a third of
/// its wall clock on, plus ~25 GB of PCIe traffic per forward, for a table whose only
/// per-block input is a 147 KB `scale_shift_table`. Appended, so every index
/// above is unchanged.
const K_ADALN_ROW: usize = 25;

/// The fused CROSS-attention kernel, `model::block::flash_cross_step`'s only
/// rung (§F.3: the tree was grepped first and had none - `flash_attn_bidir_*`
/// take one fused qkv slab at a single `tcols`, and `flash_attn_causal_gqa`
/// masks `j > i`; neither can express `nq != nk` over three separate buffers).
/// It replaces this crate's `attn_scores_cross_kt` -> `softmax_rows` ->
/// `attn_apply_cross` trio for text cross-attention, whose real cost is the
/// `[heads, nq, nk]` score slab written, read, rewritten and read again.
/// Appended, so every index above is unchanged.
const K_FLASH_CROSS_REG2: usize = 26;

/// The coalesced workgroup-per-row RMSNorm - the recurring defect class where
/// a fast sibling already existed (`crates/wan/src/block.rs` and
/// `crates/flux2/src/model.rs` both register it) and that this crate had not
/// learned about, so every norm in every block ran the one-thread-per-row
/// reference. Selected by `model::block::rms_variant`, never dispatched
/// directly - see [`rmsnorm`]. Appended, so every index above is unchanged.
const K_RMSNORM_ROWS: usize = 27;

/// Every kernel this block dispatches - all pre-existing except
/// [`K_ADALN_ROW`] and [`K_FLASH_CROSS_REG2`], all at their documented general
/// contract.
pub const KERNELS: [(&str, &str); 28] = [
    ("matmul", kernels::MATMUL),
    ("bias_add", kernels::BIAS_ADD),
    ("rmsnorm_eps", kernels::RMSNORM_EPS),
    ("gelu", kernels::GELU),
    ("mul", kernels::MUL),
    ("add2", kernels::ADD2),
    ("gate_row", kernels::GATE_ROW),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
    ("rope2d", kernels::ROPE2D),
    ("sigmoid", kernels::SIGMOID),
    ("max_abs_row", kernels::MAX_ABS_ROW),
    ("quant_pack", kernels::QUANT_PACK),
    ("matmul_i8_dyn", kernels::MATMUL_I8_DYN),
    ("matmul_q4_dyn", kernels::MATMUL_Q4_DYN),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("matmul_gemv", kernels::MATMUL_GEMV),
    ("kv_k_headt", kernels::KV_K_HEADT),
    ("attn_scores_cross_kt", kernels::ATTN_SCORES_CROSS_KT),
    ("softmax_rows", kernels::SOFTMAX_ROWS),
    // The fused self-attention path (see [`K_PACK_QKV`]). Listing them is
    // harmless on a device that never dispatches them - `attn_context` gates
    // on `DeviceCaps::workgroup_reductions`, which the CPU JIT reports false.
    ("pack_qkv", kernels::PACK_QKV),
    ("flash_attn_bidir", kernels::FLASH_ATTN_BIDIR),
    ("flash_attn_bidir_split", kernels::FLASH_ATTN_BIDIR_SPLIT),
    ("flash_attn_bidir_reg", kernels::FLASH_ATTN_BIDIR_REG),
    ("flash_attn_bidir_reg2", kernels::FLASH_ATTN_BIDIR_REG2),
    ("adaln_row", kernels::ADALN_ROW),
    ("flash_attn_cross_reg2", kernels::FLASH_ATTN_CROSS_REG2),
    ("rmsnorm_rows", kernels::RMSNORM_ROWS),
];

fn tget<'a>(w: &'a Tensors, name: &str) -> &'a [f32] {
    &w.get(name).unwrap_or_else(|| panic!("ltxv dit: missing weight {name}")).1
}

fn upload(gpu: &Gpu, w: &Tensors, name: &str) -> DeviceBuffer {
    let data = tget(w, name);
    let buf = gpu.storage(data.len() as u64);
    gpu.write_f32(&buf, data);
    buf
}

/// `to_gate_logits.{weight,bias}` - `[heads, q_dim]`+`[heads]`, present iff
/// this specific `Attention` module is gated (`crate::config::LtxDitConfig::
/// apply_gated_attention`'s doc: ONE flag per stream, shared by that
/// stream's self-/text-cross-/AV-cross-attention alike - so this is an
/// `Option` on [`AttnWeights`], not a separate struct threaded everywhere).
struct GateWeights {
    w: DeviceBuffer,
    b: DeviceBuffer,
}

/// One `Attention` module's weights (`attn1` or `attn2`).
struct AttnWeights {
    wq: DeviceBuffer,
    bq: DeviceBuffer,
    wk: DeviceBuffer,
    bk: DeviceBuffer,
    wv: DeviceBuffer,
    bv: DeviceBuffer,
    wo: DeviceBuffer,
    bo: DeviceBuffer,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
    gate: Option<GateWeights>,
}

impl AttnWeights {
    /// `gated`: whether THIS module reads `to_gate_logits` off `w` - a fact
    /// the caller decides (this stream's own `apply_gated_attention`), not
    /// derived from `w`'s contents.
    fn upload(gpu: &Gpu, w: &Tensors, prefix: &str, gated: bool) -> AttnWeights {
        AttnWeights {
            wq: upload(gpu, w, &format!("{prefix}.to_q.weight")),
            bq: upload(gpu, w, &format!("{prefix}.to_q.bias")),
            wk: upload(gpu, w, &format!("{prefix}.to_k.weight")),
            bk: upload(gpu, w, &format!("{prefix}.to_k.bias")),
            wv: upload(gpu, w, &format!("{prefix}.to_v.weight")),
            bv: upload(gpu, w, &format!("{prefix}.to_v.bias")),
            wo: upload(gpu, w, &format!("{prefix}.to_out.0.weight")),
            bo: upload(gpu, w, &format!("{prefix}.to_out.0.bias")),
            q_norm: upload(gpu, w, &format!("{prefix}.q_norm.weight")),
            k_norm: upload(gpu, w, &format!("{prefix}.k_norm.weight")),
            gate: gated.then(|| GateWeights { w: upload(gpu, w, &format!("{prefix}.to_gate_logits.weight")), b: upload(gpu, w, &format!("{prefix}.to_gate_logits.bias")) }),
        }
    }
}

/// The FFN's two linears - `net.0.proj` (GELUApprox's inner Linear) and
/// `net.2` (the output Linear). Bias-free at `ff_bias=false` (the main
/// DiT's video FFN); the connector's own FFN and the main DiT's audio FFN
/// are biased regardless (`dit.rs::push_ff`'s doc has the exact per-instance
/// breakdown) - `b1`/`b2` are therefore `Option`, not a second struct.
struct FfWeights {
    w1: DeviceBuffer, // [ff_dim, dim]
    b1: Option<DeviceBuffer>,
    w2: DeviceBuffer, // [dim, ff_dim]
    b2: Option<DeviceBuffer>,
}

/// Whether the FFN at `prefix` is BIASED, read off the tensor map rather than
/// off `cfg.ff_bias`.
///
/// The two disagree, and the map is right. `crate::dit::push_ff`'s own doc
/// records the fact this derives from: the real checkpoint's video `ff` has no
/// bias, but its `audio_ff` and both embeddings connectors' FFNs carry one
/// regardless of the single shared `ff_bias` config key, which is `false` for
/// both streams. The tiny AV golden's own `audio_ff` is bias-free, so a config
/// flag cannot describe both fixtures either. Presence in the map is the one
/// statement that is true of every checkpoint and every fixture at once.
fn ff_has_bias(w: &Tensors, prefix: &str) -> bool {
    w.contains_key(&format!("{prefix}.net.0.proj.bias"))
}

impl FfWeights {
    fn upload(gpu: &Gpu, w: &Tensors, prefix: &str, has_bias: bool) -> FfWeights {
        FfWeights {
            w1: upload(gpu, w, &format!("{prefix}.net.0.proj.weight")),
            b1: has_bias.then(|| upload(gpu, w, &format!("{prefix}.net.0.proj.bias"))),
            w2: upload(gpu, w, &format!("{prefix}.net.2.weight")),
            b2: has_bias.then(|| upload(gpu, w, &format!("{prefix}.net.2.bias"))),
        }
    }
}

/// Resident (upload-once) weights of one block.
struct BlockWeights {
    attn1: AttnWeights,
    attn2: AttnWeights,
    ff: FfWeights,
    /// `[9, dim]` host copy - combined with the shared per-token adaLN
    /// output on every forward (`dit::adaln::add_table` at `rows=T`), so
    /// kept on the host rather than uploaded once.
    scale_shift_table: Vec<f32>,
    /// `[2, dim]` host copy - `[shift_kv, scale_kv]`, static per block (see
    /// this module's doc, step 4).
    prompt_scale_shift_table: Vec<f32>,
}

impl BlockWeights {
    /// `stream`: `""` for video's own weights, `"audio_"` for audio's own
    /// (the SAME struct - both streams' per-block state is structurally
    /// identical, only the tensor name prefix and dims differ; see
    /// `ltx_core...transformer.BasicAVTransformerBlock.__init__`'s
    /// `attn1`/`audio_attn1` etc. naming).
    /// `gated`: this stream's own `apply_gated_attention` - shared
    /// identically by `attn1` and `attn2` (`crate::config::LtxDitConfig::
    /// apply_gated_attention`'s doc).
    fn upload(gpu: &Gpu, w: &Tensors, prefix: &str, stream: &str, dim: usize, gated: bool) -> BlockWeights {
        let sst = tget(w, &format!("{prefix}.{stream}scale_shift_table")).to_vec();
        assert_eq!(sst.len(), 9 * dim, "{prefix}.{stream}scale_shift_table must be [9, dim]");
        let pst = tget(w, &format!("{prefix}.{stream}prompt_scale_shift_table")).to_vec();
        assert_eq!(pst.len(), 2 * dim, "{prefix}.{stream}prompt_scale_shift_table must be [2, dim]");
        BlockWeights {
            attn1: AttnWeights::upload(gpu, w, &format!("{prefix}.{stream}attn1"), gated),
            attn2: AttnWeights::upload(gpu, w, &format!("{prefix}.{stream}attn2"), gated),
            // Biased iff the map carries the tensor - see [`ff_has_bias`].
            // The tiny AV golden's `audio_ff` is bias-free; the real
            // checkpoint's is not, and reading the flag off the map rather
            // than off `cfg.ff_bias` is what lets one code path serve both.
            ff: {
                let ffp = format!("{prefix}.{stream}ff");
                let biased = ff_has_bias(w, &ffp);
                FfWeights::upload(gpu, w, &ffp, biased)
            },
            scale_shift_table: sst,
            prompt_scale_shift_table: pst,
        }
    }
}

fn wf(gpu: &Gpu, buf: &DeviceBuffer, data: &[f32]) {
    gpu.write_f32(buf, data);
}

/// `out = x @ Wᵀ (+ b)`, `x: [m,k]`, `w: [n,k]`, `out: [m,n]` - through
/// `model::block::gemm_variant`, the same shared GEMM selection rule
/// `crates/wan/src/block.rs::linear` uses for its identical 10-linear block
/// shape (see [`K_MATMUL_REG3`]'s doc for why this was hardcoded to the
/// naive reference kernel before). `gpu.caps()` is a cheap accessor (cached
/// device capabilities, not a device round trip), so it is queried per call
/// rather than threading a precomputed `Sel` through every call site -
/// simpler for this crate's much shallower call graph than wan's.
fn linear(gpu: &Gpu, s: &mut Vec<Step>, x: &DeviceBuffer, w: &DeviceBuffer, b: Option<&DeviceBuffer>, out: &DeviceBuffer, m: u32, k: u32, n: u32) {
    let variant = if gpu.caps().workgroup_reductions {
        model::block::GemmVariants::Fast { gemv: Some(K_MATMUL_GEMV), tiled: K_MATMUL_REG3 }
    } else {
        model::block::GemmVariants::Reference(K_MATMUL)
    };
    let (kind, threads) = model::block::gemm_variant(variant, m, n);
    s.push(gpu.step(kind, &[x, w, out], &[m, k, n], threads));
    if let Some(b) = b {
        s.push(gpu.step(K_BIAS_ADD, &[out, b], &[m, n], m * n));
    }
}

/// Attention scores against a key-minor transpose of `k`, replacing the
/// `attn_scores_cross` dispatch - see [`K_KV_K_HEADT`]'s doc for why. `q`/`k`
/// are plain `[rows, inner_dim]` buffers (this crate's attention never uses a
/// fused-QKV stride - see this module's doc), so `kv_stride == inner_dim` and
/// `k_off == 0` at both the transpose and the scores step. Returns the
/// `[heads, nq, nk]` scores buffer; caller still owns softmax + apply.
#[allow(clippy::too_many_arguments)]
fn attn_scores_kt(gpu: &Gpu, s: &mut Vec<Step>, q: &DeviceBuffer, k: &DeviceBuffer, heads: u32, head_dim: u32, inner_dim: u32, nq: u32, nk: u32) -> DeviceBuffer {
    let kt = gpu.storage((inner_dim * nk) as u64);
    s.push(gpu.step(K_KV_K_HEADT, &[k, &kt], &[nk, inner_dim, inner_dim, 0], inner_dim * nk));
    let scores = gpu.storage((heads * nq * nk) as u64);
    s.push(gpu.step(K_ATTN_SCORES_CROSS_KT, &[q, &kt, &scores], &[1, heads, nq, nk, head_dim, inner_dim, 0], heads * nq * nk));
    scores
}

/// Row softmax over `[rows, cols]` scores - `softmax_rows` (one WORKGROUP
/// per row) when the device supports workgroup reductions, else
/// `attn_softmax_cross` (one thread per row, the CPU JIT's native fast
/// path) - see [`K_SOFTMAX_ROWS`]'s doc. Gated via `model::block::
/// softmax_variant` (`Op::Softmax`), not a hand-rolled `caps.
/// workgroup_reductions` check - this crate registers both kernels
/// unconditionally, so `coop` is always `Some`.
fn attn_softmax(gpu: &Gpu, s: &mut Vec<Step>, scores: &DeviceBuffer, probs: &DeviceBuffer, heads: u32, nq: u32, nk: u32) {
    let (sk, st) = model::block::softmax_variant(gpu, K_ATTN_SOFTMAX, Some(K_SOFTMAX_ROWS), heads * nq, nk);
    if sk == K_SOFTMAX_ROWS {
        s.push(gpu.step(sk, &[scores, probs], &[heads * nq, nk], st));
    } else {
        s.push(gpu.step(sk, &[scores, probs], &[1, heads, nq, nk], st));
    }
}

/// This crate's rung set for [`model::block::flash_bidir_variant`] - all four
/// registered (see [`K_PACK_QKV`]), so the selector walks the whole ladder
/// from the device's queried caps and this crate inherits any future rung the
/// shared selector learns about without a change here (§F.7).
const FLASH_IDS: model::block::FlashIds =
    model::block::FlashIds { bidir: K_FLASH, split: Some(K_FLASH_SPLIT), reg: Some(K_FLASH_REG), reg2: Some(K_FLASH_REG2) };

/// Whether THIS attention call takes the fused flash path. Three independent
/// conditions, none of them a backend name:
///
/// * `self_attn` - the flash kernels compute BIDIRECTIONAL self-attention over
///   one span of `t` rows that are simultaneously the queries, the keys and
///   the values. That is `attn1` and nothing else in this crate: text
///   cross-attention (`nq = T`, `nk = context_len`) and the audio<->video
///   cross-attention (`nq = T_v`, `nk = T_a`) attend a DIFFERENT row set, so
///   they are not expressible in this kernel family at all and go to
///   [`flash_cross_attn`]'s own kernel (or, where that is unsupported, the
///   materialized trio). It is a caller-declared flag rather than an inferred
///   `nq == nk`, because `nq == nk` is a coincidence a tiny test config can
///   satisfy for cross-attention and would then silently compute the wrong
///   thing.
/// * `head_dim <= 128` - the family's hard limit (`flash_bidir_fwd` asserts
///   it); checked here so the fallback is a decision, not a panic.
/// * `DeviceCaps::workgroup_reductions` - every kernel in the family is
///   workgroup-cooperative and forward-only, and the Cranelift CPU JIT
///   supports one top-level barrier per kernel where these need two or three.
///   `flash_bidir_fwd`'s own doc names [`attn_scores_kt`]'s materialized trio
///   as the fallback contract, which is exactly what the `else` arm below is.
///   This is the branch `make gradcheck` and every `BRAIN_DEVICE=cpu` test
///   take, so it stays the reference definition of the math.
fn flash_self_attn(gpu: &Gpu, self_attn: bool, head_dim: u32) -> bool {
    self_attn && head_dim <= 128 && gpu.caps().workgroup_reductions
}

/// Whether THIS attention call takes the fused CROSS path
/// (`model::block::flash_cross_step`). The mirror of [`flash_self_attn`], with
/// the same two non-device conditions and the shared selector's own device
/// gate - which is stricter than `workgroup_reductions` alone, because the one
/// registered rung also needs 48 KiB of workgroup memory and a 256-thread
/// workgroup. Anything it refuses keeps the materialized
/// scores/softmax/apply trio, which stays the reference definition of the math
/// (it is the branch `make gradcheck` and every `BRAIN_DEVICE=cpu` test take).
///
/// `!self_attn` rather than `nq != nk`: `nq == nk` is a coincidence a tiny test
/// config can satisfy for cross-attention, and routing a genuine
/// SELF-attention call here would cost the `pack_qkv`-free layout the flash
/// self path is built around for nothing.
fn flash_cross_attn(gpu: &Gpu, self_attn: bool, head_dim: u32) -> bool {
    !self_attn && head_dim <= 128 && model::block::flash_cross_supported(&gpu.caps())
}

/// The attention context `[nq, inner_dim]` for already-projected, already-
/// QK-normed, already-RoPE'd `q`/`k`/`v` - the ONE place this crate decides
/// between a fused flash kernel and the materialized scores/softmax/apply
/// trio - for self- and cross-attention alike, through two independent gates
/// ([`flash_self_attn`], [`flash_cross_attn`]).
///
/// Every arm computes the same softmax attention with the same `1/sqrt(hd)`
/// scale (`flash_attn_bidir*.wgsl`, `flash_attn_cross_reg2.wgsl` and
/// `attn_scores_cross_kt.wgsl` all apply `inverseSqrt(f32(head_dim))`) and
/// writes the same `[nq, heads*head_dim]` row-major layout, so everything
/// downstream - gating, `to_out`, the block's residual - is untouched by the
/// choice. They are NOT bit-identical to each other: an online softmax
/// reassociates both the score sum and the value accumulation, which is why
/// the gate below asserts a tolerance against a host f64 oracle rather than
/// equality.
///
/// The SELF arm needs its operands in ONE slab, `[t, 3*inner_dim]` per-token
/// interleaved as `[q | k | v]`, which is exactly `pack_qkv`'s output and
/// exactly the `stride`/`q_off`/`k_off`/`v_off` convention `flash_bidir_fwd`
/// documents. The CROSS arm needs no pack at all: `flash_attn_cross_reg2`
/// takes three separate buffers with their own strides, which is already the
/// shape [`attention`] produced them in. RoPE is applied BEFORE this call,
/// into the same `q`/`k` buffers the trio would have read, so the rotation is
/// bit-identical between the arms and this function only ever sees
/// already-rotated tensors - LTX's per-head `rope2d` chunking of one shared
/// table is completely unaffected by the packing.
///
/// Cost of the self arm's pack: one read+write of `3·t·inner_dim` floats -
/// against the `[heads, t, t]` score slab AND its probability twin the trio
/// allocates and streams twice each. The cross arm pays nothing at all and
/// drops a `[heads, nq, nk]` pair of the same kind.
#[allow(clippy::too_many_arguments)]
fn attn_context(
    gpu: &Gpu,
    s: &mut Vec<Step>,
    q: &DeviceBuffer,
    k: &DeviceBuffer,
    v: &DeviceBuffer,
    heads: u32,
    head_dim: u32,
    inner_dim: u32,
    nq: u32,
    nk: u32,
    self_attn: bool,
) -> DeviceBuffer {
    let ctx = gpu.storage((nq * inner_dim) as u64);
    if flash_self_attn(gpu, self_attn, head_dim) {
        assert_eq!(nq, nk, "ltxv attn_context: self_attn=true needs nq == nk (got {nq} vs {nk})");
        let qkv = gpu.storage(nq as u64 * 3 * inner_dim as u64);
        s.push(gpu.step(K_PACK_QKV, &[q, k, v, &qkv], &[nq, inner_dim], nq * 3 * inner_dim));
        model::block::flash_bidir_fwd(gpu, FLASH_IDS, heads, head_dim, inner_dim, &qkv, 3 * inner_dim, 0, inner_dim, 2 * inner_dim, &ctx, &[(0, nq)], s);
    } else if flash_cross_attn(gpu, self_attn, head_dim) {
        // No pack: q, k and v are already three plain `[rows, inner_dim]`
        // buffers, which is exactly this kernel's operand shape.
        let lay = model::block::FlashCrossLayout { q_stride: inner_dim, q_off: 0, k_stride: inner_dim, k_off: 0, v_stride: inner_dim, v_off: 0, d_out: inner_dim };
        s.push(model::block::flash_cross_step(gpu, K_FLASH_CROSS_REG2, 1, heads, head_dim, nq, nk, lay, q, k, v, &ctx));
    } else {
        attn_context_materialized(gpu, s, q, k, v, &ctx, heads, head_dim, inner_dim, nq, nk);
    }
    ctx
}

/// The materialized scores/softmax/apply arm of [`attn_context`], as its own
/// function: it is the REFERENCE definition of the math (the branch every
/// device without workgroup reductions takes, hence the one `make gradcheck`
/// and `BRAIN_DEVICE=cpu` exercise), so the A/B gate has to be able to reach
/// it on a device that would otherwise always pick a fused kernel. Calling it
/// rather than re-spelling the three dispatches in the test is what keeps the
/// reference from silently drifting away from the thing it is the reference
/// for.
#[allow(clippy::too_many_arguments)]
fn attn_context_materialized(
    gpu: &Gpu,
    s: &mut Vec<Step>,
    q: &DeviceBuffer,
    k: &DeviceBuffer,
    v: &DeviceBuffer,
    ctx: &DeviceBuffer,
    heads: u32,
    head_dim: u32,
    inner_dim: u32,
    nq: u32,
    nk: u32,
) {
    let scores = attn_scores_kt(gpu, s, q, k, heads, head_dim, inner_dim, nq, nk);
    let probs = gpu.storage((heads * nq * nk) as u64);
    attn_softmax(gpu, s, &scores, &probs, heads, nq, nk);
    s.push(gpu.step(K_ATTN_APPLY, &[&probs, v, ctx], &[1, heads, nq, nk, head_dim, inner_dim, 0, inner_dim], heads * nq * head_dim));
}

/// RMSNorm over the full row width (`dim`, never per-head) - `w` is either a
/// learnable gain (QK-norm) or an all-ones buffer (the "no learnable gain"
/// `ada_zero_function`/`post_sa_function` norms - see this module's doc).
///
/// The kernel is picked by the SHARED rule (`model::block::rms_variant` ->
/// `backend_api::select`'s `Op::RmsNorm`, keyed on `DeviceCaps`, never on a
/// backend name), not by this crate: `rmsnorm_eps` gives thread `t` row `t`
/// and walks the whole row from that one lane, so a warp's loads are `dim`
/// floats apart and each memory sector fetched serves one useful float.
/// `rmsnorm_rows` is the coalesced workgroup-per-row twin, takes the SAME
/// buffers and the SAME Params, and differs only in index and thread count -
/// which is why adopting it is this one call, and why every future rung the
/// shared selector learns about arrives here for free (§F.7).
///
/// Not bit-identical: the cooperative form reduces the row in a different
/// order, so it moves the last bits (usually toward the f64 answer, never for
/// free). That is why it is a gated seam rather than a `gpu_core::upgrade`
/// row.
fn rmsnorm(gpu: &Gpu, s: &mut Vec<Step>, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, dim: u32, rows: u32, eps: f32) {
    let (kind, threads) = model::block::rms_variant(gpu, K_RMSNORM_EPS, Some(K_RMSNORM_ROWS), rows, dim);
    s.push(gpu.step(kind, &[x, w, out], &[dim, rows, f(eps)], threads));
}

fn mul(gpu: &Gpu, s: &mut Vec<Step>, a: &DeviceBuffer, b: &DeviceBuffer, y: &DeviceBuffer, n: u32) {
    s.push(gpu.step(K_MUL, &[a, b, y], &[n], n));
}

fn add2(gpu: &Gpu, s: &mut Vec<Step>, a: &DeviceBuffer, b: &DeviceBuffer, y: &DeviceBuffer, n: u32) {
    s.push(gpu.step(K_ADD2, &[a, b, y], &[n], n));
}

/// `y[r,d] = x[r,d] + g[k,d]*h[r,d]`, `k = r/rows_per_cond`. `rows_per_cond=1`
/// (`NC=rows`, one gate row per token) is what makes this kernel - built for
/// Wan's per-FORWARD gate - serve LTX's per-TOKEN gate unchanged; see
/// `dit::adaln::add_table`'s doc for the same "rows encodes the
/// (in)dependence" generalisation one level up the stack. The AV cross-
/// attention gate is a THIRD point on that same spectrum: ONE gate row
/// shared by every token (driven by the cross modality's scalar sigma, not
/// a per-token value - see [`LtxAvBlock`]'s doc), which is exactly
/// `rows_per_cond = rows` (every token maps to condition-group 0).
#[allow(clippy::too_many_arguments)]
fn gate_row(gpu: &Gpu, s: &mut Vec<Step>, x: &DeviceBuffer, g: &DeviceBuffer, h: &DeviceBuffer, y: &DeviceBuffer, rows: u32, dim: u32, rows_per_cond: u32) {
    s.push(gpu.step(K_GATE_ROW, &[x, g, h, y], &[rows, dim, rows_per_cond], rows * dim));
}

/// `(1+scale)*rms_norm(x,eps) + shift`, per token - PixArt's `ada_zero_
/// function`. `one_plus_scale`/`shift` are `[rows, dim]` (already `1+scale`
/// baked in on upload - see [`LtxBlock::forward`]).
#[allow(clippy::too_many_arguments)]
fn ada_zero(gpu: &Gpu, s: &mut Vec<Step>, ones: &DeviceBuffer, x: &DeviceBuffer, one_plus_scale: &DeviceBuffer, shift: &DeviceBuffer, tmp: &DeviceBuffer, tmp2: &DeviceBuffer, out: &DeviceBuffer, dim: u32, rows: u32, eps: f32) {
    rmsnorm(gpu, s, x, ones, tmp, dim, rows, eps);
    mul(gpu, s, tmp, one_plus_scale, tmp2, rows * dim);
    add2(gpu, s, tmp2, shift, out, rows * dim);
}

/// Gated attention's per-head broadcast-and-double matrix (`E: [inner_dim,
/// heads]`, `E[h*head_dim+d, h] = 2.0` else `0.0`) - `linear(gate_sig, E,
/// None)` (`gate_sig: [nq, heads]`) then computes `gate_bc[t, h*head_dim+d] =
/// 2*gate_sig[t,h]` for every `d`, i.e. the per-head value BROADCAST across
/// `head_dim` AND pre-multiplied by the reference's `2.0` constant
/// (`ops.py`'s `2.0 * torch.sigmoid(gate_logits)`) in the SAME matmul - no
/// separate "scale by 2" kernel/step needed. Reuses the existing `matmul`
/// kernel exactly as every other `linear` call in this file does (§F.3: this
/// composes two already-registered kernels - `sigmoid` for the activation,
/// `matmul` for the broadcast - rather than adding a new
/// per-head-broadcast-multiply WGSL kernel, which does not otherwise exist
/// in this repo).
fn gate_expand_matrix(heads: u32, head_dim: u32) -> Vec<f32> {
    let inner_dim = (heads * head_dim) as usize;
    let mut e = vec![0f32; inner_dim * heads as usize];
    for h in 0..heads {
        for d in 0..head_dim {
            let row = (h * head_dim + d) as usize;
            e[row * heads as usize + h as usize] = 2.0;
        }
    }
    e
}

/// One (self- or cross-)attention call: QKV projections, QK-RMSNorm,
/// optional per-head RoPE (SEPARATELY for Q and K via `q_rope`/`k_rope` -
/// self-/text-cross-attention pass the SAME table for both or `None` for
/// both, but the audio<->video cross-attention rotates Q and K in
/// DIFFERENT position spaces, see [`LtxAvBlock`]'s doc), attention, GATING
/// (`w.gate`, see below), output projection.
///
/// `q_in`/`kv_in` are `[nq, q_dim]`/`[nk, kv_dim]` - equal to `inner_dim`
/// for self-/text-attention, but genuinely different for the audio<->video
/// cross-attention (A2V's `to_q` projects video's WIDER dim down to audio's
/// NARROWER `inner_dim`, and its `to_out` projects back up to `q_dim` - see
/// [`LtxAvBlock`]'s doc). `inner_dim` (`heads*head_dim`) is this call's own
/// QK-norm/attention/RoPE working width - ALWAYS the audio stream's
/// geometry for the AV cross-attention, regardless of which stream is
/// query. Returns the `[nq, q_dim]` output-projected result.
///
/// ## Gating (`w.gate`, `crate::config::LtxDitConfig::apply_gated_attention`'s
/// doc; source: `resources/ltxv/source/packages/ltx-core/src/ltx_core/
/// model/transformer/{attention,ops}.py`)
///
/// `Attention.forward` (`attention.py:575-579`): `if self.to_gate_logits is
/// not None: out = self.gated_attention_function(x, out, self)` - `x` is
/// THIS call's `q_in` (the module's raw, pre-`to_q` input, at `q_dim`
/// width - matches `to_gate_logits: Linear(query_dim, heads)`,
/// `attention.py:514`), `out` is the attention CONTEXT at `inner_dim` width
/// (post `attn_apply`, still BEFORE `to_out`). `PytorchGatedAttention.
/// __call__` (`ops.py:94-106`): `gate_logits = to_gate_logits(x)` ->
/// `(T,heads)`; `gates = 2*sigmoid(gate_logits)`; `out = out.view(T,heads,
/// head_dim) * gates.unsqueeze(-1)` - per-head, broadcast over `head_dim`,
/// THIS call's own `inner_dim` geometry (`heads`/`head_dim` params), not
/// `q_dim`'s. Implemented here as `linear`(gate logits, reusing the
/// existing `matmul`+`bias_add`) -> `sigmoid` (existing kernel, newly
/// registered in this crate's [`KERNELS`], see [`K_SIGMOID`]'s doc) ->
/// `linear` against [`gate_expand_matrix`] (broadcast-and-double, still the
/// existing `matmul`) -> elementwise `mul` (existing kernel) against `ctx`,
/// all BEFORE the `to_out` projection below.
#[allow(clippy::too_many_arguments)]
fn attention(
    gpu: &Gpu,
    s: &mut Vec<Step>,
    w: &AttnWeights,
    q_dim: u32,
    kv_dim: u32,
    inner_dim: u32,
    heads: u32,
    head_dim: u32,
    q_in: &DeviceBuffer,
    kv_in: &DeviceBuffer,
    nq: u32,
    nk: u32,
    q_rope: Option<(&[DeviceBuffer], &[DeviceBuffer])>,
    k_rope: Option<(&[DeviceBuffer], &[DeviceBuffer])>,
    kernel_rope2d: usize,
    eps: f32,
    self_attn: bool,
) -> DeviceBuffer {
    let q_pre = gpu.storage((nq * inner_dim) as u64);
    let k_pre = gpu.storage((nk * inner_dim) as u64);
    let v = gpu.storage((nk * inner_dim) as u64);
    linear(gpu, s, q_in, &w.wq, Some(&w.bq), &q_pre, nq, q_dim, inner_dim);
    linear(gpu, s, kv_in, &w.wk, Some(&w.bk), &k_pre, nk, kv_dim, inner_dim);
    linear(gpu, s, kv_in, &w.wv, Some(&w.bv), &v, nk, kv_dim, inner_dim);

    let q = gpu.storage((nq * inner_dim) as u64);
    let k = gpu.storage((nk * inner_dim) as u64);
    rmsnorm(gpu, s, &q_pre, &w.q_norm, &q, inner_dim, nq, eps);
    rmsnorm(gpu, s, &k_pre, &w.k_norm, &k, inner_dim, nk, eps);

    if let Some((cos_bufs, sin_bufs)) = q_rope {
        for h in 0..heads {
            let off = h * head_dim;
            s.push(apply_rope_step(gpu, kernel_rope2d, &q, &cos_bufs[h as usize], &sin_bufs[h as usize], nq, head_dim, inner_dim, off));
        }
    }
    if let Some((cos_bufs, sin_bufs)) = k_rope {
        for h in 0..heads {
            let off = h * head_dim;
            s.push(apply_rope_step(gpu, kernel_rope2d, &k, &cos_bufs[h as usize], &sin_bufs[h as usize], nk, head_dim, inner_dim, off));
        }
    }

    let ctx = attn_context(gpu, s, &q, &k, &v, heads, head_dim, inner_dim, nq, nk, self_attn);

    let ctx_gated = if let Some(gate) = &w.gate {
        let logits = gpu.storage((nq * heads) as u64);
        linear(gpu, s, q_in, &gate.w, Some(&gate.b), &logits, nq, q_dim, heads);
        let sig = gpu.storage((nq * heads) as u64);
        s.push(gpu.step(K_SIGMOID, &[&logits, &sig], &[nq * heads], nq * heads));
        let expand = gpu.storage((inner_dim * heads) as u64);
        wf(gpu, &expand, &gate_expand_matrix(heads, head_dim));
        let gate_bc = gpu.storage((nq * inner_dim) as u64);
        linear(gpu, s, &sig, &expand, None, &gate_bc, nq, heads, inner_dim);
        let gated = gpu.storage((nq * inner_dim) as u64);
        mul(gpu, s, &ctx, &gate_bc, &gated, nq * inner_dim);
        gated
    } else {
        ctx
    };

    let out = gpu.storage((nq * q_dim) as u64);
    linear(gpu, s, &ctx_gated, &w.wo, Some(&w.bo), &out, nq, inner_dim, q_dim);
    out
}

/// A layer's per-token modulation, sliced from the combined `[T, 9, dim]`
/// table (`dit::adaln::add_table(adaln_table, block.scale_shift_table,
/// rows=T, width=9*dim)`) - row order `shift,scale,gate` for self-attn
/// (0-2), MLP (3-5), then the cross-attention-adaln path (6-8), pinned
/// against `transformer.py`'s `get_ada_values` call sites (see this
/// module's doc).
struct Mod {
    shift_msa: Vec<f32>,
    one_plus_scale_msa: Vec<f32>,
    gate_msa: Vec<f32>,
    shift_mlp: Vec<f32>,
    one_plus_scale_mlp: Vec<f32>,
    gate_mlp: Vec<f32>,
    shift_q: Vec<f32>,
    one_plus_scale_q: Vec<f32>,
    gate_q: Vec<f32>,
}

/// Slice sub-row `i` out of a `[T, k, dim]` row-major combined table -
/// [`dit::adaln::add_table`]'s per-token output, viewed as `k` stacked
/// `[T,dim]` planes. `k=9` is [`slice_mod`]'s self-attn/MLP/text-CA layout;
/// `k=4` is the AV tables' scale/shift layout ([`av_scale_shift`]).
fn slice_row(combined: &[f32], t: usize, dim: usize, k: usize, i: usize) -> Vec<f32> {
    let mut v = vec![0f32; t * dim];
    for ti in 0..t {
        v[ti * dim..ti * dim + dim].copy_from_slice(&combined[(ti * k + i) * dim..(ti * k + i) * dim + dim]);
    }
    v
}

fn slice_mod(combined: &[f32], t: usize, dim: usize) -> Mod {
    // combined: [T, 9, dim] row-major.
    let row = |i: usize| slice_row(combined, t, dim, 9, i);
    let one_plus = |v: Vec<f32>| -> Vec<f32> { v.into_iter().map(|x| 1.0 + x).collect() };
    Mod {
        shift_msa: row(0),
        one_plus_scale_msa: one_plus(row(1)),
        gate_msa: row(2),
        shift_mlp: row(3),
        one_plus_scale_mlp: one_plus(row(4)),
        gate_mlp: row(5),
        shift_q: row(6),
        one_plus_scale_q: one_plus(row(7)),
        gate_q: row(8),
    }
}

/// [`Mod`]'s device-side form: the same nine `[t, dim]` modulation vectors,
/// as device buffers.
///
/// Two ways to fill one, and which one a caller picks is where the biggest
/// single cost of a real forward used to live:
///
/// * [`ModBufs::upload`] takes a host [`Mod`] (i.e. `add_table` + [`slice_mod`]
///   already ran) and writes the nine vectors to the card. That is what every
///   tap-producing/parity path does, and it is the reference definition of the
///   arithmetic.
/// * [`ModBufs::derive`] computes them ON the card from the model-level adaLN
///   table (uploaded ONCE per forward) and this block's own 147 KB
///   `scale_shift_table`, via `adaln_row`. Same numbers, no host combine and no
///   per-block upload - measured as roughly a third of one real 48-layer 720p
///   forward's wall clock, plus ~25 GB of PCIe traffic, removed.
struct ModBufs {
    shift_msa: DeviceBuffer,
    one_plus_scale_msa: DeviceBuffer,
    gate_msa: DeviceBuffer,
    shift_mlp: DeviceBuffer,
    one_plus_scale_mlp: DeviceBuffer,
    gate_mlp: DeviceBuffer,
    shift_q: DeviceBuffer,
    one_plus_scale_q: DeviceBuffer,
    gate_q: DeviceBuffer,
}

/// The nine `[9, dim]` table rows, in the order `slice_mod` reads them, and
/// whether each is a `1 + scale` row. ONE list, read by both
/// [`ModBufs::upload`] (indirectly, through [`slice_mod`]) and
/// [`ModBufs::derive`], so the two cannot disagree about which row is which.
const MOD_ROWS: [(u32, bool); 9] = [(0, false), (1, true), (2, false), (3, false), (4, true), (5, false), (6, false), (7, true), (8, false)];

impl ModBufs {
    /// Upload an already-computed host [`Mod`].
    fn upload(gpu: &Gpu, m: &Mod) -> ModBufs {
        let up = |v: &[f32]| -> DeviceBuffer {
            let b = gpu.storage(v.len() as u64);
            wf(gpu, &b, v);
            b
        };
        ModBufs {
            shift_msa: up(&m.shift_msa),
            one_plus_scale_msa: up(&m.one_plus_scale_msa),
            gate_msa: up(&m.gate_msa),
            shift_mlp: up(&m.shift_mlp),
            one_plus_scale_mlp: up(&m.one_plus_scale_mlp),
            gate_mlp: up(&m.gate_mlp),
            shift_q: up(&m.shift_q),
            one_plus_scale_q: up(&m.one_plus_scale_q),
            gate_q: up(&m.gate_q),
        }
    }

    /// Derive the nine vectors on the device from `tab` (the model-level
    /// adaLN table, one row per DISTINCT token timestep, uploaded once per
    /// forward), `map` (`[t]` u32, which of `tab`'s rows each token uses -
    /// `dit::adaln::RowTable::row_of`) and `tbl` (this block's own `[9, dim]`
    /// `scale_shift_table`, uploaded once per generation). Records nine
    /// `adaln_row` dispatches into `s`; nothing runs until the caller submits.
    fn derive(gpu: &Gpu, s: &mut Vec<Step>, tab: &DeviceBuffer, map: &DeviceBuffer, tbl: &DeviceBuffer, t: u32, dim: u32) -> ModBufs {
        let td = (t * dim) as u64;
        let mut out: Vec<DeviceBuffer> = Vec::with_capacity(9);
        for (row, plus_one) in MOD_ROWS {
            let b = gpu.storage(td);
            s.push(gpu.step(K_ADALN_ROW, &[tab, tbl, map, &b], &[t, dim, 9, row, u32::from(plus_one)], t * dim));
            out.push(b);
        }
        let mut it = out.into_iter();
        ModBufs {
            shift_msa: it.next().unwrap(),
            one_plus_scale_msa: it.next().unwrap(),
            gate_msa: it.next().unwrap(),
            shift_mlp: it.next().unwrap(),
            one_plus_scale_mlp: it.next().unwrap(),
            gate_mlp: it.next().unwrap(),
            shift_q: it.next().unwrap(),
            one_plus_scale_q: it.next().unwrap(),
            gate_q: it.next().unwrap(),
        }
    }
}

/// One stream's AV cross-attention `(1+scale, shift)` pair for one
/// direction - `get_av_ca_ada_values`'s scale/shift half: the model-level
/// per-token raw MLP output (`av_ca_video_scale_shift_adaln_single`/
/// `..._audio_...`, `[T, 4*dim]`) combined with this BLOCK's own `[5,dim]`
/// table's first 4 rows ([`dit::adaln::add_table`] at `rows=T,
/// width=4*dim`), then rows `(row0, row0+1)` picked - `(0,1)` for the A2V
/// direction, `(2,3)` for V2A (`scale_shift_table_a2v_ca_{video,audio}`'s
/// row order - see `crate::config`'s doc: REVERSED vs. the base 9-row
/// table, scale before shift). Returns `(1+scale, shift)`, matching
/// [`ada_zero`]'s `one_plus_scale` input directly.
fn av_scale_shift(mlp_out: &[f32], table5: &[f32], t: usize, dim: usize, row0: usize) -> (Vec<f32>, Vec<f32>) {
    let combined = dit::adaln::add_table(mlp_out, &table5[0..4 * dim], t, 4 * dim);
    let scale: Vec<f32> = slice_row(&combined, t, dim, 4, row0).into_iter().map(|x| 1.0 + x).collect();
    let shift = slice_row(&combined, t, dim, 4, row0 + 1);
    (scale, shift)
}

/// This BLOCK's own row-4 gate, combined with the model-level `[dim]`
/// single-row raw gate MLP output (driven by the CROSS modality's scalar
/// sigma, not this stream's own per-token timestep - see `crate::config`'s
/// `av_ca_timestep_scale_multiplier` doc and [`LtxAvBlock`]'s doc). ONE row,
/// broadcast across every token by `gate_row`'s `rows_per_cond` (see this
/// module's doc).
fn av_gate(gate_mlp_out: &[f32], table5: &[f32], dim: usize) -> Vec<f32> {
    dit::adaln::add_table(gate_mlp_out, &table5[4 * dim..5 * dim], 1, dim)
}

/// The internal taps a parity test bisects with - the golden's
/// `b0_attn1_out`/`b0_attn2_out`/`b0_ff_out` (module HOOK outputs, i.e.
/// `b0_attn2_out` is the RAW cross-attention output before the `*gate_q`
/// multiply the block applies afterward - see `transformer.py`'s
/// `apply_cross_attention_adaln`, where the hook sits on `attn2` itself,
/// inside the function that later scales its return).
pub struct BlockTaps {
    pub attn1_out: Vec<f32>,
    pub attn2_out: Vec<f32>,
    pub ff_out: Vec<f32>,
}

/// One `BasicAVTransformerBlock` (video-only), weights resident, for a fixed
/// token/context length.
pub struct LtxBlock {
    gpu: Gpu,
    cfg: LtxDitConfig,
    w: BlockWeights,
    context_len: u32,
    ones_t: DeviceBuffer,
}

/// One stream's self-attention + gated residual + fused re-norm + text
/// cross-attention with AdaLN modulation - `BasicAVTransformerBlock.
/// forward`'s `run_vx`/`run_ax` bodies up to (not including) the MLP
/// sublayer (this module's doc, steps 1-4). Shared VERBATIM between the
/// video-only path ([`LtxBlock::forward`]) and the AV path ([`LtxAvBlock::
/// forward`]) - both streams run exactly this sequence, only
/// dims/weights/adaLN values differ.
///
/// `x_buf`: `[t,dim]` this stream's current hidden state. `m`: this
/// stream's per-token modulation, already combined with this block's own
/// `scale_shift_table` ([`slice_mod`]'s output). `ctx_buf`/`ctx_len`: this
/// stream's RAW text context (this block's own `prompt_scale_shift_table`
/// modulates it here, fresh each block - see [`BlockWeights`]'s doc).
/// `cos_bufs`/`sin_bufs`: this stream's own self-attention RoPE tables.
///
/// Returns `(x2, attn1_out, ca_raw)`: `x2` is this stream's state after the
/// text-CA residual add (ready for the AV cross-attention step, or - in the
/// video-only path - straight to the MLP); `attn1_out`/`ca_raw` are the two
/// internal taps a parity test bisects with (`ca_raw` is the RAW
/// cross-attention output BEFORE the `*gate_q` multiply, matching
/// `transformer.py`'s hook point - see [`BlockTaps`]'s doc).
#[allow(clippy::too_many_arguments)]
fn self_attn_and_text_ca(
    gpu: &Gpu,
    s: &mut Vec<Step>,
    w: &BlockWeights,
    ones: &DeviceBuffer,
    x_buf: &DeviceBuffer,
    m: &Mod,
    ctx_buf: &DeviceBuffer,
    ctx_len: u32,
    cos_bufs: &[DeviceBuffer],
    sin_bufs: &[DeviceBuffer],
    dim: u32,
    heads: u32,
    head_dim: u32,
    t: u32,
    eps: f32,
) -> (DeviceBuffer, DeviceBuffer, DeviceBuffer) {
    let td = t * dim;
    let up = |v: &[f32]| -> DeviceBuffer {
        let b = gpu.storage(v.len() as u64);
        wf(gpu, &b, v);
        b
    };
    let shift_msa = up(&m.shift_msa);
    let one_plus_scale_msa = up(&m.one_plus_scale_msa);
    let gate_msa = up(&m.gate_msa);
    let shift_q = up(&m.shift_q);
    let one_plus_scale_q = up(&m.one_plus_scale_q);
    let gate_q = up(&m.gate_q);

    // The block's static [shift_kv, scale_kv] (rows 0,1 of
    // prompt_scale_shift_table), broadcast to every context row -
    // per-block, NOT per-token (use_prompt_adaln_single=false).
    let pst = &w.prompt_scale_shift_table;
    let (shift_kv_row, scale_kv_row) = (&pst[0..dim as usize], &pst[dim as usize..2 * dim as usize]);
    let mut shift_kv = vec![0f32; (ctx_len * dim) as usize];
    let mut one_plus_scale_kv = vec![0f32; (ctx_len * dim) as usize];
    for r in 0..ctx_len as usize {
        shift_kv[r * dim as usize..r * dim as usize + dim as usize].copy_from_slice(shift_kv_row);
        for (d, v) in scale_kv_row.iter().enumerate() {
            one_plus_scale_kv[r * dim as usize + d] = 1.0 + v;
        }
    }
    let shift_kv_buf = up(&shift_kv);
    let one_plus_scale_kv_buf = up(&one_plus_scale_kv);

    // --- self-attention ------------------------------------------------
    let tmp1 = gpu.storage(td as u64);
    let tmp2 = gpu.storage(td as u64);
    let norm_x = gpu.storage(td as u64);
    ada_zero(gpu, s, ones, x_buf, &one_plus_scale_msa, &shift_msa, &tmp1, &tmp2, &norm_x, dim, t, eps);
    let attn1_out = attention(gpu, s, &w.attn1, dim, dim, dim, heads, head_dim, &norm_x, &norm_x, t, t, Some((cos_bufs, sin_bufs)), Some((cos_bufs, sin_bufs)), K_ROPE2D, eps, true);
    let x_fma = gpu.storage(td as u64);
    gate_row(gpu, s, x_buf, &gate_msa, &attn1_out, &x_fma, t, dim, 1);

    // Fused re-norm feeding straight into text cross-attention - no
    // separate norm2 (this module's doc, step 3).
    let x_normed = gpu.storage(td as u64);
    rmsnorm(gpu, s, &x_fma, ones, &x_normed, dim, t, eps);

    // --- text cross-attention with adaLN modulation --------------------
    let attn_input_tmp1 = gpu.storage(td as u64);
    let attn_input = gpu.storage(td as u64);
    mul(gpu, s, &x_normed, &one_plus_scale_q, &attn_input_tmp1, td);
    add2(gpu, s, &attn_input_tmp1, &shift_q, &attn_input, td);

    let ctxd = ctx_len * dim;
    let enc_tmp1 = gpu.storage(ctxd as u64);
    let enc_hidden = gpu.storage(ctxd as u64);
    mul(gpu, s, ctx_buf, &one_plus_scale_kv_buf, &enc_tmp1, ctxd);
    add2(gpu, s, &enc_tmp1, &shift_kv_buf, &enc_hidden, ctxd);

    let ca_raw = attention(gpu, s, &w.attn2, dim, dim, dim, heads, head_dim, &attn_input, &enc_hidden, t, ctx_len, None, None, K_ROPE2D, eps, false);
    let ca_gated = gpu.storage(td as u64);
    mul(gpu, s, &ca_raw, &gate_q, &ca_gated, td);
    let x2 = gpu.storage(td as u64);
    add2(gpu, s, &x_fma, &ca_gated, &x2, td);

    (x2, attn1_out, ca_raw)
}

/// The MLP sublayer - `x_scaled = rms_norm(x,eps)*(1+scale_mlp)+shift_mlp`,
/// `GELU(tanh)` FFN (mult=4, `ff_bias=false`), gated residual add (this
/// module's doc, step 5). Shared between the video-only and AV paths, same
/// as [`self_attn_and_text_ca`]. Returns `(x3, ff_out)` - `ff_out` is the
/// raw FFN output before the `*gate_mlp` multiply, the third internal tap.
#[allow(clippy::too_many_arguments)]
fn mlp_sublayer(gpu: &Gpu, s: &mut Vec<Step>, w: &FfWeights, ones: &DeviceBuffer, x2: &DeviceBuffer, shift_mlp: &[f32], one_plus_scale_mlp: &[f32], gate_mlp: &[f32], dim: u32, t: u32, eps: f32) -> (DeviceBuffer, DeviceBuffer) {
    let td = t * dim;
    let up = |v: &[f32]| -> DeviceBuffer {
        let b = gpu.storage(v.len() as u64);
        wf(gpu, &b, v);
        b
    };
    let shift_mlp_buf = up(shift_mlp);
    let one_plus_scale_mlp_buf = up(one_plus_scale_mlp);
    let gate_mlp_buf = up(gate_mlp);

    let mlp_tmp1 = gpu.storage(td as u64);
    let mlp_tmp2 = gpu.storage(td as u64);
    let x_scaled = gpu.storage(td as u64);
    ada_zero(gpu, s, ones, x2, &one_plus_scale_mlp_buf, &shift_mlp_buf, &mlp_tmp1, &mlp_tmp2, &x_scaled, dim, t, eps);
    let ff_dim = dim * 4;
    let h_pre = gpu.storage((t * ff_dim) as u64);
    linear(gpu, s, &x_scaled, &w.w1, w.b1.as_ref(), &h_pre, t, dim, ff_dim);
    let h_act = gpu.storage((t * ff_dim) as u64);
    s.push(gpu.step(K_GELU, &[&h_pre, &h_act], &[t * ff_dim], t * ff_dim));
    let ff_out = gpu.storage(td as u64);
    linear(gpu, s, &h_act, &w.w2, w.b2.as_ref(), &ff_out, t, ff_dim, dim);
    let x3 = gpu.storage(td as u64);
    gate_row(gpu, s, x2, &gate_mlp_buf, &ff_out, &x3, t, dim, 1);
    (x3, ff_out)
}

impl LtxBlock {
    /// `weights`: the checkpoint's tensors, keyed by canonical name (see
    /// `crate::dit::load_tiny_weights`). `prefix`: e.g. `"transformer_blocks.0"`.
    pub fn on(gpu: Gpu, cfg: &LtxDitConfig, weights: &Tensors, prefix: &str, tokens: u32, context_len: u32) -> LtxBlock {
        cfg.assert_supported();
        let dim = cfg.inner_dim as usize;
        let w = BlockWeights::upload(&gpu, weights, prefix, "", dim, cfg.apply_gated_attention);
        let ones_t = gpu.storage(dim as u64);
        wf(&gpu, &ones_t, &vec![1.0f32; dim]);
        let _ = tokens;
        LtxBlock { gpu, cfg: *cfg, w, context_len, ones_t }
    }

    /// One block forward.
    ///
    /// `x`: `[T, dim]` current hidden state. `adaln_table`: `[T, 9*dim]` raw
    /// per-token adaLN-single linear output (shared across every block -
    /// this block's OWN `scale_shift_table` is added in here, per
    /// `get_ada_values`). `context`: `[context_len, dim]` RAW text context
    /// (this block's `prompt_scale_shift_table` modulates it here, fresh
    /// each block). `cos_bufs`/`sin_bufs`: this head's `[T, head_dim/2]`
    /// device-resident RoPE tables (built once, shared by every block -
    /// see `crate::dit`).
    ///
    /// Also returns [`BlockTaps`] (attn1/attn2/ff internal outputs) -
    /// unconditionally, since reading back three more `[T, dim]`-scale
    /// buffers is negligible at this milestone's token counts and it keeps
    /// this the ONE forward path a parity test replays (no separate
    /// "taps-enabled" build, unlike `vae3d.rs`'s env-var toggle, which
    /// exists there because the VAE's per-block taps are large real-weight
    /// buffers this crate's much smaller DiT tensors do not need to guard).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, x: &[f32], adaln_table: &[f32], context: &[f32], cos_bufs: &[DeviceBuffer], sin_bufs: &[DeviceBuffer], t: u32) -> (Vec<f32>, BlockTaps) {
        let gpu = &self.gpu;
        let cfg = &self.cfg;
        let dim = cfg.inner_dim;
        let heads = cfg.num_heads;
        let head_dim = cfg.head_dim();
        let eps = cfg.norm_eps;
        let ctx_len = self.context_len;
        assert_eq!(x.len(), (t * dim) as usize);
        assert_eq!(adaln_table.len(), (t * 9 * dim) as usize);
        assert_eq!(context.len(), (ctx_len * dim) as usize);

        // Combine the shared per-token adaLN table with this block's own
        // static table (dit::adaln::add_table at rows=T), then slice.
        let combined = dit::adaln::add_table(adaln_table, &self.w.scale_shift_table, t as usize, 9 * dim as usize);
        let m = slice_mod(&combined, t as usize, dim as usize);

        let x_buf = gpu.storage((t * dim) as u64);
        wf(gpu, &x_buf, x);
        let ctx_buf = gpu.storage((ctx_len * dim) as u64);
        wf(gpu, &ctx_buf, context);

        let mut s: Vec<Step> = Vec::new();
        let td = t * dim;

        let (x2, attn1_out, ca_raw) = self_attn_and_text_ca(gpu, &mut s, &self.w, &self.ones_t, &x_buf, &m, &ctx_buf, ctx_len, cos_bufs, sin_bufs, dim, heads, head_dim, t, eps);
        let (x3, ff_out) = mlp_sublayer(gpu, &mut s, &self.w.ff, &self.ones_t, &x2, &m.shift_mlp, &m.one_plus_scale_mlp, &m.gate_mlp, dim, t, eps);

        gpu.submit(&[], &s);
        let out = gpu.read(&x3, td as usize);
        let taps = BlockTaps { attn1_out: gpu.read(&attn1_out, td as usize), attn2_out: gpu.read(&ca_raw, td as usize), ff_out: gpu.read(&ff_out, td as usize) };
        (out, taps)
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// Build (but do not submit or read back) one block's step sequence over
    /// an already device-resident `x_buf`/`ctx_buf`, returning `(steps,
    /// output_buffer)` - the same op sequence [`LtxBlock::forward`] submits
    /// and reads back on its own, minus the host round trip.
    ///
    /// Exists for `crates/ltxv/src/bin/ltxv_bench.rs`'s per-kernel-kind
    /// profile: `forward` pays a host readback every block, which is this
    /// DiT's real op sequence at the tiny-config token counts this crate has
    /// parity-gated so far, but chaining that same round trip 48 times at
    /// REAL widths would report upload/readback cost, not kernel cost - the
    /// bench instead threads a device-resident buffer through N calls of
    /// this method and profiles the combined graph in ONE submit, the same
    /// "whole graph" shape `wan::WanDitDev` records directly. Does not
    /// change what [`LtxBlock::forward`] does or how it is dispatched.
    #[allow(clippy::too_many_arguments)]
    pub fn build_steps(&self, x_buf: &DeviceBuffer, adaln_table: &[f32], ctx_buf: &DeviceBuffer, cos_bufs: &[DeviceBuffer], sin_bufs: &[DeviceBuffer], t: u32) -> (Vec<Step>, DeviceBuffer) {
        let gpu = &self.gpu;
        let cfg = &self.cfg;
        let dim = cfg.inner_dim;
        let heads = cfg.num_heads;
        let head_dim = cfg.head_dim();
        let eps = cfg.norm_eps;
        let ctx_len = self.context_len;

        let combined = dit::adaln::add_table(adaln_table, &self.w.scale_shift_table, t as usize, 9 * dim as usize);
        let m = slice_mod(&combined, t as usize, dim as usize);

        let mut s: Vec<Step> = Vec::new();
        let (x2, _attn1_out, _ca_raw) = self_attn_and_text_ca(gpu, &mut s, &self.w, &self.ones_t, x_buf, &m, ctx_buf, ctx_len, cos_bufs, sin_bufs, dim, heads, head_dim, t, eps);
        let (x3, _ff_out) = mlp_sublayer(gpu, &mut s, &self.w.ff, &self.ones_t, &x2, &m.shift_mlp, &m.one_plus_scale_mlp, &m.gate_mlp, dim, t, eps);
        (s, x3)
    }
}

/// One block's audio<->video cross-attention state: the two Attention
/// modules (`audio_to_video_attn`/`video_to_audio_attn`, both at the AUDIO
/// stream's head geometry regardless of which stream is query - see this
/// module's doc and [`LtxAvBlock`]'s doc) plus this block's own `[5,dim]`
/// adaLN tables (`crate::config`'s doc has the row layout: 0-1 A2V
/// scale/shift, 2-3 V2A scale/shift, 4 this table's own gate).
struct AvCrossWeights {
    a2v: AttnWeights,
    v2a: AttnWeights,
    /// `[5, video.dim]`.
    table_video: Vec<f32>,
    /// `[5, audio.dim]`.
    table_audio: Vec<f32>,
}

impl AvCrossWeights {
    /// `gated`: `vcfg.apply_gated_attention` - the SAME single flag gates
    /// BOTH `audio_to_video_attn` (query=video) and `video_to_audio_attn`
    /// (query=audio), since this crate's `LtxAudioDitConfig` has no
    /// independent flag of its own (`crate::config::LtxDitConfig::
    /// apply_gated_attention`'s doc: the reference itself derives both
    /// streams' `TransformerConfig.apply_gated_attention` from ONE
    /// `LTXModel.__init__` argument, so `video_to_audio_attn`'s gate - keyed
    /// off `audio.apply_gated_attention` in `transformer.py` - is always
    /// equal to video's anyway).
    fn upload(gpu: &Gpu, w: &Tensors, prefix: &str, video_dim: usize, audio_dim: usize, gated: bool) -> AvCrossWeights {
        let table_video = tget(w, &format!("{prefix}.scale_shift_table_a2v_ca_video")).to_vec();
        assert_eq!(table_video.len(), 5 * video_dim, "{prefix}.scale_shift_table_a2v_ca_video must be [5, video.dim]");
        let table_audio = tget(w, &format!("{prefix}.scale_shift_table_a2v_ca_audio")).to_vec();
        assert_eq!(table_audio.len(), 5 * audio_dim, "{prefix}.scale_shift_table_a2v_ca_audio must be [5, audio.dim]");
        AvCrossWeights { a2v: AttnWeights::upload(gpu, w, &format!("{prefix}.audio_to_video_attn"), gated), v2a: AttnWeights::upload(gpu, w, &format!("{prefix}.video_to_audio_attn"), gated), table_video, table_audio }
    }
}

/// The AV block's internal taps a parity test bisects with - the AV
/// counterpart of [`BlockTaps`], one set per stream plus the two raw
/// cross-attention outputs (`a2v_out`/`v2a_out`, BEFORE their `*gate`
/// multiply, same convention as `BlockTaps::attn2_out`).
#[derive(Default)]
pub struct AvBlockTaps {
    pub v_attn1_out: Vec<f32>,
    pub v_attn2_out: Vec<f32>,
    pub v_ff_out: Vec<f32>,
    pub a_attn1_out: Vec<f32>,
    pub a_attn2_out: Vec<f32>,
    pub a_ff_out: Vec<f32>,
    pub a2v_out: Vec<f32>,
    pub v2a_out: Vec<f32>,
}

/// One `BasicAVTransformerBlock` (audio+video), weights resident, for fixed
/// token/context lengths on each stream.
///
/// ## Block order (pinned against `transformer.py`'s `BasicAVTransformerBlock.
/// forward`, not assumed from the class name)
///
/// 1. Video: self-attention + gated residual + fused re-norm + text
///    cross-attention with AdaLN modulation - EXACTLY [`LtxBlock::forward`]'s
///    sequence, factored out as [`self_attn_and_text_ca`].
/// 2. Audio: the SAME sequence, own weights/dims - `run_ax`, not entangled
///    with video's own self-attn/text-CA at all.
/// 3. Audio<->video cross-attention, BOTH directions reading a snapshot of
///    each stream's state taken AFTER step 1/2 (`vx_pre_av`/`ax_pre_av` in
///    the reference - so A2V and V2A both see the SAME pre-AV state,
///    regardless of which direction the reference computes first):
///    - **A2V** (`audio_to_video_attn`, video is query): `a2v_vx_scaled =
///      ada_zero(vx_pre_av, ...)` using the VIDEO table's rows 0-1
///      (scale,shift - REVERSED order vs. the base 9-row table, see
///      `crate::config`'s doc) at video's OWN per-token AV scale/shift
///      timestep; `a2v_ax_scaled = ada_zero(ax_pre_av, ...)` using the
///      AUDIO table's rows 0-1 at audio's OWN per-token AV scale/shift
///      timestep. `vx = vx_pre_av + attn(a2v_vx_scaled,
///      context=a2v_ax_scaled, pe=video_cross_pe, k_pe=audio_cross_pe) *
///      gate_a2v`, where `gate_a2v` is the VIDEO table's row 4 at a
///      timestep built from the CROSS (audio) modality's SCALAR sigma, not
///      video's own per-token timestep - and is a SINGLE row broadcast
///      across every video token (`gate_row`'s `rows_per_cond` - see this
///      module's doc), not per-token.
///    - **V2A** (`video_to_audio_attn`, audio is query): the mirror image -
///      `v2a_ax_scaled`/`v2a_vx_scaled` from rows 2-3 of the audio/video
///      tables respectively (still each operand's OWN per-token AV
///      scale/shift timestep), `gate_v2a` from the AUDIO table's row 4 at a
///      timestep built from VIDEO's scalar sigma.
///
///    Both directions run at the AUDIO stream's head geometry
///    (`heads=audio.heads, dim_head=audio.d_head`) - see [`attention`]'s
///    doc - and both use the SHARED cross-modal (time-only) RoPE space:
///    `pe` = the query stream's OWN cross positional embeddings, `k_pe` =
///    the OTHER stream's (`crate::rope`'s doc has the exact construction).
/// 4. Video MLP (on the state AFTER the A2V update), then audio MLP (on the
///    state after the V2A update) - `mlp_sublayer`, same as the video-only
///    path.
///
/// Perturbation masking and the `enabled`/`cross_attn_skip_all` shortcuts
/// (`transformer.py`'s `BlockPerturbationsProcessor`) are out of scope, same
/// simplification the video-only path already makes for `self_attention_
/// mask`/`context_mask`: the reference's own "no perturbation" config
/// resolves every mask to an effective identity (all-ones/False), so
/// omitting the machinery entirely reproduces that one configuration
/// exactly, not an approximation of it.
pub struct LtxAvBlock {
    gpu: Gpu,
    vcfg: LtxDitConfig,
    acfg: LtxAudioDitConfig,
    vw: BlockWeights,
    aw: BlockWeights,
    avw: AvCrossWeights,
    v_ctx_len: u32,
    a_ctx_len: u32,
    ones_v: DeviceBuffer,
    ones_a: DeviceBuffer,
}

impl LtxAvBlock {
    /// `weights`/`prefix`: same convention as [`LtxBlock::on`].
    pub fn on(gpu: Gpu, vcfg: &LtxDitConfig, acfg: &LtxAudioDitConfig, weights: &Tensors, prefix: &str, v_ctx_len: u32, a_ctx_len: u32) -> LtxAvBlock {
        vcfg.assert_supported();
        let vdim = vcfg.inner_dim as usize;
        let adim = acfg.inner_dim as usize;
        let vw = BlockWeights::upload(&gpu, weights, prefix, "", vdim, vcfg.apply_gated_attention);
        let aw = BlockWeights::upload(&gpu, weights, prefix, "audio_", adim, vcfg.apply_gated_attention);
        let avw = AvCrossWeights::upload(&gpu, weights, prefix, vdim, adim, vcfg.apply_gated_attention);
        let ones_v = gpu.storage(vdim as u64);
        wf(&gpu, &ones_v, &vec![1.0f32; vdim]);
        let ones_a = gpu.storage(adim as u64);
        wf(&gpu, &ones_a, &vec![1.0f32; adim]);
        LtxAvBlock { gpu, vcfg: *vcfg, acfg: *acfg, vw, aw, avw, v_ctx_len, a_ctx_len, ones_v, ones_a }
    }

    /// One AV block forward - see this struct's doc for the exact op order.
    ///
    /// `vx`/`ax`: each stream's current hidden state. `v_adaln_table`/
    /// `a_adaln_table`: each stream's own `[T,9*dim]` raw per-token
    /// adaLN-single table (shared across every block, model-level - see
    /// `crate::dit`). `v_context`/`a_context`: each stream's raw text
    /// context. `{v,a}_{cos,sin}`: each stream's own self-attention RoPE
    /// tables; `{v,a}_cross_{cos,sin}`: each stream's own cross-modal RoPE
    /// table (SHARED width, built at `audio_cross_attention_dim` - see
    /// `crate::rope`'s doc). `av_video_ss_table`/`av_audio_ss_table`:
    /// `[T,4*dim]` raw per-token AV scale/shift MLP output, one per stream.
    /// `av_a2v_gate_table`/`av_v2a_gate_table`: `[dim]` SINGLE-row raw AV
    /// gate MLP output (driven by the cross modality's scalar sigma - see
    /// `crate::dit`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        vx: &[f32],
        ax: &[f32],
        v_adaln_table: &[f32],
        a_adaln_table: &[f32],
        v_context: &[f32],
        a_context: &[f32],
        v_cos: &[DeviceBuffer],
        v_sin: &[DeviceBuffer],
        a_cos: &[DeviceBuffer],
        a_sin: &[DeviceBuffer],
        v_cross_cos: &[DeviceBuffer],
        v_cross_sin: &[DeviceBuffer],
        a_cross_cos: &[DeviceBuffer],
        a_cross_sin: &[DeviceBuffer],
        av_video_ss_table: &[f32],
        av_audio_ss_table: &[f32],
        av_a2v_gate_table: &[f32],
        av_v2a_gate_table: &[f32],
        tv: u32,
        ta: u32,
        taps: bool,
    ) -> (Vec<f32>, Vec<f32>, AvBlockTaps) {
        let gpu = &self.gpu;
        let vdim = self.vcfg.inner_dim;
        let vheads = self.vcfg.num_heads;
        let vhd = self.vcfg.head_dim();
        // A single model-level norm_eps is shared by every RMSNorm/LayerNorm
        // in BOTH streams (ltx_core...transformer.BasicAVTransformerBlock's
        // one `norm_eps` constructor arg) - not a per-stream value.
        let eps = self.vcfg.norm_eps;
        let adim = self.acfg.inner_dim;
        let aheads = self.acfg.num_heads;
        let ahd = self.acfg.head_dim();

        assert_eq!(vx.len(), (tv * vdim) as usize);
        assert_eq!(ax.len(), (ta * adim) as usize);
        assert_eq!(v_adaln_table.len(), (tv * 9 * vdim) as usize);
        assert_eq!(a_adaln_table.len(), (ta * 9 * adim) as usize);

        let v_combined = dit::adaln::add_table(v_adaln_table, &self.vw.scale_shift_table, tv as usize, 9 * vdim as usize);
        let vm = slice_mod(&v_combined, tv as usize, vdim as usize);
        let a_combined = dit::adaln::add_table(a_adaln_table, &self.aw.scale_shift_table, ta as usize, 9 * adim as usize);
        let am = slice_mod(&a_combined, ta as usize, adim as usize);

        let vx_buf = gpu.storage((tv * vdim) as u64);
        wf(gpu, &vx_buf, vx);
        let ax_buf = gpu.storage((ta * adim) as u64);
        wf(gpu, &ax_buf, ax);
        let v_ctx_buf = gpu.storage((self.v_ctx_len * vdim) as u64);
        wf(gpu, &v_ctx_buf, v_context);
        let a_ctx_buf = gpu.storage((self.a_ctx_len * adim) as u64);
        wf(gpu, &a_ctx_buf, a_context);

        let mut s: Vec<Step> = Vec::new();

        // ---- 1-2: video, then audio - self-attn + text-CA, each stream run
        // to completion before the AV step touches either (this struct's
        // doc, steps 1-2).
        let (vx1, v_attn1_out, v_ca_raw) = self_attn_and_text_ca(gpu, &mut s, &self.vw, &self.ones_v, &vx_buf, &vm, &v_ctx_buf, self.v_ctx_len, v_cos, v_sin, vdim, vheads, vhd, tv, eps);
        let (ax1, a_attn1_out, a_ca_raw) = self_attn_and_text_ca(gpu, &mut s, &self.aw, &self.ones_a, &ax_buf, &am, &a_ctx_buf, self.a_ctx_len, a_cos, a_sin, adim, aheads, ahd, ta, eps);

        // ---- 3: audio<->video cross-attention - vx1/ax1 are the pre-AV
        // snapshot BOTH directions read (this struct's doc, step 3).
        let (scale_a2v_v, shift_a2v_v) = av_scale_shift(av_video_ss_table, &self.avw.table_video, tv as usize, vdim as usize, 0);
        let (scale_a2v_a, shift_a2v_a) = av_scale_shift(av_audio_ss_table, &self.avw.table_audio, ta as usize, adim as usize, 0);
        let gate_a2v = av_gate(av_a2v_gate_table, &self.avw.table_video, vdim as usize);

        let (scale_v2a_a, shift_v2a_a) = av_scale_shift(av_audio_ss_table, &self.avw.table_audio, ta as usize, adim as usize, 2);
        let (scale_v2a_v, shift_v2a_v) = av_scale_shift(av_video_ss_table, &self.avw.table_video, tv as usize, vdim as usize, 2);
        let gate_v2a = av_gate(av_v2a_gate_table, &self.avw.table_audio, adim as usize);

        let up = |v: &[f32]| -> DeviceBuffer {
            let b = gpu.storage(v.len() as u64);
            wf(gpu, &b, v);
            b
        };
        let scale_a2v_v_buf = up(&scale_a2v_v);
        let shift_a2v_v_buf = up(&shift_a2v_v);
        let scale_a2v_a_buf = up(&scale_a2v_a);
        let shift_a2v_a_buf = up(&shift_a2v_a);
        let scale_v2a_a_buf = up(&scale_v2a_a);
        let shift_v2a_a_buf = up(&shift_v2a_a);
        let scale_v2a_v_buf = up(&scale_v2a_v);
        let shift_v2a_v_buf = up(&shift_v2a_v);
        let gate_a2v_buf = up(&gate_a2v);
        let gate_v2a_buf = up(&gate_v2a);

        let vtd = tv * vdim;
        let atd = ta * adim;

        // A2V: video is query.
        let v_tmp1 = gpu.storage(vtd as u64);
        let v_tmp2 = gpu.storage(vtd as u64);
        let a2v_vx_scaled = gpu.storage(vtd as u64);
        ada_zero(gpu, &mut s, &self.ones_v, &vx1, &scale_a2v_v_buf, &shift_a2v_v_buf, &v_tmp1, &v_tmp2, &a2v_vx_scaled, vdim, tv, eps);
        let a_tmp1 = gpu.storage(atd as u64);
        let a_tmp2 = gpu.storage(atd as u64);
        let a2v_ax_scaled = gpu.storage(atd as u64);
        ada_zero(gpu, &mut s, &self.ones_a, &ax1, &scale_a2v_a_buf, &shift_a2v_a_buf, &a_tmp1, &a_tmp2, &a2v_ax_scaled, adim, ta, eps);

        let a2v_out = attention(gpu, &mut s, &self.avw.a2v, vdim, adim, adim, aheads, ahd, &a2v_vx_scaled, &a2v_ax_scaled, tv, ta, Some((v_cross_cos, v_cross_sin)), Some((a_cross_cos, a_cross_sin)), K_ROPE2D, eps, false);
        let vx2 = gpu.storage(vtd as u64);
        gate_row(gpu, &mut s, &vx1, &gate_a2v_buf, &a2v_out, &vx2, tv, vdim, tv);

        // V2A: audio is query.
        let v_tmp3 = gpu.storage(vtd as u64);
        let v_tmp4 = gpu.storage(vtd as u64);
        let v2a_vx_scaled = gpu.storage(vtd as u64);
        ada_zero(gpu, &mut s, &self.ones_v, &vx1, &scale_v2a_v_buf, &shift_v2a_v_buf, &v_tmp3, &v_tmp4, &v2a_vx_scaled, vdim, tv, eps);
        let a_tmp3 = gpu.storage(atd as u64);
        let a_tmp4 = gpu.storage(atd as u64);
        let v2a_ax_scaled = gpu.storage(atd as u64);
        ada_zero(gpu, &mut s, &self.ones_a, &ax1, &scale_v2a_a_buf, &shift_v2a_a_buf, &a_tmp3, &a_tmp4, &v2a_ax_scaled, adim, ta, eps);

        let v2a_out = attention(gpu, &mut s, &self.avw.v2a, adim, vdim, adim, aheads, ahd, &v2a_ax_scaled, &v2a_vx_scaled, ta, tv, Some((a_cross_cos, a_cross_sin)), Some((v_cross_cos, v_cross_sin)), K_ROPE2D, eps, false);
        let ax2 = gpu.storage(atd as u64);
        gate_row(gpu, &mut s, &ax1, &gate_v2a_buf, &v2a_out, &ax2, ta, adim, ta);

        // ---- 4: MLPs, video then audio (this struct's doc, step 4) --------
        let (vx3, v_ff_out) = mlp_sublayer(gpu, &mut s, &self.vw.ff, &self.ones_v, &vx2, &vm.shift_mlp, &vm.one_plus_scale_mlp, &vm.gate_mlp, vdim, tv, eps);
        let (ax3, a_ff_out) = mlp_sublayer(gpu, &mut s, &self.aw.ff, &self.ones_a, &ax2, &am.shift_mlp, &am.one_plus_scale_mlp, &am.gate_mlp, adim, ta, eps);

        gpu.submit(&[], &s);
        let vx_out = gpu.read(&vx3, vtd as usize);
        let ax_out = gpu.read(&ax3, atd as usize);
        // Eight `[t, dim]`-scale readbacks a production forward discards -
        // see [`crate::dit::LtxAvDit::forward_blocks_av`]'s `taps` doc for
        // what they cost at a real token count.
        let tp = if taps {
            AvBlockTaps {
                v_attn1_out: gpu.read(&v_attn1_out, vtd as usize),
                v_attn2_out: gpu.read(&v_ca_raw, vtd as usize),
                v_ff_out: gpu.read(&v_ff_out, vtd as usize),
                a_attn1_out: gpu.read(&a_attn1_out, atd as usize),
                a_attn2_out: gpu.read(&a_ca_raw, atd as usize),
                a_ff_out: gpu.read(&a_ff_out, atd as usize),
                a2v_out: gpu.read(&a2v_out, vtd as usize),
                v2a_out: gpu.read(&v2a_out, atd as usize),
            }
        } else {
            AvBlockTaps::default()
        };
        (vx_out, ax_out, tp)
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }
}

// ---------------------------------------------------------------------------
// Quantized-compute (int8/int4) video-only block path - the compute-time
// sibling of `crate::int8`'s storage-only round trip, and the path a real
// generation actually takes (`crate::dit::forward_q_streamed`). Every linear
// below issues a packed-int8 GEMM; nothing here is dequantized to f32 first.
//
// A CAPACITY tier first, not a speed one - a precision change is not a speed
// change; quantization here buys the real 22B checkpoint's residency on a
// 24 GiB P40, not throughput. Every
// int8-eligible linear weight (`crate::int8::is_never_quantized`'s excluded
// set stays fp32, unchanged) is packed per-output-row (`model::int8::
// quantize_weight`, 4 int8 lanes/u32) with one fp32 scale per 32-element
// group of K (`model::int8::GROUP`); activations are
// quantized dynamically per row each forward (`max_abs_row` -> `quant_pack`,
// the SAME recipe `crates/wan/src/block.rs`'s `QTier` path uses) and the
// DP4A GEMM (`matmul_i8_dyn`) dequantizes with `sx*sw` on the way out -
// brain's established "quantized storage, fp32 arithmetic result" contract.
// int4 is W4A8 (weights only, `matmul_q4_dyn`): the same activation-quant
// steps serve both tiers, only the weight-side GEMM and its packed word
// stride differ - `crates/wan/src/block.rs`'s module doc has the full
// reasoning for why a second int4-activation tier (W4A4) is not worth
// building.
//
// The ten quantizable linears per block match `crate::int8::is_never_
// quantized`'s exclusion list exactly: `attn1`/`attn2`'s `to_q`/`to_k`/
// `to_v`/`to_out.0` (8) plus `ff.net.0.proj`/`ff.net.2` (2). `q_norm`/
// `k_norm` (1-D), `to_gate_logits` (never-quantized: small, sets the gate's
// own numeric scale) and the host-side `scale_shift_table`/
// `prompt_scale_shift_table` stay fp32 - unchanged from [`BlockWeights`].
// Only the video-only [`LtxBlock`] gets this tier in this pass (the AV
// cross-attention path is a later lift - see this crate's roadmap ledger).

/// Which packed-weight tier a quantized block uses. int4 is W4A8, not W4A4 -
/// see this section's doc.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QTier {
    Int8,
    Int4,
}

/// One quantized linear: packed weight + its per-output-row fp32 scale, plus
/// the (never-quantized) fp32 bias - `None` for a bias-free linear (the main
/// video FFN at `ff_bias=false`, `FfWeights`'s own convention).
struct QLinear {
    w: DeviceBuffer,
    sw: DeviceBuffer,
    b: Option<DeviceBuffer>,
}

/// [`QLinear`]'s host-only, GPU-free form - the CPU-side per-channel
/// quantization result [`QLinear::quantize_host`] produces and
/// [`QLinear::from_cached`] later uploads without recomputing it. This is
/// what a generation-lifetime weight cache actually stores: `packed`/`sw`
/// are a PURE function of the checkpoint's own (immutable) weight bytes, so
/// computing them once and re-uploading the same bytes on every later
/// denoise step is bit-identical to recomputing them fresh each call - see
/// `crate::pipeline::RealDit`'s doc for why this is the cache's exact-win
/// gate.
struct CachedQLinear {
    packed: Vec<u32>,
    sw: Vec<f32>,
    b: Option<Vec<f32>>,
}

impl CachedQLinear {
    /// Real host bytes this one linear's cached form holds - what
    /// `crate::pipeline::RealDit::block_cache`'s memory cost is actually
    /// made of. `Vec::len`, not `size_of`: the packed/scale/bias buffers are
    /// heap allocations, and `size_of::<CachedQLinear>()` would only ever
    /// report the three (small, fixed) `Vec` headers.
    fn byte_len(&self) -> usize {
        std::mem::size_of_val(self.packed.as_slice()) + std::mem::size_of_val(self.sw.as_slice()) + self.b.as_ref().map_or(0, |b| std::mem::size_of_val(b.as_slice()))
    }
}

impl QLinear {
    /// GGUF-decoded fp32 in, packed int8/int4 bytes + scale out, no device
    /// touched - the CPU-only half of what one `QLinear` construction used to
    /// do in a single step. Split out so a caller that already has this
    /// result cached can skip straight to [`Self::from_cached`].
    fn quantize_host(t: &Tensors, prefix: &str, tier: QTier, out_dim: usize, in_dim: usize, has_bias: bool) -> CachedQLinear {
        let data = tget(t, &format!("{prefix}.weight"));
        let (packed, sw) = match tier {
            QTier::Int8 => model::int8::quantize_weight(data, out_dim, in_dim),
            QTier::Int4 => model::int4::quantize_weight_q4(data, out_dim, in_dim),
        };
        let b = has_bias.then(|| tget(t, &format!("{prefix}.bias")).to_vec());
        CachedQLinear { packed, sw, b }
    }

    /// The device-upload half - no CPU quantization work, just three uploads
    /// of already-quantized host bytes.
    fn from_cached(gpu: &Gpu, c: &CachedQLinear) -> QLinear {
        let wb = gpu.storage(c.packed.len() as u64);
        gpu.write(&wb, &c.packed);
        let swb = gpu.storage(c.sw.len() as u64);
        wf(gpu, &swb, &c.sw);
        let b = c.b.as_ref().map(|b| {
            let buf = gpu.storage(b.len() as u64);
            gpu.write_f32(&buf, b);
            buf
        });
        QLinear { w: wb, sw: swb, b }
    }
}

/// One `Attention` module's quantized weights - [`AttnWeights`]'s int8/int4
/// analogue: `to_q`/`to_k`/`to_v`/`to_out.0` packed, everything else
/// (`q_norm`/`k_norm`/`to_gate_logits`) unchanged fp32.
struct QAttnWeights {
    wq: QLinear,
    wk: QLinear,
    wv: QLinear,
    wo: QLinear,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
    gate: Option<GateWeights>,
}

/// [`QAttnWeights`]'s host-only form - see [`CachedQLinear`]'s doc for why
/// this exists.
struct CachedQAttnWeights {
    wq: CachedQLinear,
    wk: CachedQLinear,
    wv: CachedQLinear,
    wo: CachedQLinear,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    gate: Option<(Vec<f32>, Vec<f32>)>,
}

impl CachedQAttnWeights {
    fn byte_len(&self) -> usize {
        let mut n = self.wq.byte_len() + self.wk.byte_len() + self.wv.byte_len() + self.wo.byte_len() + std::mem::size_of_val(self.q_norm.as_slice()) + std::mem::size_of_val(self.k_norm.as_slice());
        if let Some((w, b)) = &self.gate {
            n += std::mem::size_of_val(w.as_slice()) + std::mem::size_of_val(b.as_slice());
        }
        n
    }
}

impl QAttnWeights {
    fn quantize_host(w: &Tensors, prefix: &str, gated: bool, tier: QTier, q_dim: usize, kv_dim: usize, inner_dim: usize) -> CachedQAttnWeights {
        CachedQAttnWeights {
            wq: QLinear::quantize_host(w, &format!("{prefix}.to_q"), tier, inner_dim, q_dim, true),
            wk: QLinear::quantize_host(w, &format!("{prefix}.to_k"), tier, inner_dim, kv_dim, true),
            wv: QLinear::quantize_host(w, &format!("{prefix}.to_v"), tier, inner_dim, kv_dim, true),
            wo: QLinear::quantize_host(w, &format!("{prefix}.to_out.0"), tier, q_dim, inner_dim, true),
            q_norm: tget(w, &format!("{prefix}.q_norm.weight")).to_vec(),
            k_norm: tget(w, &format!("{prefix}.k_norm.weight")).to_vec(),
            gate: gated.then(|| (tget(w, &format!("{prefix}.to_gate_logits.weight")).to_vec(), tget(w, &format!("{prefix}.to_gate_logits.bias")).to_vec())),
        }
    }

    fn from_cached(gpu: &Gpu, c: &CachedQAttnWeights) -> QAttnWeights {
        QAttnWeights {
            wq: QLinear::from_cached(gpu, &c.wq),
            wk: QLinear::from_cached(gpu, &c.wk),
            wv: QLinear::from_cached(gpu, &c.wv),
            wo: QLinear::from_cached(gpu, &c.wo),
            q_norm: {
                let b = gpu.storage(c.q_norm.len() as u64);
                wf(gpu, &b, &c.q_norm);
                b
            },
            k_norm: {
                let b = gpu.storage(c.k_norm.len() as u64);
                wf(gpu, &b, &c.k_norm);
                b
            },
            gate: c.gate.as_ref().map(|(w, b)| GateWeights {
                w: {
                    let buf = gpu.storage(w.len() as u64);
                    wf(gpu, &buf, w);
                    buf
                },
                b: {
                    let buf = gpu.storage(b.len() as u64);
                    wf(gpu, &buf, b);
                    buf
                },
            }),
        }
    }
}

/// The FFN's two quantized linears - [`FfWeights`]'s int8/int4 analogue.
struct QFfWeights {
    w1: QLinear, // [ff_dim, dim]
    w2: QLinear, // [dim, ff_dim]
}

/// [`QFfWeights`]'s host-only form.
struct CachedQFfWeights {
    w1: CachedQLinear,
    w2: CachedQLinear,
}

impl CachedQFfWeights {
    fn byte_len(&self) -> usize {
        self.w1.byte_len() + self.w2.byte_len()
    }
}

impl QFfWeights {
    fn quantize_host(w: &Tensors, prefix: &str, tier: QTier, dim: usize, has_bias: bool) -> CachedQFfWeights {
        let ff_dim = dim * 4;
        CachedQFfWeights { w1: QLinear::quantize_host(w, &format!("{prefix}.net.0.proj"), tier, ff_dim, dim, has_bias), w2: QLinear::quantize_host(w, &format!("{prefix}.net.2"), tier, dim, ff_dim, has_bias) }
    }

    fn from_cached(gpu: &Gpu, c: &CachedQFfWeights) -> QFfWeights {
        QFfWeights { w1: QLinear::from_cached(gpu, &c.w1), w2: QLinear::from_cached(gpu, &c.w2) }
    }
}

/// Resident quantized weights of one video-only block - [`BlockWeights`]'s
/// int8/int4 analogue. `pub(crate)` so `crate::dit::forward_q_streamed` can
/// call [`QBlockWeights::quantize_host`] directly when populating its
/// per-generation block-weight cache.
pub(crate) struct QBlockWeights {
    attn1: QAttnWeights,
    attn2: QAttnWeights,
    ff: QFfWeights,
    scale_shift_table: Vec<f32>,
    prompt_scale_shift_table: Vec<f32>,
}

/// [`QBlockWeights`]'s host-only form - what
/// [`crate::dit::forward_q_streamed`]'s per-generation block-weight cache
/// actually stores (`crate::pipeline::RealDit::block_cache`). Produced once
/// per block by [`QBlockWeights::quantize_host`] (the same GGUF-read +
/// int8/int4-pack work `QBlockWeights::upload` always did, just no longer
/// re-run on every one of a generation's ~20-50 denoise steps) and re-hydrated
/// into a fresh [`QBlockWeights`] on every later step by
/// [`QBlockWeights::from_cached`], which touches no CPU quantization at all -
/// only device uploads of already-quantized bytes. `pub` (not `pub(crate)`):
/// [`crate::dit::forward_q_streamed`], which takes this type in its cache
/// parameter, is itself `pub` and reachable from outside the crate
/// (`crate::pipeline::RealDit` and `bin/ltxv_bench.rs` both call it), and a
/// private type behind a public function's signature does not compile. Every
/// FIELD stays private - only [`QBlockWeights::quantize_host`]/
/// [`QBlockWeights::from_cached`] construct or read one, so nothing outside
/// this module can inspect or forge cached weight bytes.
pub struct CachedQBlockWeights {
    attn1: CachedQAttnWeights,
    attn2: CachedQAttnWeights,
    ff: CachedQFfWeights,
    scale_shift_table: Vec<f32>,
    prompt_scale_shift_table: Vec<f32>,
}

impl CachedQBlockWeights {
    /// Real host bytes this one block's cache slot holds - `pub` so a
    /// real-weight test can measure `crate::pipeline::RealDit::block_cache`'s
    /// actual memory cost at the real 22B config rather than assert it from
    /// a derivation nobody checks (lesson: a memory saving is not measured by
    /// anything unless someone measures it).
    pub fn byte_len(&self) -> usize {
        self.attn1.byte_len() + self.attn2.byte_len() + self.ff.byte_len() + std::mem::size_of_val(self.scale_shift_table.as_slice()) + std::mem::size_of_val(self.prompt_scale_shift_table.as_slice())
    }

    /// Quantize one block's weights into the cache's host form - the public
    /// face of [`QBlockWeights::quantize_host`], which `crate::dit::
    /// forward_q_streamed` calls on every cache miss. `pub` so a caller
    /// outside this crate can populate a [`crate::weightcache::
    /// GenerationCache`] without going through a full forward (the residency
    /// adapter's own demote/promote gate does exactly that).
    pub fn quantize(w: &Tensors, prefix: &str, cfg: &LtxDitConfig, tier: QTier) -> CachedQBlockWeights {
        QBlockWeights::quantize_host(w, prefix, cfg.inner_dim as usize, cfg.apply_gated_attention, tier)
    }
}

/// The host-side quantized-weight cache `crate::dit::forward_q_streamed`
/// consults - defined in [`crate::weightcache`] and re-exported here because
/// that is where every caller already imports it from, and because the scope
/// change (per-generation -> per-checkpoint) must not become an import
/// churn in every test and bench that uses it.
pub use crate::weightcache::{CacheStats, CheckpointId, GenerationCache};

/// Host bytes ONE cached block occupies at `tier`, in closed form over
/// `cfg` - what a residency estimate must report BEFORE anything has been
/// loaded, since `CachedQBlockWeights::byte_len` can only measure a block
/// that already exists.
///
/// Derived from what [`QBlockWeights::quantize_host`] actually builds, and
/// pinned against it: `cached_block_bytes_matches_a_real_measured_block`
/// quantizes a real block and asserts this function reproduces its
/// `byte_len()` exactly, so the two cannot drift apart silently the way a
/// hand-derived `file_size * 1.3` estimate does.
///
/// Per quantized linear (`model::int8::quantize_weight` /
/// `int4::quantize_weight_q4`): `out*in/4` packed `u32` words at int8 (i.e.
/// `out*in` bytes) or `out*in/8` at int4, plus `out*in/32` fp32 GROUP scales
/// (`model::int8::GROUP`, the same count at either tier), plus an
/// unquantized fp32 bias where the linear has one. Everything else - the two
/// RMSNorm gains, the optional attention gate, and the two scale/shift tables
/// - stays fp32 exactly as the cache stores it.
pub fn cached_block_bytes(cfg: &LtxDitConfig, tier: QTier) -> u64 {
    /// `to_gate_logits`'s fixed row count - the same literal
    /// `crate::dit::push_attn` writes into the tensor manifest.
    const GATE_LOGIT_ROWS: u64 = 32;

    let dim = cfg.inner_dim as u64;
    let f32s = |n: u64| n * 4;
    let packed = |out: u64, inp: u64| -> u64 {
        let words = match tier {
            QTier::Int8 => out * inp / 4,
            QTier::Int4 => out * inp / 8,
        };
        words * 4 + f32s(out * inp / model::int8::GROUP as u64)
    };
    // One attention module: to_q/to_k/to_v (dim x dim, biased), to_out.0
    // (dim x dim, biased), q_norm/k_norm (dim each), optional gate.
    let attn = 4 * (packed(dim, dim) + f32s(dim)) + 2 * f32s(dim) + if cfg.apply_gated_attention { f32s(GATE_LOGIT_ROWS * dim) + f32s(GATE_LOGIT_ROWS) } else { 0 };
    // The FFN's two linears are bias-free at this config (`ff_bias=false`).
    let ff = packed(4 * dim, dim) + packed(dim, 4 * dim);
    2 * attn + ff + f32s(cfg.adaln_rows() as u64 * dim) + f32s(2 * dim)
}

impl QBlockWeights {
    pub(crate) fn quantize_host(w: &Tensors, prefix: &str, dim: usize, gated: bool, tier: QTier) -> CachedQBlockWeights {
        QBlockWeights::quantize_host_stream(w, prefix, "", dim, gated, tier)
    }

    /// [`Self::quantize_host`] for ONE stream of a block - `stream` is `""`
    /// for video's own weights and `"audio_"` for audio's, exactly
    /// [`BlockWeights::upload`]'s convention and for the same reason (both
    /// streams' per-block state is structurally identical; only the tensor
    /// name prefix, the dims and the FFN's bias differ).
    pub(crate) fn quantize_host_stream(w: &Tensors, prefix: &str, stream: &str, dim: usize, gated: bool, tier: QTier) -> CachedQBlockWeights {
        let sst = tget(w, &format!("{prefix}.{stream}scale_shift_table")).to_vec();
        assert_eq!(sst.len(), 9 * dim, "{prefix}.{stream}scale_shift_table must be [9, dim]");
        let pst = tget(w, &format!("{prefix}.{stream}prompt_scale_shift_table")).to_vec();
        assert_eq!(pst.len(), 2 * dim, "{prefix}.{stream}prompt_scale_shift_table must be [2, dim]");
        let ffp = format!("{prefix}.{stream}ff");
        let ff_bias = ff_has_bias(w, &ffp);
        CachedQBlockWeights {
            attn1: QAttnWeights::quantize_host(w, &format!("{prefix}.{stream}attn1"), gated, tier, dim, dim, dim),
            attn2: QAttnWeights::quantize_host(w, &format!("{prefix}.{stream}attn2"), gated, tier, dim, dim, dim),
            ff: QFfWeights::quantize_host(w, &ffp, tier, dim, ff_bias),
            scale_shift_table: sst,
            prompt_scale_shift_table: pst,
        }
    }

    fn from_cached(gpu: &Gpu, c: &CachedQBlockWeights) -> QBlockWeights {
        QBlockWeights { attn1: QAttnWeights::from_cached(gpu, &c.attn1), attn2: QAttnWeights::from_cached(gpu, &c.attn2), ff: QFfWeights::from_cached(gpu, &c.ff), scale_shift_table: c.scale_shift_table.clone(), prompt_scale_shift_table: c.prompt_scale_shift_table.clone() }
    }
}

/// Quantize `x` (`[rows, k]`) into `(xq, sx)` with a fresh dynamic per-row
/// scale - the shared activation-quant step every quantized linear below
/// reads. `k` must be a multiple of 4 (`quant_pack` writes four int8 per
/// `u32`); the WEIGHT side additionally needs `k % 32 == 0`
/// (`model::int8::GROUP`).
fn qquant(gpu: &Gpu, s: &mut Vec<Step>, x: &DeviceBuffer, xq: &DeviceBuffer, sx: &DeviceBuffer, rows: u32, k: u32) {
    s.push(gpu.step(K_MAX_ABS_ROW, &[x, sx], &[rows, k], rows));
    s.push(gpu.step(K_QUANT_PACK, &[x, sx, xq], &[rows, k], rows * k / 4));
}

/// `out = dequant(xq @ Wᵀ) + b`: one quantized linear, tier-dispatched -
/// `xq`/`sx` must already hold `x`'s quantization at this `(rows, k)` shape
/// (from [`qquant`]).
#[allow(clippy::too_many_arguments)]
fn qlinear(gpu: &Gpu, s: &mut Vec<Step>, tier: QTier, xq: &DeviceBuffer, sx: &DeviceBuffer, w: &QLinear, out: &DeviceBuffer, rows: u32, k: u32, n: u32) {
    match tier {
        QTier::Int8 => s.push(gpu.step(
            K_MATMUL_I8_DYN,
            &[xq, &w.w, sx, &w.sw, out],
            &[rows, k / 4, n],
            rows.div_ceil(128) * n.div_ceil(128) * 256,
        )),
        // `k` here is the LOGICAL (undivided) K - see `matmul_q4_dyn.wgsl`'s
        // own doc, `crates/wan/src/block.rs::qlinear`'s int4 branch.
        QTier::Int4 => s.push(gpu.step(K_MATMUL_Q4_DYN, &[xq, &w.w, sx, &w.sw, out], &[rows, k, n], rows * n)),
    }
    if let Some(b) = &w.b {
        s.push(gpu.step(K_BIAS_ADD, &[out, b], &[rows, n], rows * n));
    }
}

/// [`qquant`] + [`qlinear`] in one call, for the common case of a fresh
/// `(rows, k)` activation with no reuse across multiple linears.
#[allow(clippy::too_many_arguments)]
fn qquant_linear(gpu: &Gpu, s: &mut Vec<Step>, tier: QTier, x: &DeviceBuffer, w: &QLinear, out: &DeviceBuffer, rows: u32, k: u32, n: u32) {
    let xq = gpu.storage((rows * k / 4) as u64);
    let sx = gpu.storage(rows as u64);
    qquant(gpu, s, x, &xq, &sx, rows, k);
    qlinear(gpu, s, tier, &xq, &sx, w, out, rows, k, n);
}

/// [`attention`]'s quantized-compute twin: identical fp32 RMSNorm/RoPE/
/// attention-score/gating math - only the four projections (`to_q`/`to_k`/
/// `to_v`/`to_out.0`) run through [`qquant_linear`] instead of [`linear`].
#[allow(clippy::too_many_arguments)]
fn attention_q(
    gpu: &Gpu,
    s: &mut Vec<Step>,
    tier: QTier,
    w: &QAttnWeights,
    q_dim: u32,
    kv_dim: u32,
    inner_dim: u32,
    heads: u32,
    head_dim: u32,
    q_in: &DeviceBuffer,
    kv_in: &DeviceBuffer,
    nq: u32,
    nk: u32,
    q_rope: Option<(&[DeviceBuffer], &[DeviceBuffer])>,
    k_rope: Option<(&[DeviceBuffer], &[DeviceBuffer])>,
    kernel_rope2d: usize,
    eps: f32,
    self_attn: bool,
) -> DeviceBuffer {
    let q_pre = gpu.storage((nq * inner_dim) as u64);
    qquant_linear(gpu, s, tier, q_in, &w.wq, &q_pre, nq, q_dim, inner_dim);
    let k_pre = gpu.storage((nk * inner_dim) as u64);
    qquant_linear(gpu, s, tier, kv_in, &w.wk, &k_pre, nk, kv_dim, inner_dim);
    let v = gpu.storage((nk * inner_dim) as u64);
    qquant_linear(gpu, s, tier, kv_in, &w.wv, &v, nk, kv_dim, inner_dim);

    let q = gpu.storage((nq * inner_dim) as u64);
    let k = gpu.storage((nk * inner_dim) as u64);
    rmsnorm(gpu, s, &q_pre, &w.q_norm, &q, inner_dim, nq, eps);
    rmsnorm(gpu, s, &k_pre, &w.k_norm, &k, inner_dim, nk, eps);

    if let Some((cos_bufs, sin_bufs)) = q_rope {
        for h in 0..heads {
            let off = h * head_dim;
            s.push(apply_rope_step(gpu, kernel_rope2d, &q, &cos_bufs[h as usize], &sin_bufs[h as usize], nq, head_dim, inner_dim, off));
        }
    }
    if let Some((cos_bufs, sin_bufs)) = k_rope {
        for h in 0..heads {
            let off = h * head_dim;
            s.push(apply_rope_step(gpu, kernel_rope2d, &k, &cos_bufs[h as usize], &sin_bufs[h as usize], nk, head_dim, inner_dim, off));
        }
    }

    let ctx = attn_context(gpu, s, &q, &k, &v, heads, head_dim, inner_dim, nq, nk, self_attn);

    // Gating stays fp32 - `to_gate_logits` is never quantized (small,
    // fixes the gate's own numeric scale, see `crate::int8::is_never_
    // quantized`'s doc), unchanged from [`attention`].
    let ctx_gated = if let Some(gate) = &w.gate {
        let logits = gpu.storage((nq * heads) as u64);
        linear(gpu, s, q_in, &gate.w, Some(&gate.b), &logits, nq, q_dim, heads);
        let sig = gpu.storage((nq * heads) as u64);
        s.push(gpu.step(K_SIGMOID, &[&logits, &sig], &[nq * heads], nq * heads));
        let expand = gpu.storage((inner_dim * heads) as u64);
        wf(gpu, &expand, &gate_expand_matrix(heads, head_dim));
        let gate_bc = gpu.storage((nq * inner_dim) as u64);
        linear(gpu, s, &sig, &expand, None, &gate_bc, nq, heads, inner_dim);
        let gated = gpu.storage((nq * inner_dim) as u64);
        mul(gpu, s, &ctx, &gate_bc, &gated, nq * inner_dim);
        gated
    } else {
        ctx
    };

    let out = gpu.storage((nq * q_dim) as u64);
    qquant_linear(gpu, s, tier, &ctx_gated, &w.wo, &out, nq, inner_dim, q_dim);
    out
}

/// [`self_attn_and_text_ca`]'s quantized-compute twin - identical structure,
/// [`attention_q`] instead of [`attention`].
#[allow(clippy::too_many_arguments)]
fn self_attn_and_text_ca_q(
    gpu: &Gpu,
    s: &mut Vec<Step>,
    tier: QTier,
    w: &QBlockWeights,
    ones: &DeviceBuffer,
    x_buf: &DeviceBuffer,
    m: &ModBufs,
    ctx_buf: &DeviceBuffer,
    ctx_len: u32,
    cos_bufs: &[DeviceBuffer],
    sin_bufs: &[DeviceBuffer],
    dim: u32,
    heads: u32,
    head_dim: u32,
    t: u32,
    eps: f32,
) -> (DeviceBuffer, DeviceBuffer, DeviceBuffer) {
    let td = t * dim;
    let (shift_msa, one_plus_scale_msa, gate_msa) = (&m.shift_msa, &m.one_plus_scale_msa, &m.gate_msa);
    let (shift_q, one_plus_scale_q, gate_q) = (&m.shift_q, &m.one_plus_scale_q, &m.gate_q);

    // The text context's own scale/shift, broadcast to every context row.
    // Both `[dim]` rows come from this block's `prompt_scale_shift_table`, so
    // the whole `[ctx_len, dim]` pair is constant for a generation and is
    // nevertheless rebuilt per block per forward - see this crate's roadmap
    // ledger for what that costs and why removing it needs a device-side
    // broadcast rather than a bigger resident block. Two things it does NOT
    // do any more: recompute `1.0 + scale` per ELEMENT (once per row now,
    // then a `copy_from_slice` like its sibling), and hand `write_buffer` a
    // whole `[ctx_len, dim]` payload in one call - on a non-ReBAR card the
    // staging allocation that needs is the size of the payload, which is the
    // same reason `crate::devres` chunks its own table uploads.
    let pst = &w.prompt_scale_shift_table;
    let (shift_kv_row, scale_kv_row) = (&pst[0..dim as usize], &pst[dim as usize..2 * dim as usize]);
    let one_plus_scale_row: Vec<f32> = scale_kv_row.iter().map(|v| 1.0 + v).collect();
    let mut shift_kv = vec![0f32; (ctx_len * dim) as usize];
    let mut one_plus_scale_kv = vec![0f32; (ctx_len * dim) as usize];
    for r in 0..ctx_len as usize {
        let (lo, hi) = (r * dim as usize, r * dim as usize + dim as usize);
        shift_kv[lo..hi].copy_from_slice(shift_kv_row);
        one_plus_scale_kv[lo..hi].copy_from_slice(&one_plus_scale_row);
    }
    let bcast = |v: &[f32]| -> DeviceBuffer {
        let b = gpu.storage(v.len() as u64);
        gpu.write_f32_chunked(&b, v, 1 << 20);
        b
    };
    let shift_kv_buf = bcast(&shift_kv);
    let one_plus_scale_kv_buf = bcast(&one_plus_scale_kv);

    let tmp1 = gpu.storage(td as u64);
    let tmp2 = gpu.storage(td as u64);
    let norm_x = gpu.storage(td as u64);
    ada_zero(gpu, s, ones, x_buf, one_plus_scale_msa, shift_msa, &tmp1, &tmp2, &norm_x, dim, t, eps);
    let attn1_out = attention_q(gpu, s, tier, &w.attn1, dim, dim, dim, heads, head_dim, &norm_x, &norm_x, t, t, Some((cos_bufs, sin_bufs)), Some((cos_bufs, sin_bufs)), K_ROPE2D, eps, true);
    let x_fma = gpu.storage(td as u64);
    gate_row(gpu, s, x_buf, gate_msa, &attn1_out, &x_fma, t, dim, 1);

    let x_normed = gpu.storage(td as u64);
    rmsnorm(gpu, s, &x_fma, ones, &x_normed, dim, t, eps);

    let attn_input_tmp1 = gpu.storage(td as u64);
    let attn_input = gpu.storage(td as u64);
    mul(gpu, s, &x_normed, one_plus_scale_q, &attn_input_tmp1, td);
    add2(gpu, s, &attn_input_tmp1, shift_q, &attn_input, td);

    let ctxd = ctx_len * dim;
    let enc_tmp1 = gpu.storage(ctxd as u64);
    let enc_hidden = gpu.storage(ctxd as u64);
    mul(gpu, s, ctx_buf, &one_plus_scale_kv_buf, &enc_tmp1, ctxd);
    add2(gpu, s, &enc_tmp1, &shift_kv_buf, &enc_hidden, ctxd);

    let ca_raw = attention_q(gpu, s, tier, &w.attn2, dim, dim, dim, heads, head_dim, &attn_input, &enc_hidden, t, ctx_len, None, None, K_ROPE2D, eps, false);
    let ca_gated = gpu.storage(td as u64);
    mul(gpu, s, &ca_raw, gate_q, &ca_gated, td);
    let x2 = gpu.storage(td as u64);
    add2(gpu, s, &x_fma, &ca_gated, &x2, td);

    (x2, attn1_out, ca_raw)
}

/// [`mlp_sublayer`]'s quantized-compute twin - identical structure,
/// [`qquant_linear`] instead of [`linear`] for both FFN projections.
#[allow(clippy::too_many_arguments)]
fn mlp_sublayer_q(gpu: &Gpu, s: &mut Vec<Step>, w: &QFfWeights, tier: QTier, ones: &DeviceBuffer, x2: &DeviceBuffer, shift_mlp_buf: &DeviceBuffer, one_plus_scale_mlp_buf: &DeviceBuffer, gate_mlp_buf: &DeviceBuffer, dim: u32, t: u32, eps: f32) -> (DeviceBuffer, DeviceBuffer) {
    let td = t * dim;
    let mlp_tmp1 = gpu.storage(td as u64);
    let mlp_tmp2 = gpu.storage(td as u64);
    let x_scaled = gpu.storage(td as u64);
    ada_zero(gpu, s, ones, x2, one_plus_scale_mlp_buf, shift_mlp_buf, &mlp_tmp1, &mlp_tmp2, &x_scaled, dim, t, eps);
    let ff_dim = dim * 4;
    let h_pre = gpu.storage((t * ff_dim) as u64);
    qquant_linear(gpu, s, tier, &x_scaled, &w.w1, &h_pre, t, dim, ff_dim);
    let h_act = gpu.storage((t * ff_dim) as u64);
    s.push(gpu.step(K_GELU, &[&h_pre, &h_act], &[t * ff_dim], t * ff_dim));
    let ff_out = gpu.storage(td as u64);
    qquant_linear(gpu, s, tier, &h_act, &w.w2, &ff_out, t, ff_dim, dim);
    let x3 = gpu.storage(td as u64);
    gate_row(gpu, s, x2, gate_mlp_buf, &ff_out, &x3, t, dim, 1);
    (x3, ff_out)
}

/// Whether a block forward draws its temporaries from the device handle's
/// replay arena (`gpu_core::scratch`) instead of creating and destroying them
/// per block.
///
/// On by default. `BRAIN_LTXV_NO_SCRATCH_POOL=1` is the opt-out, and it exists
/// for the same reason `BRAIN_NO_COOP_LN` does: without a force switch the two
/// arms cannot be A/B'd in one binary, and a comparison that can only run the
/// shipped path confirms whatever was written. The arms must agree BIT for
/// bit - the arena changes when device memory is allocated, never what any
/// kernel reads - which is what `crates/ltxv/tests/scratch_pool.rs` asserts.
fn scratch_pool() -> bool {
    // Read per call, NOT cached in a `OnceLock`: the gate
    // (`crates/ltxv/tests/scratch_pool.rs`) runs both arms in one process and
    // compares their outputs bit for bit, which a cached first read would
    // silently turn into a comparison of one arm against itself - a green
    // test that can only ever have run one arm. One `var_os` per block
    // forward is nothing against a forward that dispatches hundreds of
    // kernels.
    std::env::var_os("BRAIN_LTXV_NO_SCRATCH_POOL").map(|v| v != "1").unwrap_or(true)
}

/// Whether a chained block stack overlaps block `l`'s HOST work with block
/// `l-1`'s DEVICE work, instead of standing still until each block's own
/// dispatches have finished.
///
/// ## What moves, and why that is the whole change
///
/// A block body is three phases: RECORD (build bind groups, upload this
/// block's weights if residency could not keep them), SUBMIT (hand the batch
/// to the queue), and WAIT (block until the card is done). Recording touches
/// no device memory - a bind group names buffers, it does not write them - so
/// the wait does not have to sit between this block's submit and the next
/// block's recording. It only has to sit between the previous block's submit
/// and THIS block's submit.
///
/// Moving it there is all this switch does:
///
/// ```text
///   serial:    record(l)  submit(l)  wait(l)          record(l+1) submit(l+1) wait(l+1)
///   pipelined: record(l)  wait(l-1)  submit(l) flush  record(l+1) wait(l)     submit(l+1) flush
/// ```
///
/// The device sees the identical submissions in the identical order, with at
/// most one block's dispatches in flight either way, so nothing about what any
/// kernel reads or when it runs changes. What changes is where the HOST is
/// while the card works: recording block `l+1` and uploading its weights,
/// rather than parked in `device.poll`.
///
/// ## Why ONE scratch arena is still enough
///
/// The recorded-not-yet-submitted phase is what makes this cheap. Block
/// `l+1`'s temporaries come from the same arena slots block `l` used, and it
/// takes them BEFORE block `l` has finished - but it only takes the handles.
/// Every dispatch that writes one of those buffers is submitted after the wait
/// that block `l` completes, so no arena slot is ever written while an
/// unfinished dispatch still reads it. A second, alternating arena would buy
/// nothing here and would cost a whole block's scratch in device memory.
///
/// The one obligation this hands the caller is that a stack must still be
/// drained before its output is used, which the chained stack already does:
/// `DitSession::run_blocks` reads the final activation back, and that read is
/// a flush plus a blocking wait.
///
/// ## What the other two backends do with it
///
/// Nothing, and correctly. `backend-cpu` executes a batch inside `submit` and
/// its `flush`/`poll_wait` are no-ops, so the ordering below collapses to
/// exactly what it always did. `backend-vulkan`'s `flush` fence-waits by
/// design, so it drains where this arm asks it to queue: no overlap, same
/// answer. Only the default wgpu backend has a queue to run ahead of.
///
/// On by default. `BRAIN_LTXV_NO_PIPELINE=1` is the opt-out, for the same
/// reason [`scratch_pool`] has one: without a switch the two arms cannot be
/// A/B'd in one binary, and both the timings and the bit-for-bit comparison in
/// `crates/ltxv/tests/block_pipeline.rs` need two arms in one process.
fn block_pipeline() -> bool {
    // Read per call, never cached, for `scratch_pool`'s reason: the gate runs
    // both arms in one process and a cached first read would silently compare
    // an arm against itself.
    std::env::var_os("BRAIN_LTXV_NO_PIPELINE").map(|v| v != "1").unwrap_or(true)
}

/// Where one block forward's wall-clock really goes.
///
/// Four buckets, because a real 48-layer forward at the production token count
/// splits between them in a way that is not guessable and was in fact guessed
/// WRONG once (device residency was built on the belief that the ~270 MB
/// per-block WEIGHT upload dominated; it turned out to be under a tenth of
/// the forward it was blamed for). Each boundary is chosen so the split is exact rather than
/// plausible - see [`LtxBlockQ::forward_timed`]'s body for why the recording
/// span really is the upload span.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlockTimings {
    /// `add_table` + `slice_mod`: building this block's own `[t, 9*dim]`
    /// combined modulation table and slicing it into nine `[t, dim]` vectors,
    /// on the host, single-threaded.
    pub modulation_host: std::time::Duration,
    /// Host->device writes (the activations, the text context, the nine
    /// modulation vectors, the two prompt scale/shift broadcasts) plus graph
    /// recording. No kernel has run yet when this span closes.
    pub record_upload: std::time::Duration,
    /// `submit` + the wait for the device - the span that contains the GPU's
    /// actual work. On the PIPELINED block stack (see [`block_pipeline`]) the
    /// wait in block `l`'s span is for block `l-1`'s dispatches, because block
    /// `l`'s own are queued at the end of it and left running; the stack total
    /// is still every device wait the stack performed, shifted by one block.
    pub compute: std::time::Duration,
    /// Device->host readback. For a chained stack
    /// ([`LtxBlockQ::forward_prod_dev`]) that is ONE `[t, dim]` read for the
    /// whole forward; for the tap-producing path
    /// ([`LtxBlockQ::forward_timed`]) it is the three parity TAP readbacks
    /// (`attn1_out`/`attn2_out`/`ff_out`), each a full `[t, dim]`, per block -
    /// all three of which a production forward discards, and which were better
    /// than a tenth of a real 48-layer 720p forward before the chained path
    /// existed.
    pub readback: std::time::Duration,
}

impl BlockTimings {
    /// Fold `other` into `self` - how a whole block stack's totals are
    /// accumulated.
    pub fn add(&mut self, other: &BlockTimings) {
        self.modulation_host += other.modulation_host;
        self.record_upload += other.record_upload;
        self.compute += other.compute;
        self.readback += other.readback;
    }
}

/// [`LtxBlock`]'s quantized-compute twin: one video-only block, weights
/// resident in packed int8/int4 form. See this section's module doc for the
/// tier's exact scope (video-only, ten quantizable linears per block).
pub struct LtxBlockQ {
    gpu: Gpu,
    cfg: LtxDitConfig,
    tier: QTier,
    w: QBlockWeights,
    context_len: u32,
    ones_t: DeviceBuffer,
    /// This block's own `[9, dim]` `scale_shift_table` ON the device - 147 KB,
    /// uploaded with the rest of the block's weights and resident for exactly
    /// as long as they are. It is what [`ModBufs::derive`] adds to the
    /// model-level adaLN table, and the reason a resident block needs nothing
    /// from the host on a warm step.
    sst_buf: DeviceBuffer,
}

impl LtxBlockQ {
    /// `weights`/`prefix`/`tokens`/`context_len`: same convention as
    /// [`LtxBlock::on`]. `weights` is a plain [`Tensors`] map, but it need not
    /// come from a whole-model fp32 materialization: [`load_block_tensors_from_source`]
    /// builds exactly this one block's own tensors straight off a real GGUF
    /// `checkpoint::TensorSource` (`crate::gguf_src::LtxvGgufSource`).
    pub fn on(gpu: Gpu, cfg: &LtxDitConfig, weights: &Tensors, prefix: &str, tokens: u32, context_len: u32, tier: QTier) -> LtxBlockQ {
        cfg.assert_supported();
        let dim = cfg.inner_dim as usize;
        let cached = QBlockWeights::quantize_host(weights, prefix, dim, cfg.apply_gated_attention, tier);
        Self::on_cached(gpu, cfg, &cached, tokens, context_len, tier)
    }

    /// [`Self::on`]'s cached twin: `cached` is an already-quantized
    /// [`CachedQBlockWeights`] (from an earlier [`QBlockWeights::
    /// quantize_host`] call, or - the real point of this constructor -
    /// carried over from a PREVIOUS denoise step by
    /// [`crate::dit::forward_q_streamed`]'s per-generation cache). No GGUF
    /// read, no CPU quantization - only device uploads of already-quantized
    /// host bytes, which is what turns the dominant share of a real denoise
    /// step that Phase 8 attributed to GGUF read+dequant+quantize into a one-time cost
    /// paid on a generation's first forward call instead of on every one of
    /// its ~20-50 steps.
    pub fn on_cached(gpu: Gpu, cfg: &LtxDitConfig, cached: &CachedQBlockWeights, tokens: u32, context_len: u32, tier: QTier) -> LtxBlockQ {
        cfg.assert_supported();
        let dim = cfg.inner_dim as usize;
        let w = QBlockWeights::from_cached(&gpu, cached);
        let ones_t = gpu.storage(dim as u64);
        wf(&gpu, &ones_t, &vec![1.0f32; dim]);
        let sst_buf = gpu.storage(w.scale_shift_table.len() as u64);
        wf(&gpu, &sst_buf, &w.scale_shift_table);
        let _ = tokens;
        LtxBlockQ { gpu, cfg: *cfg, tier, w, context_len, ones_t, sst_buf }
    }

    /// One block forward - same inputs/outputs as [`LtxBlock::forward`].
    ///
    /// Runs on this block's OWN handle, which is also the handle its weights
    /// were uploaded through. Correct for the throwaway shape (build a block,
    /// run it once, drop it); see [`Self::forward_on`] for the resident shape.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, x: &[f32], adaln_table: &[f32], context: &[f32], cos_bufs: &[DeviceBuffer], sin_bufs: &[DeviceBuffer], t: u32) -> (Vec<f32>, BlockTaps) {
        self.forward_on(&self.gpu, x, adaln_table, context, cos_bufs, sin_bufs, t)
    }

    /// [`Self::forward`] with the per-call SCRATCH allocations placed on a
    /// caller-supplied handle instead of this block's own.
    ///
    /// Same device, different `gpu_core::Gpu` handle (`Gpu::share`), and that
    /// separation is the whole point when a block is RESIDENT across a
    /// generation's forward calls (`crate::devres`). A `Gpu`'s `memauth`
    /// grants are released when the HANDLE drops, not when an individual
    /// buffer does (see `gpu_core`'s own `grants` doc), so running a resident
    /// block's transients through its long-lived weight handle would charge
    /// every step's activations against the process-wide `--limit-vram-total`
    /// ceiling forever and refuse a run that actually fits. A fresh scratch
    /// handle per forward keeps the accounting honest: the weights are charged
    /// for exactly as long as they are resident, the activations for exactly
    /// as long as the call runs.
    ///
    /// Bit-identity is not a claim about tolerance: `scratch` binds the SAME
    /// weight buffers into the SAME kernels in the SAME order, and a
    /// `Gpu::share` handle is the same adapter, queue and compiled pipelines.
    /// Only which handle recorded the dispatch changes, which no kernel can
    /// observe.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_on(&self, scratch: &Gpu, x: &[f32], adaln_table: &[f32], context: &[f32], cos_bufs: &[DeviceBuffer], sin_bufs: &[DeviceBuffer], t: u32) -> (Vec<f32>, BlockTaps) {
        self.forward_timed(scratch, x, adaln_table, context, cos_bufs, sin_bufs, t, &mut BlockTimings::default())
    }

    /// [`Self::forward_on`] with the four costs a block forward really has,
    /// split apart - see [`BlockTimings`] for why they are worth separating
    /// and what each boundary is.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_timed(&self, scratch: &Gpu, x: &[f32], adaln_table: &[f32], context: &[f32], cos_bufs: &[DeviceBuffer], sin_bufs: &[DeviceBuffer], t: u32, timings: &mut BlockTimings) -> (Vec<f32>, BlockTaps) {
        let gpu = scratch;
        let cfg = &self.cfg;
        let dim = cfg.inner_dim;
        let heads = cfg.num_heads;
        let head_dim = cfg.head_dim();
        let eps = cfg.norm_eps;
        let ctx_len = self.context_len;
        assert_eq!(x.len(), (t * dim) as usize);
        assert_eq!(adaln_table.len(), (t * 9 * dim) as usize);
        assert_eq!(context.len(), (ctx_len * dim) as usize);

        let s_mod = std::time::Instant::now();
        let combined = dit::adaln::add_table(adaln_table, &self.w.scale_shift_table, t as usize, 9 * dim as usize);
        let m = slice_mod(&combined, t as usize, dim as usize);
        timings.modulation_host += s_mod.elapsed();

        let s_rec = std::time::Instant::now();
        let x_buf = gpu.storage((t * dim) as u64);
        wf(gpu, &x_buf, x);
        let ctx_buf = gpu.storage((ctx_len * dim) as u64);
        wf(gpu, &ctx_buf, context);
        let mb = ModBufs::upload(gpu, &m);

        let mut s: Vec<Step> = Vec::new();
        let td = t * dim;

        let (x2, attn1_out, ca_raw) = self_attn_and_text_ca_q(gpu, &mut s, self.tier, &self.w, &self.ones_t, &x_buf, &mb, &ctx_buf, ctx_len, cos_bufs, sin_bufs, dim, heads, head_dim, t, eps);
        let (x3, ff_out) = mlp_sublayer_q(gpu, &mut s, &self.w.ff, self.tier, &self.ones_t, &x2, &mb.shift_mlp, &mb.one_plus_scale_mlp, &mb.gate_mlp, dim, t, eps);
        // Every `wf` above is a synchronous `queue.write_buffer` and NOTHING
        // computes until `submit`, so the span this closes is exactly the
        // host->device upload + graph recording cost, cleanly separated from
        // the GPU's own work below.
        timings.record_upload += s_rec.elapsed();

        let s_sub = std::time::Instant::now();
        gpu.submit(&[], &s);
        let out = gpu.read(&x3, td as usize);
        timings.compute += s_sub.elapsed();
        let s_taps = std::time::Instant::now();
        let taps = BlockTaps { attn1_out: gpu.read(&attn1_out, td as usize), attn2_out: gpu.read(&ca_raw, td as usize), ff_out: gpu.read(&ff_out, td as usize) };
        timings.readback += s_taps.elapsed();
        (out, taps)
    }

    /// The PRODUCTION form: modulation derived on the device, no parity taps,
    /// and only the activations crossing PCIe.
    ///
    /// Three things it does NOT do, and the reasons are measured:
    ///
    /// * it does not build this block's modulation on the host and upload it -
    ///   [`ModBufs::derive`] computes the same nine `[t, dim]` vectors on the
    ///   card from `adaln_buf` (uploaded once per FORWARD) and this block's
    ///   resident `scale_shift_table`. At the real 22B/720p shape the host
    ///   combine+slice alone was about a third of a forward, on top of ~25 GB
    ///   of PCIe traffic;
    /// * it does not upload the text context, which is uploaded once per
    ///   forward instead of once per block (~0.8 GB saved);
    /// * it does not read back the three parity TAPS, which a production
    ///   forward discards - better than a tenth of a forward, and 8.3 GB of
    ///   readback, per forward.
    ///
    /// Bit-identical to [`Self::forward_timed`] by construction - same kernels,
    /// same operands, same order; only the ROUTE the modulation bytes take to
    /// the card changes. Gated by `crates/ltxv/tests/device_residency.rs`.
    ///
    /// Takes and returns HOST floats. A stack of blocks should use
    /// [`Self::forward_prod_dev`] instead and keep the activation on the card;
    /// this form stays because a single-block gate wants to compare host
    /// floats and should not have to manage device buffers to do it.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prod(
        &self,
        scratch: &Gpu,
        x: &[f32],
        adaln_buf: &DeviceBuffer,
        adaln_map_buf: &DeviceBuffer,
        ctx_buf: &DeviceBuffer,
        cos_bufs: &[DeviceBuffer],
        sin_bufs: &[DeviceBuffer],
        t: u32,
        timings: &mut BlockTimings,
    ) -> Vec<f32> {
        let dim = self.cfg.inner_dim;
        assert_eq!(x.len(), (t * dim) as usize);
        let s_rec = std::time::Instant::now();
        let x_buf = scratch.storage((t * dim) as u64);
        wf(scratch, &x_buf, x);
        timings.record_upload += s_rec.elapsed();
        let x3 = self.forward_prod_dev(scratch, &x_buf, adaln_buf, adaln_map_buf, ctx_buf, cos_bufs, sin_bufs, t, timings);
        let s_back = std::time::Instant::now();
        let out = scratch.read(&x3, (t * dim) as usize);
        timings.readback += s_back.elapsed();
        out
    }

    /// [`Self::forward_prod`] with the activation already on the card and
    /// STAYING there: reads `x_buf`, returns this block's own output buffer.
    ///
    /// A block stack chains device buffer to device buffer, so `x` crosses
    /// PCIe once per FORWARD instead of twice per BLOCK. At the token count a
    /// real generation runs at, a `[t, dim]` activation is hundreds of MB and
    /// there are 48 blocks, so the host round trip was tens of GB per forward
    /// of traffic whose only purpose was to arrive back where it started.
    ///
    /// What the host round trip DID buy, and what this keeps buying a much
    /// cheaper way: on a non-ReBAR Pascal card under the default wgpu backend,
    /// `write_buffer` staging and dropped buffers are only retired by a
    /// BLOCKING readback, and an uploaded storage buffer otherwise costs twice
    /// its size resident (measured directly by
    /// `crates/gpu-core/tests/vram_overhead.rs`). Left unretired the
    /// allocator's pool grows several-fold and the resident weight window
    /// loses more blocks than the saved traffic is worth - which is why an
    /// earlier pass measured chaining as a NET LOSS and kept the round trip.
    /// What that pass missed is that the drain needs the readback to be
    /// BLOCKING, not to be big: one word is a full `map_async` + wait and
    /// retires exactly the same staging a whole `[t, dim]` read does
    /// (`backend_wgpu`'s `read` is `flush` + `map_async` + a bounded
    /// `poll_wait` whatever `n` is; only the copy scales). That is the same
    /// one-word probe `crate::devres::DitSession::prefill` already uses
    /// between weight uploads, for the same reason - and it is taken off `x3`
    /// itself, so it needs no buffer of its own.
    ///
    /// **Where that drain SITS is a separate question from whether it
    /// happens**, and it is the one [`block_pipeline`] answers: one blocking
    /// wait per block is what bounds the staging, but it does not have to be
    /// the last thing this function does. Placed before the submit instead of
    /// after it, the same wait bounds the same staging while the host records
    /// the next block against a card that is still busy with this one.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prod_dev(
        &self,
        scratch: &Gpu,
        x_buf: &DeviceBuffer,
        adaln_buf: &DeviceBuffer,
        adaln_map_buf: &DeviceBuffer,
        ctx_buf: &DeviceBuffer,
        cos_bufs: &[DeviceBuffer],
        sin_bufs: &[DeviceBuffer],
        t: u32,
        timings: &mut BlockTimings,
    ) -> DeviceBuffer {
        let cfg = &self.cfg;
        let (dim, heads, head_dim, eps, ctx_len) = (cfg.inner_dim, cfg.num_heads, cfg.head_dim(), cfg.norm_eps, self.context_len);
        // Every block dispatches the identical shape sequence, so the ~70
        // temporaries this body asks `scratch` for are the same sizes in the
        // same order every time. Inside the scope they are drawn from the
        // handle's replay arena instead of being created and destroyed per
        // block - see `gpu_core::scratch` for the aliasing argument, and note
        // that the two things it requires of a caller both hold here: every
        // dispatch recorded in this scope is SUBMITTED only after the previous
        // scope's work has been drained (`block_pipeline` explains why that,
        // and not "drained before the scope opens", is the condition), and the
        // one buffer that outlives the scope (`x3`, the chained activation) is
        // still held by the caller, so the arena takes a fresh allocation for
        // its slot rather than aliasing it.
        let _scope = scratch_pool().then(|| scratch.scratch_scope());
        let s_rec = std::time::Instant::now();
        let mut s: Vec<Step> = Vec::new();
        let mb = ModBufs::derive(scratch, &mut s, adaln_buf, adaln_map_buf, &self.sst_buf, t, dim);
        let (x2, _attn1_out, _ca_raw) = self_attn_and_text_ca_q(scratch, &mut s, self.tier, &self.w, &self.ones_t, x_buf, &mb, ctx_buf, ctx_len, cos_bufs, sin_bufs, dim, heads, head_dim, t, eps);
        let (x3, _ff_out) = mlp_sublayer_q(scratch, &mut s, &self.w.ff, self.tier, &self.ones_t, &x2, &mb.shift_mlp, &mb.one_plus_scale_mlp, &mb.gate_mlp, dim, t, eps);
        timings.record_upload += s_rec.elapsed();
        let s_sub = std::time::Instant::now();
        if block_pipeline() {
            // Drain what is ALREADY on the queue - every earlier block, and
            // the staging behind this block's own weight upload. This block's
            // dispatches are not on it yet, which is exactly why this wait
            // does not wait for them.
            scratch.poll_wait();
            scratch.submit(&[], &s);
            // `submit` only appends to the handle's batch; without this the
            // work would not reach the card until the NEXT block's wait
            // flushed it, which is the opposite of overlapping.
            scratch.flush();
        } else {
            scratch.submit(&[], &s);
            let _ = scratch.read(&x3, 1);
        }
        timings.compute += s_sub.elapsed();
        x3
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }
}

// ---------------------------------------------------------------------------
// Quantized-compute AUDIO+VIDEO block - the audio-extended twin of everything
// above.
//
// It quantizes exactly what the video-only tier quantizes, applied to every
// stream this block has: the four projections (`to_q`/`to_k`/`to_v`/
// `to_out.0`) of all SIX attention modules (`attn1`/`attn2`, `audio_attn1`/
// `audio_attn2`, `audio_to_video_attn`/`video_to_audio_attn`) plus both FFNs'
// two linears each - 28 quantized linears per block against the video-only
// block's 10.
//
// What stays fp32 is `crate::int8::is_never_quantized`'s set, unchanged and
// not extended: `q_norm`/`k_norm` (1-D gains), every `to_gate_logits`
// (`[32, q_dim]` - small, and it is what fixes each gate's own numeric scale),
// and every modulation table (`{audio_,}scale_shift_table`,
// `{audio_,}prompt_scale_shift_table`, `scale_shift_table_a2v_ca_{video,
// audio}`). The AV extension adds no new KIND of tensor to that decision - the
// two `[5, dim]` A<->V tables are modulation tables and are excluded by the
// same `scale_shift_table` substring the video tables are, and the two
// cross-attention modules are ordinary `Attention` modules whose gates are
// therefore excluded by the same `to_gate_logits` substring. So there is one
// keep-fp32 rule in this crate, not a second one for audio.
//
// EVERY attention here is gated, in both streams and in both cross-modal
// directions, and the A<->V directions additionally carry their own `[5, dim]`
// row-4 gate driven by the OTHER modality's scalar sigma. Both survive the
// quantization untouched: the per-head `to_gate_logits` -> `sigmoid` ->
// broadcast -> multiply chain is shared verbatim with [`attention`] (see
// [`attention_q`]'s own doc), and the cross-modal gate is [`gate_row`] against
// a fp32 `[dim]` row, exactly as in [`LtxAvBlock::forward`].
// ---------------------------------------------------------------------------

/// [`AvCrossWeights`]'s quantized twin - both cross-modal `Attention` modules
/// packed, both `[5, dim]` adaLN tables left fp32.
struct QAvCrossWeights {
    a2v: QAttnWeights,
    v2a: QAttnWeights,
    /// `[5, video.dim]`/`[5, audio.dim]` host copies, kept for the same
    /// reason [`QBlockWeights::scale_shift_table`] is: the parity path
    /// combines them with the model-level table on the HOST, the production
    /// path adds them on the CARD, and one copy of the numbers is what stops
    /// the two paths modulating against different tables.
    table_video: Vec<f32>,
    table_audio: Vec<f32>,
}

/// [`QAvCrossWeights`]'s host-only form - see [`CachedQLinear`]'s doc for why
/// the host/device split exists at all.
struct CachedQAvCrossWeights {
    a2v: CachedQAttnWeights,
    v2a: CachedQAttnWeights,
    /// `[5, video.dim]` - rows 0-1 A2V scale/shift, 2-3 V2A scale/shift, 4
    /// this stream's own cross-modal gate (`crate::config`'s doc).
    table_video: Vec<f32>,
    /// `[5, audio.dim]`.
    table_audio: Vec<f32>,
}

impl CachedQAvCrossWeights {
    fn byte_len(&self) -> usize {
        self.a2v.byte_len() + self.v2a.byte_len() + std::mem::size_of_val(self.table_video.as_slice()) + std::mem::size_of_val(self.table_audio.as_slice())
    }
}

/// Resident quantized weights of one AV block.
struct QAvBlockWeights {
    vw: QBlockWeights,
    aw: QBlockWeights,
    avw: QAvCrossWeights,
}

/// [`CachedQBlockWeights`]'s audio+video twin - what an AV
/// [`crate::weightcache::GenerationCache`] slot holds, and the unit
/// [`crate::devres::AvDitSession`] uploads. Composed of the two streams'
/// own already-existing per-stream cached form plus the cross-modal pair, so
/// the audio stream reuses the video stream's quantization code verbatim
/// rather than a transcription of it.
pub struct CachedQAvBlockWeights {
    vw: CachedQBlockWeights,
    aw: CachedQBlockWeights,
    avw: CachedQAvCrossWeights,
}

impl CachedQAvBlockWeights {
    /// Real host bytes this one AV block's cache slot holds - the same
    /// measured (not derived) number [`CachedQBlockWeights::byte_len`]
    /// reports, and what [`cached_av_block_bytes`] is pinned against.
    pub fn byte_len(&self) -> usize {
        self.vw.byte_len() + self.aw.byte_len() + self.avw.byte_len()
    }

    /// Quantize one AV block's weights into the cache's host form.
    pub fn quantize(w: &Tensors, prefix: &str, vcfg: &LtxDitConfig, acfg: &LtxAudioDitConfig, tier: QTier) -> CachedQAvBlockWeights {
        let (vdim, adim) = (vcfg.inner_dim as usize, acfg.inner_dim as usize);
        let gated = vcfg.apply_gated_attention;
        let table_video = tget(w, &format!("{prefix}.scale_shift_table_a2v_ca_video")).to_vec();
        assert_eq!(table_video.len(), 5 * vdim, "{prefix}.scale_shift_table_a2v_ca_video must be [5, video.dim]");
        let table_audio = tget(w, &format!("{prefix}.scale_shift_table_a2v_ca_audio")).to_vec();
        assert_eq!(table_audio.len(), 5 * adim, "{prefix}.scale_shift_table_a2v_ca_audio must be [5, audio.dim]");
        CachedQAvBlockWeights {
            vw: QBlockWeights::quantize_host_stream(w, prefix, "", vdim, gated, tier),
            aw: QBlockWeights::quantize_host_stream(w, prefix, "audio_", adim, gated, tier),
            avw: CachedQAvCrossWeights {
                // A2V's query is VIDEO (`q_dim = vdim`), its keys/values are
                // AUDIO (`kv_dim = adim`), and it runs at AUDIO's head
                // geometry (`inner_dim = adim`) - see [`LtxAvBlock`]'s doc.
                // V2A is the mirror image, and `inner_dim` stays audio's in
                // BOTH directions.
                a2v: QAttnWeights::quantize_host(w, &format!("{prefix}.audio_to_video_attn"), gated, tier, vdim, adim, adim),
                v2a: QAttnWeights::quantize_host(w, &format!("{prefix}.video_to_audio_attn"), gated, tier, adim, vdim, adim),
                table_video,
                table_audio,
            },
        }
    }
}

impl QAvBlockWeights {
    fn from_cached(gpu: &Gpu, c: &CachedQAvBlockWeights) -> QAvBlockWeights {
        QAvBlockWeights {
            vw: QBlockWeights::from_cached(gpu, &c.vw),
            aw: QBlockWeights::from_cached(gpu, &c.aw),
            avw: QAvCrossWeights {
                a2v: QAttnWeights::from_cached(gpu, &c.avw.a2v),
                v2a: QAttnWeights::from_cached(gpu, &c.avw.v2a),
                table_video: c.avw.table_video.clone(),
                table_audio: c.avw.table_audio.clone(),
            },
        }
    }
}

/// [`cached_block_bytes`]'s AV twin: host bytes ONE cached AV block occupies
/// at `tier`, in closed form, for a residency estimate that has to exist
/// BEFORE anything has been loaded.
///
/// Pinned against a really-quantized block by
/// `cached_av_block_bytes_matches_a_real_measured_block`, exactly as the
/// video-only one is - a derivation nobody checks drifts.
pub fn cached_av_block_bytes(vcfg: &LtxDitConfig, acfg: &LtxAudioDitConfig, tier: QTier) -> u64 {
    /// `to_gate_logits`'s fixed row count - the same literal
    /// `crate::dit::push_attn` writes into the tensor manifest.
    const GATE_LOGIT_ROWS: u64 = 32;
    let f32s = |n: u64| n * 4;
    let packed = |out: u64, inp: u64| -> u64 {
        let words = match tier {
            QTier::Int8 => out * inp / 4,
            QTier::Int4 => out * inp / 8,
        };
        words * 4 + f32s(out * inp / model::int8::GROUP as u64)
    };
    let gated = vcfg.apply_gated_attention;
    // One `Attention` at `(q_dim, kv_dim, inner_dim)`: to_q [inner,q]+bias,
    // to_k/to_v [inner,kv]+bias, to_out.0 [q,inner]+bias, two [inner] norms,
    // an optional [32, q_dim] gate + [32] bias.
    let attn = |q: u64, kv: u64, inner: u64| -> u64 {
        packed(inner, q)
            + f32s(inner)
            + 2 * (packed(inner, kv) + f32s(inner))
            + packed(q, inner)
            + f32s(q)
            + 2 * f32s(inner)
            + if gated { f32s(GATE_LOGIT_ROWS * q) + f32s(GATE_LOGIT_ROWS) } else { 0 }
    };
    let vdim = vcfg.inner_dim as u64;
    let adim = acfg.inner_dim as u64;
    let rows9 = vcfg.adaln_rows() as u64;
    // Video stream: two same-width attentions, a bias-FREE FFN, its two
    // tables. Audio stream: the same, with a BIASED FFN (`ff_has_bias` - the
    // real checkpoint carries `audio_ff`'s bias whatever `ff_bias` says).
    let video = 2 * attn(vdim, vdim, vdim) + packed(4 * vdim, vdim) + packed(vdim, 4 * vdim) + f32s(rows9 * vdim) + f32s(2 * vdim);
    let audio = 2 * attn(adim, adim, adim) + packed(4 * adim, adim) + f32s(4 * adim) + packed(adim, 4 * adim) + f32s(adim) + f32s(rows9 * adim) + f32s(2 * adim);
    let cross = attn(vdim, adim, adim) + attn(adim, vdim, adim) + f32s(5 * vdim) + f32s(5 * adim);
    video + audio + cross
}

/// One AV block's CROSS-modal modulation, as device buffers: both directions'
/// `(1+scale, shift)` pair for each operand stream, plus the two single-row
/// gates.
///
/// Same two ways to fill one as [`ModBufs`], and for the same reason:
/// [`Self::upload`] takes the host arithmetic ([`av_scale_shift`]/[`av_gate`])
/// and writes ten vectors to the card; [`Self::derive`] computes them ON the
/// card from the model-level per-token tables (uploaded once per forward) and
/// this block's own resident `[5, dim]` tables. Bit-identical by
/// construction: `adaln_row` reproduces `add_table`'s operand order and
/// `slice_mod`'s `1.0 + x` exactly (that kernel's own header states it), and
/// the gate is one
/// `add2` over the same two `[dim]` rows `av_gate` adds on the host.
struct AvModBufs {
    scale_a2v_v: DeviceBuffer,
    shift_a2v_v: DeviceBuffer,
    scale_a2v_a: DeviceBuffer,
    shift_a2v_a: DeviceBuffer,
    scale_v2a_v: DeviceBuffer,
    shift_v2a_v: DeviceBuffer,
    scale_v2a_a: DeviceBuffer,
    shift_v2a_a: DeviceBuffer,
    gate_a2v: DeviceBuffer,
    gate_v2a: DeviceBuffer,
}

/// This block's own resident AV tables, as device buffers - what
/// [`AvModBufs::derive`] adds the model-level per-token tables to.
struct AvTableBufs {
    /// `[5, video.dim]` whole; `adaln_row` reads rows 0-3 at `NR = 4`, which
    /// is why the row-4 gate needs its own buffer below (a kernel binding is a
    /// whole buffer, never an offset view).
    table_video: DeviceBuffer,
    /// `[5, audio.dim]`.
    table_audio: DeviceBuffer,
    /// `table_video`'s row 4, `[video.dim]`.
    gate_row_video: DeviceBuffer,
    /// `table_audio`'s row 4, `[audio.dim]`.
    gate_row_audio: DeviceBuffer,
}

/// The four model-level AV tables one forward shares across every block, on
/// the device: each stream's `[U, 4*dim]` per-token scale/shift table plus its
/// `[t]` row map (`dit::adaln::RowTable`, one row per DISTINCT timestep), and
/// each direction's SINGLE `[dim]` raw gate row (driven by the CROSS
/// modality's scalar sigma, so there is exactly one row however many tokens
/// there are).
pub struct AvModelTables<'a> {
    pub v_ss: &'a DeviceBuffer,
    pub v_ss_map: &'a DeviceBuffer,
    pub a_ss: &'a DeviceBuffer,
    pub a_ss_map: &'a DeviceBuffer,
    pub a2v_gate: &'a DeviceBuffer,
    pub v2a_gate: &'a DeviceBuffer,
}

impl AvModBufs {
    /// Upload an already-computed host set - the reference definition of the
    /// arithmetic, and what the parity path uses.
    #[allow(clippy::too_many_arguments)]
    fn upload(gpu: &Gpu, av_video_ss_table: &[f32], av_audio_ss_table: &[f32], av_a2v_gate_table: &[f32], av_v2a_gate_table: &[f32], tv: usize, ta: usize, vdim: usize, adim: usize, table_video: &[f32], table_audio: &[f32]) -> AvModBufs {
        let up = |v: &[f32]| -> DeviceBuffer {
            let b = gpu.storage(v.len() as u64);
            wf(gpu, &b, v);
            b
        };
        let (scale_a2v_v, shift_a2v_v) = av_scale_shift(av_video_ss_table, table_video, tv, vdim, 0);
        let (scale_a2v_a, shift_a2v_a) = av_scale_shift(av_audio_ss_table, table_audio, ta, adim, 0);
        let (scale_v2a_v, shift_v2a_v) = av_scale_shift(av_video_ss_table, table_video, tv, vdim, 2);
        let (scale_v2a_a, shift_v2a_a) = av_scale_shift(av_audio_ss_table, table_audio, ta, adim, 2);
        AvModBufs {
            scale_a2v_v: up(&scale_a2v_v),
            shift_a2v_v: up(&shift_a2v_v),
            scale_a2v_a: up(&scale_a2v_a),
            shift_a2v_a: up(&shift_a2v_a),
            scale_v2a_v: up(&scale_v2a_v),
            shift_v2a_v: up(&shift_v2a_v),
            scale_v2a_a: up(&scale_v2a_a),
            shift_v2a_a: up(&shift_v2a_a),
            gate_a2v: up(&av_gate(av_a2v_gate_table, table_video, vdim)),
            gate_v2a: up(&av_gate(av_v2a_gate_table, table_audio, adim)),
        }
    }

    /// Derive the same ten vectors on the device. Records eight `adaln_row`
    /// dispatches (`NR = 4`, rows 0-3, the scale rows carrying `plus_one`) and
    /// two `add2`s into `s`; nothing runs until the caller submits.
    #[allow(clippy::too_many_arguments)]
    fn derive(gpu: &Gpu, s: &mut Vec<Step>, m: &AvModelTables, tbl: &AvTableBufs, tv: u32, vdim: u32, ta: u32, adim: u32) -> AvModBufs {
        // `row0` is the direction's first row: 0 for A2V, 2 for V2A - and the
        // pair is (scale, shift), REVERSED against the base 9-row table (see
        // [`av_scale_shift`]'s doc), which is why `plus_one` sits on the FIRST
        // of the two here and on the SECOND there.
        let mut row = |tab: &DeviceBuffer, map: &DeviceBuffer, table: &DeviceBuffer, t: u32, dim: u32, r: u32, plus_one: bool| -> DeviceBuffer {
            let b = gpu.storage((t * dim) as u64);
            s.push(gpu.step(K_ADALN_ROW, &[tab, table, map, &b], &[t, dim, 4, r, u32::from(plus_one)], t * dim));
            b
        };
        let scale_a2v_v = row(m.v_ss, m.v_ss_map, &tbl.table_video, tv, vdim, 0, true);
        let shift_a2v_v = row(m.v_ss, m.v_ss_map, &tbl.table_video, tv, vdim, 1, false);
        let scale_a2v_a = row(m.a_ss, m.a_ss_map, &tbl.table_audio, ta, adim, 0, true);
        let shift_a2v_a = row(m.a_ss, m.a_ss_map, &tbl.table_audio, ta, adim, 1, false);
        let scale_v2a_v = row(m.v_ss, m.v_ss_map, &tbl.table_video, tv, vdim, 2, true);
        let shift_v2a_v = row(m.v_ss, m.v_ss_map, &tbl.table_video, tv, vdim, 3, false);
        let scale_v2a_a = row(m.a_ss, m.a_ss_map, &tbl.table_audio, ta, adim, 2, true);
        let shift_v2a_a = row(m.a_ss, m.a_ss_map, &tbl.table_audio, ta, adim, 3, false);
        // `av_gate` is `add_table(gate_mlp_out, table5_row4, 1, dim)`, i.e.
        // `table[i] + v[i]` over one row - one elementwise add, with the block
        // table as the first operand exactly as on the host.
        let gate_a2v = gpu.storage(vdim as u64);
        add2(gpu, s, &tbl.gate_row_video, m.a2v_gate, &gate_a2v, vdim);
        let gate_v2a = gpu.storage(adim as u64);
        add2(gpu, s, &tbl.gate_row_audio, m.v2a_gate, &gate_v2a, adim);
        AvModBufs { scale_a2v_v, shift_a2v_v, scale_a2v_a, shift_a2v_a, scale_v2a_v, shift_v2a_v, scale_v2a_a, shift_v2a_a, gate_a2v, gate_v2a }
    }
}

/// Every buffer one AV block forward produces that a caller might read -
/// the two stream outputs plus the eight [`AvBlockTaps`] taps, still on the
/// device. A production forward reads the first two and drops the rest.
struct AvQBody {
    vx3: DeviceBuffer,
    ax3: DeviceBuffer,
    v_attn1: DeviceBuffer,
    v_ca: DeviceBuffer,
    v_ff: DeviceBuffer,
    a_attn1: DeviceBuffer,
    a_ca: DeviceBuffer,
    a_ff: DeviceBuffer,
    a2v: DeviceBuffer,
    v2a: DeviceBuffer,
}

/// Every stream RoPE table set one AV block reads: each stream's own
/// self-attention tables plus each stream's cross-modal (time-only) tables.
#[derive(Clone, Copy)]
pub struct AvRope<'a> {
    pub v_cos: &'a [DeviceBuffer],
    pub v_sin: &'a [DeviceBuffer],
    pub a_cos: &'a [DeviceBuffer],
    pub a_sin: &'a [DeviceBuffer],
    pub v_cross_cos: &'a [DeviceBuffer],
    pub v_cross_sin: &'a [DeviceBuffer],
    pub a_cross_cos: &'a [DeviceBuffer],
    pub a_cross_sin: &'a [DeviceBuffer],
}

/// One AV block's whole op sequence, recorded (not submitted) - the single
/// implementation both [`LtxAvBlockQ::forward`] and
/// [`LtxAvBlockQ::forward_prod`] record, so the parity-tap path and the
/// production path cannot drift apart. Step numbering follows
/// [`LtxAvBlock`]'s doc exactly.
#[allow(clippy::too_many_arguments)]
fn av_block_body_q(
    gpu: &Gpu,
    s: &mut Vec<Step>,
    tier: QTier,
    w: &QAvBlockWeights,
    ones_v: &DeviceBuffer,
    ones_a: &DeviceBuffer,
    vx_buf: &DeviceBuffer,
    ax_buf: &DeviceBuffer,
    vm: &ModBufs,
    am: &ModBufs,
    avm: &AvModBufs,
    v_ctx: &DeviceBuffer,
    v_ctx_len: u32,
    a_ctx: &DeviceBuffer,
    a_ctx_len: u32,
    rope: AvRope,
    vcfg: &LtxDitConfig,
    acfg: &LtxAudioDitConfig,
    tv: u32,
    ta: u32,
) -> AvQBody {
    let (vdim, vheads, vhd) = (vcfg.inner_dim, vcfg.num_heads, vcfg.head_dim());
    let (adim, aheads, ahd) = (acfg.inner_dim, acfg.num_heads, acfg.head_dim());
    // ONE model-level `norm_eps` is shared by every norm in BOTH streams -
    // `BasicAVTransformerBlock`'s single constructor argument, not a
    // per-stream value ([`LtxAvBlock::forward`] makes the same call).
    let eps = vcfg.norm_eps;
    let (vtd, atd) = (tv * vdim, ta * adim);

    // ---- 1-2: each stream's self-attention + text cross-attention, run to
    // completion before the AV step touches either.
    #[rustfmt::skip]
    let (vx1, v_attn1, v_ca) = self_attn_and_text_ca_q(gpu, s, tier, &w.vw, ones_v, vx_buf, vm, v_ctx, v_ctx_len, rope.v_cos, rope.v_sin, vdim, vheads, vhd, tv, eps);
    #[rustfmt::skip]
    let (ax1, a_attn1, a_ca) = self_attn_and_text_ca_q(gpu, s, tier, &w.aw, ones_a, ax_buf, am, a_ctx, a_ctx_len, rope.a_cos, rope.a_sin, adim, aheads, ahd, ta, eps);

    // ---- 3: audio<->video cross-attention. `vx1`/`ax1` are the pre-AV
    // snapshot BOTH directions read.
    let v_tmp1 = gpu.storage(vtd as u64);
    let v_tmp2 = gpu.storage(vtd as u64);
    let a2v_vx_scaled = gpu.storage(vtd as u64);
    ada_zero(gpu, s, ones_v, &vx1, &avm.scale_a2v_v, &avm.shift_a2v_v, &v_tmp1, &v_tmp2, &a2v_vx_scaled, vdim, tv, eps);
    let a_tmp1 = gpu.storage(atd as u64);
    let a_tmp2 = gpu.storage(atd as u64);
    let a2v_ax_scaled = gpu.storage(atd as u64);
    ada_zero(gpu, s, ones_a, &ax1, &avm.scale_a2v_a, &avm.shift_a2v_a, &a_tmp1, &a_tmp2, &a2v_ax_scaled, adim, ta, eps);
    #[rustfmt::skip]
    let a2v = attention_q(gpu, s, tier, &w.avw.a2v, vdim, adim, adim, aheads, ahd, &a2v_vx_scaled, &a2v_ax_scaled, tv, ta, Some((rope.v_cross_cos, rope.v_cross_sin)), Some((rope.a_cross_cos, rope.a_cross_sin)), K_ROPE2D, eps, false);
    let vx2 = gpu.storage(vtd as u64);
    // `rows_per_cond = rows`: ONE gate row, broadcast across every token.
    gate_row(gpu, s, &vx1, &avm.gate_a2v, &a2v, &vx2, tv, vdim, tv);

    let v_tmp3 = gpu.storage(vtd as u64);
    let v_tmp4 = gpu.storage(vtd as u64);
    let v2a_vx_scaled = gpu.storage(vtd as u64);
    ada_zero(gpu, s, ones_v, &vx1, &avm.scale_v2a_v, &avm.shift_v2a_v, &v_tmp3, &v_tmp4, &v2a_vx_scaled, vdim, tv, eps);
    let a_tmp3 = gpu.storage(atd as u64);
    let a_tmp4 = gpu.storage(atd as u64);
    let v2a_ax_scaled = gpu.storage(atd as u64);
    ada_zero(gpu, s, ones_a, &ax1, &avm.scale_v2a_a, &avm.shift_v2a_a, &a_tmp3, &a_tmp4, &v2a_ax_scaled, adim, ta, eps);
    #[rustfmt::skip]
    let v2a = attention_q(gpu, s, tier, &w.avw.v2a, adim, vdim, adim, aheads, ahd, &v2a_ax_scaled, &v2a_vx_scaled, ta, tv, Some((rope.a_cross_cos, rope.a_cross_sin)), Some((rope.v_cross_cos, rope.v_cross_sin)), K_ROPE2D, eps, false);
    let ax2 = gpu.storage(atd as u64);
    gate_row(gpu, s, &ax1, &avm.gate_v2a, &v2a, &ax2, ta, adim, ta);

    // ---- 4: video MLP (on the post-A2V state), then audio MLP.
    let (vx3, v_ff) = mlp_sublayer_q(gpu, s, &w.vw.ff, tier, ones_v, &vx2, &vm.shift_mlp, &vm.one_plus_scale_mlp, &vm.gate_mlp, vdim, tv, eps);
    let (ax3, a_ff) = mlp_sublayer_q(gpu, s, &w.aw.ff, tier, ones_a, &ax2, &am.shift_mlp, &am.one_plus_scale_mlp, &am.gate_mlp, adim, ta, eps);
    AvQBody { vx3, ax3, v_attn1, v_ca, v_ff, a_attn1, a_ca, a_ff, a2v, v2a }
}

/// [`LtxAvBlock`]'s quantized-compute twin: one audio+video block, weights
/// resident in packed int8/int4 form. See this section's module doc for the
/// tier's exact scope (28 quantized linears per block; the keep-fp32 set is
/// `crate::int8::is_never_quantized`'s, unchanged).
pub struct LtxAvBlockQ {
    gpu: Gpu,
    vcfg: LtxDitConfig,
    acfg: LtxAudioDitConfig,
    tier: QTier,
    w: QAvBlockWeights,
    v_ctx_len: u32,
    a_ctx_len: u32,
    ones_v: DeviceBuffer,
    ones_a: DeviceBuffer,
    /// Each stream's own `[9, dim]` `scale_shift_table` on the device - what
    /// [`ModBufs::derive`] adds the model-level adaLN table to, and the reason
    /// a resident AV block needs nothing from the host on a warm step.
    v_sst_buf: DeviceBuffer,
    a_sst_buf: DeviceBuffer,
    /// The two `[5, dim]` A<->V tables (plus their row-4 gate rows split out),
    /// the cross-modal counterpart of the two `sst_buf`s above.
    av: AvTableBufs,
}

impl LtxAvBlockQ {
    /// `weights`/`prefix`: same convention as [`LtxAvBlock::on`].
    #[allow(clippy::too_many_arguments)]
    pub fn on(gpu: Gpu, vcfg: &LtxDitConfig, acfg: &LtxAudioDitConfig, weights: &Tensors, prefix: &str, v_ctx_len: u32, a_ctx_len: u32, tier: QTier) -> LtxAvBlockQ {
        let cached = CachedQAvBlockWeights::quantize(weights, prefix, vcfg, acfg, tier);
        LtxAvBlockQ::on_cached(gpu, vcfg, acfg, &cached, v_ctx_len, a_ctx_len, tier)
    }

    /// [`Self::on`]'s cached twin - no GGUF read and no CPU quantization, only
    /// device uploads of already-quantized host bytes. The constructor
    /// [`crate::devres::AvDitSession`] uses on every (re-)upload.
    #[allow(clippy::too_many_arguments)]
    pub fn on_cached(gpu: Gpu, vcfg: &LtxDitConfig, acfg: &LtxAudioDitConfig, cached: &CachedQAvBlockWeights, v_ctx_len: u32, a_ctx_len: u32, tier: QTier) -> LtxAvBlockQ {
        vcfg.assert_supported();
        let w = QAvBlockWeights::from_cached(&gpu, cached);
        let up = |v: &[f32]| -> DeviceBuffer {
            let b = gpu.storage(v.len() as u64);
            wf(&gpu, &b, v);
            b
        };
        let (vdim, adim) = (vcfg.inner_dim as usize, acfg.inner_dim as usize);
        let ones_v = up(&vec![1.0f32; vdim]);
        let ones_a = up(&vec![1.0f32; adim]);
        let v_sst_buf = up(&w.vw.scale_shift_table);
        let a_sst_buf = up(&w.aw.scale_shift_table);
        let av = AvTableBufs {
            table_video: up(&w.avw.table_video),
            table_audio: up(&w.avw.table_audio),
            gate_row_video: up(&w.avw.table_video[4 * vdim..5 * vdim]),
            gate_row_audio: up(&w.avw.table_audio[4 * adim..5 * adim]),
        };
        LtxAvBlockQ { gpu, vcfg: *vcfg, acfg: *acfg, tier, w, v_ctx_len, a_ctx_len, ones_v, ones_a, v_sst_buf, a_sst_buf, av }
    }

    /// One AV block forward with the full parity tap set - same inputs and
    /// outputs as [`LtxAvBlock::forward`], so a parity test can run the two
    /// against each other with one shared harness.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        vx: &[f32],
        ax: &[f32],
        v_adaln_table: &[f32],
        a_adaln_table: &[f32],
        v_context: &[f32],
        a_context: &[f32],
        rope: AvRope,
        av_video_ss_table: &[f32],
        av_audio_ss_table: &[f32],
        av_a2v_gate_table: &[f32],
        av_v2a_gate_table: &[f32],
        tv: u32,
        ta: u32,
    ) -> (Vec<f32>, Vec<f32>, AvBlockTaps) {
        let gpu = &self.gpu;
        let (vdim, adim) = (self.vcfg.inner_dim, self.acfg.inner_dim);
        assert_eq!(vx.len(), (tv * vdim) as usize);
        assert_eq!(ax.len(), (ta * adim) as usize);
        assert_eq!(v_adaln_table.len(), (tv * 9 * vdim) as usize);
        assert_eq!(a_adaln_table.len(), (ta * 9 * adim) as usize);

        let v_combined = dit::adaln::add_table(v_adaln_table, &self.w.vw.scale_shift_table, tv as usize, 9 * vdim as usize);
        let vm = ModBufs::upload(gpu, &slice_mod(&v_combined, tv as usize, vdim as usize));
        let a_combined = dit::adaln::add_table(a_adaln_table, &self.w.aw.scale_shift_table, ta as usize, 9 * adim as usize);
        let am = ModBufs::upload(gpu, &slice_mod(&a_combined, ta as usize, adim as usize));
        #[rustfmt::skip]
        let avm = AvModBufs::upload(
            gpu, av_video_ss_table, av_audio_ss_table, av_a2v_gate_table, av_v2a_gate_table,
            tv as usize, ta as usize, vdim as usize, adim as usize, &self.w.avw.table_video, &self.w.avw.table_audio,
        );

        let up = |v: &[f32]| -> DeviceBuffer {
            let b = gpu.storage(v.len() as u64);
            wf(gpu, &b, v);
            b
        };
        let (vx_buf, ax_buf) = (up(vx), up(ax));
        let (v_ctx, a_ctx) = (up(v_context), up(a_context));

        let mut s: Vec<Step> = Vec::new();
        #[rustfmt::skip]
        let b = av_block_body_q(gpu, &mut s, self.tier, &self.w, &self.ones_v, &self.ones_a, &vx_buf, &ax_buf, &vm, &am, &avm,
            &v_ctx, self.v_ctx_len, &a_ctx, self.a_ctx_len, rope, &self.vcfg, &self.acfg, tv, ta);
        gpu.submit(&[], &s);
        let (vtd, atd) = ((tv * vdim) as usize, (ta * adim) as usize);
        let taps = AvBlockTaps {
            v_attn1_out: gpu.read(&b.v_attn1, vtd),
            v_attn2_out: gpu.read(&b.v_ca, vtd),
            v_ff_out: gpu.read(&b.v_ff, vtd),
            a_attn1_out: gpu.read(&b.a_attn1, atd),
            a_attn2_out: gpu.read(&b.a_ca, atd),
            a_ff_out: gpu.read(&b.a_ff, atd),
            a2v_out: gpu.read(&b.a2v, vtd),
            v2a_out: gpu.read(&b.v2a, atd),
        };
        (gpu.read(&b.vx3, vtd), gpu.read(&b.ax3, atd), taps)
    }

    /// The PRODUCTION form: every modulation vector derived ON the card from
    /// tables uploaded once per FORWARD, no parity taps, and only the two
    /// streams' activations crossing PCIe.
    ///
    /// The same three savings [`LtxBlockQ::forward_prod`]'s doc lists, at the
    /// AV path's own scale: at the real 22B AV config a block's host-combined
    /// modulation would be `[tv, 9*4096] + [ta, 9*2048] + [tv, 4*4096] +
    /// [ta, 4*2048]`, per block, per forward. Deriving them instead leaves
    /// only the model-level tables (one row per DISTINCT timestep) crossing
    /// the bus, once per forward.
    ///
    /// Takes and returns HOST floats. A stack of blocks should use
    /// [`Self::forward_prod_dev`] instead and keep both streams on the card;
    /// this form stays because a single-block gate wants to compare host
    /// floats and should not have to manage device buffers to do it.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prod(
        &self,
        scratch: &Gpu,
        vx: &[f32],
        ax: &[f32],
        v_adaln_buf: &DeviceBuffer,
        v_adaln_map: &DeviceBuffer,
        a_adaln_buf: &DeviceBuffer,
        a_adaln_map: &DeviceBuffer,
        av: &AvModelTables,
        v_ctx: &DeviceBuffer,
        a_ctx: &DeviceBuffer,
        rope: AvRope,
        tv: u32,
        ta: u32,
        timings: &mut BlockTimings,
    ) -> (Vec<f32>, Vec<f32>) {
        let (vdim, adim) = (self.vcfg.inner_dim, self.acfg.inner_dim);
        assert_eq!(vx.len(), (tv * vdim) as usize);
        assert_eq!(ax.len(), (ta * adim) as usize);
        let s_rec = std::time::Instant::now();
        let vx_buf = scratch.storage((tv * vdim) as u64);
        wf(scratch, &vx_buf, vx);
        let ax_buf = scratch.storage((ta * adim) as u64);
        wf(scratch, &ax_buf, ax);
        timings.record_upload += s_rec.elapsed();
        #[rustfmt::skip]
        let (vx3, ax3) = self.forward_prod_dev(scratch, &vx_buf, &ax_buf, v_adaln_buf, v_adaln_map, a_adaln_buf, a_adaln_map, av, v_ctx, a_ctx, rope, tv, ta, timings);
        let s_back = std::time::Instant::now();
        let vout = scratch.read(&vx3, (tv * vdim) as usize);
        let aout = scratch.read(&ax3, (ta * adim) as usize);
        timings.readback += s_back.elapsed();
        (vout, aout)
    }

    /// [`Self::forward_prod`] with both streams already on the card and
    /// STAYING there - the AV twin of [`LtxBlockQ::forward_prod_dev`], whose
    /// doc has the measurement and the wgpu staging-drain rule this also
    /// obeys. The drain is one blocking one-word read, taken off the VIDEO
    /// output because it is the larger of the two and the wait it forces is
    /// the same wait either way.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prod_dev(
        &self,
        scratch: &Gpu,
        vx_buf: &DeviceBuffer,
        ax_buf: &DeviceBuffer,
        v_adaln_buf: &DeviceBuffer,
        v_adaln_map: &DeviceBuffer,
        a_adaln_buf: &DeviceBuffer,
        a_adaln_map: &DeviceBuffer,
        av: &AvModelTables,
        v_ctx: &DeviceBuffer,
        a_ctx: &DeviceBuffer,
        rope: AvRope,
        tv: u32,
        ta: u32,
        timings: &mut BlockTimings,
    ) -> (DeviceBuffer, DeviceBuffer) {
        let (vdim, adim) = (self.vcfg.inner_dim, self.acfg.inner_dim);
        // The video block's argument, unchanged for two streams: the shape
        // sequence is identical block to block, nothing recorded in this scope
        // is submitted until the previous scope's work has drained, and the two
        // values that outlive the scope (both streams' chained activations) are
        // still held by the caller, so the arena re-allocates their slots
        // instead of aliasing them. See `gpu_core::scratch`.
        let _scope = scratch_pool().then(|| scratch.scratch_scope());
        let s_rec = std::time::Instant::now();
        let mut s: Vec<Step> = Vec::new();
        let vm = ModBufs::derive(scratch, &mut s, v_adaln_buf, v_adaln_map, &self.v_sst_buf, tv, vdim);
        let am = ModBufs::derive(scratch, &mut s, a_adaln_buf, a_adaln_map, &self.a_sst_buf, ta, adim);
        let avm = AvModBufs::derive(scratch, &mut s, av, &self.av, tv, vdim, ta, adim);
        #[rustfmt::skip]
        let b = av_block_body_q(scratch, &mut s, self.tier, &self.w, &self.ones_v, &self.ones_a, vx_buf, ax_buf, &vm, &am, &avm,
            v_ctx, self.v_ctx_len, a_ctx, self.a_ctx_len, rope, &self.vcfg, &self.acfg, tv, ta);
        timings.record_upload += s_rec.elapsed();
        let s_sub = std::time::Instant::now();
        if block_pipeline() {
            // The video block's argument, for two streams: see
            // [`block_pipeline`]. Both chained activations are handed to the
            // caller and both streams' scratch comes from the one arena, and
            // neither is written before the wait below has completed the
            // previous block.
            scratch.poll_wait();
            scratch.submit(&[], &s);
            scratch.flush();
        } else {
            scratch.submit(&[], &s);
            let _ = scratch.read(&b.vx3, 1);
        }
        timings.compute += s_sub.elapsed();
        (b.vx3, b.ax3)
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }
}

/// [`load_block_tensors_from_source`]'s AV twin - exactly one AV block's own
/// tensors off a real GGUF `checkpoint::TensorSource`, never the whole 22B
/// checkpoint's fp32 expansion. Same bound, same reason; only the manifest
/// differs (`crate::dit::av_dit_tensor_manifest`, which carries both streams
/// and the two cross-modal modules).
pub fn load_av_block_tensors_from_source(src: &dyn checkpoint::TensorSource, cfg: &crate::config::LtxAvDitConfig, prefix: &str) -> Tensors {
    let manifest = crate::dit::av_dit_tensor_manifest(cfg);
    let want_prefix = format!("{prefix}.");
    let mut out = Tensors::new();
    for (name, shape) in manifest {
        if !name.starts_with(&want_prefix) {
            continue;
        }
        let mut data = Vec::new();
        let found = src.with_tensor(&name, &mut |d| data = d.to_vec());
        assert!(found, "load_av_block_tensors_from_source: missing {name}");
        let want: usize = shape.iter().product();
        assert_eq!(data.len(), want, "load_av_block_tensors_from_source: {name} wrong length ({} vs {want})", data.len());
        out.insert(name, (shape, data));
    }
    out
}

/// Read exactly one block's real tensors from `src` into a [`Tensors`] map -
/// the bounded loader path [`LtxBlockQ::on`] uses to build straight off a
/// real GGUF `checkpoint::TensorSource` (`crate::gguf_src::LtxvGgufSource`)
/// WITHOUT ever materializing the whole 22B checkpoint's fp32 expansion in
/// host RAM: only this one block's own tensors (a couple hundred MB at
/// most, once packed to int8) are read, each via `with_tensor` - which,
/// per `checkpoint::TensorSource`'s own contract, decodes at most one
/// tensor into a temporary at a time, never the whole source. This is the
/// same bound `model::int8::upload_dequantized`'s doc documents for the
/// general case, applied here at "one block" granularity instead of "one
/// tensor". `prefix`: e.g. `"transformer_blocks.0"`, matching
/// `crate::dit::dit_tensor_manifest`'s own naming.
pub fn load_block_tensors_from_source(src: &dyn checkpoint::TensorSource, cfg: &LtxDitConfig, prefix: &str) -> Tensors {
    let manifest = crate::dit::dit_tensor_manifest(cfg);
    let want_prefix = format!("{prefix}.");
    let mut out = Tensors::new();
    for (name, shape) in manifest {
        if !name.starts_with(&want_prefix) {
            continue;
        }
        let mut data = Vec::new();
        let found = src.with_tensor(&name, &mut |d| data = d.to_vec());
        assert!(found, "load_block_tensors_from_source: missing {name}");
        let want: usize = shape.iter().product();
        assert_eq!(data.len(), want, "load_block_tensors_from_source: {name} wrong length ({} vs {want})", data.len());
        out.insert(name, (shape, data));
    }
    assert!(!out.is_empty(), "load_block_tensors_from_source: prefix {prefix} matched nothing in the manifest");
    out
}

// ---------------------------------------------------------------------------
// Embeddings connector (`video_embeddings_connector`/
// `audio_embeddings_connector`) - `ltx_core.text_encoders.gemma.
// embeddings_connector.Embeddings1DConnector`. NOT part of
// `BasicAVTransformerBlock`/`LTXModel` in the reference at all - confirmed
// against `model_configurator.py`: neither `LTXModelConfigurator` nor
// `LTXVideoOnlyModelConfigurator` ever construct or pass an embeddings-
// connector module into `LTXModel`. In the real pipeline it is a STANDALONE
// preprocessing step the text-encoder wiring runs on the (already
// `caption_projection`'d, for 22B) Gemma-4 embeddings BEFORE they are ever
// handed to the DiT as `context` - which is exactly `caption_proj_before_
// connector`'s meaning (`_build_caption_projections`,
// `model_configurator.py:199-219`: `true` means NO `caption_projection`
// module exists inside the transformer, because that projection already ran
// upstream, before the connector). Implemented HERE (this module, not a
// separate file) because it reuses [`attention`]/[`AttnWeights`]/
// [`FfWeights`]/[`linear`]/[`rmsnorm`]/[`add2`]/[`wf`] directly, verbatim -
// the crate's own established "generalize the existing block code" style
// (see [`LtxAvBlock`] extending [`LtxBlock`]'s pieces) applied one more
// level: a plain pre-LN self-attention-only transformer, not a parallel
// reimplementation of QKV/RMSNorm/RoPE/gating.
//
// Composed by [`crate::dit::LtxDit::forward`]/[`crate::dit::LtxAvDit::
// forward`] as a preprocessing step on `context`, gated by
// `cfg.use_embeddings_connector` (this crate's own flag - the reference has
// none, since the connector lives outside `LTXModel` there; see
// `crate::config::LtxDitConfig::use_embeddings_connector`'s doc).
// ---------------------------------------------------------------------------

/// One connector block's weights - self-attention (`attn1`, optionally
/// gated per `connector_apply_gated_attention`) + a BIASED FFN (unlike the
/// main DiT's `ff_bias=false` video FFN - `embeddings_connector.py`'s
/// `_BasicTransformerBlock1D.__init__` passes `bias=ff_bias` with class
/// default `True`, and neither configurator overrides it for the real
/// checkpoint). No cross-attention, no adaLN table - `_BasicTransformerBlock1D`
/// has neither.
struct ConnectorBlockWeights {
    attn1: AttnWeights,
    ff: FfWeights,
}

impl ConnectorBlockWeights {
    fn upload(gpu: &Gpu, w: &Tensors, prefix: &str, gated: bool) -> ConnectorBlockWeights {
        ConnectorBlockWeights { attn1: AttnWeights::upload(gpu, w, &format!("{prefix}.attn1"), gated), ff: FfWeights::upload(gpu, w, &format!("{prefix}.ff"), true) }
    }
}

/// `video_embeddings_connector`/`audio_embeddings_connector`, weights
/// resident, for a fixed sequence length.
///
/// ## Op sequence (pinned against `embeddings_connector.py`, not assumed)
///
/// 1. **Register substitution** (`_replace_padded_with_learnable_registers`,
///    `embeddings_connector.py:139-152`): `learnable_registers` (`[num_
///    registers, dim]`) is TILED to `[s, dim]` (`registers.repeat(s/num_
///    registers, 1)` - REQUIRES `s % num_registers == 0`, asserted here
///    too) and BLENDS into `hidden_states` per position - `valid[i]==1` (not
///    padded) keeps the real embedding, `valid[i]==0` (padded) is REPLACED
///    by that position's tiled register row. This is a content
///    substitution at the SAME sequence length, not a prepend/append (an
///    earlier reading of this task's own brief assumed prepend-then-drop;
///    the reference source is unambiguous: same-length blend). The mask
///    this substitution leaves for every downstream block's attention is
///    then `torch.zeros_like(...)` - all-zero, i.e. UNMASKED - so once
///    substitution has run, every real config (which always sets
///    `num_learnable_registers`) resolves self-attention masking to
///    identity; this crate has no masking machinery anywhere else either
///    (`LtxAvBlock`'s doc), so [`attention`] is called with no mask
///    argument at all here - reproducing that one real configuration
///    exactly, not approximating a general masked path.
/// 2. **`num_layers` pre-LN blocks** (`_BasicTransformerBlock1D.forward`,
///    `embeddings_connector.py:41-71`): `x = x + attn1(rms_norm(x), pe=freqs)`
///    (self-attention only, own QK-norm, own RoPE, own gate per
///    [`AttnWeights::gate`]), then `x = x + ff(rms_norm(x))` (biased GELU
///    FFN, mult 4). Every norm is the SAME unweighted `rms_norm(x, weight=
///    None, eps=1e-6)` (`ltx_core.utils.rms_norm`'s own default `eps` -
///    numerically identical to `cfg.norm_eps`, which is `1e-6` in every
///    config this crate defines) [`LtxBlock`]'s `ada_zero`/`post_sa`
///    norms already dispatch.
/// 3. **RoPE**: 1-axis, `pe` built from the RAW sequential index `0..s`
///    (`Embeddings1DConnector.forward`, `embeddings_connector.py:172-184`:
///    `indices_grid = arange(s)`, passed to `precompute_freqs_cis` with
///    `use_middle_indices_grid` at its DEFAULT `False` - NOT the main DiT's
///    always-`true` midpoint-of-`[start,end)` convention, confirmed against
///    `rope.py::generate_freqs`: `use_middle_indices_grid=False` and a
///    3-D `indices_grid` (no bounds axis) leaves the raw index untouched).
///    [`ltx_rope_tables`] only implements the midpoint form, so this passes
///    DEGENERATE bounds `[s, s]` per token - `mid = (s+s)/2 = s` exactly
///    reproduces the raw-index formula with no change to that function.
///    `theta` is the class default `10000.0`
///    (`Embeddings1DConnectorConfigurator.from_metadata` never overrides
///    `positional_embedding_theta`) - `cfg.positional_embedding_theta` is
///    `10000.0` in every config this crate defines, so reusing it is exact.
/// 4. **Output norm** (`connector_norm_output`): the SAME unweighted
///    `rms_norm`, unconditionally in the reference
///    (`embeddings_connector.py:189` - no gating flag there at all); this
///    crate's `connector_norm_output` field is honored as a flag regardless
///    (both [`crate::config::LtxDitConfig::tiny_gated`]/[`ltx25_22b`] set it
///    `true`, matching the reference's unconditional behavior).
pub struct EmbeddingsConnector {
    gpu: Gpu,
    dim: u32,
    heads: u32,
    head_dim: u32,
    num_registers: u32,
    norm_output: bool,
    theta: f64,
    max_pos: Vec<u32>,
    eps: f32,
    /// `[num_registers, dim]` host copy - substitution runs on the host
    /// before the device graph is built (this call's `valid` mask is a host
    /// slice too, see [`Self::forward`]).
    registers: Vec<f32>,
    blocks: Vec<ConnectorBlockWeights>,
    ones: DeviceBuffer,
}

impl EmbeddingsConnector {
    /// `weights`/`prefix`: e.g. `"video_embeddings_connector"`. `dim`/
    /// `heads`/`head_dim` are the CONNECTOR's own geometry
    /// (`cfg.connector_inner_dim()`/`connector_num_attention_heads`/
    /// `connector_attention_head_dim`) - independent of the main DiT's,
    /// even though the real checkpoint happens to size them equal per
    /// stream (`crate::config::LtxDitConfig::connector_attention_head_dim`'s
    /// doc).
    #[allow(clippy::too_many_arguments)]
    pub fn on(gpu: Gpu, w: &Tensors, prefix: &str, dim: u32, heads: u32, head_dim: u32, num_layers: u32, num_registers: u32, gated: bool, norm_output: bool, theta: f64, max_pos: &[u32], eps: f32) -> EmbeddingsConnector {
        assert_eq!(heads * head_dim, dim, "embeddings connector: heads*head_dim ({}) must equal dim ({dim})", heads * head_dim);
        let registers = tget(w, &format!("{prefix}.learnable_registers")).to_vec();
        assert_eq!(registers.len(), (num_registers * dim) as usize, "{prefix}.learnable_registers must be [num_registers, dim]");
        let blocks = (0..num_layers).map(|l| ConnectorBlockWeights::upload(&gpu, w, &format!("{prefix}.transformer_1d_blocks.{l}"), gated)).collect();
        let ones = gpu.storage(dim as u64);
        wf(&gpu, &ones, &vec![1.0f32; dim as usize]);
        EmbeddingsConnector { gpu, dim, heads, head_dim, num_registers, norm_output, theta, max_pos: max_pos.to_vec(), eps, registers, blocks, ones }
    }

    /// `hidden`: `[s, dim]` raw (already `caption_projection`'d, per
    /// `caption_proj_before_connector`) input embeddings. `valid`: `[s]`,
    /// `1.0` keeps this position's real embedding, `0.0` substitutes the
    /// tiled register row at this position (this struct's doc, step 1) -
    /// pass all-`1.0` for "no padding" (a real configuration, see
    /// `crate::config::LtxDitConfig::use_embeddings_connector`'s doc).
    /// Returns `[s, dim]`.
    pub fn forward(&self, hidden: &[f32], valid: &[f32], s: u32) -> Vec<f32> {
        let dim = self.dim;
        assert_eq!(hidden.len(), (s * dim) as usize, "embeddings connector: hidden must be [s, dim]");
        assert_eq!(valid.len(), s as usize, "embeddings connector: valid must be [s]");
        assert_eq!(s % self.num_registers, 0, "embeddings connector: seq_len {s} must be a multiple of num_registers {}", self.num_registers);

        // ---- step 1: register substitution (host) --------------------------
        let mut x = hidden.to_vec();
        for (si, &v) in valid.iter().enumerate() {
            if v <= 0.0 {
                let reg_row = si % self.num_registers as usize;
                let (rs, re) = (reg_row * dim as usize, reg_row * dim as usize + dim as usize);
                let (xs, xe) = (si * dim as usize, si * dim as usize + dim as usize);
                x[xs..xe].copy_from_slice(&self.registers[rs..re]);
            }
        }

        // ---- step 3: RoPE table (raw sequential index, see this struct's
        // doc) -----------------------------------------------------------
        let mut positions = vec![0f32; s as usize * 2];
        for si in 0..s as usize {
            positions[si * 2] = si as f32;
            positions[si * 2 + 1] = si as f32;
        }
        let rope = ltx_rope_tables(dim, self.heads, self.theta, &self.max_pos, &positions, s as usize);
        let gpu = &self.gpu;
        let (cos_bufs, sin_bufs) = upload_rope_tables(gpu, &rope);

        let mut ops: Vec<Step> = Vec::new();
        let td = s * dim;
        let x_buf = gpu.storage(td as u64);
        wf(gpu, &x_buf, &x);

        // ---- step 2: num_layers pre-LN blocks -------------------------------
        let mut cur = x_buf;
        for blk in &self.blocks {
            let normed = gpu.storage(td as u64);
            rmsnorm(gpu, &mut ops, &cur, &self.ones, &normed, dim, s, self.eps);
            let attn_out = attention(gpu, &mut ops, &blk.attn1, dim, dim, dim, self.heads, self.head_dim, &normed, &normed, s, s, Some((&cos_bufs, &sin_bufs)), Some((&cos_bufs, &sin_bufs)), K_ROPE2D, self.eps, true);
            let x2 = gpu.storage(td as u64);
            add2(gpu, &mut ops, &attn_out, &cur, &x2, td);

            let normed2 = gpu.storage(td as u64);
            rmsnorm(gpu, &mut ops, &x2, &self.ones, &normed2, dim, s, self.eps);
            let ff_dim = dim * 4;
            let h_pre = gpu.storage((s * ff_dim) as u64);
            linear(gpu, &mut ops, &normed2, &blk.ff.w1, blk.ff.b1.as_ref(), &h_pre, s, dim, ff_dim);
            let h_act = gpu.storage((s * ff_dim) as u64);
            ops.push(gpu.step(K_GELU, &[&h_pre, &h_act], &[s * ff_dim], s * ff_dim));
            let ff_out = gpu.storage(td as u64);
            linear(gpu, &mut ops, &h_act, &blk.ff.w2, blk.ff.b2.as_ref(), &ff_out, s, ff_dim, dim);
            let x3 = gpu.storage(td as u64);
            add2(gpu, &mut ops, &ff_out, &x2, &x3, td);
            cur = x3;
        }

        // ---- step 4: output norm --------------------------------------------
        let final_buf = if self.norm_output {
            let out = gpu.storage(td as u64);
            rmsnorm(gpu, &mut ops, &cur, &self.ones, &out, dim, s, self.eps);
            out
        } else {
            cur
        };

        gpu.submit(&[], &ops);
        gpu.read(&final_buf, td as usize)
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }
}

// ---------------------------------------------------------------------------
// The self-attention gate: flash path vs the exact materialized reference, in
// ONE harness, plus a host oracle for both - two kernels agreeing cannot tell
// you WHICH of them is right, so a third, independent implementation is what
// makes the comparison mean something.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Softmax attention on the host in f64, from the SAME `[nq, inner_dim]`
    /// q and `[nk, inner_dim]` k/v the device sees - the third opinion neither
    /// device arm can influence. Deliberately the textbook max-subtract form,
    /// not any device kernel's schedule. `nq == nk` is self-attention; the two
    /// differing is cross-attention, and the SAME expression covers both,
    /// which is exactly why a wrong `nq`/`nk` swap in a kernel cannot hide
    /// here.
    fn host_attention(q: &[f32], k: &[f32], v: &[f32], heads: usize, head_dim: usize, nq: usize, nk: usize) -> Vec<f32> {
        let inner = heads * head_dim;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut out = vec![0f32; nq * inner];
        let mut row = vec![0f64; nk];
        for h in 0..heads {
            for i in 0..nq {
                let mut m = f64::NEG_INFINITY;
                for (j, r) in row.iter_mut().enumerate() {
                    let mut d = 0f64;
                    for c in 0..head_dim {
                        d += q[i * inner + h * head_dim + c] as f64 * k[j * inner + h * head_dim + c] as f64;
                    }
                    *r = d * scale;
                    m = m.max(*r);
                }
                let mut sum = 0f64;
                for r in row.iter_mut() {
                    *r = (*r - m).exp();
                    sum += *r;
                }
                for c in 0..head_dim {
                    let mut acc = 0f64;
                    for (j, &r) in row.iter().enumerate() {
                        acc += r * v[j * inner + h * head_dim + c] as f64;
                    }
                    out[i * inner + h * head_dim + c] = (acc / sum) as f32;
                }
            }
        }
        out
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (x, y) in a.iter().zip(b) {
            d += *x as f64 * *y as f64;
            na += *x as f64 * *x as f64;
            nb += *y as f64 * *y as f64;
        }
        let den = na.sqrt() * nb.sqrt();
        if den <= 0.0 {
            0.0
        } else {
            d / den
        }
    }

    /// `‖a - b‖₂ / ‖b‖₂`. The metric cosine cannot give: cosine measures only
    /// direction, so a result that is uniformly a few percent too large is
    /// invisible to it and obvious here.
    fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(a.len(), b.len());
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            num += (x as f64 - y as f64).powi(2);
            den += (y as f64).powi(2);
        }
        if den <= 0.0 {
            0.0
        } else {
            (num / den).sqrt()
        }
    }

    /// Run ONE attention in its own submit and read the context back.
    /// `fused` picks the arm DIRECTLY rather than by an input that happens to
    /// route there: `true` is whatever [`attn_context`] selects for this call
    /// (the fused kernel on a capable device), `false` is
    /// [`attn_context_materialized`], the reference trio, unconditionally.
    /// Steering the reference arm by passing `self_attn = false` was safe only
    /// while cross-attention had no fused kernel of its own; now that it does,
    /// that would compare one fused kernel against another and never touch the
    /// reference at all.
    #[allow(clippy::too_many_arguments)]
    fn run_ctx(gpu: &Gpu, q: &[f32], k: &[f32], v: &[f32], heads: u32, head_dim: u32, nq: u32, nk: u32, self_attn: bool, fused: bool) -> Vec<f32> {
        let inner = heads * head_dim;
        let up = |data: &[f32]| {
            let b = gpu.storage(data.len() as u64);
            wf(gpu, &b, data);
            b
        };
        let (qb, kb, vb) = (up(q), up(k), up(v));
        let mut s: Vec<Step> = Vec::new();
        let ctx = if fused {
            attn_context(gpu, &mut s, &qb, &kb, &vb, heads, head_dim, inner, nq, nk, self_attn)
        } else {
            let c = gpu.storage((nq * inner) as u64);
            attn_context_materialized(gpu, &mut s, &qb, &kb, &vb, &c, heads, head_dim, inner, nq, nk);
            c
        };
        gpu.submit(&[], &s);
        gpu.read(&ctx, (nq * inner) as usize)
    }

    /// The gate. For each shape: the flash arm, the exact materialized
    /// reference arm, and the host f64 oracle must all agree.
    ///
    /// `max_abs` is asserted, not only cosine: cosine is scale-invariant, so a
    /// result that is uniformly a few percent too large scores 1.000000000
    /// against the truth. That is precisely the failure a wrong `head_dim`
    /// (hence a wrong `1/sqrt(hd)` score scale) or a dropped online-softmax
    /// renormalisation produces, so cosine alone would wave both through.
    ///
    /// Shapes cover the tail (`t` not a multiple of the 128-row query tile the
    /// top rung owns), the two real parity fixtures' own geometries
    /// (`inner_dim` 64/heads 4 and `inner_dim` 24/heads 3), and the real
    /// checkpoint's `head_dim = 128`.
    #[test]
    fn flash_self_attention_matches_the_materialized_reference_and_a_host_oracle() {
        let gpu = Gpu::open(None, &KERNELS);
        if !gpu.caps().workgroup_reductions {
            // The CPU-JIT fallback branch: both arms ARE the reference trio,
            // so there is nothing to A/B here - `flash_self_attn` returned
            // false and this test would compare the reference against itself.
            // Asserting that is what makes the skip honest rather than silent.
            assert!(!flash_self_attn(&gpu, true, 128), "no workgroup reductions, yet the flash path was selected");
            eprintln!("device has no workgroup reductions - flash path not selected, reference-only (this IS the CPU fallback branch)");
            return;
        }
        assert!(flash_self_attn(&gpu, true, 128), "device reports workgroup reductions but the flash path was not selected");
        assert!(!flash_self_attn(&gpu, false, 128), "cross-attention must never take the flash path");
        assert!(!flash_self_attn(&gpu, true, 129), "head_dim > 128 must never take the flash path");

        let mut rng = data::rng::Lcg::new(0x17ac_2f01);
        let mut randn = |n: usize| -> Vec<f32> { (0..n).map(|_| rng.unit() * 2.0 - 1.0).collect() };

        for &(heads, head_dim, n) in &[(3u32, 64u32, 200u32), (2, 128, 300), (4, 16, 8), (3, 8, 37), (1, 128, 129)] {
            let inner = (heads * head_dim) as usize;
            let len = n as usize * inner;
            let (q, k, v) = (randn(len), randn(len), randn(len));

            let flash = run_ctx(&gpu, &q, &k, &v, heads, head_dim, n, n, true, true);
            let refr = run_ctx(&gpu, &q, &k, &v, heads, head_dim, n, n, true, false);
            let host = host_attention(&q, &k, &v, heads as usize, head_dim as usize, n as usize, n as usize);

            let (fr, fh, rh) = (max_abs(&flash, &refr), max_abs(&flash, &host), max_abs(&refr, &host));
            let cos = cosine(&flash, &host);
            eprintln!("heads={heads} head_dim={head_dim} t={n}: max|flash-ref|={fr:.3e} max|flash-host|={fh:.3e} max|ref-host|={rh:.3e} cos(flash,host)={cos:.12}");
            assert!(flash.iter().all(|x| x.is_finite()), "flash output has non-finite values at heads={heads} head_dim={head_dim} t={n}");
            // 1e-5 absolute on values whose magnitude is O(1): ~2 ulp of f32
            // headroom over the largest disagreement any of these shapes has
            // actually produced, and three orders of magnitude tighter than a
            // single wrong offset/stride/head_dim moves the result.
            assert!(fr < 1e-5, "flash vs materialized reference max_abs {fr:.3e} at heads={heads} head_dim={head_dim} t={n}");
            assert!(fh < 1e-5, "flash vs host oracle max_abs {fh:.3e} at heads={heads} head_dim={head_dim} t={n}");
            assert!(rh < 1e-5, "reference vs host oracle max_abs {rh:.3e} at heads={heads} head_dim={head_dim} t={n}");
            assert!(cos > 0.999999999, "flash vs host oracle cosine {cos:.12} at heads={heads} head_dim={head_dim} t={n}");
        }
    }

    /// The long-sequence accuracy case, with an ANALYTIC ground truth instead
    /// of a second float pipeline: when every one of the `t` token rows is
    /// identical, every score in a head is the same number, the softmax is
    /// exactly uniform, and the context is therefore *exactly* `v` - no
    /// oracle needed, and no tolerance to argue about.
    ///
    /// This is not a contrived shape. It is what `ltxv_bench streamed`'s
    /// all-zero latent produces after patchify + bias, and it is the WORST
    /// case for summation order: 3520 equal positive terms accumulated one
    /// after another drift by `O(t · eps)`, and they drift the same way in
    /// every row rather than cancelling as random data does. Random operands
    /// (the test above) cannot see this at all - they report ~1e-7 for both
    /// arms.
    ///
    /// The assertion is the ORDERING, not just a bound: the fused kernel's
    /// blocked online accumulation must be at least as close to the exact
    /// answer as the materialized `attn_apply_cross`'s per-thread sequential
    /// sum over all `t` keys. Measured on a P40 at `t = 3520`, `head_dim
    /// = 128`: flash 3.0e-6 vs the materialized trio's 5.1e-5. Replacing the
    /// trio here therefore made this model's self-attention MORE accurate,
    /// which is also why the real-checkpoint int8 bench's output statistics
    /// move slightly - the drift being removed is the old path's.
    #[test]
    fn flash_self_attention_beats_the_materialized_trio_on_long_sequence_accuracy() {
        let gpu = Gpu::open(None, &KERNELS);
        if !gpu.caps().workgroup_reductions {
            eprintln!("device has no workgroup reductions - flash path not selected");
            return;
        }
        let (heads, head_dim, n) = (2u32, 128u32, 3520u32);
        // Without this the `fe <= re` ordering below is vacuously true if the
        // fused path ever silently falls back - both arms would be the trio.
        assert!(flash_self_attn(&gpu, true, head_dim), "flash path not selected - the comparison below would be the trio against itself");
        let inner = (heads * head_dim) as usize;
        let mut rng = data::rng::Lcg::new(0x51e7_11d3);
        let row: Vec<f32> = (0..inner).map(|_| rng.unit() * 2.0 - 1.0).collect();
        let rep: Vec<f32> = row.iter().cycle().take(n as usize * inner).copied().collect();

        let flash = run_ctx(&gpu, &rep, &rep, &rep, heads, head_dim, n, n, true, true);
        let refr = run_ctx(&gpu, &rep, &rep, &rep, heads, head_dim, n, n, true, false);
        let (fe, re) = (max_abs(&flash, &rep), max_abs(&refr, &rep));
        eprintln!("identical rows, t={n}: max|flash-exact|={fe:.3e} max|materialized-exact|={re:.3e}");
        assert!(fe <= re, "the fused path must not be LESS accurate than the trio it replaces (flash {fe:.3e} vs materialized {re:.3e})");
        assert!(fe < 1e-5, "flash vs exact uniform-attention answer max_abs {fe:.3e} at t={n}");
    }

    /// The CROSS-attention gate, the sibling of the self-attention one above:
    /// the fused `flash_attn_cross_reg2` arm, the materialized trio, and the
    /// host f64 oracle must all agree, over shapes where `nq` and `nk` are
    /// genuinely unequal.
    ///
    /// Both `max_abs` AND relative L2 are asserted, never cosine alone: cosine
    /// is scale-invariant, so a result uniformly a few percent large scores
    /// 1.000000000 against the truth - which is exactly what a wrong
    /// `head_dim` (hence a wrong `1/sqrt(hd)`) or a dropped online-softmax
    /// renormalisation produces.
    ///
    /// The shape ladder is chosen so that no single index confusion survives
    /// it:
    ///
    /// * `nq > nk` AND `nq < nk` both appear, so a kernel that swapped the two
    ///   lengths cannot pass by symmetry (it would read out of range or
    ///   average the wrong number of keys).
    /// * `nq` both a multiple and NOT a multiple of the 128-row query tile,
    ///   and `nk` both a multiple and not a multiple of the 16-row key tile,
    ///   so the two independent tail paths are each exercised.
    /// * `head_dim` at the real checkpoint's 128 (a full tile) and well under
    ///   it (a zero-filled tile), so the padding guard is exercised too.
    /// * `nk` smaller than one key tile, where every tile is a tail.
    #[test]
    fn flash_cross_attention_matches_the_materialized_reference_and_a_host_oracle() {
        let gpu = Gpu::open(None, &KERNELS);
        if !model::block::flash_cross_supported(&gpu.caps()) {
            // The fallback branch: `attn_context` routes cross-attention to
            // the trio, so both arms below WOULD be the reference and the
            // comparison would be vacuous. Asserting the routing is what makes
            // this skip honest rather than silent.
            assert!(!flash_cross_attn(&gpu, false, 128), "device cannot run the fused cross kernel, yet it was selected");
            eprintln!("device cannot run flash_attn_cross_reg2 - cross-attention is reference-only here (this IS the fallback branch)");
            return;
        }
        assert!(flash_cross_attn(&gpu, false, 128), "device supports the fused cross kernel but it was not selected");
        assert!(!flash_cross_attn(&gpu, true, 128), "self-attention must never take the CROSS path");
        assert!(!flash_cross_attn(&gpu, false, 129), "head_dim > 128 must never take the flash path");

        let mut rng = data::rng::Lcg::new(0x3b91_c40d);
        let mut randn = |n: usize| -> Vec<f32> { (0..n).map(|_| rng.unit() * 2.0 - 1.0).collect() };

        for &(heads, head_dim, nq, nk) in
            &[(32u32, 128u32, 300u32, 128u32), (2, 128, 256, 64), (3, 64, 129, 200), (4, 16, 37, 5), (1, 128, 128, 1024), (3, 8, 200, 17)]
        {
            let inner = (heads * head_dim) as usize;
            let (q, k, v) = (randn(nq as usize * inner), randn(nk as usize * inner), randn(nk as usize * inner));

            let flash = run_ctx(&gpu, &q, &k, &v, heads, head_dim, nq, nk, false, true);
            let refr = run_ctx(&gpu, &q, &k, &v, heads, head_dim, nq, nk, false, false);
            let host = host_attention(&q, &k, &v, heads as usize, head_dim as usize, nq as usize, nk as usize);

            let (fr, fh, rh) = (max_abs(&flash, &refr), max_abs(&flash, &host), max_abs(&refr, &host));
            let (cos, l2) = (cosine(&flash, &host), rel_l2(&flash, &host));
            eprintln!(
                "heads={heads} head_dim={head_dim} nq={nq} nk={nk}: max|flash-ref|={fr:.3e} max|flash-host|={fh:.3e} max|ref-host|={rh:.3e} cos(flash,host)={cos:.12} rel_l2(flash,host)={l2:.3e}"
            );
            assert!(flash.iter().all(|x| x.is_finite()), "fused cross output has non-finite values at heads={heads} head_dim={head_dim} nq={nq} nk={nk}");
            // 1e-5 absolute on values whose magnitude is O(1): ~2 ulp of f32
            // headroom over the largest disagreement any of these shapes has
            // actually produced, and orders of magnitude tighter than a single
            // wrong offset/stride/length moves the result.
            assert!(fr < 1e-5, "fused cross vs materialized reference max_abs {fr:.3e} at heads={heads} head_dim={head_dim} nq={nq} nk={nk}");
            assert!(fh < 1e-5, "fused cross vs host oracle max_abs {fh:.3e} at heads={heads} head_dim={head_dim} nq={nq} nk={nk}");
            assert!(rh < 1e-5, "reference vs host oracle max_abs {rh:.3e} at heads={heads} head_dim={head_dim} nq={nq} nk={nk}");
            assert!(cos > 0.999999999, "fused cross vs host oracle cosine {cos:.12} at heads={heads} head_dim={head_dim} nq={nq} nk={nk}");
            assert!(l2 < 1e-5, "fused cross vs host oracle rel_l2 {l2:.3e} at heads={heads} head_dim={head_dim} nq={nq} nk={nk}");
        }
    }

    /// The V-INDEX gate, which the random test above cannot give on its own.
    ///
    /// With random q/k/v every wrong index produces a wrong number, but the
    /// magnitude of "wrong" is data-dependent and a reader cannot tell a
    /// transposed key axis from a mis-scaled one. Here the answer is
    /// ANALYTIC: `q` is zero, so every score in every head is exactly 0, the
    /// softmax is exactly uniform over all `nk` keys, and the context is
    /// therefore exactly the per-channel MEAN of `v` - the same vector for
    /// every query row. `v`'s rows are made deliberately unequal (a per-row
    /// ramp), so a kernel that read `v[j]` at the wrong stride, folded the
    /// wrong number of keys in, or indexed `v` by the query row instead of the
    /// key row lands somewhere else entirely.
    #[test]
    fn fused_cross_attention_averages_the_right_v_rows_under_a_uniform_softmax() {
        let gpu = Gpu::open(None, &KERNELS);
        if !model::block::flash_cross_supported(&gpu.caps()) {
            eprintln!("device cannot run flash_attn_cross_reg2 - nothing to check here");
            return;
        }
        let (heads, head_dim, nq, nk) = (4u32, 128u32, 300u32, 96u32);
        let inner = (heads * head_dim) as usize;
        let q = vec![0f32; nq as usize * inner];
        let k: Vec<f32> = (0..nk as usize * inner).map(|i| ((i % 13) as f32 - 6.0) * 0.25).collect();
        let v: Vec<f32> = (0..nk as usize * inner).map(|i| (i / inner) as f32 + (i % inner) as f32 * 0.001).collect();

        let got = run_ctx(&gpu, &q, &k, &v, heads, head_dim, nq, nk, false, true);
        let mut want = vec![0f32; inner];
        for c in 0..inner {
            let mut acc = 0f64;
            for j in 0..nk as usize {
                acc += v[j * inner + c] as f64;
            }
            want[c] = (acc / nk as f64) as f32;
        }
        let mut worst = 0f32;
        for i in 0..nq as usize {
            worst = worst.max(max_abs(&got[i * inner..(i + 1) * inner], &want));
        }
        eprintln!("uniform-softmax v-mean: max|got-exact| = {worst:.3e} over {nq} query rows");
        // v's entries run to ~nk here, so the tolerance is relative to that
        // magnitude rather than to O(1) values.
        assert!(worst < 1e-3, "fused cross attention did not average v's rows: max_abs {worst:.3e}");
    }
}
