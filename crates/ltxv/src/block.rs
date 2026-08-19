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
//! T`, `t_enc == context_len`) dispatch the SAME generic trio
//! (`attn_scores_cross` / `attn_softmax_cross` / `attn_apply_cross`,
//! `crates/model/src/block.rs`'s `push_cross`/Wan's non-flash path uses the
//! same trio for cross-attention) rather than a self-attention-specific
//! kernel: the trio's `q_stride`/`kv_stride`/`q_off`/`k_off`/`v_off` are
//! plain parameters, not hardcoded to a fused-QKV layout, so passing
//! separate `[T, dim]` Q/K/V buffers at `stride=dim, off=0` for BOTH
//! self- and cross-attention is exactly what the kernel already supports -
//! no new kernel, no fused-QKV packing needed at this token count. Query-
//! chunking (`model::block::chunked_bidir_fwd`'s reason to exist at Wan's
//! 30k-token scale) is not needed here: the whole `[heads, T, T]` (or
//! `[heads, T, context_len]`) score slab is at most `4*8*8*4 = 1024` bytes.
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
// Audio<->video extension (M6, second half): `LtxAvBlock` below adds the
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
const K_ATTN_SCORES: usize = 7;
const K_ATTN_SOFTMAX: usize = 8;
const K_ATTN_APPLY: usize = 9;
const K_ROPE2D: usize = 10;
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
const K_SIGMOID: usize = 11;

/// Every kernel this block dispatches - all pre-existing, all at their
/// documented general contract (see this module's doc for why no new kernel
/// was needed anywhere in the block).
pub const KERNELS: [(&str, &str); 12] = [
    ("matmul", kernels::MATMUL),
    ("bias_add", kernels::BIAS_ADD),
    ("rmsnorm_eps", kernels::RMSNORM_EPS),
    ("gelu", kernels::GELU),
    ("mul", kernels::MUL),
    ("add2", kernels::ADD2),
    ("gate_row", kernels::GATE_ROW),
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
    ("rope2d", kernels::ROPE2D),
    ("sigmoid", kernels::SIGMOID),
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
            // Both streams' tiny-golden FFN is bias-free (`ff_bias`/
            // `audio_ff_bias=False` in `tools/goldens/ltxv_av_dit_dump_
            // reference.py`'s `TINY_CONFIG` - unlike the REAL checkpoint's
            // `audio_ff`, which `dit.rs::push_ff`'s doc notes carries bias
            // regardless of config; real-weight audio FFN bias import is
            // untouched by this milestone, same as before).
            ff: FfWeights::upload(gpu, w, &format!("{prefix}.{stream}ff"), false),
            scale_shift_table: sst,
            prompt_scale_shift_table: pst,
        }
    }
}

fn wf(gpu: &Gpu, buf: &DeviceBuffer, data: &[f32]) {
    gpu.write_f32(buf, data);
}

/// `out = x @ Wᵀ (+ b)`, `x: [m,k]`, `w: [n,k]`, `out: [m,n]`.
fn linear(gpu: &Gpu, s: &mut Vec<Step>, x: &DeviceBuffer, w: &DeviceBuffer, b: Option<&DeviceBuffer>, out: &DeviceBuffer, m: u32, k: u32, n: u32) {
    s.push(gpu.step(K_MATMUL, &[x, w, out], &[m, k, n], m * n));
    if let Some(b) = b {
        s.push(gpu.step(K_BIAS_ADD, &[out, b], &[m, n], m * n));
    }
}

/// RMSNorm over the full row width (`dim`, never per-head) - `w` is either a
/// learnable gain (QK-norm) or an all-ones buffer (the "no learnable gain"
/// `ada_zero_function`/`post_sa_function` norms - see this module's doc).
fn rmsnorm(gpu: &Gpu, s: &mut Vec<Step>, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, dim: u32, rows: u32, eps: f32) {
    s.push(gpu.step(K_RMSNORM_EPS, &[x, w, out], &[dim, rows, f(eps)], rows));
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

    let scores = gpu.storage((heads * nq * nk) as u64);
    let probs = gpu.storage((heads * nq * nk) as u64);
    let ctx = gpu.storage((nq * inner_dim) as u64);
    s.push(gpu.step(K_ATTN_SCORES, &[&q, &k, &scores], &[1, heads, nq, nk, head_dim, inner_dim, inner_dim, 0, 0], heads * nq * nk));
    s.push(gpu.step(K_ATTN_SOFTMAX, &[&scores, &probs], &[1, heads, nq, nk], heads * nq));
    s.push(gpu.step(K_ATTN_APPLY, &[&probs, &v, &ctx], &[1, heads, nq, nk, head_dim, inner_dim, 0, inner_dim], heads * nq * head_dim));

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
    let attn1_out = attention(gpu, s, &w.attn1, dim, dim, dim, heads, head_dim, &norm_x, &norm_x, t, t, Some((cos_bufs, sin_bufs)), Some((cos_bufs, sin_bufs)), K_ROPE2D, eps);
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

    let ca_raw = attention(gpu, s, &w.attn2, dim, dim, dim, heads, head_dim, &attn_input, &enc_hidden, t, ctx_len, None, None, K_ROPE2D, eps);
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

        let a2v_out = attention(gpu, &mut s, &self.avw.a2v, vdim, adim, adim, aheads, ahd, &a2v_vx_scaled, &a2v_ax_scaled, tv, ta, Some((v_cross_cos, v_cross_sin)), Some((a_cross_cos, a_cross_sin)), K_ROPE2D, eps);
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

        let v2a_out = attention(gpu, &mut s, &self.avw.v2a, adim, vdim, adim, aheads, ahd, &v2a_ax_scaled, &v2a_vx_scaled, ta, tv, Some((a_cross_cos, a_cross_sin)), Some((v_cross_cos, v_cross_sin)), K_ROPE2D, eps);
        let ax2 = gpu.storage(atd as u64);
        gate_row(gpu, &mut s, &ax1, &gate_v2a_buf, &v2a_out, &ax2, ta, adim, ta);

        // ---- 4: MLPs, video then audio (this struct's doc, step 4) --------
        let (vx3, v_ff_out) = mlp_sublayer(gpu, &mut s, &self.vw.ff, &self.ones_v, &vx2, &vm.shift_mlp, &vm.one_plus_scale_mlp, &vm.gate_mlp, vdim, tv, eps);
        let (ax3, a_ff_out) = mlp_sublayer(gpu, &mut s, &self.aw.ff, &self.ones_a, &ax2, &am.shift_mlp, &am.one_plus_scale_mlp, &am.gate_mlp, adim, ta, eps);

        gpu.submit(&[], &s);
        let vx_out = gpu.read(&vx3, vtd as usize);
        let ax_out = gpu.read(&ax3, atd as usize);
        let taps = AvBlockTaps {
            v_attn1_out: gpu.read(&v_attn1_out, vtd as usize),
            v_attn2_out: gpu.read(&v_ca_raw, vtd as usize),
            v_ff_out: gpu.read(&v_ff_out, vtd as usize),
            a_attn1_out: gpu.read(&a_attn1_out, atd as usize),
            a_attn2_out: gpu.read(&a_ca_raw, atd as usize),
            a_ff_out: gpu.read(&a_ff_out, atd as usize),
            a2v_out: gpu.read(&a2v_out, vtd as usize),
            v2a_out: gpu.read(&v2a_out, atd as usize),
        };
        (vx_out, ax_out, taps)
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
        for si in 0..s as usize {
            if valid[si] <= 0.0 {
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
            let attn_out = attention(gpu, &mut ops, &blk.attn1, dim, dim, dim, self.heads, self.head_dim, &normed, &normed, s, s, Some((&cos_bufs, &sin_bufs)), Some((&cos_bufs, &sin_bufs)), K_ROPE2D, self.eps);
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
