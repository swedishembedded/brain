// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One `Gemma4UnifiedTextDecoderLayer`'s forward, as a recorded brain kernel
//! graph - a much simpler op sequence than a diffusion-transformer block (no
//! adaLN modulation, no cross-attention, no per-token gate - see this
//! module's doc on `forward` for the exact order, pinned against source):
//!
//! ```text
//! residual = x
//! x = input_layernorm(x)
//! x = self_attn(x)                         // see `attention()`'s doc
//! x = post_attention_layernorm(x)
//! x = residual + x
//!
//! residual = x
//! x = pre_feedforward_layernorm(x)
//! x = down_proj(gelu_tanh(gate_proj(x)) * up_proj(x))
//! x = post_feedforward_layernorm(x)
//! x = residual + x
//!
//! x = x * layer_scalar                     // a [1]-shaped buffer, real
//!                                           // checkpoints may hold != 1.0
//! ```
//!
//! Every RMSNorm here (`input_layernorm`/`post_attention_layernorm`/
//! `pre_feedforward_layernorm`/`post_feedforward_layernorm`) is the LEARNABLE
//! `Gemma4RMSNorm(hidden_size, eps)` (`with_scale=True`, the class default) -
//! unlike `q_norm`/`k_norm` (per-head, over `head_dim`) or `v_norm`
//! (per-head, `with_scale=False` - see [`attention`]'s doc), these four are
//! over the FULL `hidden_size` row.
//!
//! ## Attention: which kernels, and why (the two structurally different
//! ## layer types, and the two kernel-contract mismatches found)
//!
//! Both layer types dispatch the SAME `gqa_scores_win` / `attn_softmax` /
//! `gqa_apply` trio (`gqa_scores_win` degenerates to a plain causal mask when
//! `window >= tcols`, its own doc's words - so a `sliding_attention` layer
//! passes `window = cfg.sliding_window` and a `full_attention` layer passes
//! `window = t`, never a second kernel). Native GQA (`group` param) covers
//! BOTH layer types' key/value head count directly - no host-side
//! `repeat_kv` materialization needed, unlike a kernel with no GQA awareness
//! would require.
//!
//! **Mismatch 1 - the built-in `1/sqrt(head_dim)` score scale.**
//! `gqa_scores_win.wgsl` (like every other scores kernel in this repo)
//! hardcodes `scores[...] = ... * inverseSqrt(f32(head_dim))`. Gemma-4's real
//! `Gemma4TextAttention.__init__` sets `self.scaling = 1.0` UNCONDITIONALLY
//! (not `head_dim**-0.5`, not a `query_pre_attn_scalar` - verified by reading
//! both the vision and text attention constructors in `transformers.models.
//! gemma4.modular_gemma4`, and there is no `query_pre_attn_scalar` field
//! anywhere in `Gemma4TextConfig`) - so a naive dispatch would silently
//! attenuate every score by `head_dim**-0.5` relative to the reference. Fixed
//! WITHOUT a new kernel or a new elementwise-scale dispatch: RMSNorm followed
//! by RoPE is linear in a uniform per-vector scalar (RoPE is a rotation - it
//! commutes with scaling), so multiplying `q_norm`'s uploaded weight vector
//! by `sqrt(head_dim)` (done once, on the HOST, in [`AttnWeights::upload`] -
//! the golden's own `q_norm.weight` on disk is never mutated, only the
//! device-resident copy this crate uploads) makes the kernel's built-in
//! `q·k / sqrt(head_dim)` compute `q·k * sqrt(head_dim) / sqrt(head_dim) =
//! q·k` exactly - `scaling=1.0`, bit-for-bit, not an approximation.
//!
//! **Mismatch 2 - none, for k_eq_v.** The `attention_k_eq_v` global layers
//! need NO new kernel and no "skip the V matmul" parameter on any existing
//! one: `gqa_apply`'s V input is just a `[T, n_kv_heads*head_dim]` buffer,
//! and this crate feeds it `v_norm(raw k_proj output)` (computed via the SAME
//! `rmsnorm_eps` dispatch every layer already uses, just fed the k-region's
//! PRE-`k_norm`/PRE-RoPE buffer instead of a `v_proj` output) whenever
//! `cfg.k_eq_v_for(lt)` is true - a purely host-side wiring choice of which
//! upstream buffer feeds the V-side RMSNorm, exactly mirroring
//! `Gemma4TextAttention.forward`'s own `value_states = self.v_proj(...) if
//! self.v_proj is not None else key_states` (the raw, pre-norm key). No
//! `v_proj` weight is even uploaded for these layers (there is none in the
//! checkpoint - `v_proj: None` per source).
//!
//! **Mismatch 3 - `rope2d_partial` was the natural first guess for
//! `full_attention`'s RoPE, and is REFUTED** by `crate::rope`'s own doc: it
//! pairs channels at the ROTATED sub-block's own half-point, but Gemma-4
//! always pairs at the FULL head's half-point regardless of how few
//! frequencies are nonzero. `rope2d` (the SAME kernel `sliding_attention`
//! uses) is correct for BOTH layer types once the table itself carries the
//! zero-padded identity columns - see `crate::rope::full_table`'s doc. This
//! crate never dispatches `rope2d_partial` at all.

use std::collections::HashMap;

use gpu_core::{DeviceBuffer, Gpu, Step};

use crate::config::{Gemma4Config, LayerType};
use crate::rope::{apply_rope_full, DeviceRope};

pub type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

// Kernel-table indices (order matches KERNELS below).
const K_MATMUL: usize = 0;
const K_RMSNORM_EPS: usize = 1;
const K_GELU: usize = 2;
const K_MUL: usize = 3;
const K_ADD2: usize = 4;
const K_ROPE2D: usize = 5;
const K_GQA_SCORES_WIN: usize = 6;
const K_ATTN_SOFTMAX: usize = 7;
const K_GQA_APPLY: usize = 8;
const K_MAX_ABS_ROW: usize = 9;
const K_QUANT_PACK: usize = 10;
const K_MATMUL_I8_DYN: usize = 11;

/// Every kernel this block dispatches - all pre-existing, all at their
/// documented general contract (see this module's doc for the three kernel-
/// contract facts found while wiring them up - two needed no new kernel at
/// all, and the third (`rope2d_partial`) is a REFUTED guess, not something
/// this crate ships).
pub const KERNELS: [(&str, &str); 12] = [
    ("matmul", kernels::MATMUL),
    ("rmsnorm_eps", kernels::RMSNORM_EPS),
    ("gelu", kernels::GELU),
    ("mul", kernels::MUL),
    ("add2", kernels::ADD2),
    ("rope2d", kernels::ROPE2D),
    ("gqa_scores_win", kernels::GQA_SCORES_WIN),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("gqa_apply", kernels::GQA_APPLY),
    // The int8 tier ([`Precision::Int8`]). Registered unconditionally -
    // building a pipeline costs a compile at `Gpu::new` and nothing at
    // dispatch, and a kernel table that changes shape with a runtime flag is
    // how a model ends up with two incompatible kernel index spaces.
    ("max_abs_row", kernels::MAX_ABS_ROW),
    ("quant_pack", kernels::QUANT_PACK),
    ("matmul_i8_dyn", kernels::MATMUL_I8_DYN),
];

/// Which arithmetic the seven per-layer projections run in.
///
/// This is a CAPACITY-and-bandwidth choice on most hardware and a SPEED
/// choice on some. The rule for picking it is not "int8 is smaller so use
/// int8": it is what the device can actually execute fast, queried rather
/// than assumed, which is what [`Precision::for_device`] does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// Plain `matmul`. Every device supports it; nothing about it is
    /// conditional.
    Fp32,
    /// Per-output-channel symmetric int8 weights with per-token dynamic
    /// activation scales, through the packed-dot GEMM. Requires
    /// `DeviceCaps::numeric.int8_dot`.
    Int8,
}

impl Precision {
    /// Resolve a REQUESTED precision against what the device can execute.
    ///
    /// A request for int8 on a device with no packed-dot path is not an
    /// error and is not silently honoured either - it falls back to fp32 and
    /// says so, the same shape `model::ops::Weight::upload`'s
    /// `Dtype::promote` and `qwen3::serve`'s `weights_int8 && caps.numeric.
    /// int8_dot` already use. The capability is the gate; which int8 GEMM
    /// variant would be fastest is a separate, later question.
    ///
    /// Converting a weight DOWN to int8 to run it on hardware whose fp32
    /// path is the fast one would be a pure loss, which is exactly why this
    /// asks the device instead of assuming a tier.
    pub fn for_device(gpu: &Gpu, requested: Precision) -> Precision {
        if requested == Precision::Int8 && !gpu.caps().numeric.int8_dot {
            eprintln!("gemma4: int8 requested but this device exposes no packed-int8 dot path; running the projections in fp32");
            return Precision::Fp32;
        }
        requested
    }
}

fn tget<'a>(w: &'a Tensors, name: &str) -> &'a [f32] {
    &w.get(name).unwrap_or_else(|| panic!("gemma4: missing weight {name}")).1
}

fn upload(gpu: &Gpu, data: &[f32]) -> DeviceBuffer {
    let buf = gpu.storage(data.len() as u64);
    gpu.write_f32(&buf, data);
    buf
}

/// One projection's weight, resident at whatever [`Precision`] the layer was
/// built with.
///
/// The precision lives HERE, in the weight, and not in a second forward
/// pass. A quantized twin of the whole layer would be two implementations of
/// one architecture, and the failure mode of that shape is a real one this
/// engine has already paid for: a fix applied to one path and silently
/// absent from the other. [`linear`] is the only place that branches, so
/// every norm, RoPE, mask and softmax below is literally the same code in
/// both tiers.
enum Proj {
    F32(DeviceBuffer),
    /// `model::int8::quantize_weight`'s packed `[n, k/4]` u32 words plus its
    /// `[n, k/32]` f32 group scale.
    I8 { w: DeviceBuffer, sw: DeviceBuffer },
}

impl Proj {
    /// Upload `[n, k]` row-major fp32 weight data at `precision`.
    fn upload(gpu: &Gpu, data: &[f32], n: usize, k: usize, precision: Precision) -> Proj {
        match precision {
            Precision::Fp32 => Proj::F32(upload(gpu, data)),
            Precision::Int8 => {
                let (packed, sw) = ::model::int8::quantize_weight(data, n, k);
                let wb = gpu.storage(packed.len() as u64);
                gpu.write(&wb, &packed);
                Proj::I8 { w: wb, sw: upload(gpu, &sw) }
            }
        }
    }

    fn from_tensors(gpu: &Gpu, w: &Tensors, name: &str, precision: Precision) -> Proj {
        let (shape, data) = w.get(name).unwrap_or_else(|| panic!("gemma4: missing weight {name}"));
        assert_eq!(shape.len(), 2, "gemma4: {name} is a projection and must be rank 2, got {shape:?}");
        Proj::upload(gpu, data, shape[0], shape[1], precision)
    }
}

/// One layer's attention weights, uploaded once. `wv`/`k_eq_v` are mutually
/// exclusive (see this module's doc, Mismatch 2); `q_norm` already carries
/// the `sqrt(head_dim)` fold (Mismatch 1) - `v_norm_ones` is a synthesized
/// all-ones buffer since `Gemma4RMSNorm(..., with_scale=False)` has no
/// learnable weight in the checkpoint at all.
struct AttnWeights {
    wq: Proj,
    q_norm: DeviceBuffer,
    wk: Proj,
    k_norm: DeviceBuffer,
    wv: Option<Proj>,
    v_norm_ones: DeviceBuffer,
    wo: Proj,
    k_eq_v: bool,
}

impl AttnWeights {
    fn upload(gpu: &Gpu, w: &Tensors, prefix: &str, head_dim: u32, k_eq_v: bool, precision: Precision) -> AttnWeights {
        let q_norm_raw = tget(w, &format!("{prefix}.q_norm.weight"));
        // Mismatch 1 (this module's doc): fold `sqrt(head_dim)` into q_norm's
        // UPLOADED weight only - the host `Tensors` map (and hence anything
        // else that might read `q_norm.weight` later) is untouched.
        //
        // Note this fold survives the int8 tier untouched: `q_norm` is a
        // norm gain, never a projection, so it is fp32 in both tiers and the
        // `sqrt(head_dim)` factor is not exposed to any quantization step.
        let scale = (head_dim as f32).sqrt();
        let q_norm_scaled: Vec<f32> = q_norm_raw.iter().map(|v| v * scale).collect();
        AttnWeights {
            wq: Proj::from_tensors(gpu, w, &format!("{prefix}.q_proj.weight"), precision),
            q_norm: upload(gpu, &q_norm_scaled),
            wk: Proj::from_tensors(gpu, w, &format!("{prefix}.k_proj.weight"), precision),
            k_norm: upload(gpu, tget(w, &format!("{prefix}.k_norm.weight"))),
            wv: (!k_eq_v).then(|| Proj::from_tensors(gpu, w, &format!("{prefix}.v_proj.weight"), precision)),
            v_norm_ones: upload(gpu, &vec![1.0f32; head_dim as usize]),
            wo: Proj::from_tensors(gpu, w, &format!("{prefix}.o_proj.weight"), precision),
            k_eq_v,
        }
    }
}

struct MlpWeights {
    gate: Proj,
    up: Proj,
    down: Proj,
}

impl MlpWeights {
    fn upload(gpu: &Gpu, w: &Tensors, prefix: &str, precision: Precision) -> MlpWeights {
        MlpWeights {
            gate: Proj::from_tensors(gpu, w, &format!("{prefix}.gate_proj.weight"), precision),
            up: Proj::from_tensors(gpu, w, &format!("{prefix}.up_proj.weight"), precision),
            down: Proj::from_tensors(gpu, w, &format!("{prefix}.down_proj.weight"), precision),
        }
    }
}

/// `out = x @ Wᵀ`, `x: [m,k]`, `w: [n,k]`, `out: [m,n]` - no bias anywhere in
/// this crate (`attention_bias=false`, MLP linears are always bias-free per
/// `Gemma4TextMLP`/`Gemma3MLP`).
///
/// The int8 arm quantizes the ACTIVATION per row on the fly
/// (`max_abs_row` -> `quant_pack`) and lets the packed-dot GEMM dequantize
/// with `sx*sw` on the way out - the same recipe `ltxv::block`'s own
/// quantized linears use, at this model's shapes. Every `k` here (hidden
/// 3840, q_dim 4096, kv_dim 2048/512, intermediate 15360) is a multiple of
/// 4, which is the packing width the kernel requires.
fn linear(gpu: &Gpu, s: &mut Vec<Step>, x: &DeviceBuffer, w: &Proj, out: &DeviceBuffer, m: u32, k: u32, n: u32) {
    match w {
        Proj::F32(wb) => s.push(gpu.step(K_MATMUL, &[x, wb, out], &[m, k, n], m * n)),
        Proj::I8 { w: wb, sw } => {
            debug_assert_eq!(k % 4, 0, "matmul_i8_dyn packs 4 int8 lanes per u32");
            let xq = gpu.storage((m * k / 4) as u64);
            let sx = gpu.storage(m as u64);
            s.push(gpu.step(K_MAX_ABS_ROW, &[x, &sx], &[m, k], m));
            s.push(gpu.step(K_QUANT_PACK, &[x, &sx, &xq], &[m, k], m * k / 4));
            s.push(gpu.step(K_MATMUL_I8_DYN, &[&xq, wb, &sx, sw, out], &[m, k / 4, n], m.div_ceil(128) * n.div_ceil(128) * 256));
        }
    }
}

/// RMSNorm over the full row width `dim` - `w` is either a learnable gain
/// (`q_norm`/`k_norm`/the four block-level norms) or the synthesized all-ones
/// buffer (`v_norm`, `with_scale=False` - see [`AttnWeights`]'s doc).
fn rmsnorm(gpu: &Gpu, s: &mut Vec<Step>, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, dim: u32, rows: u32, eps: f32) {
    s.push(gpu.step(K_RMSNORM_EPS, &[x, w, out], &[dim, rows, gpu_core::f(eps)], rows));
}

fn add2(gpu: &Gpu, s: &mut Vec<Step>, a: &DeviceBuffer, b: &DeviceBuffer, y: &DeviceBuffer, n: u32) {
    s.push(gpu.step(K_ADD2, &[a, b, y], &[n], n));
}

fn mul(gpu: &Gpu, s: &mut Vec<Step>, a: &DeviceBuffer, b: &DeviceBuffer, y: &DeviceBuffer, n: u32) {
    s.push(gpu.step(K_MUL, &[a, b, y], &[n], n));
}

/// One self-attention call for a single layer. `x_in`: `[t, hidden]`
/// (already `input_layernorm`'d). Returns the `[t, hidden]` output-projected
/// result (matching `Gemma4TextAttention.forward`'s return value, i.e. the
/// value fed to `post_attention_layernorm` next).
#[allow(clippy::too_many_arguments)]
fn attention(gpu: &Gpu, s: &mut Vec<Step>, cfg: &Gemma4Config, lt: LayerType, w: &AttnWeights, x_in: &DeviceBuffer, hidden: u32, t: u32, rope: &DeviceRope) -> DeviceBuffer {
    let head_dim = cfg.head_dim_for(lt);
    let heads = cfg.num_attention_heads;
    let kv_heads = cfg.kv_heads_for(lt);
    let group = cfg.groups_for(lt);
    let eps = cfg.rms_norm_eps;

    let q_dim = heads * head_dim;
    let kv_dim = kv_heads * head_dim;

    let q_pre = gpu.storage((t * q_dim) as u64);
    let k_pre = gpu.storage((t * kv_dim) as u64);
    linear(gpu, s, x_in, &w.wq, &q_pre, t, hidden, q_dim);
    linear(gpu, s, x_in, &w.wk, &k_pre, t, hidden, kv_dim);

    // Mismatch 2 (this module's doc): v_pre is EITHER a fresh v_proj
    // matmul, OR (k_eq_v) the SAME raw k_pre buffer reused as v_norm's
    // input - never rotated, never k_norm'd, exactly `Gemma4TextAttention.
    // forward`'s `value_states = ... else key_states` (the pre-norm key).
    let v_pre = if let Some(wv) = &w.wv {
        let vb = gpu.storage((t * kv_dim) as u64);
        linear(gpu, s, x_in, wv, &vb, t, hidden, kv_dim);
        vb
    } else {
        assert!(w.k_eq_v);
        k_pre.clone()
    };

    let q = gpu.storage((t * q_dim) as u64);
    let k = gpu.storage((t * kv_dim) as u64);
    let v = gpu.storage((t * kv_dim) as u64);
    rmsnorm(gpu, s, &q_pre, &w.q_norm, &q, head_dim, t * heads, eps);
    rmsnorm(gpu, s, &k_pre, &w.k_norm, &k, head_dim, t * kv_heads, eps);
    rmsnorm(gpu, s, &v_pre, &w.v_norm_ones, &v, head_dim, t * kv_heads, eps);

    // RoPE: q and k only (v never rotates) - one dispatch each, every head in
    // that buffer sharing the same table (native `heads` param). `rope2d`
    // unconditionally for BOTH layer types - see this module's doc,
    // Mismatch 3, and `crate::rope::full_table`'s doc for why
    // `full_attention`'s "partial" behavior lives in the table, not a
    // different kernel.
    let (cos, sin, half) = (&rope.cos, &rope.sin, rope.half);
    s.push(apply_rope_full(gpu, K_ROPE2D, &q, cos, sin, t, heads, half, q_dim));
    s.push(apply_rope_full(gpu, K_ROPE2D, &k, cos, sin, t, kv_heads, half, kv_dim));

    // Sliding layers window; full/global layers pass `window = t`, which
    // `gqa_scores_win` documents as degenerating exactly to plain causal.
    let window = match lt {
        LayerType::Sliding => cfg.sliding_window,
        LayerType::Full => t,
    };
    let scores = gpu.storage((heads * t * t) as u64);
    let probs = gpu.storage((heads * t * t) as u64);
    let ctx = gpu.storage((t * q_dim) as u64);
    s.push(gpu.step(K_GQA_SCORES_WIN, &[&q, &k, &scores], &[1, heads, kv_heads, t, head_dim, group, window], heads * t * t));
    s.push(gpu.step(K_ATTN_SOFTMAX, &[&scores, &probs], &[1, heads, t], heads * t));
    s.push(gpu.step(K_GQA_APPLY, &[&probs, &v, &ctx], &[1, heads, kv_heads, t, head_dim, group], heads * t * head_dim));

    let out = gpu.storage((t * hidden) as u64);
    linear(gpu, s, &ctx, &w.wo, &out, t, q_dim, hidden);
    out
}

/// One `Gemma4UnifiedTextDecoderLayer`, weights resident, for a fixed token
/// count.
pub struct Gemma4Layer {
    gpu: Gpu,
    cfg: Gemma4Config,
    lt: LayerType,
    attn: AttnWeights,
    mlp: MlpWeights,
    input_ln: Vec<f32>,
    post_attn_ln: Vec<f32>,
    pre_ff_ln: Vec<f32>,
    post_ff_ln: Vec<f32>,
    layer_scalar: f32,
}

impl Gemma4Layer {
    /// Build layer `layer_idx` with its projections resident at `precision`.
    /// `weights` needs only THIS layer's tensors, which is what lets a
    /// caller stream a checkpoint layer by layer.
    pub fn on(gpu: Gpu, cfg: &Gemma4Config, weights: &Tensors, layer_idx: u32, precision: Precision) -> Gemma4Layer {
        let lt = cfg.layer_type(layer_idx);
        let head_dim = cfg.head_dim_for(lt);
        let k_eq_v = cfg.k_eq_v_for(lt);
        let prefix = format!("layers.{layer_idx}");
        let attn = AttnWeights::upload(&gpu, weights, &format!("{prefix}.self_attn"), head_dim, k_eq_v, precision);
        let mlp = MlpWeights::upload(&gpu, weights, &format!("{prefix}.mlp"), precision);
        let layer_scalar = tget(weights, &format!("{prefix}.layer_scalar"))[0];
        Gemma4Layer {
            gpu,
            cfg: *cfg,
            lt,
            attn,
            mlp,
            input_ln: tget(weights, &format!("{prefix}.input_layernorm.weight")).to_vec(),
            post_attn_ln: tget(weights, &format!("{prefix}.post_attention_layernorm.weight")).to_vec(),
            pre_ff_ln: tget(weights, &format!("{prefix}.pre_feedforward_layernorm.weight")).to_vec(),
            post_ff_ln: tget(weights, &format!("{prefix}.post_feedforward_layernorm.weight")).to_vec(),
            layer_scalar,
        }
    }

    pub fn layer_type(&self) -> LayerType {
        self.lt
    }

    /// One layer forward - see this module's doc for the exact op order.
    /// `rope`: this layer type's `(cos, sin)` device tables (built once by
    /// the caller, shared by every layer of the same type).
    /// Returns `(layer_output, self_attn_output)` - the latter purely for
    /// parity taps (see `crate::model`).
    pub fn forward(&self, x: &[f32], rope: &DeviceRope, t: u32) -> (Vec<f32>, Vec<f32>) {
        let gpu = &self.gpu;
        let cfg = &self.cfg;
        let hidden = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        assert_eq!(x.len(), (t * hidden) as usize);

        let x_buf = upload(gpu, x);
        let input_ln = upload(gpu, &self.input_ln);
        let post_attn_ln = upload(gpu, &self.post_attn_ln);
        let pre_ff_ln = upload(gpu, &self.pre_ff_ln);
        let post_ff_ln = upload(gpu, &self.post_ff_ln);

        let mut s: Vec<Step> = Vec::new();
        let td = t * hidden;

        let h1 = gpu.storage(td as u64);
        rmsnorm(gpu, &mut s, &x_buf, &input_ln, &h1, hidden, t, eps);

        let attn_out = attention(gpu, &mut s, cfg, self.lt, &self.attn, &h1, hidden, t, rope);

        let h2 = gpu.storage(td as u64);
        rmsnorm(gpu, &mut s, &attn_out, &post_attn_ln, &h2, hidden, t, eps);
        let x1 = gpu.storage(td as u64);
        add2(gpu, &mut s, &x_buf, &h2, &x1, td);

        let h3 = gpu.storage(td as u64);
        rmsnorm(gpu, &mut s, &x1, &pre_ff_ln, &h3, hidden, t, eps);
        let ff_dim = cfg.intermediate_size;
        let gate_pre = gpu.storage((t * ff_dim) as u64);
        let up = gpu.storage((t * ff_dim) as u64);
        linear(gpu, &mut s, &h3, &self.mlp.gate, &gate_pre, t, hidden, ff_dim);
        linear(gpu, &mut s, &h3, &self.mlp.up, &up, t, hidden, ff_dim);
        let gate_act = gpu.storage((t * ff_dim) as u64);
        s.push(gpu.step(K_GELU, &[&gate_pre, &gate_act], &[t * ff_dim], t * ff_dim));
        let h_act = gpu.storage((t * ff_dim) as u64);
        mul(gpu, &mut s, &gate_act, &up, &h_act, t * ff_dim);
        let mlp_out = gpu.storage(td as u64);
        linear(gpu, &mut s, &h_act, &self.mlp.down, &mlp_out, t, ff_dim, hidden);

        let h4 = gpu.storage(td as u64);
        rmsnorm(gpu, &mut s, &mlp_out, &post_ff_ln, &h4, hidden, t, eps);
        let x2 = gpu.storage(td as u64);
        add2(gpu, &mut s, &x1, &h4, &x2, td);

        gpu.submit(&[], &s);
        let mut out = gpu.read(&x2, td as usize);
        for v in out.iter_mut() {
            *v *= self.layer_scalar;
        }
        let attn_out_host = gpu.read(&attn_out, td as usize);
        (out, attn_out_host)
    }
}
