// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3.5/3.8-27B dense hybrid decoder - FORWARD ONLY, text-only, no
//! LoRA/int8/decode-step/vision-splice/sharding yet (each lands as its own
//! later milestone, mirroring `qwen35moe::model`'s own structure once it
//! does). See `crate::config` for the architecture summary.
//!
//! **Scope, strictly, matching the M5 milestone**: a single whole-sequence
//! prefill forward (`t` must already be a multiple of the derived GDN chunk
//! size - asserted loudly in [`Qwen35::new_on`], see [`gdn_chunk_size`]),
//! frozen weights only (`Role::Frozen` - no gradient/Adam buffers). Backward
//! (M6), the MTP head (M7), LoRA/full finetune (M8), the vision splice (M9),
//! and int8/decode/sharding land as their own milestones, each mirroring the
//! matching piece of `qwen35moe::model` the way this file's forward already
//! does.
//!
//! ## Layer forward, verified against the installed
//! `transformers.models.qwen3_5` reference (not a secondhand description -
//! see `tools/goldens/qwen35_dump_reference.py`, which hand-replays every
//! step below against the real reference module and reports the manual-vs-
//! real diff per layer)
//!
//! Every layer, regardless of token-mixer type: `xn1 = rmsnorm(res)`, mix
//! (GDN or GQA, below), `xmid = res + mix_out`, `xn2 = rmsnorm(xmid)`, a
//! plain dense SwiGLU MLP (`down(silu(gate(xn2)) * up(xn2))` - every layer,
//! no router, no experts, unlike `qwen35moe`), `res' = xmid + mlp_out`.
//!
//! **Gated DeltaNet** and **GQA** mixer mechanics (chunked delta-rule
//! recurrence, per-head-interleaved doubled `q_proj` gate split, partial
//! M-RoPE) are byte-identical to `qwen35moe`'s own - this file's
//! `layer_gdn_fwd`/`layer_gqa_fwd` are close copies of
//! `qwen35moe::model`'s, differing only in using this crate's own
//! `Qwen35Config` (dense) instead of qwen35moe's (MoE). The `(1+w)` RMSNorm
//! fold happens once, at import time (`crate::import`) - this file's
//! `rmsnorm_fwd` calls assume the stored weight already IS the final
//! multiplier, exactly like every other model in this engine.

use std::cell::RefCell;
use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu};
use paramstore::{ParamStore, Role};

use audio::conv::{conv1d_fwd, Conv1d, ConvKernels};
use model::block::{gqa_fwd, kv_expand_fwd, rmsnorm_fwd, rope2d_partial_fwd, Gqa, KernelIds};
use model::gdn::{gdn_chunk_fwd, GdnIds, GdnScratchBufs, GdnShape};
pub use model::gdn::gdn_chunk_size;

use crate::config::{LayerType, Qwen35Config};

// ---- kernel pipeline (order fixes the indices below) -----------------------
// Forward-only subset of qwen35moe::model::PIPELINES (no MoE - this model
// has none; no backward/int8/decode/LoRA/splice tiers yet - each lands with
// its own milestone).

pub const PIPELINES: &[(&str, &str)] = &[
    ("rmsnorm", kernels::RMSNORM), // 0
    ("matmul", kernels::MATMUL), // 1
    ("embed", kernels::EMBED), // 2
    ("sigmoid", kernels::SIGMOID), // 3
    ("silu", kernels::SILU), // 4
    ("silu_mul", kernels::SILU_MUL), // 5
    ("mul", kernels::MUL), // 6
    ("add2", kernels::ADD2), // 7
    ("l2norm_scale", kernels::L2NORM_SCALE), // 8
    ("concat_split", kernels::CONCAT_SPLIT), // 9
    ("nlc_nchw", kernels::NLC_NCHW), // 10
    ("nchw_nlc", kernels::NCHW_NLC), // 11
    ("conv1d", kernels::CONV1D), // 12
    ("gdn_decay_gate", kernels::GDN_DECAY_GATE), // 13
    ("gdn_layout_permute", kernels::GDN_LAYOUT_PERMUTE), // 14
    ("rope2d_partial", kernels::ROPE2D_PARTIAL), // 15
    ("gqa_scores", kernels::GQA_SCORES), // 16
    ("attn_softmax", kernels::ATTN_SOFTMAX), // 17
    ("gqa_apply", kernels::GQA_APPLY), // 18
    ("kv_expand", kernels::KV_EXPAND), // 19
    ("scale_row", kernels::SCALE_ROW), // 20
    ("bmm", kernels::BMM), // 21
    ("bmm_acc", kernels::BMM_ACC), // 22
    ("gdn_chunk_cumsum_step", kernels::GDN_CHUNK_CUMSUM_STEP), // 23
    ("gdn_decay_mask", kernels::GDN_DECAY_MASK), // 24
    ("gdn_mask_strict_lower", kernels::GDN_MASK_STRICT_LOWER), // 25
    ("gdn_ut_step", kernels::GDN_UT_STEP), // 26
    ("gdn_add_identity", kernels::GDN_ADD_IDENTITY), // 27
    ("gdn_row_scale_off", kernels::GDN_ROW_SCALE_OFF), // 28
    ("gdn_decay_scale", kernels::GDN_DECAY_SCALE), // 29
    ("gdn_state_decay", kernels::GDN_STATE_DECAY), // 30
    ("exp", kernels::EXP), // 31
    ("sub", kernels::SUB), // 32
    ("region_copy", kernels::REGION_COPY), // 33
];

const RMSNORM: usize = 0;
const MATMUL: usize = 1;
const EMBED: usize = 2;
const SIGMOID: usize = 3;
const SILU: usize = 4;
const SILU_MUL: usize = 5;
const MUL: usize = 6;
const ADD2: usize = 7;
const L2NORM_SCALE: usize = 8;
const CONCAT_SPLIT: usize = 9;
const NLC_NCHW: usize = 10;
const NCHW_NLC: usize = 11;
const CONV1D: usize = 12;
const GDN_DECAY_GATE: usize = 13;
const GDN_LAYOUT_PERMUTE: usize = 14;
const ROPE2D_PARTIAL: usize = 15;
const GQA_SCORES: usize = 16;
const ATTN_SOFTMAX: usize = 17;
const GQA_APPLY: usize = 18;
const KV_EXPAND: usize = 19;
const SCALE_ROW: usize = 20;
const BMM: usize = 21;
const BMM_ACC: usize = 22;
const GDN_CHUNK_CUMSUM_STEP: usize = 23;
const GDN_DECAY_MASK: usize = 24;
const GDN_MASK_STRICT_LOWER: usize = 25;
const GDN_UT_STEP: usize = 26;
const GDN_ADD_IDENTITY: usize = 27;
const GDN_ROW_SCALE_OFF: usize = 28;
const GDN_DECAY_SCALE: usize = 29;
const GDN_STATE_DECAY: usize = 30;
const EXP: usize = 31;
const SUB: usize = 32;
const REGION_COPY: usize = 33;

/// Backward-only `KernelIds` fields (`rms_inv`/`rmsnorm_dx`/`rmsnorm_dw`/
/// `gqa_d*`/`silu_d*`) point at `RMSNORM` - never dispatched by a
/// forward-only build, mirroring `qwen35moe::model::kernel_ids`'s own
/// `rope`/`rope_bwd` placeholder convention exactly (a real backward, M6,
/// gives them real indices once those kernels are registered here).
fn kernel_ids() -> KernelIds {
    KernelIds {
        rmsnorm: RMSNORM,
        rms_inv: RMSNORM,
        rmsnorm_dx: RMSNORM,
        rmsnorm_dw: RMSNORM,
        rope: RMSNORM,
        rope_bwd: RMSNORM,
        gqa_scores: GQA_SCORES,
        gqa_apply: GQA_APPLY,
        attn_softmax: ATTN_SOFTMAX,
        gqa_dscores: RMSNORM,
        gqa_dv: RMSNORM,
        gqa_dq: RMSNORM,
        gqa_dk: RMSNORM,
        silu_mul: SILU_MUL,
        silu_da: RMSNORM,
        silu_db: RMSNORM,
    }
}

fn gdn_ids() -> GdnIds {
    GdnIds {
        bmm: BMM,
        bmm_acc: BMM_ACC,
        cumsum_step: GDN_CHUNK_CUMSUM_STEP,
        decay_mask: GDN_DECAY_MASK,
        mask_strict_lower: GDN_MASK_STRICT_LOWER,
        ut_step: GDN_UT_STEP,
        add_identity: GDN_ADD_IDENTITY,
        row_scale: SCALE_ROW,
        row_scale_off: GDN_ROW_SCALE_OFF,
        decay_scale: GDN_DECAY_SCALE,
        state_decay: GDN_STATE_DECAY,
        exp: EXP,
        sub: SUB,
        mul: MUL,
        region_copy: REGION_COPY,
    }
}

fn conv_kernels() -> ConvKernels {
    // `dx`/`dw` point at `CONV1D` itself - never dispatched by a
    // forward-only build (no backward kernels registered yet, M6 gives them
    // real indices once `conv1d_dx`/`conv1d_dw` are added to `PIPELINES`).
    ConvKernels { fwd: CONV1D, dx: CONV1D, dw: CONV1D }
}

pub struct Qwen35 {
    pub gpu: Gpu,
    pub cfg: Qwen35Config,
    ps: ParamStore,
    b: u32,
    t: u32,
    /// The GDN chunk size this instance was built for - see [`gdn_chunk_size`].
    chunk: u32,

    tokens: DeviceBuffer,

    /// All-ones buffer of width `linear_key_head_dim`, bound as
    /// `l2norm_scale.wgsl`'s per-dim scale so its learnably-scaled L2-norm
    /// computes the reference's bare `l2norm(x)` (GDN's q/k norm has no
    /// learnable scale).
    ones_khd: DeviceBuffer,
    /// M-RoPE `cos`/`sin` tables, built once at construction for the fixed
    /// `(b,t)` this instance decodes: text-only, so every axis carries the
    /// same plain sequential position per sequence.
    cos: DeviceBuffer,
    sin: DeviceBuffer,

    logits: DeviceBuffer,

    /// Residual stream, one entry per layer boundary (`res[0]` = embeddings,
    /// `res[n_layers]` = input to the final norm) - the SSA activation-cache
    /// convention `qwen35moe::model` uses, kept even though nothing
    /// backprops through it yet: useful for parity debugging, any layer's
    /// residual output is independently readable via [`Self::debug_res`].
    res: RefCell<Vec<DeviceBuffer>>,
}

impl Qwen35 {
    pub fn new_on(gpu: Gpu, cfg: Qwen35Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Qwen35 {
        let chunk = gdn_chunk_size(t);
        assert_eq!(
            t % chunk,
            0,
            "qwen35: t={t} is not a multiple of the derived GDN chunk size {chunk} -- \
             model::gdn is prefill-only (no T-padding support, see its module doc); \
             gdn_chunk_size always returns a value that divides t by construction, so \
             this assert failing would mean a logic error in gdn_chunk_size itself"
        );

        let roles: Vec<(String, usize, Role)> = cfg.param_list().into_iter().map(|(n, c)| (n, c, Role::Frozen)).collect();
        let ps = ParamStore::new_with_roles_src(&gpu, roles, init);

        let ones_khd = gpu.storage_init("qwen35.ones_khd", &vec![1.0f32; cfg.linear_key_head_dim as usize]);

        // Text-only: every axis of the M-RoPE table carries the same plain
        // sequential position, reset per sequence (row = batch*t + pos).
        let positions: Vec<[u32; 3]> = (0..b).flat_map(|_| (0..t).map(|ti| [ti, ti, ti])).collect();
        let (cos, sin) = qwen3vl::mrope::mrope_tables(&positions, cfg.mrope_section, cfg.rotary_dim(), cfg.rope_theta);
        let cos = gpu.storage_init("qwen35.rope_cos", &cos);
        let sin = gpu.storage_init("qwen35.rope_sin", &sin);

        let n = (b * t) as u64;
        let tokens = gpu.storage(n);
        let logits = gpu.storage(n * cfg.vocab as u64);
        let d = cfg.d_model as u64;
        let res = RefCell::new((0..=cfg.n_layers).map(|_| gpu.storage(n * d)).collect());

        Qwen35 { gpu, cfg, ps, b, t, chunk, tokens, ones_khd, cos, sin, logits, res }
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }

    // ---- one Gated DeltaNet (Linear) layer --------------------------------

    fn layer_gdn_fwd(&self, l: usize, xn1: &DeviceBuffer, n: u32) -> DeviceBuffer {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let conv_dim = c.linear_conv_dim();
        let key_dim = c.linear_key_dim();
        let value_dim = c.linear_value_dim();
        let nkh = c.linear_num_key_heads;
        let nvh = c.linear_num_value_heads;
        let khd = c.linear_key_head_dim;
        let vhd = c.linear_value_head_dim;
        let group = c.linear_group();
        let kw = c.linear_conv_kernel_dim;
        let (b, t, chunk) = (self.b, self.t, self.chunk);
        let n_chunks = t / chunk;
        let p = |s: &str| format!("blocks.{l}.linear_attn.{s}");

        // 1. mixed_qkv = in_proj_qkv(xn1).
        let mixed_qkv = g.storage((n * conv_dim) as u64);
        g.submit(&[], &[g.step(MATMUL, &[xn1, self.w(&p("in_proj_qkv.weight")), &mixed_qkv], &[n, d, conv_dim], n * conv_dim)]);

        // 2. Depthwise causal conv1d + SiLU (activation AFTER the conv).
        // conv1d.wgsl is NCL ([N,Cin,L]); mixed_qkv is token-major ([B,T,C]).
        let ncl_in = g.storage((n * conv_dim) as u64);
        g.submit(&[], &[g.step(NLC_NCHW, &[&mixed_qkv, &ncl_in], &[n * conv_dim, conv_dim, t], n * conv_dim)]);
        let conv_shape =
            Conv1d { n: b, cin: conv_dim, l: t, cout: conv_dim, k: kw, stride: 1, pad: kw - 1, dilation: 1, groups: conv_dim, lo: t };
        let ncl_out = g.storage((n * conv_dim) as u64);
        g.submit(&[], &[conv1d_fwd(g, &conv_kernels(), &conv_shape, &ncl_in, self.w(&p("conv1d.weight")), &ncl_out)]);
        let ncl_act = g.storage((n * conv_dim) as u64);
        g.submit(&[], &[g.step(SILU, &[&ncl_out, &ncl_act], &[n * conv_dim], n * conv_dim)]);
        let mixed_act = g.storage((n * conv_dim) as u64);
        g.submit(&[], &[g.step(NCHW_NLC, &[&ncl_act, &mixed_act], &[n * conv_dim, conv_dim, t], n * conv_dim)]);

        // 3. Split into query/key/value - ONE whole-row contiguous split.
        let query = g.storage((n * key_dim) as u64);
        let key = g.storage((n * key_dim) as u64);
        let value = g.storage((n * value_dim) as u64);
        g.submit(
            &[],
            &[
                g.step(CONCAT_SPLIT, &[&mixed_act, &query], &[n, conv_dim, key_dim, 0, 1, 1], n * key_dim),
                g.step(CONCAT_SPLIT, &[&mixed_act, &key], &[n, conv_dim, key_dim, key_dim, 1, 1], n * key_dim),
                g.step(CONCAT_SPLIT, &[&mixed_act, &value], &[n, conv_dim, value_dim, 2 * key_dim, 1, 1], n * value_dim),
            ],
        );

        // 4. L2-normalize query/key - bare l2norm (no learnable scale).
        let query_n = g.storage((n * key_dim) as u64);
        let key_n = g.storage((n * key_dim) as u64);
        g.submit(
            &[],
            &[
                g.step(L2NORM_SCALE, &[&query, &self.ones_khd, &query_n], &[n * nkh, khd, f(1e-6)], n * key_dim),
                g.step(L2NORM_SCALE, &[&key, &self.ones_khd, &key_n], &[n * nkh, khd, f(1e-6)], n * key_dim),
            ],
        );

        // 5. beta = sigmoid(in_proj_b(xn1)); g = -exp(A_log)*softplus(in_proj_a(xn1)+dt_bias);
        // z = in_proj_z(xn1) (feeds the gated RMSNorm at the end, step 10).
        let bproj = g.storage((n * nvh) as u64);
        let aproj = g.storage((n * nvh) as u64);
        let z = g.storage((n * value_dim) as u64);
        g.submit(
            &[],
            &[
                g.step(MATMUL, &[xn1, self.w(&p("in_proj_b.weight")), &bproj], &[n, d, nvh], n * nvh),
                g.step(MATMUL, &[xn1, self.w(&p("in_proj_a.weight")), &aproj], &[n, d, nvh], n * nvh),
                g.step(MATMUL, &[xn1, self.w(&p("in_proj_z.weight")), &z], &[n, d, value_dim], n * value_dim),
            ],
        );
        let beta = g.storage((n * nvh) as u64);
        let g_decay = g.storage((n * nvh) as u64);
        g.submit(
            &[],
            &[
                g.step(SIGMOID, &[&bproj, &beta], &[n * nvh], n * nvh),
                g.step(GDN_DECAY_GATE, &[&aproj, self.w(&p("A_log")), self.w(&p("dt_bias")), &g_decay], &[n, nvh], n * nvh),
            ],
        );

        // 6. Repeat query/key from linear_num_key_heads to linear_num_value_heads.
        let query_w = g.storage((n * nvh * khd) as u64);
        let key_w = g.storage((n * nvh * khd) as u64);
        g.submit(
            &[],
            &[
                kv_expand_fwd(g, KV_EXPAND, &query_n, &query_w, n, nvh, group, khd, nvh * khd, 0),
                kv_expand_fwd(g, KV_EXPAND, &key_n, &key_w, n, nvh, group, khd, nvh * khd, 0),
            ],
        );

        // 7. Chunk-major permute (token-major -> chunk-major) for gdn_chunk_fwd.
        let shape = GdnShape { b, h: nvh, t, dk: khd, dv: vhd, chunk };
        let permute_fwd = |src: &DeviceBuffer, dim: u32| -> DeviceBuffer {
            let dst = g.storage(b as u64 * nvh as u64 * n_chunks as u64 * chunk as u64 * dim as u64);
            g.submit(&[], &[g.step(GDN_LAYOUT_PERMUTE, &[src, &dst], &[b, nvh, n_chunks, chunk, dim, 1], b * nvh * n_chunks * chunk * dim)]);
            dst
        };
        let query_cm = permute_fwd(&query_w, khd);
        let key_cm = permute_fwd(&key_w, khd);
        let value_cm = permute_fwd(&value, vhd);
        let g_cm = permute_fwd(&g_decay, 1);
        let beta_cm = permute_fwd(&beta, 1);

        // 8. gdn_chunk_fwd - the chunked-recurrence forward itself.
        let bh = shape.bh() as u64;
        let initial_state = g.storage(bh * khd as u64 * vhd as u64);
        let final_state = g.storage(bh * khd as u64 * vhd as u64);
        let out_cm = g.storage(shape.bhc() as u64 * chunk as u64 * vhd as u64);
        let scratch = GdnScratchBufs::new(g, &shape);
        let steps = gdn_chunk_fwd(
            g,
            &gdn_ids(),
            &shape,
            &query_cm,
            &key_cm,
            &value_cm,
            &g_cm,
            &beta_cm,
            &initial_state,
            &scratch.as_ref(),
            &out_cm,
            &final_state,
        );
        g.submit(&scratch.clears(), &steps);

        // 9. Permute back to token-major.
        let out_tok = g.storage((n * value_dim) as u64);
        g.submit(
            &[],
            &[g.step(GDN_LAYOUT_PERMUTE, &[&out_cm, &out_tok], &[b, nvh, n_chunks, chunk, vhd, 0], b * nvh * n_chunks * chunk * vhd)],
        );

        // 10. Gated RMSNorm ("norm before gate"): normed = RMSNorm(out_tok)*weight,
        // THEN * SiLU(z).
        let normed = g.storage((n * value_dim) as u64);
        let z_silu = g.storage((n * value_dim) as u64);
        let gated = g.storage((n * value_dim) as u64);
        g.submit(
            &[],
            &[
                rmsnorm_fwd(g, &kernel_ids(), &out_tok, self.w(&p("norm.weight")), &normed, vhd, n * nvh),
                g.step(SILU, &[&z, &z_silu], &[n * value_dim], n * value_dim),
                g.step(MUL, &[&normed, &z_silu, &gated], &[n * value_dim], n * value_dim),
            ],
        );

        // 11. out_proj.
        let out = g.storage((n * d) as u64);
        g.submit(&[], &[g.step(MATMUL, &[&gated, self.w(&p("out_proj.weight")), &out], &[n, value_dim, d], n * d)]);
        out
    }

    // ---- one GQA (Full) layer ----------------------------------------------

    fn layer_gqa_fwd(&self, l: usize, xn1: &DeviceBuffer, n: u32) -> DeviceBuffer {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let (nh, nkv, hd) = (c.n_heads, c.n_kv_heads, c.head_dim);
        let (qpd, qd, kvd) = (c.q_proj_dim(), c.q_dim(), c.kv_dim());
        let p = |s: &str| format!("blocks.{l}.self_attn.{s}");

        let q_full = g.storage((n * qpd) as u64);
        let k = g.storage((n * kvd) as u64);
        let v = g.storage((n * kvd) as u64);
        g.submit(
            &[],
            &[
                g.step(MATMUL, &[xn1, self.w(&p("q_proj.weight")), &q_full], &[n, d, qpd], n * qpd),
                g.step(MATMUL, &[xn1, self.w(&p("k_proj.weight")), &k], &[n, d, kvd], n * kvd),
                g.step(MATMUL, &[xn1, self.w(&p("v_proj.weight")), &v], &[n, d, kvd], n * kvd),
            ],
        );

        // Per-head de-interleaved split of q_full's [query|gate] halves -
        // NOT a whole-row split. Fold n_heads into concat_split's own N so
        // each head's 2*head_dim block splits into its own first/second half.
        let q_value = g.storage((n * qd) as u64);
        let q_gate = g.storage((n * qd) as u64);
        g.submit(
            &[],
            &[
                g.step(CONCAT_SPLIT, &[&q_full, &q_value], &[n * nh, 2 * hd, hd, 0, 1, 1], n * nh * hd),
                g.step(CONCAT_SPLIT, &[&q_full, &q_gate], &[n * nh, 2 * hd, hd, hd, 1, 1], n * nh * hd),
            ],
        );

        let q_normed = g.storage((n * qd) as u64);
        let k_normed = g.storage((n * kvd) as u64);
        g.submit(
            &[],
            &[
                rmsnorm_fwd(g, &kernel_ids(), &q_value, self.w(&p("q_norm.weight")), &q_normed, hd, n * nh),
                rmsnorm_fwd(g, &kernel_ids(), &k, self.w(&p("k_norm.weight")), &k_normed, hd, n * nkv),
            ],
        );

        let half = c.rotary_dim() / 2;
        g.submit(
            &[],
            &[
                rope2d_partial_fwd(g, ROPE2D_PARTIAL, &q_normed, &self.cos, &self.sin, n, nh, half, qd, 0, hd),
                rope2d_partial_fwd(g, ROPE2D_PARTIAL, &k_normed, &self.cos, &self.sin, n, nkv, half, kvd, 0, hd),
            ],
        );

        let scores = g.storage(self.b as u64 * nh as u64 * self.t as u64 * self.t as u64);
        let probs = g.storage(self.b as u64 * nh as u64 * self.t as u64 * self.t as u64);
        let ctx = g.storage((n * qd) as u64);
        let ga = Gqa { b: self.b, t: self.t, n_heads: nh, n_kv_heads: nkv, head_dim: hd };
        g.submit(&[], &gqa_fwd(g, &kernel_ids(), &ga, &q_normed, &k_normed, &v, &scores, &probs, &ctx));

        let gate = g.storage((n * qd) as u64);
        let ctx_gated = g.storage((n * qd) as u64);
        let out = g.storage((n * d) as u64);
        g.submit(
            &[],
            &[
                g.step(SIGMOID, &[&q_gate, &gate], &[n * qd], n * qd),
                g.step(MUL, &[&ctx, &gate, &ctx_gated], &[n * qd], n * qd),
                g.step(MATMUL, &[&ctx_gated, self.w(&p("o_proj.weight")), &out], &[n, qd, d], n * d),
            ],
        );
        out
    }

    // ---- dense SwiGLU MLP, universal for every layer -----------------------

    fn mlp_fwd(&self, l: usize, xn2: &DeviceBuffer, n: u32) -> DeviceBuffer {
        let g = &self.gpu;
        let c = &self.cfg;
        let d = c.d_model;
        let ff = c.intermediate_size;
        let p = |s: &str| format!("blocks.{l}.mlp.{s}");

        let gate_pre = g.storage((n * ff) as u64);
        let up = g.storage((n * ff) as u64);
        g.submit(
            &[],
            &[
                g.step(MATMUL, &[xn2, self.w(&p("gate.weight")), &gate_pre], &[n, d, ff], n * ff),
                g.step(MATMUL, &[xn2, self.w(&p("up.weight")), &up], &[n, d, ff], n * ff),
            ],
        );
        let h = g.storage((n * ff) as u64);
        g.submit(&[], &[g.step(SILU_MUL, &[&gate_pre, &up, &h], &[n * ff], n * ff)]);
        let down = g.storage((n * d) as u64);
        g.submit(&[], &[g.step(MATMUL, &[&h, self.w(&p("down.weight")), &down], &[n, ff, d], n * d)]);
        down
    }

    pub(crate) fn run_forward(&self) {
        let g = &self.gpu;
        let n = self.b * self.t;
        let d = self.cfg.d_model;
        let res = self.res.borrow();

        g.submit(&[], &[g.step(EMBED, &[&self.tokens, self.w("tok.weight"), &res[0]], &[d, n], n * d)]);

        let types = self.cfg.layer_types();
        for (l, ty) in types.iter().enumerate() {
            let xres = &res[l];
            let xn1 = g.storage((n * d) as u64);
            g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), xres, self.w(&format!("blocks.{l}.ln1.weight")), &xn1, d, n)]);

            let mixer_out = match ty {
                LayerType::Linear => self.layer_gdn_fwd(l, &xn1, n),
                LayerType::Full => self.layer_gqa_fwd(l, &xn1, n),
            };

            let xmid = g.storage((n * d) as u64);
            g.submit(&[], &[g.step(ADD2, &[xres, &mixer_out, &xmid], &[n * d], n * d)]);

            let xn2 = g.storage((n * d) as u64);
            g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), &xmid, self.w(&format!("blocks.{l}.ln2.weight")), &xn2, d, n)]);

            let mlp_out = self.mlp_fwd(l, &xn2, n);
            g.submit(&[], &[g.step(ADD2, &[&xmid, &mlp_out, &res[l + 1]], &[n * d], n * d)]);
        }

        let xn_final = g.storage((n * d) as u64);
        g.submit(&[], &[rmsnorm_fwd(g, &kernel_ids(), &res[self.cfg.n_layers as usize], self.w("norm.weight"), &xn_final, d, n)]);
        let v = self.cfg.vocab;
        g.submit(&[], &[g.step(MATMUL, &[&xn_final, self.w(self.cfg.head_weight()), &self.logits], &[n, d, v], n * v)]);
    }

    /// The residual stream at layer boundary `l` (`0` = embeddings, `l+1` =
    /// layer `l`'s own output, `cfg.n_layers` = input to the final norm) -
    /// parity-debugging introspection only, valid after a `run_forward()`
    /// call (via [`Self::logits_all`]).
    pub fn debug_res(&self, l: usize) -> Vec<f32> {
        let n = (self.b * self.t) as usize;
        let d = self.cfg.d_model as usize;
        self.gpu.read(&self.res.borrow()[l], n * d)
    }

    pub fn logits_all(&self, tokens: &[u32]) -> Vec<f32> {
        assert_eq!(self.b, 1, "qwen35::logits_all requires b==1 (single sequence)");
        assert_eq!(
            tokens.len() as u32,
            self.t,
            "qwen35::logits_all requires tokens.len() == the configured t (no partial-length prefill in this pass)"
        );
        self.gpu.write(&self.tokens, tokens);
        self.run_forward();
        self.gpu.read(&self.logits, (self.t * self.cfg.vocab) as usize)
    }
}
