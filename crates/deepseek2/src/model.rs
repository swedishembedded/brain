// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The decoder itself: forward AND backward assembled as WGSL dispatches on the
//! shared `gpu_core` engine, wired to the architecture-agnostic `model::Model`
//! seam (so `gradcheck`'s blanket `CheckModel` impl gates it by construction).
//!
//! Both tapes are built ONCE in [`DeepseekV2::new_on`] and stored
//! ([`DeepseekV2::fwd_steps`]/`bwd_steps` are plain dispatch descriptors, not
//! lazily-evaluated closures); `forward()`/`backward()` only submit them and
//! read back the loss. This is `crates/qwen35moe`'s and `crates/glm`'s shape,
//! not a new one.
//!
//! ## What is composed, and from where - this crate adds NO kernel
//!
//! | piece | provider |
//! |---|---|
//! | RMSNorm fwd/bwd | `model::block::{rmsnorm_fwd, rmsnorm_bwd}` |
//! | NEOX (half-split) RoPE fwd/bwd | `model::block::{rope_fwd, rope_bwd}` over `rope_base.wgsl` |
//! | causal MHA fwd/bwd | `model::block::{gqa_fwd, gqa_bwd}` at `n_kv_heads == n_heads` |
//! | routed experts + softmax router | `model::moe::{router_fwd_kind, expert_fwd, moe_layer_bwd}` |
//! | fused unweighted shared expert | `model::moe::{shared_expert_fwd, shared_expert_bwd}`, `None` arm |
//! | SwiGLU, GEMMs, CE, AdamW | the shared kernel set every decoder in this tree uses |
//!
//! **RoPE layout.** `rope_base.wgsl` is HF's `rotate_half` (GPT-NeoX)
//! convention with a configurable base theta - `out[m] = x[m]·cos − x[m+half]
//! ·sin`, `out[m+half] = x[m+half]·cos + x[m]·sin` - which is exactly what
//! llama.cpp's OCR attention branch applies, over the FULL `head_dim`, at base
//! 10000. It is NOT `rope_train.wgsl`'s interleaved-adjacent-pairs convention
//! (what `crates/glm` uses); the two are silently different, not
//! interchangeable, so [`DeepseekV2::new_on`] asserts `rotary_dim == head_dim`
//! rather than accepting a partial-rotary config this kernel cannot express.
//!
//! **RMSNorm epsilon.** `rmsnorm.wgsl`/`rmsnorm_dx.wgsl`/`rms_inv.wgsl` carry a
//! compiled-in `1e-6`, which is the checkpoint's own
//! `attention.layer_norm_rms_epsilon`. [`DeepseekV2::new_on`] asserts the config
//! agrees rather than silently normalising with a different epsilon than the
//! reference (`model::block::rmsnorm_eps_fwd` exists for a model that needs a
//! different one; this one does not).
//!
//! **Router.** `model::moe::RouterKind::Softmax` with `aux_coef = z_coef = 0`:
//! the load-balancing aux loss and the router z-loss are folded into the router
//! *gradient* but are not part of the scalar `forward()` returns, so any nonzero
//! coefficient would make the analytic router gradient inconsistent with a
//! finite difference of the loss this model actually reports. The reference has
//! neither term at inference, and this crate does not implement a training
//! schedule that would want them.
//!
//! **LoRA.** [`DeepseekV2Config::lora`], when `Some`, freezes every OTHER
//! parameter (`Role::Frozen`, weight buffer only -- see [`DeepseekV2::new_on`]'s
//! role-assignment doc) and adds a rank-`r` adapter on the targeted attention
//! projections (`q_proj`/`k_proj`/`v_proj`/`o_proj` -- never an MoE expert or
//! the router). Composed entirely from `matmul`/`matmul_dx`/`matmul_dw`/
//! `axpy`/`grad_scale`, the exact same five kernels
//! `qwen3::model::Qwen::lora_fwd`/`proj_bwd` and
//! `qwen35moe::model::Qwen35::lora_fwd`/`proj_bwd` already dispatch for their
//! own decoders -- this crate adds no kernel for it, only `axpy` gains a
//! pipeline slot (every other kernel LoRA needs was already registered for the
//! base decoder's own forward/backward). See [`DeepseekV2::lora_fwd`]/
//! [`DeepseekV2::lora_bwd`].
//!
//! Not implemented here (see this crate's lib doc for why): INT8, sharding,
//! paged-KV decode. One more known limit worth naming rather than
//! discovering: the embedding and `lm_head` matmuls are **untiled**, so at the
//! real 129280 x 1280 shape their weight binding is ~662 MB - fine on
//! Vulkan/wgpu-native and on the CPU backend, over the GL backend's 128 MB
//! per-binding budget. `qwen3` solves this with `embed_tile`/`matmul_tile` over
//! `block::vocab_tiles`; adopting that here is a drop-in change to two call
//! sites, not a structural one, and is deliberately deferred to the phase that
//! first runs this decoder at real scale.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::block::{self, Gqa, GqaDecodeIds, KernelIds};
use model::moe::{
    self, ExpertBwdScratch, ExpertGrads, MoeActs, MoeIds, MoeIdsBwd, MoeShape, RouterBwdIds, RouterKind, SharedExpertActs, SharedExpertBwdIds,
    SharedExpertBwdScratch, SharedExpertGrads, SharedExpertIds, SharedExpertScratch,
};
use optim::Optim;
use paramstore::{ParamStore, Role};

use crate::config::DeepseekV2Config;

/// Cross-entropy ignore index (masked target positions).
pub const IGNORE: u32 = 0xFFFF_FFFF;

// ---- kernel indices (order matches PIPELINES) ----
const EMBED: usize = 0;
const MATMUL: usize = 1;
const MATMUL_REG3: usize = 2;
const MATMUL_DX: usize = 3;
const MATMUL_DW: usize = 4;
const RMSNORM: usize = 5;
const RMS_INV: usize = 6;
const RMSNORM_DX: usize = 7;
const RMSNORM_DW: usize = 8;
const ROPE: usize = 9;
const ROPE_BWD: usize = 10;
const GQA_SCORES: usize = 11;
const ATTN_SOFTMAX: usize = 12;
const GQA_APPLY: usize = 13;
const GQA_DSCORES: usize = 14;
const GQA_DV: usize = 15;
const GQA_DQ: usize = 16;
const GQA_DK: usize = 17;
const SILU_MUL: usize = 18;
const SILU_DA: usize = 19;
const SILU_DB: usize = 20;
const ROUTER_GATE: usize = 21;
const ROUTER_BWD: usize = 22;
const EXPERT_COUNTS: usize = 23;
const MOE_LINEAR_GATED: usize = 24;
const MOE_LINEAR_GATED_DX: usize = 25;
const MOE_LINEAR_GATED_DW: usize = 26;
const SCALE_ADD: usize = 27;
const SCALE_ADD_DEXP: usize = 28;
const SCALE_ADD_DGATE: usize = 29;
const ADD2: usize = 30;
const CE_VALUE: usize = 31;
const CE_GRAD: usize = 32;
const EMB_BWD: usize = 33;
const ADAMW: usize = 34;
const GRADNORM_SQ: usize = 35;
const GRAD_SCALE: usize = 36;
const CLIP_COEF: usize = 37;
const GRAD_SCALE_BUF: usize = 38;
const SPLICE: usize = 39;
const SPLICE_BWD: usize = 40;
/// LoRA's fused scaled-accumulate (`y += (alpha/rank) * delta`), the one
/// kernel the base decoder's forward/backward never needed on its own --
/// every other LoRA dispatch (`MATMUL`/`MATMUL_DX`/`MATMUL_DW`/`GRAD_SCALE`)
/// reuses a slot already registered above.
const AXPY: usize = 41;
// ---- incremental KV-cache decode kernels (single new token vs the growing
// cache) -- see `DeepseekV2::step`'s doc. Appended so every index above stays
// unchanged. ----
const ROPE_AT: usize = 42;
const ATTN_DECODE_SCORES: usize = 43;
const DECODE_SOFTMAX: usize = 44;
const ATTN_DECODE_APPLY: usize = 45;
const KV_APPEND: usize = 46;

pub const PIPELINES: &[(&str, &str)] = &[
    ("embed", kernels::EMBED),
    ("matmul", kernels::MATMUL),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("rmsnorm", kernels::RMSNORM),
    ("rms_inv", kernels::RMS_INV),
    ("rmsnorm_dx", kernels::RMSNORM_DX),
    ("rmsnorm_dw", kernels::RMSNORM_DW),
    // NEOX / HF `rotate_half` RoPE with a configurable base theta -- NOT the
    // interleaved-pairs `rope_train.wgsl`; see this module's doc.
    ("rope_base", kernels::ROPE_BASE),
    ("rope_base_bwd", kernels::ROPE_BASE_BWD),
    ("gqa_scores", kernels::GQA_SCORES),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("gqa_apply", kernels::GQA_APPLY),
    ("gqa_bwd_dscores", kernels::GQA_BWD_DSCORES),
    ("gqa_bwd_dv", kernels::GQA_BWD_DV),
    ("gqa_bwd_dq", kernels::GQA_BWD_DQ),
    ("gqa_bwd_dk", kernels::GQA_BWD_DK),
    ("silu_mul", kernels::SILU_MUL),
    ("silu_bwd_da", kernels::SILU_BWD_DA),
    ("silu_bwd_db", kernels::SILU_BWD_DB),
    ("router_gate", kernels::ROUTER_GATE),
    ("router_bwd", kernels::ROUTER_BWD),
    ("expert_counts", kernels::EXPERT_COUNTS),
    ("moe_linear_gated", kernels::MOE_LINEAR_GATED),
    ("moe_linear_gated_dx", kernels::MOE_LINEAR_GATED_DX),
    ("moe_linear_gated_dw", kernels::MOE_LINEAR_GATED_DW),
    ("scale_add", kernels::SCALE_ADD),
    ("scale_add_dexp", kernels::SCALE_ADD_DEXP),
    ("scale_add_dgate", kernels::SCALE_ADD_DGATE),
    ("add2", kernels::ADD2),
    ("ce_value", kernels::CE_VALUE_MASKED),
    ("ce_grad", kernels::CE_GRAD_MASKED),
    ("emb_bwd", kernels::EMB_BWD),
    ("adamw", kernels::ADAMW),
    ("gradnorm_sq", kernels::GRADNORM_SQ),
    ("grad_scale", kernels::GRAD_SCALE),
    ("clip_coef", kernels::CLIP_COEF),
    ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
    // Vision-language embedding splice (`model::vlm`) -- appended, so every
    // index above is unchanged.
    ("splice", kernels::SPLICE),
    ("splice_bwd", kernels::SPLICE_BWD),
    // LoRA's scaled-accumulate -- appended after the base set, so every
    // index above is unchanged for a non-LoRA build too.
    ("axpy", kernels::AXPY),
    // Incremental KV-cache decode -- the O(1)-new-token twin of the O(T)
    // batched forward above. `rope_at` rotates at an explicit absolute
    // position (`rope_base` only knows `row % tcols`, unusable for a single
    // new row past position 0); the other four are the same
    // append/score/softmax/apply primitives `crates/gpt`/`crates/glm`/
    // `crates/qwen3` already use for their own decode step.
    ("rope_at", kernels::ROPE_AT),
    ("attn_decode_scores", kernels::ATTN_DECODE_SCORES),
    ("decode_softmax", kernels::DECODE_SOFTMAX),
    ("attn_decode_apply", kernels::ATTN_DECODE_APPLY),
    ("kv_append", kernels::KV_APPEND),
];

fn kernel_ids() -> KernelIds {
    KernelIds {
        rmsnorm: RMSNORM,
        rms_inv: RMS_INV,
        rmsnorm_dx: RMSNORM_DX,
        rmsnorm_dw: RMSNORM_DW,
        rope: ROPE,
        rope_bwd: ROPE_BWD,
        gqa_scores: GQA_SCORES,
        gqa_apply: GQA_APPLY,
        attn_softmax: ATTN_SOFTMAX,
        gqa_dscores: GQA_DSCORES,
        gqa_dv: GQA_DV,
        gqa_dq: GQA_DQ,
        gqa_dk: GQA_DK,
        silu_mul: SILU_MUL,
        silu_da: SILU_DA,
        silu_db: SILU_DB,
    }
}

fn moe_ids() -> MoeIds {
    MoeIds { router_gate: ROUTER_GATE, linear_gated: MOE_LINEAR_GATED, silu_mul: SILU_MUL, scale_add: SCALE_ADD }
}

fn moe_ids_bwd() -> MoeIdsBwd {
    MoeIdsBwd {
        scale_add_dexp: SCALE_ADD_DEXP,
        scale_add_dgate: SCALE_ADD_DGATE,
        silu_da: SILU_DA,
        silu_db: SILU_DB,
        linear_dx: MOE_LINEAR_GATED_DX,
        linear_dw: MOE_LINEAR_GATED_DW,
        linear_gated: true,
    }
}

fn router_bwd_ids() -> RouterBwdIds {
    RouterBwdIds { router_bwd: ROUTER_BWD, expert_counts: Some(EXPERT_COUNTS) }
}

/// Kernel ids for the shared expert. The `sigmoid`/`scale_row`/`add2` triple
/// exists only for `model::moe`'s GATED arm, which this architecture does not
/// have (there is no shared-expert gate tensor in the checkpoint) - those slots
/// carry `usize::MAX` rather than a plausible `0` that would silently dispatch
/// `embed` if the arm selection ever regressed. Same convention `crates/glm`
/// uses for the identical `None`-arm situation.
fn shared_expert_ids() -> SharedExpertIds {
    SharedExpertIds { matmul: MATMUL, silu_mul: SILU_MUL, sigmoid: usize::MAX, scale_row: usize::MAX, add2: ADD2 }
}

fn shared_expert_bwd_ids() -> SharedExpertBwdIds {
    SharedExpertBwdIds {
        linear_dx: MATMUL_DX,
        linear_dw: MATMUL_DW,
        silu_da: SILU_DA,
        silu_db: SILU_DB,
        scale_row: usize::MAX,
        row_dot: usize::MAX,
        sigmoid_bwd: usize::MAX,
    }
}

/// Index of the greatest element, ties resolving to the LOWEST index.
///
/// The strict `>` scanning upwards is llama.cpp's greedy sampler's own rule, so
/// two identical logits cannot make brain and the reference pick differently.
/// A leading `NaN` would win by default (nothing compares greater than it);
/// that is not a case a finite forward produces, and the caller that would care
/// gates finiteness upstream rather than hiding it here.
fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate().skip(1) {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

/// Per-layer MLP variant, holding the activations its backward reads back.
enum Mlp {
    /// Blocks `[0, n_dense_layers)`: a plain SwiGLU MLP at `ffn_hidden` width.
    Dense { gate_pre: DeviceBuffer, up: DeviceBuffer, h: DeviceBuffer },
    /// The rest: softmax-routed experts + the fused unweighted shared expert.
    Moe {
        router_logits: DeviceBuffer,
        gate: DeviceBuffer, // combine weights [n, E], zero outside the top-k
        acts: MoeActs,      // every expert's own gate_pre/up/h/expert_out
        sh_gate: DeviceBuffer,
        sh_up: DeviceBuffer,
        sh_h: DeviceBuffer,
    },
}

struct LayerBufs {
    xn1: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
    xmid: DeviceBuffer,
    xn2: DeviceBuffer,
    mlp: Mlp,
}

/// Per-layer / shared GPU scratch for the incremental single-token KV-cache
/// decode path, plus the persistent K/V cache -- the `O(1)`-new-token twin of
/// [`DeepseekV2::build_forward`]'s `O(T)` batched recompute. Built lazily the
/// first time [`DeepseekV2::step`] runs (inference-only; sized for `n=1` rows
/// and a `cap = self.t` cache), so the training buffers above are never
/// disturbed.
///
/// Every scratch buffer below (attention AND MLP/MoE) is reused across
/// layers -- layers run strictly sequentially in one step, so nothing needs a
/// per-layer copy except the persistent `kcache`/`vcache` (which must survive
/// from one step to the next) and `res` (the residual stream snapshot each
/// layer reads and the next layer writes). This mirrors `crates/gpt`'s
/// `Decode` struct; the MoE scratch (`moe_acts` etc.) is the one addition
/// this architecture's decoder needs over that plain-MLP shape.
struct Decode {
    cap: u32, // K/V cache capacity == self.t (max context)
    tok_id: DeviceBuffer,
    res: Vec<DeviceBuffer>, // [n_layers+1] residual-stream snapshots, [d]
    xn1: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
    proj: DeviceBuffer,
    xmid: DeviceBuffer,
    xn2: DeviceBuffer,
    // dense MLP scratch (shared by every dense layer)
    dense_gate_pre: DeviceBuffer,
    dense_up: DeviceBuffer,
    dense_h: DeviceBuffer,
    // MoE scratch (shared by every MoE layer)
    router_logits: DeviceBuffer,
    gate: DeviceBuffer,
    moe_acts: MoeActs,
    sh_gate: DeviceBuffer,
    sh_up: DeviceBuffer,
    sh_h: DeviceBuffer,
    /// The shared expert's own SwiGLU output, BEFORE it is summed with the
    /// routed accumulator - distinct from `mlp_out` (that sum's destination)
    /// on purpose, exactly like the model-level `sh_out`/`mlp_out` pair:
    /// `add2.wgsl` is out-of-place, and binding one buffer as both a
    /// read-only and a read_write operand in a single dispatch is a wgpu
    /// usage-scope violation, not merely redundant.
    sh_out: DeviceBuffer,
    moe_acc: DeviceBuffer,
    mlp_out: DeviceBuffer,
    xn_final: DeviceBuffer,
    logits: DeviceBuffer,
    kcache: Vec<DeviceBuffer>, // per layer [cap, d]
    vcache: Vec<DeviceBuffer>,
}

pub struct DeepseekV2 {
    pub gpu: Gpu,
    pub cfg: DeepseekV2Config,
    ps: ParamStore,
    opt: Optim,
    b: u32,
    t: u32,
    count: Cell<f32>,

    tokens: DeviceBuffer,
    targets: DeviceBuffer,
    ce_grad_uni: DeviceBuffer,

    /// Vision-language embedding splice, `Some((row0, n_rows))` once
    /// [`DeepseekV2::enable_mm_splice`] has been called. The forward overwrites
    /// residual rows `[row0, row0 + n_rows)` with `img_embeds` right after the
    /// token-embedding gather; the backward moves those rows' gradient into
    /// `d_img_embeds` and ZEROES them in `dres[0]` before `emb_bwd`, so the
    /// placeholder token's embedding row is never trained on the image. Same
    /// seam, same two kernels and same order as `crates/qwen3`.
    mm_splice: Cell<Option<(u32, u32)>>,
    img_embeds: DeviceBuffer,
    d_img_embeds: DeviceBuffer,

    res: Vec<DeviceBuffer>,
    dres: Vec<DeviceBuffer>,
    layers: Vec<LayerBufs>,
    xn_final: DeviceBuffer,
    logits: DeviceBuffer,
    ce_buf: DeviceBuffer,

    // shared forward temporaries
    scores: DeviceBuffer,
    proj: DeviceBuffer,
    moe_acc: DeviceBuffer,
    /// The shared expert's own SwiGLU output, BEFORE it is summed with the
    /// routed accumulator. Distinct from `mlp_out` (that sum's destination) on
    /// purpose: `add2.wgsl` is out-of-place, and binding one buffer as both a
    /// read-only and a read_write binding in a single dispatch is a wgpu
    /// usage-scope violation, not merely redundant.
    sh_out: DeviceBuffer,
    mlp_out: DeviceBuffer,
    /// One-element placeholder for `model::moe::SharedExpertScratch`'s three
    /// gated-arm buffers. The `None` (unweighted) arm this architecture uses
    /// never binds them; allocating three full-size tensors that no dispatch
    /// reads would be pure waste at the real 1280-wide shape.
    gate_stub: DeviceBuffer,

    // shared backward temporaries
    d_logits: DeviceBuffer,
    d_xn: DeviceBuffer,
    d_tmp: DeviceBuffer,
    dxmid: DeviceBuffer,
    d_ctx: DeviceBuffer,
    d_scores: DeviceBuffer,
    d_q: DeviceBuffer,
    d_k: DeviceBuffer,
    d_v: DeviceBuffer,
    d_h: DeviceBuffer,
    d_gate_pre: DeviceBuffer,
    d_up: DeviceBuffer,
    d_router_logits: DeviceBuffer,
    d_gate: DeviceBuffer,
    d_expert_out: DeviceBuffer,
    fe: DeviceBuffer,
    inv: DeviceBuffer,

    // LoRA scratch (rank `r`; `d_model`-wide, since every targetable
    // projection here is square -- see `DeepseekV2Config::param_list`'s doc).
    // Reused across all four attention sites of a layer, and across layers, in
    // one sequential Vec<Step> tape, exactly as `qwen3`/`qwen35moe` reuse
    // their own single set. `.max(1)`-sized so a non-LoRA build still gets a
    // valid (unused, one-element) buffer.
    lora_a: DeviceBuffer,
    lora_da: DeviceBuffer,
    lora_out: DeviceBuffer,

    fwd_steps: Vec<Step>,
    bwd_steps: Vec<Step>,

    // ---- incremental KV-cache decode state (lazily built on first `step`) ----
    dec: RefCell<Option<Decode>>,
    /// Absolute position the next [`DeepseekV2::step`] will decode (cache fill level).
    dec_pos: Cell<u32>,
}

impl DeepseekV2 {
    /// Trainable model on a fresh device.
    pub fn new(cfg: DeepseekV2Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> DeepseekV2 {
        DeepseekV2::new_on(Gpu::new(PIPELINES), cfg, b, t, init, true)
    }

    /// Frozen (forward-only) model on a fresh device: no gradient/AdamW buffers
    /// are allocated and no backward tape is built.
    pub fn new_inference(cfg: DeepseekV2Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> DeepseekV2 {
        DeepseekV2::new_on(Gpu::new(PIPELINES), cfg, b, t, init, false)
    }

    /// Build on an existing device handle (the pooled test device, a placed
    /// card) - see `Gpt::new_on` for the convention.
    pub fn new_on(gpu: Gpu, cfg: DeepseekV2Config, b: u32, t: u32, src: &dyn checkpoint::TensorSource, train: bool) -> DeepseekV2 {
        // `rope_base.wgsl` rotates the WHOLE `head_dim`. A partial-rotary config
        // would need `rope_partial.wgsl` and a different backward; refuse it
        // loudly rather than silently rotating dimensions the reference leaves
        // alone. The real checkpoint's `rope.dimension_count = 0` is already
        // resolved to the full head_dim by the loader.
        assert_eq!(
            cfg.shape.rotary_dim,
            cfg.head_dim(),
            "deepseekv2: rope_base.wgsl rotates the full head_dim; rotary_dim={} != head_dim={}",
            cfg.shape.rotary_dim,
            cfg.head_dim()
        );
        assert!(cfg.head_dim().is_multiple_of(2), "deepseekv2: half-split RoPE needs an even head_dim");
        assert_eq!(cfg.n_kv_heads(), cfg.n_heads(), "deepseekv2 is plain MHA (n_kv_heads == n_heads); got {} vs {}", cfg.n_kv_heads(), cfg.n_heads());
        assert_eq!(cfg.q_dim(), cfg.d_model(), "deepseekv2: the checkpoint's q/k/v/o projections are square [d_model, d_model]");
        // `rmsnorm.wgsl` and its backward carry a compiled-in eps of 1e-6.
        assert!(
            (cfg.rms_eps() - 1e-6).abs() < 1e-9,
            "deepseekv2: rmsnorm.wgsl's epsilon is compiled in at 1e-6 but this config asks for {} -- \
             use model::block::rmsnorm_eps_fwd/bwd rather than normalising with the wrong epsilon",
            cfg.rms_eps()
        );
        assert!(cfg.top_k() >= 1 && cfg.top_k() <= cfg.n_experts(), "deepseekv2: top_k must be in 1..=n_experts");

        // Role assignment, mirroring `qwen3::model.rs`'s and
        // `qwen35moe::model.rs`'s own LoRA branch exactly:
        //  - inference (`!train`): every weight Role::Frozen (weight buffer
        //    only -- no gradient, no AdamW moments).
        //  - LoRA training (`train && cfg.lora.is_some()`): only the
        //    `.lora_a`/`.lora_b` adapter tensors `cfg.param_list()` added for
        //    each targeted leaf are Trainable; every other weight -- including
        //    a LoRA-targeted leaf's own frozen base, the embeddings, the
        //    norms, the MoE router/experts/shared expert and the untied head
        //    -- is Frozen.
        //  - full training (`train && cfg.lora.is_none()`): every weight
        //    Role::Trainable, unchanged from before this field existed.
        let roles: Vec<_> = cfg
            .param_list()
            .into_iter()
            .map(|(n, c)| {
                let role = if !train {
                    Role::Frozen
                } else if cfg.lora.is_some() {
                    if n.ends_with(".lora_a") || n.ends_with(".lora_b") { Role::Trainable } else { Role::Frozen }
                } else {
                    Role::Trainable
                };
                (n, c, role)
            })
            .collect();
        // Split the two candidate costs inside what `deepseek2ocr::model::build`
        // brackets as one "decoder new_on" stage: streaming/uploading the real
        // checkpoint's weights (`ParamStore::new_with_roles_src`, one
        // `raw_words`/`with_tensor_chunks` pull per tensor) vs this
        // constructor's OWN scratch-buffer allocation below (`res`/`dres`,
        // the per-layer attention/MoE buffers, LoRA scratch). Gated on the
        // same `BRAIN_PROFILE` convention `deepseek2ocr::stage_time` and
        // `wgsl-cpu::Jit::new` already print through, so a load-time profile
        // shows all three brackets on one timeline.
        let profile = std::env::var("BRAIN_PROFILE").map(|v| v != "0").unwrap_or(false);
        let t_ps = std::time::Instant::now();
        let n_params = roles.len();
        let ps = ParamStore::new_with_roles_src(&gpu, roles, src);
        if profile {
            eprintln!("deepseekv2: new_on: ParamStore::new_with_roles_src ({n_params} tensors): {:.1} ms", t_ps.elapsed().as_secs_f64() * 1e3);
        }
        let t_scratch = std::time::Instant::now();
        let opt = Optim::new(ADAMW, GRADNORM_SQ, GRAD_SCALE, CLIP_COEF, GRAD_SCALE_BUF);

        let n = (b * t) as u64;
        let d = cfg.d_model() as u64;
        let v = cfg.vocab() as u64;
        let e = cfg.n_experts() as u64;
        let dense_ff = cfg.ffn_hidden() as u64;
        let shared_ff = cfg.shared_ff() as u64;
        let ff_max = cfg.ff_max() as u64;
        let bht2 = (b * cfg.n_heads() * t * t) as u64;
        let st = |x: u64| gpu.storage(x);

        // LoRA scratch: rank `r` (1 when unconfigured), output width `d_model`
        // (every targetable projection is square).
        let lora_r = cfg.lora.as_ref().map(|l| l.rank as u64).unwrap_or(0).max(1);
        let lora_a = st(n * lora_r);
        let lora_da = st(n * lora_r);
        let lora_out = st(n * d);

        let mut res = Vec::new();
        let mut dres = Vec::new();
        for _ in 0..=cfg.n_layers() {
            res.push(st(n * d));
            dres.push(st(n * d));
        }
        let moe_shape = MoeShape {
            rows: b * t,
            d_model: cfg.d_model(),
            moe_ff: cfg.moe_ff(),
            n_experts: cfg.n_experts(),
            top_k: cfg.top_k(),
        };
        let mut layers = Vec::new();
        for l in 0..cfg.n_layers() {
            let mlp = if cfg.is_moe_layer(l) {
                Mlp::Moe {
                    router_logits: st(n * e),
                    gate: st(n * e),
                    acts: MoeActs::new(&gpu, &moe_shape),
                    sh_gate: st(n * shared_ff),
                    sh_up: st(n * shared_ff),
                    sh_h: st(n * shared_ff),
                }
            } else {
                Mlp::Dense { gate_pre: st(n * dense_ff), up: st(n * dense_ff), h: st(n * dense_ff) }
            };
            layers.push(LayerBufs {
                xn1: st(n * d),
                q: st(n * d),
                k: st(n * d),
                v: st(n * d),
                probs: st(bht2),
                ctx: st(n * d),
                xmid: st(n * d),
                xn2: st(n * d),
                mlp,
            });
        }

        let mut m = DeepseekV2 {
            cfg,
            b,
            t,
            count: Cell::new(1.0),
            ps,
            opt,
            tokens: gpu.storage(n),
            targets: gpu.storage(n),
            ce_grad_uni: gpu.uniform_dynamic(4),
            mm_splice: Cell::new(None),
            img_embeds: gpu.storage(1),
            d_img_embeds: gpu.storage(1),
            res,
            dres,
            layers,
            xn_final: st(n * d),
            logits: st(n * v),
            ce_buf: st(n),
            scores: st(bht2),
            proj: st(n * d),
            moe_acc: st(n * d),
            sh_out: st(n * d),
            mlp_out: st(n * d),
            gate_stub: st(1),
            d_logits: st(n * v),
            d_xn: st(n * d),
            d_tmp: st(n * d),
            dxmid: st(n * d),
            d_ctx: st(n * d),
            d_scores: st(bht2),
            d_q: st(n * d),
            d_k: st(n * d),
            d_v: st(n * d),
            d_h: st(n * ff_max),
            d_gate_pre: st(n * ff_max),
            d_up: st(n * ff_max),
            d_router_logits: st(n * e),
            d_gate: st(n * e),
            d_expert_out: st(n * d),
            fe: st(e),
            inv: st(n),
            lora_a,
            lora_da,
            lora_out,
            fwd_steps: Vec::new(),
            bwd_steps: Vec::new(),
            dec: RefCell::new(None),
            dec_pos: Cell::new(0),
            gpu,
        };
        if profile {
            eprintln!("deepseekv2: new_on: scratch buffer allocation: {:.1} ms", t_scratch.elapsed().as_secs_f64() * 1e3);
        }
        let t_tape = std::time::Instant::now();
        m.fwd_steps = m.build_forward(m.b, m.t);
        m.bwd_steps = if train { m.build_backward() } else { Vec::new() };
        if profile {
            eprintln!("deepseekv2: new_on: tape build (fwd{}): {:.1} ms", if train { "+bwd" } else { "" }, t_tape.elapsed().as_secs_f64() * 1e3);
        }
        m
    }

    // ---- small dispatch helpers ----

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }
    fn g(&self, name: &str) -> &DeviceBuffer {
        self.ps.g(name)
    }
    fn trainable(&self, name: &str) -> bool {
        self.ps.grad.contains_key(name)
    }
    fn moe_shape(&self, rows: u32) -> MoeShape {
        MoeShape { rows, d_model: self.cfg.d_model(), moe_ff: self.cfg.moe_ff(), n_experts: self.cfg.n_experts(), top_k: self.cfg.top_k() }
    }
    /// The router this architecture runs: plain softmax top-k, no aux/z loss,
    /// with the checkpoint's own renormalisation and scaling policy. The SAME
    /// value feeds the forward and the backward - `model::moe`'s own doc calls
    /// out that a forward which silently defaulted one of the pair is a
    /// gradient the backward cannot check.
    fn router_kind(&self) -> RouterKind {
        RouterKind::Softmax {
            aux_coef: 0.0,
            z_coef: 0.0,
            norm_topk_prob: self.cfg.norm_topk_prob,
            routed_scaling: self.cfg.routed_scaling,
        }
    }

    /// `out = x·Wᵀ`, size-adaptive between the naive `matmul` and the
    /// register-tiled `matmul_reg3` by `block::pick_gemm`'s measured rule.
    fn mm(&self, s: &mut Vec<Step>, x: &DeviceBuffer, wname: &str, out: &DeviceBuffer, m: u32, k: u32, nout: u32) {
        let (mk, mt) = block::pick_gemm(m as usize, nout as usize, MATMUL, MATMUL_REG3, false);
        s.push(self.gpu.step(mk, &[x, self.w(wname), out], &[m, k, nout], mt));
    }

    /// Backward of `y = x·Wᵀ`: weight grad (when trainable) then input grad into
    /// `dx` (`acc = 0` initialise, `1` accumulate).
    #[allow(clippy::too_many_arguments)]
    fn mm_bwd(&self, s: &mut Vec<Step>, d_out: &DeviceBuffer, x: &DeviceBuffer, wname: &str, dx: &DeviceBuffer, m: u32, k: u32, nout: u32, acc: u32) {
        if self.trainable(wname) {
            s.push(self.gpu.step(MATMUL_DW, &[d_out, x, self.g(wname)], &[m, k, nout], nout * k));
        }
        s.push(self.gpu.step(MATMUL_DX, &[d_out, self.w(wname), dx], &[m, k, nout, acc], m * k));
    }

    /// `Some((rank, alpha/rank))` when `leaf` (e.g. `"q_proj"`) is one of
    /// [`DeepseekV2Config::lora`]'s targets, `None` otherwise (no LoRA
    /// configured at all, or this leaf is not targeted) -- mirrors
    /// `qwen3::model.rs`'s own `lora_for` exactly.
    fn lora_for(&self, leaf: &str) -> Option<(u32, f32)> {
        self.cfg.lora.as_ref().filter(|lc| lc.targets_leaf(leaf)).map(|lc| (lc.rank, lc.alpha / lc.rank as f32))
    }

    /// Forward LoRA delta for a targeted linear: `y += (alpha/r)·(x·Aᵀ)·Bᵀ`, in
    /// place on `y` (which [`Self::mm`] has already written `x·Wᵀ` into for
    /// the SAME `wname`, immediately before this call at every call site). A
    /// no-op when `leaf` is not targeted -- mirrors `qwen3::model.rs`'s own
    /// `lora_fwd` exactly (same two-matmul + `AXPY` fusion, this file's own
    /// persistent `lora_a`/`lora_out` scratch).
    fn lora_fwd(&self, s: &mut Vec<Step>, leaf: &str, x: &DeviceBuffer, wname: &str, y: &DeviceBuffer, m: u32, k: u32, nout: u32) {
        let Some((r, scale)) = self.lora_for(leaf) else { return };
        let a = format!("{wname}.lora_a");
        let bnm = format!("{wname}.lora_b");
        s.push(self.gpu.step(MATMUL, &[x, self.w(&a), &self.lora_a], &[m, k, r], m * r));
        s.push(self.gpu.step(MATMUL, &[&self.lora_a, self.w(&bnm), &self.lora_out], &[m, r, nout], m * nout));
        s.push(self.gpu.step(AXPY, &[y, &self.lora_out], &[m * nout, f(scale)], m * nout));
    }

    /// Backward for the OPTIONAL LoRA delta on a targeted linear, split out
    /// from [`Self::mm_bwd`] rather than folded into it: the base gradient
    /// (weight grad skipped automatically when the base is `Role::Frozen`,
    /// via [`Self::trainable`]) is unconditional and already correct; this
    /// adds the adapter's own `gA`/`gB` and its share of `dx` on top, always
    /// ACCUMULATING (`acc = 1`) into `dx` because [`Self::mm_bwd`]'s own write
    /// for the SAME buffer always runs first, at every one of this decoder's
    /// four call sites. No-op when `leaf` is not targeted. Mirrors
    /// `qwen3::model.rs`'s `proj_bwd` LoRA branch exactly (same seven-step
    /// derivation: `a = x·Aᵀ`, `gB += (alpha/r)·d_outᵀ·a`, `da = (alpha/r)·
    /// d_out·B`, `gA += daᵀ·x`, `dx += da·A`), over this crate's own
    /// `MATMUL`/`MATMUL_DX`/`MATMUL_DW`/`GRAD_SCALE` kernel ids.
    #[allow(clippy::too_many_arguments)]
    fn lora_bwd(&self, s: &mut Vec<Step>, leaf: &str, d_out: &DeviceBuffer, x: &DeviceBuffer, wname: &str, dx: &DeviceBuffer, m: u32, k: u32, nout: u32) {
        let Some((r, scale)) = self.lora_for(leaf) else { return };
        let a = format!("{wname}.lora_a");
        let bnm = format!("{wname}.lora_b");
        // a = x·Aᵀ ; gB += scale·d_outᵀ·a  (scale folded into `a`, private scratch)
        s.push(self.gpu.step(MATMUL, &[x, self.w(&a), &self.lora_a], &[m, k, r], m * r));
        s.push(self.gpu.step(GRAD_SCALE, &[&self.lora_a], &[m * r, f(scale)], m * r));
        s.push(self.gpu.step(MATMUL_DW, &[d_out, &self.lora_a, self.g(&bnm)], &[m, r, nout], nout * r));
        // da = scale·(d_out·B) ; gA += daᵀ·x ; dx += da·A
        s.push(self.gpu.step(MATMUL_DX, &[d_out, self.w(&bnm), &self.lora_da], &[m, r, nout, 0], m * r));
        s.push(self.gpu.step(GRAD_SCALE, &[&self.lora_da], &[m * r, f(scale)], m * r));
        s.push(self.gpu.step(MATMUL_DW, &[&self.lora_da, x, self.g(&a)], &[m, k, r], r * k));
        s.push(self.gpu.step(MATMUL_DX, &[&self.lora_da, self.w(&a), dx], &[m, k, r, 1], m * k));
    }

    fn norm_fwd(&self, s: &mut Vec<Step>, x: &DeviceBuffer, wname: &str, out: &DeviceBuffer, dim: u32, rows: u32) {
        s.push(block::rmsnorm_fwd(&self.gpu, &kernel_ids(), x, self.w(wname), out, dim, rows));
    }

    fn norm_bwd(&self, s: &mut Vec<Step>, x: &DeviceBuffer, wname: &str, dy: &DeviceBuffer, dx: &DeviceBuffer, dim: u32, rows: u32) {
        let gw = self.trainable(wname).then(|| self.g(wname));
        s.extend(block::rmsnorm_bwd(&self.gpu, &kernel_ids(), x, self.w(wname), dy, dx, &self.inv, gw, dim, rows));
    }

    pub fn set_batch(&self, x: &[u32], y: &[u32]) {
        self.gpu.write(&self.tokens, bytemuck::cast_slice(x));
        self.gpu.write(&self.targets, bytemuck::cast_slice(y));
        let c = y.iter().filter(|&&v| v != IGNORE).count();
        self.count.set(c.max(1) as f32);
    }

    // ---- forward tape ----

    fn build_forward(&self, b_use: u32, t_use: u32) -> Vec<Step> {
        let c = &self.cfg;
        let n = b_use * t_use;
        let d = c.d_model();
        let v = c.vocab();
        let e = c.n_experts();
        let hd = c.head_dim();
        let nh = c.n_heads();
        let dense_ff = c.ffn_hidden();
        let shared_ff = c.shared_ff();
        let ids = kernel_ids();
        let ga = Gqa { b: b_use, t: t_use, n_heads: nh, n_kv_heads: c.n_kv_heads(), head_dim: hd };
        let shape = self.moe_shape(n);
        let mut s: Vec<Step> = Vec::new();

        s.push(self.gpu.step(EMBED, &[&self.tokens, self.w("tok.weight"), &self.res[0]], &[d, n], n * d));
        // Vision-language splice: overwrite the image-placeholder rows of the
        // freshly-gathered residual stream with the projected image tokens.
        if let Some((row0, n_rows)) = self.mm_splice.get() {
            s.push(model::vlm::splice_fwd(&self.gpu, SPLICE, &self.img_embeds, &self.res[0], row0 * d, n_rows * d));
        }

        for l in 0..c.n_layers() as usize {
            let lb = &self.layers[l];
            let p = |name: &str| format!("blocks.{l}.{name}");

            // ---- plain MHA (no bias anywhere; q/k/v/o are all square) ----
            self.norm_fwd(&mut s, &self.res[l], &p("ln1.weight"), &lb.xn1, d, n);
            self.mm(&mut s, &lb.xn1, &p("self_attn.q_proj.weight"), &lb.q, n, d, d);
            self.lora_fwd(&mut s, "q_proj", &lb.xn1, &p("self_attn.q_proj.weight"), &lb.q, n, d, d);
            self.mm(&mut s, &lb.xn1, &p("self_attn.k_proj.weight"), &lb.k, n, d, d);
            self.lora_fwd(&mut s, "k_proj", &lb.xn1, &p("self_attn.k_proj.weight"), &lb.k, n, d, d);
            self.mm(&mut s, &lb.xn1, &p("self_attn.v_proj.weight"), &lb.v, n, d, d);
            self.lora_fwd(&mut s, "v_proj", &lb.xn1, &p("self_attn.v_proj.weight"), &lb.v, n, d, d);
            s.push(block::rope_fwd(&self.gpu, &ids, &lb.q, n, nh, hd, d, t_use, c.rope_theta()));
            s.push(block::rope_fwd(&self.gpu, &ids, &lb.k, n, nh, hd, d, t_use, c.rope_theta()));
            s.extend(block::gqa_fwd(&self.gpu, &ids, &ga, &lb.q, &lb.k, &lb.v, &self.scores, &lb.probs, &lb.ctx));
            self.mm(&mut s, &lb.ctx, &p("self_attn.o_proj.weight"), &self.proj, n, d, d);
            self.lora_fwd(&mut s, "o_proj", &lb.ctx, &p("self_attn.o_proj.weight"), &self.proj, n, d, d);
            s.push(self.gpu.step(ADD2, &[&self.res[l], &self.proj, &lb.xmid], &[n * d], n * d));

            // ---- MLP: dense (leading blocks) or MoE ----
            self.norm_fwd(&mut s, &lb.xmid, &p("ln2.weight"), &lb.xn2, d, n);
            match &lb.mlp {
                Mlp::Dense { gate_pre, up, h } => {
                    self.mm(&mut s, &lb.xn2, &p("mlp.gate.weight"), gate_pre, n, d, dense_ff);
                    self.mm(&mut s, &lb.xn2, &p("mlp.up.weight"), up, n, d, dense_ff);
                    s.push(block::swiglu_fwd(&self.gpu, &ids, gate_pre, up, h, n * dense_ff));
                    self.mm(&mut s, h, &p("mlp.down.weight"), &self.mlp_out, n, dense_ff, d);
                }
                Mlp::Moe { router_logits, gate, acts, sh_gate, sh_up, sh_h } => {
                    // Router: logits -> a dense [n, E] gate, nonzero only at the
                    // selected experts, under THIS config's norm/scale policy.
                    self.mm(&mut s, &lb.xn2, &p("mlp.router.weight"), router_logits, n, d, e);
                    s.push(moe::router_fwd_kind(&self.gpu, &moe_ids(), self.router_kind(), &shape, router_logits, None, gate, None));
                    // Routed experts: `moe_linear_gated` skips a row this expert
                    // did not win before the K-reduction, so the cost is
                    // proportional to the rows actually routed here.
                    for ei in 0..e as usize {
                        let ep = |nm: &str| format!("blocks.{l}.mlp.experts.{ei}.{nm}");
                        s.extend(moe::expert_fwd(
                            &self.gpu,
                            &moe_ids(),
                            &shape,
                            &lb.xn2,
                            gate,
                            self.w(&ep("gate.weight")),
                            self.w(&ep("up.weight")),
                            self.w(&ep("down.weight")),
                            &acts.at(ei),
                            &self.moe_acc,
                            ei as u32,
                            ei != 0,
                        ));
                    }
                    // The fused shared experts, added UNWEIGHTED (`None`): there
                    // is no shared-expert gate tensor in this checkpoint, and an
                    // unweighted sum of SwiGLU experts IS one SwiGLU of the
                    // summed width -- so this is one matmul triple, not two.
                    s.extend(moe::shared_expert_fwd(
                        &self.gpu,
                        &shared_expert_ids(),
                        n,
                        d,
                        shared_ff,
                        &lb.xn2,
                        self.w(&p("mlp.shared.gate.weight")),
                        self.w(&p("mlp.shared.up.weight")),
                        self.w(&p("mlp.shared.down.weight")),
                        None,
                        &SharedExpertScratch {
                            gate_pre: sh_gate,
                            up: sh_up,
                            h: sh_h,
                            mlp_out: &self.sh_out,
                            gate_logits: &self.gate_stub,
                            gate_scalar: &self.gate_stub,
                            scaled: &self.gate_stub,
                        },
                        &self.moe_acc,
                        &self.mlp_out,
                    ));
                }
            }
            s.push(self.gpu.step(ADD2, &[&lb.xmid, &self.mlp_out, &self.res[l + 1]], &[n * d], n * d));
        }

        // final norm + untied lm_head + masked CE
        let last = c.n_layers() as usize;
        self.norm_fwd(&mut s, &self.res[last], "norm.weight", &self.xn_final, d, n);
        self.mm(&mut s, &self.xn_final, c.head_weight(), &self.logits, n, d, v);
        s.push(self.gpu.step(CE_VALUE, &[&self.logits, &self.targets, &self.ce_buf], &[n, v, IGNORE], n));
        s
    }

    // ---- backward tape ----

    fn build_backward(&self) -> Vec<Step> {
        let c = &self.cfg;
        let (b_use, t_use) = (self.b, self.t);
        let n = b_use * t_use;
        let d = c.d_model();
        let v = c.vocab();
        let e = c.n_experts();
        let hd = c.head_dim();
        let nh = c.n_heads();
        let dense_ff = c.ffn_hidden();
        let shared_ff = c.shared_ff();
        let head = c.head_weight();
        let ids = kernel_ids();
        let ga = Gqa { b: b_use, t: t_use, n_heads: nh, n_kv_heads: c.n_kv_heads(), head_dim: hd };
        let shape = self.moe_shape(n);
        let mut s: Vec<Step> = Vec::new();

        // ---- head + final norm ----
        s.push(self.gpu.step_buf(CE_GRAD, &self.ce_grad_uni, &[&self.logits, &self.targets, &self.d_logits], n * v));
        self.mm_bwd(&mut s, &self.d_logits, &self.xn_final, head, &self.d_xn, n, d, v, 0);
        let last = c.n_layers() as usize;
        self.norm_bwd(&mut s, &self.res[last], "norm.weight", &self.d_xn, &self.dres[last], d, n);

        for l in (0..c.n_layers() as usize).rev() {
            let lb = &self.layers[l];
            let p = |name: &str| format!("blocks.{l}.{name}");

            // ===== MLP backward (output grad = dres[l+1]) -> d_xn (grad wrt xn2) =====
            match &lb.mlp {
                Mlp::Dense { gate_pre, up, h } => {
                    self.mm_bwd(&mut s, &self.dres[l + 1], h, &p("mlp.down.weight"), &self.d_h, n, dense_ff, d, 0);
                    s.extend(block::swiglu_bwd(&self.gpu, &ids, gate_pre, up, &self.d_h, &self.d_gate_pre, &self.d_up, n * dense_ff));
                    self.mm_bwd(&mut s, &self.d_up, &lb.xn2, &p("mlp.up.weight"), &self.d_xn, n, d, dense_ff, 0);
                    self.mm_bwd(&mut s, &self.d_gate_pre, &lb.xn2, &p("mlp.gate.weight"), &self.d_xn, n, d, dense_ff, 1);
                }
                Mlp::Moe { router_logits, gate, acts, sh_gate, sh_up, sh_h } => {
                    // The forward's last MoE step is `out = routed_acc + shared`,
                    // so `dres[l+1]` is the gradient w.r.t. BOTH summands
                    // unchanged -- no kernel is needed to split it, which is why
                    // `shared_expert_bwd` takes no `d_acc` output and
                    // `moe_layer_bwd` is handed the same buffer as `d_moe_acc`.
                    //
                    // Shared expert FIRST, with `accumulate = false`: it owns
                    // `d_xn`'s first touch, so the routed half's router-weight
                    // dX below accumulates on top (acc = 1) rather than clobbering it.
                    let shw = |nm: &str| self.w(&p(&format!("mlp.shared.{nm}")));
                    let shg = |nm: &str| {
                        let full = p(&format!("mlp.shared.{nm}"));
                        self.trainable(&full).then(|| self.g(&full))
                    };
                    s.extend(moe::shared_expert_bwd(
                        &self.gpu,
                        &shared_expert_bwd_ids(),
                        n,
                        d,
                        shared_ff,
                        &lb.xn2,
                        shw("gate.weight"),
                        shw("up.weight"),
                        shw("down.weight"),
                        None, // unweighted: no sigmoid shared-expert gate exists
                        &SharedExpertGrads { gate_w: shg("gate.weight"), up_w: shg("up.weight"), down_w: shg("down.weight"), shared_gate_w: None },
                        &SharedExpertActs { gate_pre: sh_gate, up: sh_up, h: sh_h, mlp_out: None, gate_logits: None, gate_scalar: None },
                        &SharedExpertBwdScratch {
                            d_h: &self.d_h,
                            d_gate_pre: &self.d_gate_pre,
                            d_up: &self.d_up,
                            d_mlp_out: None,
                            d_gate_scalar: None,
                            d_gate_logits: None,
                        },
                        &self.dres[l + 1],
                        &self.d_xn,
                        false,
                    ));
                    // The router weight's own dense-linear backward: `moe_layer_bwd`
                    // does not own this GEMM's kernel choice, it only fixes WHERE
                    // in the phase order it runs (after the router kernel's
                    // backward, before the experts').
                    let mut router_weight_bwd: Vec<Step> = Vec::new();
                    self.mm_bwd(&mut router_weight_bwd, &self.d_router_logits, &lb.xn2, &p("mlp.router.weight"), &self.d_xn, n, d, e, 1);

                    let expert_weights: Vec<(DeviceBuffer, DeviceBuffer, DeviceBuffer)> = (0..e as usize)
                        .map(|ei| {
                            let ep = |nm: &str| format!("blocks.{l}.mlp.experts.{ei}.{nm}");
                            (self.w(&ep("gate.weight")).clone(), self.w(&ep("up.weight")).clone(), self.w(&ep("down.weight")).clone())
                        })
                        .collect();
                    let expert_grads: Vec<ExpertGrads> = (0..e as usize)
                        .map(|ei| {
                            let gr = |nm: &str| {
                                let full = format!("blocks.{l}.mlp.experts.{ei}.{nm}");
                                self.trainable(&full).then(|| self.g(&full))
                            };
                            ExpertGrads { gate_w: gr("gate.weight"), up_w: gr("up.weight"), down_w: gr("down.weight") }
                        })
                        .collect();
                    s.extend(moe::moe_layer_bwd(
                        &self.gpu,
                        &router_bwd_ids(),
                        &moe_ids_bwd(),
                        self.router_kind(),
                        &shape,
                        router_logits,
                        gate,
                        Some(&self.fe),
                        &self.d_gate,
                        &self.d_router_logits,
                        &router_weight_bwd,
                        &lb.xn2,
                        &expert_weights,
                        &expert_grads,
                        acts,
                        &ExpertBwdScratch { d_expert_out: &self.d_expert_out, d_h: &self.d_h, d_gate_pre: &self.d_gate_pre, d_up: &self.d_up },
                        &self.dres[l + 1],
                        &self.d_xn,
                    ));
                }
            }
            // ln2 backward -> d_tmp ; dxmid = dres[l+1] + d_tmp
            self.norm_bwd(&mut s, &lb.xmid, &p("ln2.weight"), &self.d_xn, &self.d_tmp, d, n);
            s.push(self.gpu.step(ADD2, &[&self.dres[l + 1], &self.d_tmp, &self.dxmid], &[n * d], n * d));

            // ===== MHA backward (output grad = dxmid) =====
            self.mm_bwd(&mut s, &self.dxmid, &lb.ctx, &p("self_attn.o_proj.weight"), &self.d_ctx, n, d, d, 0);
            self.lora_bwd(&mut s, "o_proj", &self.dxmid, &lb.ctx, &p("self_attn.o_proj.weight"), &self.d_ctx, n, d, d);
            s.extend(block::gqa_bwd(&self.gpu, &ids, &ga, &lb.q, &lb.k, &lb.v, &lb.probs, &self.d_ctx, &self.d_scores, &self.d_q, &self.d_k, &self.d_v));
            // RoPE is orthogonal per (position, channel pair), so its backward is
            // the inverse rotation applied in place on the grads.
            s.push(block::rope_bwd(&self.gpu, &ids, &self.d_q, n, nh, hd, d, t_use, c.rope_theta()));
            s.push(block::rope_bwd(&self.gpu, &ids, &self.d_k, n, nh, hd, d, t_use, c.rope_theta()));
            self.mm_bwd(&mut s, &self.d_v, &lb.xn1, &p("self_attn.v_proj.weight"), &self.d_xn, n, d, d, 0);
            self.lora_bwd(&mut s, "v_proj", &self.d_v, &lb.xn1, &p("self_attn.v_proj.weight"), &self.d_xn, n, d, d);
            self.mm_bwd(&mut s, &self.d_k, &lb.xn1, &p("self_attn.k_proj.weight"), &self.d_xn, n, d, d, 1);
            self.lora_bwd(&mut s, "k_proj", &self.d_k, &lb.xn1, &p("self_attn.k_proj.weight"), &self.d_xn, n, d, d);
            self.mm_bwd(&mut s, &self.d_q, &lb.xn1, &p("self_attn.q_proj.weight"), &self.d_xn, n, d, d, 1);
            self.lora_bwd(&mut s, "q_proj", &self.d_q, &lb.xn1, &p("self_attn.q_proj.weight"), &self.d_xn, n, d, d);
            // ln1 backward -> d_tmp ; dres[l] = dxmid + d_tmp
            self.norm_bwd(&mut s, &self.res[l], &p("ln1.weight"), &self.d_xn, &self.d_tmp, d, n);
            s.push(self.gpu.step(ADD2, &[&self.dxmid, &self.d_tmp, &self.dres[l]], &[n * d], n * d));
        }

        // Vision-language splice backward: move the image rows' residual grad into
        // `d_img_embeds` and ZERO them in dres[0] BEFORE emb_bwd, so the scatter
        // below never trains the placeholder token's embedding row.
        if let Some((row0, n_rows)) = self.mm_splice.get() {
            s.push(model::vlm::splice_bwd(&self.gpu, SPLICE_BWD, &self.dres[0], &self.d_img_embeds, row0 * d, n_rows * d));
        }

        // embedding backward (untied head: only the embedding path writes tok.weight)
        if self.trainable("tok.weight") {
            s.push(self.gpu.step(EMB_BWD, &[&self.tokens, &self.dres[0], self.g("tok.weight")], &[n, d, v], v * d));
        }
        s
    }

    // ---- run ----

    pub fn forward(&self) -> f32 {
        self.gpu.submit(&[], &self.fwd_steps);
        let n = (self.b * self.t) as usize;
        self.gpu.read(&self.ce_buf, n).iter().sum::<f32>() / self.count.get()
    }

    pub fn backward(&self) {
        assert!(!self.bwd_steps.is_empty(), "DeepseekV2::backward on a forward-only (inference) instance");
        let n = self.b * self.t;
        self.gpu.write(&self.ce_grad_uni, &[n, self.cfg.vocab(), IGNORE, f(self.count.get())]);
        self.gpu.submit(&[], &self.bwd_steps);
    }

    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }
    pub fn poll_wait(&self) {
        self.gpu.poll_wait();
    }
    pub fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        self.opt.step(&self.gpu, &self.ps, t, lr, wd, 0.9, 0.95, 1e-8, clip, extra_scale);
    }
    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.ps.read_grad(&self.gpu, name)
    }
    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        self.ps.read_weight(&self.gpu, name)
    }
    pub fn write_weight(&self, name: &str, data: &[f32]) {
        self.gpu.write(self.w(name), bytemuck::cast_slice(data));
    }
    pub fn param_names(&self) -> Vec<String> {
        self.ps.trainable.iter().map(|(n, _)| n.clone()).collect()
    }
    pub fn ctx_len(&self) -> usize {
        self.t as usize
    }

    /// Per-position logits for one sequence (`b` must be 1, `len <= t`).
    pub fn logits_all(&self, tokens: &[u32]) -> Vec<f32> {
        let t_use = tokens.len() as u32;
        assert!(t_use <= self.t && self.b == 1, "deepseekv2 decoder sized too small for logits_all");
        // The rebuilt tape carries the splice too, so a short sequence must still
        // CONTAIN the image run -- otherwise `splice_fwd` writes past `res[0]`'s
        // live rows into a region no layer reads, silently dropping the image.
        if let Some((row0, n_rows)) = self.mm_splice.get() {
            assert!(row0 + n_rows <= t_use, "logits_all over {t_use} tokens does not contain the spliced image rows [{row0}, {})", row0 + n_rows);
        }
        let ignore = vec![IGNORE; t_use as usize];
        self.set_batch(tokens, &ignore);
        let s = self.build_forward(1, t_use);
        self.gpu.submit(&[], &s);
        self.gpu.read(&self.logits, (t_use * self.cfg.vocab()) as usize)
    }

    /// **Greedy autoregressive decode**: `prompt_ids` followed by `n_new`
    /// argmax-selected continuations, returned as ONE sequence (the prompt is
    /// included, so index `i + 1` of the result is the model's greedy
    /// prediction after processing `result[0..=i]`).
    ///
    /// This is the `O(T²)`-recompute tier of the two-tier arrangement
    /// `crates/gpt`/`crates/glm`/`crates/qwen3` keep (`sample::generate` vs
    /// `sample::generate_kv`); only the recompute half exists here, because
    /// incremental/paged-KV decode is explicitly out of this crate's scope (see
    /// the lib doc) and a *correctness* proof of the decode loop wants the tier
    /// that shares its graph with [`Self::forward`], not a second graph.
    ///
    /// Each step re-runs [`Self::logits_all`] over the whole sequence so far, so
    /// the RoPE positions and the causal mask are re-derived from the grown
    /// length **by the same tape builder the single-shot forward uses**. There
    /// is no cache to advance, hence nothing that can silently drift out of step
    /// with the forward this crate's stage parity already gates.
    ///
    /// The instance must be sized for the FINAL length (`prompt_ids.len() +
    /// n_new <= ctx_len()`, `b == 1`) - nothing here reallocates, so a decode
    /// that would outgrow the sized context is a panic, not a silent truncation
    /// of the context window.
    ///
    /// Ties break to the **lowest** token id (strict `>` scanning upwards),
    /// which is llama.cpp's own greedy sampler's rule - so a tie cannot make the
    /// two references disagree.
    ///
    /// Returns everything at once; [`Self::generate_greedy_cb`] is the same
    /// loop with a per-token callback, for a caller that must observe tokens as
    /// they are produced.
    pub fn generate_greedy(&self, prompt_ids: &[u32], n_new: u32) -> Vec<u32> {
        self.generate_greedy_cb(prompt_ids, n_new, |_| {})
    }

    /// [`Self::generate_greedy`] with a per-token callback - the seam a served
    /// path emits REAL streaming deltas from, instead of one emission around
    /// the whole decode (`qwen3vl::Qwen3Vl::generate_cb` is the same seam on the
    /// same argument).
    ///
    /// `on_token` fires **once per generated token**, immediately after that
    /// token is chosen and appended, so it is called exactly `n_new` times and
    /// never for a prompt id - the prompt is copied, not predicted. The
    /// returned sequence still carries the prompt ahead of those ids, so the
    /// callback's stream is `result[prompt_ids.len()..]`, not `result`.
    ///
    /// The loop is otherwise byte-for-byte [`Self::generate_greedy`]'s: same
    /// `O(T²)` recompute, same tie-break, same sized-context assertions.
    pub fn generate_greedy_cb(&self, prompt_ids: &[u32], n_new: u32, mut on_token: impl FnMut(u32)) -> Vec<u32> {
        assert!(!prompt_ids.is_empty(), "deepseekv2: greedy decode needs at least one prompt token");
        let total = prompt_ids.len() + n_new as usize;
        assert!(
            total <= self.t as usize,
            "deepseekv2: greedy decode of {} prompt + {n_new} new tokens needs a context of {total}, but this instance is sized for {}",
            prompt_ids.len(),
            self.t
        );
        let vocab = self.cfg.vocab() as usize;
        let mut ids = Vec::with_capacity(total);
        ids.extend_from_slice(prompt_ids);
        for _ in 0..n_new {
            let logits = self.logits_all(&ids);
            let next = argmax(&logits[logits.len() - vocab..]) as u32;
            ids.push(next);
            on_token(next);
        }
        ids
    }

    // ---- incremental KV-cache decode (the O(T) fast path) ----

    /// Reset the incremental KV cache to an empty sequence (next [`Self::step`]
    /// is absolute position 0).
    pub fn reset_cache(&self) {
        self.dec_pos.set(0);
    }

    /// The absolute position the next [`Self::step`] will decode (cache fill level).
    pub fn cache_pos(&self) -> u32 {
        self.dec_pos.get()
    }

    /// Build the lazy decode state (buffers + K/V cache) the first time it is
    /// needed.
    fn ensure_decode(&self) {
        if self.dec.borrow().is_some() {
            return;
        }
        let c = &self.cfg;
        let d = c.d_model() as u64;
        let e = c.n_experts() as u64;
        let dense_ff = c.ffn_hidden() as u64;
        let shared_ff = c.shared_ff() as u64;
        let vocab = c.vocab() as u64;
        let nh = c.n_heads() as u64;
        let cap = self.t;
        let g = &self.gpu;
        let st = |x: u64| g.storage(x);
        let idbuf = || g.buffer("dec_tok_id", 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST);

        let mut res = Vec::new();
        for _ in 0..=c.n_layers() as usize {
            res.push(st(d));
        }
        let (mut kcache, mut vcache) = (Vec::new(), Vec::new());
        for _ in 0..c.n_layers() as usize {
            kcache.push(st(cap as u64 * d));
            vcache.push(st(cap as u64 * d));
        }
        let moe_shape1 = MoeShape { rows: 1, d_model: c.d_model(), moe_ff: c.moe_ff(), n_experts: c.n_experts(), top_k: c.top_k() };

        let dec = Decode {
            cap,
            tok_id: idbuf(),
            res,
            xn1: st(d),
            q: st(d),
            k: st(d),
            v: st(d),
            scores: st(nh * cap as u64),
            probs: st(nh * cap as u64),
            ctx: st(d),
            proj: st(d),
            xmid: st(d),
            xn2: st(d),
            dense_gate_pre: st(dense_ff),
            dense_up: st(dense_ff),
            dense_h: st(dense_ff),
            router_logits: st(e),
            gate: st(e),
            moe_acts: MoeActs::new(g, &moe_shape1),
            sh_gate: st(shared_ff),
            sh_up: st(shared_ff),
            sh_h: st(shared_ff),
            sh_out: st(d),
            moe_acc: st(d),
            mlp_out: st(d),
            xn_final: st(d),
            logits: st(vocab),
            kcache,
            vcache,
        };
        *self.dec.borrow_mut() = Some(dec);
    }

    /// Bulk-fill the persistent KV cache from a just-run batched forward's
    /// resident per-layer `k`/`v` buffers ([`Self::layers`]) -- the prefill
    /// half of [`block::kv_cache_fill`]'s documented pattern: after a normal
    /// batched pass over the prompt's `n` positions, one flat `kv_append`
    /// dispatch per layer per buffer copies rows `0..n` into the cache, and
    /// decode steps continue from `pos = n`. Must be called immediately after
    /// a [`Self::logits_all`]/[`Self::build_forward`] call over EXACTLY the
    /// tokens the cache should hold -- `self.layers[l].k`/`.v` hold whatever
    /// that most recent forward wrote, nothing more.
    fn fill_cache_from_prefill(&self, n: u32) {
        self.ensure_decode();
        let c = &self.cfg;
        let (nh, hd) = (c.n_heads(), c.head_dim());
        let dec_ref = self.dec.borrow();
        let dec = dec_ref.as_ref().unwrap();
        let g = &self.gpu;
        let mut s: Vec<Step> = Vec::new();
        for l in 0..c.n_layers() as usize {
            s.push(block::kv_cache_fill(g, KV_APPEND, &self.layers[l].k, &dec.kcache[l], n, nh, hd));
            s.push(block::kv_cache_fill(g, KV_APPEND, &self.layers[l].v, &dec.vcache[l], n, nh, hd));
        }
        g.submit(&[], &s);
    }

    /// **Incremental KV-cache decode** of a single token id at the current
    /// cache position, returning that new token's logits (`[vocab]`). This is
    /// the `O(1)`-per-token twin of [`Self::logits_all`]'s `O(T)` batched
    /// recompute: only the new token's Q/K/V are projected and only the
    /// new token's row runs through the MoE/dense FFN; its K/V are appended to
    /// the persistent per-layer cache and a single query attends over
    /// positions `0..=pos`. Expressed entirely in the existing WGSL op set
    /// (`model::block::gqa_decode_step` plus `kernels::ROPE_AT` for the
    /// explicit-position rotation `rope_base.wgsl` cannot express at a single
    /// row past position 0), so it runs on whatever backend `Gpu` selected.
    pub fn step(&self, token_id: u32) -> Vec<f32> {
        self.ensure_decode();
        let pos = self.dec_pos.get();
        let logits = self.decode_at(token_id, pos);
        self.dec_pos.set(pos + 1);
        logits
    }

    /// Record + run the incremental decode tape for one token at absolute `pos`.
    fn decode_at(&self, token_id: u32, pos: u32) -> Vec<f32> {
        let c = &self.cfg;
        let d = c.d_model();
        let nh = c.n_heads();
        let hd = c.head_dim();
        let e = c.n_experts();
        let dense_ff = c.ffn_hidden();
        let shared_ff = c.shared_ff();
        let vocab = c.vocab() as usize;
        let theta = c.rope_theta();
        let g = &self.gpu;
        let ids = kernel_ids();
        let moe_kernel_ids = moe_ids();
        let shared_ids = shared_expert_ids();
        let decode_ids = GqaDecodeIds { kv_append: KV_APPEND, attn_decode_scores: ATTN_DECODE_SCORES, decode_softmax: DECODE_SOFTMAX, attn_decode_apply: ATTN_DECODE_APPLY };
        let shape1 = MoeShape { rows: 1, d_model: d, moe_ff: c.moe_ff(), n_experts: e, top_k: c.top_k() };

        let dec_ref = self.dec.borrow();
        let dec = dec_ref.as_ref().unwrap();
        let cap = dec.cap;
        assert!(pos < cap, "deepseekv2 decode pos {pos} exceeds the sized context {cap}");

        g.write(&dec.tok_id, &[token_id]);
        let mut s: Vec<Step> = Vec::new();
        s.push(g.step(EMBED, &[&dec.tok_id, self.w("tok.weight"), &dec.res[0]], &[d, 1], d));

        for l in 0..c.n_layers() as usize {
            let p = |name: &str| format!("blocks.{l}.{name}");

            // ---- plain MHA decode step: LN -> q/k/v -> RoPE@pos -> attend over cache ----
            self.norm_fwd(&mut s, &dec.res[l], &p("ln1.weight"), &dec.xn1, d, 1);
            self.mm(&mut s, &dec.xn1, &p("self_attn.q_proj.weight"), &dec.q, 1, d, d);
            self.mm(&mut s, &dec.xn1, &p("self_attn.k_proj.weight"), &dec.k, 1, d, d);
            self.mm(&mut s, &dec.xn1, &p("self_attn.v_proj.weight"), &dec.v, 1, d, d);
            s.push(g.step(ROPE_AT, &[&dec.q], &[1, nh, hd, d, 0, pos, f(theta)], nh * (hd / 2)));
            s.push(g.step(ROPE_AT, &[&dec.k], &[1, nh, hd, d, 0, pos, f(theta)], nh * (hd / 2)));
            s.extend(block::gqa_decode_step(g, &decode_ids, nh, nh, hd, pos, cap, &dec.q, &dec.k, &dec.v, &dec.kcache[l], &dec.vcache[l], &dec.scores, &dec.probs, &dec.ctx));
            self.mm(&mut s, &dec.ctx, &p("self_attn.o_proj.weight"), &dec.proj, 1, d, d);
            s.push(g.step(ADD2, &[&dec.res[l], &dec.proj, &dec.xmid], &[d], d));

            // ---- MLP: dense (leading blocks) or MoE, over the ONE new row ----
            self.norm_fwd(&mut s, &dec.xmid, &p("ln2.weight"), &dec.xn2, d, 1);
            if c.is_moe_layer(l as u32) {
                self.mm(&mut s, &dec.xn2, &p("mlp.router.weight"), &dec.router_logits, 1, d, e);
                s.push(moe::router_fwd_kind(g, &moe_kernel_ids, self.router_kind(), &shape1, &dec.router_logits, None, &dec.gate, None));
                // Tried and MEASURED, not shipped: skipping the ~58/64
                // non-selected experts' dispatches by reading `gate` back to
                // the host per layer (mirroring `crates/glm`'s
                // `forward_compact`/`model::moe::expert_fwd_compact` trade).
                // `moe_linear_gated` call count dropped 67584 -> 8250
                // (~8.2x), but its OWN total time was unchanged (23.3s ->
                // 23.5s) and the whole decode's profiled total went UP
                // (34.9s -> 39.3s): the per-row gate check inside
                // `moe_linear_gated.wgsl` already makes a non-selected
                // expert's dispatch cheap, so the real cost was the
                // SELECTED experts' compute all along, and the 352 extra
                // host round-trips (11 MoE layers x 32 tokens) this needs
                // cost about as much as the skipped dispatches saved, so it
                // was reverted rather than shipped - a per-kernel dispatch
                // count is not the same thing as a whole-pass win, and only
                // the whole-pass number decides whether a fix worked.
                for ei in 0..e as usize {
                    let ep = |nm: &str| format!("blocks.{l}.mlp.experts.{ei}.{nm}");
                    s.extend(moe::expert_fwd(
                        g,
                        &moe_kernel_ids,
                        &shape1,
                        &dec.xn2,
                        &dec.gate,
                        self.w(&ep("gate.weight")),
                        self.w(&ep("up.weight")),
                        self.w(&ep("down.weight")),
                        &dec.moe_acts.at(ei),
                        &dec.moe_acc,
                        ei as u32,
                        ei != 0,
                    ));
                }
                s.extend(moe::shared_expert_fwd(
                    g,
                    &shared_ids,
                    1,
                    d,
                    shared_ff,
                    &dec.xn2,
                    self.w(&p("mlp.shared.gate.weight")),
                    self.w(&p("mlp.shared.up.weight")),
                    self.w(&p("mlp.shared.down.weight")),
                    None,
                    &SharedExpertScratch {
                        gate_pre: &dec.sh_gate,
                        up: &dec.sh_up,
                        h: &dec.sh_h,
                        mlp_out: &dec.sh_out,
                        gate_logits: &self.gate_stub,
                        gate_scalar: &self.gate_stub,
                        scaled: &self.gate_stub,
                    },
                    &dec.moe_acc,
                    &dec.mlp_out,
                ));
            } else {
                self.mm(&mut s, &dec.xn2, &p("mlp.gate.weight"), &dec.dense_gate_pre, 1, d, dense_ff);
                self.mm(&mut s, &dec.xn2, &p("mlp.up.weight"), &dec.dense_up, 1, d, dense_ff);
                s.push(block::swiglu_fwd(g, &ids, &dec.dense_gate_pre, &dec.dense_up, &dec.dense_h, dense_ff));
                self.mm(&mut s, &dec.dense_h, &p("mlp.down.weight"), &dec.mlp_out, 1, dense_ff, d);
            }
            s.push(g.step(ADD2, &[&dec.xmid, &dec.mlp_out, &dec.res[l + 1]], &[d], d));
        }

        let last = c.n_layers() as usize;
        self.norm_fwd(&mut s, &dec.res[last], "norm.weight", &dec.xn_final, d, 1);
        self.mm(&mut s, &dec.xn_final, c.head_weight(), &dec.logits, 1, d, vocab as u32);
        g.submit(&[], &s);
        g.read(&dec.logits, vocab)
    }

    /// **Greedy autoregressive decode, KV-cached** - the `O(T)` twin of
    /// [`Self::generate_greedy`]'s `O(T²)` recompute, producing the SAME
    /// tokens (the cache is algebraically exact, see `tests::
    /// generate_greedy_kv_matches_recompute` and `tests/generate.rs`'s
    /// real-weight gate). The prompt is run through ONE batched forward
    /// ([`Self::logits_all`], which also handles the vision-language splice if
    /// enabled) to seed the KV cache via [`Self::fill_cache_from_prefill`];
    /// every token after that is one [`Self::step`] call instead of a full
    /// re-run over the whole sequence so far.
    ///
    /// Resets the incremental cache on entry, so instances are reusable across
    /// calls; the instance must still be sized for the FINAL length
    /// (`prompt_ids.len() + n_new <= ctx_len()`, `b == 1`).
    pub fn generate_greedy_kv(&self, prompt_ids: &[u32], n_new: u32) -> Vec<u32> {
        self.generate_greedy_kv_cb(prompt_ids, n_new, |_| {})
    }

    /// [`Self::generate_greedy_kv`] with a per-token callback - same seam as
    /// [`Self::generate_greedy_cb`].
    pub fn generate_greedy_kv_cb(&self, prompt_ids: &[u32], n_new: u32, mut on_token: impl FnMut(u32)) -> Vec<u32> {
        assert!(!prompt_ids.is_empty(), "deepseekv2: greedy decode needs at least one prompt token");
        assert_eq!(self.b, 1, "deepseekv2: KV-cache decode requires b == 1");
        let total = prompt_ids.len() + n_new as usize;
        assert!(
            total <= self.t as usize,
            "deepseekv2: greedy decode of {} prompt + {n_new} new tokens needs a context of {total}, but this instance is sized for {}",
            prompt_ids.len(),
            self.t
        );
        let vocab = self.cfg.vocab() as usize;
        let mut ids: Vec<u32> = prompt_ids.to_vec();
        if n_new == 0 {
            return ids;
        }

        self.reset_cache();
        // Prefill: one batched forward over the whole prompt (handles the
        // splice, RoPE positions and causal mask exactly like the recompute
        // path), whose resident per-layer k/v seed the persistent cache.
        let logits = self.logits_all(&ids);
        self.fill_cache_from_prefill(prompt_ids.len() as u32);
        self.dec_pos.set(prompt_ids.len() as u32);

        let mut next = argmax(&logits[logits.len() - vocab..]) as u32;
        ids.push(next);
        on_token(next);
        for _ in 1..n_new {
            let logits = self.step(next);
            next = argmax(&logits) as u32;
            ids.push(next);
            on_token(next);
        }
        ids
    }

    // ---- vision-language embedding splice seam ----

    /// Enable the VLM embedding splice at residual rows `[row0, row0 + n_rows)`.
    ///
    /// After the text token-embedding gather the forward overwrites those rows
    /// with the image tokens written via [`Self::write_img_embeds`], and the
    /// backward routes their gradient to [`Self::read_d_img_embeds`], zeroing
    /// them in `dres[0]` so `emb_bwd` never trains the placeholder token's row.
    /// Reallocates the two image buffers and rebuilds both tapes, so call it
    /// once after construction and before the first forward. Nothing about
    /// `tok.weight` or any other parameter changes.
    ///
    /// One contiguous run only, which is the single-view scope this decoder is
    /// gated at; a multi-view layout emits one call per run and is a `rows`-level
    /// change (`deepseek2ocr::rows`), not a change here.
    pub fn enable_mm_splice(&mut self, row0: u32, n_rows: u32) {
        let n = self.b * self.t;
        assert!(row0 + n_rows <= n, "splice rows [{row0}, {}) exceed the {n}-row residual stream", row0 + n_rows);
        let sz = (n_rows * self.cfg.d_model()) as u64;
        self.img_embeds = self.gpu.storage(sz);
        self.d_img_embeds = self.gpu.storage(sz);
        self.mm_splice.set(Some((row0, n_rows)));
        self.fwd_steps = self.build_forward(self.b, self.t);
        if !self.bwd_steps.is_empty() {
            self.bwd_steps = self.build_backward();
        }
    }

    /// Number of spliced image-embedding elements (`n_rows * d_model`); 0 if off.
    fn img_numel(&self) -> usize {
        self.mm_splice.get().map_or(0, |(_, n)| (n * self.cfg.d_model()) as usize)
    }

    /// Write the projected image tokens `[n_rows, d_model]` (row-major).
    pub fn write_img_embeds(&self, data: &[f32]) {
        assert_eq!(data.len(), self.img_numel(), "img_embeds size mismatch (enable_mm_splice first?)");
        self.gpu.write_f32(&self.img_embeds, data);
    }

    /// The spliced image embeddings' gradient after [`Self::backward`] -- what a
    /// vision connector/encoder backward consumes.
    pub fn read_d_img_embeds(&self) -> Vec<f32> {
        self.gpu.read(&self.d_img_embeds, self.img_numel())
    }

    /// The splice INPUT buffer, for a vision tower sharing THIS decoder's [`Gpu`]
    /// -- write into it with a `Step` and the embedding never leaves the device.
    /// Valid only after [`Self::enable_mm_splice`]; before that it is the
    /// 1-float placeholder the constructor allocates.
    pub fn img_embeds_buf(&self) -> &DeviceBuffer {
        &self.img_embeds
    }

    /// The splice GRADIENT buffer -- the device-side counterpart of
    /// [`Self::read_d_img_embeds`]. Same validity rule as
    /// [`Self::img_embeds_buf`].
    pub fn d_img_embeds_buf(&self) -> &DeviceBuffer {
        &self.d_img_embeds
    }

    // ---- per-stage taps (what a composite's parity test compares) ----

    /// The residual stream entering layer `l`; `l == n_layers` is the stream
    /// leaving the last layer. `res[0]` is the token embedding AFTER the splice.
    pub fn read_res(&self, l: usize) -> Vec<f32> {
        self.gpu.read(&self.res[l], (self.b * self.t * self.cfg.d_model()) as usize)
    }

    /// Layer `l`'s attention output BEFORE the residual add - the output
    /// projection's own result, `[b*t, d_model]`.
    ///
    /// **Reconstructed as `xmid[l] - res[l]`, not read from a buffer.** The
    /// forward writes every layer's `o_proj` result into ONE shared `proj`
    /// temporary (only the last layer's survives a whole forward), and giving
    /// each layer its own would add an eighth `[b*t, d_model]` buffer per layer
    /// - half a gigabyte at this architecture's 8192 context - to serve nothing
    /// but a parity tap. Since `xmid` is literally `add2(res[l], proj)`, the
    /// subtraction recovers `proj` up to the single rounding of that sum: exact
    /// wherever the two summands are of comparable magnitude, and carrying at
    /// most a `~2^-24 * |xmid|` absolute error where the residual stream has
    /// grown much larger than the attention output. That is a real (if tiny)
    /// term in any max_abs computed from this, and a parity test reporting one
    /// should say so rather than attribute it to the model.
    pub fn read_attn_out(&self, l: usize) -> Vec<f32> {
        let n = (self.b * self.t * self.cfg.d_model()) as usize;
        let xmid = self.gpu.read(&self.layers[l].xmid, n);
        let res = self.gpu.read(&self.res[l], n);
        xmid.iter().zip(res).map(|(x, r)| x - r).collect()
    }

    /// The final RMSNorm output, `[b*t, d_model]`.
    pub fn read_final_norm(&self) -> Vec<f32> {
        self.gpu.read(&self.xn_final, (self.b * self.t * self.cfg.d_model()) as usize)
    }

    /// `[b*t, vocab]` logits.
    pub fn read_logits(&self) -> Vec<f32> {
        self.gpu.read(&self.logits, (self.b * self.t * self.cfg.vocab()) as usize)
    }

    /// Layer `l`'s router logits `[b*t, n_experts]`; `None` on a dense layer.
    pub fn read_router_logits(&self, l: usize) -> Option<Vec<f32>> {
        match &self.layers[l].mlp {
            Mlp::Moe { router_logits, .. } => Some(self.gpu.read(router_logits, (self.b * self.t * self.cfg.n_experts()) as usize)),
            Mlp::Dense { .. } => None,
        }
    }

    /// Layer `l`'s DENSE combine weights `[b*t, n_experts]` -- zero outside the
    /// selected top-k, and under this config's `norm_topk_prob`/`routed_scaling`
    /// policy. `None` on a dense layer.
    pub fn read_router_gate(&self, l: usize) -> Option<Vec<f32>> {
        match &self.layers[l].mlp {
            Mlp::Moe { gate, .. } => Some(self.gpu.read(gate, (self.b * self.t * self.cfg.n_experts()) as usize)),
            Mlp::Dense { .. } => None,
        }
    }

    pub fn save(&self, path: &str) {
        self.save_with_itos(path, None);
    }

    pub fn save_with_itos(&self, path: &str, itos: Option<&[char]>) {
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> =
            self.ps.params.iter().map(|(name, _)| (name.clone(), vec![self.ps.numel(name) as u64], self.read_weight(name))).collect();
        let mut config = self.cfg.to_json();
        if let Some(itos) = itos {
            let arr: Vec<serde_json::Value> = itos.iter().map(|ch| serde_json::Value::from(ch.to_string())).collect();
            config["itos"] = serde_json::Value::Array(arr);
        }
        checkpoint::save_carded(path, config, &tensors, &checkpoint::st::ModelCard::new("brain/deepseekv2", "deepseekv2"));
    }
}

// ---- architecture-agnostic Model seam ----

impl model::Model for DeepseekV2 {
    type Config = DeepseekV2Config;

    fn new(cfg: DeepseekV2Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Self {
        DeepseekV2::new(cfg, b, t, init)
    }
    fn init_weights(cfg: &DeepseekV2Config, seed: u64) -> HashMap<String, Vec<f32>> {
        crate::init::init_weights(cfg, seed)
    }
    fn config(&self) -> &DeepseekV2Config {
        &self.cfg
    }
    fn set_batch(&self, batch: model::Batch) {
        match batch {
            model::Batch::Lm { tokens, targets } => DeepseekV2::set_batch(self, tokens, targets),
            _ => panic!("deepseek2::DeepseekV2 only supports Batch::Lm"),
        }
    }
    fn forward(&self) -> f32 {
        DeepseekV2::forward(self)
    }
    fn backward(&self) {
        DeepseekV2::backward(self)
    }
    fn zero_grads(&self) {
        DeepseekV2::zero_grads(self)
    }
    fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        DeepseekV2::adamw_step(self, t, lr, wd, clip, extra_scale)
    }
    fn poll_wait(&self) {
        DeepseekV2::poll_wait(self)
    }
    fn param_names(&self) -> Vec<String> {
        DeepseekV2::param_names(self)
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        DeepseekV2::read_weight(self, name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        DeepseekV2::write_weight(self, name, data)
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        DeepseekV2::read_grad(self, name)
    }
    fn logits_all(&self, tokens: &[u32]) -> Option<Vec<f32>> {
        Some(DeepseekV2::logits_all(self, tokens))
    }
    fn save(&self, path: &str) {
        DeepseekV2::save(self, path)
    }
    fn save_with_itos(&self, path: &str, itos: Option<&[char]>) {
        DeepseekV2::save_with_itos(self, path, itos)
    }
    fn config_json(&self) -> serde_json::Value {
        self.cfg.to_json()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decode loop's own bookkeeping, at toy dims and in the FAST lane.
    ///
    /// The real gate for [`DeepseekV2::generate_greedy`] is
    /// `tests/generate.rs`, which needs ~12 GB of real weights and a llama.cpp
    /// capture. This covers what can be covered without either, and none of the
    /// three claims is a tautology over [`DeepseekV2::logits_all`]:
    ///
    /// 1. the prompt comes back verbatim, ahead of the generated ids;
    /// 2. resuming from a generated prefix reproduces the straight-through run
    ///    exactly - so the loop carries no state across steps, which matters
    ///    because every step REBUILDS the forward tape at the grown length;
    /// 3. each generated id is the argmax of the **last** row, not the first -
    ///    asserted against a row that actually disagrees, since at `t = 1` the
    ///    two are the same row and the bug would be invisible.
    #[test]
    fn greedy_decode_is_prefix_stable_and_reads_the_last_row() {
        let cfg = DeepseekV2Config::tiny();
        let init = crate::init::init_weights(&cfg, 7);
        let m = DeepseekV2::new_on(gpu_core::testgpu::dev(PIPELINES), cfg, 1, 8, &init, false);
        let v = m.cfg.vocab() as usize;
        let prompt = [1u32, 5, 2];

        let all = m.generate_greedy(&prompt, 5);
        assert_eq!(all.len(), 8, "generate_greedy returns prompt + n_new");
        assert_eq!(&all[..prompt.len()], &prompt, "the prompt must come back verbatim");

        let half = m.generate_greedy(&prompt, 2);
        assert_eq!(half, all[..prompt.len() + 2], "a shorter run must be a prefix of a longer one");
        assert_eq!(m.generate_greedy(&half, 3), all, "resuming from a generated prefix diverges");

        // Each step's id IS the last row's argmax over the prefix before it.
        for i in prompt.len()..all.len() {
            let logits = m.logits_all(&all[..i]);
            assert_eq!(all[i] as usize, argmax(&logits[logits.len() - v..]), "step {i} did not take the final position's argmax");
        }
        // ... and the first row of that same forward disagrees, so claim 3 is
        // testing something. (Deterministic for this seed and config.)
        let logits = m.logits_all(&prompt);
        assert_ne!(
            argmax(&logits[..v]),
            argmax(&logits[logits.len() - v..]),
            "fixture degenerated: rows 0 and t-1 now share an argmax, so reading the wrong one would pass"
        );
    }

    /// [`DeepseekV2::generate_greedy_cb`] actually **streams**, at the same toy
    /// dims and in the same fast lane as the test above.
    ///
    /// A callback that merely compiles is worth nothing to a serving path. The
    /// three ways to get it wrong and still typecheck are firing zero times,
    /// firing once at the end with the whole generation, and **draining the
    /// finished vector after the loop** - the last of which delivers the right
    /// ids in the right order, satisfies any count assertion, and is still
    /// useless, because nothing arrives one instant earlier than the return
    /// value does. A `Progress::token` stream built on it would emit `n_new`
    /// deltas back to back after the user had already waited for the whole
    /// decode.
    ///
    /// Nothing about the ids can tell those apart - `logits_all` is a pure
    /// function of its prefix, so a callback that reconstructs "the ids seen so
    /// far" and re-derives its own token agrees with itself under either
    /// ordering (that check was written first and passed the mutation). What
    /// does tell them apart is the decoder's WORK: the callback reads the
    /// device handle's online dispatch counter (`Gpu::ops_counters`, armed by
    /// the `reset_ops_counters` below; the counters are per-handle and
    /// `testgpu::dev` hands every caller its own, so a concurrently-running
    /// test cannot contribute to them). Under a real per-token callback that
    /// counter has strictly grown between consecutive calls, by one recomputed
    /// forward's worth of dispatches; under a drain it is the finished total at
    /// every call, identical each time. Mutation-verified in both directions.
    ///
    /// The ids compare against `generate_greedy`'s tail, not its whole return -
    /// `generate_greedy` returns the prompt ahead of the generated ids, and the
    /// callback fires only for what was predicted.
    #[test]
    fn generate_greedy_cb_emits_each_token_as_it_is_produced() {
        let cfg = DeepseekV2Config::tiny();
        let init = crate::init::init_weights(&cfg, 7);
        let m = DeepseekV2::new_on(gpu_core::testgpu::dev(PIPELINES), cfg, 1, 8, &init, false);
        let prompt = [1u32, 5, 2];
        let n_new = 5u32;

        m.gpu.reset_ops_counters();
        let mut streamed: Vec<u32> = Vec::new();
        let mut dispatched: Vec<u64> = Vec::new();
        let all = m.generate_greedy_cb(&prompt, n_new, |id| {
            dispatched.push(m.gpu.ops_counters().steps);
            streamed.push(id);
        });

        assert_eq!(streamed.len(), n_new as usize, "the callback fired {} times for {n_new} new tokens", streamed.len());
        assert_eq!(streamed, all[prompt.len()..], "the streamed ids differ from the returned generation");
        assert!(dispatched[0] > 0, "the first token arrived before any forward had been submitted");
        assert!(
            dispatched.windows(2).all(|w| w[1] > w[0]),
            "tokens did not arrive as they were produced: the decoder had dispatched {dispatched:?} steps at the {n_new} calls, so at least one pair of them saw the SAME amount of work done"
        );
        // ... and the no-callback wrapper is the very same loop.
        assert_eq!(m.generate_greedy(&prompt, n_new), all, "generate_greedy diverges from generate_greedy_cb");
    }

    /// **The KV-cache decode gate** at toy dims, fast lane: [`DeepseekV2::
    /// generate_greedy_kv`] must produce the SAME greedy tokens as the `O(T²)`
    /// recompute path ([`DeepseekV2::generate_greedy`]) for every position,
    /// prompt and generated alike -- the cache is algebraically exact, so any
    /// divergence is a position/mask/append bug in the decode step, not a
    /// numerics difference (`crates/gpt`'s `generate_kv_matches_recompute_
    /// greedy` is the same shape of gate on a plain-MLP decoder; this is its
    /// MoE-decoder twin). The real-weight gate against llama.cpp is
    /// `tests/generate.rs`.
    ///
    /// Exercises BOTH MLP kinds (`tiny()`'s layer 0 is dense, layer 1 is MoE),
    /// a splice-free prompt, and re-running from a fresh `reset_cache` so the
    /// instance is provably reusable across calls.
    #[test]
    fn generate_greedy_kv_matches_recompute() {
        let cfg = DeepseekV2Config::tiny();
        let init = crate::init::init_weights(&cfg, 7);
        let m = DeepseekV2::new_on(gpu_core::testgpu::dev(PIPELINES), cfg, 1, 8, &init, false);
        let prompt = [1u32, 5, 2];
        let n_new = 5u32;

        let recompute = m.generate_greedy(&prompt, n_new);
        let kv = m.generate_greedy_kv(&prompt, n_new);
        assert_eq!(kv, recompute, "KV-cache decode diverged from O(T^2) recompute");
        assert_eq!(&kv[..prompt.len()], &prompt, "the prompt must come back verbatim");

        // Re-running (a fresh cache each call) must be deterministic, not an
        // accumulation-of-state artifact of the first run.
        let kv2 = m.generate_greedy_kv(&prompt, n_new);
        assert_eq!(kv2, kv, "generate_greedy_kv is not idempotent across calls on the same instance");

        // A shorter run's ids must be the longer run's own prefix (same
        // property `greedy_decode_is_prefix_stable_and_reads_the_last_row`
        // checks on the recompute path).
        let short = m.generate_greedy_kv(&prompt, 2);
        assert_eq!(short, kv[..prompt.len() + 2], "a shorter KV run must be a prefix of a longer one");
    }

    /// [`DeepseekV2::generate_greedy_kv_cb`] streams, same contract as
    /// [`generate_greedy_cb_emits_each_token_as_it_is_produced`] above but over
    /// the KV-cached loop -- a callback that only drains a finished vector
    /// would pass every id/count assertion, so this also checks the online
    /// dispatch counter strictly grows between calls.
    #[test]
    fn generate_greedy_kv_cb_emits_each_token_as_it_is_produced() {
        let cfg = DeepseekV2Config::tiny();
        let init = crate::init::init_weights(&cfg, 7);
        let m = DeepseekV2::new_on(gpu_core::testgpu::dev(PIPELINES), cfg, 1, 8, &init, false);
        let prompt = [1u32, 5, 2];
        let n_new = 5u32;

        m.gpu.reset_ops_counters();
        let mut streamed: Vec<u32> = Vec::new();
        let mut dispatched: Vec<u64> = Vec::new();
        let all = m.generate_greedy_kv_cb(&prompt, n_new, |id| {
            dispatched.push(m.gpu.ops_counters().steps);
            streamed.push(id);
        });

        assert_eq!(streamed.len(), n_new as usize, "the callback fired {} times for {n_new} new tokens", streamed.len());
        assert_eq!(streamed, all[prompt.len()..], "the streamed ids differ from the returned generation");
        assert!(dispatched[0] > 0, "the first token arrived before any forward had been submitted");
        assert!(
            dispatched.windows(2).all(|w| w[1] > w[0]),
            "tokens did not arrive as they were produced: the decoder had dispatched {dispatched:?} steps at the {n_new} calls, so at least one pair of them saw the SAME amount of work done"
        );
        assert_eq!(m.generate_greedy_kv(&prompt, n_new), all, "generate_greedy_kv diverges from generate_greedy_kv_cb");
        assert_eq!(m.generate_greedy(&prompt, n_new), all, "recompute diverges from the KV-cached callback loop");
    }
}
