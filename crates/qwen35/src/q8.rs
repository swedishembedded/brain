// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Int8 (DP4A) inference path for Qwen3.5's hybrid GDN/GQA/sparse-MoE
//! decoder — mirrors `crates/qwen/src/q8.rs`'s recipe (weights quantized once
//! at build via `model::int8::quantize_weight`; activations quantized
//! on-device each forward with a dynamic per-token scale, then the DP4A GEMM
//! dequantizes with `sx·sw`), adapted for this model's two distinct per-layer
//! shapes (Gated DeltaNet vs GQA) and its per-layer 256-expert sparse MoE
//! (`model::moe::expert_fwd_i8`, reused unchanged — see this module's own
//! `as_moe` adapter below for the one seam needed to call it).
//!
//! ## Which linears are quantized, and why
//!
//! **Quantized** (the attention/mixer + MoE-expert GEMMs — where nearly all
//! the FLOPs and, for the experts, nearly all the PARAMETERS live):
//! - GDN layers: `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`.
//! - GQA layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
//! - Every routed expert's `gate`/`up`/`down` (256 experts/layer at the real
//!   35B-A3B scale — by far the dominant share of total parameters: 256×3
//!   expert tensors per layer vs. 5-9 mixer tensors and 1 router tensor).
//!
//! **Left fp32** (small, precision-sensitive, or not a GEMM at all):
//! - `mlp.router.weight`: the softmax router logit projection is `[d_model,
//!   n_experts]` — tiny next to a single expert's weights, let alone 256 of
//!   them — and it picks a hard top-k routing DECISION per token; a
//!   quantization-noised logit can flip which experts get selected entirely
//!   (not just perturb a continuous output), a qualitatively worse failure
//!   mode than a noised activation. Not a throughput bottleneck either way.
//! - `mlp.shared_expert.{gate,up,down}` + `mlp.shared_expert_gate`: ONE
//!   shared expert per layer vs. 256 routed ones — even though its own
//!   `shared_expert_intermediate_size` roughly matches one routed expert's
//!   `moe_intermediate_size` (512 vs 512 at the real scale), it is 1/256th of
//!   the routed-expert parameter mass per layer. `model::moe::
//!   shared_expert_fwd` also has no int8 counterpart today (only
//!   `expert_fwd`/`expert_fwd_i8` have a quantized sibling) — adding one
//!   would be new kernel work outside this task's "integration, not new
//!   math" scope, for a path that is not the bottleneck.
//! - `tok.weight`/`lm_head.weight`: the embedding is a gather (`embed.wgsl`),
//!   not a GEMM — there is no DP4A kernel it could dispatch to. `lm_head`
//!   stays fp32 for the same precision-sensitive-logits reason as the
//!   router, mirroring `qwen::q8::Q8::LINEARS`'s own choice to leave both
//!   out.
//! - Norms/RoPE/`A_log`/`dt_bias`/conv1d: not matmuls, untouched either way.
//!
//! ## The k%4 packing constraint, and why the real checkpoint always clears it
//!
//! `model::int8::quantize_weight`/the DP4A kernels pack 4 int8 lanes per
//! `u32` along the contraction dimension `k`, so every quantized linear's `k`
//! must be a multiple of 4 (asserted in `quantize_weight` itself). At the
//! real 35B-A3B scale every `k` this module quantizes against IS a multiple
//! of 4 (`d_model=2048`, `moe_intermediate_size=512`,
//! `linear_value_dim=32*128=4096`, `q_dim=16*256=4096`, ...) — real
//! Transformer hidden widths are chosen to divide evenly for far more
//! demanding tiling reasons than this one. `Qwen35Config::tiny()`'s
//! deliberately tiny, deliberately ODD toy dimensions do NOT all clear this
//! bar (`linear_value_dim=6*5=30`, `moe_intermediate_size=10` — both feed a
//! quantized linear's `k`: GDN's `out_proj` and every expert's `down`
//! respectively), so `crates/qwen35/tests/model_i8_smoke.rs` exercises this
//! module against a bespoke small-but-int8-shaped config instead of `tiny()`
//! verbatim — see that test's own doc for why that is the right call rather
//! than adding silent per-tensor fp32-fallback logic here for a
//! toy-scale-only edge case that never occurs at any real checkpoint size.

use gpu_core::{DeviceBuffer, Gpu, Step};

use model::moe::Lin8 as MoeLin8;

pub use model::int8::quantize_weight;

use crate::config::{LayerType, Qwen35Config};

/// One int8 linear: packed int8 weight (`[n, k/4]` u32) + per-channel scale
/// `[n]` — identical layout to `qwen::q8::Lin8`, duplicated rather than
/// reused because `qwen::q8` is that crate's own private tier (its `Lin8` is
/// not `pub` beyond `qwen`) and `model::moe::Lin8` is a borrowed VIEW
/// (`&DeviceBuffer` fields, sized for one call) rather than an owner of the
/// underlying buffers — this type is the OWNER; [`Lin8::as_moe`] borrows it
/// into a `model::moe::Lin8` view at each `expert_fwd_i8` call site.
pub struct Lin8 {
    pub packed: DeviceBuffer,
    pub scale: DeviceBuffer,
    pub k: u32, // input width (contraction dim)
    pub n: u32, // output width
}

impl Lin8 {
    /// Borrow as the view `model::moe::expert_fwd_i8` expects.
    pub fn as_moe(&self) -> MoeLin8<'_> {
        MoeLin8 { wq: &self.packed, sw: &self.scale }
    }
}

/// One GQA (full-attention) layer's quantized linears.
pub struct Q8GqaLayer {
    pub q_proj: Lin8,
    pub k_proj: Lin8,
    pub v_proj: Lin8,
    pub o_proj: Lin8,
}

/// One Gated DeltaNet (linear-attention) layer's quantized linears.
pub struct Q8GdnLayer {
    pub in_proj_qkv: Lin8,
    pub in_proj_z: Lin8,
    pub in_proj_a: Lin8,
    pub in_proj_b: Lin8,
    pub out_proj: Lin8,
}

/// A layer's quantized mixer, tagged by which kind it is — mirrors
/// [`LayerType`]'s own split so a caller can `match` exactly the way
/// `model.rs`'s own `layer_types()`-driven dispatch already does.
pub enum Q8Mixer {
    Gdn(Q8GdnLayer),
    Gqa(Q8GqaLayer),
}

/// One routed expert's quantized gate/up/down.
pub struct Lin8Expert {
    pub gate: Lin8,
    pub up: Lin8,
    pub down: Lin8,
}

/// One layer's routed experts (256 at real scale), quantized. The router and
/// shared expert are never in here — see this module's doc for why.
pub struct Q8MoeLayer {
    pub experts: Vec<Lin8Expert>,
}

/// Resident int8 linears for every layer + shared activation-quant scratch.
/// Single-GPU only (no sharding — `mixers`/`moe` are plain `Vec`s indexed by
/// absolute layer index, not a `HashMap<usize, _>` of an owned subset the way
/// `qwen::q8::Q8` supports for its sharded pipeline; qwen35 multi-GPU
/// sharding is separate, already-scoped follow-on work per this task's own
/// brief, not attempted here).
pub struct Qwen35Q8 {
    pub mixers: Vec<Q8Mixer>,
    pub moe: Vec<Q8MoeLayer>,
    /// `[n_tokens]` per-token activation scale, shared by every quantized
    /// linear in the model (one distinct input is live in `xq`/`sx` at a
    /// time — see [`Qwen35Q8::quant`]'s doc).
    pub sx: DeviceBuffer,
    /// `[n_tokens * max_k/4]` packed activation, reused per linear-input
    /// group (xn1 -> q/k/v-proj or in_proj_*; ctx -> o_proj; the GDN gated
    /// output -> out_proj; xn2 -> every expert's gate/up).
    pub xq: DeviceBuffer,
    k_max_abs_row: usize,
    k_quant_pack: usize,
    k_matmul_i8: usize,
}

impl Qwen35Q8 {
    /// Is `name` (e.g. `blocks.5.self_attn.q_proj.weight` or
    /// `blocks.3.mlp.experts.17.down.weight`) one of the linears this module
    /// quantizes? Mirrors `qwen::q8::Q8::is_i8_linear`'s "leaf-name lookup"
    /// shape, extended with the per-expert-index prefix match the 256-expert
    /// MoE needs (an expert's own leaf embeds its index, so a fixed name
    /// list can't enumerate it directly).
    pub fn is_i8_linear(name: &str) -> bool {
        let Some(leaf) = name.strip_prefix("blocks.").and_then(|r| r.split_once('.')).map(|(_, leaf)| leaf) else {
            return false;
        };
        const MIXER_LINEARS: [&str; 9] = [
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.out_proj.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
        ];
        if MIXER_LINEARS.contains(&leaf) {
            return true;
        }
        // "mlp.experts.{e}.{gate,up,down}.weight" -- the router
        // ("mlp.router.weight") and shared expert ("mlp.shared_expert.*",
        // "mlp.shared_expert_gate.weight") deliberately do NOT share this
        // "mlp.experts." prefix, so they fall through to `false` below.
        leaf.strip_prefix("mlp.experts.").is_some_and(|rest| {
            rest.split_once('.').is_some_and(|(_idx, tail)| matches!(tail, "gate.weight" | "up.weight" | "down.weight"))
        })
    }

    /// Quantize+upload every layer's designated linears from `source`,
    /// streaming one tensor at a time (peak host RAM ~= one tensor of f32 —
    /// same discipline as `qwen::q8::Q8::build`/`paramstore`'s own streaming
    /// load). `n_tokens = b*t`, matching the model's own activation extent.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        gpu: &Gpu,
        source: &dyn checkpoint::TensorSource,
        cfg: &Qwen35Config,
        n_tokens: u32,
        k_max_abs_row: usize,
        k_quant_pack: usize,
        k_matmul_i8: usize,
    ) -> Qwen35Q8 {
        let mk = |name: &str, n: usize, k: usize| -> Lin8 {
            let mut lin: Option<Lin8> = None;
            let found = source.with_tensor(name, &mut |raw| {
                let (packed, sw) = quantize_weight(raw, n, k);
                let pb = gpu.storage(packed.len() as u64);
                gpu.write(&pb, &packed);
                gpu.poll_wait(); // reclaim staging before the next weight (see paramstore)
                let sb = gpu.storage(sw.len() as u64);
                gpu.write(&sb, &sw.iter().map(|v| v.to_bits()).collect::<Vec<u32>>());
                gpu.poll_wait();
                lin = Some(Lin8 { packed: pb, scale: sb, k: k as u32, n: n as u32 });
            });
            if !found {
                panic!("qwen35 q8: missing init weight {name}");
            }
            lin.unwrap()
        };

        let d = cfg.d_model as usize;
        let qd = cfg.q_dim() as usize;
        let kvd = cfg.kv_dim() as usize;
        let qpd = cfg.q_proj_dim() as usize;
        let value_dim = cfg.linear_value_dim() as usize;
        let conv_dim = cfg.linear_conv_dim() as usize;
        let nvh = cfg.linear_num_value_heads as usize;
        let ff = cfg.moe_intermediate_size as usize;

        let types = cfg.layer_types();
        let mut mixers = Vec::with_capacity(cfg.n_layers as usize);
        let mut moe = Vec::with_capacity(cfg.n_layers as usize);
        for (l, ty) in types.iter().enumerate() {
            let mixer = match ty {
                LayerType::Linear => {
                    let p = |s: &str| format!("blocks.{l}.linear_attn.{s}");
                    Q8Mixer::Gdn(Q8GdnLayer {
                        in_proj_qkv: mk(&p("in_proj_qkv.weight"), conv_dim, d),
                        in_proj_z: mk(&p("in_proj_z.weight"), value_dim, d),
                        in_proj_a: mk(&p("in_proj_a.weight"), nvh, d),
                        in_proj_b: mk(&p("in_proj_b.weight"), nvh, d),
                        out_proj: mk(&p("out_proj.weight"), d, value_dim),
                    })
                }
                LayerType::Full => {
                    let p = |s: &str| format!("blocks.{l}.self_attn.{s}");
                    Q8Mixer::Gqa(Q8GqaLayer {
                        q_proj: mk(&p("q_proj.weight"), qpd, d),
                        k_proj: mk(&p("k_proj.weight"), kvd, d),
                        v_proj: mk(&p("v_proj.weight"), kvd, d),
                        o_proj: mk(&p("o_proj.weight"), d, qd),
                    })
                }
            };
            mixers.push(mixer);

            let mut experts = Vec::with_capacity(cfg.n_experts as usize);
            for e in 0..cfg.n_experts {
                let pe = |s: &str| format!("blocks.{l}.mlp.experts.{e}.{s}");
                experts.push(Lin8Expert {
                    gate: mk(&pe("gate.weight"), ff, d),
                    up: mk(&pe("up.weight"), ff, d),
                    down: mk(&pe("down.weight"), d, ff),
                });
            }
            moe.push(Q8MoeLayer { experts });
        }

        // The widest `k` any quantized linear's ACTIVATION side reads: xn1
        // (d_model, feeds q/k/v-proj or in_proj_*), ctx (q_dim, feeds
        // o_proj), the GDN gated output (linear_value_dim, feeds out_proj),
        // and xn2 (d_model, feeds every expert's gate/up -- the expert's own
        // `h` is quantized separately inside `expert_fwd_i8`'s own scratch,
        // not through this shared buffer).
        let max_k = d.max(qd).max(value_dim) as u32;
        let sx = gpu.storage(n_tokens.max(1) as u64);
        let xq = gpu.storage(((n_tokens as u64) * (max_k as u64) / 4).max(1));
        Qwen35Q8 { mixers, moe, sx, xq, k_max_abs_row, k_quant_pack, k_matmul_i8 }
    }

    /// Quantize activation `x` `[n_tokens · k]` into `self.xq` with fresh
    /// per-token scales `self.sx`. Call once per distinct input (shared by
    /// every linear that reads that SAME input, e.g. xn1 -> q/k/v-proj); a
    /// later `quant` call for a DIFFERENT input safely overwrites `xq`/`sx`
    /// once every earlier consumer's `mm8`/`expert_fwd_i8` step has already
    /// been pushed ahead of it in the same (or an earlier, already-submitted)
    /// step list — identical in spirit to `qwen::q8::Q8::quant`'s own doc.
    pub fn quant(&self, gpu: &Gpu, s: &mut Vec<Step>, x: &DeviceBuffer, k: u32, n_tokens: u32) {
        s.push(gpu.step(self.k_max_abs_row, &[x, &self.sx], &[n_tokens, k], n_tokens));
        s.push(gpu.step(self.k_quant_pack, &[x, &self.sx, &self.xq], &[n_tokens, k], n_tokens * k / 4));
    }

    /// `out = dequant(xq @ wᵀ)`: dynamic per-token scale `self.sx` × per-
    /// channel weight scale. Must be preceded by a matching [`Qwen35Q8::quant`]
    /// on the same input.
    pub fn mm8(&self, gpu: &Gpu, s: &mut Vec<Step>, w: &Lin8, out: &DeviceBuffer, n_tokens: u32) {
        s.push(gpu.step(
            self.k_matmul_i8,
            &[&self.xq, &w.packed, &self.sx, &w.scale, out],
            &[n_tokens, w.k / 4, w.n],
            n_tokens.div_ceil(128) * w.n.div_ceil(128) * 256,
        ));
    }
}
