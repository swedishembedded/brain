// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Int8 (DP4A) inference path for Qwen3.5's per-layer 256-expert sparse MoE
//! (`model::moe::expert_fwd_i8`, reused unchanged - see this module's own
//! `as_moe` adapter below for the one seam needed to call it). The GDN/GQA
//! mixer linears are quantized separately, through the shared
//! `model::ops::{Ops, Act, Weight}` façade driven by `model.rs` itself - this
//! module no longer owns or dispatches them (`Qwen35Q8::is_i8_linear` still
//! names both groups, since `model.rs`'s fp32-`ParamStore` role filter needs
//! one combined predicate regardless of which struct quantizes which leaf).
//!
//! ## Which linears are quantized, and why
//!
//! **Quantized** (the attention/mixer + MoE-expert GEMMs - where nearly all
//! the FLOPs and, for the experts, nearly all the PARAMETERS live):
//! - GDN layers: `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`
//!   (via `model::ops::Weight` in `model.rs`, not this module).
//! - GQA layers: `q_proj`, `k_proj`, `v_proj`, `o_proj` (ditto).
//! - Every routed expert's `gate`/`up`/`down` (256 experts/layer at the real
//!   35B-A3B scale - by far the dominant share of total parameters: 256×3
//!   expert tensors per layer vs. 5-9 mixer tensors and 1 router tensor;
//!   quantized by this module).
//!
//! **Left fp32** (small, precision-sensitive, or not a GEMM at all):
//! - `mlp.router.weight`: the softmax router logit projection is `[d_model,
//!   n_experts]` - tiny next to a single expert's weights, let alone 256 of
//!   them - and it picks a hard top-k routing DECISION per token; a
//!   quantization-noised logit can flip which experts get selected entirely
//!   (not just perturb a continuous output), a qualitatively worse failure
//!   mode than a noised activation. Not a throughput bottleneck either way.
//! - `mlp.shared_expert.{gate,up,down}` + `mlp.shared_expert_gate`: ONE
//!   shared expert per layer vs. 256 routed ones - even though its own
//!   `shared_expert_intermediate_size` roughly matches one routed expert's
//!   `moe_intermediate_size` (512 vs 512 at the real scale), it is 1/256th of
//!   the routed-expert parameter mass per layer. `model::moe::
//!   shared_expert_fwd` also has no int8 counterpart today (only
//!   `expert_fwd`/`expert_fwd_i8` have a quantized sibling) - adding one
//!   would be new kernel work outside this task's "integration, not new
//!   math" scope, for a path that is not the bottleneck.
//! - `tok.weight`/`lm_head.weight`: the embedding is a gather (`embed.wgsl`),
//!   not a GEMM - there is no DP4A kernel it could dispatch to. `lm_head`
//!   stays fp32 for the same precision-sensitive-logits reason as the
//!   router, mirroring `qwen3::q8::Q8::LINEARS`'s own choice to leave both
//!   out.
//! - Norms/RoPE/`A_log`/`dt_bias`/conv1d: not matmuls, untouched either way.
//!
//! ## The k%32 scale-group constraint, and why the real checkpoint always clears it
//!
//! `model::int8::quantize_weight` scales a weight per 32-element GROUP of the
//! contraction dimension `k` (`model::int8::GROUP`, GGUF `Q8_0`'s own block),
//! so every quantized linear's `k` must be a multiple of 32 (asserted in
//! `quantize_weight` itself - the same
//! constraint applies to the mixer linears' `model::ops::Weight::upload` in
//! `model.rs`). At the real 35B-A3B scale every `k` this module quantizes
//! against IS a multiple of 32 (`d_model=2048`, `moe_intermediate_size=512`,
//! ...) - real Transformer hidden widths are chosen to divide evenly for far
//! more demanding tiling reasons than this one. `Qwen35Config::tiny()`'s
//! deliberately tiny, deliberately ODD toy dimensions do NOT all clear this
//! bar (`moe_intermediate_size=10`, feeding every expert's `down`), so
//! `crates/qwen35moe/tests/model_i8_smoke.rs` exercises this module against a
//! bespoke small-but-int8-shaped config instead of `tiny()` verbatim - see
//! that test's own doc for why that is the right call rather than adding
//! silent per-tensor fp32-fallback logic here for a toy-scale-only edge case
//! that never occurs at any real checkpoint size.

use gpu_core::{DeviceBuffer, Gpu, Step};
use model::ops::Weight;

pub use model::int8::quantize_weight;

use crate::config::Qwen35Config;

/// One projection of every expert of a layer as ONE fused int8 bank: expert
/// `e`'s `[n, k]` matrix is rows `e*n .. (e+1)*n` of a `[blocks * n, k/4]`
/// packed-word buffer with a `[blocks * n, k/32]` group-scale sibling - exactly
/// llama.cpp's own stacked-expert layout, so a Q8_0 GGUF becomes this by a byte
/// repack. One buffer per projection per layer (not one per expert) is what
/// lets a single `moe_i8_gemv_gather` dispatch serve every expert a token
/// routes to.
///
/// `blocks` is `n_experts`, plus one when the shared expert rides in the bank
/// as block `n_experts` (see [`Q8MoeLayer::router_ext`]).
pub struct Bank8 {
    pub packed: DeviceBuffer,
    pub scale: DeviceBuffer,
    /// Output width of one expert's matrix.
    pub n: u32,
    /// Input width (contraction dimension).
    pub k: u32,
    pub blocks: u32,
}

/// One layer's routed experts (256 at real scale) as three fused banks, and -
/// when the shared expert fits the same shape - the shared expert as the bank's
/// last block.
pub struct Q8MoeLayer {
    pub gate: Bank8,
    pub up: Bank8,
    pub down: Bank8,
    /// `[n_experts + 1, d_model]` f32 router weight with the shared expert's
    /// gate row appended, so ONE matmul produces the routed logits and the
    /// shared gate's logit. `Some` exactly when the shared expert is in the
    /// banks; `None` leaves the shared expert on the fp32 path.
    pub router_ext: Option<Weight>,
}

impl Q8MoeLayer {
    pub fn shared_in_bank(&self) -> bool {
        self.router_ext.is_some()
    }
}

/// Resident int8 MoE-expert banks for every layer + shared
/// activation-quant scratch. Single-GPU only (no sharding - `moe` is a plain
/// `Vec` indexed by absolute layer index, not a `HashMap<usize, _>` of an
/// owned subset the way `qwen3::q8::Q8` supports for its sharded pipeline;
/// qwen35 multi-GPU sharding is separate, already-scoped follow-on work per
/// this task's own brief, not attempted here). The GDN/GQA mixer linears
/// live in `model.rs`'s own `weights: HashMap<String, model::ops::Weight>`
/// instead - see this module's doc.
pub struct Qwen35Q8 {
    pub moe: Vec<Q8MoeLayer>,
    /// `[n_tokens]` per-token activation scale, shared by every quantized
    /// linear this module dispatches (one distinct input is live in
    /// `xq`/`sx` at a time - see [`Qwen35Q8::quant`]'s doc).
    pub sx: DeviceBuffer,
    /// `[n_tokens * d_model/4]` packed activation - `d_model` is the only
    /// width this module's sole quant call site (`xn2` feeding every
    /// expert's gate/up) ever reads.
    pub xq: DeviceBuffer,
    k_max_abs_row: usize,
    k_quant_pack: usize,
}

impl Qwen35Q8 {
    /// Is `name` (e.g. `blocks.5.self_attn.q_proj.weight` or
    /// `blocks.3.mlp.experts.17.down.weight`) one of the linears this tier
    /// quantizes? Mirrors `qwen3::q8::Q8::is_i8_linear`'s "leaf-name lookup"
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

    /// Is `name` one of the shared expert's three projections - the leaves that
    /// live in the int8 banks (as block `n_experts`) instead of the fp32 store
    /// whenever [`Self::shared_fits_bank`].
    pub fn is_shared_expert_linear(name: &str) -> bool {
        name.strip_prefix("blocks.")
            .and_then(|r| r.split_once('.'))
            .is_some_and(|(_, leaf)| matches!(leaf, "mlp.shared_expert.gate.weight" | "mlp.shared_expert.up.weight" | "mlp.shared_expert.down.weight"))
    }

    /// The shared expert can ride in the routed experts' banks only if its
    /// matrices have the routed experts' shape.
    pub fn shared_fits_bank(cfg: &Qwen35Config) -> bool {
        cfg.shared_expert_intermediate_size == cfg.moe_intermediate_size
    }

    /// Quantize+upload every layer's expert banks from `source`, one
    /// projection at a time (peak host RAM ~= one bank's packed words - a
    /// quarter of the fp32 bank - never the model; same discipline as
    /// `paramstore`'s own streaming load). `n_tokens = b*t`, matching the
    /// model's own activation extent.
    ///
    /// A source that lends llama.cpp's stacked tensors under `gguf_load::
    /// bank_name` is read whole (a Q8_0 stack is a byte repack); any other
    /// source (the tiny test checkpoints) is read expert by expert and
    /// assembled, so both land in the same bank layout.
    pub fn build(
        gpu: &Gpu,
        source: &dyn checkpoint::TensorSource,
        cfg: &Qwen35Config,
        n_tokens: u32,
        k_max_abs_row: usize,
        k_quant_pack: usize,
    ) -> Qwen35Q8 {
        let mut up = paramstore::upload::Uploader::new(gpu);
        let (d, ff) = (cfg.d_model as usize, cfg.moe_intermediate_size as usize);
        let shared = Self::shared_fits_bank(cfg);

        let mut moe = Vec::with_capacity(cfg.n_layers as usize);
        for l in 0..cfg.n_layers as usize {
            let gate = build_bank(&mut up, source, cfg, l, "gate", ff, d, shared);
            let up_bank = build_bank(&mut up, source, cfg, l, "up", ff, d, shared);
            let down = build_bank(&mut up, source, cfg, l, "down", d, ff, shared);
            let router_ext = shared.then(|| {
                let mut w = Vec::with_capacity((cfg.n_experts as usize + 1) * d);
                for name in [format!("blocks.{l}.mlp.router.weight"), format!("blocks.{l}.mlp.shared_expert_gate.weight")] {
                    assert!(source.with_tensor(&name, &mut |t| w.extend_from_slice(t)), "qwen35 q8: missing {name}");
                }
                let (n, k) = (cfg.n_experts + 1, cfg.d_model);
                Weight::F32 { w: gpu.storage_init(&format!("blocks.{l}.mlp.router_ext"), &w), n, k }
            });
            moe.push(Q8MoeLayer { gate, up: up_bank, down, router_ext });
        }

        // `d_model` is the only width this module's sole quant call site
        // (`xn2`, feeding every expert's gate/up) ever reads -- the expert's
        // own `h` is quantized separately by `moe_swiglu_quant`.
        let sx = gpu.storage(n_tokens.max(1) as u64);
        let xq = gpu.storage(((n_tokens as u64) * (d as u64) / 4).max(1));
        Qwen35Q8 { moe, sx, xq, k_max_abs_row, k_quant_pack }
    }

    /// Quantize activation `x` `[n_tokens · k]` into `self.xq` with fresh
    /// per-token scales `self.sx`. Call once per distinct input (shared by
    /// every linear that reads that SAME input, e.g. xn1 -> q/k/v-proj); a
    /// later `quant` call for a DIFFERENT input safely overwrites `xq`/`sx`
    /// once every earlier consumer's step has already been pushed ahead of it
    /// in the same (or an earlier, already-submitted) step list - identical in
    /// spirit to `qwen3::q8::Q8::quant`'s own doc.
    pub fn quant(&self, gpu: &Gpu, s: &mut Vec<Step>, x: &DeviceBuffer, k: u32, n_tokens: u32) {
        s.push(gpu.step(self.k_max_abs_row, &[x, &self.sx], &[n_tokens, k], n_tokens));
        s.push(gpu.step(self.k_quant_pack, &[x, &self.sx, &self.xq], &[n_tokens, k], n_tokens * k / 4));
    }
}

/// One fused bank of `proj` (`gate`/`up`: `[ff, d]` per expert; `down`: `[d,
/// ff]`), the shared expert appended as the last block when `with_shared`.
#[allow(clippy::too_many_arguments)]
fn build_bank(
    up: &mut paramstore::upload::Uploader,
    source: &dyn checkpoint::TensorSource,
    cfg: &Qwen35Config,
    layer: usize,
    proj: &str,
    n: usize,
    k: usize,
    with_shared: bool,
) -> Bank8 {
    let e = cfg.n_experts as usize;
    // (source tensor, rows it contributes)
    let mut parts: Vec<(String, usize)> = Vec::new();
    let stacked = crate::gguf_load::bank_name(layer, proj);
    if source.numel(&stacked) == Some(e * n * k) {
        parts.push((stacked, e * n));
    } else {
        parts.extend((0..e).map(|ei| (format!("blocks.{layer}.mlp.experts.{ei}.{proj}.weight"), n)));
    }
    if with_shared {
        parts.push((format!("blocks.{layer}.mlp.shared_expert.{proj}.weight"), n));
    }
    let rows: usize = parts.iter().map(|p| p.1).sum();
    let gpu = up.gpu();
    let (packed, scale) = (gpu.storage((rows * k / 4) as u64), gpu.storage((rows * k / model::int8::GROUP) as u64));
    let (mut word_off, mut scale_off) = (0u64, 0u64);
    for (name, part_rows) in &parts {
        let (words, scales) =
            model::int8::quantize_from(source, name, *part_rows, k).unwrap_or_else(|| panic!("qwen35 q8: '{name}' is not present in this source"));
        gpu.write_at(&packed, word_off, &words);
        gpu.write_f32_at(&scale, scale_off, &scales);
        word_off += words.len() as u64;
        scale_off += scales.len() as u64;
        up.account(4 * (words.len() + scales.len()) as u64);
        up.maybe_drain(&packed);
    }
    Bank8 { packed, scale, n: n as u32, k: k as u32, blocks: (e + usize::from(with_shared)) as u32 }
}
