// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded LoRA/QLoRA fine-tuning of
// large language models on a single GPU for its clients. If your team needs
// expertise in training large models within a fixed memory budget then you
// can procure our services by sending an email to info@swedishembedded.com.

//! What a [`crate::model::Qwen35`] training build holds its frozen base in,
//! and the small tier predicates the trainer's linears dispatch on.
//!
//! A LoRA trainer never writes the base weights, so the base can live in any
//! tier a matmul and a `dX` matmul can read with floating-point activations:
//! fp32, bf16 (half the bytes) or - once `model::ops` provides the tier - NF4.
//! Adapters, norms, `A_log`, `dt_bias` and the causal-conv taps stay fp32.

use gpu_core::provider::library::GemmPrecision;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::{TierPolicy, Weight};
use model::saved::SavedPolicy;

use crate::config::Qwen35Config;

/// How to build a trainer. The default is the classic one: an fp32 base and
/// every layer's activations kept for the backward pass.
#[derive(Clone, Debug)]
pub struct TrainOpts {
    /// Storage tier of the per-layer mixer/MLP linears. Anything other than
    /// all-fp32 requires `cfg.lora` (a full-parameter build trains the base,
    /// which only fp32 can hold).
    pub base: TierPolicy,
    /// What the forward keeps for the backward (`model::saved`): every layer's
    /// activations ([`SavedPolicy::Keep`]); only the layer-boundary residuals,
    /// each layer rebuilt inside the backward at the price of one extra
    /// forward ([`SavedPolicy::Recompute`]); or those boundary residuals held
    /// as bf16 / FP8 ([`SavedPolicy::Quantized`], whole-model non-MTP builds
    /// only). Normally the planner's choice (`model::memplan::saved_policy`).
    pub saved: SavedPolicy,
    /// Words of logits the chunked LM head keeps alive at once (it computes
    /// `rows x vocab` at a time instead of `tokens x vocab`); `None` takes
    /// the default of 2^28 words (1 GiB).
    pub head_budget_words: Option<u64>,
    /// Shortest sequence whose attention runs the flash-attention kernels
    /// (memory linear in the sequence instead of `heads * T * T`), on a device
    /// that can run them; `None` takes the shared default. `Some(u32::MAX)`
    /// keeps the materialised chain.
    pub flash_min_t: Option<u32>,
    /// Run each Gated-DeltaNet layer of a training pass in segments of about
    /// this many tokens (rounded down to whole chunks), so a layer's
    /// activation memory follows the segment instead of the sequence. `None`
    /// keeps the whole sequence's intermediates; needs a batch of one.
    pub gdn_segment_tokens: Option<u32>,
    /// Run each layer's dense MLP in chunks of this many tokens on a
    /// recomputing build, rebuilding a chunk's activations when its gradient
    /// is taken: the MLP's gate, up and hidden activations (and their three
    /// gradients) are most of a layer's working set once the Gated-DeltaNet
    /// part is segmented. `None` runs it whole.
    pub mlp_rows: Option<u32>,
    /// Arithmetic of the GEMMs against bf16 weights: `Auto` (the default) runs
    /// them on bf16 tensor cores where the device has them and on the fp32
    /// kernels everywhere else; `Fp32Reference` pins the fp32 kernels, which is
    /// what gradient checks and cross-backend parity are stated against. The
    /// CPU and Vulkan backends have no tensor-core path and always run fp32.
    pub gemm_precision: GemmPrecision,
    /// What a full-parameter build stores each trainable parameter as: the device AdamW state in f32, bf16 or int8
    /// blocks (the fused kernels); anything else is refused by name ([`model::TrainPrecision::resolve`]). Ignored
    /// by an adapter build, whose base is frozen.
    pub precision: model::TrainPrecision,
    /// A resident low-precision full fine-tune ([`model::resident_ft`]): the per-layer linears live in the plan's
    /// encoding (all of them on the device, or a window of layers streamed through it) with a master copy and the
    /// optimizer state on the host; the embedding, head and norms train in fp32 on the device as in any full
    /// fine-tune. Whole-model, no adapter, no multi-token head. `None` is the plain fp32 full fine-tune.
    pub resident: Option<model::resident_ft::ResidentPlan>,
}

impl Default for TrainOpts {
    fn default() -> Self {
        TrainOpts {
            base: TierPolicy::uniform(Dtype::F32),
            saved: SavedPolicy::Keep,
            head_budget_words: None,
            flash_min_t: None,
            gdn_segment_tokens: None,
            mlp_rows: None,
            gemm_precision: GemmPrecision::Auto,
            precision: model::TrainPrecision::default(),
            resident: None,
        }
    }
}

/// The tier a narrow-base LoRA build holds its frozen embedding table and LM
/// head at, or `None` when `name` stays an fp32 store tensor. Both are plain
/// `[vocab, d_model]` matrices the adapters never touch, and at the real
/// shape each is 5 GB in fp32 - more than one storage binding. bf16 halves
/// that and is the floor: unlike the layer linears they are not held at 4
/// bits, so a narrower request for the base gives them bf16. MTP shares the
/// tables with its own fp32 path and keeps them fp32.
pub(crate) fn table_tier(tier: &TierPolicy, cfg: &Qwen35Config, train: bool, name: &str) -> Option<Dtype> {
    let narrow_lora_base = train && cfg.lora.is_some() && tier.quantizes_anything();
    // An inference build at the bf16 tier holds the tables in bf16 too: a decode step reads the head once, and
    // at fp32 that is the largest single read of the step.
    let bf16_inference = !train && tier.want(name) == Dtype::BF16;
    ((narrow_lora_base || bf16_inference) && !cfg.mtp && is_table(cfg, name)).then_some(Dtype::BF16)
}

/// Whether the anchor memory's read-side linear `name` (`blocks.{l}.memory.<leaf>.weight`, not the admission scorer)
/// is held in bf16 on an inference build at the bf16 tier. The scorer, the norms and the read gate stay exact fp32.
pub(crate) fn memory_linear_bf16(tier: &TierPolicy, cfg: &Qwen35Config, train: bool, name: &str) -> bool {
    let Some((_, leaf)) = name.split_once(".memory.") else { return false };
    !train && !cfg.mtp && tier.want(name) == Dtype::BF16 && cfg.memory_linears().iter().any(|(l, _, _)| *l == leaf)
}

/// Whether `name` is one of the `[vocab, d_model]` tables: the embedding, or an untied LM head.
pub(crate) fn is_table(cfg: &Qwen35Config, name: &str) -> bool {
    name == "tok.weight" || name == cfg.head_weight()
}

/// `true` when `w` is read with fp32 (unquantized) activations: the tiers
/// whose matmul decodes the WEIGHT on the fly, so a LoRA correction computed
/// from the same fp32 activation is exact and the int8-packed activation view
/// would be dead work. The int8/int4/K-quant tiers read packed activations
/// and are inference-only.
pub(crate) fn reads_float_acts(w: &Weight) -> bool {
    !matches!(w, Weight::I8 { .. } | Weight::Q4 { .. } | Weight::KQuant { .. })
}

/// Overwrite a weight held in the `Ops` weight map with `data` (row-major
/// `[n, k]`, fp32 host values), re-encoding to the weight's own tier.
pub(crate) fn write_weight_host(gpu: &Gpu, w: &Weight, data: &[f32]) {
    assert_eq!(data.len(), w.n() as usize * w.k() as usize, "qwen35: write_weight length does not match the weight's [n, k]");
    match w {
        Weight::F32 { w, .. } => gpu.write(w, bytemuck::cast_slice(data)),
        Weight::BF16 { w, .. } => gpu.write(w, bytemuck::cast_slice(&model::half::pack_bf16(data))),
        other => panic!("qwen35: writing a {:?} base weight is not supported", other.dtype()),
    }
}

/// The widest per-token activation a training pass binds as ONE buffer of
/// `b * t` rows: `(name, floats per token)`. Every kernel binds a whole
/// buffer, so its size is capped by the device's storage binding and by the
/// u32 element index the kernels compute with.
fn widest_row_buffers(cfg: &Qwen35Config) -> [(&'static str, u64); 5] {
    [
        ("the MLP gate/up/hidden activations", cfg.intermediate_size as u64),
        ("the GDN mixed q/k/v projection", cfg.linear_conv_dim() as u64),
        ("the GQA doubled query projection", cfg.q_proj_dim() as u64),
        ("the GDN gated value", cfg.linear_value_dim() as u64),
        ("the residual stream", cfg.d_model as u64),
    ]
}

/// The most tokens (`b * t`) a training pass over `cfg` can run before one of
/// its per-token activation buffers no longer fits a single storage binding of
/// `binding_bytes` or a u32 element index, and the buffer that sets it.
pub(crate) fn longest_batch_tokens(cfg: &Qwen35Config, binding_bytes: u64) -> (u64, &'static str) {
    widest_row_buffers(cfg)
        .into_iter()
        .map(|(name, width)| ((binding_bytes / (width * 4)).min(u32::MAX as u64 / width), name))
        .min_by_key(|(tokens, _)| *tokens)
        .expect("a non-empty list")
}

/// Refuse, at construction, a `b * t` whose widest activation overflows the
/// device's storage binding or the u32 element index - the alternative is a
/// buffer that wraps silently or fails deep in a kernel launch.
pub(crate) fn check_batch_fits(cfg: &Qwen35Config, b: u32, t: u32, binding_bytes: u64) {
    let (limit, buffer) = longest_batch_tokens(cfg, binding_bytes);
    let n = b as u64 * t as u64;
    assert!(
        n <= limit,
        "qwen35: {b} x {t} = {n} tokens do not fit this device: {buffer} would not fit one storage binding of {binding_bytes} bytes \
         (or a u32 element index); at most {limit} tokens per pass"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mlp_activation_sets_the_longest_pass_at_the_real_width() {
        let cfg = Qwen35Config::qwen38_27b();
        let (tokens, buffer) = longest_batch_tokens(&cfg, 4 << 30);
        assert_eq!(tokens, (4u64 << 30) / (17408 * 4));
        assert!(buffer.contains("MLP"));
        // A 2 GiB binding (Vulkan) halves it.
        assert_eq!(longest_batch_tokens(&cfg, 2 << 30).0, (2u64 << 30) / (17408 * 4));
    }

    #[test]
    #[should_panic(expected = "do not fit this device")]
    fn a_pass_past_the_binding_is_refused_naming_the_buffer() {
        check_batch_fits(&Qwen35Config::qwen38_27b(), 1, 70_000, 4 << 30);
    }

    #[test]
    fn a_pass_inside_the_binding_is_accepted() {
        check_batch_fits(&Qwen35Config::qwen38_27b(), 1, 32_768, 4 << 30);
    }
}
