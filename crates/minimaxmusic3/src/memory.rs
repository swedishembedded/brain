// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What one MiniMax Music 3 generation costs in DEVICE memory, stage by
//! stage - the numbers `crates/cli/src/resident_minimaxmusic3.rs`'s
//! [`residency::ResidentModel::estimate`] budgets against.
//!
//! Swedish Embedded AB implements memory-accurate model residency for
//! production inference servers. If your team needs expertise in sizing a
//! multi-stage generative pipeline so a scheduler can place it without
//! either exhausting a card or leaving one idle, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! # Why this is here and not in the adapter
//!
//! The adapter lives in `crates/cli` and cannot be unit-tested against this
//! model's own configs; these figures are pure functions of
//! [`crate::config`]'s `::real()` dims and of a small number of measurements
//! this crate's roadmap ledger records. Keeping them beside the configs is
//! what lets `cargo test -p brain-minimaxmusic3 --lib` pin them - and a
//! number nothing checks is a number that silently goes stale.
//!
//! # Derived where it can be, measured where it cannot - and never guessed
//!
//! Every weight figure below is a closed form over a config, so a different
//! config gives a different (still correct) answer. Three figures are NOT
//! derivable from a config and are named constants carrying the measurement
//! that produced them:
//!
//! * [`GLOBAL_LLM_INT8_LINEAR_BYTES`] - the Global LLM is a real Qwen3-8B
//!   whose per-layer linears are packed int8 by `qwen3::q8`; this crate has
//!   no model of that packing, so the ledger's measured figure stands.
//! * [`DIT_DEVICE_OVERHEAD_BYTES`] - the margin between the DiT's derivable
//!   block stack and a whole-device `nvidia-smi` peak (non-block tensors,
//!   the driver/Vulkan context, and the per-forward scratch).
//! * [`VOCODER_CHUNK_DEVICE_PEAK_BYTES`] - the vocoder's peak is dominated by
//!   activation and kernel scratch across a 512-fold upsample, not by its 207 MB
//!   of weights; nothing here models that, so the measurement stands.
//!
//! **The stage figures are a MAX, not a sum.** `crate::generate::generate`
//! puts each stage in its own block scope and drops it before the next one
//! loads (that module's "sequential-stage RAM discipline"), so the three
//! stages are never co-resident and the peak is whichever is largest.
//!
//! # What these numbers do NOT cover, deliberately
//!
//! They are the footprint on **one** card - the card the residency manager
//! assigned. A generation on a two-card box also borrows the OTHER card (the
//! AR stage's second Global LLM branch, [`crate::generate`]'s
//! `ar_branch_devices`; the denoise stage's second CFG branch,
//! [`crate::devplan`]). That second card is not charged to any budget today;
//! see `crate::devplan`'s own "honest gap" note and the resident adapter's
//! `estimate` doc for why, and for the seam
//! (`residency::MultiDeviceResidentModel`) that would close it.

use crate::config::{DepthDecoderConfig, DitConfig};

/// Per-layer linear weights of the flow-matching DiT, in fp32 bytes.
///
/// One block is `attn.to_{q,k,v,out}` (four `inner x inner` projections)
/// plus the gated FF pair `ff_in` (`[2*ff_inner, inner]` - GEGLU, hence the
/// 2) and `ff_out` (`[inner, ff_inner]`), exactly as
/// [`crate::dit::BlockW`] declares them. Norms and biases are `inner`-sized
/// vectors and are dropped as a rounding error, the same simplification
/// `crates/cli/src/resident_wan.rs::dit_weight_bytes` makes.
///
/// At [`DitConfig::real`] this is 36 x 64 Mi params = 2 415 919 104 params =
/// **9.664 GB**, which is the figure a real run reports for this stack.
pub fn dit_weight_bytes(cfg: &DitConfig) -> u64 {
    let inner = cfg.num_attention_heads as u64 * cfg.attention_head_dim as u64;
    let ff = cfg.ff_inner_dim as u64;
    let per_block = 4 * inner * inner + 2 * ff * inner + inner * ff;
    per_block * cfg.num_layers as u64 * 4
}

/// Device bytes a `dit::Resident` costs beyond [`dit_weight_bytes`].
///
/// Measured, not derived: `nvidia-smi` sampled once a second across a whole
/// real run on an otherwise idle Tesla P40 at [`DitConfig::real`] and a real
/// 689-latent chunk peaked at **9553 MiB** (this crate's roadmap ledger,
/// Phase 15), against a 9216 MiB derivable block stack. The 337 MiB
/// difference is the non-block tensors (`proj_in`/`proj_out`, the two 1-tap
/// convs, the time embedding), the driver's own context, and the
/// per-forward transient scratch - which that same phase measured
/// separately at 123 MiB once the flash-attention path stopped materialising
/// the `[32, 690, 690]` `scores`/`probs` slabs.
///
/// Rounded UP to 384 MiB: over-budgeting a scheduler costs a little
/// placement flexibility, under-budgeting it costs an out-of-memory abort
/// mid-generation.
pub const DIT_DEVICE_OVERHEAD_BYTES: u64 = 384 << 20;

/// Device bytes ONE `dit::Resident` holds on ONE card.
pub fn dit_device_bytes(cfg: &DitConfig) -> u64 {
    dit_weight_bytes(cfg) + DIT_DEVICE_OVERHEAD_BYTES
}

/// The Global LLM's vocabulary, from the released `language_model/
/// config.json` (`crate::global_llm::import` reads the same file). Named
/// here because the tables it sizes are 6.55 GB of the AR stage's peak and
/// nothing else in this crate states them.
pub const GLOBAL_LLM_VOCAB: u64 = 200_000;

/// The Global LLM's hidden width (same source as [`GLOBAL_LLM_VOCAB`]).
pub const GLOBAL_LLM_D_MODEL: u64 = 4096;

/// Device bytes ONE Global LLM instance's packed int8 per-layer linears
/// hold.
///
/// **Measured**, not derived: `qwen3::q8`'s packing (4 int8 lanes per `u32`
/// plus one f32 scale per output row, over exactly the projections
/// `Q8::LINEARS` names) is `crates/qwen3`'s business, not this crate's, and
/// transcribing its rule here would be a second copy that could drift. The
/// figure is this crate's roadmap ledger's own Phase 14 measurement -
/// "int8 got the linears to a MEASURED 6.95 GB" - on a real P40, which is a
/// backend that genuinely executes int8 (`backend-wgpu` reports
/// `int8_dot: true`; `backend-vulkan` queries the real DP4A property, which
/// GP102 reports as accelerated).
pub const GLOBAL_LLM_INT8_LINEAR_BYTES: u64 = 6_950_000_000;

/// Device bytes ONE Global LLM instance holds while the AR stage runs.
///
/// The int8 linears plus `tok.weight` and `lm_head.weight`, which stay
/// **fp32**: they are gathered/vocab-tiled rather than run through a
/// packed-weight GEMM, so no int8 tier applies to them
/// (`crate::generate::ar_branch_devices`'s own doc says the same). Each is
/// `[GLOBAL_LLM_VOCAB, GLOBAL_LLM_D_MODEL]` = 3.28 GB.
pub fn global_llm_device_bytes() -> u64 {
    GLOBAL_LLM_INT8_LINEAR_BYTES + 2 * GLOBAL_LLM_VOCAB * GLOBAL_LLM_D_MODEL * 4
}

/// Every weight tensor of the RVQ depth decoder, in fp32 bytes - the closed
/// form over [`crate::depth_decoder::DepthDecoderWeights`]' declared shapes:
/// per layer `attn.to_{q,k,v,out}` (4 x `h*h`) and the SwiGLU MLP triple
/// (3 x `h*intermediate`), plus `audio_embeddings`, the `num_codebooks - 1`
/// `audio_heads`, `projection` and `pos_embedding`. Layer norms are
/// `h`-sized vectors and are dropped as a rounding error.
///
/// At [`DepthDecoderConfig::real`] this is 645 988 352 params = **2.584 GB**
/// fp32.
pub fn depth_decoder_weight_bytes(cfg: &DepthDecoderConfig) -> u64 {
    let h = cfg.hidden_size as u64;
    let i = cfg.intermediate_size as u64;
    let v = cfg.audio_vocab_size as u64;
    let residual_books = cfg.num_codebooks.saturating_sub(1) as u64;
    let per_layer = 4 * h * h + 3 * h * i;
    let params = per_layer * cfg.num_layers as u64
        + residual_books * v * h  // audio_embeddings
        + residual_books * v * h  // audio_heads
        + h * h                   // projection
        + cfg.max_position_embeddings as u64 * h; // pos_embedding
    params * 4
}

/// Device bytes the vocoder stage peaks at decoding ONE chunk.
///
/// **Measured**, not derived: a P40 running `crate::vocoder::forward` over a
/// real 689-latent chunk peaked at **12264 MiB** (this crate's roadmap
/// ledger, Phase 13, which also records the two hypotheses tested and
/// rejected against that same number). The peak is activation and kernel
/// scratch across a 512-fold upsample, not weights - the whole `vocoder/`
/// checkpoint is 207 MB - and nothing in this crate models a conv's
/// transient buffers, so a closed form here would be a guess dressed as
/// arithmetic.
///
/// This is a bound for EVERY chunk, not just the one measured, and that is
/// the fact that makes a constant honest here: `crate::denoise::CHUNK_FRAMES`
/// caps a chunk at 200 AR frames, so 689 latents is the longest chunk this
/// pipeline ever vocodes regardless of how long the song is. A four-minute
/// track is ~59 chunks of this size, decoded one after another through one
/// `Gpu`, never a bigger one.
///
/// (The ledger's prose renders this as "12.26 GB". 12264 MiB is the
/// `nvidia-smi` reading it came from and the larger of the two readings, so
/// it is what is charged.)
pub const VOCODER_CHUNK_DEVICE_PEAK_BYTES: u64 = 12264 << 20;

/// The device-memory peak of each of the three sequential pipeline stages,
/// on ONE card.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StagePeaks {
    /// AR stage: one Global LLM instance plus the depth decoder, which
    /// `crate::generate::depth_decoder_device` deliberately places on the
    /// SAME card as the conditional LM branch.
    pub ar: u64,
    /// Denoise stage: one `dit::Resident`. (The condition encoder is pure
    /// host math - `crate::condition_encoder` opens no `Gpu` at all.)
    pub denoise: u64,
    /// Vocoder stage: one chunk through `crate::vocoder::forward`.
    pub vocode: u64,
}

impl StagePeaks {
    /// The largest stage - what a card must hold, since the three are never
    /// co-resident.
    pub fn peak(&self) -> u64 {
        self.ar.max(self.denoise).max(self.vocode)
    }
}

/// [`StagePeaks`] for the released checkpoint's dims.
pub fn stage_peaks() -> StagePeaks {
    StagePeaks {
        ar: global_llm_device_bytes() + depth_decoder_weight_bytes(&DepthDecoderConfig::real()),
        denoise: dit_device_bytes(&DitConfig::real()),
        vocode: VOCODER_CHUNK_DEVICE_PEAK_BYTES,
    }
}

/// Host bytes the two DERIVABLE warm-cacheable components hold once
/// [`crate::weightcache`] has read them: the DiT's block stack and the depth
/// decoder's whole weight set, both materialised fp32 by this repo's
/// safetensors reader.
///
/// The condition encoder (four tensors) and the vocoder (207 MB on disk) are
/// deliberately not in this figure - they are sized from their checkpoint
/// directories by the caller that knows where those directories are, since
/// neither has a shape this crate can close a form over without reading the
/// file.
pub fn derivable_warm_host_bytes() -> u64 {
    dit_weight_bytes(&DitConfig::real()) + depth_decoder_weight_bytes(&DepthDecoderConfig::real())
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: f64 = 1e9;

    /// The DiT's block stack, pinned to the exact figure a real run reports.
    /// This is the closed form's whole point: it reproduces the measurement
    /// rather than restating it, so a config change moves it correctly.
    #[test]
    fn the_dit_block_stack_is_9_66_gb_at_the_released_dims() {
        let bytes = dit_weight_bytes(&DitConfig::real());
        assert_eq!(bytes, 9_663_676_416, "36 blocks x 64 Mi params x 4 bytes");
        assert_eq!(bytes / 4 / (1 << 20), 2304, "2.4B params - the '2.4B DiT' this port's ledger names");
        // Below the 9553 MiB whole-device peak the ledger measured, and the
        // overhead constant must close exactly that gap or more.
        assert!(
            bytes + DIT_DEVICE_OVERHEAD_BYTES >= 9553u64 << 20,
            "the charged DiT figure ({:.2} GB) must cover the measured 9553 MiB peak",
            (bytes + DIT_DEVICE_OVERHEAD_BYTES) as f64 / GB
        );
    }

    /// The depth decoder's closed form must reproduce the 2.58 GB fp32
    /// figure measured for the released checkpoint - which is also what
    /// pins the MLP as a SwiGLU triple rather than a plain up/down pair (a
    /// pair would give 2.18 GB and this assertion would catch it).
    #[test]
    fn the_depth_decoder_is_2_58_gb_fp32_at_the_released_dims() {
        let bytes = depth_decoder_weight_bytes(&DepthDecoderConfig::real());
        assert_eq!(bytes, 2_583_953_408);
        assert!((bytes as f64 / GB - 2.58).abs() < 0.01, "{:.3} GB", bytes as f64 / GB);
    }

    /// One Global LLM instance is ~13.5 GB: the measured int8 linears plus
    /// the two fp32 vocab tables at 3.28 GB each. Both halves matter - it
    /// was charging the tables at the int8 rate that let an earlier estimate
    /// believe the pair fit one 24 GB card.
    #[test]
    fn one_global_llm_instance_is_the_int8_linears_plus_two_fp32_vocab_tables() {
        assert_eq!(GLOBAL_LLM_VOCAB * GLOBAL_LLM_D_MODEL * 4, 3_276_800_000, "tok.weight / lm_head at fp32");
        assert_eq!(global_llm_device_bytes(), 13_503_600_000);
    }

    /// The AR stage is the tallest of the three, and the whole model must
    /// fit one 24 GB card once the serving default 2 GiB headroom is kept
    /// free - otherwise `residency::place::could_ever_fit` reports
    /// `TooLarge` and the model is not merely unplaced but unservable.
    #[test]
    fn the_ar_stage_is_the_peak_and_still_fits_a_24_gb_card_with_headroom() {
        let p = stage_peaks();
        assert_eq!(p.peak(), p.ar, "AR {:.2} GB, denoise {:.2} GB, vocode {:.2} GB", p.ar as f64 / GB, p.denoise as f64 / GB, p.vocode as f64 / GB);
        assert_eq!(p.ar, 16_087_553_408);
        let card = 24u64 << 30;
        let reserve = 2u64 << 30; // `brain serve --reserve-gb`'s own default
        assert!(p.peak() < card - reserve, "peak {:.2} GB must fit a 24 GiB card minus the 2 GiB serving reserve", p.peak() as f64 / GB);
        // ...but it must NOT fit twice: this is exactly why the two AR
        // branches go on two cards, and why two concurrent generations
        // cannot share one.
        assert!(2 * p.peak() > card - reserve, "two AR branches must NOT be believed to fit one card");
    }

    /// A tiny config must produce a small, non-zero figure - the closed
    /// forms have to be functions of their config, not constants wearing a
    /// parameter.
    #[test]
    fn the_closed_forms_follow_their_config() {
        let tiny = dit_weight_bytes(&DitConfig::tiny());
        assert!(tiny > 0 && tiny < 1 << 20, "{tiny}");
        let tiny_dd = depth_decoder_weight_bytes(&DepthDecoderConfig::tiny());
        assert!(tiny_dd > 0 && tiny_dd < 1 << 20, "{tiny_dd}");
        assert!(dit_weight_bytes(&DitConfig::real()) > tiny * 1000);
    }
}
