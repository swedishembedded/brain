// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving large language models that size themselves to the
// accelerator they land on for its clients. If your team needs expertise in planning context, batch and cache
// precision from a card's real memory then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The serving defaults the released-checkpoint LLM residents share: the KV-cache tier ranking, the shortest
//! context a default may leave, and the one way an unset knob is chosen and reported. Each resident decides what
//! "fits" means for its own engine; what is ranked, and how the choice is recorded, is the same for all of them.

use gpu_core::defaults::{Hardware, Rule};
use gpu_core::published_peak::{Metric, PublishedPeak};

/// The KV-cache tier a resident takes when `--kv-cache` is not given, fastest first. Decode reads every cached row of
/// every attention layer each step, so the most compact tier is also the fastest, and it holds the most context.
pub const KV_DEFAULTS: &[Rule<model::kv_tier::KvTier>] = &[Rule {
    when: |_| true,
    ranked: &[model::kv_tier::KvTier::INT8, model::kv_tier::KvTier::BF16, model::kv_tier::KvTier::F32],
    why: "decode reads every cached K/V row each step: the most compact tier is the fastest and holds the most context",
}];

/// The shortest automatic context a default may leave (or the checkpoint's own trained window, when shorter): a
/// default that would squeeze the context below it gives way to the next-ranked one.
pub const DEFAULT_CONTEXT_FLOOR: u32 = 32_768;

/// Sequences a resident decodes together when `--max-batch` is not given, most first. A decode step's cost is the
/// weights it reads, which every row of the batch shares, so the widest batch serves the most; the sequences share one
/// KV pool, so a sequence costs its recurrent-state row and its decode activations up front and its KV only as its
/// tokens fill blocks. The widest batch that still leaves one sequence of [`DEFAULT_CONTEXT_FLOOR`] is taken. A batch
/// narrower than the clients a server has is also a queue in front of it: a request beyond it waits for a running one
/// to finish (prefill and every output token) before its prompt is read in, which on a 16-client load of 8K prompts
/// put half the requests seconds behind a full batch (knowledge 499).
pub const MAX_BATCH_DEFAULTS: &[Rule<u32>] = &[Rule {
    when: |_| true,
    ranked: &[64, 32, 16, 8, 4, 2, 1],
    why: "a decode step reads the weights once for every row it carries, and the sequences share one KV pool: the widest batch that leaves one sequence the context floor",
}];

/// The operator's value, else the best-ranked default of `rules` that `fits`, recorded in `chosen` as
/// `knob value: why`.
pub fn given_or_chosen<T: Copy + std::fmt::Display>(given: Option<T>, knob: &str, rules: &[Rule<T>], hw: &Hardware, chosen: &mut Vec<String>, fits: impl FnMut(T) -> Result<(), String>) -> T {
    given.unwrap_or_else(|| {
        let c = gpu_core::defaults::choose_or_leanest(rules, hw, fits);
        chosen.push(format!("{knob} {}: {}", c.value, c.why));
        c.value
    })
}

/// The widest batch a default ranks: past it a decode step's arithmetic grows faster than its throughput.
pub const MAX_OPERATING_BATCH: u32 = 1024;

/// The bytes one weight of tier `dtype` occupies and the tensor-core rate its products run at.
fn weight_rate(dtype: gpu_core::select::Dtype) -> (f64, Metric) {
    use gpu_core::select::Dtype;
    match dtype {
        Dtype::F32 => (4.0, Metric::Tf32),
        Dtype::F16 => (2.0, Metric::Fp16),
        Dtype::BF16 => (2.0, Metric::Bf16),
        Dtype::I8 | Dtype::Q8K => (1.0, Metric::Int8),
        Dtype::F8E4M3 | Dtype::F8E5M2 => (1.0, Metric::Fp8),
        Dtype::Q4 | Dtype::Q4K | Dtype::Cq4 => (0.5, Metric::Int8),
        Dtype::NF4 | Dtype::F4E2M1 => (0.5, Metric::Fp8),
        Dtype::Cq3 => (0.375, Metric::Int8),
        Dtype::Cq2 => (0.25, Metric::Int8),
    }
}

/// The decode batch at which reading the weights once takes as long as multiplying every row by them (two
/// operations per weight per row): `bytes_per_weight * tensor_rate / (2 * memory_bandwidth)`, the nearest power of
/// two, the smallest over `cards` (a placement runs at its slowest card). Below it a step is bound by the weight read
/// and a wider batch is nearly free throughput; above it the arithmetic grows with the rows. It depends on the card
/// and the weight tier, not on the model's size. `None` when no card has rates for the tier.
pub fn decode_operating_point(cards: &[&PublishedPeak], dtype: gpu_core::select::Dtype) -> Option<u32> {
    let (bytes, metric) = weight_rate(dtype);
    cards
        .iter()
        .map(|card| {
            let rows = bytes * card.of(metric)? * 1e12 / (2.0 * card.memory_gbs * 1e9);
            Some((rows.max(1.0).log2().round() as u32).min(MAX_OPERATING_BATCH.ilog2()))
        })
        .collect::<Option<Vec<u32>>>()?
        .into_iter()
        .min()
        .map(|log2| 1 << log2)
}

/// The batch widths an unset `--max-batch` tries, widest first, and why: halving from the cards' decode operating
/// point ([`decode_operating_point`]) when their rates are known, else [`MAX_BATCH_DEFAULTS`]. The caller keeps the
/// widest that fits beside the weights and one sequence at the context floor.
pub fn max_batch_ranking(cards: &[&PublishedPeak], dtype: gpu_core::select::Dtype) -> (Vec<u32>, String) {
    match decode_operating_point(cards, dtype) {
        Some(point) => (
            std::iter::successors(Some(point), |b| (*b > 1).then_some(b / 2)).collect(),
            format!(
                "the cards' decode operating point at {dtype:?} weights (one weight read per step costs as long as {point} rows of products, from their published memory and tensor rates): a narrower batch leaves the card waiting on the weight read, so the widest batch up to it that leaves one sequence the context floor"
            ),
        ),
        None => {
            let (ranked, why) = gpu_core::defaults::candidates(MAX_BATCH_DEFAULTS, gpu_core::defaults::Hardware::detected());
            (ranked.to_vec(), format!("{why} (no published rates for these cards at {dtype:?} weights)"))
        }
    }
}

/// The published rates of every GPU of `hw`, or none when any card is not listed (a placement is planned at its
/// slowest card, which an unknown card would hide).
pub fn published_rates(hw: &Hardware) -> Vec<&'static PublishedPeak> {
    let found: Option<Vec<_>> = hw.cards.iter().map(|c| gpu_core::published_peak::lookup(&c.name)).collect();
    found.unwrap_or_default()
}

/// [`given_or_chosen`] over a ranking computed for the device ([`max_batch_ranking`]) instead of a fixed rule.
pub fn given_or_chosen_ranked<T: Copy + std::fmt::Display>(given: Option<T>, knob: &str, ranked: &[T], why: &str, chosen: &mut Vec<String>, fits: impl FnMut(T) -> Result<(), String>) -> T {
    given.unwrap_or_else(|| {
        let c = gpu_core::defaults::choose_or_leanest_from(ranked, why, fits);
        chosen.push(format!("{knob} {}: {}", c.value, c.why));
        c.value
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_core::published_peak::lookup;
    use gpu_core::select::Dtype;

    /// A decode step reads every weight once whatever its row count and multiplies each weight once per row: the
    /// batch at which the two take the same time is the card's operating point. A GH200 reaches it at about 250
    /// rows at every weight tier (its tensor rate grows as the bytes shrink), measured at batch 256 on Qwen3-4B
    /// (knowledge 538: 5.4k tok/s at 64, 7.6k at 256).
    #[test]
    fn the_operating_point_is_where_the_weight_read_meets_the_row_arithmetic() {
        let gh200 = lookup("NVIDIA GH200 120GB").unwrap();
        for dtype in [Dtype::BF16, Dtype::I8, Dtype::F8E4M3] {
            assert_eq!(decode_operating_point(&[gh200], dtype), Some(256), "{dtype:?}");
        }
        let a100 = lookup("NVIDIA A100-SXM4-80GB").unwrap();
        assert_eq!(decode_operating_point(&[a100], Dtype::BF16), Some(128));
        assert_eq!(decode_operating_point(&[gh200, a100], Dtype::BF16), Some(128), "the slowest card of a placement decides");
        assert_eq!(decode_operating_point(&[], Dtype::BF16), None, "a device with no known rates has no operating point");
    }

    /// The ranked widths descend from the operating point; a device without one keeps the fixed ranking, and the
    /// reason given names which it was.
    #[test]
    fn the_batch_ranking_starts_at_the_operating_point() {
        let gh200 = lookup("NVIDIA GH200 120GB").unwrap();
        let (ranked, why) = max_batch_ranking(&[gh200], Dtype::BF16);
        assert_eq!(ranked, vec![256, 128, 64, 32, 16, 8, 4, 2, 1]);
        assert!(why.contains("operating point"), "{why}");
        let (ranked, why) = max_batch_ranking(&[], Dtype::BF16);
        assert_eq!(ranked, MAX_BATCH_DEFAULTS[0].ranked.to_vec());
        assert!(why.contains("no published rates"), "{why}");
    }
}
