// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **Capability axes** — the grouping that turns a flat list of benchmark scores
//! into a small, interpretable architecture profile.
//!
//! Each benchmark probes a *capability*; several benchmarks probe the same one
//! (e.g. MQAR, MAD-recall, fuzzy-recall and noisy-recall all probe in-context
//! **recall**). An axis's score is the **mean of its benchmarks' headline
//! scores**, so two architectures can be compared on "how good is each at recall
//! / copying / state-tracking / …" rather than on a dozen individual numbers.
//! This is the layer the predictive-scaling + tuning-advisor work builds on: it
//! reasons per-axis, not per-benchmark.
//!
//! Axes (and the benchmarks mapped onto them):
//! - **recall** — in-context key→value lookup: `mqar`, `mad_recall`,
//!   `mad_fuzzy_recall`, `mad_noisy_recall`.
//! - **copying** — copy/route spans of the input: `mad_selective_copy`,
//!   `toolcall`.
//! - **memory** — store & reproduce content with no lookup cue: `mad_memorize`.
//! - **state_tracking** — carry algorithmic / hierarchical state: `parity`,
//!   `dyck`.
//! - **compression** — bottleneck reconstruction: `mad_compress`.
//! - **arithmetic** — modular-addition generalization (grokking): `mod_add`
//!   (this benchmark is *informational*; see [`crate::Benchmark::informational`]).

/// The canonical list of capability axes, in display order.
pub const AXES: &[&str] =
    &["recall", "copying", "memory", "state_tracking", "compression", "arithmetic"];

/// The capability axis a benchmark belongs to. Unknown names map to `"other"` so
/// a newly-registered benchmark is still surfaced (and a `debug_assert` in tests
/// catches the missing mapping).
pub fn axis_of(name: &str) -> &'static str {
    match name {
        "mqar" | "mad_recall" | "mad_fuzzy_recall" | "mad_noisy_recall" => "recall",
        "mad_selective_copy" | "toolcall" => "copying",
        "mad_memorize" => "memory",
        "parity" | "dyck" => "state_tracking",
        "mad_compress" => "compression",
        "mod_add" => "arithmetic",
        _ => "other",
    }
}

/// The canonical axis list (owned `&str`s, in display order).
pub fn axes() -> Vec<&'static str> {
    AXES.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_registered_benchmark_has_a_known_axis() {
        for b in crate::registry() {
            let ax = axis_of(b.name());
            assert!(
                AXES.contains(&ax),
                "benchmark '{}' maps to unknown axis '{}'; add it to axes.rs",
                b.name(),
                ax
            );
        }
    }

    #[test]
    fn axis_examples() {
        assert_eq!(axis_of("mqar"), "recall");
        assert_eq!(axis_of("toolcall"), "copying");
        assert_eq!(axis_of("parity"), "state_tracking");
        assert_eq!(axis_of("mad_compress"), "compression");
        assert_eq!(axis_of("mod_add"), "arithmetic");
        assert_eq!(axis_of("???"), "other");
    }
}
