// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **brain performance benchmarking** — how much *correct work meeting its SLO*
//! brain delivers per unit of hardware, memory, energy and time.
//!
//! Sibling to `crates/bench`, and deliberately not the same thing:
//!
//! | crate | question | compares |
//! |---|---|---|
//! | `bench` | can this architecture **learn** task X? | architectures |
//! | `perf`  | how **fast**, and at what cost — and is it still correct? | models × hardware × config |
//!
//! # What makes this brain's suite rather than a generic LLM harness
//!
//! brain serves detection, depth, TTS, image generation, forecasting, 3D and
//! world models from one engine across three backends (plus a separate
//! whole-graph NPU compiler path), so the harness cannot be
//! written in terms of "tokens". It measures **artifacts arriving over time**
//! along the timeline `submit → admit → first → … → done`, which specialises to
//! TTFT/ITL/TPOT for a decoder and collapses cleanly to a single latency for a
//! one-shot model. `capability::Action`'s existing `Progress` callback supplies
//! that timeline for free, so any model implementing `capability::Provider` is
//! benchmarkable with no new benchmark code.
//!
//! # Rules the harness enforces rather than trusting callers with
//!
//! * warm-up requests never enter a statistic;
//! * a failed or unfinished request is never goodput, and never leaves the
//!   denominator;
//! * unmeasured fields serialise as `null`, never `0`;
//! * a run whose correctness gate did not pass is marked `valid: false` and is
//!   excluded from comparison;
//! * results in different artifact units are never ranked against each other;
//! * a software rasteriser is labelled as one wherever it appears — a machine
//!   with no real GPU still serves `--device gpu`, and reporting that as a "GPU
//!   number" is worse than reporting nothing.
//!
//! ```no_run
//! # fn main() -> Result<(), String> {
//! use perf::{scenarios, workload};
//! # struct T; impl perf::target::PerfTarget for T {
//! #   fn describe(&self) -> perf::target::TargetInfo { perf::target::TargetInfo::new("m","token") }
//! #   fn submit(&mut self, _r: perf::target::PerfRequest) -> u64 { 0 }
//! #   fn step(&mut self, _o: &mut Vec<perf::target::Emission>) -> bool { false }
//! #   fn busy(&self) -> bool { false } }
//! let mut target = T;
//! let opt = scenarios::Options::default();
//! let art = scenarios::run("sweep", &mut target, "chat", 0, &opt)?;
//! println!("{}", scenarios::render(&art));
//! art.write(&art.default_path(opt.seed)).map_err(|e| e.to_string())?;
//! # Ok(()) }
//! ```

pub mod devicetel;
pub mod driver;
pub mod env;
pub mod fidelity;
pub mod gate;
pub mod metrics;
pub mod report;
pub mod scenarios;
pub mod schema;
pub mod stats;
pub mod target;
pub mod targets;
pub mod workload;

pub use schema::Artifact;
pub use scenarios::{Options, SCENARIOS};
pub use target::{PerfRequest, PerfTarget, TargetInfo};
pub use workload::{Arrival, Workload, STANDARD};

/// One-line descriptions for `brain perf list`.
pub fn list() -> String {
    let mut s = String::from("scenarios:\n");
    for (name, desc) in SCENARIOS {
        s.push_str(&format!("  {name:<12} {desc}\n"));
    }
    s.push_str("\nworkloads (input/output artifacts):\n");
    for name in STANDARD {
        if let Some(w) = workload::standard(name, Arrival::Saturated, 1, 0) {
            let r = &w.requests()[0];
            s.push_str(&format!(
                "  {:<15} {:>6} / {:<6}\n",
                name, r.input_artifacts, r.output_artifacts
            ));
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_covers_every_scenario_and_workload() {
        let s = list();
        for (name, _) in SCENARIOS {
            assert!(s.contains(name), "list must mention scenario {name}");
        }
        for name in STANDARD {
            assert!(s.contains(name), "list must mention workload {name}");
        }
    }
}
