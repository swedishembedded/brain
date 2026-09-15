// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Installs the production GPU/CPU placer at process startup.
//!
//! The placement policy itself (`BudgetPlacer`, `budgets`, `probe_free_vram`)
//! moved to `loader::placement` -- it depends on nothing CLI-local (see
//! that module's own doc), so any embedder wanting `Device::Auto` to have a
//! real answer can call `loader::install_default_placer` directly. What
//! stays here is exactly the one CLI-specific fact that function cannot know
//! on its own: which cards `--device` made schedulable for THIS process.

/// Install the production placer for this process.
///
/// Called once from `main` after `--device` has been resolved, so the
/// candidate set is exactly what the user made schedulable. With no GPU
/// present this still installs (the host tier answers), and with an explicit
/// `--device gpu<i>` the placer is never consulted - `gpu_core::devices::
/// selected_device` only asks when the user expressed no preference.
pub fn install() {
    // `--device` narrows the candidate set; with no `--device` every card is
    // a candidate, which is the "use all the hardware" default. How FULL those
    // cards are is re-probed on a TTL by the placer itself rather than frozen
    // here - `brain serve` calls this once at startup and then lives for weeks.
    let gpus = crate::compute_set().map(|s| s.gpus.iter().copied().collect());
    // `--device gpu`/`gpu<i>` must keep CPU out of every later re-probe too,
    // not just this first snapshot - the reported bug was exactly this:
    // `--device gpu` excluded CPU from the candidate SET but the placer kept
    // adding a CPU tier anyway, so a part too big for the GPUs landed on
    // `Home::Cpu` and then panicked inside a model that cannot run int8 there
    // at all. No `--device` (`compute_set()` still `None` this early, or a
    // ComputeSet with CPU enabled) keeps the historical "everything" default.
    let cpu_allowed = crate::compute_set().map(|s| s.cpu_enabled()).unwrap_or(true);
    loader::install_default_placer(gpus, cpu_allowed);
}
