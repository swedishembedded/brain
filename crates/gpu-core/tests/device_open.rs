// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gate for [`Gpu::open`] - the ONE "an `Option<&str>` device token to a
//! `Gpu` handle" mapping.
//!
//! Before this existed, seventeen byte-identical private copies of
//! ```ignore
//! match device {
//!     Some("cpu")               => Gpu::new_cpu(&KERNELS),
//!     Some("gpu") | Some("wgpu") => Gpu::new_wgpu(&KERNELS),
//!     _                          => Gpu::new(&KERNELS),
//! }
//! ```
//! were spread across `vae`, `diamond`, `s3dit`, `wan`, `gemma4`, `ltxv`
//! and `minimaxmusic3`. Every one of them silently mangled an INDEXED card
//! (`gpu0`/`gpu1`) into the ambient selection through the `_` arm, so a
//! model asked to run on the second card ran on the first - the same
//! defect class `tests/device_grammar.rs` records for `BRAIN_DEVICE`'s own
//! weak ladder, one layer up. Indexing is what a two-card box needs to
//! place two stages on two cards, so it is gated here rather than left to
//! each caller to rediscover.
//!
//! Kept cheap on purpose: it asserts the SELECTION, not any compute. The
//! `gpu*` cases skip themselves when the box has no discrete card, per the
//! workspace's absent-hardware convention (a skip that names itself, never
//! a bare `return`).

use gpu_core::Gpu;

/// One real kernel - this gate never dispatches, it only builds, but the
/// source still has to COMPILE on every backend (the CPU JIT requires the
/// `global_invocation_id`/`num_workgroups` builtins to be taken). Reusing
/// the shared `add2` rather than hand-rolling a toy is the same choice
/// `gpu_core`'s own in-crate tests make.
const KERNELS: &[(&str, &str)] = &[("add2", kernels::ADD2)];

/// Every test in this file builds its own independent real `Gpu` (`Gpu::open`/
/// `Gpu::new` below `testgpu`'s shared pool entirely) rather than sharing one,
/// so - like `device_churn.rs` and `device_sharing.rs` - nothing here is
/// protected from a sibling test's concurrent device construction racing
/// against it under `cargo test`'s default multi-threaded harness. Found by
/// direct reproduction: a full `cargo test -p brain-gpu-core --lib --tests`
/// run hung with this file's own test binary pinned at ~100% CPU on one
/// thread (GPUs otherwise idle) - the same busy-wait-on-a-fence-that-never-
/// signals signature already diagnosed for this exact hazard class, not
/// reproducible running this file alone. Same one-lock-per-file fix as
/// those two files.
static DEVICE_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn an_explicit_cpu_token_selects_the_cpu_backend() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    assert_eq!(Gpu::open(Some("cpu"), KERNELS).kind(), "cpu");
}

#[test]
fn none_matches_the_ambient_constructor() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    // `None` must mean exactly "whatever `Gpu::new` would have done" -
    // callers pass it to opt OUT of overriding, not to pick a default of
    // their own.
    assert_eq!(Gpu::open(None, KERNELS).kind(), Gpu::new(KERNELS).kind());
}

#[test]
fn an_indexed_card_is_honoured_not_silently_mangled_to_ambient() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let n = gpu_core::devices::gpus().len();
    if n < 2 {
        eprintln!("SKIP an_indexed_card_is_honoured_not_silently_mangled_to_ambient: needs 2 discrete GPUs, found {n}");
        return;
    }
    // The regression: `Some("gpu1")` used to fall through the `_` arm to
    // `Gpu::new`, landing on card 0. Both handles must report a GPU
    // backend, and card 1 must be reachable at all.
    for tok in ["gpu0", "gpu1"] {
        let g = Gpu::open(Some(tok), KERNELS);
        assert_ne!(g.kind(), "cpu", "{tok} must not resolve to the CPU backend");
    }
}

#[test]
#[should_panic(expected = "gpu9")]
fn an_out_of_range_card_is_an_error_never_a_silent_clamp() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    // AGENTS.md: "Out-of-range indices are errors, never silent clamps."
    let _ = Gpu::open(Some("gpu9"), KERNELS);
}
