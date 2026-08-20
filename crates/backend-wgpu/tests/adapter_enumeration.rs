// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Adapter enumeration must stay STABLE for the whole life of a process.
//!
//! A `wgpu::Instance` makes the Vulkan loader `dlopen` every installed ICD,
//! and destroying the last instance makes it `dlclose` them again. Some
//! vendor ICDs do not survive that cycle: after a handful of
//! create/destroy rounds the loader still finds the shared object but can no
//! longer resolve `vkCreateInstance` through it, and from then on the
//! process enumerates ZERO physical GPUs while `nvidia-smi` (and any other
//! process) still sees the cards perfectly.
//!
//! The consequence is not a clean error - `WgpuBackend::new_on_async` falls
//! back to `request_adapter`, which happily hands back the software
//! rasteriser, and the run continues on a CPU adapter with a fraction of the
//! real card's buffer limits until some allocation blows past them. So the
//! invariant worth gating is the enumeration itself: a process that opens
//! many devices in sequence (every "fresh device per forward call" model
//! does) must still see the same cards at the end that it saw at the start.

/// Enough rounds to clear the observed exhaustion point by a wide margin;
/// the failure it guards against appears well inside single digits.
const ROUNDS: usize = 24;

#[test]
fn repeated_enumeration_keeps_finding_the_same_physical_cards() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let first = backend_wgpu::enumerate_gpus();
    if first.is_empty() {
        eprintln!("no physical GPU present; nothing to keep enumerating");
        return;
    }
    for round in 1..ROUNDS {
        let now = backend_wgpu::enumerate_gpus();
        assert_eq!(
            now.len(),
            first.len(),
            "enumeration {round} found {} card(s), the first found {}: \
             the process has lost sight of hardware that is still present",
            now.len(),
            first.len()
        );
        for (a, b) in now.iter().zip(first.iter()) {
            assert!(
                a.same_device(b),
                "enumeration {round} reordered or replaced a card: {a:?} vs {b:?}"
            );
        }
    }
}
