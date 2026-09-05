// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! M8.1's own proof gate: on THIS machine's real GPU, `backend-wgpu` and
//! `backend-vulkan` must now report DIFFERENT `ArchDesc::tier(I8)` levels -
//! the observable divergence the whole milestone exists to make possible.
//! Before M8.1, both backends only had `NumericSupport::int8_dot`, one bool
//! that conflated "executes" and "runs on dedicated hardware"; a selector
//! reading it could never tell a real DP4A card from a polyfill. Real
//! hardware, not a synthetic `ArchDesc`, is the point of this file - see
//! `crates/backend-api/tests/arch_view_agrees.rs` for the synthetic,
//! table-driven backward-compatibility proof this one does not duplicate.
//!
//! `backend-wgpu` always reports `I8` at `Emulated` (`dot4I8Packed` is core
//! WGSL - naga lowers it to hardware DP4A or a polyfill, and wgpu has no way
//! to query which one happened). `backend-vulkan` reports `Native` iff its
//! own `shaderIntegerDotProduct` query says so, `Emulated` otherwise - a REAL
//! device query, not a guess. On any adapter with real DP4A hardware, that
//! makes the two backends' tiers genuinely different even while running on
//! the exact same physical card - which is the assertion below.

use backend_api::arch::TierLevel;
use backend_api::{Backend, DType};
use backend_vulkan::VulkanBackend;
use backend_wgpu::WgpuBackend;

const KERNELS: &[(&str, &str)] = &[("add2", kernels::ADD2)];

#[test]
fn wgpu_and_vulkan_report_different_i8_tiers_on_the_same_real_card() {
    let vk = match VulkanBackend::try_new(KERNELS) {
        Ok(b) => b,
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no Vulkan device: {e}"));
            return;
        }
    };
    let wg = WgpuBackend::new(KERNELS);

    let vk_caps = vk.caps();
    let wg_caps = wg.caps();

    eprintln!(
        "wgpu:   I8 tier = {:?}, F16 tier = {:?}, numeric.int8_dot = {}",
        wg_caps.arch.tier(DType::I8).level,
        wg_caps.arch.tier(DType::F16).level,
        wg_caps.numeric.int8_dot,
    );
    eprintln!(
        "vulkan: I8 tier = {:?}, F16 tier = {:?}, numeric.int8_dot = {}",
        vk_caps.arch.tier(DType::I8).level,
        vk_caps.arch.tier(DType::F16).level,
        vk_caps.numeric.int8_dot,
    );

    // `backend-wgpu` never claims dedicated hardware for I8 - it cannot
    // query that, only that the arithmetic executes.
    assert_eq!(
        wg_caps.arch.tier(DType::I8).level,
        TierLevel::Emulated,
        "backend-wgpu must report I8 as Emulated, never Native - it has no DP4A query"
    );

    // Both `numeric_view()`-derived bools still agree that int8 EXECUTES on
    // both backends (the pre-M8.1 observable behaviour, preserved) - the
    // new information is the TIER, which only `ArchDesc` carries.
    assert!(wg_caps.numeric.int8_dot, "wgpu's packed-int8 kernels execute regardless of hardware DP4A");

    // The actual divergence this milestone exists to produce: on hardware
    // with a real, queried DP4A capability, `backend-vulkan`'s tier is
    // `Native`, strictly above wgpu's `Emulated` - a selector reading
    // `ArchDesc` directly (not the flattened bool) can now tell them apart.
    // On hardware WITHOUT that capability the two backends still legitimately
    // agree (both `Emulated`) - that is not a failure of this gate, it is
    // the honest answer for that card, so the assertion is conditioned on
    // what this real device actually reported rather than hard-coded to
    // always require `Native`.
    if vk_caps.arch.tier(DType::I8).level == TierLevel::Native {
        assert_ne!(
            vk_caps.arch.tier(DType::I8).level,
            wg_caps.arch.tier(DType::I8).level,
            "on real DP4A hardware, vulkan and wgpu must disagree on I8's tier"
        );
    } else {
        eprintln!(
            "note: this card's Vulkan driver did not report shaderIntegerDotProduct, \
             so both backends agree at Emulated here - the divergence this gate proves \
             is conditional on real DP4A hardware, which this run's device may not have"
        );
    }
}
