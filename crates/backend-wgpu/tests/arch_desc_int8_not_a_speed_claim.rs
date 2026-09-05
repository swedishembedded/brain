// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! M8.1 gate: `backend-wgpu`'s `I8` tier is `Emulated`, never `Native`, and
//! never reads as fast - `dot4I8Packed` is core WGSL (naga lowers it to
//! hardware DP4A or a polyfill, and wgpu has no way to query which one
//! happened), so this backend can only ever make the weaker, honest claim.
//! This is the exact conflation M8.1 exists to stop: before it, wgpu
//! hardcoded `NumericSupport::int8_dot: true` unconditionally, which read
//! identically to `backend-vulkan`'s truthful, hardware-queried `true` - a
//! selector had no way to prefer real hardware over a polyfill. Real device,
//! not a synthetic `ArchDesc` (see `crates/backend-api/tests/
//! arch_view_agrees.rs` for the synthetic table).

use backend_api::arch::TierLevel;
use backend_api::{Backend, DType};
use backend_wgpu::WgpuBackend;

#[test]
fn int8_dot_is_not_a_speed_claim_on_wgpu() {
    let wg = WgpuBackend::new(&[("add2", kernels::ADD2)]);
    let caps = wg.caps();

    assert_eq!(
        caps.arch.tier(DType::I8).level,
        TierLevel::Emulated,
        "wgpu must never claim Native for I8 - it cannot query real DP4A hardware"
    );
    assert!(!caps.arch.is_fast(DType::I8), "Emulated must never read as fast without a real measurement");
}
