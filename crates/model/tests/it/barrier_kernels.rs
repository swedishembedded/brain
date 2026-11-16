// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The one check a test of a workgroup-barrier kernel makes before it builds anything on its device.
//!
//! The CPU JIT compiles no kernel that synchronises a workgroup (`DeviceCaps::workgroup_reductions` is false there),
//! so a test of such a kernel - the register-tiled int8 and K-quant GEMMs, the flash attention trio, the fused prefill
//! attention - has nothing to run under `BRAIN_BACKEND=cpu`; `BRAIN_DEVICE=software` runs the same kernels on lavapipe
//! instead. Such a test reports itself skipped by name there; on every device that runs the kernels it keeps every
//! assertion it has.

use gpu_core::Gpu;

/// True when `g` cannot run workgroup-barrier kernels: the caller has reported `what` skipped and returns.
pub fn unavailable(g: &Gpu, what: &str) -> bool {
    if g.caps().workgroup_reductions {
        return false;
    }
    brain_testutil::skip_unavailable(&format!("{what}: workgroup-barrier kernels, which the {} backend does not run (BRAIN_DEVICE=software runs them)", g.kind()));
    true
}
