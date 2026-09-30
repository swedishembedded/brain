// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Finite-difference gate for the SAM tower's **HD branch and neck resize**
//! (DeepSeek-VL's `sam_b_downsample`), on [`sam1::SamViTConfig::tiny_hd`].
//!
//! Two things are new relative to the plain tower `crates/sam1/tests/gradcheck.rs`
//! already gates, and each gets its own check:
//!
//!  * the scalar `hd_alpha`, whose gradient is a whole-output reduction
//!    `<d_out, hd_out>` rather than anything a conv or norm backward produces;
//!  * the compressor weights are SHARED: the main path and the HD branch both
//!    read them, so their gradient is the sum of two conv backwards. A
//!    backward that dropped either use still produces a plausible, nonzero
//!    gradient, and a single ±1 direction contracts a missing share into
//!    near-invisibility (the same blindness `crates/sam1/tests/gradcheck.rs`
//!    measured for the windowed pad rows), so those tensors are checked PER
//!    ENTRY.
//!
//! [`check_sam_hd`] is the directional check over every tensor (both necks,
//! the resize adjoint in both paths, the join of the HD adjoint into the middle
//! of the block stack); [`check_sam_hd_shared`] is the per-entry check of
//! `hd_alpha` and both compressor convs.
//!
//! ## Results, and the mutation that proves the per-entry check is needed
//!
//! Green: the directional check 60/60 tensors, worst rel 2.73e-3; the
//! per-entry check 586 entries, worst rel 6.27e-3.
//!
//! Dropping the HD branch's share of the compressor gradient (restoring both
//! compressor gradients to their pre-HD-backward values) turned
//! `compress.conv2.weight` red in the directional check (rel 0.43) but left
//! `compress.conv1.weight` PASSING it -- a whole missing use of a shared
//! weight, invisible to the best-of-four ±1 projections. The per-entry check
//! turned both red (conv1's first three entries at rel 0.74, 0.85, 1.86). The mutation was reverted
//! and both checks re-run green.
//!
//! `hd_alpha` is set to 0.7 in both: the checkpoint's own value (~5.6e-3) or
//! the reference's zero init would leave the HD branch's share of the shared
//! gradient below finite-difference resolution, and the check would pass
//! whether or not that share was accumulated.

use sam1::config::HD_ALPHA;
use sam1::{SamEncoder, SamViTConfig};

use crate::{directional_check, elementwise_check, CheckModel, Report};

/// The HD-branch scale the checks run at; see the module header.
const ALPHA: f32 = 0.7;

/// Orphan-rule wrapper, as in `crates/sam1/tests/gradcheck.rs`.
struct Check(SamEncoder);

impl CheckModel for Check {
    fn param_names(&self) -> Vec<String> {
        self.0.param_names()
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        self.0.read_weight(name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        self.0.write_weight(name, data);
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        self.0.read_grad(name)
    }
    fn loss(&self) -> f32 {
        self.0.objective()
    }
    fn zero_grads(&self) {
        self.0.zero_grads();
    }
    fn backward(&self) {
        self.0.backward();
    }
}

fn fixture(seed: u64) -> Check {
    let g = gpu_core::testgpu::dev(sam1::PIPELINES);
    let enc = SamEncoder::with_dense_init(g, SamViTConfig::tiny_hd(), seed);
    enc.write_weight(HD_ALPHA, &[ALPHA]);
    Check(enc)
}

/// Directional check over every tensor of the HD tower.
pub fn check_sam_hd(seed: u64) -> Report {
    directional_check(&fixture(seed), 5e-4, 4, seed ^ 0x4d)
}

/// Per-entry check of `hd_alpha` and of the two shared compressor convs.
pub fn check_sam_hd_shared(seed: u64) -> Report {
    let h = fixture(seed);
    let mut checks = Vec::new();
    for name in [HD_ALPHA, "vision.sam.compress.conv1.weight", "vision.sam.compress.conv2.weight"] {
        checks.extend(elementwise_check(&h, name, 5e-3).checks);
    }
    Report { checks }
}
