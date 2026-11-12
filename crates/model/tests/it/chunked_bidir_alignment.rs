// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements GPU attention kernels for long-context training for its clients. If your team needs
// expertise in chunked attention on devices with strict storage-binding alignment then you can procure our services
// by sending an email to info@swedishembedded.com.

//! The chunked attention entry points bind row windows at `row * width` words, so `stride` and `d_out` must be whole
//! storage-binding grains (64 words). A call that breaks that must fail at the call, naming the width, not at
//! whichever chunk first starts off the grain. A call whose windows happen to start on aligned rows stays legal: a
//! single chunk from row 0 binds only offset 0 and takes any width (`chunked_bidir_fwd_win.rs` runs one at `d = 8`).

use gpu_core::Gpu;
use model::block::{self, CrossBwdIds, CrossIds};

fn dev() -> Gpu {
    gpu_core::testgpu::dev(&[("attn_scores_cross", kernels::ATTN_SCORES_CROSS)])
}

fn ids() -> (CrossIds, CrossBwdIds) {
    (CrossIds { scores: 0, softmax: 0, apply: 0 }, CrossBwdIds { dscores: 0, dscores_rows: None, dq: 0, dk_acc: 0, dv_acc: 0 })
}

/// `stride = 3 * 16 = 48` and `d_out = 16` are legal widths for the kernels but not for a sliced binding.
const MISALIGNED: (u32, u32) = (48, 16);

#[test]
#[should_panic(expected = "chunked_bidir_fwd: the window starting at row 2 binds stride at word 2 * 48 = 96")]
fn forward_refuses_a_stride_off_the_binding_grain() {
    let g = dev();
    let (fwd, _) = ids();
    let b = g.storage(1);
    let mut steps = Vec::new();
    // Two chunks of one span: the second binds row 2 at `2 * 48` words, which is off the grain.
    block::chunked_bidir_fwd(&g, &fwd, None, 1, 16, MISALIGNED.1, &b, MISALIGNED.0, 0, 16, 32, &b, &b, &b, &[(0, 4)], 2, None, &mut steps);
}

#[test]
#[should_panic(expected = "chunked_bidir_bwd: the window starting at row 2 binds d_out at word 2 * 16 = 32")]
fn backward_refuses_a_width_off_the_binding_grain() {
    let g = dev();
    let (fwd, bwd) = ids();
    let b = g.storage(1);
    let mut steps = Vec::new();
    // A legal stride and a misaligned context width: `d_ctx` and `d_qkv` windows are bound at `row * d_out`.
    block::chunked_bidir_bwd(&g, &fwd, None, &bwd, 1, 16, MISALIGNED.1, &b, 192, 0, 64, 128, &b, &b, &b, &b, &b, &[(0, 4)], 2, None, &mut steps);
}
