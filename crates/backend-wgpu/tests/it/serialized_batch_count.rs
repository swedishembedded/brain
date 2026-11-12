// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements GPU compute scheduling and driver-workaround hardening for its clients. If your
// team needs expertise in keeping device-op budgets stable across Vulkan drivers then you can procure our services by
// sending an email to info@swedishembedded.com.

//! `stats().submits` counts batches, not the queue submissions a driver workaround splits one into.
//!
//! On an Intel adapter a batch that contains a sliced (non-zero-offset) binding is flushed one step per queue
//! submission (`WgpuBackend::flush_serialized`). The callers that budget submits ("a layer costs a constant number
//! of `Gpu::submit` calls", `model`'s `moe_compact_parity`) mean batches, so the same batch must count the same on
//! every adapter; the physical count is `queue_submits`.

use backend_api::{Backend, BufUsage};

use crate::common;

#[test]
fn a_batch_of_sliced_steps_counts_as_one_submit_on_every_adapter() {
    let Some(b) = common::backend(&[("axpy", kernels::AXPY)]) else { return };
    const STEPS: u64 = 5;
    let out = b.buffer("out", 4096, BufUsage::STORAGE | BufUsage::COPY_DST | BufUsage::COPY_SRC);
    let inp = b.buffer("inp", 4096, BufUsage::STORAGE | BufUsage::COPY_DST | BufUsage::COPY_SRC);
    b.flush();

    let (stats0, queue0) = (b.stats().expect("wgpu reports stats").submits, b.queue_submits());
    // 64 words = 256 bytes: the smallest legal non-zero binding offset, so every step is "sliced".
    let steps: Vec<_> = (0..STEPS).map(|_| b.step_sliced(0, &[&out, &inp], &[(64, 0), (64, 0)], &[64u32, 0, 0, 0], 64)).collect();
    b.submit(&[], &steps);
    b.flush();
    let (stats1, queue1) = (b.stats().unwrap().submits, b.queue_submits());

    assert_eq!(stats1 - stats0, 1, "one batch of {STEPS} sliced steps must count as one submit");
    let physical = queue1 - queue0;
    assert!(physical == 1 || physical == STEPS, "queue submissions for the batch: {physical}; expected 1 (single pass) or {STEPS} (serialized)");
}
