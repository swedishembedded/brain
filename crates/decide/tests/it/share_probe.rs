// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Can a step dispatched on one `Gpu` handle READ a buffer allocated on
//! another handle of the same device?
//!
//! `Gpu::share` is documented as handing out another handle to the same
//! device, which a model composed of two halves depends on: the head reads the
//! encoder's hidden states. If that read silently yields zeros rather than
//! failing, every downstream stage is zero and the model still trains, still
//! answers, and is simply always wrong - so the answer is pinned here.

use decide::kern::PIPELINES;

#[test]
fn a_shared_handle_sees_the_other_handles_buffer() {
    let a = gpu_core::testgpu::dev(PIPELINES);
    let b = a.share();
    let src = a.storage(8);
    a.write_f32(&src, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    // Read it back through the SAME handle first, so a failure below cannot be
    // blamed on the write.
    assert_eq!(a.read(&src, 8), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "same-handle readback");

    // Now have the OTHER handle dispatch a kernel that reads it: gather all
    // eight values through `embed` with a width of 1.
    let idx = b.buffer("idx", 8 * 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST);
    b.write(&idx, &[7u32, 6, 5, 4, 3, 2, 1, 0]);
    let out = b.storage(8);
    let embed = b.kernel_index("embed").expect("embed registered");
    b.submit(&[], &[b.step(embed, &[&idx, &src, &out], &[1, 8], 8)]);
    assert_eq!(
        b.read(&out, 8),
        vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        "a step on a shared handle read zeros instead of the other handle's buffer"
    );
}
