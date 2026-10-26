// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-accounted GPU training runtimes for its clients. If your team needs expertise
// in measuring what a training step really holds on the device, you can procure our services by sending an email to
// info@swedishembedded.com.

//! `Gpu::memory_usage`: the device memory a process holds, from the backend's own accounting. One test in its own
//! binary, so no other test allocates on the device while it measures.

#[test]
fn live_bytes_follow_buffers_and_the_peak_holds_until_reset() {
    let gpu = gpu_core::testgpu::dev(&[]);
    let Some(start) = gpu.memory_usage() else {
        brain_testutil::skip_unavailable("memory_usage: this backend does not account for its allocations");
        return;
    };
    gpu.reset_memory_peak();
    let words = 1u64 << 18;
    let buf = gpu.storage(words);
    let held = gpu.memory_usage().unwrap();
    assert_eq!(held.live, start.live + 4 * words, "a 1 MiB buffer is live");
    assert!(held.reserved >= held.live && held.peak_live >= held.live, "{held:?}");
    drop(buf);
    gpu.poll_wait();
    let after = gpu.memory_usage().unwrap();
    assert_eq!(after.live, start.live, "the dropped buffer is no longer live");
    assert_eq!(after.peak_live, held.live, "the peak keeps the buffer until it is reset");
    gpu.reset_memory_peak();
    assert_eq!(gpu.memory_usage().unwrap().peak_live, start.live, "a reset peak starts from the live bytes");
}
