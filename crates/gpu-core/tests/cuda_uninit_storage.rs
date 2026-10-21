// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `Gpu::storage_uninit` hands out device storage WITHOUT the clear `Gpu::storage` pays for, and a debug build
//! poisons it so a read of a word no kernel wrote cannot pass as zero.
//!
//! Swedish Embedded AB implements GPU training and inference pipelines for its clients, where a clear of every
//! fresh buffer costs terabytes of memory traffic per step. If your team needs expertise in removing that
//! traffic without risking reads of stale device memory, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! * the clear is skipped: the call is counted as uninitialised and not as a clear;
//! * in a debug build (every `cargo test`) the contents are 0xFFFFFFFF, an f32 NaN, which is the canary: a
//!   kernel under test that leaves a word of such a buffer unwritten fails its comparison;
//! * a plain `storage` allocated after an uninitialised one of the same size is still all zeros, so the
//!   poison never leaks into a zeroed allocation (the allocator recycles blocks);
//! * a step's uniform is not cleared either: its parameters are uploaded over all of it before the dispatch runs
//!   (a prefill round over fresh buffers paid one clear per dispatch for it, about 2,000 a round on the 27B).
//!
//! Skipped without a CUDA device.

use gpu_core::Gpu;

const WORDS: u64 = 1 << 20;

/// The clear counters are process-wide; the tests here take turns so neither sees the other's clears.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn uninitialised_storage_skips_the_clear_and_is_poisoned_in_a_debug_build() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(gpu) = Gpu::try_new_cuda(&[]) else {
        brain_testutil::skip_unavailable("cuda_uninit_storage: no CUDA device on this box");
        return;
    };
    let before = gpu_core::cuda_call_totals();
    let raw = gpu.storage_uninit(WORDS);
    let after = gpu_core::cuda_call_totals();
    assert_eq!(after.uninit_calls - before.uninit_calls, 1);
    assert_eq!(after.uninit_bytes - before.uninit_bytes, WORDS * 4);
    assert_eq!(after.zero_calls, before.zero_calls, "an uninitialised allocation is not cleared");
    if cfg!(debug_assertions) {
        let v = gpu.read(&raw, WORDS as usize);
        assert!(v.iter().all(|x| x.to_bits() == 0xFFFF_FFFF), "a debug build poisons uninitialised storage");
    }
    drop(raw);
    gpu.poll_wait();

    let zeroed = gpu.storage(WORDS);
    let v = gpu.read(&zeroed, WORDS as usize);
    assert!(v.iter().all(|x| x.to_bits() == 0), "zeroed storage stays zeroed after poisoned storage of the same size was freed");
}

/// Writes the uniform's value into every one of the first `n` words.
const FILL: &str = "struct P { n: u32, v: u32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) { if (id.x < p.n) { out[id.x] = p.v; } }
";

#[test]
fn a_step_over_fresh_buffers_does_not_clear_its_uniform() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(gpu) = Gpu::try_new_cuda(&[("fill", FILL)]) else {
        brain_testutil::skip_unavailable("cuda_uninit_storage: no CUDA device on this box");
        return;
    };
    for round in 0..3u32 {
        // A fresh output each round, as a prefill round's activations are: a new uniform address every time.
        let out = gpu.storage_uninit(256);
        let before = gpu_core::cuda_call_totals();
        let step = gpu.step(0, &[&out], &[256, 7 + round, 0], 256);
        gpu.submit(&[], &[step]);
        gpu.poll_wait();
        let after = gpu_core::cuda_call_totals();
        assert_eq!(after.zero_calls, before.zero_calls, "round {round}: the step's uniform was cleared before its parameters overwrote it");
        let v: Vec<u32> = gpu.read(&out, 256).iter().map(|x| x.to_bits()).collect();
        assert!(v.iter().all(|&x| x == 7 + round), "round {round}: the kernel read the parameters it was given");
    }
}
