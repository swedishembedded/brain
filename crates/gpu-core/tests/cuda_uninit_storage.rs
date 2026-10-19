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
//!   poison never leaks into a zeroed allocation (the allocator recycles blocks).
//!
//! Skipped without a CUDA device.

use gpu_core::Gpu;

const WORDS: u64 = 1 << 20;

#[test]
fn uninitialised_storage_skips_the_clear_and_is_poisoned_in_a_debug_build() {
    let Ok(gpu) = Gpu::try_new_cuda(&[]) else {
        eprintln!("cuda_uninit_storage: no CUDA device on this box - skipping");
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
