// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A scratch-arena slot reused by the next scope is not overwritten before the previous scope's readers have
//! finished with it, with NO host drain between the scopes.
//!
//! Swedish Embedded AB implements low-latency GPU inference for its clients, where an idle gap between two layers
//! of a prefill round is throughput nobody gets back. If your team needs expertise in keeping the device fed
//! while buffers are recycled safely, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! The reader is a long chain of dispatches over the slot (about as long as a layer's work, far longer than the
//! host takes to record the writer); the writer of the next scope stores different values into the same slot.
//! Were the writer able to start early, the reader's output would carry the writer's values. The test fails if
//! the arena hands the slot to a writer that runs before the reader is done.
//!
//! Skipped without a CUDA device.

use gpu_core::Gpu;

const KERNELS: [(&str, &str); 1] = [("add2", kernels::ADD2)];
const WORDS: u64 = 1 << 24;
const READS: usize = 48;

#[test]
fn a_reused_slot_is_not_written_before_the_previous_scopes_readers_finish() {
    let Ok(gpu) = Gpu::try_new_cuda(&KERNELS) else {
        eprintln!("cuda_arena_reuse: no CUDA device on this box - skipping");
        return;
    };
    assert!(gpu.submissions_ordered());
    let add2 = gpu.kernel_index("add2").expect("add2 is registered");
    let ones = gpu.storage_init("ones", &vec![1.0f32; WORDS as usize]);
    let sevens = gpu.storage_init("sevens", &vec![7.0f32; WORDS as usize]);
    let out = gpu.storage(WORDS);

    // Scope 1: slot S holds 1.0 + 1.0 (written here), then READS chained dispatches read it into `out`.
    let slot_id = {
        let _s = gpu.scratch_scope();
        let slot = gpu.storage(WORDS);
        let mut steps = vec![gpu.step(add2, &[&ones, &ones, &slot], &[WORDS as u32], WORDS as u32)];
        for _ in 0..READS {
            steps.push(gpu.step(add2, &[&slot, &slot, &out], &[WORDS as u32], WORDS as u32));
        }
        gpu.submit(&[], &steps);
        gpu.flush();
        slot.alloc_id()
    };
    // Scope 2, no poll_wait in between: the same slot is handed out again and overwritten with 7.0 + 7.0.
    {
        let _s = gpu.scratch_scope();
        let slot = gpu.storage(WORDS);
        assert_eq!(slot.alloc_id(), slot_id, "the arena replays the slot, so the hazard under test exists");
        gpu.submit(&[], &[gpu.step(add2, &[&sevens, &sevens, &slot], &[WORDS as u32], WORDS as u32)]);
        gpu.flush();
    }
    // Every reader saw 2.0: out = slot + slot = 4.0.
    let got = gpu.read(&out, WORDS as usize);
    assert!(got.iter().all(|&v| v == 4.0), "a reader saw the next scope's writer: first bad value {:?}", got.iter().find(|&&v| v != 4.0));
    gpu.scratch_release();
}
