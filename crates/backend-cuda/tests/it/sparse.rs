// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Sparse buffers: an address range reserved once, with physical memory attached to and taken back from parts
//! of it on demand.
//!
//! Swedish Embedded AB implements GPU memory management for LLM serving for its clients, where a cache that
//! holds memory for sequences that are not running costs the sequences that are. If your team needs expertise in
//! sharing one device memory budget between caches, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # What is asserted
//!
//! - **backed granules compute**: a kernel writes and reads a backed granule at the buffer's own addresses, and a
//!   granule backed later reads as zero;
//! - **the memory is shared**: backing a granule takes a granule of the card's free memory, unbacking returns it,
//!   so two sparse buffers that are never both full fit in the memory of one;
//! - **idempotent and bounded**: backing a backed granule or unbacking an unbacked one changes nothing, and a range
//!   past the buffer is refused;
//! - **returned on drop**: the live counters come back to baseline.
//!
//! Skip-if-absent, and tiny: a few granules, so it does not contend with anything resident on the card. Counters
//! and free memory are process-global, so every test holds [`serial`].

use std::sync::{Mutex, MutexGuard};

use backend_api::{Backend as _, DeviceBuffer};
use backend_cuda::exec::Context;
use backend_cuda::{live_resources, CudaBackend};

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

const KERNELS: &[(&str, &str)] = &[("axpy", kernels::AXPY)];
const AXPY: usize = 0;

fn backend() -> Option<CudaBackend> {
    match CudaBackend::try_new(KERNELS) {
        Ok(b) if b.sparse_granule().is_some() => Some(b),
        Ok(_) => {
            brain_testutil::skip_unavailable("the driver has no virtual memory management");
            None
        }
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no usable CUDA backend: {e}"));
            None
        }
    }
}

/// `out += 2 * x[offset_words ..][..n]` for a sparse `x`, read back from the dense `out`.
fn read_through_axpy(b: &CudaBackend, x: &DeviceBuffer, offset_words: u64, n: usize) -> Vec<f32> {
    let out = b.storage(n as u64);
    let step = b.step_sliced(AXPY, &[&out, x], &[(0, 0), (offset_words, 0)], &[n as u32, 1.0f32.to_bits()], n as u32);
    b.submit(&[], &[step]);
    b.read(&out, n)
}

/// `x[offset_words ..][..n] += 1 * src`, with `src` the values `1.0 ..`.
fn write_through_axpy(b: &CudaBackend, x: &DeviceBuffer, offset_words: u64, n: usize) {
    let src = b.storage_init("src", &(1..=n).map(|i| i as f32).collect::<Vec<_>>());
    let step = b.step_sliced(AXPY, &[x, &src], &[(offset_words, 0), (0, 0)], &[n as u32, 1.0f32.to_bits()], n as u32);
    b.submit(&[], &[step]);
}

#[test]
fn a_backed_granule_computes_and_a_granule_backed_later_reads_zero() {
    let _g = serial();
    let Some(b) = backend() else { return };
    let granule = b.sparse_granule().expect("checked") as usize;
    assert!(granule.is_power_of_two() && granule >= 4096, "granule {granule}");
    let words_per_granule = (granule / 4) as u64;
    let x = b.alloc_sparse("x", (4 * granule) as u64 - 1).expect("sparse allocation");
    assert_eq!(b.back_sparse(&x, 2, 1).expect("back"), 1);
    write_through_axpy(&b, &x, 2 * words_per_granule, 8);
    assert_eq!(read_through_axpy(&b, &x, 2 * words_per_granule, 8), (1..=8).map(|i| i as f32).collect::<Vec<_>>(), "a backed granule holds what a kernel wrote");

    // The granule next to it, backed afterwards, is fresh memory and leaves the first alone.
    assert_eq!(b.back_sparse(&x, 3, 1).expect("back"), 1);
    assert_eq!(read_through_axpy(&b, &x, 3 * words_per_granule, 8), vec![0.0; 8], "a granule backed later reads as zero");
    assert_eq!(read_through_axpy(&b, &x, 2 * words_per_granule, 8)[7], 8.0, "backing a neighbour left the first granule alone");

    // Taking the memory back and attaching it again gives zeros, not the old contents.
    assert_eq!(b.unback_sparse(&x, 2, 1).expect("unback"), 1);
    assert_eq!(b.back_sparse(&x, 2, 1).expect("back"), 1);
    assert_eq!(read_through_axpy(&b, &x, 2 * words_per_granule, 8), vec![0.0; 8]);
}

#[test]
fn backing_takes_free_memory_and_unbacking_returns_it() {
    let _g = serial();
    let Some(b) = backend() else { return };
    let granule = b.sparse_granule().expect("checked");
    let ctx = Context::open(0).expect("context");
    let free = || ctx.mem_info().expect("mem_info").0;
    let x = b.alloc_sparse("x", 64 * granule).expect("sparse allocation");
    let y = b.alloc_sparse("y", 64 * granule).expect("sparse allocation");
    let before = free();
    assert_eq!(b.back_sparse(&x, 0, 32).expect("back"), 32);
    let held = before.saturating_sub(free());
    // Other processes may move the figure by a little; a quarter of the expected drop is slack enough to be sure.
    assert!(held >= 24 * granule, "32 granules backed took {held} bytes of free memory");
    assert_eq!(b.unback_sparse(&x, 0, 32).expect("unback"), 32);
    assert_eq!(b.back_sparse(&y, 0, 32).expect("back"), 32);
    let held_by_y = before.saturating_sub(free());
    assert!(held_by_y <= 40 * granule, "memory taken back from one buffer did not serve the other: {held_by_y} bytes held");
    assert_eq!(b.unback_sparse(&y, 0, 32).expect("unback"), 32);
}

#[test]
fn backing_is_idempotent_and_a_range_past_the_buffer_is_refused() {
    let _g = serial();
    let Some(b) = backend() else { return };
    let granule = b.sparse_granule().expect("checked");
    let x = b.alloc_sparse("x", 8 * granule).expect("sparse allocation");
    assert_eq!(b.back_sparse(&x, 1, 3).expect("back"), 3);
    assert_eq!(b.back_sparse(&x, 0, 4).expect("back"), 1, "only the granule not yet backed is new");
    assert_eq!(b.unback_sparse(&x, 4, 4).expect("unback"), 0, "nothing was backed there");
    assert_eq!(b.unback_sparse(&x, 0, 4).expect("unback"), 4);
    assert_eq!(b.unback_sparse(&x, 0, 4).expect("unback"), 0);
    assert!(b.back_sparse(&x, 6, 3).is_err(), "granules 6..9 of an 8-granule buffer");
    assert!(b.unback_sparse(&x, 8, 1).is_err());
    let dense = b.storage(4);
    assert!(b.back_sparse(&dense, 0, 1).is_err(), "a dense buffer has nothing to attach memory to");
}

#[test]
fn dropping_a_sparse_buffer_returns_what_it_held() {
    let _g = serial();
    let before = live_resources();
    let Some(b) = backend() else { return };
    let granule = b.sparse_granule().expect("checked");
    let during = {
        let x = b.alloc_sparse("x", 16 * granule).expect("sparse allocation");
        b.back_sparse(&x, 0, 5).expect("back");
        write_through_axpy(&b, &x, 0, 4);
        live_resources()
    };
    assert_eq!(during.sparse_allocs, before.sparse_allocs + 1);
    assert_eq!(during.sparse_backed_bytes, before.sparse_backed_bytes + 5 * granule);
    b.poll_wait();
    drop(b);
    assert_eq!(live_resources(), before, "a sparse buffer left a driver object behind");
}
