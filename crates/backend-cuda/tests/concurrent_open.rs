// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A handle can be opened while another thread is capturing a graph.
//!
//! Swedish Embedded AB implements accelerator runtimes that stay correct when
//! several threads share one device. If your team needs expertise in
//! multi-threaded GPU runtimes, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! In a file of its own because it hammers the driver with stream creation from
//! a second thread, and the timing assertions in `cuda_graphs.rs` would read
//! that as load: separate test binaries run one after the other.

use backend_api::Backend as _;
use backend_cuda::CudaBackend;

const KERNELS: &[(&str, &str)] = &[("axpy", kernels::AXPY)];
const AXPY: usize = 0;
const N: usize = 256;

fn input() -> Vec<f32> {
    (0..N).map(|i| 1.0 + i as f32 * 0.125).collect()
}

fn round(b: &dyn backend_api::Backend, outs: &[backend_api::DeviceBuffer], inp: &backend_api::DeviceBuffer, s: f32) {
    let params = [N as u32, s.to_bits()];
    let steps: Vec<_> = outs.iter().map(|o| b.step(AXPY, &[o, inp], &params, N as u32)).collect();
    b.submit(&[], &steps);
}

fn backend() -> Option<CudaBackend> {
    match CudaBackend::try_new(KERNELS) {
        Ok(b) => Some(b),
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no usable CUDA backend: {e}"));
            None
        }
    }
}

/// Opening a handle must not fail because a different thread is in the middle
/// of capturing a graph on its own handle.
///
/// A new handle drains the device once, so work another handle issued before
/// it could know a second handle existed has finished. A context-wide drain
/// waits on every stream in the context, and a stream that is being captured
/// cannot be waited on: the driver refuses with "operation not permitted when
/// stream is capturing". `share` reports a refusal as "this backend cannot
/// share a device", which a sharded model takes to mean it must run unsharded
/// - a different, slower answer chosen at random by whichever thread happened
/// to be capturing.
///
/// One thread captures a stream of fresh shapes (every distinct chain length
/// is a new capture) while another opens handles back to back. Every open must
/// succeed.
#[test]
fn opening_a_handle_while_another_thread_captures_succeeds() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let Some(parent) = backend() else { return };
    let capturing = AtomicBool::new(true);
    let (opens, refused) = std::thread::scope(|scope| {
        let capturer = scope.spawn(|| {
            let b = parent.share().expect("the capturing handle");
            let inp = b.storage_init("inp", &input());
            for shape in 0..60 {
                // A new chain length is a new signature, hence a new capture.
                let outs: Vec<_> = (0..8 + shape).map(|_| b.storage(N as u64)).collect();
                for _ in 0..4 {
                    round(b.as_ref(), &outs, &inp, 0.5);
                }
                b.poll_wait();
            }
            capturing.store(false, Ordering::Release);
        });
        let opener = scope.spawn(|| {
            let (mut opens, mut refused) = (0usize, 0usize);
            while capturing.load(Ordering::Acquire) {
                opens += 1;
                if parent.share().is_none() {
                    refused += 1;
                }
                // Not a spin: tests in this binary measure host time, and an
                // opener that owns a core would be the load that fails them.
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
            (opens, refused)
        });
        capturer.join().expect("capturing thread");
        opener.join().expect("opening thread")
    });
    assert_eq!(
        refused, 0,
        "{refused} of {opens} opens were refused while another thread captured"
    );
}
