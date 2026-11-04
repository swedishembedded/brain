// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The urgent lane (`gpu_core::lane`): work issued inside `lane::urgent` runs
//! on a second, high-priority stream of the same device, so a decode step is
//! not queued behind a prefill round that is already running.
//!
//! Swedish Embedded AB implements latency-critical inference serving for its
//! clients. If your team needs expertise in keeping interactive generation
//! responsive while a GPU is busy with bulk work, you can procure our services
//! by sending an email to info@swedishembedded.com.
//!
//! The gates: a lane changes where work runs, never what it computes (the same
//! dispatch gives the same bits in either lane, on buffers made in the other);
//! and a read in the urgent lane does not wait for the normal lane, whose
//! `is_idle` answers without waiting while it is busy. The `#[ignore]`d
//! measurement prints a decode GEMV's latency alone, queued behind a running
//! prefill GEMM, and beside it in the urgent lane.

use gpu_core::{lane, Dispatch, Gpu};
use std::time::Instant;

const KERNELS: &[(&str, &str)] = &[("matmul_i8_gemv", kernels::MATMUL_I8_GEMV), ("matmul_i8_dyn", kernels::MATMUL_I8_DYN)];
const K_GEMV: usize = 0;
const K_GEMM: usize = 1;

fn urgent_cuda() -> Option<Gpu> {
    let gpu = Gpu::new(KERNELS);
    if gpu.kind() != "cuda" || !gpu.caps().numeric.int8_dot || !gpu.has_urgent_lane() {
        brain_testutil::skip_unavailable("the urgent lane needs a CUDA device with a packed int8 dot");
        return None;
    }
    Some(gpu)
}

/// Packed int8 operands of an `m x k` by `n x k` product with group scales.
struct Operands {
    x: gpu_core::DeviceBuffer,
    sx: gpu_core::DeviceBuffer,
    w: gpu_core::DeviceBuffer,
    sw: gpu_core::DeviceBuffer,
    out: gpu_core::DeviceBuffer,
    /// Zeroed by `submit_clearing`; a buffer a captured graph names without any step's uniform naming it.
    scratch: gpu_core::DeviceBuffer,
    m: u32,
    k: u32,
    n: u32,
}

impl Operands {
    fn new(gpu: &Gpu, m: u32, k: u32, n: u32, seed: u64) -> Operands {
        let mut r = data::rng::Lcg::new(seed);
        let words = |len: u32, r: &mut data::rng::Lcg| (0..len).map(|_| r.next_u32()).collect::<Vec<u32>>();
        let x = gpu.storage(u64::from(m * k / 4));
        gpu.write(&x, &words(m * k / 4, &mut r));
        let w = gpu.storage(u64::from(n) * u64::from(k / 4));
        gpu.write(&w, &words(n * (k / 4), &mut r));
        let sx = gpu.storage(u64::from(m * (k / 32)));
        gpu.write_f32(&sx, &vec![0.01; (m * (k / 32)) as usize]);
        let sw = gpu.storage(u64::from(n * (k / 32)));
        gpu.write_f32(&sw, &vec![0.02; (n * (k / 32)) as usize]);
        Operands { x, sx, w, sw, out: gpu.storage(u64::from(m * n)), scratch: gpu.storage(1), m, k, n }
    }

    fn submit(&self, gpu: &Gpu, kind: usize) {
        let threads = if kind == K_GEMV { Dispatch::Workgroups(self.n) } else { Dispatch::Threads(self.m * self.n) };
        let step = gpu.dispatch(kind, &[&self.x, &self.w, &self.sx, &self.sw, &self.out], &[self.m, self.k / 4, self.n], threads);
        gpu.submit(&[], &[step]);
    }

    /// [`Self::submit`] with `scratch` as the submission's clear.
    fn submit_clearing(&self, gpu: &Gpu, kind: usize) {
        let threads = if kind == K_GEMV { Dispatch::Workgroups(self.n) } else { Dispatch::Threads(self.m * self.n) };
        let step = gpu.dispatch(kind, &[&self.x, &self.w, &self.sx, &self.sw, &self.out], &[self.m, self.k / 4, self.n], threads);
        gpu.submit(&[&self.scratch], &[step]);
    }

    fn bits(&self, gpu: &Gpu) -> Vec<u32> {
        gpu.read(&self.out, (self.m * self.n) as usize).iter().map(|f| f.to_bits()).collect()
    }
}

#[test]
fn a_lane_changes_where_work_runs_not_what_it_computes() {
    let Some(gpu) = urgent_cuda() else { return };
    let normal = Operands::new(&gpu, 1, 5120, 1024, 7);
    normal.submit(&gpu, K_GEMV);
    let want = normal.bits(&gpu);
    // The same operands, made in the normal lane, multiplied in the urgent one.
    // Inside a scratch scope and a pass, as a decode step records itself: both
    // must close on the lane they opened in.
    let got = lane::urgent(|| {
        gpu.write(&normal.out, &vec![0u32; 1024]);
        {
            let _scratch = gpu.scratch_scope_in("urgent-lane-test");
            let _pass = gpu.pass_scope();
            normal.submit(&gpu, K_GEMV);
        }
        normal.bits(&gpu)
    });
    assert_eq!(got, want, "the urgent lane computed a different product");
    // And operands made in the urgent lane, read back in the normal one.
    let urgent = lane::urgent(|| {
        let o = Operands::new(&gpu, 1, 5120, 1024, 7);
        o.submit(&gpu, K_GEMV);
        o
    });
    assert_eq!(urgent.bits(&gpu), want, "a buffer the urgent lane wrote reads back the same in the normal lane");
}

#[test]
fn an_urgent_read_does_not_wait_for_the_normal_lane() {
    let Some(gpu) = urgent_cuda() else { return };
    let bulk = Operands::new(&gpu, 64, 5120, 17408, 3);
    let decode = Operands::new(&gpu, 1, 5120, 1024, 4);
    gpu.poll_wait();
    // Queue far more normal-lane work than the urgent read below takes.
    let t = Instant::now();
    for _ in 0..40 {
        bulk.submit(&gpu, K_GEMM);
    }
    assert!(!gpu.is_idle(), "forty prefill GEMMs were finished as soon as they were queued");
    let urgent_ms = lane::urgent(|| {
        let t = Instant::now();
        decode.submit(&gpu, K_GEMV);
        let _ = decode.bits(&gpu);
        t.elapsed().as_secs_f64() * 1e3
    });
    let still_busy = !gpu.is_idle();
    gpu.poll_wait();
    let bulk_ms = t.elapsed().as_secs_f64() * 1e3;
    assert!(still_busy, "the normal lane finished its {bulk_ms:.1} ms of work before the urgent read returned ({urgent_ms:.2} ms)");
    assert!(gpu.is_idle(), "a drained lane is idle");
}

/// A decode step whose captured graph was invalidated (a buffer it named was
/// freed) retires that graph on the urgent lane's next submission. Retiring must
/// not call anything that waits for the whole device: the driver's page-locked
/// free does, and with a prefill queued on the normal lane it held the decode
/// loop for the rest of that queue (knowledge 374).
#[test]
fn retiring_an_invalidated_graph_in_the_urgent_lane_does_not_wait_for_the_normal_lane() {
    let Some(gpu) = urgent_cuda() else { return };
    let bulk = Operands::new(&gpu, 64, 5120, 17408, 3);
    let mut decode = lane::urgent(|| Operands::new(&gpu, 1, 5120, 1024, 4));
    // Three identical submissions: the second is captured, the third replays it.
    lane::urgent(|| {
        for _ in 0..3 {
            decode.submit_clearing(&gpu, K_GEMV);
            let _ = decode.bits(&gpu);
        }
        // Freeing the buffer the graph clears makes the graph stale.
        decode.scratch = gpu.storage(1);
    });
    for _ in 0..40 {
        bulk.submit(&gpu, K_GEMM);
    }
    assert!(!gpu.is_idle(), "forty prefill GEMMs were finished as soon as they were queued");
    let urgent_ms = lane::urgent(|| {
        let t = Instant::now();
        decode.submit_clearing(&gpu, K_GEMV);
        let _ = decode.bits(&gpu);
        t.elapsed().as_secs_f64() * 1e3
    });
    let still_busy = !gpu.is_idle();
    gpu.poll_wait();
    assert!(still_busy, "the urgent lane's first step after an invalidation waited for the normal lane's whole queue ({urgent_ms:.1} ms)");
}

/// A decode GEMV's latency: alone, behind a running prefill GEMM in the same
/// lane, and beside it in the urgent lane - one host thread, as a scheduler
/// that keeps the prefill in flight between decode steps issues them.
#[test]
#[ignore]
fn decode_latency_beside_a_prefill() {
    let Some(gpu) = urgent_cuda() else { return };
    let bulk = Operands::new(&gpu, 64, 5120, 17408, 3);
    let decode = Operands::new(&gpu, 1, 5120, 17408, 4);
    let time = |urgent: bool, busy: bool| {
        let mut ts = Vec::new();
        for _ in 0..40 {
            if busy {
                for _ in 0..4 {
                    bulk.submit(&gpu, K_GEMM);
                }
            }
            let t = Instant::now();
            let step = || {
                decode.submit(&gpu, K_GEMV);
                let _ = decode.bits(&gpu);
            };
            if urgent {
                lane::urgent(step);
            } else {
                step();
            }
            ts.push(t.elapsed().as_secs_f64() * 1e3);
            gpu.poll_wait();
        }
        ts.sort_by(f64::total_cmp);
        ts[ts.len() / 2]
    };
    println!("decode GEMV 1x5120x17408, median ms: alone {:.3}, behind a prefill {:.3}, urgent beside a prefill {:.3}", time(false, false), time(false, true), time(true, true));
}
