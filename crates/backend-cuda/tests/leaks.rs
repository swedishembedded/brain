// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Every owner of a driver object in `backend-cuda` returns everything it took.
//!
//! Swedish Embedded AB implements long-running GPU services for its clients,
//! where a leaked allocation is an outage rather than a nuisance. If your team
//! needs expertise in proving that an accelerator runtime returns every byte
//! and handle it takes, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # What is asserted, and why two measures
//!
//! Each test runs a create/drop workload and then asserts two independent
//! things:
//!
//! 1. [`backend_cuda::live_resources`] is EXACTLY what it was before - the
//!    crate's own count of driver objects, by kind, with no tolerance; and
//! 2. `cuMemGetInfo` free memory has not shrunk beyond a small tolerance - the
//!    driver's own opinion, which catches an owner that forgot to register with
//!    the counters at all. It is a tolerance check only, because other
//!    processes allocate on the same card.
//!
//! The counters are process-global, so every test holds [`serial`] for its
//! whole body: two tests running at once would each see the other's objects.
//!
//! Skip-if-absent, like the rest of this crate's device tests.

use std::sync::{Mutex, MutexGuard};

use backend_api::{Backend as _, BufUsage, DeviceBuffer};
use backend_cuda::exec::Context;
use backend_cuda::{live_resources, CudaBackend, LiveResources};

const KERNELS: &[(&str, &str)] = &[("axpy", kernels::AXPY)];
const AXPY: usize = 0;

/// How much `cuMemGetInfo` free memory may drop across a workload that returned
/// everything. The driver keeps small internal pools of its own (module images,
/// graph bookkeeping) that this process does not control; a real leak in these
/// tests is hundreds of MiB.
const FREE_TOLERANCE: u64 = 48 << 20;

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// A context that stays open for the whole test, so the baseline already
/// contains the primary-context retain, its stream, and the driver's lazily
/// created per-context state.
fn probe() -> Option<Context> {
    match Context::open(0) {
        Ok(c) => Some(c),
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no usable CUDA device: {e}"));
            None
        }
    }
}

struct Baseline {
    live: LiveResources,
    free: u64,
}

fn baseline(probe: &Context) -> Baseline {
    Baseline { live: live_resources(), free: probe.mem_info().expect("cuMemGetInfo").0 }
}

/// Assert everything taken since `base` has been returned.
fn assert_returned(probe: &Context, base: &Baseline, what: &str) {
    assert_eq!(live_resources(), base.live, "{what}: the crate's live-resource counters did not return to baseline");
    // Free memory is shared with other processes; a transient dip from someone
    // else's allocation is retried rather than reported as ours.
    let mut free = 0;
    for _ in 0..20 {
        free = probe.mem_info().expect("cuMemGetInfo").0;
        if free + FREE_TOLERANCE >= base.free {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    panic!(
        "{what}: cuMemGetInfo free memory fell from {} MiB to {} MiB - the driver still holds {} MiB this workload took",
        base.free >> 20,
        free >> 20,
        (base.free - free) >> 20
    );
}

const WORDS: usize = 4 << 20; // 16 MiB per buffer

fn backend() -> Option<CudaBackend> {
    match CudaBackend::try_new(KERNELS) {
        Ok(b) => Some(b),
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no usable CUDA backend: {e}"));
            None
        }
    }
}

/// One submission of `outs.len()` axpy steps, `out += s * inp`.
fn round(b: &dyn backend_api::Backend, outs: &[DeviceBuffer], inp: &DeviceBuffer, n: u32, s: f32) {
    let params = [n, s.to_bits()];
    let steps: Vec<_> = outs.iter().map(|o| b.step(AXPY, &[o, inp], &params, n)).collect();
    b.submit(&[], &steps);
}

/// The smallest kernel that is a kernel: enough to capture and replay.
const TOUCH: &str = r#"extern "C" __global__ void touch(unsigned int* p) { p[threadIdx.x] = threadIdx.x + 1u; }"#;

// ---------------------------------------------------------------------------
// Context-level owners
// ---------------------------------------------------------------------------

#[test]
fn device_allocations_are_returned() {
    let _s = serial();
    let Some(p) = probe() else { return };
    let base = baseline(&p);
    for _ in 0..6 {
        let held: Vec<_> = (0..8).map(|_| p.alloc(32 << 20).expect("alloc")).collect();
        let live = live_resources();
        assert_eq!(live.device_allocs - base.live.device_allocs, 8);
        assert_eq!(live.device_bytes - base.live.device_bytes, 8 * (32 << 20));
        drop(held);
    }
    // A zero-byte request is a real 1-byte allocation and must balance too.
    drop(p.alloc(0).expect("alloc 0"));
    assert_returned(&p, &base, "device allocations");
}

#[test]
fn pinned_blocks_events_and_streams_are_returned() {
    let _s = serial();
    let Some(p) = probe() else { return };
    let base = baseline(&p);
    for _ in 0..20 {
        let pins: Vec<_> = (0..4).map(|_| p.pinned(1 << 20).expect("pinned")).collect();
        let evs: Vec<_> = (0..16).map(|i| if i % 2 == 0 { p.event() } else { p.ordering_event() }.expect("event")).collect();
        let live = live_resources();
        assert_eq!(live.pinned_allocs - base.live.pinned_allocs, 4);
        assert_eq!(live.events - base.live.events, 16);
        drop((pins, evs));
    }
    // Opening and dropping handles creates and destroys a stream and a retain.
    for _ in 0..8 {
        let extra = Context::open(0).expect("second handle");
        assert_eq!(live_resources().streams - base.live.streams, 1);
        drop(extra);
    }
    assert_returned(&p, &base, "pinned blocks, events, streams");
}

#[test]
fn modules_are_unloaded() {
    let _s = serial();
    let Some(p) = probe() else { return };
    let Ok(cubin) = p.cubin(TOUCH, "touch") else {
        brain_testutil::skip_unavailable("no NVRTC");
        return;
    };
    let warm = p.load(&cubin.cubin).expect("load");
    drop(warm);
    let base = baseline(&p);
    for _ in 0..50 {
        let m = p.load(&cubin.cubin).expect("load");
        assert_eq!(live_resources().modules - base.live.modules, 1);
        let _f = m.function("touch").expect("entry point");
    }
    assert_returned(&p, &base, "modules");
}

#[test]
fn captured_graphs_and_their_execs_are_destroyed() {
    let _s = serial();
    let Some(p) = probe() else { return };
    let Ok(module) = p.compile(TOUCH, "touch") else {
        brain_testutil::skip_unavailable("no NVRTC");
        return;
    };
    if p.graphs().is_err() {
        brain_testutil::skip_unavailable("no graph entry points");
        return;
    }
    let f = module.function("touch").expect("entry point");
    let out = p.alloc(1024).expect("alloc");
    let capture_once = || {
        let cap = p.begin_capture().expect("begin capture");
        p.launch(&f, (1, 1, 1), (32, 1, 1), &[&out]).expect("launch");
        let graph = cap.finish().expect("finish");
        let exec = p.instantiate(&graph).expect("instantiate");
        (graph, exec)
    };
    drop(capture_once());
    let base = baseline(&p);
    for _ in 0..30 {
        let (graph, exec) = capture_once();
        p.launch_graph(&exec).expect("replay");
        p.sync().expect("sync");
        let live = live_resources();
        assert_eq!((live.graphs - base.live.graphs, live.graph_execs - base.live.graph_execs), (1, 1));
        // Both orders: the instantiated graph outlives its source and vice versa.
        if live.graphs.is_multiple_of(2) {
            drop((graph, exec));
        } else {
            drop(graph);
            drop(exec);
        }
    }
    assert_returned(&p, &base, "captured graphs");
    drop(out);
}

#[test]
fn an_abandoned_capture_leaves_neither_a_graph_nor_a_capturing_stream() {
    let _s = serial();
    let Some(p) = probe() else { return };
    let Ok(module) = p.compile(TOUCH, "touch") else {
        brain_testutil::skip_unavailable("no NVRTC");
        return;
    };
    if p.graphs().is_err() {
        brain_testutil::skip_unavailable("no graph entry points");
        return;
    }
    let f = module.function("touch").expect("entry point");
    let out = p.alloc(1024).expect("alloc");
    let base = baseline(&p);
    for _ in 0..20 {
        let cap = p.begin_capture().expect("begin capture");
        p.launch(&f, (1, 1, 1), (32, 1, 1), &[&out]).expect("launch");
        drop(cap); // an error path that never finishes
    }
    // The stream must work again: a capture left open swallows every operation.
    p.launch(&f, (1, 1, 1), (32, 1, 1), &[&out]).expect("launch after abandoned captures");
    p.sync().expect("sync after abandoned captures");
    assert_returned(&p, &base, "abandoned captures");
    drop(out);
}

#[test]
fn failed_operations_do_not_leave_anything_behind() {
    let _s = serial();
    let Some(p) = probe() else { return };
    let base = baseline(&p);
    for _ in 0..10 {
        // An allocation the card cannot satisfy.
        assert!(p.alloc(usize::MAX / 2).is_err());
        assert!(p.pinned(usize::MAX / 16).is_err());
        // Bounds errors on a real allocation.
        let m = p.alloc(64).expect("alloc");
        assert!(p.upload(&m, &[0u8; 65]).is_err());
        assert!(p.upload_at(&m, 60, &[0u8; 8]).is_err());
        assert!(p.download(&m, &mut [0u8; 65]).is_err());
        // A device that does not exist.
        assert!(Context::open(9999).is_err());
    }
    assert_returned(&p, &base, "error paths");
}

// ---------------------------------------------------------------------------
// Backend-level owners
// ---------------------------------------------------------------------------

#[test]
fn buffers_created_through_the_backend_are_returned() {
    let _s = serial();
    let Some(p) = probe() else { return };
    let b = backend().expect("a context opened, so a backend must");
    let warm = b.storage(WORDS as u64);
    drop(warm);
    let base = baseline(&p);
    for _ in 0..5 {
        let held = vec![
            b.storage(WORDS as u64),
            b.storage_init("init", &vec![1.0f32; WORDS]),
            b.buffer("buf", (WORDS * 4) as u64, BufUsage::STORAGE),
            b.uniform_dynamic(WORDS),
        ];
        let _ = b.read(&held[1], 4);
        drop(held);
    }
    assert_returned(&p, &base, "backend buffers");
}

#[test]
fn a_backend_that_captured_graphs_returns_everything_when_dropped() {
    let _s = serial();
    let Some(p) = probe() else { return };
    drop(backend()); // warm the driver's lazy state
    let base = baseline(&p);
    for _ in 0..4 {
        let b = backend().expect("backend");
        let inp = b.storage_init("inp", &vec![1.0f32; WORDS]);
        let outs: Vec<_> = (0..8).map(|_| b.storage(WORDS as u64)).collect();
        for s in 0..6 {
            round(&b, &outs, &inp, 1024, 0.5 + s as f32);
        }
        b.poll_wait();
        assert!(b.launch_stats().graph_captures > 0, "no graph formed, so this proves nothing about graphs");
        // Dropped with buffers still alive, in the order a model drops them:
        // the device handle first, its buffers after.
        drop(b);
        drop((inp, outs));
    }
    assert_returned(&p, &base, "backend with captured graphs");
}

#[test]
fn staging_for_unsynchronised_submissions_is_bounded_and_returned() {
    let _s = serial();
    let Some(p) = probe() else { return };
    drop(backend());
    let base = baseline(&p);
    let b = backend().expect("backend").with_graph_capture(false);
    let inp = b.storage_init("inp", &[1.0; 256]);
    let out = b.storage(256);
    // Every submission has new parameters and nothing ever reads: the eager
    // path stages each one in page-locked memory. 30k steps is far more than a
    // decode token and must neither grow without bound nor allocate per step.
    let mut peak = 0;
    for i in 0..30_000u32 {
        round(&b, std::slice::from_ref(&out), &inp, 256, i as f32);
        if i % 1000 == 0 {
            peak = peak.max(live_resources().pinned_bytes - base.live.pinned_bytes);
        }
    }
    b.poll_wait();
    peak = peak.max(live_resources().pinned_bytes - base.live.pinned_bytes);
    assert!(peak <= 32 << 20, "eager staging grew to {} MiB of page-locked memory", peak >> 20);
    let _ = b.read(&out, 1);
    drop((inp, out, b));
    assert_returned(&p, &base, "eager staging");
}

#[test]
fn the_eager_path_reuses_staging_across_synchronised_steps() {
    let _s = serial();
    let Some(p) = probe() else { return };
    let b = backend().expect("backend").with_graph_capture(false);
    let inp = b.storage_init("inp", &[1.0; 256]);
    let outs: Vec<_> = (0..32).map(|_| b.storage(256)).collect();
    // A decode loop: many steps, then a read that drains.
    round(&b, &outs, &inp, 256, 1.0);
    b.poll_wait();
    let steady = live_resources().pinned_allocs;
    for i in 0..200 {
        round(&b, &outs, &inp, 256, i as f32);
        b.poll_wait();
    }
    assert_eq!(live_resources().pinned_allocs, steady, "a synchronised decode loop allocated page-locked memory after warm-up");
    drop((inp, outs, b, p));
}

#[test]
fn shared_handles_return_everything_in_any_drop_order() {
    let _s = serial();
    let Some(p) = probe() else { return };
    drop(backend());
    let base = baseline(&p);
    for order in 0..3 {
        let a = backend().expect("backend");
        let c = a.share().expect("share");
        let inp = a.storage_init("inp", &vec![1.0f32; WORDS]);
        let outs: Vec<_> = (0..4).map(|_| c.storage(WORDS as u64)).collect();
        for s in 0..5 {
            round(c.as_ref(), &outs, &inp, 1024, s as f32);
            round(&a, &outs, &inp, 1024, s as f32);
        }
        c.poll_wait();
        a.poll_wait();
        match order {
            0 => {
                drop(a);
                drop(c);
                drop((inp, outs));
            }
            1 => {
                drop(c);
                drop((inp, outs));
                drop(a);
            }
            _ => {
                drop((inp, outs));
                drop(a);
                drop(c);
            }
        }
    }
    assert_returned(&p, &base, "shared handles");
}

#[test]
fn kernel_timing_events_are_returned() {
    let _s = serial();
    let Some(p) = probe() else { return };
    drop(backend());
    let base = baseline(&p);
    {
        let b = backend().expect("backend");
        assert!(b.set_kernel_timing(true));
        let inp = b.storage_init("inp", &[1.0; 256]);
        let out = b.storage(256);
        for i in 0..40 {
            round(&b, std::slice::from_ref(&out), &inp, 256, i as f32);
        }
        b.poll_wait();
        assert!(b.kernel_times().is_some_and(|t| !t.is_empty()));
        b.set_kernel_timing(false);
    }
    assert_returned(&p, &base, "kernel timing");
}

#[test]
fn registered_native_kernels_are_unloaded_with_the_backend() {
    let _s = serial();
    let Some(p) = probe() else { return };
    let spec = backend_api::NativeSpec::Cuda {
        src: TOUCH,
        entry: "touch",
        block_dim: 32,
        bindings: &[backend_api::BindKind::StorageReadWrite],
        shared_bytes: 0,
    };
    {
        let warm = backend().expect("backend");
        if warm.register_native(&spec).is_none() {
            brain_testutil::skip_unavailable("no NVRTC");
            return;
        }
    }
    let base = baseline(&p);
    for _ in 0..10 {
        let b = backend().expect("backend");
        let sibling = b.share().expect("share");
        assert!(b.register_native(&spec).is_some());
        assert_eq!(live_resources().modules - base.live.modules, 1);
        drop(b);
        // The module is shared with the sibling and must outlive the handle
        // that compiled it...
        assert_eq!(live_resources().modules - base.live.modules, 1);
        drop(sibling);
    }
    // ...and be unloaded with the last one.
    assert_returned(&p, &base, "native kernel modules");
}
