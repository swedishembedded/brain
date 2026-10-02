// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Memory placement: what the driver says about sharing memory with the host,
//! and the explicit allocation policies built on it.
//!
//! Swedish Embedded AB implements memory-placement strategies for coherent
//! CPU-GPU systems for its clients, where the same bytes can live in HBM or in
//! CPU memory and the choice decides the bandwidth. If your team needs
//! expertise in placing tensors across a coherent memory hierarchy, you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! # What is asserted
//!
//! - **facts are asked, not inferred**: the capability report's placement facts
//!   come from driver attributes, and the integrated-GPU flag is untouched by
//!   them (a coherent node is not integrated and still shares);
//! - **policies are honest**: a managed or system allocation is made exactly
//!   when the facts say it can be, and refused with a reason otherwise - never
//!   handed back as device memory;
//! - **placed memory computes**: a kernel reads and writes managed and system
//!   buffers and gets the numbers it would on device memory, before and after a
//!   prefetch each way and under every advice;
//! - **placed memory is returned**: the live counters (managed and system, as
//!   their own kinds) come back to baseline, system memory is not freed under a
//!   running kernel, and the host sees the bytes it wrote through the host
//!   pointer.
//!
//! Skip-if-absent, and tiny: it must not contend with anything resident on the
//! card. Counters are process-global, so every test holds [`serial`].

use std::sync::{Mutex, MutexGuard};

use backend_api::{AllocPolicy, Backend as _, MemAdvice, PrefetchTarget};
use backend_cuda::exec::Context;
use backend_cuda::{live_resources, CudaBackend};

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

const KERNELS: &[(&str, &str)] = &[("axpy", kernels::AXPY)];
const AXPY: usize = 0;
const N: usize = 1 << 16;

fn backend() -> Option<CudaBackend> {
    match CudaBackend::try_new(KERNELS) {
        Ok(b) => Some(b),
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no usable CUDA backend: {e}"));
            None
        }
    }
}

/// `out += 2 * inp` on `out` allocated under `policy`; `inp` is device memory.
fn axpy_on(b: &CudaBackend, policy: AllocPolicy) {
    let inp: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let src = b.storage_init("src", &inp);
    let out = b.alloc_placed("out", (N * 4) as u64, policy).expect("alloc_placed");
    assert!(b.read(&out, N).iter().all(|&v| v == 0.0), "a placed allocation must start zeroed");
    let run = |b: &CudaBackend| {
        b.submit(&[], &[b.step(AXPY, &[&out, &src], &[N as u32, 2.0f32.to_bits()], N as u32)]);
    };
    run(b);
    let got = b.read(&out, N);
    assert!(got.iter().enumerate().all(|(i, &v)| v == 2.0 * i as f32), "{policy:?}: kernel result differs from device memory's");

    if policy != AllocPolicy::Device {
        // Prefetch both ways and every advice leave the numbers alone.
        b.prefetch(&out, 0, (N * 4) as u64, PrefetchTarget::Device).expect("prefetch to device");
        run(b);
        b.prefetch(&out, 0, (N * 4) as u64, PrefetchTarget::Host).expect("prefetch to host");
        for a in [
            MemAdvice::ReadMostly,
            MemAdvice::UnsetReadMostly,
            MemAdvice::PreferDevice,
            MemAdvice::PreferHost,
            MemAdvice::UnsetPreferred,
            MemAdvice::AccessedByDevice,
            MemAdvice::UnsetAccessedByDevice,
        ] {
            b.advise(&out, a).unwrap_or_else(|e| panic!("{policy:?} advice {a:?}: {e}"));
        }
        run(b);
        let got = b.read(&out, N);
        assert!(got.iter().enumerate().all(|(i, &v)| v == 6.0 * i as f32), "{policy:?}: prefetch or advice changed the data (three runs of += 2 * i)");
        assert!(b.prefetch(&out, (N * 4) as u64 - 4, 8, PrefetchTarget::Device).is_err(), "a range past the end must be refused");
    }
}

#[test]
fn placement_facts_are_queried_and_do_not_touch_the_integrated_flag() {
    let _s = serial();
    let Some(b) = backend() else { return };
    let facts = b.placement_facts();
    let caps = b.caps();
    // The integrated flag still means exactly what the driver's integrated
    // attribute says; sharing facts neither set nor clear it.
    assert_eq!(caps.unified_memory, backend_cuda::driver().unwrap().devices().unwrap()[0].integrated);
    if facts.pageable_via_host_page_tables {
        assert!(facts.pageable_memory_access, "host page tables are how pageable access works");
    }
    eprintln!("placement facts: {facts:?} (integrated: {})", caps.unified_memory);
}

#[test]
fn a_policy_is_honoured_exactly_when_the_facts_allow_it() {
    let _s = serial();
    let Some(b) = backend() else { return };
    let facts = b.placement_facts();
    for policy in [AllocPolicy::Device, AllocPolicy::Managed, AllocPolicy::System] {
        let r = b.alloc_placed("probe", 4096, policy);
        assert_eq!(r.is_ok(), facts.supports(policy), "{policy:?}: allocation outcome disagrees with the reported facts: {:?}", r.err());
    }
}

#[test]
fn placed_memory_computes_and_is_returned() {
    let _s = serial();
    let before = live_resources();
    let Some(b) = backend() else { return };
    let facts = b.placement_facts();
    axpy_on(&b, AllocPolicy::Device);
    if facts.supports(AllocPolicy::Managed) {
        axpy_on(&b, AllocPolicy::Managed);
    }
    if facts.supports(AllocPolicy::System) {
        axpy_on(&b, AllocPolicy::System);
    }
    b.poll_wait();
    let now = live_resources();
    assert_eq!((now.managed_allocs, now.system_allocs), (before.managed_allocs, before.system_allocs), "placed memory was not returned");
    assert_eq!((now.managed_bytes, now.system_bytes), (before.managed_bytes, before.system_bytes));
    drop(b);
    assert_eq!(live_resources(), before, "dropping the backend left a driver object behind");
}

#[test]
fn the_host_sees_what_it_wrote_in_placed_memory() {
    let _s = serial();
    let before = live_resources();
    let Ok(ctx) = Context::open(0) else {
        brain_testutil::skip_unavailable("no usable CUDA device");
        return;
    };
    let facts = ctx.device_info().placement;
    let mut blocks = Vec::new();
    if facts.supports(AllocPolicy::Managed) {
        blocks.push(ctx.alloc_managed(1 << 20).expect("managed"));
    }
    if facts.supports(AllocPolicy::System) {
        blocks.push(ctx.alloc_system(1 << 20).expect("system"));
    }
    for m in &blocks {
        let p = m.host_ptr().expect("placed memory has a host address");
        // SAFETY: the block is `1 << 20` bytes the host owns for the test.
        let host = unsafe { std::slice::from_raw_parts_mut(p, 1 << 20) };
        assert!(host.iter().all(|&b| b == 0), "placed memory must start zeroed");
        host.iter_mut().enumerate().for_each(|(i, b)| *b = (i % 251) as u8);
        let mut back = vec![0u8; 1 << 20];
        ctx.download(m, &mut back).expect("download");
        assert_eq!(back, host, "the device copy engine read different bytes than the host wrote");
    }
    let dev = ctx.alloc(4096).expect("device block");
    assert!(dev.host_ptr().is_none(), "device memory has no host address");
    drop(dev);
    drop(blocks);
    drop(ctx);
    assert_eq!(live_resources(), before);
}
