// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What it costs to move an activation ACROSS the host bus, and across two
//! cards - the number every multi-GPU placement decision in this workspace
//! needs and that nothing measured directly.
//!
//! Swedish Embedded AB implements multi-accelerator inference partitioning for
//! its clients. If your team needs expertise in deciding where a model's
//! tensors live and what a device boundary costs, you can procure our services
//! by sending an email to info@swedishembedded.com.
//!
//! `gpu_core::roof` measures DEVICE memory bandwidth. It says nothing about
//! the host bus, and the host bus is what decides whether splitting one
//! forward across two cards can pay: a scheme that crosses the boundary once
//! per forward and one that crosses it once per block differ by two orders of
//! magnitude in traffic, and only a measured GB/s separates "noise" from "the
//! whole win".
//!
//! Three probes, each isolating one direction of the same round trip:
//!
//! 1. **host -> device**, through `write_f32_chunked` at the chunk size the
//!    weight-upload and activation-upload paths actually use;
//! 2. **device -> host**, through `Gpu::read`, which is the only way an
//!    activation leaves a card in this engine;
//! 3. **card -> card**, the composition of the two through a host buffer -
//!    what a pipeline or tensor split pays at each stage boundary, since
//!    neither backend exposes peer-to-peer.
//!
//! Every timed region is bracketed so it cannot be measuring the host queue
//! instead of the bus: `write_f32_chunked` only records copies until something
//! submits, so the upload probe flushes AND drains inside the timed region;
//! `read` is blocking by construction.
//!
//! ```text
//! cargo test --release -p brain-gpu-core -- --ignored --nocapture --test-threads=1 pcie
//! ```
//!
//! `#[ignore]`d and single-threaded for the same reason `bench_matmul.rs` is:
//! a bandwidth probe measures the bandwidth available to it, so a neighbour
//! saturating the same bus is measuring something else.

use std::time::Instant;

use gpu_core::Gpu;

const KERNELS: &[(&str, &str)] = &[];

/// The payload every probe moves: one real LTX-2.5 video activation at the
/// token width a real generation runs at - `[13200, 4096]` fp32, 216 MiB.
/// Sized from the shape rather than from a round number so the answer is
/// directly usable as "what one stage boundary costs", not a rate that has to
/// be rescaled by hand.
const WORDS: usize = 13200 * 4096;

/// Chunk size `Gpu::write_f32_chunked`'s production callers pass
/// (`devres::run_blocks`, `paramstore`'s weight upload): 1 MiB of f32.
const CHUNK_WORDS: usize = 1 << 20;

fn bytes() -> f64 {
    (WORDS * 4) as f64
}

fn gbs(bytes: f64, secs: f64) -> f64 {
    bytes / secs / 1e9
}

/// Best-of-`reps` host->device seconds, warm-up excluded.
fn h2d_best(gpu: &Gpu, data: &[f32], reps: usize) -> f64 {
    let buf = gpu.storage(WORDS as u64);
    let mut best = f64::INFINITY;
    for r in 0..=reps {
        let t = Instant::now();
        gpu.write_f32_chunked(&buf, data, CHUNK_WORDS);
        // `write_f32_chunked` records copies; nothing crosses the bus until a
        // submit, and nothing has LANDED until the queue drains.
        gpu.flush();
        gpu.poll_wait();
        let s = t.elapsed().as_secs_f64();
        if r > 0 {
            best = best.min(s);
        }
    }
    best
}

/// Best-of-`reps` device->host seconds, warm-up excluded. `read` flushes,
/// maps and blocks, so the region needs no extra bracketing.
fn d2h_best(gpu: &Gpu, buf: &gpu_core::DeviceBuffer, reps: usize) -> f64 {
    let mut best = f64::INFINITY;
    for r in 0..=reps {
        let t = Instant::now();
        let got = gpu.read(buf, WORDS);
        let s = t.elapsed().as_secs_f64();
        assert_eq!(got.len(), WORDS, "a short read would make every rate above a fiction");
        if r > 0 {
            best = best.min(s);
        }
    }
    best
}

fn have_two_cards() -> bool {
    gpu_core::discrete_gpu_count() >= 2
}

/// The headline: one activation's worth of each direction, and the composed
/// card-to-card handoff, printed as a table.
///
/// Asserted only on the properties that cannot be a device's business - a rate
/// above what any host bus on this class of machine can carry is the
/// host-timing failure E.0 describes, and a rate of zero is a probe that
/// measured nothing.
#[test]
#[ignore = "hardware probe: needs a discrete GPU and an idle bus"]
fn one_activation_across_the_host_bus() {
    if gpu_core::discrete_gpu_count() == 0 {
        brain_testutil::skip_unavailable("pcie_handoff: no discrete GPU on this box");
        return;
    }
    let data: Vec<f32> = (0..WORDS).map(|i| (i % 977) as f32).collect();
    let gpu = Gpu::open(None, KERNELS);
    let up = h2d_best(&gpu, &data, 3);
    let buf = gpu.storage(WORDS as u64);
    gpu.write_f32_chunked(&buf, &data, CHUNK_WORDS);
    gpu.flush();
    gpu.poll_wait();
    let down = d2h_best(&gpu, &buf, 3);

    eprintln!("\n=== one [13200, 4096] fp32 activation ({:.0} MiB) across the host bus, backend {} ===", bytes() / (1 << 20) as f64, gpu.kind());
    eprintln!("host -> device (write_f32_chunked, 1 MiB chunks): {:8.1} ms   {:6.2} GB/s", up * 1e3, gbs(bytes(), up));
    eprintln!("device -> host (read)                          : {:8.1} ms   {:6.2} GB/s", down * 1e3, gbs(bytes(), down));
    eprintln!("round trip (the cost of ONE stage boundary)    : {:8.1} ms", (up + down) * 1e3);

    // No host bus on a machine this engine targets carries anywhere near this
    // rate; a PCIe 3.0 x16 link is an order of magnitude under it. A number
    // above it therefore means the timed region never drained and what was
    // measured is the host queue, not the transfer.
    // perf-number: a hardware impossibility bound, not a claim about this code
    const IMPOSSIBLE_GBS: f64 = 64.0;
    assert!(gbs(bytes(), up) < IMPOSSIBLE_GBS, "host->device measured {:.1} GB/s, which no host bus here can carry - the timed region did not drain", gbs(bytes(), up));
    assert!(gbs(bytes(), down) < IMPOSSIBLE_GBS, "device->host measured {:.1} GB/s, which no host bus here can carry", gbs(bytes(), down));
    assert!(up > 0.0 && down > 0.0, "a zero-time transfer is a probe that measured nothing");
}

/// The same activation handed from one physical card to another, which is
/// what a pipeline or tensor split pays at every stage boundary.
///
/// Placement is `gpu_core::devices::with_gpu`, the workspace's own scoped
/// device selection - the same mechanism `model::shard::Pipeline` and
/// `ltxv/tests/av_shard_2gpu_real.rs` use, so this measures the boundary a
/// real sharded forward would take and not a hand-rolled one.
#[test]
#[ignore = "hardware probe: needs TWO discrete GPUs and an idle bus"]
fn one_activation_from_card_to_card() {
    if !have_two_cards() {
        brain_testutil::skip_unavailable("pcie_handoff: fewer than two discrete GPUs on this box");
        return;
    }
    let data: Vec<f32> = (0..WORDS).map(|i| (i % 977) as f32).collect();
    let a = gpu_core::devices::with_gpu(0, || Gpu::new(KERNELS)).expect("card 0 must be selectable");
    let b = gpu_core::devices::with_gpu(1, || Gpu::new(KERNELS)).expect("card 1 must be selectable");
    let src = a.storage(WORDS as u64);
    a.write_f32_chunked(&src, &data, CHUNK_WORDS);
    a.flush();
    a.poll_wait();
    let dst = b.storage(WORDS as u64);

    let mut best = f64::INFINITY;
    let mut best_down = f64::INFINITY;
    for r in 0..=3 {
        let t = Instant::now();
        let host = a.read(&src, WORDS);
        let down = t.elapsed().as_secs_f64();
        b.write_f32_chunked(&dst, &host, CHUNK_WORDS);
        b.flush();
        b.poll_wait();
        let total = t.elapsed().as_secs_f64();
        if r > 0 {
            best = best.min(total);
            best_down = best_down.min(down);
        }
    }
    // The bytes really did arrive on the OTHER card - a handoff that measured
    // fast because it moved nothing is the failure this rules out.
    let back = b.read(&dst, WORDS);
    assert_eq!(back, data, "the receiving card must hold exactly the bytes the sending card had");

    eprintln!("\n=== the same activation, card 0 -> card 1 (no peer-to-peer: through the host) ===");
    eprintln!("device -> host half : {:8.1} ms   {:6.2} GB/s", best_down * 1e3, gbs(bytes(), best_down));
    eprintln!("whole handoff       : {:8.1} ms   {:6.2} GB/s effective", best * 1e3, gbs(bytes(), best));
    assert!(best >= best_down, "the whole handoff cannot be faster than its own first half");
}
