// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The per-iteration reclaim contract: a streaming loop must hand a layer's
//! device memory back BEFORE the next layer's is allocated, and the only
//! shapes this workspace offers for writing that loop
//! (`gpu_core::Transient`, `gpu_core::reclaiming`) must make it impossible to
//! get the order wrong.
//!
//! Swedish Embedded AB implements device-memory lifetime management for
//! inference engines for its clients. If your team needs expertise in GPU
//! weight streaming and per-layer residency then you can procure our services
//! by sending an email to info@swedishembedded.com.
//!
//! ## Why this file exists rather than a per-model test
//!
//! The failure it pins - abandoned device buffers piling up across a block
//! stack until a real allocation fails - is invisible at the sizes a model
//! crate's unit tests use. Every one of them runs a tiny synthetic config
//! whose whole weight set fits in a few megabytes, which is exactly why the
//! bug reached production twice on the same day (a 12B text encoder and a 22B
//! DiT) with every model test green. Testing the MECHANISM, at the seam every
//! model shares, is what generalises to model #61.
//!
//! Two knobs make that affordable in a fast lane:
//!
//! * `Gpu::pending_reclaim_bytes` - the accumulation, observable directly,
//!   so the ORDERING property needs kilobytes and no threshold at all.
//! * `BRAIN_GPU_RECLAIM_CEILING` - the backend's own refuse-to-allocate
//!   threshold, lowered so a loop can actually cross it with megabyte
//!   buffers instead of the gigabytes a real checkpoint needs. It resolves
//!   once per process (`backend_api::hardware::reclaim_ceiling`), so each
//!   case that needs it runs in its own child process - the same shape, for
//!   the same reason, as `memory_limit.rs`.

use gpu_core::{reclaiming, DeviceBuffer, Gpu, Transient};

const KERNELS: &[(&str, &str)] = &[("add2", kernels::ADD2)];

/// Words per buffer and buffers per "layer": 4 x 256 KiB = 1 MiB per layer,
/// small enough to be free and big enough that 16 of them cross the lowered
/// ceiling below by 4x.
const WORDS: u64 = 64 * 1024;
const BUFS: usize = 4;
const LAYER_BYTES: u64 = WORDS * 4 * BUFS as u64;

/// The lowered ceiling the child processes run under: four layers' worth, so
/// an unreclaimed loop crosses it on the fifth iteration and a reclaimed one
/// never approaches it.
const CEILING: u64 = 4 * LAYER_BYTES;

/// How many layers each loop below runs. Well past the ceiling when nothing
/// is reclaimed.
const LAYERS: usize = 16;

/// A stand-in for a streamed transformer block: an object that owns this
/// iteration's device buffers and nothing else. What a real one additionally
/// does - upload weights into them, record a dispatch - changes none of the
/// memory behaviour under test.
struct Layer {
    _w: Vec<DeviceBuffer>,
}

impl Layer {
    /// The UNGUARDED constructor, kept only so the negative controls below
    /// can write the loop the wrong way on purpose. Every real per-layer
    /// constructor in this workspace returns a [`Transient`] instead - that
    /// is the fix, and `streamed` is what it replaced.
    fn streamed(gpu: &Gpu) -> Layer {
        Layer { _w: (0..BUFS).map(|_| gpu.storage(WORDS)).collect() }
    }

    /// The GUARDED constructor, shaped exactly like the migrated ones
    /// (`ltxv::block::LtxBlock::on`, `gemma4::block::Gemma4Layer::on`,
    /// `wan::block::WanBlock::on`).
    fn on(gpu: &Gpu) -> Transient<'_, Layer> {
        Transient::on(gpu, Layer::streamed(gpu))
    }
}

/// One device at a time in this binary, child processes included.
///
/// Cargo runs a test binary's tests on parallel threads and several
/// concurrent devices on one card are hostile to this driver (see
/// `backend_wgpu`'s own device-creation lock and `AGENTS.md`'s
/// one-device-per-process rule), so every case here - including the two that
/// only SPAWN a device-opening child - holds this for its whole body.
fn one_device_at_a_time() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(())).lock().unwrap_or_else(|e| e.into_inner())
}

fn gpu() -> Option<Gpu> {
    if gpu_core::discrete_gpu_count() == 0 {
        brain_testutil::skip_unavailable("transient_reclaim: no discrete GPU on this box");
        return None;
    }
    Some(Gpu::open(Some("gpu"), KERNELS))
}

/// **The trap, stated as a test.** A poll placed in the loop body - the fix
/// that looks right, and was written first for real - reclaims nothing,
/// because the object holding the buffers is still alive when it runs.
///
/// This is the property that makes the whole class of bug survive review: the
/// wrong code and the right code differ only in the order of two statements,
/// and the wrong order is the one that reads naturally ("run the layer, then
/// wait for the GPU").
#[test]
fn polling_before_the_layer_drops_reclaims_nothing() {
    let _one = one_device_at_a_time();
    let Some(gpu) = gpu() else { return };
    gpu.poll_wait();
    assert_eq!(gpu.pending_reclaim_bytes(), 0, "a freshly polled device has nothing pending");

    {
        let layer = Layer::streamed(&gpu);
        // The naive fix: poll inside the loop body, while `layer` is live.
        gpu.poll_wait();
        assert_eq!(gpu.pending_reclaim_bytes(), 0, "nothing is pending yet - the poll had nothing to reclaim");
        drop(layer);
    }
    assert!(
        gpu.pending_reclaim_bytes() >= LAYER_BYTES,
        "the layer's {LAYER_BYTES} bytes must still be pending after a poll that ran BEFORE the drop, \
         got {} - if this ever reads 0, the ordering trap this whole abstraction exists for is gone \
         and `gpu_core::transient` should be re-argued from scratch",
        gpu.pending_reclaim_bytes()
    );

    // The same layer, built through the guard: dropped first, polled after.
    {
        let _layer = Layer::on(&gpu);
    }
    assert_eq!(gpu.pending_reclaim_bytes(), 0, "`Transient` must leave nothing pending once it goes out of scope");
}

/// [`reclaiming`] carries the same guarantee for a loop body that allocates
/// device buffers directly, with no single object owning them - the closure
/// boundary is what puts every drop before the poll.
#[test]
fn reclaiming_hands_back_everything_its_body_allocated() {
    let _one = one_device_at_a_time();
    let Some(gpu) = gpu() else { return };
    gpu.poll_wait();

    let kept = reclaiming(&gpu, || {
        let _scratch: Vec<DeviceBuffer> = (0..BUFS).map(|_| gpu.storage(WORDS)).collect();
        // Returned OUT of the body, so it is deliberately still alive after
        // the reclaim - a value the next iteration needs must survive one.
        gpu.storage(WORDS)
    });
    assert_eq!(gpu.pending_reclaim_bytes(), 0, "`reclaiming` must reclaim everything its body dropped");

    drop(kept);
    assert!(gpu.pending_reclaim_bytes() >= WORDS * 4, "a buffer returned out of the body is not reclaimed by it");
}

/// A `Transient` must not restrict what a caller can hold: a residual that
/// spans two layers, or a parity test comparing the same block in two tiers,
/// legitimately keeps two alive at once. Each reclaims on ITS own scope end,
/// and neither reclaims the other's memory early.
#[test]
fn two_transients_may_be_alive_at_once() {
    let _one = one_device_at_a_time();
    let Some(gpu) = gpu() else { return };
    gpu.poll_wait();
    {
        let a = Layer::on(&gpu);
        let b = Layer::on(&gpu);
        assert_eq!(a._w.len(), BUFS);
        assert_eq!(b._w.len(), BUFS);
        assert_eq!(gpu.pending_reclaim_bytes(), 0, "nothing has been dropped yet");
    }
    assert_eq!(gpu.pending_reclaim_bytes(), 0, "both guards reclaimed on their own scope end");
}

// ---- the ceiling itself, in child processes --------------------------------

/// Run one `#[ignore]`d helper below in a child process under a lowered
/// reclaim ceiling, and return `(succeeded, combined output)`.
fn child(helper: &str) -> (bool, String) {
    let exe = std::env::current_exe().expect("current_exe");
    let out = std::process::Command::new(exe)
        .args(["--exact", helper, "--ignored", "--nocapture", "--test-threads=1"])
        .env("BRAIN_GPU_RECLAIM_CEILING", CEILING.to_string())
        .output()
        .expect("spawn subprocess");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    (out.status.success(), text)
}

/// The negative control: the hand-rolled loop this workspace no longer
/// contains must still be REFUSED by the backend, loudly and by name.
///
/// Without this, a future change that relaxed `WgpuBackend::track` (or lost
/// the `TrackedBuffer` accounting behind it) would leave every migrated call
/// site correct and the safety net gone, with nothing red.
#[test]
fn a_loop_that_never_reclaims_is_refused_at_the_ceiling() {
    let _one = one_device_at_a_time();
    if gpu_core::discrete_gpu_count() == 0 {
        brain_testutil::skip_unavailable("transient_reclaim: no discrete GPU on this box");
        return;
    }
    let (ok, out) = child("unreclaimed_loop_helper");
    assert!(!ok, "a loop that never reclaims must not be allowed to run to completion; output:\n{out}");
    assert!(
        out.contains("dropped without an intervening poll_wait"),
        "the refusal must name the real cause, not fail as a generic OOM; output:\n{out}"
    );
    assert!(
        out.contains("gpu_core::Transient"),
        "the refusal must point at the one correct way to write this loop; output:\n{out}"
    );
}

/// ...and the migrated shape must run the identical loop, at the identical
/// sizes, under the identical ceiling, to completion.
#[test]
fn the_same_loop_through_transient_stays_under_the_ceiling() {
    let _one = one_device_at_a_time();
    if gpu_core::discrete_gpu_count() == 0 {
        brain_testutil::skip_unavailable("transient_reclaim: no discrete GPU on this box");
        return;
    }
    let (ok, out) = child("transient_loop_helper");
    assert!(ok, "the guarded loop must complete under a ceiling the unguarded one crosses; output:\n{out}");
    assert!(out.contains("PEAK_PENDING="), "the helper must report what it measured; output:\n{out}");
    let peak: u64 = out
        .split("PEAK_PENDING=")
        .nth(1)
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("no PEAK_PENDING in child output:\n{out}"));
    assert!(
        peak <= LAYER_BYTES,
        "a guarded loop must never hold more than one iteration's {LAYER_BYTES} bytes pending, peaked at {peak}"
    );
}

/// The unguarded loop, run only as [`a_loop_that_never_reclaims_is_refused_at_the_ceiling`]'s
/// child. `#[ignore]`d so the fast lane never runs it directly - it is
/// supposed to abort.
#[test]
#[ignore = "child process of a_loop_that_never_reclaims_is_refused_at_the_ceiling"]
fn unreclaimed_loop_helper() {
    let gpu = Gpu::open(Some("gpu"), KERNELS);
    for _ in 0..LAYERS {
        let layer = Layer::streamed(&gpu);
        std::hint::black_box(&layer);
    }
    panic!("this loop was expected to be refused before it finished");
}

/// The guarded loop, run only as
/// [`the_same_loop_through_transient_stays_under_the_ceiling`]'s child.
#[test]
#[ignore = "child process of the_same_loop_through_transient_stays_under_the_ceiling"]
fn transient_loop_helper() {
    let gpu = Gpu::open(Some("gpu"), KERNELS);
    let mut peak = 0u64;
    for _ in 0..LAYERS {
        let layer = Layer::on(&gpu);
        std::hint::black_box(&*layer);
        peak = peak.max(gpu.pending_reclaim_bytes());
    }
    peak = peak.max(gpu.pending_reclaim_bytes());
    println!("PEAK_PENDING={peak}");
}
