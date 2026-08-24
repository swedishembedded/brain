// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **Checkpoint-free reproduction of the wgpu 3+-block corruption** recorded
//! in `tests/parity.rs`'s module header: at production shape the wgpu
//! backend's per-block buffers go wrong as soon as the SAM tower's graph
//! holds three or more blocks, while `backend-cpu` is exact at the same
//! block counts.
//!
//! ## The invariant this gate checks
//!
//! Block 0's own forward output depends ONLY on the input image and block 0's
//! own weights -- nothing about a LATER block (block 2, appended after it) can
//! legally change it, on any correct backend. So: build two encoders that
//! share an identical PREFIX (same config for blocks 0/1, same seed, so
//! `init_dense`'s sequential RNG draws the same weights for that prefix -- see
//! its doc comment), differing only in whether a third, GLOBAL-attention block
//! exists past it, run both forwards, and diff `patch_tokens()` / `block_out(0)`
//! / `block_out(1)` between them. Equal on a correct backend; this is what the
//! real-weight parity test found NOT equal on `backend-wgpu`.
//!
//! ## This is a RACE, not a deterministic bug -- found by running the gate repeatedly
//!
//! A first hypothesis (narrowed by bisection: disabling just the
//! `attn_relpos_add` dispatch, or shrinking `attn_chunk` so it never needs
//! `backend_api::MAX_GROUPS_PER_DIM`'s 2D grid-tiling path, or
//! `BRAIN_GPU_CHECKED=1`'s bounds-checked shaders, each independently made ONE
//! run clean) was an out-of-bounds device write reachable through a REPEATED
//! 2D-tiled in-place dispatch of that kernel (`SamViTConfig::deepseek_ocr()`'s
//! global blocks chunk `attn_chunk=256` against `T=4096`, so
//! `attn_relpos_add` -- `scores[idx] += rel_h[..] + rel_w[..]` in place --
//! dispatches `heads*qn*kn = 12 582 912` threads, past the 65 535-workgroup
//! limit, 16 times per global block). `attn_relpos_add.wgsl` was rewritten to
//! block `JB=8` keys per invocation, cutting the real shape to 1 572 864
//! threads / 24 576 workgroups -- clear of the tiling threshold at any SAM-1
//! config this repo defines.
//!
//! **That kernel change does NOT reliably fix this test.** Run five times in
//! a row after it landed: FAIL, ok, FAIL, FAIL, ok (2/5 clean). Every FAILING
//! run reported the IDENTICAL delta (`patch_tokens=8.202e0,
//! block_out(0)=2.816e2, block_out(1)=3.894e2`) -- not noise that grows or
//! shrinks run to run, a BINARY outcome: either every tap matches exactly, or
//! every tap is off by one SPECIFIC, repeatable wrong value. That shape --
//! deterministic-if-lost, sometimes-lost -- is the signature of a missing
//! synchronization barrier (a read observing stale pre-write device memory on
//! some schedules and the correct post-write value on others), not of a fixed
//! index formula. `backend-cpu`'s dispatcher has no concept of overlapping
//! device work at all, which is consistent with it never racing.
//!
//! So the kernel rewrite is kept (it removes an unnecessary and genuinely
//! oversized dispatch, and each of the three interventions above changed the
//! failure rate, which a pure red herring would not do), but at the time it
//! landed it MITIGATED rather than fixed the defect -- see the ROOT CAUSE
//! section below for what actually closed it.
//!
//! This fixture mirrors `SamViTConfig::deepseek_ocr()` (real `d_model=768`,
//! 12 heads, window 14, `attn_chunk=256`) at the real 1024x1024 image / 64x64
//! grid, varying only `n_layers` -- an inference build (`train=false`, no
//! backward scratch) to keep this in the checkpoint-free fast lane.
//!
//! ## ROOT CAUSE, FIX, AND CONFIRMATION (this repo's fourth pass on this bug)
//!
//! `crates/backend-vulkan` had already root-caused the identical bug class
//! (commit `b6295e36`, Khronos-validation-clean): the Intel ANV Vulkan driver
//! on this box's Arc iGPU does not reliably honor an in-command-buffer
//! compute-compute pipeline barrier across a non-zero-offset ("sliced")
//! storage buffer binding -- only a queue submit+fence boundary is honored.
//! `backend-wgpu::WgpuBackend::flush_serialized` (commit `04675800`) mirrors
//! that fix: any flush batch on an Intel adapter containing a `step_sliced`
//! dispatch is serialized (one pass + one submit + one `device.poll(Wait)`
//! per dispatch) instead of single-pass flushed. `crates/model/src/
//! block.rs::chunked_bidir_fwd` (the path `sam1`'s global-attention blocks
//! route through) uses `step_sliced` for every query chunk past the first, so
//! this applies directly to the corruption above.
//!
//! That fix landed unable to be confirmed against a live failure in the same
//! session (46 consecutive clean pre-fix baseline trials found no failure to
//! compare against). The follow-up session that closed this out did two
//! things a clean-but-quiet run cannot: (1) ran
//! `crates/sam1/tests/wgpu_real_weight_parity.rs` (full 12-layer, real mmproj
//! checkpoint) FIVE times with the fix in place -- worst cosine
//! `1.0000000000` on every tap, every run, including `patch_tokens`, which a
//! stale doc comment on that file had recorded as diverging at cosine ~0.11
//! pre-fix; (2) reproduced the ORIGINAL bisection's high-contention
//! conditions deliberately rather than just re-running the test: 14 CPU-bound
//! busy-loop processes pushed this 22-core box's load average to ~20-21 (the
//! original bisection's own reported range), FOUR of this file's `#[ignore]`d
//! tests were run concurrently against the same physical GPU (32 total
//! trials across `wgpu_backend_block0_is_unaffected_by_a_third_block`,
//! `..._at_full_twelve_layers`, and `wgpu_backend_block1_is_unaffected_at_
//! five_layers`, half of them with `BRAIN_GPU_CHECKED=1`, each trial a FRESH
//! process per this file's own re-derivation discipline), and every trial
//! passed -- 32/32, zero failures, under conditions that reproduce the
//! original bisection's reported ~40-60% failure rate almost exactly. That
//! combination (real-weight confirmation + a genuine attempt at the
//! documented highest-failure-rate conditions, both clean) is the evidence
//! this investigation's own prior pass said was the bar for calling the fix
//! confirmed rather than merely plausible.
//!
//! **These tests stay `#[ignore]`d** -- not because the race is still
//! suspected, but because they build a real `wgpu` adapter with no
//! GPU-absence self-skip beyond the manual `MOE_SKIP_GPU_TESTS` override, so
//! running them by default risks a hard panic on a GPU-less CI runner. Run
//! explicitly (`cargo test -p brain-sam1 --release -- --ignored`) to
//! re-validate after any future change to `backend-wgpu`'s flush path or
//! `crates/model/src/block.rs`'s sliced dispatches.
//!
//! **The CPU pin this confirmation was chasing is now LIFTED.**
//! `crates/deepseek2ocr::caps::Session::load` builds the vision encoder
//! (SAM + CLIP + glue) on `gpu_core::Gpu::new_wgpu` and only the decoder on
//! `gpu_core::Gpu::new_cpu`, and `crates/cli/src/resident_deepseekocr.rs`
//! declares the resulting two-device footprint to the scheduler
//! (`residency::multi::MultiDeviceCost`) instead of the RAM-only cost it used
//! to report. This test remains the record of WHY that was safe to do.

use gpu_core::Gpu;
use sam1::{init_dense, SamEncoder, SamViTConfig};

/// `SamViTConfig::deepseek_ocr()` truncated to `n_layers` blocks, keeping only
/// the global-attention layers that still exist. Blocks 0/1 are IDENTICAL
/// (same geometry) between every `n_layers`, so `init_dense`'s sequential draw
/// gives them identical weights too (see the module doc).
fn cfg(n_layers: u32) -> SamViTConfig {
    let base = SamViTConfig::deepseek_ocr();
    SamViTConfig {
        n_layers,
        global_attn_layers: base.global_attn_layers.into_iter().filter(|&l| l < n_layers).collect(),
        ..base
    }
}

const SEED: u64 = 7;

/// `(patch_tokens, block0_out, block1_out)`, read back to the host. An
/// INFERENCE build (`train=false`): the corruption in `tests/parity.rs` was
/// found on exactly that build, and it needs no backward scratch.
fn taps(gpu: Gpu, n_layers: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let c = cfg(n_layers);
    let init = init_dense(&c, SEED);
    let enc = SamEncoder::new_on(gpu, c.clone(), &init, SEED, false);
    let obj = enc.forward();
    assert!(obj.is_finite(), "n_layers={n_layers}: forward produced a non-finite objective");
    let rows = (c.rows() * c.d_model) as usize;
    let patch = enc.gpu.read(enc.patch_tokens(), rows);
    let b0 = enc.gpu.read(enc.block_out(0), rows);
    let b1 = enc.gpu.read(enc.block_out(1), rows);
    (patch, b0, b1)
}

/// Assert every tap is EXACTLY equal (not a tolerance -- both runs share a
/// seed and a graph prefix, so a correct backend produces bit-identical
/// floats; any difference at all is the defect this test exists to catch).
fn assert_prefix_unaffected_by_a_later_block(backend_name: &str, gpu_a: Gpu, n_a: u32, gpu_b: Gpu, n_b: u32) {
    let (patch_a, b0_a, b1_a) = taps(gpu_a, n_a);
    let (patch_b, b0_b, b1_b) = taps(gpu_b, n_b);
    let worst = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let (dp, d0, d1) = (worst(&patch_a, &patch_b), worst(&b0_a, &b0_b), worst(&b1_a, &b1_b));
    println!("{backend_name} ({n_a} vs {n_b}): max|delta| patch_tokens={dp:.3e} block_out(0)={d0:.3e} block_out(1)={d1:.3e}");
    assert_eq!(patch_a, patch_b, "{backend_name}: patch-embed tokens changed when going from {n_a} to {n_b} blocks");
    assert_eq!(b0_a, b0_b, "{backend_name}: block 0's output changed when going from {n_a} to {n_b} blocks");
    assert_eq!(b1_a, b1_b, "{backend_name}: block 1's output changed when going from {n_a} to {n_b} blocks");
}

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

/// Control: the CPU backend must hold the invariant. If this fails the
/// fixture itself is broken (e.g. the two configs do not actually share
/// block 0/1's weights), not the wgpu backend.
#[test]
fn cpu_backend_block0_is_unaffected_by_a_third_block() {
    assert_prefix_unaffected_by_a_later_block(
        "cpu",
        Gpu::new_cpu(sam1::PIPELINES),
        2,
        Gpu::new_cpu(sam1::PIPELINES),
        3,
    );
}

/// The reproduction -- matches `tests/parity.rs`'s real-weight finding (and
/// the CPU-backend pin it documents) at a fraction of the size and time.
/// `#[ignore]`d: not for flakiness (this module's doc comment records the
/// root cause as fixed and confirmed under heavy induced contention), but
/// because it builds a real `wgpu` adapter with no GPU-absence self-skip.
#[test]
#[ignore = "needs a real GPU adapter (no auto-skip) - confirmed fixed, see this module's doc comment"]
fn wgpu_backend_block0_is_unaffected_by_a_third_block() {
    if skip() {
        return;
    }
    assert_prefix_unaffected_by_a_later_block(
        "wgpu",
        Gpu::new_wgpu(sam1::PIPELINES),
        2,
        Gpu::new_wgpu(sam1::PIPELINES),
        3,
    );
}

/// A SECOND angle on the same race (or possibly a second race -- see below):
/// at 5 layers `global_attn_layers` filtered to `< 5` keeps only layer 2, the
/// SAME single global block as the 3-layer case above, just with two more
/// WINDOWED blocks (3, 4) after block 1. One early run of this found
/// `block_out(1)` (computed entirely by windowed attention, before block 2's
/// global attention ever runs) disagreeing between the 2-layer and 5-layer
/// builds by max|delta| 1.209e2, while `patch_tokens`/`block_out(0)` moved by
/// only ~1e-6 (float round-off) -- suspected at the time to possibly be a
/// second, distinct race. This module's doc comment records this test
/// passing 8/8 under heavy induced contention with `BRAIN_GPU_CHECKED=1`
/// (the confirmation run's stream D), consistent with it being the SAME
/// sliced-binding race as the other tests here, now fixed, not a second one.
#[test]
#[ignore = "needs a real GPU adapter (no auto-skip) - confirmed fixed, see this module's doc comment"]
fn wgpu_backend_block1_is_unaffected_at_five_layers() {
    if skip() {
        return;
    }
    assert_prefix_unaffected_by_a_later_block(
        "wgpu",
        Gpu::new_wgpu(sam1::PIPELINES),
        2,
        Gpu::new_wgpu(sam1::PIPELINES),
        5,
    );
}

/// Maximum-pressure variant: the FULL `SamViTConfig::deepseek_ocr()` (12
/// layers, all 4 global-attention blocks at `[2, 5, 8, 11]` -- 64
/// `attn_relpos_add`/scores-matmul/apply chunk iterations total, 4x the
/// single-global-block repro above) against the 2-layer prefix. More queued
/// dispatches to the same handful of buffers is a wider window for a
/// scheduling race, so this is the version most likely to catch the defect
/// when the smaller repro happens not to on a given (load-dependent) run.
/// This is exactly the variant the confirmation run's streams A/B repeated
/// 8x each (16 total trials, half with `BRAIN_GPU_CHECKED=1`) under induced
/// ~load-average-20 contention with zero failures - see this module's doc
/// comment.
#[test]
#[ignore = "needs a real GPU adapter (no auto-skip) - confirmed fixed, see this module's doc comment"]
fn wgpu_backend_block0_is_unaffected_at_full_twelve_layers() {
    if skip() {
        return;
    }
    assert_prefix_unaffected_by_a_later_block(
        "wgpu",
        Gpu::new_wgpu(sam1::PIPELINES),
        2,
        Gpu::new_wgpu(sam1::PIPELINES),
        12,
    );
}

/// Repeats [`wgpu_backend_block0_is_unaffected_by_a_third_block`]'s check
/// `N` times in a row. A single clean run proves nothing here -- the
/// measured flake rate on the un-mitigated race was roughly 40-60%, so only
/// many CONSECUTIVE clean runs distinguish "fixed" from "got lucky". The
/// root cause: `backend-wgpu::WgpuBackend::flush_serialized` now
/// auto-serializes any flush batch that both runs on an Intel adapter and
/// contains a `step_sliced` dispatch (see that method's doc comment),
/// mirroring `backend-vulkan`'s already-confirmed Intel ANV sliced-binding
/// barrier workaround. This module's doc comment records the "several
/// independent invocations of the whole binary" bar this comment asks for as
/// now met (32 fresh-process trials, four separate concurrent streams, zero
/// failures) - kept `#[ignore]`d regardless, same GPU-adapter reason as the
/// other tests in this file, not because more re-derivation is still wanted.
#[test]
#[ignore = "needs a real GPU adapter (no auto-skip) - slow (10x device re-init), confirmed fixed"]
fn wgpu_backend_block0_is_unaffected_by_a_third_block_repeated_10x() {
    if skip() {
        return;
    }
    const N: usize = 10;
    for i in 0..N {
        assert_prefix_unaffected_by_a_later_block(
            &format!("wgpu run {}/{N}", i + 1),
            Gpu::new_wgpu(sam1::PIPELINES),
            2,
            Gpu::new_wgpu(sam1::PIPELINES),
            3,
        );
    }
}
