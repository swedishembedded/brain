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
//! failure rate, which a pure red herring would not do), but it MITIGATES
//! rather than fixes -- this test is `#[ignore]`d rather than asserted, and
//! `crates/cli/src/resident_deepseekocr.rs`'s CPU pin stays. Finding the
//! actual missing barrier (likely in how `backend-wgpu`'s single-pass flush
//! or the underlying driver serializes many large dispatches queued in one
//! submit) is unfinished work, not delegated to a workaround that happens to
//! shift the odds.
//!
//! This fixture mirrors `SamViTConfig::deepseek_ocr()` (real `d_model=768`,
//! 12 heads, window 14, `attn_chunk=256`) at the real 1024x1024 image / 64x64
//! grid, varying only `n_layers` -- an inference build (`train=false`, no
//! backward scratch) to keep this in the checkpoint-free fast lane.

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
/// `#[ignore]`d: per this module's doc comment, it is intermittent (measured
/// 2/5 clean) rather than reliably green, so it must not gate unrelated
/// changes to this crate while the underlying race is still open.
#[test]
#[ignore = "known-flaky: reproduces a wgpu race roughly half the time - see this module's doc comment"]
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
/// WINDOWED blocks (3, 4) after block 1. One run of this found
/// `block_out(1)` (computed entirely by windowed attention, before block 2's
/// global attention ever runs) disagreeing between the 2-layer and 5-layer
/// builds by max|delta| 1.209e2, while `patch_tokens`/`block_out(0)` moved by
/// only ~1e-6 (float round-off). No kernel in blocks 0/1's own path needs 2D
/// grid tiling at this shape, so if this IS a distinct race it is not the
/// `attn_relpos_add` one -- but given the test above's measured 2/5 flake
/// rate on the FIRST race, one non-repeated run finding a different-looking
/// symptom is not enough to call this a confirmed second defect; it could be
/// the same race with a different observable when more total dispatches are
/// queued. Left here, `#[ignore]`d, as a reproduction to re-run (several
/// times, per the lesson above) rather than a settled finding.
#[test]
#[ignore = "known-failing: a second wgpu defect, not yet root-caused - see this test's doc comment"]
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
