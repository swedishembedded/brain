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
//! ## Why this needs real production DIMS, not just a real block COUNT
//!
//! A first version of this test kept every dimension tiny (`d_model=96`) and
//! only sized the global span so ONE dispatch (`attn_relpos_add`, whose thread
//! count is `heads*qn*kn`) crossed `backend_api::MAX_GROUPS_PER_DIM`'s 2D
//! grid-tiling threshold -- on the theory that the tiled-dispatch path
//! (`gid.y*(nwg.x*64u)+gid.x`, needed once a dispatch exceeds 65 535
//! workgroups) was the wgpu-specific mechanism, since `backend-cpu` has no
//! such per-dimension dispatch limit and is proven correct at the same block
//! counts. That reproduction was clean on BOTH backends -- ruling out
//! dispatch-grid tiling in isolation as sufficient, and matching the roadmap's
//! own finding that the defect needs "a few blocks' worth of ~500 MB-per-block
//! buffers", i.e. total DEVICE ALLOCATION VOLUME, not merely one large
//! dispatch. So this fixture mirrors `SamViTConfig::deepseek_ocr()` (real
//! `d_model=768`, 12 heads, window 14, `attn_chunk=256`) at the real
//! 1024x1024 image / 64x64 grid, varying only `n_layers` -- an inference build
//! (`train=false`, no backward scratch) to keep this in the checkpoint-free
//! fast lane at ~250-340 MB per block rather than the ~3x-larger training
//! build.

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
fn assert_prefix_unaffected_by_a_later_block(backend_name: &str, gpu_2layer: Gpu, gpu_3layer: Gpu) {
    let (patch_a, b0_a, b1_a) = taps(gpu_2layer, 2);
    let (patch_b, b0_b, b1_b) = taps(gpu_3layer, 3);
    let worst = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let (dp, d0, d1) = (worst(&patch_a, &patch_b), worst(&b0_a, &b0_b), worst(&b1_a, &b1_b));
    println!("{backend_name}: max|delta| patch_tokens={dp:.3e} block_out(0)={d0:.3e} block_out(1)={d1:.3e}");
    assert_eq!(patch_a, patch_b, "{backend_name}: patch-embed tokens changed when a 3rd block was appended");
    assert_eq!(b0_a, b0_b, "{backend_name}: block 0's output changed when a 3rd block was appended");
    assert_eq!(b1_a, b1_b, "{backend_name}: block 1's output changed when a 3rd block was appended");
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
        Gpu::new_cpu(sam1::PIPELINES),
    );
}

/// The reproduction -- matches `tests/parity.rs`'s real-weight finding (and
/// the CPU-backend pin it documents) at a fraction of the size and time.
#[test]
fn wgpu_backend_block0_is_unaffected_by_a_third_block() {
    if skip() {
        return;
    }
    assert_prefix_unaffected_by_a_later_block(
        "wgpu",
        Gpu::new_wgpu(sam1::PIPELINES),
        Gpu::new_wgpu(sam1::PIPELINES),
    );
}
