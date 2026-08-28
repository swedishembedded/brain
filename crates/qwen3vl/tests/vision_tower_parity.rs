// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What the Qwen3-VL vision tower is allowed to change when its dispatch does.
//!
//! Two claims are made about the tower's kernel selection, and each gets the
//! gate its claim deserves:
//!
//! * **On the GPU, the register-tiled GEMM is bit-identical to the row-blocked
//!   one.** `matmul_reg3` tiles the OUTPUT, never the contraction, so every
//!   output element is still one thread summing `k` in order. That is a claim
//!   of EXACT equality, so it is gated at `max|delta| == 0` - anything looser
//!   would pass a kernel that quietly reassociated. It is gated on the GPU
//!   and NOT on the CPU JIT, because there the two names are two different
//!   implementations rather than two schedules of one: `backend-cpu` routes
//!   every `matmul*_reg*` name to its native AVX2 GEMM and has no fast path
//!   for `matmul_rows` at all, so the reference side runs Cranelift-compiled
//!   WGSL. That asymmetry is the point - it is why the swap is a large CPU win
//!   as well as a GPU one - but it means CPU equality is numeric, and this
//!   file says which is which instead of asserting the stronger claim
//!   everywhere and quietly loosening the threshold until it passes.
//! * **The fused flash attention computes the same attention as the
//!   scores/softmax/apply trio.** This one genuinely reassociates - the online
//!   softmax rescales its running maximum as it walks the key tiles - so no
//!   exact claim is made or gated. It is held to the same two-sided numeric
//!   bar as the cross-backend comparison, on the same fixture, so the two are
//!   directly comparable.
//! * **The tower computes the same function on both backends.** Two different
//!   GEMM implementations on two different devices cannot be bit-equal, so
//!   this is gated numerically - on **cosine AND rel_l2**, never cosine alone,
//!   because cosine is blind to a uniform scale and this repo has four
//!   separate components where a mutation scored 0.99999+ and was caught only
//!   by the magnitude term.
//!
//! Both gates are **mutation-verified in the test file itself**: a gate nobody
//! has watched fail is a hypothesis, so each one is re-run against a
//! deliberately perturbed weight set and REQUIRED to reject it.
//!
//! The config is small but structurally real: two blocks, a DeepStack tap, a
//! patch grid past the 128-row tile threshold (so the tiled GEMM is actually
//! the kernel under test rather than the fallback), and the same
//! `VisionEncoder` + `PatchMerger` pair the model itself drives.

use std::collections::HashMap;

use gpu_core::Gpu;
use qwen3vl::config::VisionConfig;
use qwen3vl::encoder::{vision_pipelines, PatchMerger, VisionEncoder, BLOCK_LEAVES};

/// Two blocks, one DeepStack tap, hidden 128 and a 16x16 patch grid: 256 rows
/// and 128-wide outputs, so `m >= 128 && n >= 128` holds and the register-tiled
/// GEMM is the kernel this file actually exercises.
fn cfg() -> VisionConfig {
    VisionConfig {
        depth: 2,
        hidden: 128,
        num_heads: 4,
        intermediate: 256,
        patch_size: 4,
        temporal_patch_size: 2,
        spatial_merge_size: 2,
        num_position_embeddings: 64,
        out_hidden_size: 128,
        in_channels: 3,
        deepstack_indexes: vec![0],
    }
}

const GRID: u32 = 16;

fn weights(v: &VisionConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = data::rng::Rng::new(seed);
    let mut fill = |n: usize| -> Vec<f32> { (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect() };
    let (c, pv, mlp) = (v.hidden as usize, v.patch_vec_dim() as usize, v.intermediate as usize);
    let mut w = HashMap::new();
    w.insert("patch_embed.weight".into(), fill(c * pv));
    w.insert("patch_embed.bias".into(), fill(c));
    w.insert("pos_embed".into(), fill(v.num_position_embeddings as usize * c));
    for b in 0..v.depth {
        for leaf in BLOCK_LEAVES {
            let n = match *leaf {
                "qkv.weight" => 3 * c * c,
                "qkv.bias" => 3 * c,
                "proj.weight" => c * c,
                "fc1.weight" => mlp * c,
                "fc1.bias" => mlp,
                "fc2.weight" => c * mlp,
                _ => c,
            };
            w.insert(format!("blocks.{b}.{leaf}"), fill(n));
        }
    }
    w
}

/// `postshuffle` selects the DeepStack merger's shape: its LayerNorm runs over
/// the SHUFFLED `hidden * merge^2` width, where the main merger's runs over
/// `hidden` per patch. Getting that wrong is not a small error - it reads past
/// the norm buffer - so the two fixtures are built from one function that
/// takes the flag, exactly as `PatchMerger::new` does.
fn merger_weights(v: &VisionConfig, seed: u64, postshuffle: bool) -> HashMap<String, Vec<f32>> {
    let mut rng = data::rng::Rng::new(seed);
    let mut fill = |n: usize| -> Vec<f32> { (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect() };
    let merged = (v.hidden * v.spatial_merge_size * v.spatial_merge_size) as usize;
    let out = v.out_hidden_size as usize;
    let ln = if postshuffle { merged } else { v.hidden as usize };
    HashMap::from([
        ("ln.weight".to_string(), vec![1.0; ln]),
        ("ln.bias".to_string(), fill(ln)),
        ("fc1.weight".to_string(), fill(merged * merged)),
        ("fc1.bias".to_string(), fill(merged)),
        ("fc2.weight".to_string(), fill(out * merged)),
        ("fc2.bias".to_string(), fill(out)),
    ])
}

fn pixels(v: &VisionConfig) -> Vec<f32> {
    let n = (GRID * GRID * v.patch_vec_dim()) as usize;
    (0..n).map(|i| ((i % 251) as f32 / 251.0) - 0.5).collect()
}

/// The tower + main merger, end to end, on `gpu`.
fn tower(gpu: &Gpu, v: &VisionConfig, vw: &HashMap<String, Vec<f32>>) -> Vec<f32> {
    let enc = VisionEncoder::new(gpu, v.clone(), vw);
    let (feats, taps) = enc.encode_with_taps(gpu, GRID, GRID, &pixels(v), &v.deepstack_indexes);
    assert_eq!(taps.len(), 1, "the DeepStack tap must actually be taken");
    let merger = PatchMerger::new(gpu, &merger_weights(v, 5, false), v.hidden, v.spatial_merge_size, v.out_hidden_size, false);
    let mut out = merger.merge(gpu, &feats, GRID * GRID);
    // Fold the tap in so a change confined to it cannot pass unnoticed.
    let ds = PatchMerger::new(gpu, &merger_weights(v, 9, true), v.hidden, v.spatial_merge_size, v.out_hidden_size, true);
    out.extend(ds.merge(gpu, &taps[0], GRID * GRID));
    out
}

/// `vision_pipelines()` with the register-tiled GEMM taken back out - the
/// row-blocked/naive reference the tower dispatched before it was registered.
/// Removal keeps every OTHER slot at its hand-numbered index, which is what
/// makes this a valid A/B rather than a different model.
fn pipelines_without_reg3() -> Vec<(&'static str, &'static str)> {
    let v: Vec<_> = vision_pipelines().iter().copied().filter(|(n, _)| *n != "matmul_reg3").collect();
    assert_eq!(v.len() + 1, vision_pipelines().len(), "matmul_reg3 must be registered to be removable");
    v
}

/// `vision_pipelines()` with the flash family taken back out, leaving the
/// chunked scores/softmax/apply trio that `model::vit::flash_ids` falls back
/// to. Dropping `flash_attn_bidir` alone would be enough (it is the required
/// slot), but all four go so the A/B cannot be confused by a half-registered
/// set.
fn pipelines_without_flash() -> Vec<(&'static str, &'static str)> {
    let v: Vec<_> = vision_pipelines().iter().copied().filter(|(n, _)| !n.starts_with("flash_attn_")).collect();
    assert_eq!(v.len() + 4, vision_pipelines().len(), "the flash family must be registered to be removable");
    v
}

/// The two-sided threshold every NUMERIC comparison in this file uses. Cosine
/// alone is not a gate: the mutation below scores 0.999999998 against it and
/// is rejected only by the magnitude term.
const COS_FLOOR: f64 = 0.9999999;
const REL_L2_CEIL: f64 = 1e-5;

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// Perturb one weight by a relative amount small enough that a cosine-only
/// gate waves it through - the mutation this file exists to prove is caught.
fn mutate(w: &HashMap<String, Vec<f32>>, key: &str) -> HashMap<String, Vec<f32>> {
    let mut m = w.clone();
    let v = m.get_mut(key).unwrap_or_else(|| panic!("no such weight: {key}"));
    for x in v.iter_mut() {
        *x *= 1.0005;
    }
    m
}

/// On the GPU, swapping the row-blocked reference GEMM for the register-tiled
/// one must not change a single bit of the tower's output. The tiling is over
/// the output, so each element is still one in-order `k` reduction; if that
/// ever stops being true this is where it shows.
#[test]
fn the_register_tiled_gemm_is_bit_identical_on_the_gpu() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        brain_testutil::skip("MOE_SKIP_GPU_TESTS set");
        return;
    }
    let v = cfg();
    let vw = weights(&v, 3);

    let fast = tower(&Gpu::new_wgpu(vision_pipelines()), &v, &vw);
    let reference = tower(&Gpu::new_wgpu(&pipelines_without_reg3()), &v, &vw);
    assert!(fast.iter().all(|x| x.is_finite()), "tower produced non-finite output");
    assert_eq!(max_abs(&fast, &reference), 0.0, "the tiled GEMM changed the tower's output");

    // Mutation-verify: the SAME comparison must reject a perturbed tower, or
    // the equality above is proving nothing.
    let mutated = tower(&Gpu::new_wgpu(vision_pipelines()), &v, &mutate(&vw, "blocks.1.fc1.weight"));
    assert!(max_abs(&fast, &mutated) > 0.0, "the bit-equality gate cannot fail, so it is not a gate");
}

/// On the CPU JIT the same swap changes IMPLEMENTATION, not just schedule (see
/// this file's header), so the claim there is numeric - and held to the same
/// two-sided gate as the cross-backend comparison, never to cosine alone.
#[test]
fn the_register_tiled_gemm_agrees_with_the_reference_on_the_cpu_jit() {
    let v = cfg();
    let vw = weights(&v, 3);
    let fast = tower(&Gpu::new_cpu(vision_pipelines()), &v, &vw);
    let reference = tower(&Gpu::new_cpu(&pipelines_without_reg3()), &v, &vw);
    assert!(fast.iter().all(|x| x.is_finite()), "tower produced non-finite output");

    let (cos, max) = brain_testutil::parity::compare(&fast, &reference);
    let rel = brain_testutil::parity::rel_l2(&fast, &reference);
    println!("cpu reg3 vs matmul_rows: cosine={cos:.9} rel_l2={rel:.3e} max_abs={max:.3e}");
    assert!(cos >= COS_FLOOR, "cosine {cos:.9} below floor {COS_FLOOR}");
    assert!(rel <= REL_L2_CEIL, "rel_l2 {rel:.3e} above ceiling {REL_L2_CEIL:.0e}");

    let mutated = tower(&Gpu::new_cpu(vision_pipelines()), &v, &mutate(&vw, "blocks.1.fc1.weight"));
    let mrel = brain_testutil::parity::rel_l2(&mutated, &reference);
    assert!(mrel > REL_L2_CEIL, "the gate cannot fail, so it is not a gate: rel_l2 {mrel:.3e}");
}

/// The tower must compute the same function wherever it is placed. This is the
/// gate that lets the vision half follow the decoder onto a card instead of
/// being pinned to the CPU JIT: two backends, so numeric rather than exact,
/// and on BOTH cosine and rel_l2.
#[test]
fn the_tower_agrees_between_the_cpu_jit_and_the_gpu() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        brain_testutil::skip("MOE_SKIP_GPU_TESTS set");
        return;
    }
    let v = cfg();
    let vw = weights(&v, 3);
    let want = tower(&Gpu::new_cpu(vision_pipelines()), &v, &vw);
    let got = tower(&Gpu::new_wgpu(vision_pipelines()), &v, &vw);

    let (cos, max) = brain_testutil::parity::compare(&got, &want);
    let rel = brain_testutil::parity::rel_l2(&got, &want);
    println!("qwen3vl tower cpu vs gpu: cosine={cos:.9} rel_l2={rel:.3e} max_abs={max:.3e}");
    assert!(cos >= COS_FLOOR, "cosine {cos:.9} below floor {COS_FLOOR}");
    assert!(rel <= REL_L2_CEIL, "rel_l2 {rel:.3e} above ceiling {REL_L2_CEIL:.0e}");

    // Mutation-verify BOTH halves of the gate on the same fixture: a 5e-4
    // relative scale on one weight is exactly the size of defect a
    // cosine-only gate misses.
    let mutated = tower(&Gpu::new_wgpu(vision_pipelines()), &v, &mutate(&vw, "blocks.1.fc2.weight"));
    let (mcos, _) = brain_testutil::parity::compare(&mutated, &want);
    let mrel = brain_testutil::parity::rel_l2(&mutated, &want);
    println!("mutated: cosine={mcos:.9} rel_l2={mrel:.3e}");
    assert!(mcos < COS_FLOOR || mrel > REL_L2_CEIL, "the parity gate cannot fail, so it is not a gate");
    assert!(mrel > REL_L2_CEIL, "rel_l2 must be the term that catches a small uniform scale: {mrel:.3e}");
}

/// The fused flash dispatch must compute the attention the chunked trio
/// computes. It reassociates, so this is numeric by nature - and gated on
/// cosine AND rel_l2 at the same thresholds as everything else here, because
/// "it reassociates" is a reason to state a numeric bound, not a licence to
/// skip one.
#[test]
fn the_fused_flash_attention_matches_the_chunked_trio() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        brain_testutil::skip("MOE_SKIP_GPU_TESTS set");
        return;
    }
    let v = cfg();
    let vw = weights(&v, 3);
    let flash = Gpu::new_wgpu(vision_pipelines());
    assert!(model::vit::flash_ids(&flash).is_some(), "this device cannot run the fused path, so this test proves nothing");

    let got = tower(&flash, &v, &vw);
    let want = tower(&Gpu::new_wgpu(&pipelines_without_flash()), &v, &vw);
    assert!(got.iter().all(|x| x.is_finite()), "the fused path produced non-finite output");

    let (cos, max) = brain_testutil::parity::compare(&got, &want);
    let rel = brain_testutil::parity::rel_l2(&got, &want);
    println!("flash vs chunked trio: cosine={cos:.9} rel_l2={rel:.3e} max_abs={max:.3e}");
    assert!(cos >= COS_FLOOR, "cosine {cos:.9} below floor {COS_FLOOR}");
    assert!(rel <= REL_L2_CEIL, "rel_l2 {rel:.3e} above ceiling {REL_L2_CEIL:.0e}");

    // Mutation-verify against the SAME reference, so the threshold above is
    // demonstrably tight enough to reject a real defect rather than merely
    // loose enough to accept a real kernel.
    let mutated = tower(&flash, &v, &mutate(&vw, "blocks.0.qkv.weight"));
    let mrel = brain_testutil::parity::rel_l2(&mutated, &want);
    let (mcos, _) = brain_testutil::parity::compare(&mutated, &want);
    println!("mutated: cosine={mcos:.9} rel_l2={mrel:.3e}");
    assert!(mrel > REL_L2_CEIL, "the gate cannot fail, so it is not a gate: rel_l2 {mrel:.3e}");
}
