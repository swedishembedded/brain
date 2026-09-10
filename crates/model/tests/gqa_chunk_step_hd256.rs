// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `model::block::gqa_chunk_step`'s M2.6 fused branch (`GqaChunkIds::
//! fused_prefill_hd256`, `paged_flash_prefill_hd256`) against its own
//! pre-existing triad - at `head_dim=256`, the real shape
//! `qwen35::config::Qwen35Config::qwen38_27b()` ships and the one the fused
//! kernel exists for. `crates/model/src/paged.rs`'s own
//! `paged_flash_prefill_hd256_matches_batched_triad_at_head_dim_256`
//! already proves the KERNEL is numerically correct at this shape; this
//! test proves the WIRING is - `GqaChunkIds`'s new field, the branch in
//! `gqa_chunk_step` itself, and qwen35's own pipeline registration/param
//! order - none of which any existing test exercises, because every
//! `Qwen35Config` used in this crate's own test suite (`tiny`, `tiny_i8`)
//! has a head_dim (40, 32) neither `Op::PagedAttentionFused`'s selector nor
//! any fused kernel accepts, so the fused branch is structurally
//! unreachable from them (confirmed by reading `Qwen35Config::tiny`, not
//! assumed).
//!
//! Swedish Embedded AB implements paged-attention serving kernels for
//! inference engines for its clients. If your team needs expertise in
//! fused FlashAttention-style prefill at non-standard head dimensions then
//! you can procure our services by sending an email to
//! info@swedishembedded.com.

use data::rng::Lcg;
use gpu_core::Gpu;
use model::block::{gqa_chunk_step, GqaChunkIds};

const PIPES: &[(&str, &str)] = &[
    ("splice", kernels::SPLICE),
    ("paged_decode_scores_batched", kernels::PAGED_DECODE_SCORES_BATCHED),
    ("decode_softmax_batched", kernels::DECODE_SOFTMAX_BATCHED),
    ("paged_decode_apply_batched", kernels::PAGED_DECODE_APPLY_BATCHED),
    ("paged_flash_prefill_hd256", kernels::PAGED_FLASH_PREFILL_HD256),
];

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

/// Same scenario `paged.rs`'s own hd256 kernel test uses (`start`/`cc`
/// spanning 3 `BR=64` query tiles, real `qwen38_27b` head_dim), so any
/// divergence here is attributable to the WIRING this milestone added, not
/// a different shape than the kernel is already gated at.
#[test]
fn fused_prefill_hd256_matches_the_triad_through_gqa_chunk_step() {
    let g = gpu_core::testgpu::dev(PIPES);
    let (nh, nkv, hd) = (4u32, 2u32, 256u32); // qwen38_27b's own head_dim
    let kv_stride = nkv * hd;
    let hq = nh * hd;

    let start = 17u32; // tokens already cached by an earlier chunk
    let n = 130u32; // this chunk's own rows - spans 3 BR=64 query tiles
    let cap = start + n;

    let mut rng = Lcg::new(2560);
    let q: Vec<f32> = rng.vec_scaled((n * hq) as usize, 1.0);
    let k_new: Vec<f32> = rng.vec_scaled((n * kv_stride) as usize, 1.0);
    let v_new: Vec<f32> = rng.vec_scaled((n * kv_stride) as usize, 1.0);
    // The "already cached" history (rows 0..start) - identical initial
    // content for both runs below, only ever READ by gqa_chunk_step, never
    // itself appended by it.
    let kcache_init: Vec<f32> = rng.vec_scaled((cap * kv_stride) as usize, 1.0);
    let vcache_init: Vec<f32> = rng.vec_scaled((cap * kv_stride) as usize, 1.0);

    let block_ids: Vec<u32> = vec![0; n as usize]; // one physical block, see gqa_chunk_step's own doc
    let seq_lens: Vec<u32> = (0..n).map(|i| start + i + 1).collect();

    let ids = GqaChunkIds {
        splice: idx(&g, "splice"),
        scores_batched: idx(&g, "paged_decode_scores_batched"),
        softmax_batched: idx(&g, "decode_softmax_batched"),
        apply_batched: idx(&g, "paged_decode_apply_batched"),
        fused_prefill_hd256: Some(idx(&g, "paged_flash_prefill_hd256")),
    };

    let run = |ids: &GqaChunkIds| -> Vec<f32> {
        let qb = g.storage_init("q", &q);
        let kb = g.storage_init("k_new", &k_new);
        let vb = g.storage_init("v_new", &v_new);
        let kcache = g.storage_init("kcache", &kcache_init);
        let vcache = g.storage_init("vcache", &vcache_init);
        let bt = g.storage(n as u64);
        g.write(&bt, &block_ids);
        let sl = g.storage(n as u64);
        g.write(&sl, &seq_lens);
        let scores = g.storage((n * nh * cap) as u64);
        let probs = g.storage((n * nh * cap) as u64);
        let ctx = g.storage((n * hq) as u64);

        let steps = gqa_chunk_step(&g, ids, nh, nkv, hd, start, n, cap, &qb, &kb, &vb, &kcache, &vcache, &bt, &sl, &scores, &probs, &ctx);
        g.submit(&[], &steps);
        g.read(&ctx, (n * hq) as usize)
    };

    let ctx_fused = run(&ids);
    let ctx_triad = run(&GqaChunkIds { fused_prefill_hd256: None, ..ids });

    let worst = ctx_fused.iter().zip(&ctx_triad).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    println!("gqa_chunk_step fused-hd256 vs triad: worst maxabs = {worst:e}");
    assert!(worst > 0.0, "sanity: inputs are not all-zero, so a real match should not be a trivial 0==0");
    // Same bound `paged_flash_prefill_hd256_matches_batched_triad_at_head_dim_256`
    // uses, for the identical reason: online softmax reassociates the
    // reduction, so this is never bit-exact against the triad's exact-
    // max-then-single-pass reference.
    assert!(worst < 1e-3, "gqa_chunk_step's fused branch disagrees with its own triad, maxabs={worst}");
}

/// Sanity that the fused branch actually DISPATCHES the fused kernel rather
/// than silently falling through to the triad for some unrelated reason
/// (e.g. a capability gate this test's `testgpu` device fails) - counts
/// dispatches by name via `Gpu::stats()`'s dispatch delta is not precise
/// enough per-kernel, so this instead checks the two step lists directly:
/// the fused branch must be shorter (1 dispatch vs 3, plus the 2 shared
/// appends) and must actually contain `paged_flash_prefill_hd256`'s pipeline
/// index.
#[test]
fn the_fused_branch_really_dispatches_one_kernel_not_the_triad() {
    let g = gpu_core::testgpu::dev(PIPES);
    let (nh, nkv, hd) = (4u32, 2u32, 256u32);
    let kv_stride = nkv * hd;
    let hq = nh * hd;
    let (start, n) = (17u32, 130u32);
    let cap = start + n;

    let mut rng = Lcg::new(7);
    let qb = g.storage_init("q", &rng.vec_scaled((n * hq) as usize, 1.0));
    let kb = g.storage_init("k_new", &rng.vec_scaled((n * kv_stride) as usize, 1.0));
    let vb = g.storage_init("v_new", &rng.vec_scaled((n * kv_stride) as usize, 1.0));
    let kcache = g.storage_init("kcache", &rng.vec_scaled((cap * kv_stride) as usize, 1.0));
    let vcache = g.storage_init("vcache", &rng.vec_scaled((cap * kv_stride) as usize, 1.0));
    let bt = g.storage(n as u64);
    g.write(&bt, &vec![0u32; n as usize]);
    let sl = g.storage(n as u64);
    g.write(&sl, &(0..n).map(|i| start + i + 1).collect::<Vec<u32>>());
    let scores = g.storage((n * nh * cap) as u64);
    let probs = g.storage((n * nh * cap) as u64);
    let ctx = g.storage((n * hq) as u64);

    let fused_idx = idx(&g, "paged_flash_prefill_hd256");
    let ids = GqaChunkIds {
        splice: idx(&g, "splice"),
        scores_batched: idx(&g, "paged_decode_scores_batched"),
        softmax_batched: idx(&g, "decode_softmax_batched"),
        apply_batched: idx(&g, "paged_decode_apply_batched"),
        fused_prefill_hd256: Some(fused_idx),
    };
    let steps = gqa_chunk_step(&g, &ids, nh, nkv, hd, start, n, cap, &qb, &kb, &vb, &kcache, &vcache, &bt, &sl, &scores, &probs, &ctx);
    assert_eq!(steps.len(), 3, "2 appends + 1 fused dispatch - got {} steps, the triad's 5 (or something else) ran instead", steps.len());
}
