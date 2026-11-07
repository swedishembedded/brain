// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The fused paged flash attention of the serving engine against the score-slab triad, at Ripple's group of 10.
//!
//! Swedish Embedded AB implements solutions for serving long-context decoders whose attention runs at the memory
//! roofline instead of materializing a score slab per query, for its clients. If your team needs expertise in
//! fused decode and prefill attention kernels then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! The fused kernels are workgroup-cooperative: the CPU JIT cannot run them and keeps the triad (these tests skip by
//! name there); the software Vulkan adapter runs them (`BRAIN_DEVICE=software`). The model is synthetic with the
//! published attention geometry scaled down in width only: 20 query heads over 2 kv heads of 128, a group of 10 as
//! Ripple's 80 over 8, so the specialisation under test (head_dim 128, group above the kernel's old limit of 8) is
//! the real one. A window of 4 wraps the rings over a dozen tokens.

use model::kv_tier::KvTier;
use model::paged::BlockTable;
use model::serve::conformance::{self, Conformance};
use model::serve::PagedDecoder;
use ricursive::{RicursiveEngine, ServeConfig};
use std::sync::OnceLock;

use crate::serve::{prefill_in_slices, stream};
use crate::synthetic::{synthetic, Model, Shape};

const LAYERS: [&str; 5] = ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "full_attention"];

fn model() -> &'static Model {
    static MODEL: OnceLock<Model> = OnceLock::new();
    MODEL.get_or_init(|| synthetic(&LAYERS, Shape { heads: 20, kv_heads: 2, head_dim: 128 }))
}

fn runs_here() -> bool {
    let runs = gpu_core::testgpu::dev(&[]).caps().workgroup_reductions;
    if !runs {
        brain_testutil::skip_unavailable("fused attention: this backend does not run workgroup-barrier kernels (the triad is its path)");
    }
    runs
}

fn serve(kv: KvTier, fused: bool, chunk: u32) -> ServeConfig {
    ServeConfig { kv_cache: kv, fused_attention: fused, ..ServeConfig::new(4, 160, 96, 4, 2, chunk) }
}

/// Prefill 21 tokens in slices of 5 (chunks that straddle the window of 4 and the blocks of 4), then decode 14: every
/// logits row.
fn trace(e: &mut RicursiveEngine, ids: &[u32]) -> Vec<Vec<f32>> {
    let mut t = BlockTable::new();
    let h = prefill_in_slices(e, &mut t, &ids[..21], 5);
    let mut rows = vec![e.logits(&h)];
    for &token in &ids[21..] {
        rows.push(e.decode_logits(&mut [&mut t], &[token]).remove(0));
    }
    e.release_table(&mut t);
    rows
}

/// Spec: the fused kernels compute the triad's attention (a different reduction order, the same value): the fused engine
/// answers as the triad engine at every KV tier, through prefill chunks (the fused prefill on the global layers, the
/// split-key decode kernel over the rings' windows) and decode, across window wraps.
#[test]
fn the_fused_attention_equals_the_triad_at_every_kv_tier_through_window_wraps() {
    if !runs_here() {
        return;
    }
    let m = model();
    let ids = stream(5, 21 + 14, m.cfg.vocab_size as u32);
    let reference = trace(&mut m.engine(serve(KvTier::F32, false, 8)), &ids);
    let distance = |a: &[Vec<f32>], b: &[Vec<f32>]| a.iter().zip(b).map(|(x, y)| crate::taps::scaled_error("row", y, x)).fold(0.0f32, f32::max);
    for tier in [KvTier::F32, KvTier::BF16, KvTier::INT8, KvTier::FP8] {
        let mut triad = m.engine(serve(tier, false, 8));
        let mut fused = m.engine(serve(tier, true, 8));
        assert!(!triad.fused_attention() && fused.fused_attention(), "{tier}: the engine picked the kernels it was asked for");
        let (a, b) = (trace(&mut triad, &ids), trace(&mut fused, &ids));
        let (worst, from_triad, from_fused) = (distance(&a, &b), distance(&reference, &a), distance(&reference, &b));
        eprintln!("fused vs triad at {tier}: worst scaled error {worst:.3e} over {} rows (to the f32 cache: triad {from_triad:.3e}, fused {from_fused:.3e})", a.len());
        // The fused kernels are exact against f64 attention over the decoded planes (2e-6; `model`'s kv_tier gates, group 10
        // at head_dim 128 included). The triad's bf16, int8 and fp8 variants are gated there at 1e-4 per operation, which
        // five random layers amplify to the distance between the two engines' logits. What the quantization costs against
        // the f32 cache must not grow with the fused kernels.
        assert!(worst <= 2e-3, "{tier}: fused differs from the triad by {worst:.3e}");
        assert!(from_fused <= from_triad * 1.05 + 1e-5, "{tier}: the fused engine is {from_fused:.3e} from the f32 cache, the triad {from_triad:.3e}");
    }
}

/// Spec: without workgroup barriers the engine keeps the triad and says so; with `fused_attention` off it keeps the triad
/// on any device.
#[test]
fn the_engine_picks_fused_kernels_by_capability_and_keeps_the_triad_as_the_reference() {
    let m = model();
    let pinned = m.engine(serve(KvTier::F32, false, 8));
    assert!(!pinned.fused_attention());
    let wanted = m.engine(serve(KvTier::F32, true, 8));
    let capable = gpu_core::testgpu::dev(&[]).caps().workgroup_reductions;
    assert_eq!(wanted.fused_attention(), capable, "fused where the device runs workgroup barriers, the triad on the CPU JIT");
}

/// Spec: with the fused kernels there is no score slab, so the dense path's context is not capped by `dense_keys`: an
/// engine whose triad scratch would refuse a read past its cap serves it.
#[test]
fn the_fused_dense_path_is_not_capped_by_the_score_scratch() {
    if !runs_here() {
        return;
    }
    let m = model();
    let ids = stream(9, 70, m.cfg.vocab_size as u32);
    let cap = |fused: bool| ServeConfig { sparse_attention: true, dense_keys: Some(16), ..serve(KvTier::F32, fused, 8) };
    // 70 keys on a global layer against a triad scratch of 16 keys: the triad engine refuses by name.
    let mut triad = m.engine(cap(false));
    let mut t = BlockTable::new();
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| prefill_in_slices(&mut triad, &mut t, &ids, 8)));
    assert!(refused.is_err(), "the triad cannot read 70 keys with a 16-key scratch");
    let mut fused = m.engine(cap(true));
    let mut t = BlockTable::new();
    prefill_in_slices(&mut fused, &mut t, &ids, 8);
    let again = fused.decode_logits(&mut [&mut t], &[ids[3]]);
    let mut reference = m.engine(serve(KvTier::F32, false, 8));
    let mut r = BlockTable::new();
    prefill_in_slices(&mut reference, &mut r, &ids, 8);
    let want = reference.decode_logits(&mut [&mut r], &[ids[3]]);
    assert!(crate::taps::scaled_error("fused past the cap", &again[0], &want[0]) <= 1e-4);
}

fn conformance_engine() -> RicursiveEngine {
    model().engine(ServeConfig { fused_attention: true, ..ServeConfig::new(4, 160, 96, 6, 4, 8) })
}

/// The shared serving suite (continuous batching, no head-of-line blocking, prefix reuse) on the fused kernels.
#[test]
fn the_shared_serving_conformance_suite_passes_on_the_fused_kernels() {
    if !runs_here() {
        return;
    }
    assert!(conformance_engine().fused_attention());
    conformance::run_all(&Conformance { make: conformance_engine, prefix_reuse: true, prefix_prompt_tokens: 24, load_adapter: None });
}
