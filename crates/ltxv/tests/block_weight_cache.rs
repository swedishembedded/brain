// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `crate::dit::forward_q_streamed`'s per-generation cache
//! (`ltxv::block::GenerationCache`: block weights AND embeddings-connector
//! routing)
//! (Phase 9's exact win over Phase 8's own "~86% of one real denoise step is
//! GGUF re-read + re-quantize of the SAME immutable weights, every single
//! step" finding) - gated on BIT-IDENTICAL output, not a cosine threshold,
//! since a cached forward is supposed to compute the exact same function as
//! an uncached one, just skip redundant work getting there.
//!
//! Two things, matching this crate's own `int8_compute.rs`/`int8_storage.rs`
//! style:
//!
//! 1. A tiny synthetic config (`LtxDitConfig::tiny()`, `random_tiny_weights` -
//!    a plain `Tensors` `HashMap` already implements `checkpoint::
//!    TensorSource`, so no real GGUF is needed here) - always runs, no fixture
//!    dependency.
//! 2. A REAL-weight-gated check on one small shape of the actual production
//!    checkpoint, matching `dit_parity.rs`'s `real_weight` module's own GGUF
//!    resolution convention.

use ltxv::block::{GenerationCache, QTier};
use ltxv::dit::{forward_q_streamed, load_head_tensors_from_source, random_tiny_weights};
use ltxv::modelgrad::Cfg;
use ltxv::LtxDitConfig;

struct Inputs {
    latent: Vec<f32>,
    timesteps: Vec<f32>,
    positions: Vec<f32>,
    keyframes_mask: Vec<f32>,
    context: Vec<f32>,
    context_valid: Vec<f32>,
    context_len: usize,
    t: usize,
}

fn synthetic_inputs(cfg: &LtxDitConfig, t: usize, context_len: usize) -> Inputs {
    let mcfg = Cfg::from_ltx(cfg, t, context_len);
    let positions = mcfg.simple_positions();
    let latent: Vec<f32> = (0..t * cfg.in_channels as usize).map(|i| ((i % 23) as f32 / 23.0 - 0.5) * 1.1).collect();
    let context: Vec<f32> = (0..context_len * cfg.cross_attention_dim as usize).map(|i| ((i % 7) as f32 / 7.0 - 0.5) * 1.4).collect();
    let timesteps: Vec<f32> = (0..t).map(|i| 0.2 + 0.05 * (i % 5) as f32).collect();
    let mut keyframes_mask = vec![0f32; t];
    keyframes_mask[0] = 1.0;
    let context_valid = vec![1.0f32; context_len];
    Inputs { latent, timesteps, positions, keyframes_mask, context, context_valid, context_len, t }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "max_abs_diff: length mismatch ({} vs {})", a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

// ------------------------------------------------------------ 1. synthetic

/// The core claim: a forward that populates an empty cache (cache miss on
/// every layer) and a LATER forward on the SAME cache (cache hit on every
/// layer) produce bit-identical output - `max_abs == 0.0`, not merely close.
/// `model::int8::quantize_weight`/`quantize_weight_q4` are pure functions of
/// the checkpoint's own weight bytes, so this is not a tolerance the cache
/// happens to meet, it is what "the cache changes no math" means concretely.
#[test]
fn cached_forward_is_bit_identical_to_an_uncached_one() {
    let cfg = LtxDitConfig { num_layers: 3, ..LtxDitConfig::tiny() };
    let w = random_tiny_weights(&cfg, 0x000C_ACE1);
    let head = load_head_tensors_from_source(&w, &cfg);
    let inputs = synthetic_inputs(&cfg, 7, 5);

    // Baseline: two independent forwards, each with its OWN fresh (empty)
    // cache - both are cache-miss-only runs, so this is also a determinism
    // sanity check before the cache is even exercised.
    let fresh_a = GenerationCache::default();
    let out_fresh_a = forward_q_streamed(&cfg, &w, &head, None, QTier::Int8, &inputs.latent, &inputs.timesteps, &inputs.positions, &inputs.keyframes_mask, &inputs.context, inputs.context_len, inputs.t, &inputs.context_valid, &fresh_a);
    let fresh_b = GenerationCache::default();
    let out_fresh_b = forward_q_streamed(&cfg, &w, &head, None, QTier::Int8, &inputs.latent, &inputs.timesteps, &inputs.positions, &inputs.keyframes_mask, &inputs.context, inputs.context_len, inputs.t, &inputs.context_valid, &fresh_b);
    assert_eq!(max_abs_diff(&out_fresh_a, &out_fresh_b), 0.0, "two independent cache-miss-only forwards of the SAME weights/inputs must already be bit-identical");

    // The real claim: one SHARED cache, called twice. Call 1 is a cache miss
    // on every layer (identical work to `fresh_a`/`fresh_b` above); call 2 is
    // a cache HIT on every layer (the GGUF-read-equivalent + CPU-quantize
    // work is skipped entirely - see `forward_q_streamed`'s own doc).
    let shared = GenerationCache::default();
    let out_shared_1 = forward_q_streamed(&cfg, &w, &head, None, QTier::Int8, &inputs.latent, &inputs.timesteps, &inputs.positions, &inputs.keyframes_mask, &inputs.context, inputs.context_len, inputs.t, &inputs.context_valid, &shared);
    assert_eq!(max_abs_diff(&out_shared_1, &out_fresh_a), 0.0, "a cache-populating (miss) forward must match an independent cache-free forward exactly");

    // Structural check that caching actually happened, not vacuously: every
    // layer's slot is populated after call 1, before call 2 ever runs.
    assert_eq!(shared.blocks().borrow().len(), cfg.num_layers as usize);
    assert!(shared.blocks().borrow().iter().all(|slot| slot.is_some()), "every layer must be cached after the first (cache-miss) forward");

    let out_shared_2 = forward_q_streamed(&cfg, &w, &head, None, QTier::Int8, &inputs.latent, &inputs.timesteps, &inputs.positions, &inputs.keyframes_mask, &inputs.context, inputs.context_len, inputs.t, &inputs.context_valid, &shared);
    assert_eq!(max_abs_diff(&out_shared_2, &out_fresh_a), 0.0, "a cache-HIT forward must be bit-identical to the cache-free reference - the cache must change no math");
}

/// Different `context` (the CFG conditional/unconditional split
/// `crate::pipeline::denoise` actually drives) still hits the SAME
/// block-weight cache correctly, since the cache is keyed by LAYER, never by
/// `context` - the block WEIGHTS a cache slot holds have nothing to do with
/// which branch is calling. Guards against a cache design that accidentally
/// only works because the two calls happened to pass identical arguments.
#[test]
fn cache_hit_is_correct_across_different_contexts_sharing_one_cache() {
    let cfg = LtxDitConfig { num_layers: 2, ..LtxDitConfig::tiny() };
    let w = random_tiny_weights(&cfg, 0x000C_ACE2);
    let head = load_head_tensors_from_source(&w, &cfg);
    let inputs_a = synthetic_inputs(&cfg, 6, 4);
    let mut inputs_b = synthetic_inputs(&cfg, 6, 4);
    for v in inputs_b.context.iter_mut() {
        *v *= -1.3;
    }
    assert_ne!(inputs_a.context, inputs_b.context, "test setup: the two branches must actually differ");

    // Reference: each context run cache-free (fresh cache each time).
    let ref_a = {
        let c = GenerationCache::default();
        forward_q_streamed(&cfg, &w, &head, None, QTier::Int8, &inputs_a.latent, &inputs_a.timesteps, &inputs_a.positions, &inputs_a.keyframes_mask, &inputs_a.context, inputs_a.context_len, inputs_a.t, &inputs_a.context_valid, &c)
    };
    let ref_b = {
        let c = GenerationCache::default();
        forward_q_streamed(&cfg, &w, &head, None, QTier::Int8, &inputs_b.latent, &inputs_b.timesteps, &inputs_b.positions, &inputs_b.keyframes_mask, &inputs_b.context, inputs_b.context_len, inputs_b.t, &inputs_b.context_valid, &c)
    };
    assert_ne!(ref_a, ref_b, "test setup: different context must actually change the output");

    // The `denoise` loop's real shape: one shared cache, branch A then
    // branch B (A populates every layer's cache slot; B hits it).
    let shared = GenerationCache::default();
    let out_a = forward_q_streamed(&cfg, &w, &head, None, QTier::Int8, &inputs_a.latent, &inputs_a.timesteps, &inputs_a.positions, &inputs_a.keyframes_mask, &inputs_a.context, inputs_a.context_len, inputs_a.t, &inputs_a.context_valid, &shared);
    let out_b = forward_q_streamed(&cfg, &w, &head, None, QTier::Int8, &inputs_b.latent, &inputs_b.timesteps, &inputs_b.positions, &inputs_b.keyframes_mask, &inputs_b.context, inputs_b.context_len, inputs_b.t, &inputs_b.context_valid, &shared);

    assert_eq!(max_abs_diff(&out_a, &ref_a), 0.0, "branch A (the cache-populating call) must match its cache-free reference exactly");
    assert_eq!(max_abs_diff(&out_b, &ref_b), 0.0, "branch B (the cache-HIT call, different context) must match ITS OWN cache-free reference exactly - a stale/wrong-layer cache bug would show up here as a wrong number, not a crash");
}

// ------------------------------------------ 1b. connector routing caching

/// The connector half of the cache, on a config that actually HAS a connector
/// (`LtxDitConfig::tiny_gated` enables it; plain `tiny` does not, which is why
/// the tests above cannot see this path at all).
///
/// Same claim, same standard as the block half: a forward that populates an
/// empty cache and a later forward on the same cache must be bit-identical to
/// two independent cache-free forwards. The connector reads only `context`,
/// `context_valid` and `context_len` - all fixed for a generation - so reusing
/// its output is skipping repeated work, not approximating it.
#[test]
fn cached_connector_routing_is_bit_identical_to_recomputing_it() {
    let cfg = LtxDitConfig { num_layers: 2, ..LtxDitConfig::tiny_gated() };
    assert!(cfg.use_embeddings_connector, "test setup: this config must actually route through the connector");
    let w = random_tiny_weights(&cfg, 0x000C_ACE3);
    let head = load_head_tensors_from_source(&w, &cfg);
    // `context_len` must be a multiple of the connector's register count.
    let inputs = synthetic_inputs(&cfg, 6, cfg.connector_num_learnable_registers as usize * 2);

    let reference = forward_q_streamed(&cfg, &w, &head, None, QTier::Int8, &inputs.latent, &inputs.timesteps, &inputs.positions, &inputs.keyframes_mask, &inputs.context, inputs.context_len, inputs.t, &inputs.context_valid, &GenerationCache::default());

    let shared = GenerationCache::default();
    let miss = forward_q_streamed(&cfg, &w, &head, None, QTier::Int8, &inputs.latent, &inputs.timesteps, &inputs.positions, &inputs.keyframes_mask, &inputs.context, inputs.context_len, inputs.t, &inputs.context_valid, &shared);
    assert_eq!(max_abs_diff(&miss, &reference), 0.0, "the connector-cache-populating forward must match a cache-free forward exactly");
    // Not vacuous: something was actually stored, so call 2 really is a hit.
    assert!(shared.connector_byte_len() > 0, "the connector routing must actually have been cached by the first forward");

    let hit = forward_q_streamed(&cfg, &w, &head, None, QTier::Int8, &inputs.latent, &inputs.timesteps, &inputs.positions, &inputs.keyframes_mask, &inputs.context, inputs.context_len, inputs.t, &inputs.context_valid, &shared);
    assert_eq!(max_abs_diff(&hit, &reference), 0.0, "a connector-cache-HIT forward must be bit-identical to the cache-free reference");
}

/// The bug class a same-context test cannot see: two DIFFERENT contexts
/// sharing one cache (exactly what classifier-free guidance does - a
/// conditional and an unconditional branch against one `RealDit`). Each branch
/// must get its OWN connector routing back, not whichever one ran first.
///
/// This is the assertion that makes the cache key load-bearing: keyed on the
/// full context/mask/length, branch B misses and computes its own answer;
/// keyed on anything coarser (a step index, "the connector output", a
/// truncated hash) branch B would silently receive branch A's conditioning and
/// the two outputs would collapse onto each other.
#[test]
fn two_contexts_sharing_one_cache_each_get_their_own_connector_routing() {
    let cfg = LtxDitConfig { num_layers: 2, ..LtxDitConfig::tiny_gated() };
    let w = random_tiny_weights(&cfg, 0x000C_ACE4);
    let head = load_head_tensors_from_source(&w, &cfg);
    let regs = cfg.connector_num_learnable_registers as usize;
    let inputs_a = synthetic_inputs(&cfg, 6, regs * 2);
    let mut inputs_b = synthetic_inputs(&cfg, 6, regs * 2);
    for v in inputs_b.context.iter_mut() {
        *v = *v * -0.8 + 0.21;
    }
    // Also differ in the VALIDITY mask, not only the values: the mask selects
    // which positions the connector substitutes its learnable registers into,
    // so a key that ignored it would be just as wrong and is not otherwise
    // exercised anywhere.
    inputs_b.context_valid[regs] = 0.0;

    let run = |i: &Inputs, cache: &GenerationCache| forward_q_streamed(&cfg, &w, &head, None, QTier::Int8, &i.latent, &i.timesteps, &i.positions, &i.keyframes_mask, &i.context, i.context_len, i.t, &i.context_valid, cache);

    let ref_a = run(&inputs_a, &GenerationCache::default());
    let ref_b = run(&inputs_b, &GenerationCache::default());
    assert_ne!(ref_a, ref_b, "test setup: the two branches must actually produce different output");

    let shared = GenerationCache::default();
    let out_a = run(&inputs_a, &shared);
    let out_b = run(&inputs_b, &shared);
    assert_eq!(max_abs_diff(&out_a, &ref_a), 0.0, "branch A must match its own cache-free reference exactly");
    assert_eq!(max_abs_diff(&out_b, &ref_b), 0.0, "branch B must get ITS OWN connector routing, not branch A's - a key that ignored the context (or its validity mask) would show up here as branch B returning branch A's answer");

    // And a third call on branch A's context must still hit A's entry, not B's.
    let out_a2 = run(&inputs_a, &shared);
    assert_eq!(max_abs_diff(&out_a2, &ref_a), 0.0, "returning to branch A after branch B must still produce branch A's answer");
}

// --------------------------------------------------------- 2. real-weight

const REPO: &str = "Lightricks/LTX-2.5";

fn gguf_path() -> Option<String> {
    if let Ok(p) = std::env::var("BRAIN_LTXV_DIT") {
        if !p.is_empty() {
            return Some(p);
        }
    }
    let dir = brain_testutil::model_dir(REPO)?;
    let mut found: Vec<String> = std::fs::read_dir(&dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.contains("Q8_0") && n.ends_with(".gguf")))
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    found.sort();
    found.into_iter().next()
}

/// The same bit-identical claim as [`cached_forward_is_bit_identical_to_an_uncached_one`],
/// on the real production path: the real Q8_0 GGUF, `LtxDitConfig::ltx25_22b()`
/// reduced to a few layers, small token/context counts (this task's own
/// "start small" constraint - this is a correctness gate, not a timing run).
#[test]
fn real_checkpoint_cached_forward_is_bit_identical_to_an_uncached_one() {
    let Some(path) = gguf_path() else {
        brain_testutil::skip(&format!("set BRAIN_LTXV_DIT to a real {REPO} distilled Q8_0 GGUF (none in the model store)"));
        return;
    };
    let cfg = LtxDitConfig { num_layers: 2, use_embeddings_connector: false, ..LtxDitConfig::ltx25_22b() };
    let src = ltxv::gguf_src::LtxvGgufSource::open(&path).unwrap_or_else(|e| panic!("opening {path}: {e}"));
    let head = load_head_tensors_from_source(&src, &cfg);
    let inputs = synthetic_inputs(&cfg, 8, 6);

    let fresh = GenerationCache::default();
    let out_fresh = forward_q_streamed(&cfg, &src, &head, None, QTier::Int8, &inputs.latent, &inputs.timesteps, &inputs.positions, &inputs.keyframes_mask, &inputs.context, inputs.context_len, inputs.t, &inputs.context_valid, &fresh);

    let shared = GenerationCache::default();
    let out_shared_1 = forward_q_streamed(&cfg, &src, &head, None, QTier::Int8, &inputs.latent, &inputs.timesteps, &inputs.positions, &inputs.keyframes_mask, &inputs.context, inputs.context_len, inputs.t, &inputs.context_valid, &shared);
    assert_eq!(max_abs_diff(&out_shared_1, &out_fresh), 0.0, "real-checkpoint cache-populating forward must match the cache-free reference exactly");

    let out_shared_2 = forward_q_streamed(&cfg, &src, &head, None, QTier::Int8, &inputs.latent, &inputs.timesteps, &inputs.positions, &inputs.keyframes_mask, &inputs.context, inputs.context_len, inputs.t, &inputs.context_valid, &shared);
    assert_eq!(max_abs_diff(&out_shared_2, &out_fresh), 0.0, "real-checkpoint cache-HIT forward must be bit-identical to the cache-free reference");
    println!("real Q8_0 checkpoint: cache-miss and cache-hit forwards agree exactly (max_abs=0.0), t={}, context_len={}, layers={}", inputs.t, inputs.context_len, cfg.num_layers);

    // Memory-footprint measurement, not assertion-from-a-derivation (lesson:
    // a memory saving is not measured by anything unless someone measures
    // it): `shared` now holds `cfg.num_layers` REAL cached blocks at the
    // real 22B checkpoint's width - read their actual byte count and
    // extrapolate to the full 48-layer model the way `device_bytes_real.rs`
    // already extrapolates the device-side int8 ratio.
    let cache_ref = shared.blocks().borrow();
    let per_layer_bytes: Vec<usize> = cache_ref.iter().map(|c| c.as_ref().expect("populated above").byte_len()).collect();
    let avg_bytes = per_layer_bytes.iter().sum::<usize>() as f64 / per_layer_bytes.len() as f64;
    let extrapolated_48_gb = avg_bytes * 48.0 / 1e9;
    println!("measured per-block host cache footprint: {:.1} MB/block (real 22B width) -> extrapolated 48-layer total: {extrapolated_48_gb:.2} GB", avg_bytes / 1e6);
    // Sanity band, not a tight bound: the real 22B config's own device-bytes
    // test (`device_bytes_real.rs`) measured a real (3.5, 4.0)x int8
    // compression ratio against the fp32 block size, never the flat
    // theoretical 4x - this cache stores the identical packed bytes host-side,
    // so its total footprint should land well under the ~42 GB bf16 model
    // size and comfortably within this class of hardware's RAM (184 GiB).
    assert!(extrapolated_48_gb > 5.0 && extrapolated_48_gb < 40.0, "extrapolated 48-layer cache footprint {extrapolated_48_gb:.2} GB is outside the sane band for an int8-quantized 22B model's block weights");
}
