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
    assert_eq!(shared.stats().blocks, cfg.num_layers as usize);
    assert!((0..cfg.num_layers as usize).all(|l| shared.is_cached(l, QTier::Int8)), "every layer must be cached after the first (cache-miss) forward");

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

// -------------------------------------------- 1c. the footprint estimator

/// `block::cached_block_bytes` is what `resident_ltxv.rs` budgets against
/// BEFORE anything is loaded, so it must be the same number a real cached
/// block actually weighs - not a `file_size * 1.3` guess. Pinned against a
/// really-quantized block at both tiers and both gate settings, so a change
/// to what `quantize_host` stores cannot silently make every residency
/// estimate wrong.
#[test]
fn cached_block_bytes_matches_a_real_measured_block() {
    for (label, base) in [("tiny", LtxDitConfig::tiny()), ("tiny_gated", tiny_gated_quantizable())] {
        for tier in [QTier::Int8, QTier::Int4] {
            let cfg = LtxDitConfig { num_layers: 1, ..base };
            let w = random_tiny_weights(&cfg, 0x000C_ACE7);
            let head = load_head_tensors_from_source(&w, &cfg);
            let inputs = synthetic_inputs(&cfg, 6, cfg.connector_num_learnable_registers.max(1) as usize * 2);
            let cache = GenerationCache::default();
            forward_q_streamed(&cfg, &w, &head, None, tier, &inputs.latent, &inputs.timesteps, &inputs.positions, &inputs.keyframes_mask, &inputs.context, inputs.context_len, inputs.t, &inputs.context_valid, &cache);
            let measured = cache.block_byte_lens()[0] as u64;
            let predicted = ltxv::block::cached_block_bytes(&cfg, tier);
            assert_eq!(predicted, measured, "{label}/{tier:?}: the closed-form footprint must equal a really-quantized block's own byte_len");
        }
    }
}

/// [`LtxDitConfig::tiny_gated`] at dims the quantized tiers can actually
/// take. `tiny_gated` is a GOLDEN-parity fixture: its `inner_dim = 24` is
/// transcribed from the reference dumper and must not move. But
/// `model::int8::quantize_weight` scales per 32-element group of a linear's
/// contraction axis (`model::int8::GROUP`), so 24 is not quantizable at all.
/// 96 keeps everything this config exists to exercise - the gates on, the
/// connector on, and TWO different factorizations of one `inner_dim`
/// (3 x 32 for the main attention, 4 x 24 for the connector, so a
/// heads/head_dim transpose between them still cannot hide) - while being
/// three whole scale groups wide. `head_dim` stays even and `inner_dim / 2`
/// stays a multiple of `num_heads`, `crate::rope::ltx_rope_tables`'s own
/// divisibility rule.
fn tiny_gated_quantizable() -> LtxDitConfig {
    LtxDitConfig {
        inner_dim: 96,
        cross_attention_dim: 96,
        connector_attention_head_dim: 24,
        ..LtxDitConfig::tiny_gated()
    }
}

// ------------------------------- 1a. cross-GENERATION reuse and eviction

/// Write `bytes` to a unique temp path and return it - a stand-in checkpoint
/// FILE, used only for its identity (path + length + mtime). The weights
/// these tests actually forward through stay the synthetic in-memory
/// `Tensors`; what is under test here is which STORE a handle resolves to,
/// which is exactly the thing that decides whether generation B re-reads the
/// disk.
fn identity_file(tag: &str, bytes: &[u8]) -> String {
    let dir = std::env::temp_dir().join(format!("ltxv-bwc-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("checkpoint.gguf");
    std::fs::write(&p, bytes).unwrap();
    p.to_string_lossy().into_owned()
}

/// **The milestone's core claim.** Two SEPARATE generations - each obtaining
/// its own cache handle the way `pipeline::RealDit` does, from the
/// checkpoint-keyed registry, with nothing carried between them but the
/// checkpoint path - share cache entries, and generation B's output is
/// bit-identical to a cache-free forward.
///
/// Before this, the cache was owned by the `RealDit` and died with the
/// `generate()` call, so B re-read and re-quantized every block from scratch.
/// The two assertions that make this non-vacuous are the hit COUNTERS: B must
/// record `num_layers` hits and ZERO block misses, from its very first layer.
#[test]
fn a_second_generation_reuses_the_first_generations_entries_bit_identically() {
    let cfg = LtxDitConfig { num_layers: 3, ..LtxDitConfig::tiny() };
    let w = random_tiny_weights(&cfg, 0x000C_ACE5);
    let head = load_head_tensors_from_source(&w, &cfg);
    let inputs_a = synthetic_inputs(&cfg, 7, 5);
    // Generation B is a DIFFERENT prompt/latent, as two back-to-back real
    // generations would be - the entries being shared are weights, which
    // depend on neither.
    let mut inputs_b = synthetic_inputs(&cfg, 7, 5);
    for v in inputs_b.latent.iter_mut() {
        *v = *v * 0.4 - 0.17;
    }

    let path = identity_file("reuse", b"a stand-in checkpoint's bytes");
    let reference_b = forward_q_streamed(&cfg, &w, &head, None, QTier::Int8, &inputs_b.latent, &inputs_b.timesteps, &inputs_b.positions, &inputs_b.keyframes_mask, &inputs_b.context, inputs_b.context_len, inputs_b.t, &inputs_b.context_valid, &GenerationCache::default());

    // ---- generation A: its own handle, its own scope, dropped at the end.
    {
        let cache_a = GenerationCache::for_checkpoint(&path);
        forward_q_streamed(&cfg, &w, &head, None, QTier::Int8, &inputs_a.latent, &inputs_a.timesteps, &inputs_a.positions, &inputs_a.keyframes_mask, &inputs_a.context, inputs_a.context_len, inputs_a.t, &inputs_a.context_valid, &cache_a);
        assert_eq!(cache_a.stats().blocks, cfg.num_layers as usize, "generation A must have populated every layer");
    }

    // ---- generation B: a fresh handle, resolved only from the path.
    let cache_b = GenerationCache::for_checkpoint(&path);
    let before = cache_b.stats();
    assert_eq!(before.blocks, cfg.num_layers as usize, "generation B's handle must resolve to the store generation A populated, not an empty one");
    let out_b = forward_q_streamed(&cfg, &w, &head, None, QTier::Int8, &inputs_b.latent, &inputs_b.timesteps, &inputs_b.positions, &inputs_b.keyframes_mask, &inputs_b.context, inputs_b.context_len, inputs_b.t, &inputs_b.context_valid, &cache_b);
    let after = cache_b.stats();
    assert_eq!(after.hits - before.hits, cfg.num_layers as u64, "every one of generation B's layers must be a cache HIT, from its first layer");
    assert_eq!(after.misses - before.misses, 0, "generation B must not miss on any layer");
    assert_eq!(max_abs_diff(&out_b, &reference_b), 0.0, "generation B's cache-HIT output must be bit-identical to a cache-free forward of the same inputs");

    // A different checkpoint identity must NOT see any of it - otherwise the
    // key is not doing its job and a replaced checkpoint would serve stale
    // weights.
    let other = GenerationCache::for_checkpoint(&identity_file("reuse-other", b"different bytes entirely, different length"));
    assert_eq!(other.stats().blocks, 0, "a different checkpoint identity must start empty");
}

/// Eviction correctness: under a budget too small to hold the whole model,
/// entries are dropped by the shared cost-aware policy and a later access
/// RE-POPULATES correctly - it does not return a stale entry and does not
/// silently change a number. An eviction here can only ever cost time.
#[test]
fn eviction_under_a_tight_ram_ceiling_repopulates_correctly() {
    let cfg = LtxDitConfig { num_layers: 4, ..LtxDitConfig::tiny() };
    let w = random_tiny_weights(&cfg, 0x000C_ACE6);
    let head = load_head_tensors_from_source(&w, &cfg);
    let inputs = synthetic_inputs(&cfg, 6, 4);
    let run = |cache: &GenerationCache| forward_q_streamed(&cfg, &w, &head, None, QTier::Int8, &inputs.latent, &inputs.timesteps, &inputs.positions, &inputs.keyframes_mask, &inputs.context, inputs.context_len, inputs.t, &inputs.context_valid, cache);

    let reference = run(&GenerationCache::default());

    // Size the ceiling off a REAL measured block, not a guess: two blocks'
    // worth, against a four-layer model, so eviction is forced and provably
    // partial (some entries survive, so this is not just "caching disabled").
    let measured = GenerationCache::with_budget(None);
    run(&measured);
    let per_block = measured.block_byte_lens().into_iter().max().expect("populated") as u64;
    let tight = GenerationCache::with_budget(Some(per_block * 2));

    let out1 = run(&tight);
    let s1 = tight.stats();
    assert_eq!(max_abs_diff(&out1, &reference), 0.0, "a forward under a tight ceiling must still be bit-identical");
    assert!(s1.evictions > 0, "the ceiling must actually have forced evictions, got {s1:?}");
    assert!(s1.blocks > 0 && s1.blocks < cfg.num_layers as usize, "eviction must be partial, not total: {s1:?}");
    assert!(tight.block_byte_len() <= per_block * 2, "the cache must stay under its budget, got {} > {}", tight.block_byte_len(), per_block * 2);

    // The claim that matters: a SECOND forward, which necessarily misses on
    // the evicted layers, re-reads and re-quantizes them and produces the
    // exact same answer. A stale or wrong-keyed entry would show up here as a
    // wrong number, not as a crash.
    let out2 = run(&tight);
    assert!(tight.stats().misses > s1.misses, "the second forward must genuinely miss on the evicted layers");
    assert_eq!(max_abs_diff(&out2, &reference), 0.0, "a forward that re-populates evicted entries must be bit-identical to the cache-free reference");

    // And a block larger than the whole budget is simply not retained - the
    // forward still runs and is still exact.
    let starved = GenerationCache::with_budget(Some(per_block / 2));
    let out3 = run(&starved);
    assert_eq!(starved.stats().blocks, 0, "no entry may be retained when one block alone exceeds the budget");
    assert_eq!(max_abs_diff(&out3, &reference), 0.0, "a starved cache must still compute the exact same function");
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
    let cfg = LtxDitConfig { num_layers: 2, ..tiny_gated_quantizable() };
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
    let cfg = LtxDitConfig { num_layers: 2, ..tiny_gated_quantizable() };
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
        // Discriminate on the file's OWN declared architecture, not on
        // its name. The model store legitimately holds several Q8_0
        // GGUFs for one repo - the DiT and, since the text encoder was
        // quantized too, Gemma-4 - and a name glob picked whichever
        // sorted first, which surfaced as an architecture mismatch deep
        // inside an importer rather than as "no fixture here".
        .filter(|p| {
            checkpoint::gguf::MmapGguf::open(&p.to_string_lossy())
                .ok()
                .and_then(|g| g.kv().get("general.architecture").and_then(|v| v.as_str()).map(str::to_string))
                .as_deref()
                == Some(ltxv::import::GGUF_ARCHITECTURE)
        })
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
    // The cross-GENERATION claim on the real production path: a second
    // generation resolves its handle from the checkpoint PATH alone (exactly
    // what `pipeline::RealDit` does) and must start warm.
    {
        let gen_a = ltxv::block::GenerationCache::for_checkpoint(&path);
        let out_a = forward_q_streamed(&cfg, &src, &head, None, QTier::Int8, &inputs.latent, &inputs.timesteps, &inputs.positions, &inputs.keyframes_mask, &inputs.context, inputs.context_len, inputs.t, &inputs.context_valid, &gen_a);
        assert_eq!(max_abs_diff(&out_a, &out_fresh), 0.0, "real-checkpoint generation A must match the cache-free reference exactly");
        drop(gen_a);
        let gen_b = ltxv::block::GenerationCache::for_checkpoint(&path);
        let before = gen_b.stats();
        assert_eq!(before.blocks, cfg.num_layers as usize, "a second generation's handle must resolve to the store the first one populated");
        let out_b = forward_q_streamed(&cfg, &src, &head, None, QTier::Int8, &inputs.latent, &inputs.timesteps, &inputs.positions, &inputs.keyframes_mask, &inputs.context, inputs.context_len, inputs.t, &inputs.context_valid, &gen_b);
        let after = gen_b.stats();
        assert_eq!(after.misses - before.misses, 0, "the second real generation must miss on no layer");
        assert_eq!(after.hits - before.hits, cfg.num_layers as u64, "the second real generation must hit on every layer, from its first");
        assert_eq!(max_abs_diff(&out_b, &out_fresh), 0.0, "the second real generation's output must be bit-identical to the cache-free reference");
        println!("real Q8_0 checkpoint: a SECOND generation's handle started warm ({} blocks, {:.2} GB) and hit every layer", before.blocks, before.bytes as f64 / 1e9);
    }

    let per_layer_bytes: Vec<usize> = shared.block_byte_lens();
    assert_eq!(per_layer_bytes.len(), cfg.num_layers as usize);
    let avg_bytes = per_layer_bytes.iter().sum::<usize>() as f64 / per_layer_bytes.len() as f64;
    let extrapolated_48_gb = avg_bytes * 48.0 / 1e9;
    println!("measured per-block host cache footprint: {:.1} MB/block (real 22B width) -> extrapolated 48-layer total: {extrapolated_48_gb:.2} GB", avg_bytes / 1e6);
    // Sanity band, not a tight bound: the real 22B config's own device-bytes
    // test (`device_bytes_real.rs`) measured a real (3.5, 4.0)x int8
    // compression ratio against the fp32 block size, never the flat
    // theoretical four-to-one - this cache stores the identical packed bytes host-side,
    // so its total footprint should land well under the ~42 GB bf16 model
    // size and comfortably within this class of hardware's RAM (184 GiB).
    assert!(extrapolated_48_gb > 5.0 && extrapolated_48_gb < 40.0, "extrapolated 48-layer cache footprint {extrapolated_48_gb:.2} GB is outside the sane band for an int8-quantized 22B model's block weights");
}
