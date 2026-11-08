// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The whole 80-layer release against the official reference: how far each weight tier, KV format and sparse budget
//! moves the next-token distribution away from the bf16 model's.
//!
//! Swedish Embedded AB implements solutions for measuring what quantization and sparse attention cost a deployed
//! language model in output quality, on the real weights and not on a proxy, for its clients. If your team needs
//! expertise in validating a long-context serving stack against its reference then you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! The reference is `tools/goldens/ricursive_dump_reference.py full`: the official `modeling_ricursive.py` run layer
//! by layer over the sharded bf16 release (never more than one layer resident), its last `rows` logits rows written
//! under `golden/ricursive/full/`. The engine is replayed the same way those rows arise (a prefill that ends on the
//! first of them, then teacher-forced decode steps), at every combination below, and each row is scored by
//! [`quality`]: the mean KL divergence of the reference's next-token distribution from the engine's and the share of
//! positions whose top-1 token is the same.
//!
//! Inputs under the testdata root: `ricursive/ripple-65b-bf16/` and `ricursive/ripple-65b-nvfp4/` (the two releases,
//! each with `config.json` and `model.safetensors.index.json`; a link is enough), optionally
//! `ricursive/ripple-lsa.safetensors` (the distilled indexer, `brain lsa distill`). Absent inputs skip by name.
//! The engine is placed by [`ricursive::placement::auto_plan`] on whatever `BRAIN_DEVICE` selects, so the same test
//! covers one card and a staged placement (`gpu0,gpu1`).
//!
//! The first test needs nothing and runs everywhere: it holds the replay and its alignment with the reference rows to
//! the tiny model, where the whole-model forward is the reference.

use std::path::Path;
use std::time::Instant;

use model::kv_tier::KvTier;
use model::lsa::indexer::Indexer;
use model::paged::BlockTable;
use model::serve::PagedDecoder;
use ricursive::placement::auto_plan;
use ricursive::{open_shared_checkpoint, resolve_tiers, LsaCharge, RicursiveConfig, RicursiveEngine, RicursiveModel, ServeConfig};

use crate::serve::{prefill_in_slices, serve_config, shared_weights, stream, tiny_config, tiny_dir, weights};
use crate::taps::Taps;
use crate::tiers::{quality, Trace};

const DUMPER: &str = "tools/goldens/ricursive_dump_reference.py full";

/// Prefill `ids` up to the first of the last `rows` positions, then decode the rest teacher-forced (each input the
/// prompt's own next token): one logits row per position of the tail, the order of the reference's `logits`.
pub(crate) fn replay_tail(engine: &mut RicursiveEngine, ids: &[u32], rows: usize, prefill_chunk: usize) -> Trace {
    assert!(rows >= 1 && rows <= ids.len(), "{rows} rows of a {}-token prompt", ids.len());
    let prefill = ids.len() + 1 - rows;
    let mut table = BlockTable::new();
    let hidden = prefill_in_slices(engine, &mut table, &ids[..prefill], prefill_chunk);
    let mut out = vec![engine.logits(&hidden)];
    for &token in &ids[prefill..] {
        out.push(engine.decode_logits(&mut [&mut table], &[token]).remove(0));
    }
    engine.release_table(&mut table);
    out
}

/// A flat `[rows, vocab]` array as one vector per row.
pub(crate) fn split_rows(flat: &[f32], vocab: usize) -> Trace {
    assert_eq!(flat.len() % vocab, 0, "{} values are not whole rows of {vocab}", flat.len());
    flat.chunks(vocab).map(<[f32]>::to_vec).collect()
}

/// `max|got - want|` over `max|want|` across all rows.
fn scaled(reference: &Trace, got: &Trace) -> f32 {
    let (mut diff, mut scale) = (0.0f32, f32::MIN_POSITIVE);
    for (r, g) in reference.iter().zip(got) {
        for (a, b) in r.iter().zip(g) {
            diff = diff.max((a - b).abs());
            scale = scale.max(a.abs());
        }
    }
    diff / scale
}

/// The numbers the roadmap ledger records for one configuration; the bounds are sanity floors, to be tightened from
/// the first card run (no 80-layer number exists yet).
fn score(label: &str, reference: &Trace, got: &Trace, seconds: f64) {
    assert_eq!(reference.len(), got.len());
    assert!(got.iter().flatten().all(|v| v.is_finite()), "{label}: non-finite logits");
    let (kl, top1) = quality(reference, got);
    eprintln!("ripple-full {label}: mean KL {kl:.3e}, top-1 {:.1}% over {} positions, worst scaled logits error {:.3e}, replay {seconds:.1}s", top1 * 100.0, reference.len(), scaled(reference, got));
    assert!(kl < 1.0 && top1 >= 0.5, "{label}: the engine left the reference (KL {kl:.3e}, top-1 {top1})");
}

/// Spec: the replay produces the rows the reference dump holds. On the tiny model the whole-model forward is the
/// reference: its last `rows` rows, against the f32 engine replayed over the same prompt in two prefill chunk sizes.
#[test]
fn the_replay_reproduces_the_last_rows_of_the_full_forward() {
    let Some(dir) = tiny_dir() else { return };
    let cfg = tiny_config(&dir);
    let vocab = cfg.vocab_size;
    let ids = stream(11, 24, vocab as u32);
    let rows = 9;
    let forward = RicursiveModel::new(cfg.clone(), &weights(&dir, &cfg), None).forward(&ids).expect("forward").logits;
    let reference = split_rows(&forward[(ids.len() - rows) * vocab..], vocab);
    for chunk in [5, 16] {
        let mut engine = RicursiveEngine::from_source(cfg.clone(), shared_weights(&dir, &cfg), serve_config(chunk as u32, 0), None).expect("engine");
        let got = replay_tail(&mut engine, &ids, rows, chunk);
        assert_eq!(got.len(), rows);
        let worst = scaled(&reference, &got);
        eprintln!("replay of the last {rows} rows, prefill chunk {chunk}: worst scaled error {worst:.3e}");
        assert!(worst <= 1e-5, "chunk {chunk}: {worst:.3e}");
    }
    let (kl, top1) = quality(&reference, &reference);
    assert_eq!((kl, top1), (0.0, 1.0), "a trace against itself");

    // The score of a compact cache is a real, finite distance from the reference, not zero.
    let mut int8 = RicursiveEngine::from_source(cfg.clone(), shared_weights(&dir, &cfg), ServeConfig { kv_cache: KvTier::INT8, ..serve_config(5, 0) }, None).expect("int8 engine");
    let got = replay_tail(&mut int8, &ids, rows, 5);
    score("tiny, int8 KV", &reference, &got, 0.0);
    assert!(scaled(&reference, &got) > 1e-6, "int8 KV answered as f32 exactly: the tier is not in use");
}

/// A release directory under the testdata root, if it holds a sharded checkpoint.
fn release(name: &str) -> Option<String> {
    let dir = brain_testutil::testdata(&format!("ricursive/{name}"));
    Path::new(&dir).join("model.safetensors.index.json").exists().then_some(dir)
}

/// The engine of `release` at `serve`, placed over the selected cards, with the seconds it took to be ready.
fn open(dir: &str, cfg: &RicursiveConfig, serve: ServeConfig, indexer: Option<&Indexer>) -> (RicursiveEngine, f64) {
    let started = Instant::now();
    let layers: Vec<usize> = (0..cfg.num_hidden_layers).collect();
    let src = open_shared_checkpoint(dir, cfg, &layers).unwrap_or_else(|e| panic!("{dir}: {e}"));
    let tiers = resolve_tiers(cfg, src.as_ref(), serve.weights.as_ref());
    let charge = indexer.map_or_else(LsaCharge::default, |ix| LsaCharge::of(ix, &serve));
    let plan = auto_plan(cfg, &serve, &tiers, charge, None).unwrap_or_else(|e| panic!("{dir}: {e}"));
    eprintln!("ripple-full: placement {}", plan.describe());
    let mut engine = RicursiveEngine::from_source_staged(cfg.clone(), src, serve, None, &plan).unwrap_or_else(|e| panic!("{dir}: {e}"));
    if let Some(ix) = indexer {
        engine.set_lsa(ix).expect("set_lsa");
        engine.set_lsa_dense_until(0);
    }
    (engine, started.elapsed().as_secs_f64())
}

/// Spec (GPU phase): over the 80 layers of the real release, the
/// bf16 weights and the NVFP4 release each answer within quantization noise of the official reference at a bf16, int8
/// and fp8 KV cache, and the distilled indexer at its budget and at a quarter of it costs retention, not coherence.
#[test]
fn the_80_layer_releases_stay_close_to_the_official_reference_at_every_kv_format_and_budget() {
    let golden = brain_testutil::testdata("golden/ricursive/full");
    let Some(source) = brain_testutil::golden::Source::open(Path::new(&golden), DUMPER) else { return };
    let releases: Vec<(&str, String)> = ["ripple-65b-bf16", "ripple-65b-nvfp4"].into_iter().filter_map(|n| release(n).map(|d| (n, d))).collect();
    if releases.is_empty() {
        brain_testutil::skip("no release under testdata/ricursive/ripple-65b-{bf16,nvfp4} (a link to a directory with model.safetensors.index.json)");
        return;
    }
    let taps = Taps::read(&format!("{golden}/ricursive_full.safetensors"));
    let ids = taps.input_ids();
    let indexer = Path::new(&brain_testutil::testdata("ricursive/ripple-lsa.safetensors")).exists().then(|| Indexer::load(&brain_testutil::testdata("ricursive/ripple-lsa.safetensors")).expect("indexer"));
    for (name, dir) in releases {
        let cfg = RicursiveConfig::from_file(&format!("{dir}/config.json")).expect("config");
        if !source.require(&[("hidden_size", cfg.hidden_size as i64), ("num_hidden_layers", cfg.num_hidden_layers as i64), ("vocab_size", cfg.vocab_size as i64), ("sliding_window", cfg.sliding_window as i64)]) {
            return;
        }
        let reference = split_rows(taps.get("logits"), cfg.vocab_size);
        let rows = reference.len();
        let block = 64u32;
        let sizing = |kv: KvTier, sparse: bool| ServeConfig { kv_cache: kv, sparse_attention: sparse, ..ServeConfig::new(block, (ids.len() as u32).div_ceil(block) + 8, ids.len() as u32 + 8, 1, 0, 512) };
        for (label, kv) in [("bf16 KV", KvTier::BF16), ("int8 KV", KvTier::INT8), ("fp8 KV", KvTier::FP8)] {
            let (mut engine, ready) = open(&dir, &cfg, sizing(kv, false), None);
            eprintln!("ripple-full {name}, {label}: load-to-ready {ready:.1}s");
            let started = Instant::now();
            let got = replay_tail(&mut engine, &ids, rows, 512);
            score(&format!("{name}, {label}, dense"), &reference, &got, started.elapsed().as_secs_f64());
        }
        let Some(ix) = &indexer else {
            brain_testutil::skip("no testdata/ricursive/ripple-lsa.safetensors: the sparse budgets of this release are not measured");
            continue;
        };
        for divisor in [1usize, 4] {
            let mut budget = ix.clone();
            budget.cfg.top_k = (ix.cfg.top_k / divisor).max(1);
            budget.cfg.top_p = (ix.cfg.top_p / divisor).max(1);
            let (mut engine, ready) = open(&dir, &cfg, sizing(KvTier::INT8, true), Some(&budget));
            let started = Instant::now();
            let got = replay_tail(&mut engine, &ids, rows, 512);
            eprintln!("ripple-full {name}: load-to-ready with the index {ready:.1}s");
            score(&format!("{name}, int8 KV, top_k {} top_p {}", budget.cfg.top_k, budget.cfg.top_p), &reference, &got, started.elapsed().as_secs_f64());
        }
    }
}
