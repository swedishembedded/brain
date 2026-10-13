// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving many fine-tuned
// variants of one model from a single resident copy of its weights for its
// clients. If your team needs expertise in multi-tenant inference engines
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The serving engine's adapter bank: one resident base, many LoRA adapters.
//!
//! The spec, each requirement its own test:
//!
//! * a batch whose rows run under different adapters (and none) gives every
//!   row the logits of that row run alone through `Qwen::attach_adapter` with
//!   its adapter, at fp32 and int8 weights, fp32 and int8 KV;
//! * base rows in such a batch are bit-identical to the same batch on an
//!   engine with no bank, and a base-only batch costs exactly what it cost
//!   before the bank existed;
//! * K/V computed under one adapter is never served to another, or to the
//!   base, including after a slot is reused;
//! * the lifecycle: cold loads never block decode, an unload under a running
//!   sequence is deferred and completes on the last unpin, a cancel releases
//!   the pin, an unknown adapter is rejected, a resident adapter is not
//!   evicted between the availability answer and the pin, and an adapter that
//!   does not fit the bank is refused.

use gpu_core::Gpu;
use model::paged::BlockTable;
use model::serve::{AdapterAvailability, AdapterId, PagedDecoder, RejectReason, Request, SampleParams, Scheduler};
use qwen3::lora::{AdapterSite, DeviceAdapter};
use qwen3::serve::{AdapterBankConfig, AdapterError, Engine, LoraSite, UnloadStatus, VerifiedAdapter};
use qwen3::{Dtype, Qwen, QwenConfig, Shard};
use std::collections::HashMap;

const BLOCK: u32 = 4;
const R_MAX: u32 = 4;
/// (rank, seed) of the two adapters the mixed batches use.
const ADAPTERS: [(u32, u64); 2] = [(3, 101), (2, 202)];

fn parent() -> Gpu {
    gpu_core::testgpu::dev(&[])
}

#[derive(Clone)]
struct Setup {
    cfg: QwenConfig,
    w: HashMap<String, Vec<f32>>,
    weights_int8: bool,
    kv_int8: bool,
}

impl Setup {
    fn new(cfg: QwenConfig, weights_int8: bool, kv_int8: bool) -> Setup {
        let w = qwen3::init_weights(&cfg, 7);
        Setup { cfg, w, weights_int8, kv_int8 }
    }

    fn engine(&self, bank_slots: Option<u32>) -> Engine {
        let mut e = Engine::from_map_on(&parent(), self.cfg.clone(), &self.w, BLOCK, 96, 8, 8, 16, self.kv_int8, self.weights_int8);
        if let Some(slots) = bank_slots {
            e.enable_adapters(AdapterBankConfig::new(slots, R_MAX, &LoraSite::ALL)).unwrap();
        }
        e
    }
}

/// A synthetic adapter over every linear of every layer, big enough to move
/// the logits visibly.
fn adapter(cfg: &QwenConfig, rank: u32, seed: u64) -> DeviceAdapter {
    let (d, ff, hq, hkv) = (cfg.d_model as usize, cfg.d_ff as usize, cfg.q_dim() as usize, cfg.kv_dim() as usize);
    let leaves = [("attn.wq", hq, d), ("attn.wk", hkv, d), ("attn.wv", hkv, d), ("attn.wo", d, hq), ("mlp.gate", ff, d), ("mlp.up", ff, d), ("mlp.down", d, ff)];
    let mut rng = seed | 1;
    let mut next = move |amp: f32| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        ((rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * amp
    };
    let mut sites = Vec::new();
    for l in 0..cfg.n_layers {
        for (leaf, out, inn) in leaves {
            let a = (0..rank as usize * inn).map(|_| next(1.0 / (inn as f32).sqrt())).collect();
            let b = (0..out * rank as usize).map(|_| next(0.8)).collect();
            sites.push(AdapterSite { base: format!("blocks.{l}.{leaf}.weight"), a, b });
        }
    }
    DeviceAdapter { rank, alpha: 2.0 * rank as f32, sites }
}

fn verified(cfg: &QwenConfig, rank: u32, seed: u64) -> VerifiedAdapter {
    VerifiedAdapter { label: format!("adapter-{seed}"), digest: format!("sha256:{seed:064x}"), adapter: adapter(cfg, rank, seed) }
}

fn register(eng: &mut Engine, cfg: &QwenConfig, which: usize) -> AdapterId {
    let (rank, seed) = ADAPTERS[which];
    eng.load_adapter(verified(cfg, rank, seed)).unwrap()
}

/// Ask until the adapter is resident; the scheduler does the same every
/// iteration.
fn make_resident(eng: &mut Engine, id: AdapterId) {
    for _ in 0..10_000 {
        if eng.adapter_availability(id) == AdapterAvailability::Resident {
            return;
        }
    }
    panic!("adapter {} never became resident", id.kv_namespace());
}

/// Admit one sequence the way the scheduler does: availability, pin, table in
/// the adapter's namespace, prefill. Returns the table and the prompt's logits.
fn admit(eng: &mut Engine, adapter: Option<AdapterId>, prompt: &[u32]) -> (BlockTable, Vec<f32>) {
    let mut table = match adapter {
        Some(a) => {
            make_resident(eng, a);
            eng.pin_adapter(a);
            BlockTable::with_namespace(a.kv_namespace())
        }
        None => BlockTable::new(),
    };
    let hidden = PagedDecoder::prefill(eng, &mut table, prompt);
    let logits = PagedDecoder::logits(eng, &hidden);
    (table, logits)
}

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) }).0 as u32
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
    let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

fn prompt_of(row: usize, vocab: u32) -> Vec<u32> {
    (0..5 + row as u32).map(|j| (j * 7 + row as u32 * 3 + 1) % vocab).collect()
}

/// `[row][step]` logits: step 0 is the prompt's, step `s` the one after the
/// `s`-th token fed.
type Trajectory = Vec<Vec<Vec<f32>>>;

/// Run `rows` (each: which adapter, if any) together as one batch for `steps`
/// decode steps. `forced` feeds the given tokens instead of the batch's own
/// greedy ones, so two runs can be compared logit for logit. Returns the
/// trajectory and the tokens fed.
fn run_engine(s: &Setup, with_bank: bool, rows: &[(Option<usize>, usize)], steps: usize, forced: Option<&[Vec<u32>]>) -> (Trajectory, Vec<Vec<u32>>) {
    let mut eng = s.engine(with_bank.then_some(2));
    let ids: Vec<Option<AdapterId>> = if with_bank {
        let reg: Vec<AdapterId> = (0..2).map(|w| register(&mut eng, &s.cfg, w)).collect();
        rows.iter().map(|(r, _)| r.map(|w| reg[w])).collect()
    } else {
        vec![None; rows.len()]
    };
    let vocab = s.cfg.vocab as usize;
    let mut tables = Vec::new();
    let mut traj: Trajectory = vec![Vec::new(); rows.len()];
    let mut last = Vec::new();
    for (i, id) in ids.iter().enumerate() {
        let (t, logits) = admit(&mut eng, *id, &prompt_of(rows[i].1, s.cfg.vocab));
        last.push(argmax(&logits));
        traj[i].push(logits);
        tables.push(t);
    }
    let mut fed = vec![Vec::new(); rows.len()];
    for step in 0..steps {
        let inputs: Vec<u32> = (0..rows.len()).map(|i| forced.map_or(last[i], |f| f[i][step])).collect();
        let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
        let top = PagedDecoder::forward_batched_topk(&mut eng, &mut refs, &inputs, vocab);
        for (i, cand) in top.iter().enumerate() {
            let mut logits = vec![f32::NAN; vocab];
            for &(tok, l) in cand {
                logits[tok as usize] = l;
            }
            assert!(logits.iter().all(|l| !l.is_nan()), "top-k over the whole vocabulary must return every logit");
            last[i] = argmax(&logits);
            fed[i].push(inputs[i]);
            traj[i].push(logits);
        }
    }
    (traj, fed)
}

/// The same row run alone through `Qwen` with `attach_adapter_pairs`: the
/// single-adapter runtime path, the numerical oracle.
fn run_oracle(s: &Setup, which: Option<usize>, row: usize, fed: &[u32]) -> Vec<Vec<f32>> {
    let dt = if s.weights_int8 { Dtype::I8 } else { Dtype::F32 };
    let mut m = Qwen::new_shard_dt_decode(s.cfg.clone(), 64, &s.w, Shard::whole(s.cfg.n_layers as usize), dt);
    if let Some(w) = which {
        let (rank, seed) = ADAPTERS[w];
        m.attach_adapter_pairs(&adapter(&s.cfg, rank, seed)).unwrap();
    }
    let mut out = Vec::new();
    for &t in &prompt_of(row, s.cfg.vocab) {
        m.step(t);
    }
    out.push(m.decode_logits());
    for &t in fed {
        m.step(t);
        out.push(m.decode_logits());
    }
    out
}

const ROWS: [Option<usize>; 5] = [None, Some(0), Some(1), Some(0), None];
const STEPS: usize = 6;

/// `ROWS` with their prompt indices.
fn rows() -> Vec<(Option<usize>, usize)> {
    ROWS.iter().copied().enumerate().map(|(i, r)| (r, i)).collect()
}

/// Check every row of the mixed batch against the oracle (cosine floor
/// `oracle_cos`; the greedy token may differ from the oracle's only when the
/// oracle's own logits for the two are within `tie_eps` of the logit range,
/// i.e. a tie that the quantisation noise of int8 weights or an int8 KV
/// cache, which the oracle keeps in fp32, decides) and against the engine's own solo
/// run (near bit-exact: a mixed batch must not change what one row computes).
fn mixed_batch_matches_solo(s: &Setup, oracle_cos: f64, tie_eps: f32, tag: &str) {
    let mut min_cos = 1.0f64;
    let (mixed, fed) = run_engine(s, true, &rows(), STEPS, None);
    for (i, row) in ROWS.iter().enumerate() {
        // The oracle: the row alone, adapter attached at runtime.
        let want = run_oracle(s, *row, i, &fed[i]);
        for (step, (got, want)) in mixed[i].iter().zip(&want).enumerate() {
            let c = cosine(got, want);
            min_cos = min_cos.min(c);
            assert!(c >= oracle_cos, "{tag}: row {i} ({row:?}) step {step}: cosine {c} against Qwen::attach_adapter is under {oracle_cos}");
            let (tok, best) = (argmax(got), argmax(want));
            let range = want.iter().fold(f32::MIN, |m, &v| m.max(v)) - want.iter().fold(f32::MAX, |m, &v| m.min(v));
            assert!(tok == best || want[best as usize] - want[tok as usize] <= tie_eps * range, "{tag}: row {i} step {step}: greedy token {tok} differs from the solo oracle's {best} beyond a tie");
        }
        // The engine's own solo run of that row: a mixed batch must not
        // change what one row computes.
        let (solo, _) = run_engine(s, true, &[(*row, i)], STEPS, Some(&[fed[i].clone()]));
        for (step, (got, want)) in mixed[i].iter().zip(&solo[0]).enumerate() {
            let c = cosine(got, want);
            assert!(c >= 0.99999, "{tag}: row {i} step {step}: mixed vs solo engine cosine {c}");
            assert_eq!(argmax(got), argmax(want), "{tag}: row {i} step {step}: mixed vs solo engine token");
        }
    }
    eprintln!("{tag}: lowest per-row cosine against Qwen::attach_adapter: {min_cos}");
    // The adapters must move the model, or everything above is vacuous.
    let base = run_oracle(s, None, 1, &fed[1]);
    assert!(cosine(&mixed[1][0], &base[0]) < 0.9999, "{tag}: adapter A does not change the logits");
}

#[test]
fn mixed_batch_rows_equal_the_same_rows_run_alone_fp32() {
    mixed_batch_matches_solo(&Setup::new(QwenConfig::tiny(), false, false), 0.9999, 0.0, "fp32 weights, fp32 kv");
}

#[test]
fn mixed_batch_rows_equal_the_same_rows_run_alone_fp32_weights_int8_kv() {
    mixed_batch_matches_solo(&Setup::new(QwenConfig::tiny(), false, true), 0.999, 0.01, "fp32 weights, int8 kv");
}

#[test]
fn mixed_batch_rows_equal_the_same_rows_run_alone_int8_weights() {
    mixed_batch_matches_solo(&Setup::new(QwenConfig::tiny_i8(), true, false), 0.9999, 0.01, "int8 weights, fp32 kv");
}

#[test]
fn mixed_batch_rows_equal_the_same_rows_run_alone_int8_weights_int8_kv() {
    mixed_batch_matches_solo(&Setup::new(QwenConfig::tiny_i8(), true, true), 0.995, 0.02, "int8 weights, int8 kv");
}

fn bits_of(l: &[f32]) -> Vec<u32> {
    l.iter().map(|v| v.to_bits()).collect()
}

fn bits(t: &[Vec<f32>]) -> Vec<Vec<u32>> {
    t.iter().map(|l| bits_of(l)).collect()
}

/// Base rows of a mixed batch are the rows of an engine that has no bank.
fn base_rows_are_bit_identical(s: &Setup, tag: &str) {
    let (mixed, fed) = run_engine(s, true, &rows(), STEPS, None);
    // The reference batch has the same shape and the same base rows; the
    // other rows run the base too, which base rows must not be able to see.
    let (plain, _) = run_engine(s, false, &rows(), STEPS, Some(&fed));
    for i in [0, 4] {
        assert_eq!(bits(&mixed[i]), bits(&plain[i]), "{tag}: base row {i} of a mixed batch differs from an engine with no bank");
    }
    // A batch of only base rows on a bank engine takes the base tape.
    let base_only: Vec<(Option<usize>, usize)> = (0..5).map(|i| (None, i)).collect();
    let (with_bank, fed2) = run_engine(s, true, &base_only, STEPS, None);
    let (without, _) = run_engine(s, false, &base_only, STEPS, Some(&fed2));
    for i in 0..5 {
        assert_eq!(bits(&with_bank[i]), bits(&without[i]), "{tag}: base-only row {i} differs between a bank engine and one without");
    }
}

#[test]
fn base_rows_in_a_mixed_batch_are_bit_identical_fp32() {
    base_rows_are_bit_identical(&Setup::new(QwenConfig::tiny(), false, false), "fp32");
}

#[test]
fn base_rows_in_a_mixed_batch_are_bit_identical_int8() {
    base_rows_are_bit_identical(&Setup::new(QwenConfig::tiny_i8(), true, true), "int8");
}

fn dispatches(e: &Engine) -> u64 {
    e.device_stats().expect("device stats").dispatches
}

/// A base-only decode step on an engine with a bank records the base tape:
/// the same number of dispatches as an engine without one. An adapter step
/// adds exactly the two bank dispatches per targeted linear.
#[test]
fn a_base_only_step_costs_what_it_cost_without_a_bank_and_an_adapter_step_adds_two_per_site() {
    let s = Setup::new(QwenConfig::tiny(), false, false);
    let step_cost = |eng: &mut Engine, id: Option<AdapterId>| {
        let (mut t, _) = admit(eng, id, &prompt_of(0, s.cfg.vocab));
        // Warm: the first step at a key records and caches its tape.
        PagedDecoder::forward_batched_greedy(eng, &mut [&mut t], &[3]);
        let before = dispatches(eng);
        PagedDecoder::forward_batched_greedy(eng, &mut [&mut t], &[3]);
        dispatches(eng) - before
    };
    let mut plain = s.engine(None);
    let mut banked = s.engine(Some(2));
    let a = register(&mut banked, &s.cfg, 0);
    let base = step_cost(&mut plain, None);
    assert_eq!(step_cost(&mut banked, None), base, "a base-only step must record the unchanged base tape");
    let with_adapter = step_cost(&mut banked, Some(a));
    assert_eq!(with_adapter, base + 2 * 7 * s.cfg.n_layers as u64, "an adapter step adds two dispatches per targeted linear per layer, however many adapters the batch holds");
}

fn prefix_hits(e: &Engine) -> u64 {
    e.prefix_stats().0
}

/// K/V computed under one adapter is invisible to every other namespace, in
/// every order, and a slot handed to a new adapter inherits nothing.
#[test]
fn prefix_cache_never_crosses_base_and_adapters_including_after_slot_reuse() {
    let s = Setup::new(QwenConfig::tiny(), false, false);
    let prompt: Vec<u32> = (0..9).map(|j| (j * 5 + 2) % s.cfg.vocab).collect(); // two full blocks of 4
    let orders: [[Option<usize>; 3]; 3] = [[None, Some(0), Some(1)], [Some(1), None, Some(0)], [Some(0), Some(1), None]];
    // Each namespace's cold result, from an engine that has seen nothing else.
    let cold = |who: Option<usize>| {
        let mut e = s.engine(Some(2));
        let ids: Vec<AdapterId> = (0..2).map(|w| register(&mut e, &s.cfg, w)).collect();
        let (_, l) = admit(&mut e, who.map(|w| ids[w]), &prompt);
        l
    };
    let colds = [cold(None), cold(Some(0)), cold(Some(1))];
    for order in orders {
        let mut e = s.engine(Some(2));
        let ids: Vec<AdapterId> = (0..2).map(|w| register(&mut e, &s.cfg, w)).collect();
        for who in order {
            let before = prefix_hits(&e);
            let (mut t, l) = admit(&mut e, who.map(|w| ids[w]), &prompt);
            assert_eq!(prefix_hits(&e), before, "{who:?} in order {order:?} adopted blocks cached under another namespace");
            assert_eq!(bits_of(&l), bits_of(&colds[who.map_or(0, |w| w + 1)]), "{who:?} in order {order:?}: result differs from its cold run");
            e.release_table(&mut t);
        }
        // Second pass: now every namespace has its own cache and must hit it, and only it.
        for who in order {
            let before = prefix_hits(&e);
            let (mut t, l) = admit(&mut e, who.map(|w| ids[w]), &prompt);
            assert_eq!(prefix_hits(&e) - before, 8, "{who:?} in order {order:?} should reuse its own two blocks");
            assert_eq!(bits_of(&l), bits_of(&colds[who.map_or(0, |w| w + 1)]), "{who:?}: warm differs from cold");
            e.release_table(&mut t);
        }
    }

    // Slot reuse: one slot, adapter A cached, A unloaded, a different adapter
    // takes the slot. It must find nothing, and match its own cold result.
    let mut e = s.engine(Some(1));
    let a = register(&mut e, &s.cfg, 0);
    let (mut t, _) = admit(&mut e, Some(a), &prompt);
    e.release_table(&mut t);
    e.unpin_adapter(a);
    assert_eq!(e.unload_adapter(a), Ok(UnloadStatus::Unloaded));
    let b = register(&mut e, &s.cfg, 1);
    assert_ne!(a, b, "a namespace is never reused");
    let before = prefix_hits(&e);
    let (_, l) = admit(&mut e, Some(b), &prompt);
    assert_eq!(prefix_hits(&e), before, "the reused slot's new adapter adopted the old adapter's blocks");
    assert_eq!(bits_of(&l), bits_of(&colds[2]), "adapter B in A's old slot differs from its cold result");
}

/// An adapter evicted from the bank to make room keeps its cached KV: the
/// namespace belongs to the adapter, not to the slot.
#[test]
fn eviction_from_a_slot_keeps_the_adapters_cache_and_reload_is_exact() {
    let s = Setup::new(QwenConfig::tiny(), false, false);
    let prompt: Vec<u32> = (0..9).map(|j| (j * 5 + 2) % s.cfg.vocab).collect();
    let mut e = s.engine(Some(1));
    let (a, b) = (register(&mut e, &s.cfg, 0), register(&mut e, &s.cfg, 1));
    let (mut t, first) = admit(&mut e, Some(a), &prompt);
    e.release_table(&mut t);
    e.unpin_adapter(a);
    // B takes the only slot, evicting A.
    let (mut tb, _) = admit(&mut e, Some(b), &prompt);
    assert!(!e.adapters().iter().find(|i| i.id == a).unwrap().resident);
    e.release_table(&mut tb);
    e.unpin_adapter(b);
    let before = prefix_hits(&e);
    let (_, again) = admit(&mut e, Some(a), &prompt);
    assert_eq!(prefix_hits(&e) - before, 8, "A's cached blocks survive its eviction from the slot");
    // Reloaded into the (same) slot, A computes exactly what it did.
    assert_eq!(bits_of(&again), bits_of(&first));
}

#[test]
fn lifecycle_unload_is_deferred_while_running_and_cancel_releases_the_pin() {
    let s = Setup::new(QwenConfig::tiny(), false, false);
    let mut e = s.engine(Some(2));
    let a = register(&mut e, &s.cfg, 0);
    let prompt = prompt_of(0, s.cfg.vocab);
    let (mut t, _) = admit(&mut e, Some(a), &prompt);
    assert_eq!(e.adapters()[0].pins, 1);

    // Unload under a running sequence: no new admissions, still serving.
    assert_eq!(e.unload_adapter(a), Ok(UnloadStatus::Draining));
    assert!(e.adapters()[0].draining);
    assert_eq!(e.adapter_availability(a), AdapterAvailability::Unknown, "a draining adapter takes no new sequences");
    let tok = PagedDecoder::forward_batched_greedy(&mut e, &mut [&mut t], &[3]);
    assert_eq!(tok.len(), 1, "the running sequence still decodes");
    assert_eq!(e.adapters().len(), 1);

    // The scheduler's cancel path: release the table, then unpin.
    e.release_table(&mut t);
    e.unpin_adapter(a);
    assert!(e.adapters().is_empty(), "the last unpin completes the deferred unload");
    assert_eq!(e.adapter_availability(a), AdapterAvailability::Unknown);
    assert_eq!(e.unload_adapter(a), Err(AdapterError::Unknown(a)));
}

#[test]
fn an_adapter_is_not_evicted_between_the_availability_answer_and_the_pin() {
    let s = Setup::new(QwenConfig::tiny(), false, false);
    let mut e = s.engine(Some(1));
    let (a, b) = (register(&mut e, &s.cfg, 0), register(&mut e, &s.cfg, 1));
    make_resident(&mut e, a);
    // A has answered Resident and is not pinned yet. A new unload of it must wait for the pin.
    assert_eq!(e.unload_adapter(a), Ok(UnloadStatus::Draining), "an offered adapter counts as held");
    e.pin_adapter(a);
    e.unpin_adapter(a);
    assert!(e.adapters().iter().all(|i| i.id != a), "the deferred unload completes");
    // B can now have the slot.
    make_resident(&mut e, b);
}

#[test]
fn a_full_bank_makes_a_cold_adapter_wait_without_blocking_and_lru_picks_the_victim() {
    let s = Setup::new(QwenConfig::tiny(), false, false);
    let mut e = s.engine(Some(2));
    let ids: Vec<AdapterId> = [(2, 301u64), (2, 302), (2, 303)].iter().map(|&(r, sd)| e.load_adapter(verified(&s.cfg, r, sd)).unwrap()).collect();
    make_resident(&mut e, ids[0]);
    e.pin_adapter(ids[0]);
    make_resident(&mut e, ids[1]);
    e.pin_adapter(ids[1]);
    // Both slots pinned: the third waits (Loading, immediately), it is not an error.
    assert_eq!(e.adapter_availability(ids[2]), AdapterAvailability::Loading);
    // Releasing the older pin frees the older one, not the newer.
    e.unpin_adapter(ids[1]);
    e.unpin_adapter(ids[0]);
    make_resident(&mut e, ids[2]);
    let resident: Vec<bool> = e.adapters().iter().map(|i| i.resident).collect();
    assert_eq!(resident, [false, true, true], "the least recently used unpinned adapter is evicted");
}

#[test]
fn a_cold_load_is_spread_over_calls_and_unknown_or_unfit_adapters_are_refused() {
    let s = Setup::new(QwenConfig::tiny(), false, false);
    let mut e = s.engine(None);
    assert_eq!(e.load_adapter(verified(&s.cfg, 2, 1)).err(), Some(AdapterError::NoBank));
    assert_eq!(e.adapter_availability(AdapterId::new(9).unwrap()), AdapterAvailability::Unknown);

    let mut cfg = AdapterBankConfig::new(2, R_MAX, &LoraSite::ATTENTION);
    cfg.upload_quantum_bytes = 1; // one linear per call
    e.enable_adapters(cfg).unwrap();
    assert_eq!(e.adapter_availability(AdapterId::new(9).unwrap()), AdapterAvailability::Unknown, "an unregistered adapter is rejected, not waited for");

    // Rank above r_max, and a site the bank does not target.
    assert!(matches!(e.load_adapter(verified(&s.cfg, R_MAX + 1, 5)), Err(AdapterError::Incompatible(_))));
    assert!(matches!(e.load_adapter(verified(&s.cfg, 2, 6)), Err(AdapterError::Incompatible(why)) if why.contains("mlp")), "an adapter with MLP sites does not fit an attention-only bank");
    let mut attn_only = adapter(&s.cfg, 2, 7);
    attn_only.sites.retain(|st| st.base.contains("attn"));
    let id = e.load_adapter(VerifiedAdapter { label: "attn".into(), digest: "sha256:attn".into(), adapter: attn_only }).unwrap();
    assert_eq!(e.load_adapter(VerifiedAdapter { label: "again".into(), digest: "sha256:attn".into(), adapter: adapter(&s.cfg, 2, 7) }).unwrap(), id, "the same digest is the same adapter");

    let mut calls = 1;
    while e.adapter_availability(id) == AdapterAvailability::Loading {
        calls += 1;
        assert!(calls < 1000);
    }
    assert!(calls > 1, "a cold load must not complete inside one availability call when the quantum is one linear");
    assert!(calls <= s.cfg.n_layers as usize * 4 + 1, "one linear per call at most");
}

/// The whole path through the scheduler: base and adapter requests share
/// steps, adapter requests produce exactly what the solo oracle produces,
/// an unknown adapter is rejected in the step report, and a cold adapter
/// does not hold up base traffic.
#[test]
fn the_scheduler_serves_base_and_adapter_requests_together() {
    let s = Setup::new(QwenConfig::tiny(), false, false);
    // A slow loader (one linear per availability call) keeps the adapter cold
    // for several scheduler iterations.
    let mut cold = AdapterBankConfig::new(2, R_MAX, &LoraSite::ALL);
    cold.upload_quantum_bytes = 1;
    let mut e = s.engine(None);
    e.enable_adapters(cold).unwrap();
    let a = register(&mut e, &s.cfg, 0);
    let mut sched = Scheduler::new(e, 4);
    let req = |row: usize| Request { prompt: prompt_of(row, s.cfg.vocab), max_new: 6, stop: vec![] };
    let adapter_req = sched.submit_for_adapter(req(1), a, SampleParams::greedy(), 0);
    let ghost = sched.submit_for_adapter(req(2), AdapterId::new(77).unwrap(), SampleParams::greedy(), 0);
    let base_req = sched.submit(req(0));

    let first = sched.step_report();
    assert!(first.admitted.contains(&base_req), "base traffic is admitted while the adapter is still loading");
    assert!(!first.admitted.contains(&adapter_req));
    assert!(first.rejected.iter().any(|(id, why)| *id == ghost && matches!(why, RejectReason::AdapterUnavailable { .. })), "an unknown adapter is rejected, not waited for");
    let out = sched.run();

    assert_eq!(out[&adapter_req], run_oracle_greedy(&s, Some(0), 1, 6), "the adapter request equals the solo oracle");
    assert_eq!(out[&base_req], run_oracle_greedy(&s, None, 0, 6), "the base request equals the solo base");
    let e = sched.into_decoder();
    assert_eq!(e.adapters()[0].pins, 0, "every pin is released when the requests finish");
}

fn run_oracle_greedy(s: &Setup, which: Option<usize>, row: usize, n: usize) -> Vec<u32> {
    let dt = if s.weights_int8 { Dtype::I8 } else { Dtype::F32 };
    let mut m = Qwen::new_shard_dt_decode(s.cfg.clone(), 64, &s.w, Shard::whole(s.cfg.n_layers as usize), dt);
    if let Some(w) = which {
        let (rank, seed) = ADAPTERS[w];
        m.attach_adapter_pairs(&adapter(&s.cfg, rank, seed)).unwrap();
    }
    for &t in &prompt_of(row, s.cfg.vocab) {
        m.step(t);
    }
    let mut out = vec![argmax(&m.decode_logits())];
    while out.len() < n {
        m.step(*out.last().unwrap());
        out.push(argmax(&m.decode_logits()));
    }
    out
}
