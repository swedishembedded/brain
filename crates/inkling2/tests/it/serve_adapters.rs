// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements multi-tenant serving of fine-tuned mixture-of-experts models from one resident copy of
// their quantized expert banks for its clients. If your team needs expertise in per-request adapters over routed experts
// then you can procure our services by sending an email to info@swedishembedded.com.

//! Spec of the served engine's adapter bank (`Engine::enable_adapters`):
//!
//! - an adapter the trainer saved (rank 2 over every site it trains: the linears, the message gates' factors, the
//!   depth, readout and message query banks and both stacked expert sites, over a release with published NVFP4 banks)
//!   is served with the logits of the trainer's own adapted forward at every position, prefilled whole or in chunks and
//!   decoded, beside a base sequence that keeps the base's logits in the same batches; so is one over the query banks
//!   and gates alone;
//! - the engine passes the generic serving conformance suite with adapters (no K/V shared across adapters and the base,
//!   embedded prompts under adapters, mixed batches);
//! - expert targets over a checkpoint whose experts are dense are refused by name.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use inkling2::serve::{AdapterBankConfig, Engine, EngineOptions, VerifiedAdapter};
use inkling2::train::BuildOptions;
use inkling2::Inkling2Train;
use model::adapter::device::LoraCfg;
use model::kv_tier::KvTier;
use model::paged::BlockTable;
use model::serve::conformance::{self, Conformance};
use model::serve::visit_state::VisitPoolSizing;
use model::serve::{AdapterAvailability, AdapterId, PagedDecoder, PrefillProgress};
use model::Model;

/// The f32 reference's own spread, relative to the tensor's scale; the trainer sums its corrections in its own order.
const TOL: f32 = 1e-4;
/// Every site the engine serves: every site the trainer adapts.
const SERVED: &[&str] = inkling2::train::sites::DEFAULT_TARGETS;
/// The sites the forward does not correct beside a linear product in training: the query banks and the gates' factors.
const MERGED: [&str; 8] = ["attn_queries", "mlp_queries", "output_query", "middle_cycle_cross_pass_queries", "middle_cycle_cross_pass_kv_queries", "down_weight", "forget_weight", "gain_weight"];
const RANK: u32 = 2;

fn options(chunk: u32) -> EngineOptions {
    EngineOptions { kv_tier: KvTier::F32, sizing: VisitPoolSizing { block_size: 8, num_blocks: 64, live_rows: 4, snapshot_rows: 4, undo_rows: false }, max_seq_len: 128, prefill_chunk: chunk }
}

fn engine(dir: &Path, chunk: u32) -> Engine {
    let mut e = Engine::open(crate::tiny_parity::ops(), dir, options(chunk)).expect("the release opens");
    e.enable_adapters(AdapterBankConfig::new(2, RANK, SERVED)).expect("the bank fits");
    e
}

/// A trainer's LoRA build over `dir` with every served site, its adapter factors drawn from `seed` (B too, so the
/// adapter moves the logits), saved where the trainer saves an adapter. Returns the build and the file.
fn trained_adapter(dir: &Path, seed: u64, seq: u32, out: &Path) -> Inkling2Train {
    trained_adapter_over(dir, SERVED, seed, seq, out)
}

/// [`trained_adapter`] over the sites `targets` names.
fn trained_adapter_over(dir: &Path, targets: &[&str], seed: u64, seq: u32, out: &Path) -> Inkling2Train {
    let lora = LoraCfg::new(RANK, 4.0, targets.iter().map(|s| s.to_string()).collect());
    let base = model::ops::TierPolicy::uniform(gpu_core::select::Dtype::F32);
    let t = Inkling2Train::from_dir_lora(dir, lora, &BuildOptions { batch: 1, seq, base: &base, seed, continue_from: None }).expect("a LoRA build over the release");
    let mut rng = data::rng::Lcg::new(seed);
    // Factors of a size training reaches: every site of every layer moves the logits well past the comparison's
    // tolerance, without driving the looped stack where f32 rounding differences between the two recipes grow.
    for name in t.param_names() {
        let n = t.read_weight(&name).len();
        t.write_weight(&name, &rng.vec_scaled(n, 0.15));
    }
    inkling2::finetune::save_adapter_file(&t, out).expect("the adapter saves");
    t
}

fn verified(path: &Path, seed: u64) -> VerifiedAdapter {
    let adapter = model::lora::device_adapter::read_adapter_with_experts(path.to_str().unwrap()).expect("the saved adapter reads with its expert pairs");
    VerifiedAdapter { label: format!("adapter-{seed}"), digest: format!("sha256:{seed:064x}"), adapter }
}

/// Make `id` resident and pin it, as the scheduler does before admitting a sequence under it.
fn admit(e: &mut Engine, id: AdapterId) {
    for _ in 0..1000 {
        match e.adapter_availability(id) {
            AdapterAvailability::Resident => {
                e.pin_adapter(id);
                return;
            }
            AdapterAvailability::Loading => {}
            AdapterAvailability::Unknown => panic!("adapter {id:?} is not served"),
        }
    }
    panic!("adapter {id:?} never became resident");
}

fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "length");
    let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) / scale
}

fn prefill(e: &mut Engine, table: &mut BlockTable, prompt: &[u32], slice: usize) -> Vec<f32> {
    let hidden = loop {
        if let PrefillProgress::Done(h) = e.prefill_chunk(table, prompt, slice) {
            break h;
        }
    };
    e.logits(&hidden)
}

#[test]
fn a_trained_adapter_is_served_with_the_trainers_adapted_logits_beside_the_base() {
    served_at_the_trainers_logits("adapters", SERVED);
}

#[test]
fn an_adapter_of_the_query_banks_and_gates_alone_is_served_with_the_trainers_adapted_logits() {
    served_at_the_trainers_logits("merged-adapters", &MERGED);
}

/// An adapter the trainer saved over `targets`, served beside the base: the trainer's adapted logits on its rows, the
/// base's on the base's, prefilled whole or in slices and decoded in batches mixing the two.
fn served_at_the_trainers_logits(tag: &str, targets: &[&str]) {
    let Some(dir) = crate::declared_tier::declared_release(tag) else { return };
    let seq: Vec<u32> = (0..20u32).map(|i| (i * 17 + 4) % 60 + 3).collect();
    let prompt = 8;
    let scratch = brain_testutil::Scratch::dir("inkling2-served-adapter");
    let path = scratch.path().join("adapter.safetensors");
    // seed: any factors; both sides read the same file.
    let trainer = trained_adapter_over(&dir, targets, 41, seq.len() as u32, &path);
    let want = trainer.logits(&seq).unwrap();
    drop(trainer);
    let mut e = engine(&dir, 4);
    let base = e.model().forward(&seq).unwrap();
    let id = e.load_adapter(verified(&path, 41)).expect("the trained adapter fits the bank");
    let v = e.vocab();
    for slice in [prompt, 3] {
        admit(&mut e, id);
        let (mut adapted, mut plain) = (BlockTable::with_namespace(id.kv_namespace()), BlockTable::new());
        let mut got = vec![prefill(&mut e, &mut adapted, &seq[..prompt], slice)];
        let mut got_base = vec![prefill(&mut e, &mut plain, &seq[..prompt], slice)];
        // Decoded together: one batch holds a row under the adapter and a row under the base.
        for &t in &seq[prompt..seq.len() - 1] {
            let rows = e.forward_batched_logit_rows(&mut [&mut adapted, &mut plain], &[t, t]);
            got.push(rows[0].clone());
            got_base.push(rows[1].clone());
        }
        for (i, (a, b)) in got.iter().zip(&got_base).enumerate() {
            let p = prompt - 1 + i;
            let err = rel_err(a, &want[p * v..(p + 1) * v]);
            assert!(err <= TOL, "slices of {slice}: position {p} under the adapter is {err:.3e} from the trainer's adapted forward");
            let err = rel_err(b, &base[p * v..(p + 1) * v]);
            assert!(err <= 5e-5, "slices of {slice}: the base row at position {p} is {err:.3e} from the base's forward");
        }
        assert!(rel_err(&got[0], &base[(prompt - 1) * v..prompt * v]) > 1e-3, "the adapter moves the logits");
        e.release_table(&mut adapted);
        e.release_table(&mut plain);
        e.unpin_adapter(id);
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Spec: an adapter the trainer saves over every site kind (linear, merged and stacked expert) reads back through the
/// reader a serving bank uses, every pair with the trainer's bits, under the card's declared targets.
#[test]
fn an_adapter_over_every_site_kind_reads_back_with_the_trainers_factors() {
    let Some(dir) = crate::declared_tier::declared_release("adapter-roundtrip") else { return };
    let scratch = brain_testutil::Scratch::dir("inkling2-adapter-roundtrip");
    let path = scratch.path().join("adapter.safetensors");
    let trainer = trained_adapter_over(&dir, inkling2::train::sites::DEFAULT_TARGETS, 7, 8, &path);
    let read = model::lora::device_adapter::read_adapter_with_experts(path.to_str().unwrap()).expect("the saved adapter reads back");
    let cfg = inkling2::Inkling2Config::from_dir(&dir).unwrap();
    let kinds = inkling2::train::sites::lora_sites(&cfg);
    assert!(kinds.iter().any(|(_, k)| *k == inkling2::train::sites::SiteKind::Merged) && kinds.iter().any(|(_, k)| *k == inkling2::train::sites::SiteKind::Linear));
    for (site, _) in &kinds {
        let pair = read.sites.iter().find(|p| p.base == site.name).unwrap_or_else(|| panic!("{} is not read back", site.name));
        assert_eq!(pair.a, trainer.read_weight(&format!("{}.lora_a", site.name)), "{} A", site.name);
        assert_eq!(pair.b, trainer.read_weight(&format!("{}.lora_b", site.name)), "{} B", site.name);
    }
    let experts = inkling2::train::sites::expert_sites(&cfg);
    assert_eq!(read.sites.len(), kinds.len(), "one linear pair per linear and merged site");
    assert_eq!(read.experts.len(), experts.len(), "one stacked pair per expert site");
    for site in &experts {
        let pair = read.experts.iter().find(|p| p.base == site.name).unwrap_or_else(|| panic!("{} is not read back", site.name));
        assert_eq!(pair.a, trainer.read_weight(&format!("{}.lora_a", site.name)), "{} A", site.name);
    }
    std::fs::remove_dir_all(&dir).ok();
}

static DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
static ADAPTER: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

fn conformance_engine() -> Engine {
    // The chunk the engine's own conformance run uses (`serve.rs`).
    engine(DIR.get().expect("written by the test"), 7)
}

/// A distinct adapter per call: the saved file's factors, each scaled by its own factor (a new digest, a new id).
fn load_adapter(e: &mut Engine) -> AdapterId {
    static NEXT: AtomicU64 = AtomicU64::new(100);
    let seed = NEXT.fetch_add(1, Ordering::SeqCst);
    let mut a = verified(ADAPTER.get().expect("written by the test"), seed);
    let scale = 0.5 + (seed % 7) as f32 * 0.1;
    for site in &mut a.adapter.sites {
        site.b.iter_mut().for_each(|x| *x *= scale);
    }
    for pair in &mut a.adapter.experts {
        pair.b.iter_mut().for_each(|x| *x *= scale);
    }
    e.load_adapter(a).unwrap()
}

#[test]
fn the_engine_with_adapters_passes_the_serving_conformance_suite() {
    let Some(dir) = crate::declared_tier::declared_release("adapter-conformance") else { return };
    let scratch = brain_testutil::Scratch::dir("inkling2-conformance-adapter");
    let path = scratch.path().join("adapter.safetensors");
    trained_adapter(&dir, 43, 32, &path);
    DIR.set(dir.clone()).expect("set once");
    ADAPTER.set(path).expect("set once");
    conformance::run_all(&Conformance { make: conformance_engine, prefix_reuse: true, prefix_prompt_tokens: 20, load_adapter: Some(load_adapter) });
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn expert_targets_over_dense_experts_are_refused_by_name() {
    // The tiny reference checkpoint stores its experts dense f32: no served expert product to correct.
    let dense = brain_testutil::testdata_path("golden/inkling2/tiny/checkpoint");
    if !dense.join("config.json").is_file() {
        brain_testutil::skip("testdata/golden/inkling2/tiny absent (make testdata/fetch, or run tools/goldens/inkling2_dump_reference.py tiny)");
        return;
    }
    let mut e = Engine::open(crate::tiny_parity::ops(), &dense, options(4)).unwrap();
    let err = format!("{:?}", e.enable_adapters(AdapterBankConfig::new(2, RANK, &["w13_weight"])).unwrap_err());
    assert!(err.contains("dense"), "{err}");
    e.enable_adapters(AdapterBankConfig::new(2, RANK, &["wq_du"])).expect("linear sites are served over dense experts");
}
