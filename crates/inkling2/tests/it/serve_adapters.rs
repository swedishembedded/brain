// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements multi-tenant serving of fine-tuned mixture-of-experts models from one resident copy of
// their quantized expert banks for its clients. If your team needs expertise in per-request adapters over routed experts
// then you can procure our services by sending an email to info@swedishembedded.com.

//! Spec of the served engine's adapter bank (`Engine::enable_adapters`):
//!
//! - an adapter the trainer saved (rank 2 over every linear site and both stacked expert sites, over a release with
//!   published NVFP4 banks) is served with the logits of the trainer's own adapted forward at every position, prefilled
//!   whole or in chunks and decoded, beside a base sequence that keeps the base's logits in the same batches;
//! - the engine passes the generic serving conformance suite with adapters (no K/V shared across adapters and the base,
//!   embedded prompts under adapters, mixed batches);
//! - an adapter that carries a merged site (a matrix the forward reads whole) is refused by name, and so are expert
//!   targets over a checkpoint whose experts are dense.

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
/// Every site the engine serves: the linears and the two stacked expert banks.
const SERVED: [&str; 9] = ["wq_du", "wk_dv", "wv_dv", "wo_ud", "gate_attn", "attn_sink_proj", "wr_du", "w13_weight", "w2_weight"];
const RANK: u32 = 2;

fn options(chunk: u32) -> EngineOptions {
    EngineOptions { kv_tier: KvTier::F32, sizing: VisitPoolSizing { block_size: 8, num_blocks: 64, live_rows: 4, snapshot_rows: 4, undo_rows: false }, max_seq_len: 128, prefill_chunk: chunk }
}

fn engine(dir: &Path, chunk: u32) -> Engine {
    let mut e = Engine::open(crate::tiny_parity::ops(), dir, options(chunk)).expect("the release opens");
    e.enable_adapters(AdapterBankConfig::new(2, RANK, &SERVED)).expect("the bank fits");
    e
}

/// A trainer's LoRA build over `dir` with every served site, its adapter factors drawn from `seed` (B too, so the
/// adapter moves the logits), saved where the trainer saves an adapter. Returns the build and the file.
fn trained_adapter(dir: &Path, seed: u64, seq: u32, out: &Path) -> Inkling2Train {
    let lora = LoraCfg::new(RANK, 4.0, SERVED.iter().map(|s| s.to_string()).collect());
    let base = model::ops::TierPolicy::uniform(gpu_core::select::Dtype::F32);
    let t = Inkling2Train::from_dir_lora(dir, lora, &BuildOptions { batch: 1, seq, base: &base, seed, continue_from: None }).expect("a LoRA build over the release");
    let mut rng = data::rng::Lcg::new(seed);
    for name in t.param_names() {
        let n = t.read_weight(&name).len();
        t.write_weight(&name, &rng.vec_scaled(n, 0.3));
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
    let Some(dir) = crate::declared_tier::declared_release("adapters") else { return };
    let seq: Vec<u32> = (0..20u32).map(|i| (i * 17 + 4) % 60 + 3).collect();
    let prompt = 8;
    let scratch = brain_testutil::Scratch::dir("inkling2-served-adapter");
    let path = scratch.path().join("adapter.safetensors");
    // seed: any factors; both sides read the same file.
    let trainer = trained_adapter(&dir, 41, seq.len() as u32, &path);
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
fn merged_sites_and_expert_targets_over_dense_experts_are_refused_by_name() {
    let Some(dir) = crate::declared_tier::declared_release("adapter-refusals") else { return };
    // A trainer's adapter of the served sites, with a pair for a merged site added (a depth query of layer 1).
    let scratch = brain_testutil::Scratch::dir("inkling2-merged-adapter");
    let path = scratch.path().join("adapter.safetensors");
    trained_adapter(&dir, 5, 8, &path);
    let mut with_merged = verified(&path, 5);
    let d = inkling2::Inkling2Config::from_dir(&dir).unwrap().hidden as usize;
    let queries = inkling2::Inkling2Config::from_dir(&dir).unwrap().attn_queries(1) as usize;
    with_merged.adapter.sites.push(model::lora::device_adapter::AdapterSite { base: inkling2::names::attn_queries(1), a: vec![0.1; RANK as usize * d], b: vec![0.1; queries * RANK as usize] });
    let mut e = Engine::open(crate::tiny_parity::ops(), &dir, options(4)).unwrap();
    e.enable_adapters(AdapterBankConfig::new(2, RANK, &SERVED)).unwrap();
    let err = format!("{:?}", e.load_adapter(with_merged).unwrap_err());
    assert!(err.contains("attn_queries") && err.contains("merged site"), "{err}");
    std::fs::remove_dir_all(&dir).ok();
    // The tiny reference checkpoint stores its experts dense f32: no served expert product to correct.
    let dense = brain_testutil::testdata_path("golden/inkling2/tiny/checkpoint");
    let mut e = Engine::open(crate::tiny_parity::ops(), &dense, options(4)).unwrap();
    let err = format!("{:?}", e.enable_adapters(AdapterBankConfig::new(2, RANK, &["w13_weight"])).unwrap_err());
    assert!(err.contains("dense"), "{err}");
    e.enable_adapters(AdapterBankConfig::new(2, RANK, &["wq_du"])).expect("linear sites are served over dense experts");
}
