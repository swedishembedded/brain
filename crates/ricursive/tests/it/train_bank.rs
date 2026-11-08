// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Many adapters over one resident Ripple: the engine's adapter bank corrects each row of a batch by that row's own
//! adapter (none for a base row), on one or several stages.
//!
//! Swedish Embedded AB implements solutions for serving many fine-tuned variants of one very large model from a
//! single resident copy of its weights. If your team needs expertise in multi-tenant inference of fine-tuned
//! language models then you can procure our services by sending an email to info@swedishembedded.com.

use model::adapter::bank::{AdapterBankConfig, AdapterError, UnloadStatus, VerifiedAdapter};
use model::adapter::device::LoraCfg;
use model::paged::BlockTable;
use model::serve::{AdapterAvailability, AdapterId, PagedDecoder};
use model::{Batch, Model};
use optim::Adam;
use ricursive::train::{adapter, default_targets, AdapterInit};
use ricursive::{open_safetensors, RicursiveEngine, Ripple, TrainConfig, TrainOpts};

use crate::serve::{prefill_in_slices, serve_config, stream};
use crate::synthetic::{synthetic, Model as Synthetic, Shape};
use crate::taps::scaled_error;

/// A trainer over the synthetic model with an adapter trained a few steps on `seed`'s stream, and its adapter file.
fn trained(model: &Synthetic, seed: u32, steps: u32, dir: &std::path::Path) -> (Ripple, VerifiedAdapter) {
    let layers: Vec<usize> = (0..model.cfg.num_hidden_layers).collect();
    let vocab = model.cfg.vocab_size as u32;
    let base = open_safetensors(&model.path(), &model.cfg, &layers).unwrap();
    let cfg = TrainConfig::new(model.cfg.clone(), 16).with_lora(LoraCfg::new(2, 8.0, default_targets()));
    let init = AdapterInit::fresh(&cfg, &base, u64::from(seed));
    let trainer = Ripple::build(cfg, 1, 16, &init, &TrainOpts::default(), true).unwrap();
    if steps > 0 {
        let tokens = stream(seed, 12, vocab);
        let targets: Vec<u32> = tokens[1..].iter().copied().chain([ricursive::train::IGNORE]).collect();
        // The batch is 12 tokens in a model sized for 16: pad both with unsupervised positions.
        let (mut x, mut y) = (tokens, targets);
        x.resize(16, 0);
        y.resize(16, ricursive::train::IGNORE);
        trainer.set_batch(Batch::Lm { tokens: &x, targets: &y });
        for step in 1..=steps {
            trainer.zero_grads();
            trainer.forward();
            trainer.backward();
            trainer.adamw_step(step, 3e-2, 0.0, Adam::default(), None, 1.0);
        }
    }
    let file = dir.join(format!("adapter-{seed}.safetensors"));
    trainer.save_adapter(file.to_str().unwrap(), &format!("test/adapter-{seed}"), "test/base", None, None).unwrap();
    let verified = VerifiedAdapter { label: format!("test/adapter-{seed}"), digest: format!("sha256:{seed}"), adapter: adapter::read_adapter(file.to_str().unwrap()).unwrap() };
    (trainer, verified)
}

fn last(logits: &[f32], vocab: usize) -> Vec<f32> {
    logits[logits.len() - vocab..].to_vec()
}

/// Make `id` resident and pinned, as the scheduler does before it admits a sequence.
fn admit(engine: &mut RicursiveEngine, id: AdapterId) {
    for _ in 0..64 {
        if engine.adapter_availability(id) == AdapterAvailability::Resident {
            engine.pin_adapter(id);
            return;
        }
    }
    panic!("adapter {id:?} never became resident");
}

fn bank_config() -> AdapterBankConfig {
    AdapterBankConfig::new(2, 4, &["q_proj", "k_proj", "v_proj", "o_proj", "g_proj", "gate_proj", "up_proj", "down_proj"])
}

/// Spec: one batch holding a base row and the rows of two adapters over ONE resident base answers each row as that
/// adapter alone answers it, and as the training model with that adapter computes it, on one stage and on two.
#[test]
fn a_mixed_batch_decodes_each_row_as_its_adapter_alone_and_as_the_training_forward_does() {
    let model = synthetic(&["sliding_attention", "sliding_attention", "full_attention"], Shape { heads: 4, kv_heads: 2, head_dim: 16 });
    let vocab = model.cfg.vocab_size;
    let tmp = tempfile::tempdir().unwrap();
    let (train_a, a) = trained(&model, 11, 8, tmp.path());
    let (train_b, b) = trained(&model, 12, 8, tmp.path());
    let (train_base, _) = trained(&model, 13, 0, tmp.path());
    let prompt = stream(5, 12, vocab as u32);
    let next = 7u32;
    let mut full = prompt.clone();
    full.push(next);
    let want: Vec<Vec<f32>> = [&train_base, &train_a, &train_b].iter().map(|t| last(&t.logits_all(&full).unwrap(), vocab)).collect();

    for staged in [false, true] {
        let mut engine = if staged { model.staged(serve_config(4, 0)) } else { model.engine(serve_config(4, 0)) };
        assert!(matches!(engine.load_adapter(a.clone()), Err(AdapterError::NoBank)), "no bank, no adapters");
        engine.enable_adapters(bank_config()).unwrap();
        let (ida, idb) = (engine.load_adapter(a.clone()).unwrap(), engine.load_adapter(b.clone()).unwrap());
        assert_eq!(engine.load_adapter(a.clone()).unwrap(), ida, "one digest is one adapter");
        assert_ne!(ida, idb);
        admit(&mut engine, ida);
        admit(&mut engine, idb);
        let namespaces = [model::paged::BASE_KV_NAMESPACE, ida.kv_namespace(), idb.kv_namespace()];

        // Alone: each sequence decoded by itself.
        let mut alone = Vec::new();
        for ns in namespaces {
            let mut table = BlockTable::with_namespace(ns);
            prefill_in_slices(&mut engine, &mut table, &prompt, 5);
            alone.push(engine.decode_logits(&mut [&mut table], &[next]).remove(0));
            engine.release_table(&mut table);
        }
        // Together: one batch of the three.
        let mut tables: Vec<BlockTable> = namespaces.iter().map(|&ns| BlockTable::with_namespace(ns)).collect();
        for t in &mut tables {
            prefill_in_slices(&mut engine, t, &prompt, 5);
        }
        let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
        let together = engine.decode_logits(&mut refs, &[next, next, next]);
        for (i, name) in ["base", "adapter a", "adapter b"].iter().enumerate() {
            let vs_alone = scaled_error(&format!("{name} together vs alone"), &together[i], &alone[i]);
            let vs_trained = scaled_error(&format!("{name} served vs trained"), &together[i], &want[i]);
            eprintln!("staged {staged}: {name}: vs alone {vs_alone:.2e}, vs training forward {vs_trained:.2e}");
            assert!(vs_alone <= 1e-6, "{name}: a batch answers a row as it is answered alone ({vs_alone:.2e})");
            assert!(vs_trained <= 5e-6, "{name}: the bank answers what the training model does ({vs_trained:.2e})");
        }
        let distinct = scaled_error("a vs base", &together[1], &together[0]).min(scaled_error("b vs a", &together[2], &together[1]));
        assert!(distinct > 1e-3, "the adapters change the answer ({distinct:.2e})");
        for (t, id) in tables.iter_mut().zip([None, Some(ida), Some(idb)]) {
            engine.release_table(t);
            if let Some(id) = id {
                engine.unpin_adapter(id);
            }
        }
        assert!(engine.adapter_bank_bytes() > 0);
    }
}

/// Spec: an adapter loaded after serving began is usable at once and one unloaded stops being offered, with its
/// cached prefixes gone, while a sequence under it finishes first; a third adapter is served by evicting an idle slot.
#[test]
fn adapters_load_and_unload_while_the_engine_serves() {
    let model = synthetic(&["sliding_attention", "full_attention"], Shape { heads: 4, kv_heads: 2, head_dim: 16 });
    let tmp = tempfile::tempdir().unwrap();
    let mut engine = model.engine(serve_config(4, 0));
    engine.enable_adapters(bank_config()).unwrap();
    let ids: Vec<AdapterId> = (21..24).map(|seed| engine.load_adapter(trained(&model, seed, 0, tmp.path()).1).unwrap()).collect();
    assert_eq!(engine.adapters().len(), 3);
    // Two slots, three adapters: each is served in turn by evicting the least recently used idle one.
    for &id in &ids {
        admit(&mut engine, id);
        engine.unpin_adapter(id);
    }
    admit(&mut engine, ids[0]);
    assert_eq!(engine.unload_adapter(ids[0]).unwrap(), UnloadStatus::Draining, "a sequence still runs under it");
    assert_eq!(engine.adapter_availability(ids[0]), AdapterAvailability::Unknown, "a draining adapter takes no new users");
    engine.unpin_adapter(ids[0]);
    assert_eq!(engine.adapters().len(), 2, "the last sequence finishing completed the unload");
    assert_eq!(engine.unload_adapter(ids[1]).unwrap(), UnloadStatus::Unloaded);
    assert!(matches!(engine.unload_adapter(ids[1]), Err(AdapterError::Unknown(_))));
}

/// A distinct random adapter on every linear of the conformance model, loaded into `e`'s bank.
fn conformance_adapter(e: &mut RicursiveEngine) -> AdapterId {
    use model::adapter::sites::LoraSites;
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let mut rng = data::rng::Rng::new(u64::from(n)); // seed: any distinct factors serve
    let rank = 2usize;
    let sites = TrainConfig::new(e.config().clone(), 1).lora_sites();
    let pairs = sites
        .iter()
        .map(|s| model::lora::device_adapter::AdapterSite { base: s.name.clone(), a: (0..rank * s.spec.inn).map(|_| 0.3 * rng.next_gaussian() as f32).collect(), b: (0..s.spec.out * rank).map(|_| 0.3 * rng.next_gaussian() as f32).collect() })
        .collect();
    e.load_adapter(VerifiedAdapter { label: format!("conformance/{n}"), digest: format!("sha256:conformance-{n}"), adapter: model::lora::device_adapter::DeviceAdapter { rank: rank as u32, alpha: 4.0, sites: pairs, saved: Vec::new() } }).unwrap()
}

fn conformance_engine() -> RicursiveEngine {
    let model = synthetic(&["sliding_attention", "sliding_attention", "full_attention"], Shape { heads: 4, kv_heads: 2, head_dim: 16 });
    let mut engine = model.engine(ricursive::ServeConfig::new(4, 160, 96, 6, 4, 8));
    engine.enable_adapters(bank_config()).unwrap();
    engine
}

/// Spec: the shared serving suite every engine runs (continuous batching, no head-of-line blocking, prefix reuse,
/// speculation) passes with the adapter bank on, including its adapter checks: adapters never share KV with the base, a
/// mixed batch does not wait, embedded prompts are kept apart.
#[test]
fn the_shared_serving_conformance_suite_passes_with_adapters() {
    model::serve::conformance::run_all(&model::serve::conformance::Conformance { make: conformance_engine, prefix_reuse: true, prefix_prompt_tokens: 24, load_adapter: Some(conformance_adapter) });
}
