// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Adapters that carry full tensors (`modules_to_save`): norm gains and value-residual gates trained beside the LoRA
//! factors, written into the adapter file and listed on its card, read back, applied by the trainer that continues
//! them, folded into a base, and served per row of a mixed batch by the engine's bank.
//!
//! Swedish Embedded AB implements solutions for serving many fine-tuned variants of one very large model from a
//! single resident copy of its weights, including variants that retrain norms and gates. If your team needs expertise
//! in multi-tenant inference of fine-tuned language models then you can procure our services by sending an email to
//! info@swedishembedded.com.

use model::adapter::bank::{AdapterBankConfig, AdapterError, VerifiedAdapter};
use model::adapter::device::LoraCfg;
use model::paged::BlockTable;
use model::serve::{AdapterAvailability, AdapterId, PagedDecoder};
use model::{Batch, Model};
use optim::Adam;
use ricursive::train::{adapter, default_targets, expand_modules_to_save, AdapterInit};
use ricursive::{open_safetensors, RicursiveEngine, Ripple, TrainConfig, TrainOpts};

use crate::serve::{prefill_in_slices, serve_config, stream};
use crate::synthetic::{synthetic, Model as Synthetic, Shape};
use crate::taps::scaled_error;

const SEQ: u32 = 16;

fn cfg_of(model: &Synthetic, saved: &[&str]) -> TrainConfig {
    let patterns: Vec<String> = saved.iter().map(|s| (*s).to_string()).collect();
    TrainConfig::new(model.cfg.clone(), SEQ).with_lora(LoraCfg::new(2, 8.0, default_targets()).with_modules_to_save(expand_modules_to_save(&patterns)))
}

fn batch(seed: u32, vocab: u32) -> (Vec<u32>, Vec<u32>) {
    let tokens = stream(seed, 12, vocab);
    let targets: Vec<u32> = tokens[1..].iter().copied().chain([ricursive::train::IGNORE]).collect();
    let (mut x, mut y) = (tokens, targets);
    x.resize(SEQ as usize, 0);
    y.resize(SEQ as usize, ricursive::train::IGNORE);
    (x, y)
}

/// A trainer over the synthetic model with an adapter that also trains `saved`, `steps` steps on `seed`'s stream.
fn trained(model: &Synthetic, saved: &[&str], seed: u32, steps: u32) -> Ripple {
    let layers: Vec<usize> = (0..model.cfg.num_hidden_layers).collect();
    let base = open_safetensors(&model.path(), &model.cfg, &layers).unwrap();
    let cfg = cfg_of(model, saved);
    let init = AdapterInit::fresh(&cfg, &base, u64::from(seed));
    let trainer = Ripple::build(cfg, 1, SEQ, &init, &TrainOpts::default(), true).unwrap();
    let (x, y) = batch(seed, model.cfg.vocab_size as u32);
    trainer.set_batch(Batch::Lm { tokens: &x, targets: &y });
    for step in 1..=steps {
        trainer.zero_grads();
        trainer.forward();
        trainer.backward();
        trainer.adamw_step(step, 5e-2, 0.0, Adam::default(), None, 1.0);
    }
    trainer
}

/// The base's tensors as a host map, as `fold_adapter_into` takes them.
fn host_map(model: &Synthetic) -> std::collections::HashMap<String, Vec<f32>> {
    use checkpoint::TensorSource;
    let layers: Vec<usize> = (0..model.cfg.num_hidden_layers).collect();
    let base = open_safetensors(&model.path(), &model.cfg, &layers).unwrap();
    ricursive::import::manifest(&model.cfg, &layers)
        .into_iter()
        .map(|(name, _)| {
            let mut data = Vec::new();
            assert!(base.with_tensor(&name, &mut |d| data = d.to_vec()), "{name}");
            (name, data)
        })
        .collect()
}

fn verified(trainer: &Ripple, label: &str, dir: &std::path::Path) -> VerifiedAdapter {
    let file = dir.join(format!("{}.safetensors", label.replace('/', "-")));
    trainer.save_adapter(file.to_str().unwrap(), label, "test/base", None, None).unwrap();
    VerifiedAdapter { label: label.into(), digest: format!("sha256:{label}"), adapter: adapter::read_adapter(file.to_str().unwrap()).unwrap() }
}

fn last(logits: &[f32], vocab: usize) -> Vec<f32> {
    logits[logits.len() - vocab..].to_vec()
}

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

fn synth() -> Synthetic {
    synthetic(&["sliding_attention", "sliding_attention", "full_attention"], Shape { heads: 4, kv_heads: 2, head_dim: 16 })
}

/// Spec: the norms and the value-residual gates train beside the factors (nothing else of the base does), learn, and
/// the base linears stay as they were.
#[test]
fn norms_and_gates_train_beside_the_factors() {
    let model = synth();
    let cfg = cfg_of(&model, &["norms", "gates"]);
    let base = host_map(&model);
    let trainer = trained(&model, &["norms", "gates"], 11, 0);
    let names = trainer.param_names();
    let full: Vec<&String> = names.iter().filter(|n| !model::adapter::device::is_adapter_param(n)).collect();
    assert_eq!(full.len(), cfg.trained_in_full().len(), "every norm and gate of the model trains: {full:?}");
    assert!(full.iter().all(|n| n.contains("norm") || n.ends_with("value_res_lambda")), "only norms and gates: {full:?}");
    assert!(full.iter().any(|n| n.ends_with("value_res_lambda")) && full.iter().any(|n| n.ends_with("q_norm.weight")) && full.iter().any(|n| *n == "model.norm.weight"));

    let (x, y) = batch(11, model.cfg.vocab_size as u32);
    trainer.set_batch(Batch::Lm { tokens: &x, targets: &y });
    let start = trainer.forward();
    for step in 1..=20 {
        trainer.zero_grads();
        trainer.forward();
        trainer.backward();
        trainer.adamw_step(step, 5e-2, 0.0, Adam::default(), None, 1.0);
    }
    let end = trainer.forward();
    eprintln!("loss {start:.4} -> {end:.4}");
    assert!(end < 0.8 * start);
    let moved = full.iter().filter(|n| trainer.read_weight(n) != base[n.as_str()]).count();
    assert_eq!(moved, full.len(), "every full tensor moved");
    for (name, want) in base.iter().filter(|(n, _)| !full.iter().any(|f| f.as_str() == n.as_str()) && !model::adapter::device::is_adapter_param(n)) {
        assert_eq!(&trainer.read_weight(name), want, "{name} is frozen");
    }
}

/// Spec: the adapter file lists its full tensors on the card and holds each; read back they are what the trainer held,
/// a model that loads the adapter computes what the trainer did, a fold into the base replaces the tensors, a run that
/// continues the adapter starts from them, and a run that does not train a tensor the adapter carries refuses it.
#[test]
fn an_adapter_file_lists_and_holds_its_full_tensors() {
    let model = synth();
    let trainer = trained(&model, &["norms", "gates"], 11, 6);
    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join("adapter.safetensors");
    trainer.save_adapter(file.to_str().unwrap(), "test/saved", "test/base", None, None).unwrap();
    let card = checkpoint::st::read_card(file.to_str().unwrap()).unwrap().unwrap();
    let listed = card.adapter.unwrap().modules_to_save.expect("the card lists the full tensors");
    let read = adapter::read_adapter(file.to_str().unwrap()).unwrap();
    assert_eq!(read.saved.iter().map(|t| t.base.clone()).collect::<Vec<_>>(), listed);
    assert!(listed.contains(&"model.norm.weight".to_string()) && listed.iter().any(|n| n.ends_with("value_res_lambda")), "{listed:?}");
    for t in &read.saved {
        assert_eq!(t.data, trainer.read_weight(&t.base), "{}", t.base);
    }

    // Continuing: the run starts from the adapter's tensors and factors.
    let layers: Vec<usize> = (0..model.cfg.num_hidden_layers).collect();
    let base_source = open_safetensors(&model.path(), &model.cfg, &layers).unwrap();
    let base = &base_source;
    let cfg = cfg_of(&model, &["norms", "gates"]);
    let init = AdapterInit::continued(&cfg, base, &read).unwrap();
    let again = Ripple::build(cfg.clone(), 1, SEQ, &init, &TrainOpts::default(), true).unwrap();
    let (x, y) = batch(11, model.cfg.vocab_size as u32);
    let logits = |m: &Ripple| m.logits_all(&x[..12]).unwrap();
    assert_eq!(logits(&again), logits(&trainer), "a continued run computes what the adapter computed");
    // Loading the adapter into a fresh model of the same configuration does too.
    let fresh = Ripple::build(cfg.clone(), 1, SEQ, &AdapterInit::fresh(&cfg, base, 3), &TrainOpts::default(), true).unwrap();
    assert_ne!(logits(&fresh), logits(&trainer));
    fresh.load_adapter(&read).unwrap();
    assert_eq!(logits(&fresh), logits(&trainer));
    drop(y);

    // A run that trains only the factors does not take tensors it would never train.
    let factors_only = cfg_of(&model, &[]);
    let err = AdapterInit::continued(&factors_only, base, &read).err().expect("refused");
    assert!(err.contains("modules_to_save") || err.contains("in full"), "{err}");

    // Folding replaces the base's tensors by the adapter's.
    let base = host_map(&model);
    let mut map = base.clone();
    model::lora::device_adapter::fold_adapter_into(&mut map, file.to_str().unwrap()).unwrap();
    for t in &read.saved {
        assert_eq!(map[&t.base], t.data, "{}", t.base);
        assert_ne!(map[&t.base], base[&t.base], "{} was trained away from the base", t.base);
    }

    // PEFT: norms are modules (PEFT's modules_to_save); a bare gate has no PEFT form and is refused by name.
    let refused = ricursive::train::adapter::export_peft(file.to_str().unwrap(), &tmp.path().join("peft"), None).expect_err("a bare parameter cannot be a PEFT module");
    assert!(refused.contains("value_res_lambda") && refused.contains("bare parameter"), "{refused}");
    let norms = trained(&model, &["norms"], 11, 2);
    let nfile = tmp.path().join("norms.safetensors");
    norms.save_adapter(nfile.to_str().unwrap(), "test/norms", "test/base", None, None).unwrap();
    let out = tmp.path().join("peft-norms");
    ricursive::train::adapter::export_peft(nfile.to_str().unwrap(), &out, None).unwrap();
    let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(out.join("adapter_config.json")).unwrap()).unwrap();
    let modules: Vec<&str> = config["modules_to_save"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert!(modules.contains(&"model.norm") && modules.contains(&"model.layers.1.self_attn.q_norm"), "{modules:?}");
    let tensors = checkpoint::safetensors::read(out.join("adapter_model.safetensors").to_str().unwrap()).unwrap();
    let t = tensors.iter().find(|t| t.name == "base_model.model.model.layers.0.input_layernorm.modules_to_save.default.weight").expect("PEFT's name for a saved module");
    assert_eq!(t.data, norms.read_weight("model.layers.0.input_layernorm.weight"));
}

/// Spec: a batch of a base row and the rows of two adapters that carry norms and gates (one of them also a plain one)
/// answers each row as that adapter alone answers it and as the training forward does, on one stage and on two; the
/// full tensors change the answer (the same factors without them answer differently).
#[test]
fn a_mixed_batch_serves_each_rows_own_norms_and_gates() {
    let model = synth();
    let vocab = model.cfg.vocab_size;
    let tmp = tempfile::tempdir().unwrap();
    let (a_train, b_train) = (trained(&model, &["norms", "gates"], 11, 10), trained(&model, &["norms"], 12, 10));
    let base_train = trained(&model, &[], 13, 0);
    let (a, b) = (verified(&a_train, "test/a", tmp.path()), verified(&b_train, "test/b", tmp.path()));
    assert!(!a.adapter.saved.is_empty() && !b.adapter.saved.is_empty());
    let stripped = VerifiedAdapter { label: "test/a-factors".into(), digest: "sha256:a-factors".into(), adapter: model::lora::device_adapter::DeviceAdapter { saved: Vec::new(), ..a.adapter.clone() } };

    let prompt = stream(5, 12, vocab as u32);
    let next = 7u32;
    let mut full = prompt.clone();
    full.push(next);
    let want: Vec<Vec<f32>> = [&base_train, &a_train, &b_train].iter().map(|t| last(&t.logits_all(&full).unwrap(), vocab)).collect();

    for staged in [false, true] {
        let mut engine = if staged { model.staged(serve_config(4, 0)) } else { model.engine(serve_config(4, 0)) };
        engine.enable_adapters(bank_config()).unwrap();
        let (ida, idb) = (engine.load_adapter(a.clone()).unwrap(), engine.load_adapter(b.clone()).unwrap());
        admit(&mut engine, ida);
        admit(&mut engine, idb);
        let namespaces = [model::paged::BASE_KV_NAMESPACE, ida.kv_namespace(), idb.kv_namespace()];
        let mut alone = Vec::new();
        for ns in namespaces {
            let mut table = BlockTable::with_namespace(ns);
            prefill_in_slices(&mut engine, &mut table, &prompt, 5);
            alone.push(engine.decode_logits(&mut [&mut table], &[next]).remove(0));
            engine.release_table(&mut table);
        }
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
            assert!(vs_alone <= 1e-5, "{name}: a batch answers a row as it is answered alone ({vs_alone:.2e})");
            assert!(vs_trained <= 2e-5, "{name}: the bank answers what the training model does ({vs_trained:.2e})");
        }
        for (t, id) in tables.iter_mut().zip([None, Some(ida), Some(idb)]) {
            engine.release_table(t);
            if let Some(id) = id {
                engine.unpin_adapter(id);
            }
        }

        // The same factors without the full tensors answer differently: the norms and gates are applied.
        let ids = engine.load_adapter(stripped.clone()).unwrap();
        admit(&mut engine, ids);
        let mut table = BlockTable::with_namespace(ids.kv_namespace());
        prefill_in_slices(&mut engine, &mut table, &prompt, 5);
        let factors_only = engine.decode_logits(&mut [&mut table], &[next]).remove(0);
        engine.release_table(&mut table);
        engine.unpin_adapter(ids);
        let applied = scaled_error("with vs without the full tensors", &alone[1], &factors_only);
        assert!(applied > 1e-3, "the adapter's full tensors change the answer ({applied:.2e})");
    }
}

/// Spec: an adapter that carries a tensor the engine cannot gather per row (a whole linear) is refused at load by name.
#[test]
fn a_full_linear_cannot_be_served_per_row_and_is_refused_by_name() {
    let model = synth();
    let tmp = tempfile::tempdir().unwrap();
    let trainer = trained(&model, &["g_proj"], 11, 2);
    let adapter = verified(&trainer, "test/g", tmp.path());
    assert!(adapter.adapter.saved.iter().all(|t| t.base.ends_with("g_proj.weight")) && adapter.adapter.saved.len() == 3);
    let mut engine = model.engine(serve_config(4, 0));
    engine.enable_adapters(bank_config()).unwrap();
    let err = engine.load_adapter(adapter).expect_err("refused");
    assert!(matches!(err, AdapterError::Incompatible(_)), "{err}");
    assert!(err.to_string().contains("g_proj") && err.to_string().contains("modules_to_save"), "{err}");
}
