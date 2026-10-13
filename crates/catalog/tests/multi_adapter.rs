// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving many fine-tuned
// variants of one model from a single resident copy of its weights for its
// clients. If your team needs expertise in multi-tenant inference engines
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Multi-adapter serving through the real executor: one resident Qwen3, the
//! base and several adapters as model ids, requests for all of them in the
//! same batches.
//!
//! The spec, each requirement its own test:
//!
//! * a request for an adapter id is answered exactly as a server that folded
//!   that adapter into its weights answers it, and a request for the base id,
//!   made at the same time, exactly as a server with no adapters answers it;
//! * base and adapter requests share one instance and one batch;
//! * a name the server does not offer is an unknown model, and the reserved
//!   param that carries the selection cannot be set by a caller;
//! * an adapter withdrawn while a request for it runs does not interrupt that
//!   request, and a new release offered beside it answers at once;
//! * the bank is part of the resident's footprint, and an adapter the bank
//!   cannot hold is refused with the setting to change.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use brain_testutil::adapters::{scratch_dir, write_byte_tokenizer, write_lora_adapter, Linear};
use capability::{Invocation, Outcome};
use catalog::adapter_registry::{AdapterBankSettings, AdapterTargets, LiveAdapters, ADAPTER_PARAM};
use catalog::adapter_release::verify_adapter;
use catalog::resident_llm::{QwenResident, QwenServeConfig};
use checkpoint::st::ModelCard;
use qwen3::config::QwenConfig;
use residency::budget::Budgets;
use residency::{Device, Executor, InstanceKey, Policy, ResidentModel};
use serde_json::json;

const BASE: &str = "Qwen/Tiny";
const ONE: &str = "Qwen/Tiny:lab:one:v1";
const TWO: &str = "Qwen/Tiny:lab:two:v1";
const RANK: u32 = 4;
const PROMPT: &str = "the sky is";
const NEW_TOKENS: u64 = 16;

fn cfg() -> QwenConfig {
    QwenConfig { vocab: 128, ..QwenConfig::tiny() }
}

/// Every attention linear of the tiny model, in the shapes an adapter file
/// stores them.
fn attention_linears(cfg: &QwenConfig) -> Vec<Linear> {
    let (d, hq, hkv) = (cfg.d_model as usize, cfg.q_dim() as usize, cfg.kv_dim() as usize);
    (0..cfg.n_layers)
        .flat_map(|l| {
            [("wq", hq, d), ("wk", hkv, d), ("wv", hkv, d), ("wo", d, hq)].map(|(leaf, out, inn)| Linear { base: format!("blocks.{l}.attn.{leaf}.weight"), out, inn })
        })
        .collect()
}

fn write_checkpoint(path: &Path, cfg: &QwenConfig, weights: &std::collections::HashMap<String, Vec<f32>>) {
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg.param_list().into_iter().map(|(name, n)| (name.clone(), vec![n as u64], weights[&name].clone())).collect();
    checkpoint::save(path.to_str().unwrap(), cfg.to_json(), &tensors);
}

/// The files of one scenario: a base, two adapters on it, and the same base
/// with each adapter folded in.
struct Fixture {
    dir: PathBuf,
    tokenizer: String,
    base: PathBuf,
    adapters: [PathBuf; 2],
    folded: [PathBuf; 2],
}

fn fixture(tag: &str) -> Fixture {
    let dir = scratch_dir(tag);
    let tokenizer = write_byte_tokenizer(&dir.join("tok")).to_string_lossy().into_owned();
    let cfg = cfg();
    let weights = qwen3::init_weights(&cfg, 7);
    let base = dir.join("base.safetensors");
    write_checkpoint(&base, &cfg, &weights);
    let linears = attention_linears(&cfg);
    let adapters = [(ONE, 11u64), (TWO, 22u64)].map(|(id, seed)| {
        let path = dir.join(format!("{}.safetensors", id.replace([':', '/'], "_")));
        write_lora_adapter(&path, id, BASE, RANK, &linears, seed);
        path
    });
    let folded = [0, 1].map(|i| {
        let mut w = weights.clone();
        qwen3::lora::fold_adapter_into(&mut w, adapters[i].to_str().unwrap()).expect("the adapter folds into the base");
        let path = dir.join(format!("folded-{i}.safetensors"));
        write_checkpoint(&path, &cfg, &w);
        path
    });
    Fixture { dir, tokenizer, base, adapters, folded }
}

fn serve_config(bank: Option<AdapterBankSettings>) -> QwenServeConfig {
    QwenServeConfig { ctx: Some(256), kv_int8: false, max_batch: 4, max_prefill_cap: 64, adapter_bank: bank, ..QwenServeConfig::default() }
}

fn bank() -> AdapterBankSettings {
    AdapterBankSettings { slots: 2, rank_cap: RANK, targets: AdapterTargets::ATTENTION }
}

fn resident(f: &Fixture, id: &str, path: &Path, bank: Option<AdapterBankSettings>) -> Arc<QwenResident> {
    Arc::new(QwenResident::from_card_configured(path.to_str().unwrap(), &ModelCard::new(id, "qwen"), Some(&f.tokenizer), serve_config(bank)))
}

fn executor(models: Vec<Arc<dyn ResidentModel>>) -> Executor {
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 8 << 30, 0);
    Executor::start(models, budgets, Policy::default())
}

fn offer(live: &LiveAdapters, resident: &QwenResident, path: &Path) {
    let loaded = verify_adapter(path, &resident.served_base()).expect("the adapter verifies against its base");
    live.serve(loaded).expect("the adapter is offered");
}

fn greedy(max_new: u64) -> Invocation {
    Invocation::new().set("prompt", json!(PROMPT)).set("chat", json!(false)).set("max_new", json!(max_new)).set("temp", json!(0.0)).set("top_k", json!(1)).set("seed", json!(1))
}

fn text(outcome: Outcome) -> String {
    outcome.outputs["text"].as_str().expect("a text output").to_string()
}

/// Run `model` once and return its text.
fn ask(exec: &Executor, model: &str, inv: Invocation) -> Result<String, String> {
    exec.run_blocking(model, "generate", inv, |_| {}).map(text)
}

/// A request for each adapter and for the base, all submitted before any is
/// answered, reply exactly as the folded server and the adapter-free server
/// reply to the same request; the adapters differ from each other and from
/// the base, so the equalities are not vacuous.
#[test]
fn every_id_answers_as_its_standalone_server_does_when_asked_at_once() {
    let f = fixture("multi-adapter-equal");
    let served = resident(&f, BASE, &f.base, Some(bank()));
    let exec = executor(vec![served.clone()]);
    let live = LiveAdapters::new(served.clone(), exec.clone());
    offer(&live, &served, &f.adapters[0]);
    offer(&live, &served, &f.adapters[1]);

    // The reference servers: the base alone with no bank, and each adapter folded.
    let plain = executor(vec![resident(&f, BASE, &f.base, None)]);
    let folded_0 = executor(vec![resident(&f, BASE, &f.folded[0], None)]);
    let folded_1 = executor(vec![resident(&f, BASE, &f.folded[1], None)]);
    let want_base = ask(&plain, BASE, greedy(NEW_TOKENS)).unwrap();
    let want_one = ask(&folded_0, BASE, greedy(NEW_TOKENS)).unwrap();
    let want_two = ask(&folded_1, BASE, greedy(NEW_TOKENS)).unwrap();
    assert!(want_base != want_one && want_base != want_two && want_one != want_two, "the fixture must tell the three apart: {want_base:?} {want_one:?} {want_two:?}");

    // Rounds, in a different order each time, every request on its own thread.
    for round in 0..3 {
        let order = [[BASE, ONE, TWO], [TWO, BASE, ONE], [ONE, TWO, BASE]][round];
        let asked: Vec<_> = order.iter().map(|&model| (model, { let exec = exec.clone(); std::thread::spawn(move || ask(&exec, model, greedy(NEW_TOKENS))) })).collect();
        for (model, handle) in asked {
            let got = handle.join().unwrap().unwrap_or_else(|e| panic!("round {round}: {model}: {e}"));
            let want = match model {
                BASE => &want_base,
                ONE => &want_one,
                _ => &want_two,
            };
            assert_eq!(&got, want, "round {round}: {model} must answer as its standalone server does");
        }
    }
    for e in [exec, plain, folded_0, folded_1] {
        e.shutdown();
    }
}

/// The views and the base are one instance: one activation, one key, and a
/// group that mixes them runs as one batch.
#[test]
fn base_and_adapters_share_one_instance_and_one_batch() {
    let f = fixture("multi-adapter-batch");
    let served = resident(&f, BASE, &f.base, Some(bank()));
    let exec = executor(vec![served.clone()]);
    let live = LiveAdapters::new(served.clone(), exec.clone());
    offer(&live, &served, &f.adapters[0]);
    offer(&live, &served, &f.adapters[1]);

    let key = InstanceKey::new(BASE, "default");
    let inv = Invocation::new();
    assert_eq!(served.instance_key("generate", &inv), key);

    let asked: Vec<_> = [BASE, ONE, TWO, ONE].into_iter().map(|model| { let exec = exec.clone(); std::thread::spawn(move || ask(&exec, model, greedy(8))) }).collect();
    for handle in asked {
        handle.join().unwrap().expect("the request is answered");
    }
    let stats = exec.stats();
    assert_eq!(stats.builds, 1, "one activation served the base and both adapters: {stats:?}");
    assert_eq!(exec.residency().placements.len(), 1, "one resident instance");
    assert!(stats.max_batch >= 2, "requests for different ids ran in one batch: {stats:?}");
    exec.shutdown();
}

/// A name that is not offered is an unknown model, whatever shape it has; a
/// caller cannot select an adapter by setting the reserved param, on the base
/// or on an adapter.
#[test]
fn an_unoffered_name_is_an_unknown_model_and_selection_cannot_be_forged() {
    let f = fixture("multi-adapter-forge");
    let served = resident(&f, BASE, &f.base, Some(bank()));
    let exec = executor(vec![served.clone()]);
    let live = LiveAdapters::new(served.clone(), exec.clone());
    offer(&live, &served, &f.adapters[0]);

    let want_base = ask(&exec, BASE, greedy(NEW_TOKENS)).unwrap();
    let want_one = ask(&exec, ONE, greedy(NEW_TOKENS)).unwrap();
    assert_ne!(want_base, want_one);

    for ghost in [TWO, "Qwen/Tiny:lab:ghost:v9", "Qwen/Tiny:lab:../x:v1", "pinned"] {
        let err = ask(&exec, ghost, greedy(4)).unwrap_err();
        assert!(err.starts_with("no model"), "{ghost}: {err}");
        assert!(!err.contains(".safetensors"), "the answer carries no path: {err}");
    }

    // The reserved param, set by the caller.
    assert_eq!(ask(&exec, BASE, greedy(NEW_TOKENS).set(ADAPTER_PARAM, json!(ONE))).unwrap(), want_base, "a caller reaching the base cannot select an adapter");
    assert_eq!(ask(&exec, ONE, greedy(NEW_TOKENS).set(ADAPTER_PARAM, json!(TWO))).unwrap(), want_one, "and a caller reaching one adapter cannot select another");
    exec.shutdown();
}

/// A withdrawn adapter's running request finishes; the new release offered
/// beside it answers; the old id is then unknown and the engine lets go of it.
#[test]
fn a_hot_swap_leaves_a_running_request_alone_and_serves_the_new_release_at_once() {
    let f = fixture("multi-adapter-swap");
    let served = resident(&f, BASE, &f.base, Some(bank()));
    let exec = executor(vec![served.clone()]);
    let live = LiveAdapters::new(served.clone(), exec.clone());
    offer(&live, &served, &f.adapters[0]);
    let before = ask(&exec, ONE, greedy(NEW_TOKENS)).unwrap();

    // A long request under the old release, running when it is swapped out.
    let long = {
        let exec = exec.clone();
        std::thread::spawn(move || ask(&exec, ONE, greedy(160)))
    };
    let deadline = Instant::now() + Duration::from_secs(60);
    while exec.in_flight().is_empty() {
        assert!(Instant::now() < deadline, "the long request never reached the executor");
        std::thread::sleep(Duration::from_millis(2));
    }

    offer(&live, &served, &f.adapters[1]);
    live.retire(ONE);
    assert!(!exec.manifests().iter().any(|m| m.model == ONE), "the old id is unlisted at once");
    assert!(exec.manifests().iter().any(|m| m.model == TWO), "the new id is listed");
    assert_eq!(exec.served_adapter(ONE), None);
    assert!(exec.served_adapter(TWO).is_some_and(|a| a.id == TWO));
    assert!(ask(&exec, ONE, greedy(4)).unwrap_err().starts_with("no model"), "a request after the swap for the old id is an unknown model");

    let finished = long.join().unwrap().expect("the request that was running when its adapter was withdrawn finishes");
    assert!(finished.starts_with(&before), "and finishes on the adapter it began on: {finished:?} vs {before:?}");
    let after = ask(&exec, TWO, greedy(NEW_TOKENS)).unwrap();
    assert_ne!(after, before, "the new release answers differently from the old");

    // Once the old id's last job is done the registry lets go of it.
    let deadline = Instant::now() + Duration::from_secs(30);
    while served.adapters().ids() != [TWO.to_string()] {
        assert!(Instant::now() < deadline, "the withdrawn adapter was never released: {:?}", served.adapters().ids());
        std::thread::sleep(Duration::from_millis(20));
    }
    exec.shutdown();
}

/// The same card id under a new digest replaces the adapter under that id.
#[test]
fn a_new_release_under_the_same_card_id_replaces_the_adapter_under_that_id() {
    let f = fixture("multi-adapter-replace");
    let served = resident(&f, BASE, &f.base, Some(bank()));
    let exec = executor(vec![served.clone()]);
    let live = LiveAdapters::new(served.clone(), exec.clone());
    offer(&live, &served, &f.adapters[0]);
    let first = exec.served_adapter(ONE).unwrap();
    let want_first = ask(&exec, ONE, greedy(NEW_TOKENS)).unwrap();

    // A second file, same card id, other factors.
    let second_path = f.dir.join("one-again.safetensors");
    write_lora_adapter(&second_path, ONE, BASE, RANK, &attention_linears(&cfg()), 99);
    offer(&live, &served, &second_path);
    let second = exec.served_adapter(ONE).unwrap();
    assert_ne!(first.digest, second.digest, "the card reports the release now served");
    assert_eq!(exec.manifests().iter().filter(|m| m.model == ONE).count(), 1, "one model, not two");
    let want_second = ask(&exec, ONE, greedy(NEW_TOKENS)).unwrap();
    assert_ne!(want_first, want_second, "the id now answers as the new release");
    exec.shutdown();
}

/// The bank is part of what placement budgets; an adapter the bank cannot
/// hold, or an id that is not an adapter reference, or one that would shadow
/// a model, is refused with the reason.
#[test]
fn the_bank_is_budgeted_and_what_it_cannot_hold_is_refused_with_the_setting() {
    let f = fixture("multi-adapter-budget");
    let with_bank = resident(&f, BASE, &f.base, Some(bank()));
    let without = resident(&f, BASE, &f.base, None);
    let key = InstanceKey::new(BASE, "default");
    let extra = with_bank.estimate(&key).vram - without.estimate(&key).vram;
    // Rows the engine sizes its row buffers for: the larger of the decode batch (4) and the prefill chunk (64).
    assert_eq!(extra, bank().device_bytes(&cfg(), 64), "the estimate counts exactly the bank the engine will allocate");
    assert!(extra > 0);

    let exec = executor(vec![with_bank.clone(), Arc::new(QwenResident::from_card_configured(f.base.to_str().unwrap(), &ModelCard::new("Qwen/Other", "qwen"), Some(&f.tokenizer), serve_config(None)))]);
    let live = LiveAdapters::new(with_bank.clone(), exec.clone());

    // An adapter trained at a higher rank than the bank holds.
    let wide = f.dir.join("wide.safetensors");
    write_lora_adapter(&wide, "Qwen/Tiny:lab:wide:v1", BASE, RANK * 2, &attention_linears(&cfg()), 5);
    let loaded = verify_adapter(&wide, &with_bank.served_base()).unwrap();
    assert!(live.serve(loaded).unwrap_err().contains("--adapter-rank-cap"));

    // An id that names a model, not an adapter.
    let mut renamed = verify_adapter(&f.adapters[0], &with_bank.served_base()).unwrap();
    renamed.release.id = "pinned".to_string();
    assert!(live.serve(renamed).unwrap_err().contains("not a model reference"));

    // An adapter whose id is a served model's.
    let mut shadow = verify_adapter(&f.adapters[0], &with_bank.served_base()).unwrap();
    shadow.release.id = "Qwen/Other".to_string();
    assert!(live.serve(shadow).is_err(), "an adapter cannot take a model's id");
    assert!(exec.manifests().iter().any(|m| m.model == "Qwen/Other"), "and the model is untouched");
    live.retire("Qwen/Other");
    assert!(exec.manifests().iter().any(|m| m.model == "Qwen/Other"), "retiring cannot withdraw a model that is not one of this resident's adapters");
    assert!(exec.manifests().iter().any(|m| m.model == BASE));

    // A server started without a bank offers no adapter at all.
    let bare = resident(&f, "Qwen/Bare", &f.base, None);
    let exec_bare = executor(vec![bare.clone()]);
    let loaded = verify_adapter(&f.adapters[0], &bare.served_base());
    assert!(loaded.is_err(), "bound by id, the adapter is for {BASE}, not Qwen/Bare");
    exec.shutdown();
    exec_bare.shutdown();
}
