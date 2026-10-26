// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving many fine-tuned
// variants of one large model from a single resident copy of its weights for
// its clients. If your team needs expertise in multi-tenant inference of
// hybrid-attention models then you can procure our services by sending an
// email to info@swedishembedded.com.

//! Gate A3 (second architecture, hybrid GDN + GQA): an rsLoRA adapter saved by
//! the trainer, reloaded from the file, and served as an overlay by the
//! adapter bank (no folding into the base) computes what the training forward
//! computes. The file carries the scale the adapter trained with, so a reader
//! that knows only `alpha / rank` applies it correctly; serving the same
//! factors at the plain-LoRA scale does NOT match, which is what makes the
//! parity above informative.

use std::collections::HashMap;

use data::rng::Rng;
use gpu_core::Gpu;
use model::adapter::bank::{AdapterBankConfig, AdapterTargets, VerifiedAdapter};
use model::adapter::sites::LoraSites;
use model::lora::device_adapter::{read_adapter, AdapterSite, DeviceAdapter};
use model::paged::BlockTable;
use model::serve::{AdapterAvailability, PagedDecoder};
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35};
use qwen35::serve::Engine;

// seed: the TOL below was measured on this weight draw
const SEED: u64 = 7;
const RANK: u32 = 4;
const ALPHA: f32 = 8.0;
const TOL: f32 = 2e-3;

fn rs_cfg() -> Qwen35Config {
    let mut lora = lora_cfg(RANK, ALPHA);
    lora.rank_stabilized = true;
    Qwen35Config { lora: Some(lora), ..Qwen35Config::tiny() }
}

fn maxabs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// Base weights, and the LoRA build's init with both factors made non-zero.
fn weights(cfg: &Qwen35Config) -> (HashMap<String, Vec<f32>>, HashMap<String, Vec<f32>>) {
    let base = qwen35::init::init_weights(&Qwen35Config::tiny(), SEED);
    let mut with_lora = qwen35::init::init_weights(cfg, SEED);
    with_lora.extend(base.clone());
    // seed: the TOL below was measured on this adapter draw
    let mut rng = Rng::new(99);
    for (name, w) in with_lora.iter_mut().filter(|(n, _)| model::adapter::device::is_adapter_param(n)) {
        let spread = if model::adapter::device::init_of(name) == Some(model::adapter::ParamInit::Zeros) { 2.0 } else { 0.5 };
        w.iter_mut().for_each(|x| *x = spread * (rng.next_f32() - 0.5));
    }
    (base, with_lora)
}

fn served(cfg: &Qwen35Config, alpha: f32, with_lora: &HashMap<String, Vec<f32>>, label: &str) -> VerifiedAdapter {
    let sites = cfg
        .lora_sites()
        .iter()
        .map(|s| AdapterSite {
            base: s.name.clone(),
            a: with_lora[&model::adapter::device::param_name(&s.name, model::adapter::kind::LORA_A)].clone(),
            b: with_lora[&model::adapter::device::param_name(&s.name, model::adapter::kind::LORA_B)].clone(),
        })
        .collect();
    VerifiedAdapter { label: label.into(), digest: format!("sha256:{label}"), adapter: DeviceAdapter { rank: RANK, alpha, sites } }
}

/// The last position's logits of a sequence prefilled under `adapter`.
fn served_logits(base: &HashMap<String, Vec<f32>>, adapter: VerifiedAdapter, toks: &[u32]) -> Vec<f32> {
    let base_cfg = Qwen35Config::tiny();
    let mut e = Engine::from_map_on(&Gpu::new_cpu(pipelines()), base_cfg.clone(), base, base_cfg.block_size, 4);
    e.enable_adapters(AdapterBankConfig::new(1, 8, &AdapterTargets::ALL.resolve(&base_cfg).unwrap())).unwrap();
    let id = e.load_adapter(adapter).unwrap();
    assert_eq!(e.adapter_availability(id), AdapterAvailability::Resident);
    e.pin_adapter(id);
    let mut table = BlockTable::with_namespace(id.kv_namespace());
    let hidden = e.prefill(&mut table, toks);
    e.logits(&hidden)
}

#[test]
fn an_rslora_adapter_saved_reloaded_and_served_as_an_overlay_matches_the_training_forward() {
    let cfg = rs_cfg();
    let (base, with_lora) = weights(&cfg);
    let toks: Vec<u32> = (0..cfg.block_size).map(|i| (i * 5 + 3) % cfg.vocab).collect();
    let v = cfg.vocab as usize;

    let trained = Qwen35::new_train_on(Gpu::new_cpu(pipelines()), cfg.clone(), 1, cfg.block_size, &with_lora);
    let want = trained.logits_all(&toks);
    let want_last = &want[(toks.len() - 1) * v..toks.len() * v];

    // Save and reload: the file's alpha / rank is the trained scale.
    let path = std::env::temp_dir().join(format!("brain-qwen35-rslora-{}.safetensors", std::process::id()));
    qwen35::lora::save_adapter(path.to_str().unwrap(), &trained, "test/rs", "test/base", None).unwrap();
    let reloaded = read_adapter(path.to_str().unwrap()).unwrap();
    std::fs::remove_file(&path).ok();
    let scale = cfg.lora.as_ref().unwrap().scale();
    assert!((reloaded.scale() - scale).abs() < 1e-6, "file scale {} vs trained {scale}", reloaded.scale());
    assert!(scale != ALPHA / RANK as f32, "the fixture must separate rsLoRA from plain LoRA");

    // Overlay serving at the reloaded adapter's own scale matches training...
    let adapter = VerifiedAdapter { label: "rs".into(), digest: "sha256:rs".into(), adapter: reloaded };
    let got = served_logits(&base, adapter, &toks);
    assert!(maxabs(&got, want_last) < TOL, "overlay vs training forward: {}", maxabs(&got, want_last));

    // ... and the same factors at the plain scale do not.
    let plain = served_logits(&base, served(&cfg, ALPHA, &with_lora, "plain-scale"), &toks);
    assert!(maxabs(&plain, want_last) > 10.0 * TOL, "the plain scale must be distinguishable from the rsLoRA one");
}

/// Gate A2/A3 for the kinds with no low-rank form on the hybrid model: an IA3
/// or LoKr adapter saved by the trainer folds into a plain base
/// (`AdapterFile::fold_into`) and reloads into a fresh adapter build, each
/// computing what the live adapter computes.
#[test]
fn ia3_lokr_and_dora_fold_and_reload_on_the_hybrid_model() {
    for kind in [model::adapter::DeviceKind::Ia3, model::adapter::DeviceKind::Lokr, model::adapter::DeviceKind::Dora] {
        let mut lora = lora_cfg(RANK, ALPHA);
        lora.kind = kind;
        let cfg = Qwen35Config { lora: Some(lora), ..Qwen35Config::tiny() };
        let (base, with_lora) = weights(&cfg);
        let toks: Vec<u32> = (0..cfg.block_size).map(|i| (i * 5 + 3) % cfg.vocab).collect();
        let trained = Qwen35::new_train_on(Gpu::new_cpu(pipelines()), cfg.clone(), 1, cfg.block_size, &with_lora);
        let want = trained.logits_all(&toks);
        let scale = want.iter().fold(1e-6f32, |m, v| m.max(v.abs()));

        let path = std::env::temp_dir().join(format!("brain-qwen35-kind-{}-{}.safetensors", std::process::id(), kind.name()));
        let path_str = path.to_str().unwrap();
        qwen35::lora::save_adapter(path_str, &trained, "test/kind", "test/base", None).unwrap();
        let file = model::adapter::file::AdapterFile::read(path_str).unwrap();
        assert_eq!(file.kind, kind);
        std::fs::remove_file(&path).ok();

        // Fused: fold into the plain base.
        let mut folded = base.clone();
        file.fold_into(&mut folded).unwrap();
        let plain_cfg = Qwen35Config::tiny();
        let fused = Qwen35::new_on(Gpu::new_cpu(pipelines()), plain_cfg.clone(), 1, plain_cfg.block_size, &folded).logits_all(&toks);
        let plain = Qwen35::new_on(Gpu::new_cpu(pipelines()), plain_cfg.clone(), 1, plain_cfg.block_size, &base).logits_all(&toks);
        assert!(maxabs(&plain, &want) > 20.0 * 1e-4 * scale, "{}: the adapter must move the logits well beyond the parity bound (moved {} on scale {scale})", kind.name(), maxabs(&plain, &want));
        assert!(maxabs(&fused, &want) <= 1e-4 * scale, "{}: fused vs live: {}", kind.name(), maxabs(&fused, &want));

        // Reload: a fresh adapter build given the file's tensors.
        let init = qwen35::init::init_weights(&cfg, 7);
        let mut start = init.clone();
        start.extend(base.clone());
        let fresh = Qwen35::new_train_on(Gpu::new_cpu(pipelines()), cfg.clone(), 1, cfg.block_size, &start);
        for (name, data) in &file.tensors {
            fresh.write_weight(name, data);
        }
        assert!(maxabs(&fresh.logits_all(&toks), &want) <= 1e-6 * scale.max(1.0), "{}: reloaded vs trained", kind.name());
    }
}
