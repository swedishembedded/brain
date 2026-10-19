// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A tune exported as a PEFT adapter computes in `transformers` what brain
//! computes with the tune itself. The tune is deterministic and touches every
//! kind of trained tensor the export handles: decoder LoRA on all twelve
//! projections, tower LoRA (at another rank and scale, on the fused `qkv`
//! too) and the connector trained in full. brain serves it
//! (`MiniCpmV::load_tuned`); `tools/goldens/minicpmv_peft_reference.py`
//! applies the export to the HF model; their next-token logits and greedy
//! continuations must agree.
//!
//! The test writes the export to `testdata/minicpmv/peft/adapter/` for the
//! reference script to read, and compares only against a golden made from
//! that exact file (its sha256).

mod common;

use std::collections::HashMap;
use std::path::Path;

use checkpoint::st::{Adapter, ModelCard};
use data::rng::Rng;
use minicpmv::prompt::{Message, Part};
use minicpmv::train::{TUNE_DECODER_ADAPTER, TUNE_MANIFEST, TUNE_VISION};
use minicpmv::{GenOptions, MiniCpmV, MiniCpmVConfig};
use model::serve::PagedDecoder;
use sha2::Digest;

const SCRIPT: &str = "tools/goldens/minicpmv_peft_reference.py";
const OUT_DIMS_FIRST: &[&str] = &["o_proj", "out_proj", "down", "attn.proj", "mlp.fc2"];

/// Small seeded values: a delta that moves the output without wrecking it.
fn noise(rng: &mut Rng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| (rng.next_f32() * 2.0 - 1.0) * scale).collect()
}

/// `(out, in)` of a linear of `numel` values whose `d`-wide side is its
/// output for the projections back into the residual stream, else its input.
fn dims(name: &str, numel: usize, d: usize) -> (usize, usize) {
    if OUT_DIMS_FIRST.iter().any(|leaf| name.contains(&format!(".{leaf}."))) {
        (d, numel / d)
    } else {
        (numel / d, d)
    }
}

/// Write the deterministic tune into `tune`.
fn write_tune(dir: &Path, cfg: &MiniCpmVConfig, tune: &Path) {
    std::fs::create_dir_all(tune).unwrap();
    // seed: the export's bytes, and so the golden's adapter hash, must not change between runs
    let mut rng = Rng::new(1234);
    let (rank, alpha) = (8u32, 16.0f32);
    let d = cfg.text.d_model as usize;
    let mut decoder = Vec::new();
    for (name, numel) in cfg.text.param_list() {
        if qwen35::import::hf_linear_name(&name).is_none() {
            continue;
        }
        let (o, i) = dims(&name, numel, d);
        decoder.push((format!("{name}.lora_a"), vec![(rank as usize * i) as u64], noise(&mut rng, rank as usize * i, 0.02)));
        decoder.push((format!("{name}.lora_b"), vec![(o * rank as usize) as u64], noise(&mut rng, o * rank as usize, 0.02)));
    }
    let mut card = ModelCard::new("test:peft:v1", "qwen35");
    card.adapter = Some(Adapter {
        kind: "lora".into(),
        rank: Some(rank),
        base: Some(common::REFERENCE.into()),
        alpha: Some(alpha),
        targets: Some(qwen35::config::lora_targets()),
        dataset_id: None,
        per_target: None,
    });
    checkpoint::st::save_safetensors(tune.join(TUNE_DECODER_ADAPTER).to_str().unwrap(), &decoder, &serde_json::json!({}), Some(&card)).unwrap();

    let base = minicpmv::import::vision_weights(dir, cfg).unwrap();
    let (vrank, h) = (4usize, cfg.vision.hidden as usize);
    let mut vision: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
    for l in 0..cfg.vision.n_layers {
        for leaf in minicpmv::vision_train::LORA_LEAVES {
            let name = format!("blocks.{l}.{leaf}.weight");
            let (o, i) = dims(&name, base[&name].len(), h);
            vision.push((format!("{name}.lora_a"), vec![(vrank * i) as u64], noise(&mut rng, vrank * i, 0.02)));
            vision.push((format!("{name}.lora_b"), vec![(o * vrank) as u64], noise(&mut rng, o * vrank, 0.02)));
        }
    }
    let mut merger: Vec<&String> = base.keys().filter(|n| n.starts_with("merger.")).collect();
    merger.sort();
    for name in merger {
        let w: Vec<f32> = base[name].iter().zip(noise(&mut rng, base[name].len(), 0.01)).map(|(w, e)| w + e).collect();
        vision.push((name.clone(), vec![w.len() as u64], w));
    }
    checkpoint::save(tune.join(TUNE_VISION).to_str().unwrap(), serde_json::json!({"lora_rank": vrank, "lora_alpha": 8.0}), &vision);
    let manifest = serde_json::json!({
        "format": minicpmv::train::TUNE_FORMAT, "base": common::REFERENCE, "label": "test:peft:v1",
        "decoder": "lora:8:16", "connector": true, "vision": "lora:4:8", "downsample": cfg.downsample.name(), "steps": 0,
    });
    std::fs::write(tune.join(TUNE_MANIFEST), serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
}

#[test]
fn a_peft_export_computes_in_transformers_what_brain_serves() {
    let Some(dir) = common::checkpoint() else { return };
    let cfg = MiniCpmVConfig::from_dir(&dir).unwrap();
    let work = std::env::temp_dir().join(format!("brain-minicpmv-peft-{}", std::process::id()));
    std::fs::remove_dir_all(&work).ok();
    let tune = work.join("tune");
    write_tune(&dir, &cfg, &tune);

    let golden = brain_testutil::testdata_path("minicpmv").join("peft");
    let adapter = golden.join("adapter");
    std::fs::remove_dir_all(&adapter).ok();
    minicpmv::export::export_peft(&tune, &dir, &adapter).expect("export");
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(adapter.join("adapter_config.json")).unwrap()).unwrap();
    let targets = config["target_modules"].as_array().unwrap();
    let decoder_linears = cfg.text.param_list().iter().filter(|(n, _)| qwen35::import::hf_linear_name(n).is_some()).count();
    assert_eq!(targets.len(), decoder_linears + cfg.vision.n_layers as usize * 6, "decoder linears, and per tower layer q, k, v, out, fc1, fc2");
    assert_eq!(config["modules_to_save"], serde_json::json!(["model.merger"]));
    assert_eq!((config["r"].as_u64(), config["lora_alpha"].as_f64()), (Some(8), Some(8.0)), "mixed scales fold into B at scale 1");

    let sha = format!("{:x}", sha2::Sha256::digest(std::fs::read(adapter.join("adapter_model.safetensors")).unwrap()));
    let manifest: Option<serde_json::Value> = std::fs::read(golden.join("manifest.json")).ok().and_then(|b| serde_json::from_slice(&b).ok());
    let Some(manifest) = manifest.filter(|m| m["adapter_sha256"].as_str() == Some(sha.as_str())) else {
        brain_testutil::skip(&format!("no golden for this adapter - run python {SCRIPT} --model {}", dir.display()));
        std::fs::remove_dir_all(&work).ok();
        return;
    };
    let src = brain_testutil::golden::Source::open_manifest(&golden.join("manifest.json"), SCRIPT).expect("golden source");
    let identity = [
        ("text_n_layers", cfg.text.n_layers as i64),
        ("text_d_model", cfg.text.d_model as i64),
        ("vocab_size", cfg.text.vocab as i64),
        ("vision_n_layers", cfg.vision.n_layers as i64),
        ("vision_d_model", cfg.vision.hidden as i64),
        ("insert_layer_id", cfg.vision.insert_layer_id as i64),
    ];
    assert!(src.require(&identity), "the golden was made from another checkpoint");
    let reference: HashMap<String, Vec<f32>> =
        checkpoint::safetensors::read(golden.join("logits.safetensors").to_str().unwrap()).unwrap().into_iter().map(|t| (t.name, t.data)).collect();

    let gpu = gpu_core::testgpu::dev(qwen35::model::pipelines());
    let mut m = MiniCpmV::load_tuned(&dir, Some(&tune), &gpu, 2048, 1, model::kv_tier::KvTier::F32).expect("load the tune");
    let image = imaging::codec::load(golden.join("image.png")).unwrap();
    let question = manifest["question"].as_str().unwrap();
    let want = common::ids(&reference["greedy"]);
    let opts = GenOptions { max_new: want.len(), ..GenOptions::default() };
    let prepared = m.prepare(&[Message::user(vec![Part::Image, Part::Text(question.into())])], &[image], &opts).unwrap();
    assert_eq!(prepared.prompt.ids, common::ids(&reference["input_ids"]), "prompt ids");
    let mut tbl = model::paged::BlockTable::new();
    let hidden = m.engine.prefill_embeds(&mut tbl, &prepared.embeds);
    let logits = m.engine.logits(&hidden);
    m.engine.release_table(&mut tbl);
    let mut table = brain_testutil::parity::Table::new(0.9999, 2e-3);
    table.check("tuned next-token logits", &logits, &reference["logits_last"]);
    table.print();
    assert!(table.failures.is_empty(), "{:#?} (reference applied by {})", table.failures, manifest["applied"]);
    let got = m.generate_prepared(&prepared, &opts, &mut |_| true).unwrap();
    let want: Vec<u32> = want.into_iter().take_while(|t| !m.eos.contains(t)).collect();
    assert_eq!(got, want, "greedy continuation\n got: {:?}\nwant: {:?}", m.decode(&got), m.decode(&want));

    // The agreement means something only if the tune moves the logits well
    // past the tolerance: the base alone must not match the reference.
    drop(m);
    gpu.poll_wait(); // reclaim the tuned build before the base allocates
    let mut base = MiniCpmV::load(&dir, &gpu, 2048, 1, model::kv_tier::KvTier::F32).expect("load the base");
    let prepared = base.prepare(&[Message::user(vec![Part::Image, Part::Text(question.into())])], &[imaging::codec::load(golden.join("image.png")).unwrap()], &opts).unwrap();
    let mut tbl = model::paged::BlockTable::new();
    let hidden = base.engine.prefill_embeds(&mut tbl, &prepared.embeds);
    let base_logits = base.engine.logits(&hidden);
    base.engine.release_table(&mut tbl);
    let rel = brain_testutil::parity::rel_l2(&base_logits, &reference["logits_last"]);
    assert!(rel > 10.0 * 2e-3, "the tune barely changes the logits (base vs tuned rel_l2 {rel}), so the comparison proves nothing");
    std::fs::remove_dir_all(&work).ok();
}
