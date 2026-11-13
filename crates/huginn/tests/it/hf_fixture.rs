// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for loading HuggingFace model releases as published for its clients. If
// your team needs expertise in keeping a model zoo loadable from upstream layouts then you can procure our services
// by sending an email to info@swedishembedded.com.

// hf-fixture: huginn
//! A tiny Huginn checkpoint in the release's own layout (`transformer.{prelude,core_block,coda}.{i}.*`, fused
//! `Wqkv`, the Q and K bias as one `[2, 1, heads, dim]` tensor, a tied head that is also stored as `lm_head.weight`
//! and a stored `freqs_cis` table) read through the importer: the config, the rename plan's two-way coverage with the
//! two documented drops, the bias split into its halves, an unknown tensor refused by name, the import into a brain
//! container, and a forward on the imported weights.

use checkpoint::TensorSource;
use huginn::config::HuginnConfig;
use huginn::import::{import_as, open_dir};
use huginn::init::init_weights;
use huginn::model::Huginn;

/// The HF name and shape of the brain parameter `name`; `None` for the two halves of the bias, which the release keeps
/// as one tensor.
fn hf_layout(cfg: &HuginnConfig, name: &str) -> Option<(String, Vec<u64>)> {
    let (d, ff, v) = (u64::from(cfg.d_model), u64::from(cfg.d_ff), u64::from(cfg.vocab));
    Some(match name {
        "tok.weight" => ("transformer.wte.weight".into(), vec![v, d]),
        "ln_f.weight" => ("transformer.ln_f.weight".into(), vec![d]),
        "adapter.weight" => ("transformer.adapter.weight".into(), vec![d, 2 * d]),
        _ => {
            let (group, rest) = name.split_once('.').unwrap();
            let (i, leaf) = rest.split_once('.').unwrap();
            let group = if group == "core" { "core_block" } else { group };
            let (hf, shape) = match leaf {
                "ln1.weight" => ("norm_1.weight", vec![d]),
                "ln2.weight" => ("norm_2.weight", vec![d]),
                "ln3.weight" => ("norm_3.weight", vec![d]),
                "ln4.weight" => ("norm_4.weight", vec![d]),
                "attn.wqkv.weight" => ("attn.Wqkv.weight", vec![u64::from(cfg.qkv_dim()), d]),
                "attn.wo.weight" => ("attn.proj.weight", vec![d, d]),
                "mlp.fc.weight" => ("mlp.fc.weight", vec![2 * ff, d]),
                "mlp.proj.weight" => ("mlp.proj.weight", vec![d, ff]),
                "attn.q_bias" | "attn.k_bias" => return None,
                other => panic!("no HF name for {other}"),
            };
            (format!("transformer.{group}.{i}.{hf}"), shape)
        }
    })
}

fn fixture(tag: &str, extra: &[(&str, Vec<u64>)]) -> (std::path::PathBuf, HuginnConfig, std::collections::HashMap<String, Vec<f32>>) {
    let cfg = HuginnConfig::tiny();
    let values = init_weights(&cfg, 5);
    let dir = std::env::temp_dir().join(format!("brain-huginn-hf-{tag}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
    for (name, numel) in cfg.param_list() {
        match hf_layout(&cfg, &name) {
            Some((hf, shape)) => {
                assert_eq!(shape.iter().product::<u64>() as usize, numel, "{name}");
                tensors.push((hf, shape, values[&name].clone()));
            }
            // `q_bias` opens the pair: the release's tensor is the two halves one after the other.
            None if name.ends_with("q_bias") => {
                let prefix = name.strip_suffix("q_bias").unwrap();
                let (group, rest) = prefix.split_once('.').unwrap();
                let (i, _) = rest.split_once('.').unwrap();
                let group = if group == "core" { "core_block" } else { group };
                let both: Vec<f32> = values[&name].iter().chain(&values[&format!("{prefix}k_bias")]).copied().collect();
                tensors.push((format!("transformer.{group}.{i}.attn.qk_bias"), vec![2, 1, u64::from(cfg.n_heads), u64::from(cfg.head_dim)], both));
            }
            None => {}
        }
    }
    // The two tensors the importer documents it drops.
    tensors.push(("lm_head.weight".into(), vec![u64::from(cfg.vocab), u64::from(cfg.d_model)], values["tok.weight"].clone()));
    tensors.push(("freqs_cis".into(), vec![4], vec![0.0; 4]));
    for (name, shape) in extra {
        tensors.push(((*name).to_string(), shape.clone(), vec![0.0; shape.iter().product::<u64>() as usize]));
    }
    checkpoint::st::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &tensors, &serde_json::Value::Null, None).unwrap();
    std::fs::write(dir.join("config.json"), cfg.to_json().to_string()).unwrap();
    (dir, cfg, values)
}

#[test]
fn the_release_layout_loads_end_to_end() {
    let (dir, cfg, values) = fixture("loads", &[]);
    let (read, src) = open_dir(&dir).expect("every parameter is covered and every tensor consumed");
    assert_eq!(format!("{read:?}"), format!("{cfg:?}"));
    for (name, want) in &values {
        let mut got = Vec::new();
        assert!(src.with_tensor(name, &mut |t| got = t.to_vec()), "{name} has a source");
        assert_eq!(&got, want, "{name}: the fused bias splits into the halves that were written");
    }

    let out = dir.join("huginn.brain.safetensors");
    import_as(dir.to_str().unwrap(), out.to_str().unwrap(), Some("fixture/huginn")).expect("import");
    let reader = checkpoint::weightio::WeightReader::open(out.to_str().unwrap()).expect("the brain container opens");
    for (name, want) in &values {
        let mut got = Vec::new();
        assert!(reader.with_tensor(name, &mut |t| got = t.to_vec()), "{name}");
        assert_eq!(&got, want, "{name}");
    }
    assert!(!reader.names().any(|n| n == "lm_head.weight" || n == "freqs_cis"), "the tied head and the rotary table are not parameters");

    let model = Huginn::new_inference(cfg.clone(), 1, 4, &reader);
    model.set_tokens(&[3, 4, 5, 6]);
    let logits = model.forward(cfg.mean_recurrence);
    assert!(!logits.is_empty() && logits.iter().all(|x| x.is_finite()));
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn a_tensor_the_architecture_does_not_have_is_refused_by_name() {
    let (dir, ..) = fixture("refused", &[("transformer.rotary_emb.inv_freq", vec![2])]);
    let err = open_dir(&dir).err().expect("an unmapped tensor is an error, not a silent drop");
    assert!(err.contains("inv_freq"), "{err}");
    std::fs::remove_dir_all(dir).ok();
}
