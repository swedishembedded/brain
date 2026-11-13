// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for loading HuggingFace model releases as published for its clients. If
// your team needs expertise in keeping a model zoo loadable from upstream layouts then you can procure our services
// by sending an email to info@swedishembedded.com.

// hf-fixture: ouro
//! A tiny Ouro checkpoint in the release's own layout (HF tensor names and `[out, in]` linear shapes, the exit gate as
//! `model.early_exit_gate.*`, an untied `lm_head.weight`, a `config.json` of the HF keys) read through the importer:
//! the config, the two-way coverage of the rename plan, the values the files were written from, an unknown tensor
//! refused by name, the import into a brain container, and a forward on the imported weights.

use checkpoint::TensorSource;
use ouro::config::OuroConfig;
use ouro::import::{import_as, open_dir};
use ouro::init::init_weights;
use ouro::model::Ouro;

/// The HF name and shape of the brain parameter `name`.
fn hf_layout(cfg: &OuroConfig, name: &str) -> (String, Vec<u64>) {
    let (d, ff, v) = (u64::from(cfg.d_model), u64::from(cfg.d_ff), u64::from(cfg.vocab));
    let (hq, hkv) = (u64::from(cfg.q_dim()), u64::from(cfg.kv_dim()));
    let layer = |l: &str, leaf: &str| format!("model.layers.{l}.{leaf}");
    match name {
        "tok.weight" => ("model.embed_tokens.weight".into(), vec![v, d]),
        "norm.weight" => ("model.norm.weight".into(), vec![d]),
        "gate.weight" => ("model.early_exit_gate.weight".into(), vec![1, d]),
        "gate.bias" => ("model.early_exit_gate.bias".into(), vec![1]),
        "lm_head.weight" => ("lm_head.weight".into(), vec![v, d]),
        _ => {
            let rest = name.strip_prefix("blocks.").expect("a block parameter");
            let (l, leaf) = rest.split_once('.').unwrap();
            let (hf, shape) = match leaf {
                "ln1.weight" => ("input_layernorm.weight", vec![d]),
                "ln1_2.weight" => ("input_layernorm_2.weight", vec![d]),
                "ln2.weight" => ("post_attention_layernorm.weight", vec![d]),
                "ln2_2.weight" => ("post_attention_layernorm_2.weight", vec![d]),
                "attn.wq.weight" => ("self_attn.q_proj.weight", vec![hq, d]),
                "attn.wk.weight" => ("self_attn.k_proj.weight", vec![hkv, d]),
                "attn.wv.weight" => ("self_attn.v_proj.weight", vec![hkv, d]),
                "attn.wo.weight" => ("self_attn.o_proj.weight", vec![d, hq]),
                "mlp.gate.weight" => ("mlp.gate_proj.weight", vec![ff, d]),
                "mlp.up.weight" => ("mlp.up_proj.weight", vec![ff, d]),
                "mlp.down.weight" => ("mlp.down_proj.weight", vec![d, ff]),
                other => panic!("no HF name for {other}"),
            };
            (layer(l, hf), shape)
        }
    }
}

fn fixture(tag: &str, extra: &[(&str, Vec<u64>)]) -> (std::path::PathBuf, OuroConfig, std::collections::HashMap<String, Vec<f32>>) {
    let cfg = OuroConfig::tiny();
    let values = init_weights(&cfg, 5);
    let dir = std::env::temp_dir().join(format!("brain-ouro-hf-{tag}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
        .param_list()
        .into_iter()
        .map(|(name, numel)| {
            let (hf, shape) = hf_layout(&cfg, &name);
            assert_eq!(shape.iter().product::<u64>() as usize, numel, "{name}");
            (hf, shape, values[&name].clone())
        })
        .collect();
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

    // The config.json reads to the preset the fixture was written from, and the rename plan covers every parameter
    // with nothing left over.
    let (read, src) = open_dir(&dir).expect("every parameter is covered and every tensor consumed");
    assert_eq!(format!("{read:?}"), format!("{cfg:?}"));
    for (name, want) in &values {
        let mut got = Vec::new();
        assert!(src.with_tensor(name, &mut |t| got = t.to_vec()), "{name} has a source");
        assert_eq!(&got, want, "{name}");
    }

    // The import writes a brain container whose card names the family and whose tensors are the same values.
    let out = dir.join("ouro.brain.safetensors");
    import_as(dir.to_str().unwrap(), out.to_str().unwrap(), Some("fixture/ouro")).expect("import");
    let reader = checkpoint::weightio::WeightReader::open(out.to_str().unwrap()).expect("the brain container opens");
    for (name, want) in &values {
        let mut got = Vec::new();
        assert!(reader.with_tensor(name, &mut |t| got = t.to_vec()), "{name}");
        assert_eq!(&got, want, "{name}");
    }

    // The imported weights run the forward.
    let model = Ouro::new_inference(cfg.clone(), 1, 4, &reader);
    model.set_tokens(&[3, 4, 5, 6]);
    model.trunk_forward();
    model.head_forward(0);
    assert!(model.read_logits().iter().all(|x| x.is_finite()));
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn a_tensor_the_architecture_does_not_have_is_refused_by_name() {
    let (dir, ..) = fixture("refused", &[("model.layers.0.self_attn.rotary_emb.inv_freq", vec![2])]);
    let err = open_dir(&dir).err().expect("an unmapped tensor is an error, not a silent drop");
    assert!(err.contains("inv_freq"), "{err}");
    std::fs::remove_dir_all(dir).ok();
}
