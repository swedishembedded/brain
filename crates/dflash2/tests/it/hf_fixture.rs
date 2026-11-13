// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for loading HuggingFace model releases as published for its clients. If
// your team needs expertise in keeping a model zoo loadable from upstream layouts then you can procure our services
// by sending an email to info@swedishembedded.com.

// hf-fixture: dflash2
//! A tiny DFlash2 draft in the release's own layout (`layers.{i}.*`, `fc.weight`, `hidden_norm.weight`,
//! `candidate_selector.*`, a `config.json` with `dflash_config`, `layer_types` and `no_rope_layers`) read through
//! [`HfDraft`]: the config, every tensor's shape and values, the selector's codebook rows, a load onto a device and a
//! denoising round on it, and a draft missing a tensor or shaped against its config refused by name.

use dflash2::{Dflash2, Dflash2Config, HfDraft};
use dflash2::source::{DraftSource, Leaf, Tensor};

const CONFIG: &str = r#"{
  "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 3, "num_attention_heads": 4,
  "num_key_value_heads": 2, "head_dim": 4, "rms_norm_eps": 1e-6, "sliding_window": 8, "use_sliding_window": true,
  "layer_types": ["sliding_attention", "full_attention", "full_attention"], "no_rope_layers": [1, 1, 0],
  "rope_parameters": {"rope_theta": 10000.0}, "draft_vocab_size": 23, "num_target_layers": 6,
  "dflash_config": {"block_size": 4, "conv_kernel_size": 3, "conv_group_size": 4, "selector_rank": 4,
                    "selector_top_k": 4, "mask_token_id": 5, "target_layer_ids": [1, 3]}
}"#;

/// Every tensor of the draft, by role, with the shape the release stores it in.
fn layout(cfg: &Dflash2Config) -> Vec<(Tensor, Vec<u64>)> {
    let (d, ff) = (u64::from(cfg.d_model), u64::from(cfg.d_ff));
    let (hq, hkv, hd) = (u64::from(cfg.q_dim()), u64::from(cfg.kv_dim()), u64::from(cfg.head_dim));
    let (taps, dynamic, rank, vocab) = (u64::from(cfg.conv_kernel), u64::from(cfg.dyn_width()), u64::from(cfg.selector_rank), u64::from(cfg.vocab));
    let mut all = vec![
        (Tensor::Fc, vec![d, cfg.tap_width() as u64]),
        (Tensor::HiddenNorm, vec![d]),
        (Tensor::OutNorm, vec![d]),
        (Tensor::SelectorHidden, vec![rank, d]),
        (Tensor::SelectorPredecessor, vec![vocab, rank]),
        (Tensor::SelectorSuccessor, vec![vocab, rank]),
    ];
    for l in 0..cfg.n_layers {
        let t = |leaf| Tensor::Layer(l, leaf);
        all.extend([
            (t(Leaf::InputNorm), vec![d]),
            (t(Leaf::PostAttnNorm), vec![d]),
            (t(Leaf::Q), vec![hq, d]),
            (t(Leaf::K), vec![hkv, d]),
            (t(Leaf::V), vec![hkv, d]),
            (t(Leaf::O), vec![d, hq]),
            (t(Leaf::QNorm), vec![hd]),
            (t(Leaf::KNorm), vec![hd]),
            (t(Leaf::Gate), vec![ff, d]),
            (t(Leaf::Up), vec![ff, d]),
            (t(Leaf::Down), vec![d, ff]),
            (t(Leaf::AttnConvBase), vec![2, taps, d]),
            (t(Leaf::AttnConvProj), vec![dynamic, d]),
            (t(Leaf::MlpConvBase), vec![2, taps, d]),
            (t(Leaf::MlpConvProj), vec![dynamic, d]),
        ]);
    }
    all
}

/// Small deterministic values; a norm gain is near one so the forward stays in range.
fn values(tensor: Tensor, n: usize, seed: u64) -> Vec<f32> {
    let norm = matches!(tensor, Tensor::HiddenNorm | Tensor::OutNorm | Tensor::Layer(_, Leaf::InputNorm | Leaf::PostAttnNorm | Leaf::QNorm | Leaf::KNorm));
    // A fixed scatter of values in [0, 1): enough that no two rows are alike.
    let unit = |i: usize| ((i as u64 * 2_654_435_761 + seed * 40_503) % 1009) as f32 / 1009.0;
    (0..n).map(|i| if norm { 1.0 + 0.1 * unit(i) } else { 0.1 * (unit(i) - 0.5) }).collect()
}

/// A draft directory with every tensor but `without`, and the tensors it was written from.
fn fixture(tag: &str, without: Option<Tensor>, reshape: Option<(Tensor, Vec<u64>)>) -> (std::path::PathBuf, Dflash2Config, Vec<(Tensor, Vec<f32>)>) {
    let cfg = Dflash2Config::from_json(CONFIG).expect("the config parses");
    let dir = std::env::temp_dir().join(format!("brain-dflash2-hf-{tag}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    let mut written = Vec::new();
    let mut tensors = Vec::new();
    for (i, (tensor, shape)) in layout(&cfg).into_iter().enumerate() {
        let data = values(tensor, shape.iter().product::<u64>() as usize, 100 + i as u64);
        written.push((tensor, data.clone()));
        if Some(tensor) == without {
            continue;
        }
        let shape = match &reshape {
            Some((t, s)) if *t == tensor => s.clone(),
            _ => shape,
        };
        tensors.push((tensor.name_hf(), shape, data));
    }
    checkpoint::st::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &tensors, &serde_json::Value::Null, None).unwrap();
    std::fs::write(dir.join("config.json"), CONFIG).unwrap();
    (dir, cfg, written)
}

#[test]
fn the_release_layout_loads_and_denoises_a_block() {
    let (dir, cfg, written) = fixture("loads", None, None);
    assert_eq!((cfg.n_layers, cfg.vocab, cfg.block_size, cfg.target_layers.clone()), (3, 23, 4, vec![1, 3]));
    assert_eq!(cfg.layers.iter().map(|l| l.rope).collect::<Vec<_>>(), [true, true, false], "no_rope_layers marks the NoPE layer");

    // Every tensor is there with the shape it was written in and the values it was written with.
    let src = HfDraft::open(&dir).expect("the draft opens");
    for (tensor, data) in &written {
        let shape = layout(&cfg).into_iter().find(|(t, _)| t == tensor).unwrap().1;
        assert_eq!(src.shape(*tensor), Some(shape.iter().map(|&x| x as usize).collect()), "{}", tensor.name_hf());
        if !matches!(tensor, Tensor::SelectorPredecessor | Tensor::SelectorSuccessor) {
            let got = src.vector(*tensor).unwrap();
            assert!(got.iter().zip(data).all(|(a, b)| (a - b).abs() <= 1e-2 * b.abs().max(1e-3)), "{}", tensor.name_hf());
        }
    }
    // The selector's codebooks are read by row.
    let (_, codebook) = written.iter().find(|(t, _)| *t == Tensor::SelectorPredecessor).unwrap();
    let rows = src.rows(Tensor::SelectorPredecessor, 7, 2, cfg.selector_rank as usize).unwrap();
    let rank = cfg.selector_rank as usize;
    assert!(rows.iter().zip(&codebook[7 * rank..9 * rank]).all(|(a, b)| (a - b).abs() <= 1e-2 * b.abs().max(1e-3)));

    // The draft loads on a device and denoises a block of its own width.
    let gpu = gpu_core::Gpu::new_cpu(&dflash2::pipelines(&cfg));
    let draft = Dflash2::load(gpu, cfg.clone(), Box::new(src), 1, 16).expect("the draft loads");
    let wide = cfg.tap_width();
    let context: Vec<f32> = (0..6 * wide).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();
    draft.append_context(0, &context, 0).expect("the target's taps are fed in");
    let noise: Vec<f32> = (0..cfg.block_size as usize * cfg.d_model as usize).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();
    let out = draft.denoise_block(&noise, 6).expect("a round");
    assert_eq!(out.len(), noise.len());
    assert!(out.iter().all(|v| v.is_finite()));
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn a_draft_missing_a_tensor_or_shaped_against_its_config_is_refused_by_name() {
    let missing = Tensor::Layer(1, Leaf::AttnConvProj);
    let (dir, cfg, _) = fixture("missing", Some(missing), None);
    let gpu = gpu_core::Gpu::new_cpu(&dflash2::pipelines(&cfg));
    let err = Dflash2::load(gpu, cfg.clone(), Box::new(HfDraft::open(&dir).unwrap()), 1, 16).err().expect("a missing tensor is an error");
    assert!(err.contains(&missing.name_hf()), "{err}");
    std::fs::remove_dir_all(dir).ok();

    let wrong = Tensor::Layer(0, Leaf::Q);
    let (dir, cfg, _) = fixture("reshaped", None, Some((wrong, vec![8, 32])));
    let gpu = gpu_core::Gpu::new_cpu(&dflash2::pipelines(&cfg));
    let err = Dflash2::load(gpu, cfg, Box::new(HfDraft::open(&dir).unwrap()), 1, 16).err().expect("a shape the config does not imply is an error");
    assert!(err.contains(&wrong.name_hf()) && err.contains("config implies"), "{err}");
    std::fs::remove_dir_all(dir).ok();
}
