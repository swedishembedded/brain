// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements parameter-efficient fine-tuning on
// consumer GPUs for its clients. If your team needs expertise in adapting
// open-weight decoders to your data then you can procure our services by
// sending an email to info@swedishembedded.com.

//! A LoRA run that asks for snapshots writes the adapter of every held-out
//! evaluation, each a loadable adapter file of the shape the run trains, and
//! the last of them is the adapter the run ends with.

use std::path::PathBuf;

use data::binio::{self, Meta};
use model::{AdapterSnapshots, FitControl, Model};
use qwen3::{Dtype, LoraCfg, Qwen, QwenConfig};

const VOCAB: u32 = 24;
const PROMPT: [u32; 3] = [2, 5, 8];
const TARGET: u32 = 15;

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-lora-snapshots-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_dataset(dir: &std::path::Path) {
    let mut tokens = Vec::new();
    let mut mask = Vec::new();
    for _ in 0..200 {
        tokens.extend_from_slice(&PROMPT);
        tokens.push(TARGET);
        mask.extend_from_slice(&[false, false, false, true]);
    }
    for split in ["train", "val"] {
        binio::write_u32_bin(&dir.join(format!("{split}.u32.bin")), &tokens).unwrap();
        binio::write_mask_bin(&dir.join(format!("{split}.mask.bin")), &mask).unwrap();
    }
    std::fs::write(dir.join("meta.json"), Meta::vocab_only(VOCAB as usize)).unwrap();
}

#[test]
fn every_evaluation_of_a_lora_run_leaves_a_loadable_adapter() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let scratch = tmp("run");
    write_dataset(&scratch);
    let cfg = QwenConfig {
        vocab: VOCAB,
        block_size: 16,
        n_layers: 2,
        d_model: 16,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        d_ff: 32,
        max_position_embeddings: 16,
        lora: Some(LoraCfg::attn(3, 6.0)),
        ..QwenConfig::tiny()
    };
    let init = qwen3::init_weights(&cfg, 7);
    let opts = model::FitOpts {
        steps: 12,
        batch_size: 1,
        block_size: 16,
        lr: 2e-2,
        min_lr: 2e-3,
        warmup: 2,
        decay_iters: 12,
        weight_decay: 0.0,
        grad_clip: 1.0,
        eval_interval: 4,
        eval_batches: 0,
        seed: 1234,
        ..Default::default()
    };
    let (train, val, bcfg, _, itos) = model::load_dataset_with_itos(&scratch, &opts).unwrap();
    let objective = model::causal_lm::<Qwen>(train, val, bcfg, itos);
    let snapshots = scratch.join("snapshots");
    std::fs::create_dir_all(&snapshots).unwrap();
    let control = FitControl {
        snapshots: Some(AdapterSnapshots {
            dir: &snapshots,
            card_id: "test:adapter",
            base_id: "test:base",
            family: "qwen",
            rank: 3,
            alpha: 6.0,
            targets: LoraCfg::attn(3, 6.0).targets,
        }),
        ..Default::default()
    };
    let model = Qwen::new_lora_dt(cfg, 1, 16, &init, Dtype::F32);
    let (report, trained) = model::fit_controlled(model, objective, &opts, None, control).unwrap();
    assert_eq!(report.evaluations.len(), 3);

    for step in [4, 8, 12] {
        let path = snapshots.join(format!("step-{step}.safetensors"));
        assert!(path.is_file(), "an adapter after step {step}");
        let adapter = qwen3::lora::read_adapter(path.to_str().unwrap()).unwrap();
        assert_eq!((adapter.rank, adapter.alpha), (3, 6.0));
        assert!(!adapter.sites.is_empty());
    }
    // The run's own end is the last snapshot: what was written is what the
    // model holds.
    let last = qwen3::lora::read_adapter(snapshots.join("step-12.safetensors").to_str().unwrap()).unwrap();
    let held = trained.param_names().into_iter().find(|n| n.ends_with(".lora_b")).unwrap();
    let site = last
        .sites
        .iter()
        .find(|s| held.starts_with(&s.base.trim_end_matches(".weight").to_string()))
        .unwrap();
    assert_eq!(site.b, trained.read_weight(&held));
    let first = qwen3::lora::read_adapter(snapshots.join("step-4.safetensors").to_str().unwrap()).unwrap();
    assert_ne!(first.sites[0].b, last.sites[0].b, "the adapter moved between evaluations");
}
