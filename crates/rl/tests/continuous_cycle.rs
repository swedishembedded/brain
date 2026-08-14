// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`rl::continuous::run_cycle`] coverage. The real-training branch (ingest
//! -> `fit_weighted` -> save adapter) reuses machinery already proven
//! elsewhere: `fit_weighted` itself by `qwen3_fit_weighted.rs`'s
//! convergence test, `save_adapter`/`fold_adapter_into` by `qwen3::lora`'s
//! own round-trip test and the cross-crate ones in `qwen35moe`/
//! `deepseek2`. What's net-new here is `run_cycle`'s OWN wiring - the
//! empty-trajectories short circuit, and (given a real training pass) that
//! it actually produces a loadable, versioned adapter file - so that is
//! what this file checks, not re-proving the training math again.

use qwen3::config::QwenConfig;

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-rl-run-cycle-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_base_checkpoint(path: &std::path::Path, vocab: u32, seed: u64) {
    let cfg = QwenConfig { vocab, ..QwenConfig::tiny() };
    let init = qwen3::init_weights(&cfg, seed);
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
        .param_list()
        .into_iter()
        .map(|(name, n)| {
            let v = init.get(&name).unwrap_or_else(|| panic!("init missing {name}")).clone();
            (name, vec![n as u64], v)
        })
        .collect();
    checkpoint::save(path.to_str().unwrap(), cfg.to_json(), &tensors);
}

fn tiny_tok() -> data::qwen_tokenizer::QwenBpe {
    use checkpoint::gguf::GgufTokenizer;
    let gt = GgufTokenizer {
        model: "gpt2".into(),
        pre: Some("qwen2".into()),
        tokens: vec!["<|endoftext|>".into(), "<|im_start|>".into(), "<|im_end|>".into(), "h".into(), "i".into(), "hi".into()],
        merges: vec!["h i".into()],
        token_types: vec![3, 3, 3, 1, 1, 1],
        bos: Some(0),
        eos: Some(2),
        unk: None,
        pad: None,
    };
    data::qwen_tokenizer::QwenBpe::from_gguf(&gt).unwrap()
}

fn tiny_tmpl() -> data::chat_template::ChatTemplate {
    data::chat_template::ChatTemplate::compile("{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}").unwrap()
}

#[test]
fn run_cycle_is_a_quiet_no_op_when_no_trajectories_are_waiting() {
    if skip() {
        return;
    }
    let dir = tmp("empty");
    let trajectories = dir.join("trajectories");
    std::fs::create_dir_all(&trajectories).unwrap();
    let base_path = dir.join("base.safetensors");
    write_base_checkpoint(&base_path, 23, 1);

    let result = rl::continuous::run_cycle(
        &trajectories,
        &base_path,
        &dir.join("train.safetensors"),
        &dir.join("adapters"),
        &tiny_tok(),
        &tiny_tmpl(),
        2,
        4.0,
        &model::FitOpts::default(),
    )
    .expect("run_cycle");
    assert!(result.is_none(), "no trajectories waiting must be a quiet no-op, not an error or a spurious adapter");
    assert!(!dir.join("train.safetensors").exists(), "must not even start training when there is nothing to train on");
}

#[test]
fn run_cycle_trains_and_produces_a_versioned_loadable_adapter() {
    if skip() {
        return;
    }
    let dir = tmp("real");
    let trajectories = dir.join("trajectories");
    std::fs::create_dir_all(&trajectories).unwrap();
    let base_path = dir.join("base.safetensors");
    write_base_checkpoint(&base_path, 23, 1);

    // One reward-stamped trajectory: enough for ingest_dir to produce a
    // non-empty weighted dataset. Content stays within the tiny synthetic
    // tokenizer's own vocab ("h"/"i"/"hi") and is long enough to clear
    // TokenDataset's `block_size + 1` minimum window requirement.
    let traj_json = serde_json::json!({
        "schema_version": "ATIF-v1.7",
        "agent": {"name": "test-agent", "version": "0.0.1"},
        "final_metrics": {"extra": {"reward": 1.0}},
        "steps": [
            {"step_id": 1, "source": "user", "message": "hihihihihi"},
            {"step_id": 2, "source": "agent", "message": "hihihihihihihihihi"}
        ]
    });
    std::fs::write(trajectories.join("t1.json"), serde_json::to_string(&traj_json).unwrap()).unwrap();

    let opts = model::FitOpts { steps: 5, batch_size: 2, block_size: 4, ..Default::default() };
    let adapter_path = rl::continuous::run_cycle(
        &trajectories,
        &base_path,
        &dir.join("train.safetensors"),
        &dir.join("adapters"),
        &tiny_tok(),
        &tiny_tmpl(),
        2,
        4.0,
        &opts,
    )
    .expect("run_cycle")
    .expect("a reward-stamped trajectory must produce an adapter");

    assert_eq!(adapter_path.file_name().unwrap().to_str().unwrap(), "adapter-000000.safetensors", "the first adapter of a fresh out dir must be version 0");
    assert!(dir.join("train.safetensors").exists(), "the resumable full training checkpoint must exist after a real cycle");

    // The produced adapter must be a real, loadable LoRA adapter -- fold it
    // into a copy of the base and confirm it actually changes something
    // (same assertion shape as qwen3::lora's own round-trip test).
    let base = checkpoint::load(base_path.to_str().unwrap());
    let mut folded = base.by_role("");
    let before = folded.clone();
    qwen3::lora::fold_adapter_into(&mut folded, adapter_path.to_str().unwrap()).expect("fold_adapter_into");
    let any_changed = before.iter().any(|(name, v)| folded.get(name).map(|after| after != v).unwrap_or(false));
    assert!(any_changed, "the produced adapter must actually change the folded base weights");
}
