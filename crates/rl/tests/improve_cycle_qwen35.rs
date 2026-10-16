// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for self-improving fine-tuning loops
// on hybrid linear-attention language models for its clients. If your team
// needs expertise in rollout, verification, GRPO and gated promotion over
// Gated-DeltaNet decoders then you can procure our services by sending an
// email to info@swedishembedded.com.

//! One improve cycle (rollout, verify, GRPO, evaluate, gate) on a tiny
//! `qwen35::Qwen35`, where it used to panic for want of token objectives. The
//! cycle must run through, keep the structural no-label property, and leave a
//! well-formed adapter file when it promotes.

mod grpo_common;

use grpo_common::{always_variable, config, parity_reward, Env, Judge, T};
use model::FitOpts;
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::Qwen35;
use rl::env::Environment;
use rl::gate::GateConfig;
use rl::improve::{self, AdapterMeta, CycleArtifacts, Evaluation, ProvenanceInput};
use rl::objective::grpo::{CycleLog, Grpo, GrpoOptions};

fn write_base(path: &std::path::Path, cfg: &Qwen35Config, seed: u64) {
    let init = qwen35::init::init_weights(cfg, seed);
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg.param_list().into_iter().map(|(name, n)| (name.clone(), vec![n as u64], init[&name].clone())).collect();
    checkpoint::save(path.to_str().unwrap(), cfg.to_json(), &tensors);
}

#[test]
fn an_improve_cycle_runs_on_qwen35() {
    let seed = data::rng::random_seed();
    println!("seed {seed}");
    let dir = tempfile::tempdir().unwrap();
    let cfg = Qwen35Config { block_size: T as u32, lora: Some(lora_cfg(3, 6.0)), ..Qwen35Config::tiny() };
    let base = dir.path().join("base.safetensors");
    write_base(&base, &cfg, seed);

    let env = Env { variable: always_variable };
    let held_out = env.tasks(900_000);
    let log = CycleLog::new();
    let (judge, _) = Judge::new(parity_reward);
    let objective = Grpo::new(env, judge, config(2, 1.0, GrpoOptions::default())).with_log(log.clone());
    let opts = FitOpts { steps: 6, grad_accum: 2, lr: 5e-3, min_lr: 5e-4, decay_iters: 6, grad_clip: 1.0, seed, ..FitOpts::default() };
    let targets = qwen35::config::lora_targets();
    let (eval_judge, _) = Judge::new(parity_reward);
    let outcome = improve::cycle::<Qwen35, _>(
        &base,
        objective,
        &Evaluation {
            held_out: &held_out,
            anchor: &[],
            verifier: &eval_judge,
            rollout: &model::rollout::RolloutParams { max_new: 4, sample: model::serve::SampleParams::greedy(), eos: None },
            gate_cfg: &GateConfig { min_entropy_ratio: 0.0, ..GateConfig::default() },
            anchor_block_len: 0,
        },
        &opts,
        CycleArtifacts {
            train_out: &dir.path().join("train.safetensors"),
            adapter_out_dir: &dir.path().join("adapters"),
            adapter: AdapterMeta { rank: 3, alpha: 6.0, targets: &targets, family: "qwen35", base_id: "base", dataset_id: None },
            provenance: ProvenanceInput { regime: "grpo".to_string(), seed, hyperparams: serde_json::json!({"group_size": 2}), environment: "cpu/test".to_string(), cycle: 0 },
        },
    )
    .expect("the cycle runs to a decision");

    improve::assert_trained_spans_were_sampled(&log.trained(), &log.sampled());
    assert!(!log.trained().is_empty(), "the cycle trained on sampled spans");
    if let Some(adapter) = outcome.adapter_path {
        let st = checkpoint::st::load_safetensors(adapter.to_str().unwrap()).expect("a promoted adapter loads");
        assert_eq!(st.card().expect("a model card").family, "qwen35");
    }
}
