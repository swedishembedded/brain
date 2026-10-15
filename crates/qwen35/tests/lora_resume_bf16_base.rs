// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Resume gate for a LoRA run on a streamed bf16 base (the counterpart of
//! `lora_resume_gate.rs`, which covers the fp32 base): a second cycle with
//! `resume` must start from the adapter the first cycle wrote, not from zero
//! delta, and the written adapter must be an adapter-only file.

use data::binio::{self, Meta};
use qwen35::base_source::BaseDtype;
use qwen35::config::Qwen35Config;
use qwen35::finetune::{finetune_streamed, Mode};
use qwen35::model::{pipelines, Qwen35};

#[test]
#[ignore = "red until lane A2's bf16 base tier is wired into qwen35::finetune::finetune_streamed"]
fn a_resumed_bf16_base_lora_run_starts_from_the_saved_adapter() {
    let dir = std::env::temp_dir().join(format!("brain-qwen35-resume-bf16-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let cfg = Qwen35Config::tiny();
    let t = cfg.block_size;
    // seed: any base weights; the gate compares cycle 2's start against cycle 1's end
    let init = qwen35::init::init_weights(&cfg, 7);
    let base = dir.join("base.safetensors");
    Qwen35::new_on(gpu_core::Gpu::new(pipelines()), cfg.clone(), 1, t, &init).save(base.to_str().unwrap());
    let data = dir.join("data");
    std::fs::create_dir_all(&data).unwrap();
    let tokens: Vec<u32> = (0..800).map(|i| [2, 5, 8, 15][i % 4]).collect();
    binio::write_u32_bin(&data.join("train.u32.bin"), &tokens).unwrap();
    binio::write_u32_bin(&data.join("val.u32.bin"), &[]).unwrap();
    std::fs::write(data.join("meta.json"), Meta::vocab_only(cfg.vocab as usize)).unwrap();
    let out = dir.join("adapter.safetensors");
    let out_s = out.to_str().unwrap();
    let mode = Mode::Lora { rank: 5, alpha: 8.0 };
    let opts = |seed| model::FitOpts { steps: 8, batch_size: 1, block_size: t, lr: 5e-2, min_lr: 5e-3, warmup: 2, decay_iters: 8, checkpoint_secs: 0, eval_interval: 0, eval_batches: 0, seed, ..Default::default() };

    let (first_start, _) = finetune_streamed(&base, &data, &opts(1), &mode, out_s, false, BaseDtype::Bf16, true).expect("cycle 1");
    let adapter = checkpoint::st::load_safetensors(out_s).unwrap();
    assert!(adapter.tensors.keys().all(|k| k.ends_with(".lora_a") || k.ends_with(".lora_b")), "the output must be adapter-only");
    let (second_start, _) = finetune_streamed(&base, &data, &opts(2), &mode, out_s, true, BaseDtype::Bf16, true).expect("cycle 2");
    assert!(second_start < first_start, "resume restarted from zero delta: {first_start} then {second_start}");
}
