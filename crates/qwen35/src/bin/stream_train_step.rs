// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One phase of a real streamed LoRA fine-tune against the real 64-layer
//! `Qwen/Qwen3.8-27B-FP8` checkpoint - a standalone CLI so a single training
//! run can be CHECKPOINTED across several short process invocations. This
//! exists because one real forward+backward step against the real
//! checkpoint costs tens of minutes (`qwen35::stream_train`'s own module
//! doc: a step re-streams all 64 layers' weights TWICE), which this
//! development environment's own background-process lifetime is too short
//! to run end-to-end in one invocation - the adapter's own small `.lora_a`/
//! `.lora_b` state is written to `--adapter-out` after every phase and read
//! back from `--adapter-in` by the next one, so the SAME logical training
//! run spans several short processes without losing progress.
//!
//! Swedish Embedded AB implements streaming LoRA fine-tuning over
//! disk-streamed transformer checkpoints for its clients. If your team
//! needs expertise in training large models that do not fit in device or
//! host memory, you can procure our services by emailing
//! info@swedishembedded.com.
//!
//! Usage:
//! ```text
//! stream_train_step --dir <checkpoint dir> --phase before|step|after|load \
//!     --tokenizer <tokenizer.json> [--adapter-in FILE] --adapter-out FILE \
//!     [--device cpu|gpu|gpuN|vulkan|auto --base-dtype stored|nf4 --upload-staging-mib N --step N --lr X --window-budget N --n-tokens N \
//!      --rank N --alpha X --prompt "..." --max-new N --corpus FILE]
//! ```
//!
//! `--device` defaults to `cpu`, the only backend proven safe for `--phase
//! step` against this model's real checkpoint: the resident fp32 `lm_head`
//! (~4.74 GiB as one buffer) exceeds `wgpu`'s `max_buffer_size` on any
//! non-NVIDIA Linux adapter, so `gpu` refuses to allocate it at all; the
//! native Vulkan backend (`vulkan`) allocates it but crashes the device
//! with `ERROR_DEVICE_LOST` on the backward pass. Both remain selectable
//! here for forward-only smoke testing and for re-validating once
//! `lm_head` is chunked into sub-`max_buffer_size` buffers.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use gpu_core::Gpu;
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::pipelines;
use qwen35::stream_train::StreamTrainer;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn save_adapter(path: &str, tensors: &HashMap<String, Vec<f32>>) {
    let list: Vec<(String, Vec<u64>, Vec<f32>)> = tensors.iter().map(|(k, v)| (k.clone(), vec![v.len() as u64], v.clone())).collect();
    checkpoint::st::save_safetensors(path, &list, &serde_json::json!({}), None).unwrap_or_else(|e| panic!("save_adapter: {path}: {e}"));
}

fn load_adapter(path: &str) -> HashMap<String, Vec<f32>> {
    checkpoint::st::load_safetensors(path).unwrap_or_else(|e| panic!("load_adapter: {path}: {e}")).tensors
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(arg(&args, "--dir").expect("--dir required"));
    let phase = arg(&args, "--phase").expect("--phase required (before|step|after)");
    let tokenizer = PathBuf::from(arg(&args, "--tokenizer").unwrap_or_else(|| dir.join("tokenizer.json").to_string_lossy().into_owned()));
    let adapter_out = arg(&args, "--adapter-out").unwrap_or_default();
    let adapter_in = arg(&args, "--adapter-in");
    let step: u32 = arg(&args, "--step").and_then(|s| s.parse().ok()).unwrap_or(1);
    let lr: f32 = arg(&args, "--lr").and_then(|s| s.parse().ok()).unwrap_or(0.05);
    let window_budget: u32 = arg(&args, "--window-budget").and_then(|s| s.parse().ok()).unwrap_or(2);
    let n_tokens: u32 = arg(&args, "--n-tokens").and_then(|s| s.parse().ok()).unwrap_or(16);
    let rank: u32 = arg(&args, "--rank").and_then(|s| s.parse().ok()).unwrap_or(4);
    let alpha: f32 = arg(&args, "--alpha").and_then(|s| s.parse().ok()).unwrap_or(8.0);
    let prompt = arg(&args, "--prompt").unwrap_or_else(|| "The capital of France is".to_string());
    let max_new: usize = arg(&args, "--max-new").and_then(|s| s.parse().ok()).unwrap_or(3);
    let corpus = arg(&args, "--corpus");
    // Pinned staging ring the layer uploads go through on a transfer stream (0: plain uploads on the compute stream).
    let upload_staging_mib: u64 = arg(&args, "--upload-staging-mib").and_then(|s| s.parse().ok()).unwrap_or(64);
    let device = arg(&args, "--device").unwrap_or_else(|| "cpu".to_string());

    let mut cfg = Qwen35Config::qwen38_27b();
    cfg.lora = Some(lora_cfg(rank, alpha));

    let lora_init = match &adapter_in {
        Some(p) => load_adapter(p),
        None => qwen35::init::init_lora_only(&cfg, 20260820),
    };

    let gpu = match device.as_str() {
        "cpu" => Gpu::new_cpu(pipelines()),
        "gpu" => Gpu::new_gpu(pipelines()),
        "auto" => Gpu::new(pipelines()),
        "vulkan" => Gpu::try_new_vulkan(pipelines()).unwrap_or_else(|e| panic!("try_new_vulkan: {e}")),
        // `gpuN`: the physical card the reservation helper granted (`bin/card` exports `--device gpuN`).
        card if card.strip_prefix("gpu").is_some_and(|n| n.parse::<u32>().is_ok()) => {
            let index: u32 = card["gpu".len()..].parse().expect("checked above");
            Gpu::new_on_index(index, pipelines()).unwrap_or_else(|e| panic!("--device {card}: {e}"))
        }
        other => panic!("unknown --device {other} (expected cpu|gpu|gpuN|vulkan|auto)"),
    };
    let t0 = Instant::now();
    let trainer = StreamTrainer::new_real_staged(gpu, &cfg, &dir, n_tokens, window_budget, &lora_init, upload_staging_mib << 20).unwrap_or_else(|e| panic!("new_real: {e}"));
    eprintln!("stream_train_step: trainer construction: {:.1}s", t0.elapsed().as_secs_f64());
    let convert = match arg(&args, "--base-dtype").as_deref() {
        None | Some("stored") => None,
        Some("nf4") => Some(gpu_core::select::Dtype::NF4),
        Some(other) => panic!("unknown --base-dtype {other} (expected stored|nf4)"),
    };
    let loader = trainer.real_loader_as(&cfg, &dir, convert).unwrap_or_else(|e| panic!("real_loader: {e}"));

    match phase.as_str() {
        "before" | "after" => {
            let t0 = Instant::now();
            let text = trainer.generate_greedy(&cfg, &loader, &dir, &tokenizer, &prompt, max_new).unwrap_or_else(|e| panic!("generate_greedy: {e}"));
            println!("PHASE={phase} MINUTES={:.2} PROMPT={prompt:?} OUTPUT={text:?}", t0.elapsed().as_secs_f64() / 60.0);
        }
        "step" => {
            assert!(!adapter_out.is_empty(), "--adapter-out required for --phase step");
            let corpus = corpus.expect("--corpus required (path to the Step 2 dataset's corpus.txt) for --phase step");
            let text = std::fs::read_to_string(&corpus).unwrap_or_else(|e| panic!("read {corpus}: {e}"));
            let tok = data::qwen_tokenizer::QwenBpe::from_file(tokenizer.to_str().unwrap()).unwrap_or_else(|e| panic!("load tokenizer: {e}"));
            let ids = data::tokenizer::Tokenizer::encode(&tok, &text);
            assert!(ids.len() > n_tokens as usize, "corpus too short: {} tokens", ids.len());
            let tokens = ids[0..n_tokens as usize].to_vec();
            let targets = ids[1..(n_tokens as usize + 1)].to_vec();
            let x0 = StreamTrainer::embed_real(&dir, &tokens, cfg.d_model as usize).unwrap_or_else(|e| panic!("embed_real: {e}"));

            let t0 = Instant::now();
            trainer.lora.zero_grads(&trainer.gpu);
            // `BRAIN_PROFILE=1` prints the per-kernel device time of the step (the shared profile table).
            let loss = gpu_core::profile::device_table(&trainer.gpu, "streamed step", || trainer.forward_backward(&cfg, &loader, &x0, &targets));
            trainer.lora.adamw_step(&trainer.gpu, step, lr);
            trainer.gpu.poll_wait();
            println!("PHASE=step STEP={step} LOSS={loss:.6} MINUTES={:.2}", t0.elapsed().as_secs_f64() / 60.0);
            let (waited, computed) = trainer.pass_timing();
            println!("PASS_TIMING waited_for_load_s={waited:.2} computed_s={computed:.2}");

            let mut out: HashMap<String, Vec<f32>> = HashMap::new();
            for (name, _) in cfg.param_list() {
                if model::adapter::device::is_adapter_param(&name) {
                    out.insert(name.clone(), trainer.lora.ps.read_weight(&trainer.gpu, &name));
                }
            }
            save_adapter(&adapter_out, &out);
        }
        // The loader alone: every layer read, converted if asked, and uploaded once, with no compute, through the same
        // bounded prefetch a step uses (`--window-budget` layers alive, one fewer loader threads). Separates the
        // streaming cost (disk, host staging, upload) from the device time of a step.
        "load" => {
            let t0 = Instant::now();
            let window = window_budget.max(2) as usize;
            let order: Vec<usize> = (0..cfg.n_layers as usize).collect();
            let stats = model::layer_stream::for_each_prefetched(&order, window - 1, window, &loader, |_, _| trainer.gpu.poll_wait());
            println!("PHASE=load LAYERS={} SECONDS={:.2} WAITED={:.2}", cfg.n_layers, t0.elapsed().as_secs_f64(), stats.waited.as_secs_f64());
        }
        other => panic!("unknown --phase {other} (expected before|step|after|load)"),
    }
}
