// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements QLoRA fine-tuning of published quantized mixture-of-experts releases for its clients.
// If your team needs expertise in adapting such a model without dequantizing it then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: a LoRA run over the released checkpoint as published - NVFP4 and MXFP8 expert banks read in place, bf16 dense
//! linears, every adapter site including the stacked expert pairs - lowers the loss of a short supervised row in a few
//! steps, and the adapter it saves, read back over a fresh load of the release, computes the trained loss.

use data::tokenizer::Tokenizer;
use inkling2::train::{targets, BuildOptions, Inkling2Train};
use model::Model;

const REPO: &str = "thinkingmachines/Inkling-2-Mini";
const TEXT: &str = "The capital of France is Paris. The capital of Italy is Rome. The capital of Spain is Madrid.";

#[test]
fn a_qlora_run_over_the_released_banks_lowers_the_loss_and_its_adapter_reloads() {
    let Some(dir) = brain_testutil::model_dir(REPO).map(std::path::PathBuf::from).filter(|d| d.join("hf_quant_config.json").is_file()) else {
        brain_testutil::skip_unavailable(&format!("{REPO} is not in the model store"));
        return;
    };
    let bpe = data::qwen_tokenizer::QwenBpe::from_file(&dir.join("tokenizer.json").to_string_lossy()).expect("the release's tokenizer");
    let tokens = bpe.encode(TEXT);
    let t = tokens.len() as u32 - 1;
    let (x, y) = (tokens[..t as usize].to_vec(), tokens[1..].to_vec());
    let seed = data::rng::random_seed();
    eprintln!("seed {seed}, {t} tokens");
    let lora = model::adapter::device::LoraCfg::new(4, 8.0, targets(true));
    let base = model::ops::TierPolicy::uniform(gpu_core::select::Dtype::BF16);
    let opts = BuildOptions { batch: 1, seq: t, base: &base, seed, continue_from: None };
    let started = std::time::Instant::now();
    let m = Inkling2Train::from_dir_lora(&dir, lora.clone(), &opts).unwrap();
    eprintln!("loaded in {:.0} s", started.elapsed().as_secs_f64());
    assert!(m.param_names().iter().any(|n| n.contains("w13_weight.lora_a")), "the stacked expert banks are adapted");
    let mut losses = Vec::new();
    for step in 1..=3u32 {
        let at = std::time::Instant::now();
        m.zero_grads();
        m.set_batch(model::Batch::Lm { tokens: &x, targets: &y });
        let loss = m.forward();
        m.backward();
        m.adamw_step(step, 2e-3, 0.0, model::Adam::default(), Some(1.0), 1.0);
        eprintln!("step {step}: loss {loss:.4} ({:.0} s)", at.elapsed().as_secs_f64());
        assert!(loss.is_finite(), "step {step}: loss {loss}");
        losses.push(loss);
    }
    m.set_batch(model::Batch::Lm { tokens: &x, targets: &y });
    let trained = m.forward();
    eprintln!("after the steps: loss {trained:.4}");
    assert!(trained < losses[0], "the loss went from {} to {trained}", losses[0]);

    let scratch = brain_testutil::Scratch::dir("inkling2-real-adapter");
    let path = scratch.path().join("adapter.safetensors");
    inkling2::finetune::save_adapter_file(&m, &path).unwrap();
    drop(m);
    let reopened = BuildOptions { continue_from: Some(&path), ..opts };
    let r = Inkling2Train::from_dir_lora(&dir, lora, &reopened).unwrap();
    r.set_batch(model::Batch::Lm { tokens: &x, targets: &y });
    let again = r.forward();
    assert!((again - trained).abs() <= 1e-4 * trained.abs().max(1.0), "the reloaded adapter's loss {again}, the trained model's {trained}");
}
