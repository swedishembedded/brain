// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded low-precision full fine-tuning for its clients. If your team needs
// expertise in training with 8-bit optimizer state then you can procure our services by sending an email to
// info@swedishembedded.com.

//! A full fine-tune of a tiny random Qwen3 takes a precision policy through the
//! generic optimizer-state seam: int8 and bf16 moments with the state host-offloaded train
//! like the fp32 default (host-offloaded and on the device), and an unsupported choice
//! is refused rather than ignored.

use std::collections::HashMap;

use model::TrainPrecision;
use paramstore::precision::{MasterPolicy, ParamPrecision, StateFormat};
use qwen3::model::FullFtBuild;
use qwen3::{Qwen, QwenConfig};

fn tiny() -> QwenConfig {
    QwenConfig {
        vocab: 64,
        block_size: 16,
        n_layers: 2,
        d_model: 32,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        d_ff: 64,
        rope_theta: 1.0e6,
        rms_eps: 1e-6,
        max_position_embeddings: 16,
        tie_embeddings: false,
        qk_norm: true,
        attn_bias: false,
        lora: None,
        rope_scaling: None,
    }
    .with_defaults()
}

fn init(cfg: &QwenConfig) -> HashMap<String, Vec<f32>> {
    let mut s = 0x1234_5678_9abc_def0u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2
    };
    cfg.param_list()
        .into_iter()
        .map(|(name, numel)| {
            let one = name.ends_with("norm.weight");
            (name, (0..numel).map(|_| if one { 1.0 + next() } else { next() }).collect())
        })
        .collect()
}

/// Mean loss after `steps` AdamW steps on one fixed batch, or the builder's refusal.
fn train(precision: &TrainPrecision, offload: bool, steps: u32) -> Result<f32, String> {
    let cfg = tiny();
    let w = init(&cfg);
    let t = 8u32;
    let x: Vec<u32> = (0..t).map(|i| (i * 7 + 3) % cfg.vocab).collect();
    let y: Vec<u32> = (0..t).map(|i| (i * 7 + 10) % cfg.vocab).collect();
    let full = FullFtBuild { precision: precision.clone(), offload, resident: None };
    let m = Qwen::new_full(cfg, 1, t, &w, gpu_core::provider::library::GemmPrecision::Fp32Reference, &full)?;
    let mut last = 0.0;
    for step in 1..=steps {
        m.zero_grads();
        m.set_batch(&x, &y);
        last = model::Model::forward(&m);
        model::Model::backward(&m);
        m.adamw_step(step, 3e-3, 0.0, Default::default(), Some(1.0), 1.0);
    }
    Ok(last)
}

#[test]
fn int8_moments_offloaded_train_like_fp32_and_unsupported_choices_are_refused() {
    let fp32 = TrainPrecision::default();
    let int8 = TrainPrecision::uniform(ParamPrecision { state: StateFormat::Int8Block { block: 64 }, stochastic_rounding: true, ..ParamPrecision::FP32 });
    let base = train(&fp32, true, 12).unwrap();
    let q8 = train(&int8, true, 12).unwrap();
    println!("loss after 12 steps: fp32 moments {base:.4}, int8 moments {q8:.4}");
    assert!(q8 < 1.2 * base + 0.05, "int8 moments {q8} should track fp32 {base}");

    // int8 moments run on the device optimiser too (the fused adamw_q8 kernels)
    let q8_dev = train(&int8, false, 12).unwrap();
    println!("device int8 moments {q8_dev:.4}");
    assert!(q8_dev < 1.2 * base + 0.05, "device int8 moments {q8_dev} should track fp32 {base}");

    // bf16 moments run on the device optimiser too (the fused adamw_bf16 kernel)
    let bf16_dev = train(&TrainPrecision::uniform(ParamPrecision { state: StateFormat::Bf16, stochastic_rounding: true, ..ParamPrecision::FP32 }), false, 12).unwrap();
    println!("device bf16 moments {bf16_dev:.4}");
    assert!(bf16_dev < 1.2 * base + 0.05, "device bf16 moments {bf16_dev} should track fp32 {base}");

    // logarithmic 8-bit moments on the device optimiser and a master copy are refused with an error naming the parameter
    let log8 = TrainPrecision::uniform(ParamPrecision { state: StateFormat::Log8Block { block: 64 }, ..ParamPrecision::FP32 });
    let e = train(&log8, false, 1).expect_err("refused");
    assert!(e.contains("host-offloaded") && e.contains("tok.weight"), "{e}");
    let master = TrainPrecision::uniform(ParamPrecision { master: MasterPolicy::F32, ..ParamPrecision::FP32 });
    let e = train(&master, true, 1).expect_err("refused");
    assert!(e.contains("no master") && e.contains("tok.weight"), "{e}");
}
