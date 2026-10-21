// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded low-precision full fine-tuning for its clients. If your team needs
// expertise in training hybrid linear-attention models with 8-bit and bf16 optimizer state then you can procure
// our services by sending an email to info@swedishembedded.com.

//! A full fine-tune of the tiny hybrid Qwen3.5 takes a precision policy through the generic optimizer-state
//! seam: the device AdamW keeps its moments in bf16 or int8 blocks (the fused `adamw_bf16` and `adamw_q8`
//! kernels) and trains like the f32 default; a choice no kernel runs is refused with the parameter's name.

use gpu_core::Gpu;
use paramstore::precision::{ParamPrecision, PrecisionPolicy, StateFormat};
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35};

fn loss_after(policy: &PrecisionPolicy, steps: u32) -> f32 {
    let cfg = Qwen35Config::tiny();
    let t = cfg.block_size;
    let init = qwen35::init::init_weights(&cfg, 11);
    let m = model::offload::with_precision_policy(policy, || Qwen35::new_train_on(Gpu::new(pipelines()), cfg.clone(), 2, t, &init));
    let x: Vec<u32> = (0..2 * t).map(|i| (i * 7) % cfg.vocab).collect();
    let y: Vec<u32> = (0..2 * t).map(|i| (i * 7 + 1) % cfg.vocab).collect();
    m.set_batch(&x, &y);
    for step in 1..=steps {
        m.zero_grads();
        m.forward();
        m.backward();
        m.adamw_step(step, 1e-2, 0.0, Default::default(), Some(1.0), 1.0);
        m.poll_wait();
    }
    m.forward()
}

#[test]
fn bf16_and_int8_device_moments_train_like_f32_and_unsupported_choices_are_refused() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    let with = |state, sr| PrecisionPolicy::uniform(ParamPrecision { state, stochastic_rounding: sr, ..ParamPrecision::FP32 });
    let base = loss_after(&with(StateFormat::F32, false), 6);
    let bf16 = loss_after(&with(StateFormat::Bf16, true), 6);
    let int8 = loss_after(&with(StateFormat::Int8Block { block: 64 }, true), 6);
    println!("loss after 6 steps: f32 {base:.4}, bf16 moments {bf16:.4}, int8 moments {int8:.4}");
    assert!(bf16 < 1.3 * base + 0.05, "bf16 moments {bf16} should track f32 {base}");
    assert!(int8 < 1.3 * base + 0.05, "int8 moments {int8} should track f32 {base}");

    let log8 = with(StateFormat::Log8Block { block: 64 }, false);
    let e = std::panic::catch_unwind(|| loss_after(&log8, 1)).expect_err("a state no device kernel runs is refused");
    let msg = e.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(msg.contains("host-offloaded"), "{msg}");
}
