// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for verifying that every
// parameter-efficient fine-tuning method trains correctly on every model, for
// its clients. If your team needs expertise in gradient verification for PEFT
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Gate A1 on the second architecture (hybrid Gated-DeltaNet and attention):
//! LoRA, IA3 and LoKr adapters on every targetable linear match central finite
//! differences.

use gpu_core::Gpu;
use gradcheck::directional_check;
use model::adapter::DeviceKind;
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35};

#[test]
fn every_adapter_kind_matches_finite_differences_on_the_hybrid_model() {
    for kind in DeviceKind::ALL {
        let cfg = Qwen35Config { lora: Some(lora_cfg(2, 4.0).with_kind(kind)), ..Qwen35Config::tiny() };
        let t = cfg.block_size;
        let init = qwen35::init::init_weights(&cfg, 7);
        let m = Qwen35::new_train_on(Gpu::new_cpu(pipelines()), cfg.clone(), 1, t, &init);
        let x: Vec<u32> = (0..t).map(|i| (i * 3 + 1) % cfg.vocab).collect();
        let y: Vec<u32> = (0..t).map(|i| (i * 3 + 2) % cfg.vocab).collect();
        m.set_batch(&x, &y);
        // Off the invisible start, so every tensor carries gradient.
        for step in 1..=5 {
            m.zero_grads();
            m.forward();
            m.backward();
            m.adamw_step(step, 5e-2, 0.0, Default::default(), Some(1.0), 1.0);
            m.poll_wait();
        }
        let report = directional_check(&m, 1.25e-3, 4, 7 ^ 0x1234);
        let fails = report.failures(4e-3, 8e-2);
        assert!(fails.is_empty(), "{}: {:?}", kind.name(), fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>());
        assert!(report.dead_gradients().is_empty(), "{}: dead gradients", kind.name());
    }
}
