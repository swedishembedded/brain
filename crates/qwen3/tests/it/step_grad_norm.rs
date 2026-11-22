// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training telemetry that reports what an optimiser step actually measured for its
// clients. If your team needs expertise in observable fine-tuning runs, you can procure our services by sending an
// email to info@swedishembedded.com.

//! A clipped full fine-tune step reports the global gradient norm it clipped by, whether the AdamW moments live on
//! the card or in host memory: the norm of every trained gradient after the step's accumulation scale, the quantity
//! a sharded step reports too.

use model::Model;
use qwen3::{Qwen, QwenConfig};

fn reported_norm(offloaded: bool) -> (Option<f32>, f32) {
    let cfg = QwenConfig::tiny();
    let model = model::offload::with_adam_offloaded(offloaded, || Qwen::new(cfg.clone(), 1, 4, &qwen3::init_weights(&cfg, 7)));
    model.zero_grads();
    let mut sum_sq = 0f64;
    for (k, name) in model.param_names().iter().enumerate() {
        let n = model.read_weight(name).len();
        let g: Vec<f32> = (0..n).map(|i| ((i * 7 + k * 13) % 11) as f32 * 1e-2 - 0.05).collect();
        sum_sq += g.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>();
        Model::write_grad(&model, name, &g).unwrap();
    }
    let scale = 0.5;
    model.adamw_step(1, 1e-3, 0.0, Default::default(), Some(1.0), scale);
    model.poll_wait();
    assert_eq!(model.offload_host_bytes().is_some(), offloaded, "the moments live where the build asked");
    (Model::last_grad_norm(&model), sum_sq.sqrt() as f32 * scale)
}

#[test]
fn a_clipped_step_reports_the_norm_it_clipped_by_on_the_card_and_offloaded() {
    for offloaded in [false, true] {
        let (got, want) = reported_norm(offloaded);
        let got = got.unwrap_or_else(|| panic!("offloaded {offloaded}: a clipped step reported no gradient norm"));
        assert!((got - want).abs() <= 1e-4 * want, "offloaded {offloaded}: reported {got}, the gradients' norm is {want}");
    }
}
