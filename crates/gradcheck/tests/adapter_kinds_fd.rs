// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for verifying that every
// parameter-efficient fine-tuning method trains correctly on every model, for
// its clients. If your team needs expertise in gradient verification for PEFT
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Gate A1: the device-side adapter variants (plain LoRA, rsLoRA, LoRA+,
//! per-target rank and alpha, LoRA-FA, IA3, LoKr, DoRA) match central finite differences on a tiny decoder with adapters on
//! every projection, and the parameter groups do what the variant says:
//! LoRA-FA leaves `A` exactly at its init while `B` trains, LoRA+ moves `B`
//! `lr_ratio` times as far as plain LoRA on the first Adam step.

use gradcheck::directional_check;
use model::adapter::device::{group_of, is_adapter_param};
use model::adapter::DeviceKind;
use model::Model;
use qwen3::{LoraCfg, Qwen, QwenConfig};

// seed: the finite-difference step and error tolerances below were tuned on this weight draw
const SEED: u64 = 7;

fn variants() -> Vec<(&'static str, LoraCfg)> {
    let plain = LoraCfg::new(2, 4.0, qwen3::finetune::LORA_TARGETS.iter().map(|t| t.to_string()).collect());
    vec![
        ("lora", plain.clone()),
        ("rslora", LoraCfg { rank_stabilized: true, ..plain.clone() }),
        ("lora+", LoraCfg { lr_ratio: 4.0, ..plain.clone() }),
        ("per-target", plain.clone().with_override("wq", Some(3), Some(3.0)).with_override("up", Some(1), Some(5.0))),
        ("lora-fa", LoraCfg { freeze_a: true, ..plain.clone() }),
        ("ia3", plain.clone().with_kind(DeviceKind::Ia3)),
        ("lokr", plain.clone().with_kind(DeviceKind::Lokr)),
        ("dora", plain.with_kind(DeviceKind::Dora)),
    ]
}

fn built(lora: LoraCfg) -> Qwen {
    let cfg = QwenConfig { lora: Some(lora), ..QwenConfig::tiny() };
    let init = qwen3::init_weights(&cfg, SEED);
    let model = Qwen::new(cfg, 2, 6, &init);
    let x: Vec<u32> = (0..12).map(|i| (i * 5 + 1) % 23).collect();
    let y: Vec<u32> = (0..12).map(|i| (i * 5 + 2) % 23).collect();
    model.set_batch(&x, &y);
    model
}

fn adapter_weights(m: &Qwen) -> Vec<(String, Vec<f32>)> {
    m.param_names().into_iter().filter(|n| is_adapter_param(n)).map(|n| (n.clone(), m.read_weight(&n))).collect()
}

fn train(m: &Qwen, steps: u32) {
    for step in 1..=steps {
        m.zero_grads();
        m.forward();
        m.backward();
        m.adamw_step(step, 5e-2, 0.0, Default::default(), Some(1.0), 1.0);
        m.poll_wait();
    }
}

#[test]
fn every_adapter_variant_matches_finite_differences() {
    for (name, lora) in variants() {
        let model = built(lora);
        // Off the zero-B start, so both factors carry gradient (see the LoRA
        // twin in decoder_variants_fd.rs for why the step is this small).
        train(&model, 5);
        let report = directional_check(&model, 1.25e-3, 4, SEED ^ 0x1234);
        let fails = report.failures(4e-3, 8e-2);
        assert!(fails.is_empty(), "{name}: {:?}", fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>());
        assert!(report.dead_gradients().is_empty(), "{name}: dead gradients");
    }
}

#[test]
fn lora_fa_keeps_a_at_its_init_and_trains_b() {
    let lora = variants().into_iter().find(|(n, _)| *n == "lora-fa").unwrap().1;
    let model = built(lora);
    let before = adapter_weights(&model);
    train(&model, 3);
    let hp = model.config().lora.as_ref().unwrap().hp();
    for ((name, was), (_, now)) in before.iter().zip(adapter_weights(&model)) {
        let frozen = !group_of(name, &hp).unwrap().trainable;
        assert_eq!(was == &now, frozen, "{name}: frozen={frozen} but unchanged={}", was == &now);
    }
}

#[test]
fn lora_plus_moves_b_by_the_ratio_on_the_first_step() {
    let moved = |lora: LoraCfg| -> f32 {
        let model = built(lora);
        let before = adapter_weights(&model);
        train(&model, 1);
        let hp = model.config().lora.as_ref().unwrap().hp();
        let mut total = 0.0f32;
        let mut count = 0usize;
        for ((name, was), (_, now)) in before.iter().zip(adapter_weights(&model)) {
            if group_of(name, &hp).unwrap().lr_mult != 1.0 || name.ends_with(model::adapter::kind::LORA_B) {
                total += was.iter().zip(&now).map(|(a, b)| (a - b).abs()).sum::<f32>();
                count += was.len();
            }
        }
        total / count as f32
    };
    let plain = moved(variants().remove(0).1);
    let plus = moved(variants().remove(2).1);
    let ratio = plus / plain;
    assert!((ratio - 4.0).abs() < 0.2, "LoRA+ (ratio 4) moved B {ratio}x as far as plain LoRA on step 1");
}

/// rsLoRA is live in the forward: the same factors give a different loss than
/// plain LoRA, and the difference is the scale (`alpha/sqrt(r)` vs `alpha/r`).
#[test]
fn rslora_changes_the_scale_the_forward_applies() {
    let loss_with = |lora: LoraCfg| -> f32 {
        let model = built(lora);
        for (name, w) in adapter_weights(&model) {
            let filled: Vec<f32> = (0..w.len()).map(|i| 0.05 * (((i * 7 + name.len()) % 11) as f32 - 5.0) / 5.0).collect();
            model.write_weight(&name, &filled);
        }
        model.forward()
    };
    let all = variants();
    let (plain, rs) = (loss_with(all[0].1.clone()), loss_with(all[1].1.clone()));
    assert!((plain - rs).abs() > 1e-6, "rsLoRA applied the plain scale: {plain} vs {rs}");
}

/// DoRA over a frozen base held in 4-bit NormalFloat or block-scaled FP8 (never expanded to fp32 on the device) is
/// differentiated correctly: the norm of the adapted weight rows is formed from tile-decoded base
/// values and the gradient reaches both factors and the magnitude vector.
#[test]
fn dora_over_a_packed_base_matches_finite_differences() {
    // Every contraction width a multiple of the 4-bit group (32).
    let cfg = QwenConfig {
        d_model: 64,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 16,
        d_ff: 128,
        lora: Some(LoraCfg::new(2, 4.0, qwen3::finetune::LORA_TARGETS.iter().map(|t| t.to_string()).collect()).with_kind(DeviceKind::Dora)),
        ..QwenConfig::tiny()
    };
    for dtype in [qwen3::Dtype::NF4, qwen3::Dtype::F8E4M3] {
        let init = qwen3::init_weights(&cfg, SEED);
        let model = Qwen::new_lora_dt(cfg.clone(), 2, 6, &init, dtype);
        assert_eq!(model.linear_dtype(), Some(dtype), "the base is held as asked, not widened");
        let x: Vec<u32> = (0..12).map(|i| (i * 5 + 1) % 23).collect();
        let y: Vec<u32> = (0..12).map(|i| (i * 5 + 2) % 23).collect();
        model.set_batch(&x, &y);
        train(&model, 5);
        let report = directional_check(&model, 1.25e-3, 4, SEED ^ 0x1234);
        let fails = report.failures(4e-3, 8e-2);
        assert!(fails.is_empty(), "{dtype:?}: {:?}", fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>());
        assert!(report.dead_gradients().is_empty(), "{dtype:?}: dead gradients");
    }
}
