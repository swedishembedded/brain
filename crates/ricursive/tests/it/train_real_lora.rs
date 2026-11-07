// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One LoRA training step over two real layers of Ripple-65B (layer 0, sliding, and layer 4, global, with the real
//! embedding, final norm and `lm_head`), against PyTorch autograd of the official implementation in fp32 on the CPU.
//!
//! Swedish Embedded AB implements solutions for proving that adapters train over a frozen very large model exactly
//! as the reference implementation says they do, on its real weights. If your team needs expertise in partial-
//! checkpoint LoRA parity then you can procure our services by sending an email to info@swedishembedded.com.
//!
//! The adapter is rank 8, scale 2, on all eight linears of both layers (`tools/goldens/ricursive_dump_reference.py
//! real-lora` seeds and dumps its factors; the base is frozen). The test starts from those factors and compares the
//! loss, the gradient of every A and B in full, and the factors and loss after one `torch.optim.AdamW` step. The base
//! is 5.2 GB of bf16 promoted to f32, so it runs only where `testdata/ricursive/ripple-65b-partial/` and the golden
//! are present.

use std::path::Path;

use model::adapter::device::{lora_names, LoraCfg};
use model::{Adam, Batch, Model};
use ricursive::train::{default_targets, linear_suffix, AdapterInit, LORA_LEAVES};
use ricursive::{open_safetensors, RicursiveConfig, Ripple, TrainConfig, TrainOpts};

use crate::taps::{scaled_error, Taps};

const DUMPER: &str = "tools/goldens/ricursive_dump_reference.py real-lora";
const SOURCE_LAYERS: [usize; 2] = [0, 4];
const RANK: u32 = 8;
const ALPHA: f32 = 16.0;
const LR: f32 = 1e-3;

/// Largest scaled error of a gradient tensor, and of the loss. The gradients pass through 5120- and 40960-wide f32
/// contractions, so these are looser than the tiny model's.
const GRAD_TOLERANCE: f32 = 3e-4;
const LOSS_TOLERANCE: f32 = 1e-5;
/// After the step: the first AdamW step moves every entry by `lr * sign(g)`, so an entry whose gradient is within
/// rounding of zero may move the other way (2 * lr). The factors are compared in the aggregate: the share of entries
/// that differ by more than a tenth of the step, and the step's loss.
const FLIPPED_SHARE: f64 = 2e-3;
const LOSS_AFTER_TOLERANCE: f32 = 2e-4;

fn base_names() -> Vec<String> {
    (0..2).flat_map(|layer| LORA_LEAVES.iter().map(move |leaf| ricursive::import::layer_tensor(layer, &linear_suffix(leaf)))).collect()
}

/// Spec: starting from the same factors, one LoRA step over the real 2-layer model has PyTorch's loss, gradients,
/// updated factors and loss.
#[test]
fn one_lora_step_over_real_layers_0_and_4_matches_pytorch() {
    let checkpoint = brain_testutil::testdata("ricursive/ripple-65b-partial/layers-0-4.safetensors");
    let config = brain_testutil::testdata("ricursive/ripple-65b-partial/config.json");
    let golden = brain_testutil::testdata("golden/ricursive/real_lora");
    for path in [&checkpoint, &config] {
        if !Path::new(path).exists() {
            brain_testutil::skip(&format!("{path} absent: place the partial Ripple-65B checkpoint under testdata/ricursive/ripple-65b-partial/"));
            return;
        }
    }
    let Some(source) = brain_testutil::golden::Source::open(Path::new(&golden), DUMPER) else { return };
    let model_cfg = RicursiveConfig::from_file(&config).expect("config").truncated(&SOURCE_LAYERS).expect("truncated config");
    if !source.require(&[("hidden_size", model_cfg.hidden_size as i64), ("num_hidden_layers", model_cfg.num_hidden_layers as i64), ("sliding_window", model_cfg.sliding_window as i64)]) {
        return;
    }
    let taps = Taps::read(&format!("{golden}/ricursive_real_lora.safetensors"));
    let ids = taps.input_ids();
    let targets: Vec<u32> = taps.get("targets").iter().map(|&t| if t < 0.0 { ricursive::train::IGNORE } else { t as u32 }).collect();

    let base = open_safetensors(&checkpoint, &model_cfg, &SOURCE_LAYERS).expect("open checkpoint");
    let cfg = TrainConfig::new(model_cfg, ids.len() as u32).with_lora(LoraCfg::new(RANK, ALPHA, default_targets()));
    let init = AdapterInit::fresh(&cfg, &base, 0); // the factors are overwritten with the reference's below
    let started = std::time::Instant::now();
    let model = Ripple::build(cfg, 1, ids.len() as u32, &init, &TrainOpts::default(), true).expect("build");
    eprintln!("built in {:.1}s", started.elapsed().as_secs_f64());
    let names = base_names();
    for base in &names {
        let (a, b) = lora_names(base);
        model.write_weight(&a, taps.get(&format!("a.{base}")));
        model.write_weight(&b, taps.get(&format!("b.{base}")));
    }
    assert_eq!(model.param_names().len(), 2 * names.len(), "only the adapter trains");

    model.set_batch(Batch::Lm { tokens: &ids, targets: &targets });
    model.zero_grads();
    let started = std::time::Instant::now();
    let loss = model.forward();
    model.backward();
    model.poll_wait();
    eprintln!("forward and backward in {:.1}s", started.elapsed().as_secs_f64());
    let want = taps.get("loss")[0];
    let loss_err = (loss - want).abs() / want.abs();
    eprintln!("loss {loss} vs {want} (relative {loss_err:.2e})");
    assert!(loss.is_finite() && loss_err <= LOSS_TOLERANCE, "loss {loss} vs {want}");

    let mut worst = (0.0f32, String::new());
    for base in &names {
        let (a, b) = lora_names(base);
        for (name, key) in [(a, "grad_a"), (b, "grad_b")] {
            let grad = model.read_grad(&name);
            assert!(grad.iter().all(|g| g.is_finite()), "{name}: non-finite gradient");
            let err = scaled_error(&format!("{key} {}", base.trim_start_matches("model.layers.")), &grad, taps.get(&format!("{key}.{base}")));
            if err > worst.0 {
                worst = (err, name);
            }
        }
    }
    eprintln!("worst adapter-gradient error {:.2e} ({})", worst.0, worst.1);
    assert!(worst.0 <= GRAD_TOLERANCE, "{}: gradient off by {:.2e}", worst.1, worst.0);

    model.adamw_step(1, LR, 0.0, Adam::default(), None, 1.0);
    model.poll_wait();
    let (mut total, mut flipped, mut worst_after) = (0usize, 0usize, 0.0f32);
    for base in &names {
        let (a, b) = lora_names(base);
        for (name, key) in [(a, "after_a"), (b, "after_b")] {
            let got = model.read_weight(&name);
            let reference = taps.get(&format!("{key}.{base}"));
            total += got.len();
            flipped += got.iter().zip(reference).filter(|(g, w)| (*g - *w).abs() > LR / 10.0).count();
            worst_after = worst_after.max(scaled_error(&format!("{key} {}", base.trim_start_matches("model.layers.")), &got, reference));
        }
    }
    let share = flipped as f64 / total as f64;
    eprintln!("after one AdamW step: {flipped} of {total} factor entries differ by more than a tenth of the step ({share:.2e}), worst scaled error {worst_after:.2e}");
    assert!(share <= FLIPPED_SHARE, "{flipped} of {total} entries moved differently");

    model.set_batch(Batch::Lm { tokens: &ids, targets: &targets });
    let after = model.forward();
    let want_after = taps.get("loss_after")[0];
    let after_err = (after - want_after).abs() / want_after.abs();
    eprintln!("loss after the step {after} vs {want_after} (relative {after_err:.2e}); before it {loss}");
    assert!(after < loss, "the step did not lower the loss: {loss} -> {after}");
    assert!(after_err <= LOSS_AFTER_TOLERANCE, "loss after the step {after} vs {want_after}");
}
