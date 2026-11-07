// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Training parity against PyTorch autograd of the official `modeling_ricursive.py`: the loss and the gradient of
//! every parameter on a batch, and a short `torch.optim.AdamW` trajectory with global-norm clipping.
//!
//! Swedish Embedded AB implements solutions for proving that a ported model trains exactly as its reference does,
//! gradient by gradient, for its clients. If your team needs expertise in backward-pass parity validation of large
//! language models then you can procure our services by sending an email to info@swedishembedded.com.
//!
//! Run `python -I tools/goldens/ricursive_dump_reference.py grads --modeling-dir <dir with modeling_ricursive.py>
//! --out testdata/golden/ricursive` to produce the fixture (the weights are the reference's own random draw, so
//! the fixture is self-contained); without it the tests skip by name.
//!
//! The reference model is the 3-layer tiny of the forward parity test: a 4-key window over a 12-token prompt, a
//! nonzero `value_res_lambda` per layer and a clip of 0.5 that both fires and saturates, so every backward path is
//! on. The tolerances state what the two f32 implementations can differ by: summation order, the `exp`/`sin`/`cos`
//! of each implementation and the folded value-residual lerp.

use std::path::Path;

use model::{Batch, Model};
use optim::Adam;
use ricursive::{open_safetensors, Ripple, RicursiveConfig, TrainConfig, TrainOpts};

use crate::taps::{scaled_error, Taps};

const DUMPER: &str = "tools/goldens/ricursive_dump_reference.py grads";

/// Largest scaled error (`max|got - want| / max|want|`) accepted per gradient tensor and per updated parameter.
const GRAD_TOLERANCE: f32 = 1e-4;
/// Relative error of a loss.
const LOSS_TOLERANCE: f32 = 1e-5;

struct Fixture {
    cfg: TrainConfig,
    taps: Taps,
    dir: String,
    train: serde_json::Value,
}

fn fixture() -> Option<Fixture> {
    let dir = brain_testutil::testdata("golden/ricursive/tiny_train");
    let source = brain_testutil::golden::Source::open(Path::new(&dir), DUMPER)?;
    let model = RicursiveConfig::from_file(Path::new(&dir).join("config.json").to_str().unwrap()).expect("config");
    if !source.require(&[("hidden_size", model.hidden_size as i64), ("num_hidden_layers", model.num_hidden_layers as i64), ("sliding_window", model.sliding_window as i64)]) {
        return None;
    }
    let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(Path::new(&dir).join("manifest.json")).unwrap()).unwrap();
    let taps = Taps::read(Path::new(&dir).join("ricursive_tiny_train.safetensors").to_str().unwrap());
    Some(Fixture { cfg: TrainConfig::new(model, taps.get("input_ids").len() as u32), taps, dir, train: manifest["train"].clone() })
}

fn build(f: &Fixture) -> Ripple {
    let layers: Vec<usize> = (0..f.cfg.model.num_hidden_layers).collect();
    let weights = open_safetensors(Path::new(&f.dir).join("ricursive_tiny_train_weights.safetensors").to_str().unwrap(), &f.cfg.model, &layers).expect("weights");
    let model = Ripple::build(f.cfg.clone(), 1, f.cfg.block_size, &weights, &TrainOpts::default(), true).expect("build");
    let ids: Vec<u32> = f.taps.input_ids();
    // The reference marks unsupervised positions -100.
    let targets: Vec<u32> = f.taps.get("targets").iter().map(|&t| if t < 0.0 { ricursive::train::IGNORE } else { t as u32 }).collect();
    model.set_batch(Batch::Lm { tokens: &ids, targets: &targets });
    model
}

fn relative(label: &str, got: f32, want: f32) -> f32 {
    let err = (got - want).abs() / want.abs().max(f32::MIN_POSITIVE);
    eprintln!("{label:>34}: got {got:.7} want {want:.7} relative {err:.3e}");
    err
}

/// Spec: the loss and every parameter's gradient equal PyTorch autograd's on the same weights and batch.
#[test]
fn the_loss_and_every_gradient_match_pytorch_autograd() {
    let Some(f) = fixture() else { return };
    let model = build(&f);
    model.zero_grads();
    let loss = model.forward();
    model.backward();
    model.poll_wait();
    let loss_err = relative("loss", loss, f.taps.get("loss")[0]);
    assert!(loss_err <= LOSS_TOLERANCE, "loss {loss} vs {} ({loss_err:.2e})", f.taps.get("loss")[0]);

    let mut worst = (0.0f32, String::new());
    let names = model.param_names();
    assert_eq!(names.len(), 3 * 15 + 3, "every parameter of the 3-layer model trains: 15 per layer, the embedding, the final norm and the head");
    for name in &names {
        let err = scaled_error(&format!("grad {}", name.trim_start_matches("model.")), &model.read_grad(name), f.taps.get(&format!("grad.{name}")));
        if err > worst.0 {
            worst = (err, name.clone());
        }
    }
    eprintln!("worst gradient error {:.3e} ({}), tolerance {GRAD_TOLERANCE:.0e}", worst.0, worst.1);
    assert!(worst.0 <= GRAD_TOLERANCE, "{}: scaled error {:.3e} exceeds {GRAD_TOLERANCE:.0e}", worst.1, worst.0);
}

/// Spec: three AdamW steps with global-norm clipping follow `torch.optim.AdamW` + `clip_grad_norm_`: the losses on
/// the way and every parameter at the end.
#[test]
fn an_adamw_trajectory_follows_torch() {
    let Some(f) = fixture() else { return };
    let t = &f.train;
    let (lr, wd, clip) = (t["lr"].as_f64().unwrap() as f32, t["weight_decay"].as_f64().unwrap() as f32, t["clip"].as_f64().unwrap() as f32);
    let adam = Adam { beta1: t["beta1"].as_f64().unwrap() as f32, beta2: t["beta2"].as_f64().unwrap() as f32, eps: t["eps"].as_f64().unwrap() as f32 };
    let steps = t["steps"].as_u64().unwrap() as u32;
    let model = build(&f);
    let want_losses = f.taps.get("trajectory.loss");
    for step in 1..=steps {
        model.zero_grads();
        let loss = model.forward();
        model.backward();
        let err = relative(&format!("loss at step {step}"), loss, want_losses[step as usize - 1]);
        assert!(err <= 1e-4, "step {step}: loss {loss} vs {}", want_losses[step as usize - 1]);
        model.adamw_step(step, lr, wd, adam, Some(clip), 1.0);
        model.poll_wait();
    }
    let after = model.forward();
    let err = relative("loss after the last step", after, f.taps.get("trajectory.loss_after")[0]);
    assert!(err <= 1e-4, "loss after {after} vs {}", f.taps.get("trajectory.loss_after")[0]);
    assert!(after < want_losses[0], "the steps lower the loss");

    let mut worst = (0.0f32, String::new());
    for name in model.param_names() {
        let e = scaled_error(&format!("final {}", name.trim_start_matches("model.")), &model.read_weight(&name), f.taps.get(&format!("final.{name}")));
        if e > worst.0 {
            worst = (e, name);
        }
    }
    eprintln!("worst parameter error after {steps} steps {:.3e} ({})", worst.0, worst.1);
    assert!(worst.0 <= GRAD_TOLERANCE, "{}: scaled error {:.3e} exceeds {GRAD_TOLERANCE:.0e}", worst.1, worst.0);
}
