// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One backward through two real layers of Ripple-65B (layer 0, sliding, and layer 4, global, with the real
//! embedding, final norm and `lm_head`), against PyTorch autograd of the official implementation in fp32 on the CPU.
//!
//! Swedish Embedded AB implements solutions for proving that ported large language models train exactly as their
//! reference implementations do, on real weights. If your team needs expertise in partial-checkpoint backward-pass
//! parity then you can procure our services by sending an email to info@swedishembedded.com.
//!
//! The weights are 5.2 GB of bf16 promoted to f32 (10 GB resident, plus gradients and the optimizer's moments),
//! so the test runs only where `testdata/ricursive/ripple-65b-partial/` and the golden
//! `tools/goldens/ricursive_dump_reference.py real-grads` writes are present. The reference's full gradients are
//! 10 GB, so the golden keeps each parameter's L2 norm and 4096 evenly strided entries; both are compared.

use std::path::Path;

use model::{Batch, Model};
use ricursive::{open_safetensors, Ripple, RicursiveConfig, TrainConfig, TrainOpts};

use crate::taps::{scaled_error, Taps};

const DUMPER: &str = "tools/goldens/ricursive_dump_reference.py real-grads";
const SOURCE_LAYERS: [usize; 2] = [0, 4];
const SAMPLES: usize = 4096;

/// Largest relative error of a gradient tensor's L2 norm, and largest scaled error of its sampled entries (measured
/// worst: 1.4e-5 and 4.8e-5, and 9.5e-7 for the loss; the stream passes through 5120- and 40960-wide f32
/// contractions, so these are looser than the tiny model's by the summation order over wider sums).
const NORM_TOLERANCE: f32 = 1e-4;
const SAMPLE_TOLERANCE: f32 = 3e-4;
const LOSS_TOLERANCE: f32 = 1e-5;

/// Spec: the loss of the real 2-layer model and, for every parameter, the gradient's norm and sampled entries
/// match PyTorch autograd.
#[test]
fn one_backward_through_real_layers_0_and_4_matches_pytorch() {
    let checkpoint = brain_testutil::testdata("ricursive/ripple-65b-partial/layers-0-4.safetensors");
    let config = brain_testutil::testdata("ricursive/ripple-65b-partial/config.json");
    let golden = brain_testutil::testdata("golden/ricursive/real_grads");
    for path in [&checkpoint, &config] {
        if !Path::new(path).exists() {
            brain_testutil::skip(&format!("{path} absent: place the partial Ripple-65B checkpoint under testdata/ricursive/ripple-65b-partial/"));
            return;
        }
    }
    let Some(source) = brain_testutil::golden::Source::open(Path::new(&golden), DUMPER) else { return };
    let cfg = RicursiveConfig::from_file(&config).expect("config").truncated(&SOURCE_LAYERS).expect("truncated config");
    if !source.require(&[("hidden_size", cfg.hidden_size as i64), ("num_hidden_layers", cfg.num_hidden_layers as i64), ("sliding_window", cfg.sliding_window as i64)]) {
        return;
    }
    let taps = Taps::read(&format!("{golden}/ricursive_real_grads.safetensors"));
    let ids = taps.input_ids();
    let targets: Vec<u32> = taps.get("targets").iter().map(|&t| if t < 0.0 { ricursive::train::IGNORE } else { t as u32 }).collect();

    let weights = open_safetensors(&checkpoint, &cfg, &SOURCE_LAYERS).expect("open checkpoint");
    let started = std::time::Instant::now();
    let model = Ripple::build(TrainConfig::new(cfg, ids.len() as u32), 1, ids.len() as u32, &weights, &TrainOpts::default(), true).expect("build");
    eprintln!("built in {:.1}s", started.elapsed().as_secs_f64());
    model.set_batch(Batch::Lm { tokens: &ids, targets: &targets });
    let started = std::time::Instant::now();
    model.zero_grads();
    let loss = model.forward();
    model.backward();
    model.poll_wait();
    eprintln!("forward and backward in {:.1}s", started.elapsed().as_secs_f64());

    let want = taps.get("loss")[0];
    let loss_err = (loss - want).abs() / want.abs();
    eprintln!("loss {loss} vs {want} (relative {loss_err:.2e})");
    assert!(loss_err <= LOSS_TOLERANCE, "loss {loss} vs {want}");

    let (mut worst_norm, mut worst_sample) = ((0.0f32, String::new()), (0.0f32, String::new()));
    for name in model.param_names() {
        let grad = model.read_grad(&name);
        let norm = (grad.iter().map(|g| f64::from(*g).powi(2)).sum::<f64>().sqrt()) as f32;
        let want_norm = taps.get(&format!("norm.{name}"))[0];
        let norm_err = (norm - want_norm).abs() / want_norm.max(f32::MIN_POSITIVE);
        let stride = (grad.len() / SAMPLES).max(1);
        let sample: Vec<f32> = grad.iter().step_by(stride).take(SAMPLES).copied().collect();
        let sample_err = scaled_error(&format!("grad sample {}", name.trim_start_matches("model.")), &sample, taps.get(&format!("sample.{name}")));
        if norm_err > worst_norm.0 {
            worst_norm = (norm_err, name.clone());
        }
        if sample_err > worst_sample.0 {
            worst_sample = (sample_err, name.clone());
        }
        assert!(want_norm > 0.0, "{name}: the reference gradient is dead");
    }
    eprintln!("worst gradient-norm error {:.2e} ({}), worst sampled-entry error {:.2e} ({})", worst_norm.0, worst_norm.1, worst_sample.0, worst_sample.1);
    assert!(worst_norm.0 <= NORM_TOLERANCE, "{}: gradient norm off by {:.2e}", worst_norm.1, worst_norm.0);
    assert!(worst_sample.0 <= SAMPLE_TOLERANCE, "{}: sampled gradient off by {:.2e}", worst_sample.1, worst_sample.0);
}
