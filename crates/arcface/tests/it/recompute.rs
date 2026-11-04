// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of face recognition models for its clients. If your team
// needs expertise in fine-tuning embedding networks within a fixed device budget, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Activation recompute for the IResNet backbone, held to the contract `model::saved::check_recompute` states: a
//! trainer that keeps only each residual block's output and rebuilds the block's `bn1 -> conv1 -> prelu` branch from
//! the block's input before its backward computes the loss and every gradient of the trainer that keeps them, bit for
//! bit, at a lower step peak. `ArcFaceTrainer` has no `model::Model` impl for that function to take (there is no
//! image/label batch variant), so this test runs the same clauses on the trainer itself. One test in its own binary,
//! so nothing else allocates while the device bytes are compared.

use arcface::{ArcFaceTrainConfig, ArcFaceTrainer};
use model::saved::{SavedFormat, SavedPolicy};

/// A step's loss, every parameter's gradient bits and the live device bytes it peaked at.
type Measured = (f32, Vec<(String, Vec<u32>)>, Option<u64>);

fn init(name: &str, n: usize) -> Vec<f32> {
    if name.ends_with(".running_var") || name.ends_with("bn1.weight") {
        vec![1.0; n]
    } else {
        (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect()
    }
}

fn trainer() -> ArcFaceTrainer {
    let gpu = gpu_core::testgpu::dev(arcface::model::PIPELINES);
    ArcFaceTrainer::new_on(gpu, ArcFaceTrainConfig::tiny(), &init)
}

/// One training step under `policy`.
fn measured(policy: SavedPolicy) -> Measured {
    let mut t = trainer();
    t.set_saved_policy(policy).unwrap();
    let cfg = t.config().clone();
    let n = (cfg.batch * 3 * cfg.arc.image_size * cfg.arc.image_size) as usize;
    let img: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let labels: Vec<u32> = (0..cfg.batch).map(|i| i % cfg.classes).collect();
    t.set_batch(&img, &labels);
    t.gpu().reset_memory_peak();
    t.zero_grads();
    let loss = t.loss();
    t.backward();
    t.gpu().poll_wait();
    let peak = t.gpu().memory_usage().map(|u| u.peak_live);
    let mut names = t.param_names();
    names.sort();
    let grads = names.into_iter().map(|n| (n.clone(), t.read_grad(&n).into_iter().map(f32::to_bits).collect())).collect();
    (loss, grads, peak)
}

#[test]
fn arcface_recomputes_its_block_branches_exactly() {
    let (want, want_grads, keep_peak) = measured(SavedPolicy::Keep);
    let (got, got_grads, recompute_peak) = measured(SavedPolicy::Recompute);
    if let (Some(keep), Some(recompute)) = (keep_peak, recompute_peak) {
        assert!(recompute < keep, "the recomputing step peaks at {recompute} live device bytes, keeping every branch {keep}");
    }
    assert_eq!(got.to_bits(), want.to_bits(), "recomputing gives loss {got}, keeping {want}");
    assert!(want_grads.iter().any(|(_, g)| g.iter().any(|b| f32::from_bits(*b) != 0.0)), "no parameter received a gradient to compare");
    assert_eq!(want_grads.len(), got_grads.len());
    for ((n, a), (_, b)) in want_grads.iter().zip(&got_grads) {
        assert!(a == b, "{n}: gradient differs recomputing");
    }

    // Narrow saved formats do not apply, and a recomputing trainer does not go back to keeping.
    let mut t = trainer();
    assert!(t.set_saved_policy(SavedPolicy::Quantized(SavedFormat::Bf16)).unwrap_err().contains("arcface"));
    t.set_saved_policy(SavedPolicy::Recompute).unwrap();
    assert!(t.set_saved_policy(SavedPolicy::Keep).unwrap_err().contains("arcface"));
}
