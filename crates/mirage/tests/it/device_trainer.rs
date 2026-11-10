// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements resumable GPU fine-tuning of compact perception models for its clients. If your
// team needs expertise in full, head-only and low-rank fine-tuning with exact resume on accelerators then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: the device trainer fine-tunes in every mode like the host trainer (same losses, same updates), resumes
//! exactly, and writes checkpoints and adapters in checkpoint layout that load again.

use mirage::device::train::trainer::DevTrainer;
use mirage::device::Device;
use mirage::train::trainer::{FineTuner, Head, Mode, Prepared, Trainer};

use crate::common::tiny_checkpoint;
use crate::train_fit::{cfg_for, prepared, scratch};

fn device(test: &str) -> Option<Device> {
    crate::common::cap_threads();
    match Device::open() {
        Ok(d) => Some(d),
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("{test}: no device: {e}"));
            None
        }
    }
}

fn dev_trainer(dev: Device, mode: Mode) -> (DevTrainer, Prepared) {
    let ck = tiny_checkpoint();
    let p = prepared(&ck.config);
    (DevTrainer::new(dev, ck, cfg_for(mode)).unwrap(), p)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
    let (na, nb): (f64, f64) = (a.iter().map(|x| f64::from(*x).powi(2)).sum(), b.iter().map(|x| f64::from(*x).powi(2)).sum());
    if na == 0.0 && nb == 0.0 {
        1.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

#[test]
fn device_steps_follow_the_host_trainer_in_every_mode() {
    let _g = gpu_core::testgpu::exclusive();
    for mode in [Mode::Full, Mode::Heads(vec![Head::Reader, Head::Dense]), Mode::Lora { rank: 2, alpha: 4.0, seed: 5 }] {
        let Some(dev) = device("device vs host trainer") else { return };
        let ck = tiny_checkpoint();
        let p = prepared(&ck.config);
        let batch = [p.clone_prepared(), p.clone_prepared()];
        let on_cpu = dev.gpu().kind() == "cpu";
        // A GPU's tensor-core GEMMs round at ~1e-3: the losses agree to that and the Adam updates a little less.
        let (rel, cos_floor) = if on_cpu { (1e-4, 0.999) } else { (2e-3, 0.98) };
        let mut host = Trainer::new(ck.clone(), cfg_for(mode.clone())).unwrap();
        let mut device = DevTrainer::new(dev, ck.clone(), cfg_for(mode.clone())).unwrap();
        for s in 0..3 {
            let (h, d) = (host.step(&batch).unwrap(), device.step(&batch).unwrap());
            eprintln!("{mode:?} step {s}: loss host {:.6} device {:.6}, grad norm host {:.4e} device {:.4e}", h.loss, d.loss, h.grad_norm, d.grad_norm);
            assert!((h.loss - d.loss).abs() <= rel * h.loss.abs().max(1.0), "loss {} vs {}", h.loss, d.loss);
            assert!((h.grad_norm - d.grad_norm).abs() <= 10.0 * rel * h.grad_norm.max(1e-6), "grad norm {} vs {}", h.grad_norm, d.grad_norm);
        }
        // The trained tensors agree (the updates are Adam steps of near-identical gradients).
        let (hc, dc) = (host.checkpoint(), device.checkpoint());
        let mut worst = 1.0f64;
        for (name, t) in &hc.tensors {
            let base = &ck.tensors[name].data;
            let (dh, dd): (Vec<f32>, Vec<f32>) = (t.data.iter().zip(base).map(|(a, b)| a - b).collect(), dc.tensors[name].data.iter().zip(base).map(|(a, b)| a - b).collect());
            // The residual-routing tensors receive a gradient below f32 resolution (~1e-8, noise on both sides); Adam
            // normalises its size away, so their update direction is not comparable.
            if dh.iter().any(|v| *v != 0.0) && !name.contains("_res") {
                let c = cosine(&dh, &dd);
                if c < 0.99 {
                    eprintln!("  {name}: update cosine {c:.4}");
                }
                worst = worst.min(c);
            }
        }
        eprintln!("{mode:?}: worst cosine of the weight updates {worst:.6}");
        assert!(worst >= cos_floor, "{mode:?}: weight updates differ (cosine {worst})");
        if let (Some(h), Some(d)) = (host.adapter_set(), device.adapter_set()) {
            for (key, (ha, hb)) in &h.pairs {
                let (da, db) = &d.pairs[key];
                assert!(cosine(ha, da) >= cos_floor && cosine(hb, db) >= cos_floor, "adapter `{key}` differs");
            }
        }
        assert_eq!(host.steps(), device.steps());
    }
}

trait CloneP {
    fn clone_prepared(&self) -> Prepared;
}

impl CloneP for Prepared {
    fn clone_prepared(&self) -> Prepared {
        Prepared { sample: self.sample.clone(), target: self.target.clone() }
    }
}

fn exact_resume(mode: Mode, label: &str) {
    let Some(dev) = device(label) else { return };
    let (mut straight, p) = dev_trainer(dev, mode.clone());
    let batch = std::slice::from_ref(&p);
    for _ in 0..4 {
        straight.step(batch).unwrap();
    }
    let (mut first, _) = dev_trainer(device(label).unwrap(), mode.clone());
    for _ in 0..2 {
        first.step(batch).unwrap();
    }
    let dir = scratch(label);
    first.save(&dir).unwrap();
    let mut resumed = DevTrainer::resume(device(label).unwrap(), &dir, cfg_for(mode)).unwrap();
    assert_eq!(resumed.steps(), 2);
    for _ in 0..2 {
        resumed.step(batch).unwrap();
    }
    assert_eq!(resumed.checkpoint().tensors, straight.checkpoint().tensors, "weights after resume differ from the uninterrupted run");
    assert_eq!(resumed.adapter().map(|a| a.pairs), straight.adapter().map(|a| a.pairs), "adapter after resume differs");
    assert_eq!(resumed.eval_loss(batch), straight.eval_loss(batch));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_resumed_device_full_run_ends_exactly_where_the_uninterrupted_run_does() {
    let _g = gpu_core::testgpu::exclusive();
    exact_resume(Mode::Full, "dev-resume-full");
}

#[test]
fn a_resumed_device_lora_run_ends_exactly_where_the_uninterrupted_run_does() {
    let _g = gpu_core::testgpu::exclusive();
    exact_resume(Mode::Lora { rank: 2, alpha: 4.0, seed: 5 }, "dev-resume-lora");
}

#[test]
fn a_device_run_directory_resumes_on_the_host_trainer() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("cross resume") else { return };
    let rel = if dev.gpu().kind() == "cpu" { 1e-4 } else { 2e-3 };
    let (mut t, p) = dev_trainer(dev, Mode::Heads(vec![Head::Reader]));
    let batch = std::slice::from_ref(&p);
    t.step(batch).unwrap();
    let dir = scratch("dev-to-host");
    t.save(&dir).unwrap();
    let mut host = Trainer::resume(&dir, cfg_for(Mode::Heads(vec![Head::Reader]))).unwrap();
    assert_eq!(host.steps(), 1);
    let (d, h) = (t.step(batch).unwrap(), host.step(batch).unwrap());
    assert!((d.loss - h.loss).abs() <= rel * h.loss.abs().max(1.0), "second step: device {} vs host {}", d.loss, h.loss);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_device_lora_adapter_is_a_no_op_at_init_learns_and_merges_exactly() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("device lora") else { return };
    let rel = if dev.gpu().kind() == "cpu" { 1e-5 } else { 2e-3 };
    let (mut t, p) = dev_trainer(dev, Mode::Lora { rank: 2, alpha: 4.0, seed: 5 });
    t.set_lr(2e-2);
    let batch = std::slice::from_ref(&p);
    let ck = tiny_checkpoint();
    let host_base = Trainer::new(ck.clone(), cfg_for(Mode::Full)).unwrap();
    let before = t.eval_loss(batch);
    assert!((before - host_base.eval_loss(batch)).abs() <= rel * before.abs().max(1.0), "B starts at zero, so a fresh adapter changes nothing");
    for _ in 0..20 {
        t.step(batch).unwrap();
    }
    let after = t.eval_loss(batch);
    eprintln!("device lora on one tiny sample: loss {before:.4} -> {after:.4}");
    assert!(after < 0.97 * before, "the adapter did not learn: {before} -> {after}");
    assert_eq!(t.checkpoint().tensors, ck.tensors, "the base stays untouched");
    let merged = Trainer::new(t.merged().unwrap(), cfg_for(Mode::Full)).unwrap();
    let (live, folded) = (t.eval_loss(batch), merged.eval_loss(batch));
    assert!((live - folded).abs() <= 10.0 * rel * live.abs().max(1.0), "live {live} vs merged {folded}");
}

#[test]
fn a_device_full_fine_tune_overfits_and_its_checkpoint_loads_through_the_importer() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("device full") else { return };
    let (mut t, p) = dev_trainer(dev, Mode::Full);
    let batch = std::slice::from_ref(&p);
    let before = t.eval_loss(batch);
    for _ in 0..30 {
        t.step(batch).unwrap();
    }
    let after = t.eval_loss(batch);
    eprintln!("device full fine-tune on one tiny sample: loss {before:.4} -> {after:.4}");
    assert!(after < 0.7 * before, "held loss fell only from {before} to {after}");
    let dir = scratch("dev-export");
    t.save(&dir).unwrap();
    let loaded = mirage::import::load(&dir.join("mirage.safetensors"), &mirage::import::ImportOptions::default()).unwrap();
    assert_eq!(loaded.tensors, t.checkpoint().tensors);
    assert_eq!(loaded.config, mirage::config::MirageConfig::tiny());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_recompute_step_equals_a_keep_step_on_both_trainers() {
    use model::saved::{SavedFormat, SavedPolicy};
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("recompute step") else { return };
    let on_cpu = dev.gpu().kind() == "cpu";
    let ck = tiny_checkpoint();
    let p = prepared(&ck.config);
    let batch = [p.clone_prepared(), p.clone_prepared()];
    for mode in [Mode::Full, Mode::Lora { rank: 2, alpha: 4.0, seed: 5 }] {
        let (mut keep, mut recompute) = (Trainer::new(ck.clone(), cfg_for(mode.clone())).unwrap(), Trainer::new(ck.clone(), cfg_for(mode.clone())).unwrap());
        recompute.set_saved_policy(SavedPolicy::Recompute).unwrap();
        let mut device_recompute = DevTrainer::new(device("recompute step").unwrap(), ck.clone(), cfg_for(mode.clone())).unwrap();
        let mut device_keep = DevTrainer::new(device("recompute step").unwrap(), ck.clone(), cfg_for(mode.clone())).unwrap();
        device_recompute.set_saved_policy(SavedPolicy::Recompute).unwrap();
        for _ in 0..2 {
            let (a, b) = (keep.step(&batch).unwrap(), recompute.step(&batch).unwrap());
            assert_eq!(a.loss.to_bits(), b.loss.to_bits(), "{mode:?}: host loss");
            assert_eq!(a.grad_norm.to_bits(), b.grad_norm.to_bits(), "{mode:?}: host gradient norm");
            let (a, b) = (device_keep.step(&batch).unwrap(), device_recompute.step(&batch).unwrap());
            assert!(on_cpu && a.loss.to_bits() == b.loss.to_bits() && a.grad_norm.to_bits() == b.grad_norm.to_bits() || !on_cpu && (a.loss - b.loss).abs() <= 2e-3 * a.loss.abs().max(1.0), "{mode:?}: device loss {} vs {}", a.loss, b.loss);
        }
        let same = |x: &mirage::import::Checkpoint, y: &mirage::import::Checkpoint| x.tensors.iter().all(|(n, t)| t.data.iter().zip(&y.tensors[n].data).all(|(a, b)| a.to_bits() == b.to_bits()));
        assert!(same(&keep.checkpoint(), &recompute.checkpoint()), "{mode:?}: host weights differ after recompute steps");
        if on_cpu {
            assert!(same(&device_keep.checkpoint(), &device_recompute.checkpoint()), "{mode:?}: device weights differ after recompute steps");
        }
    }
    let refused = Trainer::new(ck.clone(), cfg_for(Mode::Full)).unwrap().set_saved_policy(SavedPolicy::Quantized(SavedFormat::Bf16));
    assert!(refused.is_err_and(|e| e.contains("full precision")), "a narrow boundary format must be refused by name");
}
