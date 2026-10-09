// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The whole detector's loss and EVERY parameter gradient on the selected
//! accelerator agree with the CPU backend, from identical weights and data.
//!
//! Swedish Embedded AB implements training stacks for vision models on GPUs.
//! If your team needs expertise in proving a fast backward pass computes the
//! same gradients as the reference, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! `p3_gradcheck` proves the backward against finite differences on the CPU
//! backend. This test carries that proof to a device whose kernels differ - on
//! CUDA the dense convolutions, their input and weight gradients run as native
//! kernels that accumulate in a different order - by holding the device's
//! analytic gradients to the CPU's. Proxy loss (`<r, logits>`, linear in the
//! logits) so no discrete step (an anchor assignment, a top-k) can turn a
//! last-bit difference into a different loss.
//!
//! Tolerance: every kernel of both backends is fp32 with fp32 accumulation, so
//! two correct implementations differ only by summation order, a relative
//! `O(L u)` per reduction of length `L` that compounds through the depth of the
//! net. The bar is a fraction of each tensor's own largest gradient, set more
//! than an order of magnitude above what correct kernels produce on this model
//! - on CUDA both the generated WGSL tier and the native conv kernels measured
//! under 4e-5 - and two below what a dropped or misplaced term does (one
//! missing input column in the native forward moved the proxy loss by 38%).

use model::Model;
use yolov8::net::PIPELINES;
use yolov8::{LossMode, Yolo, YoloConfig};

/// Largest allowed `max|g_dev - g_cpu| / max|g_cpu|` per tensor.
const REL_TOL: f32 = 1e-3;

fn randvec(seed: u64, n: usize) -> Vec<f32> {
    let mut r = data::rng::Lcg::new(seed);
    r.vec(n)
}

fn run(model: &Yolo, img: &[f32]) -> (f32, Vec<(String, Vec<f32>)>) {
    model.set_mode(LossMode::Proxy);
    model.set_batch(model::Batch::Tensor { tokens: None, inputs: img, targets: &[] });
    model.zero_grads();
    let loss = model.forward();
    model.backward();
    model.poll_wait();
    let grads = model.param_names().into_iter().map(|n| (n.clone(), model.read_grad(&n))).collect();
    (loss, grads)
}

#[test]
fn every_gradient_on_the_device_matches_the_cpu_backend() {
    let dev = gpu_core::testgpu::dev(PIPELINES);
    if dev.kind() == "cpu" {
        brain_testutil::skip_unavailable("no accelerator selected (set BRAIN_BACKEND/BRAIN_DEVICE)");
        return;
    }
    let cfg = YoloConfig::tiny(2);
    let b = 4u32;
    let init = yolov8::init_weights(&cfg, 7);
    let img = randvec(0xA5A5, (b * 3 * cfg.input * cfg.input) as usize);

    let on_dev = Yolo::new_on(dev, cfg.clone(), b, cfg.input, &init);
    let on_cpu = Yolo::new_on(gpu_core::Gpu::new_cpu(PIPELINES), cfg.clone(), b, cfg.input, &init);
    let (loss_d, grads_d) = run(&on_dev, &img);
    let (loss_c, grads_c) = run(&on_cpu, &img);

    let loss_rel = (loss_d - loss_c).abs() / loss_c.abs().max(1e-6);
    eprintln!("proxy loss: device {loss_d} cpu {loss_c} (rel {loss_rel:.2e})");
    assert!(loss_rel <= REL_TOL, "proxy loss differs: device {loss_d} vs cpu {loss_c}");

    let mut worst = (0.0f32, String::new());
    for ((name, gd), (_, gc)) in grads_d.iter().zip(&grads_c) {
        assert_eq!(gd.len(), gc.len(), "{name}: length");
        let scale = gc.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let diff = gd.iter().zip(gc).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(gd.iter().all(|v| v.is_finite()), "{name}: non-finite gradient on the device");
        let rel = if scale > 0.0 { diff / scale } else { diff };
        if rel > worst.0 {
            worst = (rel, name.clone());
        }
        assert!(rel <= REL_TOL, "{name}: max|g_dev - g_cpu| = {diff:e} is {rel:.2e} of max|g_cpu| = {scale:e}");
    }
    eprintln!("{} tensors; worst relative gradient difference {:.2e} ({})", grads_c.len(), worst.0, worst.1);
}
