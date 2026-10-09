// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which conv lowering a VAE graph takes on a device with a native conv
//! kernel, and that taking it does not move the answer beyond fp32 rounding.
//!
//! Swedish Embedded AB implements fast, verified image autoencoders for its
//! clients. If your team needs expertise in convolution lowerings for
//! diffusion VAEs, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! A conv the device's native implicit-GEMM kernel serves (`conv_bias_reg`,
//! redirected by `gpu_core::native_upgrade`) must be dispatched DIRECT - not
//! lowered to `im2col_at` + a GEMM, which materialises a nine-times-wider
//! operand the native kernel never needs. A conv it does not serve (the
//! encoder's padded stride-2 downsample) keeps the lowering. Both halves are
//! asserted from the recorded graph itself.
//!
//! The answer is held to the CPU reference graph (direct WGSL convs on the CPU
//! JIT, fp32 throughout) built from the same seeded weights: the native conv
//! accumulates the same reduction with fused multiply-adds, so it agrees to
//! fp32 rounding, not bit for bit - the bound below is that rounding with
//! headroom, two orders of magnitude under a wrong tap or a dropped channel.
//!
//! GPU-gated: `BRAIN_VAE_DEVICE` names the card (default `gpu`), and the test
//! skips where that device offers no native conv.

mod zeros;

use vae::{VaeConfig, VaeDecoder, VaeEncoder};

fn device() -> String {
    std::env::var("BRAIN_VAE_DEVICE").unwrap_or_else(|_| "gpu".to_string())
}

/// Seeded weights at every shape `zeros` lists: conv weights uniform in
/// `±1/sqrt(fan_in)`, biases and GroupNorm shifts small, GroupNorm gains near
/// one - the scale a trained VAE keeps its activations at.
fn seeded(mut t: vae::blocks::Tensors) -> vae::blocks::Tensors {
    let mut names: Vec<String> = t.keys().cloned().collect();
    names.sort();
    let mut r = data::rng::Lcg::new(0x7a3e_11d0);
    for name in names {
        let (shape, data) = t.get_mut(&name).expect("listed");
        let fan_in: usize = shape.iter().skip(1).product::<usize>().max(1);
        let gain = name.contains("norm") && name.ends_with(".weight");
        for v in data.iter_mut() {
            let u = (r.next_u32() % 20_001) as f32 / 10_000.0 - 1.0;
            *v = if gain {
                1.0 + 0.1 * u
            } else if shape.len() == 1 {
                0.05 * u
            } else {
                u / (fan_in as f32).sqrt()
            };
        }
    }
    t
}

/// Slot of `name` on `gpu`'s kernel list.
fn slot(gpu: &gpu_core::Gpu, name: &str) -> usize {
    gpu.kernel_index(name).unwrap_or_else(|| panic!("the VAE kernel set registers {name}"))
}

/// Recorded dispatches of kernel `name`.
fn count(gpu: &gpu_core::Gpu, steps: &[gpu_core::Step], name: &str) -> usize {
    let k = slot(gpu, name);
    steps.iter().filter(|s| s.meta().is_some_and(|m| m.kernel == k)).count()
}

/// Whether `gpu` redirects a dense 3x3 conv to a native kernel at all.
fn offers_native_conv(gpu: &gpu_core::Gpu) -> bool {
    gpu.native_kernel_for(slot(gpu, "conv_bias_reg"), &[1, 128, 32, 32, 128, 3, 1, 1, 32, 32]).is_some()
}

fn compare(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: output length");
    let peak = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let worst = got.iter().zip(want).fold(0.0f32, |m, (g, w)| m.max((g - w).abs()));
    eprintln!("{what}: max |native - cpu reference| = {worst:.3e} against a peak of {peak:.3e}");
    assert!(peak > 0.1, "{what}: a degenerate output ({peak}) would agree with anything");
    assert!(worst <= 2e-4 * peak, "{what}: max |delta| {worst:.3e} exceeds the fp32 rounding bound {:.3e}", 2e-4 * peak);
}

#[test]
fn a_decoder_takes_the_native_conv_and_matches_the_reference() {
    let cfg = VaeConfig::flux2();
    let ts = seeded(zeros::decoder(&cfg));
    let (lh, lw) = (8u32, 6u32);
    let dec = VaeDecoder::from_diffusers(cfg.clone(), &ts, lh, lw, Some(&device()));
    if !offers_native_conv(dec.gpu()) {
        brain_testutil::skip_unavailable("this device offers no native conv kernel");
        return;
    }
    // Every FLUX.2 decoder conv is a stride-1 square conv the native kernel
    // serves, so the lowering must not appear at all.
    assert_eq!(count(dec.gpu(), dec.steps(), "im2col_at"), 0, "a served conv was lowered to im2col + GEMM");
    let mut r = data::rng::Lcg::new(17);
    let z: Vec<f32> = (0..cfg.latent_channels * lh * lw).map(|_| (r.next_u32() % 2001) as f32 / 1000.0 - 1.0).collect();
    let got = dec.decode(&z);
    let want = VaeDecoder::from_diffusers(cfg, &ts, lh, lw, Some("cpu")).decode(&z);
    compare("decode", &got, &want);
}

#[test]
fn an_encoder_lowers_only_the_convs_the_native_kernel_does_not_serve() {
    let cfg = VaeConfig::flux2();
    let ts = seeded(zeros::encoder(&cfg));
    // Large enough that every downsample's output plane is past the lowering's
    // own position threshold, so all of them would be lowered on any GPU.
    let (h, w) = (128u32, 96u32);
    let enc = VaeEncoder::from_diffusers(cfg.clone(), &ts, h, w, Some(&device()));
    if !offers_native_conv(enc.gpu()) {
        brain_testutil::skip_unavailable("this device offers no native conv kernel");
        return;
    }
    // The three padded stride-2 downsamples are the convs the native kernel
    // declines (their output extent is forced, not derived), so exactly
    // those three stay lowered.
    let downsamples = cfg.block_out_channels.len() - 1;
    assert_eq!(count(enc.gpu(), enc.steps(), "im2col_at"), downsamples, "only the downsamples are lowered");
    let mut r = data::rng::Lcg::new(23);
    let img: Vec<f32> = (0..cfg.in_channels * h * w).map(|_| (r.next_u32() % 2001) as f32 / 1000.0 - 1.0).collect();
    let got = enc.encode(&img);
    let want = VaeEncoder::from_diffusers(cfg, &ts, h, w, Some("cpu")).encode(&img);
    compare("encode", &got, &want);
}
