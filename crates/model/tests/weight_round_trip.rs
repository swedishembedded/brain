// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for simulating quantized model
// weights exactly as the deployed kernels see them for its clients. If your
// team needs expertise in quantization-aware model optimization then you can
// procure our services by sending an email to info@swedishembedded.com.

//! `Weight::round_trip` is the weight the device computes with: for every
//! tier `Weight::upload` accepts, the host round trip equals what the device
//! decodes from the uploaded weight. A model optimizer that ranks tiers by
//! simulated quantization error, or trains through fake quantization, is only
//! measuring the deployed model if this holds.
//!
//! The device's decoded weight is read through the ordinary matmul: one-hot
//! activation rows (`x = I_k`) give `out[r, j] = W[j, r]`. A one-hot row is
//! exact under every activation quantization `Ops::act` applies (its absmax
//! is the single 1), so the only rounding left is the weight's own decode and
//! one scale multiply.

use data::rng::Lcg;
use gpu_core::select::Dtype;
use model::ops::{Ops, Weight};

const TIERS: &[Dtype] = &[Dtype::F32, Dtype::BF16, Dtype::F16, Dtype::I8, Dtype::Q4, Dtype::NF4, Dtype::F8E4M3, Dtype::F8E5M2];

#[test]
fn round_trip_is_what_the_device_decodes() {
    let (n, k) = (64usize, 128usize);
    let w = Lcg::new(7).vec_scaled(n * k, 0.5);
    let mut identity = vec![0f32; k * k];
    for r in 0..k {
        identity[r * k + r] = 1.0;
    }
    let ops = Ops::new(gpu_core::testgpu::dev(model::ops::training_kernel_list())).expect("Ops::new");
    let g = ops.gpu();
    for &tier in TIERS {
        let weight = Weight::upload(&ops, &w, n, k, tier);
        if weight.dtype() != tier {
            brain_testutil::skip_unavailable(&format!("{tier:?}: this device holds it as {:?}, so its decode is not exercised here", weight.dtype()));
            continue;
        }
        let x = g.storage_init("x", &identity);
        let mut steps = Vec::new();
        let act = ops.act(&mut steps, &x, 0, k as u32, k as u32);
        let out = g.storage((k * n) as u64);
        ops.matmul(&mut steps, &weight, &act, &out, 0);
        g.submit(&[], &steps);
        let device = g.read(&out, k * n);

        let host = Weight::round_trip(&w, n, k, tier);
        assert_eq!(host.len(), n * k, "{tier:?}: round trip keeps the shape");
        for j in 0..n {
            for c in 0..k {
                let (h, d) = (host[j * k + c], device[c * n + j]);
                assert!((h - d).abs() <= 1e-6 * h.abs().max(1e-3), "{tier:?} w[{j},{c}]: host round trip {h} != device decode {d}");
            }
        }
        if tier != Dtype::F32 {
            assert!(host.iter().zip(&w).any(|(a, b)| a != b), "{tier:?}: a narrower tier must actually round");
        }
    }
}

#[test]
#[should_panic(expected = "round_trip")]
fn round_trip_refuses_a_tier_upload_does_not_take() {
    Weight::round_trip(&[0.0; 64], 2, 32, Dtype::Q4K);
}

#[test]
fn packs_names_the_shapes_each_codec_takes() {
    for &tier in TIERS {
        assert!(Weight::packs(tier, 64, 128), "{tier:?} packs a whole-group shape");
    }
    for tier in [Dtype::I8, Dtype::Q4, Dtype::NF4] {
        assert!(!Weight::packs(tier, 64, 48), "{tier:?} needs whole groups along k");
    }
    assert!(Weight::packs(Dtype::F8E4M3, 64, 48) && !Weight::packs(Dtype::F8E4M3, 64, 50));
    assert!(!Weight::packs(Dtype::Q4K, 64, 128) && !Weight::packs(Dtype::BF16, 0, 128));
}
