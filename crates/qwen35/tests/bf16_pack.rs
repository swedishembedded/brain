// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded LoRA/QLoRA fine-tuning of
// large language models on a single GPU for its clients. If your team needs
// expertise in training large models within a fixed memory budget then you
// can procure our services by sending an email to info@swedishembedded.com.

//! `pack_bf16` / `unpack_bf16`: the device rounds an f32 buffer to bf16 exactly
//! as `model::half::pack_bf16` does on the host, for odd and even lengths and
//! for the awkward values (ties, subnormals, infinities, NaN), and decoding
//! gives back the rounded values.

use data::rng::Lcg;
use gpu_core::Gpu;
use qwen35::model::pipelines;

fn kernel(gpu: &Gpu, name: &str) -> usize {
    gpu.kernel_index(name).unwrap_or_else(|| panic!("{name} is not registered"))
}

fn round_trip(gpu: &Gpu, x: &[f32]) -> (Vec<u32>, Vec<f32>) {
    let n = x.len() as u32;
    let xb = gpu.storage_init("x", x);
    let words = x.len().div_ceil(2) as u64;
    let packed = gpu.storage(words);
    gpu.submit(&[], &[gpu.step(kernel(gpu, "pack_bf16"), &[&xb, &packed], &[n], n.div_ceil(2))]);
    let out = gpu.storage(x.len() as u64);
    gpu.submit(&[], &[gpu.step(kernel(gpu, "unpack_bf16"), &[&packed, &out], &[n], n)]);
    (gpu.read(&packed, words as usize).into_iter().map(f32::to_bits).collect(), gpu.read(&out, x.len()))
}

fn values() -> Vec<f32> {
    let mut rng = Lcg::new(data::rng::random_seed());
    let mut v: Vec<f32> = (0..4097).map(|_| rng.scaled(30.0)).collect();
    v.extend([
        0.0,
        -0.0,
        f32::from_bits(0x3f80_8000), // a tie rounding to even (down)
        f32::from_bits(0x3f81_8000), // a tie rounding to even (up)
        f32::from_bits(0x0000_0001), // smallest subnormal
        f32::from_bits(0x007f_ffff), // largest subnormal
        f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ]);
    v
}

fn check(gpu: &Gpu) {
    for take in [0usize, 1, 2, 3, 64, 65, values().len() - 1, values().len()] {
        let x = &values()[..take.max(1)];
        let (packed, back) = round_trip(gpu, x);
        assert_eq!(packed, model::half::pack_bf16(x), "{} elements: packed words differ from the host", x.len());
        for (i, (&got, &want)) in back.iter().zip(x).enumerate() {
            let rounded = f32::from_bits((model::half::f32_to_bf16(want) as u32) << 16);
            assert!(got.to_bits() == rounded.to_bits() || (got.is_nan() && rounded.is_nan()), "element {i}: {got} vs {rounded}");
        }
    }
}

#[test]
fn the_cpu_backend_packs_and_unpacks_bf16_like_the_host() {
    check(&Gpu::new_cpu(pipelines()));
}

#[test]
fn the_default_device_packs_and_unpacks_bf16_like_the_host() {
    check(&Gpu::new(pipelines()));
}
