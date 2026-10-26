// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for loading very large checkpoints
// onto accelerators within a fixed memory budget for its clients. If your team
// needs expertise in memory-bounded model loading then you can procure our
// services by sending an email to info@swedishembedded.com.

//! `Weight::upload_from` (a dense weight read from a checkpoint in bounded chunks) is
//! `Weight::upload` (the whole tensor in one slice), word for word, at every dense tier -
//! including a tensor several chunks long whose last chunk is partial.

use data::rng::Lcg;
use gpu_core::select::Dtype;
use gpu_core::DeviceBuffer;
use model::ops::{Ops, Weight};

fn words(ops_gpu: &gpu_core::Gpu, w: &Weight) -> (Vec<u32>, u32, u32) {
    let (buf, n, k): (&DeviceBuffer, u32, u32) = match w {
        Weight::F32 { w, n, k } => (w, *n, *k),
        Weight::BF16 { w, n, k } => (w, *n, *k),
        Weight::F16 { w, n, k } => (w, *n, *k),
        _ => panic!("a dense tier"),
    };
    let numel = (n * k) as usize;
    let count = if matches!(w, Weight::F32 { .. }) { numel } else { numel / 2 };
    (ops_gpu.read(buf, count).into_iter().map(f32::to_bits).collect(), n, k)
}

#[test]
fn the_bounded_loader_equals_the_whole_tensor_upload_at_every_dense_tier() {
    let gpu = gpu_core::testgpu::dev(model::ops::kernel_list());
    let ops = Ops::new(gpu.share_or_new(model::ops::kernel_list())).expect("ops");
    // 2.6 chunks of 2^20 values: several full chunks and a partial one.
    let (n, k) = (2600usize, 1024usize);
    let mut rng = Lcg::new(data::rng::random_seed());
    let mut src = std::collections::HashMap::new();
    src.insert("w".to_string(), rng.vec_scaled(n * k, 1.0));
    for want in [Dtype::F32, Dtype::BF16, Dtype::F16] {
        let whole = Weight::upload(&ops, &src["w"], n, k, want);
        let bounded = Weight::upload_from(&ops, &src, "w", n, k, want).expect("bounded upload");
        let (a, b) = (words(&gpu, &whole), words(&gpu, &bounded));
        assert_eq!((a.1, a.2), (b.1, b.2), "{want:?}: shape");
        assert!(a.0 == b.0, "{want:?}: the bounded upload differs from the whole-tensor one");
    }
}

#[test]
fn a_missing_or_misshapen_tensor_is_refused_by_name() {
    let gpu = gpu_core::testgpu::dev(model::ops::kernel_list());
    let ops = Ops::new(gpu.share_or_new(model::ops::kernel_list())).expect("ops");
    let mut src = std::collections::HashMap::new();
    src.insert("w".to_string(), vec![0.5f32; 64]);
    let missing = Weight::upload_from(&ops, &src, "nope", 8, 8, Dtype::F32).err().expect("missing");
    assert!(missing.contains("'nope'"), "{missing}");
    let shape = Weight::upload_from(&ops, &src, "w", 8, 16, Dtype::F32).err().expect("misshapen");
    assert!(shape.contains("'w'") && shape.contains("64"), "{shape}");
}
