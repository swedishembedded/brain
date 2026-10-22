// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements serving of quantized checkpoints in the
// precision they were released in for its clients. If your team needs
// expertise in running GGUF K-quant models on the GPUs it already owns, you
// can procure our services by sending an email to info@swedishembedded.com.

//! The spec for `gguf::upload`: a GGUF tensor lands at the tier that holds
//! what it stores - Q4_K/Q5_K as K-quants, everything else at int8 - and the
//! product with it is the product with the stored values.

use checkpoint::gguf::{GgmlType, MmapGguf};
use checkpoint::gguf_write::{write, TensorOut};
use checkpoint::packed::{BoundedWorkspace, MemoryBudget};
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::Ops;

const N: usize = 16;
const K: usize = 512;
const M: usize = 4;

fn values(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.731 + seed).sin() * 1.5 + (i % 7) as f32 * 0.01).collect()
}

#[test]
fn each_format_lands_at_the_tier_that_holds_it_and_multiplies_as_stored() {
    let _device = gpu_core::testgpu::exclusive();
    let formats = [(GgmlType::Q4K, Dtype::Q4K), (GgmlType::Q5K, Dtype::Q8K), (GgmlType::Q6K, Dtype::I8), (GgmlType::Q8_0, Dtype::I8)];
    let path = std::env::temp_dir().join(format!("brain-gguf-upload-{}.gguf", std::process::id())).to_string_lossy().into_owned();
    let tensors: Vec<TensorOut> = formats
        .iter()
        .enumerate()
        .map(|(i, (ty, _))| TensorOut { name: format!("w{i}"), shape: vec![K, N], ty: ty.id(), data: checkpoint::quant::quantize_par(ty.id(), &values(N * K, i as f32)).unwrap() })
        .collect();
    write(&path, &[], &tensors, 32).unwrap();
    let mg = MmapGguf::open(&path).unwrap();

    let ops = Ops::new(Gpu::new(model::ops::kernel_list())).expect("Ops::new");
    let g = ops.gpu();
    let staging = BoundedWorkspace::new(MemoryBudget::new(64 << 20, 0));
    let x_h = values(M * K, 9.0);
    let x = g.storage_init("x", &x_h);
    for (i, &(ty, tier)) in formats.iter().enumerate() {
        let name = format!("w{i}");
        assert_eq!(gguf::held_tier_of(&mg, &name), tier, "{} is held at {tier:?}", ty.name());
        let w = gguf::upload_weight(&ops, &mg, &name, N, K, tier, &staging).unwrap_or_else(|| panic!("{} uploads at {tier:?}", ty.name()));
        assert_eq!(w.dtype(), tier);
        let stored = mg.tensor(&name).unwrap().unwrap();
        let want: Vec<f32> = (0..M * N).map(|o| (0..K).map(|c| x_h[(o / N) * K + c] * stored[(o % N) * K + c]).sum()).collect();
        let mut s = Vec::new();
        let act = ops.act(&mut s, &x, 0, M as u32, K as u32);
        let y = g.storage((M * N) as u64);
        ops.matmul(&mut s, &w, &act, &y, 0);
        g.submit(&[], &s);
        let got = g.read(&y, M * N);
        let err = got.iter().zip(&want).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt() / want.iter().map(|b| b * b).sum::<f32>().sqrt();
        assert!(err < 1e-2, "{}: relative error {err} against the stored values (only the activation is quantized)", ty.name());
    }
    assert!(gguf::upload_weight(&ops, &mg, "w3", N, K, Dtype::Q4K, &staging).is_none(), "a K-quant is never made from another format");
    std::fs::remove_file(&path).ok();
}
