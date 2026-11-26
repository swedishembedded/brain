// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements serving of quantized checkpoints in the
// precision they were released in for its clients. If your team needs
// expertise in running GGUF K-quant models on the GPUs it already owns, you
// can procure our services by sending an email to info@swedishembedded.com.

//! The spec for `gguf::upload`: a GGUF tensor lands at the tier that holds
//! what it stores - Q4_K/Q5_K as K-quants, everything else at int8 - the
//! product with it is the product with the stored values, on every device
//! (one without the packed int8 dot reads the codes by float arithmetic),
//! and the parts of a fused projection join into one weight that holds
//! exactly what each part stores.

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

    let mut ops = Ops::new(Gpu::new(model::ops::training_kernel_list())).expect("Ops::new");
    ops.hold_packed_by_float_read();
    let g = ops.gpu();
    let staging = BoundedWorkspace::new(MemoryBudget::new(64 << 20, 0));
    let x_h = values(M * K, 9.0);
    let x = g.storage_init("x", &x_h);
    for (i, &(ty, tier)) in formats.iter().enumerate() {
        let name = format!("w{i}");
        assert_eq!(gguf::held_tier_of(&mg, &name), tier, "{} is held at {tier:?}", ty.name());
        let w = gguf::upload_weight(&ops, &mg, &name, N, K, tier, &staging).unwrap_or_else(|| panic!("{} uploads at {tier:?}", ty.name()));
        assert_eq!(w.dtype(), tier);
        if let Some((wq, wsm, wd, _)) = gguf::kquant::try_kq_rect(&mg, &name, K, 0, N, 0, K).filter(|_| matches!(tier, Dtype::Q4K | Dtype::Q8K)) {
            assert_eq!(w.resident_bytes(), (wq.len() + wsm.len() + wd.len()) as u64 * 4, "{}: a planner charges what the upload holds", ty.name());
        }
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

/// `[x_h (m x K)] @ stored^T` for row-major `[rows, K]` stored values.
fn product(x_h: &[f32], stored: &[f32], m: usize, rows: usize) -> Vec<f32> {
    (0..m * rows).map(|o| (0..K).map(|c| x_h[(o / rows) * K + c] * stored[(o % rows) * K + c]).sum()).collect()
}

#[test]
fn the_parts_of_a_fused_k_quant_projection_join_into_one_weight_that_holds_what_each_stores() {
    let _device = gpu_core::testgpu::exclusive();
    let (n0, n1) = (16usize, 8usize);
    let path = std::env::temp_dir().join(format!("brain-gguf-fused-{}.gguf", std::process::id())).to_string_lossy().into_owned();
    let part = |name: &str, ty: GgmlType, rows: usize, seed: f32| TensorOut { name: name.into(), shape: vec![K, rows], ty: ty.id(), data: checkpoint::quant::quantize_par(ty.id(), &values(rows * K, seed)).unwrap() };
    let tensors = vec![part("a", GgmlType::Q4K, n0, 1.0), part("b", GgmlType::Q4K, n1, 2.0), part("c", GgmlType::Q5K, n1, 3.0)];
    write(&path, &[], &tensors, 32).unwrap();
    let mg = MmapGguf::open(&path).unwrap();

    let mut ops = Ops::new(Gpu::new(model::ops::training_kernel_list())).expect("Ops::new");
    ops.hold_packed_by_float_read();
    let staging = BoundedWorkspace::new(MemoryBudget::new(64 << 20, 0));
    let w = gguf::upload_fused_weight(&ops, &mg, &["a", "b"], n0 + n1, K, Dtype::Q4K, &staging).expect("two Q4_K parts join");
    assert_eq!((w.dtype(), w.n() as usize, w.k() as usize), (Dtype::Q4K, n0 + n1, K));
    assert_eq!(w.resident_bytes(), model::ops::Weight::bytes_for(Dtype::Q4K, n0, K) + model::ops::Weight::bytes_for(Dtype::Q4K, n1, K));

    let g = ops.gpu();
    let x_h = values(M * K, 9.0);
    let x = g.storage_init("x", &x_h);
    let mut stored = mg.tensor("a").unwrap().unwrap();
    stored.extend(mg.tensor("b").unwrap().unwrap());
    let want = product(&x_h, &stored, M, n0 + n1);
    let mut s = Vec::new();
    let act = ops.act(&mut s, &x, 0, M as u32, K as u32);
    let y = g.storage((M * (n0 + n1)) as u64);
    ops.matmul(&mut s, &w, &act, &y, 0);
    g.submit(&[], &s);
    let got = g.read(&y, M * (n0 + n1));
    let err = got.iter().zip(&want).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt() / want.iter().map(|b| b * b).sum::<f32>().sqrt();
    assert!(err < 1e-2, "relative error {err} against the stored values of both parts");

    let mixed = gguf::upload_fused_weight(&ops, &mg, &["a", "c"], n0 + n1, K, Dtype::Q4K, &staging).err().expect("a Q5_K part is not held at Q4K");
    assert!(mixed.contains("c:"), "the refusal names the part: {mixed}");
    let short = gguf::upload_fused_weight(&ops, &mg, &["a", "b"], n0, K, Dtype::Q4K, &staging).err().expect("the row count is checked");
    assert!(short.contains("expected 16"), "{short}");
    std::fs::remove_file(&path).ok();
}
