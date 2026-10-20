// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded loading of quantized
// checkpoints straight into device memory for its clients. If your team needs
// expertise in uploading packed weights tile by tile without ever expanding a
// tensor to full precision, you can procure our services by sending an email
// to info@swedishembedded.com.

//! Spec of `upload_from_source`: a named tensor of a packed source becomes the resident tier asked for,
//! identical to uploading its f32 decode, while the host holds one tile of staging instead of the tensor.

use checkpoint::packed::{BoundedWorkspace, Binding, CodecRegistry, MemoryBudget, PackedReader, PackedSource, Pool};
use checkpoint::TensorSource;
use gpu_core::Gpu;
use gpu_core::select::Dtype;
use model::ops::{training_kernel_list, upload_from_source, Ops, Weight};

const N: usize = 256;
const K: usize = 512;

/// A one-tensor FP8 safetensors file with a bf16 128x128 scale grid.
fn write_fp8_checkpoint(dir: &std::path::Path) {
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let w: Vec<u8> = (0..N * K)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let b = (s >> 33) as u8;
            if b & 0x7f == 0x7f { b & 0xfe } else { b }
        })
        .collect();
    let grid: Vec<u8> = (0..(N / 128) * (K / 128)).flat_map(|i| half::bf16::from_f32(0.5 + i as f32 * 0.25).to_le_bytes()).collect();
    let header = serde_json::json!({
        "w": {"dtype": "F8_E4M3", "shape": [N, K], "data_offsets": [0, w.len()]},
        "w_scale_inv": {"dtype": "BF16", "shape": [N / 128, K / 128], "data_offsets": [w.len(), w.len() + grid.len()]},
    });
    let h = serde_json::to_vec(&header).unwrap();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(w);
    out.extend(grid);
    std::fs::write(dir.join("m.safetensors"), out).unwrap();
}

fn decoded(ops: &Ops, w: &Weight) -> Vec<f32> {
    let g = ops.gpu();
    let out = g.storage((w.n() * w.k()) as u64);
    let mut steps = Vec::new();
    ops.decode_tile(&mut steps, w, 0, w.n(), &out);
    g.submit(&[], &steps);
    g.read(&out, (w.n() * w.k()) as usize)
}

#[test]
fn a_packed_tensor_uploads_as_the_requested_tier_with_one_tile_of_staging() {
    let dir = std::env::temp_dir().join(format!("upload_from_source_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    write_fp8_checkpoint(&dir);
    let reader = PackedReader::open(&dir.join("m.safetensors")).unwrap();
    let src = PackedSource::new(Box::new(reader), CodecRegistry::reference(), [("blocks.0.w".to_string(), Binding::new("w"))]).unwrap();
    let ops = Ops::new(Gpu::new(training_kernel_list())).expect("Ops::new");

    let mut whole = Vec::new();
    assert!(src.with_tensor("blocks.0.w", &mut |d| whole = d.to_vec()));
    // A small budget: far below the tensor's f32 size (512 KiB), above one 128-row tile of this shape.
    for want in [Dtype::BF16, Dtype::NF4, Dtype::I8, Dtype::F8E4M3] {
        let staging = BoundedWorkspace::new(MemoryBudget::new(300 << 10, 0));
        let got = upload_from_source(&ops, &src, "blocks.0.w", N, K, want, &staging).expect("the tensor exists");
        let reference = Weight::upload(&ops, &whole, N, K, want);
        assert_eq!(got.n(), reference.n());
        let (a, b) = (decoded(&ops, &got), decoded(&ops, &reference));
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "{want:?}: tiled upload differs from the whole-tensor upload");
        assert!(staging.peak() <= 300 << 10, "{want:?}: staging peak {} B", staging.peak());
        assert_eq!(staging.budget().stats(Pool::HostStaging).in_use, 0);
    }
    assert!(upload_from_source(&ops, &src, "missing", N, K, Dtype::BF16, &BoundedWorkspace::new(MemoryBudget::new(1 << 20, 0))).is_none());
    std::fs::remove_dir_all(&dir).ok();
}
