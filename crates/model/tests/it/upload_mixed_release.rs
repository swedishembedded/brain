// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded loading of quantized
// checkpoints straight into device memory for its clients. If your team needs
// expertise in serving mixed-precision FP8 and NVFP4 releases on the GPU you
// already own, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: a mixed-precision release (FP8 with one static `weight_scale` per tensor next to NVFP4 with block and
//! global scales) reaches the device as the tiers it is stored in, bit for bit. A per-tensor f32 scale that bf16
//! would round is held exactly (on a tile grid, alone or joined into a fused projection of tensors with different
//! scales), and NVFP4 is the microscaling tier.

use checkpoint::packed::{BoundedWorkspace, CodecRegistry, MemoryBudget, PackedReader, PackedSource};
use checkpoint::safetensors::e4m3fn_to_f32;
use checkpoint::TensorSource;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::{stored_tier, training_kernel_list, upload_fused_from_source, upload_from_source, Ops, Weight};

const ROWS: usize = 128;
const K: usize = 256;
/// Not a bf16 value (the bf16 of it differs), so a tile grid that rounded it would be caught.
const SCALE_A: f32 = 0.012_345_679;
const SCALE_B: f32 = 0.034_567_89;

fn fp8_bytes(seed: u32, n: usize) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let b = (s >> 24) as u8;
            if b & 0x7f == 0x7f { b & 0xfe } else { b }
        })
        .collect()
}

fn write_release(path: &std::path::Path) -> (Vec<u8>, Vec<u8>) {
    let (a, b) = (fp8_bytes(1, ROWS * K), fp8_bytes(2, ROWS * K));
    // NVFP4 [ROWS, K]: two codes per byte, an E4M3 scale per 16, a global f32 scale.
    let codes: Vec<u8> = (0..ROWS * K / 2).map(|i| (i * 37 % 251) as u8).collect();
    let block_scales: Vec<u8> = (0..ROWS * K / 16).map(|i| 0x30 + (i % 9) as u8).collect();
    let bf16: Vec<u8> = (0..ROWS * K).flat_map(|i| half::bf16::from_f32((i % 13) as f32 / 7.0).to_le_bytes()).collect();
    let tensors: Vec<(&str, &str, Vec<usize>, Vec<u8>)> = vec![
        ("a.weight", "F8_E4M3", vec![ROWS, K], a.clone()),
        ("a.weight_scale", "F32", vec![], SCALE_A.to_le_bytes().to_vec()),
        ("b.weight", "F8_E4M3", vec![ROWS, K], b.clone()),
        ("b.weight_scale", "F32", vec![], SCALE_B.to_le_bytes().to_vec()),
        ("m.weight", "U8", vec![ROWS, K / 2], codes),
        ("m.weight_scale", "F8_E4M3", vec![ROWS, K / 16], block_scales),
        ("m.weight_scale_2", "F32", vec![], 0.003f32.to_le_bytes().to_vec()),
        ("h.weight", "BF16", vec![ROWS, K], bf16),
    ];
    let mut header = serde_json::Map::new();
    let mut off = 0;
    for (name, dtype, shape, bytes) in &tensors {
        header.insert((*name).into(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [off, off + bytes.len()]}));
        off += bytes.len();
    }
    let h = serde_json::to_vec(&header).unwrap();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    for (.., bytes) in &tensors {
        out.extend(bytes);
    }
    std::fs::write(path, out).unwrap();
    (a, b)
}

fn decoded(ops: &Ops, w: &Weight) -> Vec<f32> {
    ops.read_host(w)
}

#[test]
fn a_mixed_fp8_nvfp4_release_uploads_in_its_stored_tiers_exactly() {
    let dir = std::env::temp_dir().join(format!("upload_mixed_release_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (a, b) = write_release(&dir.join("m.safetensors"));
    let src = PackedSource::identity_lenient(Box::new(PackedReader::open(&dir.join("m.safetensors")).unwrap()), CodecRegistry::reference());
    let ops = Ops::new(Gpu::new(training_kernel_list())).expect("Ops::new");
    let staging = BoundedWorkspace::new(MemoryBudget::new(8 << 20, 0));

    // Each tensor is held as the tier its encoding is.
    assert_eq!(stored_tier(&src, "a.weight"), Some(Dtype::F8E4M3));
    assert_eq!(stored_tier(&src, "m.weight"), Some(Dtype::F4E2M1));
    assert_eq!(stored_tier(&src, "h.weight"), Some(Dtype::BF16));

    // Per-tensor FP8: the f32 scale on every tile, so the product is the scalar f32 product with no rounding of the scale.
    let want = |bytes: &[u8], scale: f32| bytes.iter().map(|&x| e4m3fn_to_f32(x) * scale).collect::<Vec<f32>>();
    let got = upload_from_source(&ops, &src, "a.weight", ROWS, K, Dtype::F8E4M3, &staging).expect("a exists");
    assert!(matches!(got, Weight::FP8 { .. }));
    assert_eq!(decoded(&ops, &got).iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want(&a, SCALE_A).iter().map(|v| v.to_bits()).collect::<Vec<_>>());

    // Two tensors with different scales join into one FP8 weight, each keeping its own scale.
    let fused = upload_fused_from_source(&ops, &src, &["a.weight", "b.weight"], 2 * ROWS, K, Dtype::F8E4M3, &staging).expect("fusable");
    let mut both = want(&a, SCALE_A);
    both.extend(want(&b, SCALE_B));
    assert_eq!(decoded(&ops, &fused).iter().map(|v| v.to_bits()).collect::<Vec<_>>(), both.iter().map(|v| v.to_bits()).collect::<Vec<_>>());

    // NVFP4 is the microscaling tier and decodes to what the reference codec decodes the same bytes to.
    let micro = upload_from_source(&ops, &src, "m.weight", ROWS, K, Dtype::F4E2M1, &staging).expect("m exists");
    assert!(matches!(micro, Weight::Micro { .. }));
    let mut reference = Vec::new();
    assert!(src.with_tensor("m.weight", &mut |d| reference = d.to_vec()));
    assert_eq!(decoded(&ops, &micro).iter().map(|v| v.to_bits()).collect::<Vec<_>>(), reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>());

    // A tensor not stored as NVFP4 is never turned into the NVFP4 tier (nor the W4A8 int4 layout) behind the caller's back.
    assert!(upload_from_source(&ops, &src, "h.weight", ROWS, K, Dtype::F4E2M1, &staging).is_none());
    std::fs::remove_dir_all(&dir).ok();
}
