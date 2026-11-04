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
use checkpoint::remap::{Fetch, RemapSource};
use checkpoint::TensorSource;
use gpu_core::Gpu;
use gpu_core::select::Dtype;
use model::ops::{training_kernel_list, upload_fused_from_source, upload_from_source, upload_group_preserved, FusedGroup, Ops, Weight};

const N: usize = 256;
const K: usize = 512;

const N2: usize = 128;

/// An FP8 safetensors file with a bf16 128x128 scale grid per tensor: `w` is `[N, K]`, `v` is `[N2, K]`;
/// `u` is a plain bf16 `[N2, K]` tensor (a part stored differently from the other two).
fn write_fp8_checkpoint(dir: &std::path::Path) {
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let mut tensor = |rows: usize, grid0: f32| {
        let w: Vec<u8> = (0..rows * K)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let b = (s >> 33) as u8;
                if b & 0x7f == 0x7f { b & 0xfe } else { b }
            })
            .collect();
        let grid: Vec<u8> = (0..(rows / 128) * (K / 128)).flat_map(|i| half::bf16::from_f32(grid0 + i as f32 * 0.25).to_le_bytes()).collect();
        (w, grid)
    };
    let (w, wg) = tensor(N, 0.5);
    let (v, vg) = tensor(N2, 1.5);
    let u: Vec<u8> = (0..N2 * K).flat_map(|i| half::bf16::from_f32(((i * 29 % 197) as f32 - 98.0) / 50.0).to_le_bytes()).collect();
    let mut header = serde_json::Map::new();
    let mut off = 0usize;
    for (name, shape, bytes) in [("w", vec![N, K], &w), ("w_scale_inv", vec![N / 128, K / 128], &wg), ("v", vec![N2, K], &v), ("v_scale_inv", vec![N2 / 128, K / 128], &vg), ("u", vec![N2, K], &u)] {
        let dtype = if name.ends_with("scale_inv") || name == "u" { "BF16" } else { "F8_E4M3" };
        header.insert(name.into(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [off, off + bytes.len()]}));
        off += bytes.len();
    }
    let h = serde_json::to_vec(&header).unwrap();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    for b in [&w, &wg, &v, &vg, &u] {
        out.extend_from_slice(b);
    }
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
    let src = PackedSource::new(Box::new(reader), CodecRegistry::reference(), [("blocks.0.w".to_string(), Binding::new("w")), ("blocks.0.v".to_string(), Binding::new("v")), ("blocks.0.u".to_string(), Binding::new("u"))]).unwrap();
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

    // A fused projection joins the parts packed: same weight as uploading the concatenated f32 rows.
    let (mut v, mut fused_ref) = (Vec::new(), whole.clone());
    assert!(src.with_tensor("blocks.0.v", &mut |d| v = d.to_vec()));
    fused_ref.extend_from_slice(&v);
    for want in [Dtype::BF16, Dtype::NF4] {
        let staging = BoundedWorkspace::new(MemoryBudget::new(300 << 10, 0));
        let got = upload_fused_from_source(&ops, &src, &["blocks.0.w", "blocks.0.v"], N + N2, K, want, &staging).expect("both parts exist");
        assert!(staging.peak() > 0, "{want:?}: the fused load must take the packed path");
        let reference = Weight::upload(&ops, &fused_ref, N + N2, K, want);
        let (a, b) = (decoded(&ops, &got), decoded(&ops, &reference));
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "{want:?}: fused tiled upload differs from the concatenated whole upload");
    }

    // A renamed view of the source (a RemapSource over it) keeps the packed path.
    let plan = [("up".to_string(), Fetch::Whole("blocks.0.w".to_string()))].into_iter().collect();
    let remap = RemapSource::new(&src, plan);
    let staging = BoundedWorkspace::new(MemoryBudget::new(300 << 10, 0));
    upload_from_source(&ops, &remap, "up", N, K, Dtype::BF16, &staging).expect("renamed tensor");
    assert!(staging.peak() > 0, "a Whole remap must forward the packed view");
    std::fs::remove_dir_all(&dir).ok();
}

/// Spec: a fused group of parts stored in different encodings (FP8 next to bf16) is converted part by part into
/// one weight of the requested tier, equal to uploading the concatenated decoded rows, with bounded staging and no
/// whole-tensor f32 copy; asked to keep the stored encodings, it is split into one weight per part instead.
#[test]
fn a_fused_group_of_mixed_encodings_is_converted_per_part_or_split() {
    let dir = std::env::temp_dir().join(format!("upload_mixed_group_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    write_fp8_checkpoint(&dir);
    let reader = PackedReader::open(&dir.join("m.safetensors")).unwrap();
    let src = PackedSource::new(
        Box::new(reader),
        CodecRegistry::reference(),
        [("w", "w"), ("v", "v"), ("u", "u")].map(|(canonical, stored)| (canonical.to_string(), Binding::new(stored))),
    )
    .unwrap();
    let ops = Ops::new(Gpu::new(training_kernel_list())).expect("Ops::new");
    let rows_of = |name: &str| {
        let mut out = Vec::new();
        assert!(src.with_tensor(name, &mut |d| out = d.to_vec()));
        out
    };
    let mixed_ref = [rows_of("w"), rows_of("u")].concat();
    for want in [Dtype::NF4, Dtype::BF16, Dtype::F8E4M3] {
        // An FP8 tile is a whole 128-row scale band, so its smallest legal tile is larger than the others'.
        let budget: u64 = if want == Dtype::F8E4M3 { 2 << 20 } else { 300 << 10 };
        let staging = BoundedWorkspace::new(MemoryBudget::new(budget, 0));
        let got = upload_fused_from_source(&ops, &src, &["w", "u"], N + N2, K, want, &staging).unwrap_or_else(|e| panic!("{want:?}: {e}"));
        assert_eq!((got.n() as usize, got.dtype()), (N + N2, want), "{want:?}: the fused weight is held as asked");
        assert!(staging.peak() > 0 && staging.peak() <= budget, "{want:?}: staging peak {} B", staging.peak());
        // What converting each part on its own gives, joined: the fused weight is exactly that.
        let separate: Vec<f32> = [("w", N), ("u", N2)].into_iter().flat_map(|(name, n)| decoded(&ops, &upload_from_source(&ops, &src, name, n, K, want, &staging).expect("part"))).collect();
        let fused = decoded(&ops, &got);
        assert!(fused.iter().zip(&separate).all(|(x, y)| x.to_bits() == y.to_bits()), "{want:?}: the fused conversion differs from converting the parts separately");
        // And it is a faithful conversion of the stored values: the tier's own rounding error only (relative L2).
        let norm = |v: &mut dyn Iterator<Item = f32>| v.map(|x| (x as f64).powi(2)).sum::<f64>().sqrt();
        let rel = norm(&mut fused.iter().zip(&mixed_ref).map(|(x, y)| x - y)) / norm(&mut mixed_ref.iter().copied());
        let bound = match want {
            Dtype::BF16 => 5e-3,
            Dtype::F8E4M3 => 4e-2,
            _ => 0.2,
        };
        assert!(rel < bound, "{want:?}: relative L2 deviation from the stored values {rel}");
    }

    let staging = BoundedWorkspace::new(MemoryBudget::new(300 << 10, 0));
    match upload_group_preserved(&ops, &src, &["w", "u"], &[N, N2], K, &staging).expect("a mixed group splits") {
        FusedGroup::Split(parts) => assert_eq!(parts.iter().map(Weight::dtype).collect::<Vec<_>>(), [Dtype::F8E4M3, Dtype::BF16]),
        FusedGroup::Joined(_) => panic!("FP8 and bf16 parts cannot be one weight in their stored encodings"),
    }
    match upload_group_preserved(&ops, &src, &["w", "v"], &[N, N2], K, &staging).expect("a uniform group joins") {
        FusedGroup::Joined(w) => assert_eq!((w.n() as usize, w.dtype()), (N + N2, Dtype::F8E4M3)),
        FusedGroup::Split(_) => panic!("two FP8 parts join in place"),
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// A brain checkpoint with a bf16 linear, a packed-int8 linear (`u32` words plus the `.scale` grid) and an f32 norm.
fn write_mixed_brain_checkpoint(path: &std::path::Path) {
    let bf16: Vec<u8> = (0..4 * 32).flat_map(|i| half::bf16::from_f32(i as f32 * 0.01).to_le_bytes()).collect();
    let words: Vec<u8> = (0..4 * 8 * 4).map(|i| i as u8).collect();
    let scales: Vec<u8> = [0.5f32, 1.0, 1.5, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    let norm: Vec<u8> = (0..32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let mut header = serde_json::Map::new();
    let mut off = 0usize;
    for (name, dtype, shape, bytes) in [
        ("blocks.0.mlp.gate.weight", "BF16", vec![4, 32], &bf16),
        ("blocks.0.mlp.up.weight", "U32", vec![4, 8], &words),
        ("blocks.0.mlp.up.weight.scale", "F32", vec![4, 1], &scales),
        ("blocks.0.ln1.weight", "F32", vec![32], &norm),
    ] {
        header.insert(name.into(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [off, off + bytes.len()]}));
        off += bytes.len();
    }
    header.insert("__metadata__".into(), serde_json::json!({ "brain.config": "{}" }));
    let h = serde_json::to_vec(&header).unwrap();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    for b in [&bf16, &words, &scales, &norm] {
        out.extend_from_slice(b);
    }
    std::fs::write(path, out).unwrap();
}

/// Spec: the tier policy of a checkpoint keeps each linear in the encoding the file stores it in - bf16 stays bf16,
/// brain's packed int8 stays int8 - and everything else (an f32 tensor, a tensor the file lacks) at f32, so a
/// serving build never widens what is stored narrow.
#[test]
fn the_stored_tier_policy_keeps_each_tensor_in_its_stored_encoding() {
    let dir = std::env::temp_dir().join(format!("stored_tier_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.safetensors");
    write_mixed_brain_checkpoint(&path);
    let src = checkpoint::packed::stored_source(checkpoint::weightio::WeightReader::open(path.to_str().unwrap()).unwrap());

    assert_eq!(model::ops::stored_tier(&*src, "blocks.0.mlp.gate.weight"), Some(Dtype::BF16));
    assert_eq!(model::ops::stored_tier(&*src, "blocks.0.mlp.up.weight"), Some(Dtype::I8));
    assert_eq!(model::ops::stored_tier(&*src, "blocks.0.ln1.weight"), Some(Dtype::F32));
    assert_eq!(model::ops::stored_tier(&*src, "blocks.0.absent.weight"), None);

    let names = ["blocks.0.mlp.gate.weight", "blocks.0.mlp.up.weight", "blocks.0.ln1.weight", "blocks.0.absent.weight"];
    let policy = model::ops::stored_tier_policy(&*src, names);
    let tiers: Vec<Dtype> = names.iter().map(|n| policy.want(n)).collect();
    assert_eq!(tiers, [Dtype::BF16, Dtype::I8, Dtype::F32, Dtype::F32]);
    std::fs::remove_dir_all(&dir).ok();
}

/// Spec: a linear a checkpoint stores as packed int8 comes back from `quantize_from` as the stored codes and
/// scales, not a requantization of their decode (which moves any group whose largest code is under 127).
#[test]
fn a_stored_int8_linear_is_quantized_from_as_its_own_codes() {
    let dir = std::env::temp_dir().join(format!("stored_int8_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.safetensors");
    write_mixed_brain_checkpoint(&path);
    let src = checkpoint::packed::stored_source(checkpoint::weightio::WeightReader::open(path.to_str().unwrap()).unwrap());

    let (words, scales) = model::int8::quantize_from(&*src, "blocks.0.mlp.up.weight", 4, 32).expect("the tensor is stored");
    let want_words: Vec<u32> = (0..4 * 8).map(|w| u32::from_le_bytes([4 * w as u8, 4 * w as u8 + 1, 4 * w as u8 + 2, 4 * w as u8 + 3])).collect();
    assert_eq!(words, want_words);
    assert_eq!(scales, [0.5f32, 1.0, 1.5, 2.0]);
    std::fs::remove_dir_all(&dir).ok();
}
