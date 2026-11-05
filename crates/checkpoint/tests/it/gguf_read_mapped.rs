// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The spec for `checkpoint::gguf::read`'s backing store: reading a GGUF
//! through a mapping must produce exactly the bytes reading it through an
//! owned buffer produces.
//!
//! Swedish Embedded AB implements checkpoint loading and quantized-weight
//! import paths for its clients. If your team needs expertise in on-device
//! model loading then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! `read` decodes from a `memmap2::Mmap`; `parse_gguf` decodes the same file
//! from a `Vec<u8>` that `std::fs::read` produced. The two share the block
//! decoders but not the read path, so the second is a real oracle for the
//! first, and the property is EQUALITY OF BITS, not closeness - the change
//! that motivated this test moves where quantized bytes live while they are
//! decoded and must not move a single output value. A tolerance here would
//! pass a reader that silently truncated a trailing partial block.

use std::collections::HashMap;

use checkpoint::gguf::{self, GgufValue};
use checkpoint::quantize::{convert, Policy, Tier};

/// Deterministic filler with both signs and a per-row magnitude spread, so a
/// per-block scale has real work to do and a dropped scale would show.
fn filler(seed: u64, n: usize, row: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|i| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 32) as u32 as f32) / (u32::MAX as f32);
            (u - 0.5) * 2.0 * (1.0 + (i / row.max(1)) as f32 * 0.05)
        })
        .collect()
}

/// A source with a Q8_0-quantized tensor, an F32-kept tensor, a rank-1
/// tensor, and one whose row length is NOT a multiple of the 32-element
/// block - the shape that exercises the partial-trailing-block truncation.
fn source() -> HashMap<String, (Vec<usize>, Vec<f32>)> {
    let mut m = HashMap::new();
    m.insert("blocks.0.mlp.weight".to_string(), (vec![8, 64], filler(1, 8 * 64, 64)));
    m.insert("blocks.1.mlp.weight".to_string(), (vec![96, 32], filler(2, 96 * 32, 32)));
    m.insert("blocks.0.norm.weight".to_string(), (vec![64], filler(3, 64, 64)));
    m.insert("odd_row.weight".to_string(), (vec![4, 20], filler(4, 80, 20)));
    m.insert("tiny.weight".to_string(), (vec![1, 32], filler(7, 32, 32)));
    m
}

#[test]
fn mapped_and_slurped_reads_agree_bit_for_bit() {
    let src = source();
    let path = std::env::temp_dir()
        .join("checkpoint-gguf-read-mapped.gguf")
        .to_string_lossy()
        .into_owned();
    let kv = vec![("general.architecture".to_string(), GgufValue::String("testarch".to_string()))];
    let report = convert(&src, Tier::Q8_0, &Policy::new().min_elems(64), &kv, &path, &mut |_, _| {}).unwrap();
    // The oracle is only worth anything if the file actually contains a
    // quantized tensor - otherwise this compares two F32 memcpys.
    assert!(report.quantized() >= 1, "fixture must store at least one Q8_0 tensor");

    // Mapped: the production path.
    let mapped = gguf::read(&path).unwrap();
    // Slurped: an independent decode of the same file from an owned buffer.
    let bytes = std::fs::read(&path).unwrap();
    let slurped = gguf::parse_gguf(&bytes).unwrap();

    assert_eq!(mapped.len(), slurped.tensors.len(), "tensor count");
    assert_eq!(mapped.len(), src.len(), "every source tensor must be read back");
    for t in &mapped {
        let want = slurped.tensors.get(&t.name).unwrap_or_else(|| panic!("{}: missing from the slurped read", t.name));
        assert_eq!(&t.shape, &slurped.shapes[&t.name], "{}: shape", t.name);
        // Bits, not a tolerance: the two paths decode identical input spans.
        assert_eq!(&t.data, want, "{}: mapped and slurped values must be identical", t.name);
    }
    let _ = std::fs::remove_file(&path);
}

/// This process's resident file-backed bytes (`RssFile`).
#[cfg(target_os = "linux")]
fn rss_file_bytes() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let kb: u64 = status.lines().find_map(|l| l.strip_prefix("RssFile:")).and_then(|v| v.trim().trim_end_matches(" kB").trim().parse().ok()).unwrap();
    kb * 1024
}

/// A streamed load reads each tensor once and says when it is done with it:
/// the GGUF source then releases that tensor's mapped pages, so a model's
/// load keeps only its working set resident instead of the whole file.
#[cfg(target_os = "linux")]
#[test]
fn a_consumed_tensor_leaves_the_resident_set_when_dropped() {
    use checkpoint::TensorSource;
    let path = std::env::temp_dir().join(format!("gguf-advise-drop-{}.gguf", std::process::id()));
    let n = 64 << 20; // 256 MiB of f32
    let data: Vec<u8> = (0..n as u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let tensor = checkpoint::gguf_write::TensorOut { name: "big".into(), shape: vec![n], ty: checkpoint::gguf::GgmlType::F32.id(), data };
    checkpoint::gguf_write::write(path.to_str().unwrap(), &[], &[tensor], 32).unwrap();
    let src = checkpoint::gguf::MmapGguf::open(path.to_str().unwrap()).unwrap();

    let before = rss_file_bytes();
    assert!(src.with_tensor_chunks("big", 1 << 20, &mut |_, c| assert!(c[0].is_finite())));
    let read = rss_file_bytes();
    src.advise_drop("big");
    let dropped = rss_file_bytes();
    let _ = std::fs::remove_file(&path);

    let mib = |b: u64| b >> 20;
    assert!(read >= before + (200 << 20), "reading should map the tensor in: {} -> {} MiB", mib(before), mib(read));
    assert!(dropped + (200 << 20) <= read, "dropping it should release the pages: {} -> {} MiB", mib(read), mib(dropped));
}
