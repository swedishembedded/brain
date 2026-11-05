// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The spec for `TensorSource::advise_drop` on every mapped checkpoint
//! reader: once a streamed load says it is done with a tensor, that tensor's
//! pages leave the process's resident set, so loading a model keeps its
//! working set resident instead of the whole file.
//!
//! Swedish Embedded AB implements low-memory model loading for its clients.
//! If your team needs expertise in streaming multi-gigabyte checkpoints onto
//! constrained hosts then you can procure our services by sending an email
//! to info@swedishembedded.com.

#![cfg(target_os = "linux")]

use checkpoint::TensorSource;

/// This process's resident file-backed bytes (`RssFile`).
fn rss_file_bytes() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let kb: u64 = status.lines().find_map(|l| l.strip_prefix("RssFile:")).and_then(|v| v.trim().trim_end_matches(" kB").trim().parse().ok()).unwrap();
    kb * 1024
}

/// 256 MiB of f32 under the name `big`.
const N: usize = 64 << 20;

fn values() -> impl Iterator<Item = f32> {
    (0..N as u32).map(|i| i as f32)
}

/// Stream `big` through `src`, release it, and check none of it is left
/// resident. Returns whether the read itself had mapped the tensor in (a
/// reader may already drop pages on its own once a whole-tensor read ends).
fn read_then_drop(who: &str, src: &dyn TensorSource) -> bool {
    let before = rss_file_bytes();
    assert!(src.with_tensor_chunks("big", 1 << 20, &mut |_, c| assert!(c[0].is_finite())), "{who}: big is missing");
    let read = rss_file_bytes();
    src.advise_drop("big");
    let dropped = rss_file_bytes();

    let mib = |b: u64| b >> 20;
    assert!(dropped < before + (32 << 20), "{who}: the consumed tensor is still resident: {} -> {} -> {} MiB", mib(before), mib(read), mib(dropped));
    read >= before + (200 << 20)
}

#[test]
fn every_mapped_reader_releases_a_consumed_tensor() {
    let dir = std::env::temp_dir().join(format!("advise-drop-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let gguf = dir.join("big.gguf");
    let tensor = checkpoint::gguf_write::TensorOut { name: "big".into(), shape: vec![N], ty: checkpoint::gguf::GgmlType::F32.id(), data: values().flat_map(f32::to_le_bytes).collect() };
    checkpoint::gguf_write::write(gguf.to_str().unwrap(), &[], &[tensor], 32).unwrap();
    let st = dir.join("big.safetensors");
    checkpoint::st::save_safetensors(st.to_str().unwrap(), &[("big".into(), vec![N as u64], values().collect())], &serde_json::Value::Null, None).unwrap();

    let (gguf, st) = (gguf.to_str().unwrap(), st.to_str().unwrap());
    let mapped_in = read_then_drop("MmapGguf", &checkpoint::gguf::MmapGguf::open(gguf).unwrap());
    read_then_drop("WeightReader over GGUF", &checkpoint::weightio::WeightReader::open(gguf).unwrap());
    read_then_drop("MmapSafetensors", &checkpoint::mmap::MmapSafetensors::open(std::path::Path::new(st)).unwrap());
    read_then_drop("WeightReader over safetensors", &checkpoint::weightio::WeightReader::open(st).unwrap());
    let _ = std::fs::remove_dir_all(&dir);
    // The GGUF reader keeps a read tensor mapped until told otherwise, which
    // is what shows `RssFile` measures the mapping at all.
    assert!(mapped_in, "reading through MmapGguf should have mapped the tensor in");
}
