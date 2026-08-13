// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Throwaway diagnostic, not a correctness gate: which of two candidate costs
//! explains `ParamStore::new_with_roles_src`'s ~35 s "read+write" bracket when
//! streaming the real DeepSeek-OCR decoder's cached fp32 expansion (~11.7 GB,
//! 2234 tensors) -- the dominant share of `deepseekocr::caps::Session::load`'s
//! 20-28 s model-construction cost, isolated by real per-stage timers this
//! same investigation added to
//! `crates/paramstore`/`crates/deepseekv2`/`crates/wgsl-cpu`.
//!
//! `dd if=<file> of=/dev/null` on the exact same file read the whole 11.7 GB
//! in ~8 s (1.5 GB/s), so a pure sequential scan is not the bottleneck. Two
//! candidates remained: (a) the upload loop visits tensors in
//! `DeepseekV2Config::param_list()`'s construction order (layer 0..11, each
//! layer's attention then MoE experts 0..63), which need not match the
//! tensors' PHYSICAL byte order in the safetensors file -- if it does not,
//! every tensor fetch is effectively a random seek across an 11.7 GB mmap
//! instead of a sequential scan; (b) first-touch page faults on the
//! DESTINATION buffers (`gpu.storage`'s fresh `vec![0u32; n]` per tensor) under
//! this box's memory pressure (swap observed pinned near 100% full for the
//! whole investigation), independent of the source read pattern.
//!
//! This reads tensors via [`checkpoint::weightio::WeightReader::raw_words`]
//! (zero-copy, no destination buffer, no GPU) in two orders and prints wall
//! time for each -- run as two SEPARATE process invocations (not two
//! `#[test]`s in one run) so neither benefits from the other's warmed page
//! cache:
//!
//! ```text
//! cargo test --release -p brain-deepseekv2 --test weight_read_order_bench \
//!   -- --ignored --nocapture construction_order
//! cargo test --release -p brain-deepseekv2 --test weight_read_order_bench \
//!   -- --ignored --nocapture file_order
//! ```

use checkpoint::weightio::WeightReader;
use checkpoint::TensorSource;
use deepseek2::DeepseekV2Config;

#[path = "common/real_lm.rs"]
mod real_lm;

/// Touch every word of every named tensor (construction or file order),
/// without allocating a destination buffer -- isolates the SOURCE read cost
/// (mmap page-in through `raw_words`) from `ParamStore`'s own per-tensor
/// `gpu.storage`/`write_at`.
fn scan(reader: &WeightReader, names: &[String], label: &str) {
    let t0 = std::time::Instant::now();
    let mut checksum: u64 = 0;
    let mut total_words: u64 = 0;
    for name in names {
        let words = reader.raw_words(name).unwrap_or_else(|| panic!("missing tensor {name}"));
        // Touch one word per 4 KiB page (1024 u32 words), not every word --
        // this measures page-fault/read cost, not memcpy bandwidth, which is
        // what `ParamStore`'s own `t_read_write` bracket already conflates
        // with the fetch. A stride keeps this diagnostic itself cheap.
        let mut i = 0;
        while i < words.len() {
            checksum ^= words[i] as u64;
            i += 1024;
        }
        total_words += words.len() as u64;
    }
    let elapsed = t0.elapsed();
    println!(
        "{label}: {} tensors, {:.2} GiB, {:.1} ms ({:.0} MB/s), checksum={checksum:x}",
        names.len(),
        total_words as f64 * 4.0 / (1 << 30) as f64,
        elapsed.as_secs_f64() * 1e3,
        total_words as f64 * 4.0 / elapsed.as_secs_f64() / 1e6,
    );
}

#[test]
#[ignore = "real-weight diagnostic, not a correctness gate; run in its own process, see module doc"]
fn construction_order() {
    let Some((gguf, st)) = real_lm::paths() else {
        eprintln!("skip: real DeepSeek-OCR checkpoint not in the model store");
        return;
    };
    let st_path = real_lm::expanded(&gguf, &st);
    let reader = WeightReader::open(&st_path).unwrap_or_else(|e| panic!("open expansion: {e}"));
    let cfg = deepseek2::import::config_from_file(gguf.to_str().expect("utf-8 path"), 512).unwrap_or_else(|e| panic!("config_from_file: {e}"));
    assert_eq!(cfg, DeepseekV2Config::deepseek_ocr(512));
    let names: Vec<String> = cfg.param_list().into_iter().map(|(n, _)| n).collect();
    scan(&reader, &names, "construction order (ParamStore's real upload order)");
}

#[test]
#[ignore = "real-weight diagnostic, not a correctness gate; run in its own process, see module doc"]
fn file_order() {
    let Some((gguf, st)) = real_lm::paths() else {
        eprintln!("skip: real DeepSeek-OCR checkpoint not in the model store");
        return;
    };
    let st_path = real_lm::expanded(&gguf, &st);
    let reader = WeightReader::open(&st_path).unwrap_or_else(|e| panic!("open expansion: {e}"));
    let names: Vec<String> = reader.names().map(str::to_string).collect();
    scan(&reader, &names, "file order (physical safetensors layout)");
}
