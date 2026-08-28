// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Isolated profile of the two candidate stages of `qwen35::import::
//! import_layer` (the per-layer weight loader the sliding-window streaming
//! forward path, `crates/qwen35/src/stream.rs`, calls) named as suspects for
//! the streaming forward pass's 75-minute full-run wall-clock, measured one
//! at a time at REAL `Qwen35Config::qwen38_27b()` layer scale against the
//! REAL checkpoint on disk:
//!
//!   1. `MmapSafetensors::tensor_f32` - the raw-FP8-byte-to-f32 decode
//!      `import_layer` actually calls (one byte per element, unscaled).
//!   2. `model::fp8::dequant_block128` - the per-128x128-block scale
//!      multiply, given already-decoded raw f32 + scale arrays.
//!   3. `model::int8::quantize_weight` - the per-channel absmax + int8 pack
//!      `ops::Weight::upload` runs on the dequantized weight.
//!
//! Raw disk throughput (the 4th measurement the task asks for) is NOT
//! reimplemented here - `dd if=<shard> of=/dev/null bs=4M iflag=direct`
//! against the same real shard is the simpler, more trustworthy way to
//! measure it (bypasses the page cache outright, which a same-process `File`
//! read cannot do without root to drop caches), and this session already
//! re-measured it that way; see the profiling report for the number.
//!
//! This binary exists because the 75-minute full-run number is a SINGLE
//! end-to-end measurement, not a stage breakdown - guessing which of these
//! stages is the real cost, rather than measuring it, is exactly the mistake
//! this tool rules out. Swedish Embedded AB builds this kind of
//! profile-before-you-optimize tooling for its clients' own
//! performance-sensitive ML/edge pipelines; if your team needs the same
//! discipline applied to a model import or inference path, reach out at
//! info@swedishembedded.com.
//!
//! Usage: `import_profile [shard_path] [tensor_base_name] [reps]`
//!   shard_path       default: `$BRAIN_QWEN35_DIR/layers-5.safetensors` (set
//!                    `BRAIN_QWEN35_DIR` to a downloaded Qwen/Qwen3.8-27B-FP8
//!                    dir, same env var `qwen35/tests/streaming_forward.rs`
//!                    uses, or pass a shard path explicitly)
//!   tensor_base_name default: model.language_model.layers.5.mlp.down_proj
//!                    (a real [5120, 17408] FP8 blockwise-quantized weight -
//!                    one of the three MLP tensors the task names as
//!                    "~267M params" combined)
//!   reps             default: 3 (stage 1 - the mmap decode - is timed once,
//!                    since a repeat read of the SAME still-mapped tensor
//!                    mostly re-measures the page cache rather than the real
//!                    decode cost; stages 2/3 are pure-CPU and repeated for a
//!                    stable mean)

use std::time::Instant;

fn default_shard_path() -> String {
    let dir = std::env::var("BRAIN_QWEN35_DIR").unwrap_or_else(|_| {
        panic!("import_profile: pass a shard path as arg 1, or set BRAIN_QWEN35_DIR to a downloaded Qwen/Qwen3.8-27B-FP8 dir")
    });
    format!("{dir}/layers-5.safetensors")
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let shard_path = a.get(1).cloned().unwrap_or_else(default_shard_path);
    let base = a.get(2).cloned().unwrap_or_else(|| "model.language_model.layers.5.mlp.down_proj".to_string());
    let reps: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);

    let weight_name = format!("{base}.weight");
    let scale_name = format!("{base}.weight_scale_inv");

    println!("import_profile: shard={shard_path} tensor={base} reps={reps}");

    let reader = checkpoint::mmap::MmapSafetensors::open(&shard_path).unwrap_or_else(|e| panic!("open {shard_path}: {e}"));
    let shape = reader.shape(&weight_name).unwrap_or_else(|| panic!("missing shape for {weight_name}")).to_vec();
    assert_eq!(shape.len(), 2, "{weight_name}: expected a 2-D weight, got {shape:?}");
    let (rows, cols) = (shape[0], shape[1]);
    let numel = rows * cols;
    let dtype = reader.dtype(&weight_name).unwrap_or_default().to_string();
    assert_eq!(dtype, "F8_E4M3", "{weight_name}: expected F8_E4M3, got {dtype}");
    let scale_shape = reader.shape(&scale_name).unwrap_or_else(|| panic!("missing shape for {scale_name}")).to_vec();
    println!("tensor shape: [{rows}, {cols}] = {numel} elements (F8_E4M3, {numel} bytes raw); scale {scale_name} shape {scale_shape:?}");

    // --- Stage 1: mmap decode (raw FP8 byte -> f32, unscaled) ------------
    // Timed ONCE per fresh MmapSafetensors handle - `tensor_f32` calls
    // `advise_dontneed_tensor` on the SAME handle after decoding, so a
    // second call on the same `reader` would legitimately re-fault from
    // disk/cache and is a different (also interesting, but distinct)
    // measurement from "first touch of this tensor in this process".
    let t0 = Instant::now();
    let raw = reader.tensor_f32(&weight_name).unwrap_or_else(|| panic!("missing data for {weight_name}"));
    let decode_s = t0.elapsed().as_secs_f64();
    assert_eq!(raw.len(), numel);
    let decode_meps = numel as f64 / decode_s / 1e6;
    let decode_gbs = (numel as f64 * 4.0 / 1e9) / decode_s; // output bytes/s (f32)
    println!("\n[1] MmapSafetensors::tensor_f32 (raw FP8 decode): {:>9.2} ms  ({decode_meps:>8.1} Melem/s, {decode_gbs:>6.2} GB/s of f32 output)", decode_s * 1e3);

    let scale = reader.tensor_f32(&scale_name).unwrap_or_else(|| panic!("missing data for {scale_name}"));

    // --- Stage 2: dequant_block128, isolated, repeated for a stable mean -
    let mut dequant_total = 0f64;
    let mut dequant_out = Vec::new();
    for _ in 0..reps {
        let t1 = Instant::now();
        dequant_out = model::fp8::dequant_block128(&raw, &scale, rows, cols, 128);
        dequant_total += t1.elapsed().as_secs_f64();
    }
    let dequant_s = dequant_total / reps as f64;
    let dequant_meps = numel as f64 / dequant_s / 1e6;
    let dequant_gbs = (numel as f64 * 8.0 / 1e9) / dequant_s; // 4B read + 4B write per element
    println!("[2] fp8::dequant_block128 ({reps} reps):        {:>9.2} ms  ({dequant_meps:>8.1} Melem/s, {dequant_gbs:>6.2} GB/s r+w)", dequant_s * 1e3);

    // --- Stage 3: quantize_weight, isolated, repeated for a stable mean --
    assert_eq!(cols % model::int8::GROUP, 0, "quantize_weight needs k%32==0 (k={cols})");
    let mut quant_total = 0f64;
    for _ in 0..reps {
        let t2 = Instant::now();
        let _ = model::int8::quantize_weight(&dequant_out, rows, cols);
        quant_total += t2.elapsed().as_secs_f64();
    }
    let quant_s = quant_total / reps as f64;
    let quant_meps = numel as f64 / quant_s / 1e6;
    let quant_gbs = (numel as f64 * 5.0 / 1e9) / quant_s; // 4B read + 1B write per element
    println!("[3] int8::quantize_weight ({reps} reps):        {:>9.2} ms  ({quant_meps:>8.1} Melem/s, {quant_gbs:>6.2} GB/s r+w)", quant_s * 1e3);

    let total_import_ms = (decode_s + dequant_s + quant_s) * 1e3;
    println!(
        "\nsum of the three stages for THIS ONE tensor: {total_import_ms:.2} ms (decode {:.1}%, dequant {:.1}%, quantize {:.1}%)",
        100.0 * decode_s / (decode_s + dequant_s + quant_s),
        100.0 * dequant_s / (decode_s + dequant_s + quant_s),
        100.0 * quant_s / (decode_s + dequant_s + quant_s),
    );
}
