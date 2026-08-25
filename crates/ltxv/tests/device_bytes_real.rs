// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device-bytes assertion for int8 storage, at REAL LTX-2.5 22B dims - this
//! port's own roadmap ledger's lesson #34: "a memory saving is not measured
//! by anything unless someone measures it". `crate::int8::is_never_
//! quantized`'s exclusion list means the real ratio is NOT the theoretical
//! flat four-to-one an all-eligible-tensor model would get - this file computes the
//! real number from the real config's own tensor-size breakdown
//! ([`ltxv::dit::av_dit_tensor_manifest`] at [`LtxAvDitConfig::ltx25`]) and
//! asserts against it, rather than assuming a round number.
//!
//! No fixture dependency: this needs only the config's own shapes (a
//! manifest walk, no tensor DATA), so it always runs, at millisecond cost -
//! the same "shapes alone, no download" gate `import::dry_run`'s own tests
//! already use for two-way coverage.

use ltxv::dit::av_dit_tensor_manifest;
use ltxv::int8::is_never_quantized;
use ltxv::LtxAvDitConfig;

/// A tensor is int8-storage-eligible iff it is a plain `[n, k]` matrix with
/// `k` a multiple of 4 (`model::int8::quantize_weight`'s packing width) and
/// not on the never-quantize list - the exact predicate `crate::int8::
/// quantize_tensors` applies (re-derived here rather than imported, since
/// `is_eligible` there is private - the two must still agree, which is
/// exactly what `int8_storage.rs`'s own predicate-pin test already checks
/// against the video-only manifest).
fn is_eligible(name: &str, shape: &[usize]) -> bool {
    shape.len() == 2 && shape[1].is_multiple_of(4) && !is_never_quantized(name)
}

#[test]
fn real_22b_int8_storage_ratio_is_measured_not_assumed() {
    let cfg = LtxAvDitConfig::ltx25();
    let manifest = av_dit_tensor_manifest(&cfg);
    assert_eq!(manifest.len(), 4349, "sanity: real 22B/4349-tensor manifest");

    let mut fp32_bytes: u64 = 0;
    let mut int8_bytes: u64 = 0;
    let mut eligible_count = 0usize;
    let mut eligible_numel: u64 = 0;
    let mut total_numel: u64 = 0;

    for (name, shape) in &manifest {
        let numel: u64 = shape.iter().map(|&d| d as u64).product();
        total_numel += numel;
        fp32_bytes += numel * 4;
        if is_eligible(name, shape) {
            eligible_count += 1;
            eligible_numel += numel;
            // Packed int8 (1 byte/element) + one fp32 scale per output row.
            let n = shape[0] as u64;
            int8_bytes += numel + n * 4;
        } else {
            // Never-quantized / ineligible tensors stay fp32 in the storage
            // format too (`crate::int8::quantize_tensors`'s `full` map).
            int8_bytes += numel * 4;
        }
    }

    let ratio = fp32_bytes as f64 / int8_bytes as f64;
    let eligible_frac = eligible_numel as f64 / total_numel as f64;
    println!(
        "real 22B AV DiT int8 storage: {eligible_count}/{} tensors eligible ({eligible_frac:.4} of params), fp32={fp32_bytes} bytes, int8={int8_bytes} bytes, ratio={ratio:.4}x",
        manifest.len()
    );

    // Measured on the real config's own manifest: the vast majority of
    // parameters live in the ten quantizable linears per block (attention
    // Q/K/V/O + FFN, both streams, both AV cross-attention directions) at 48
    // layers - the never-quantized set (patchify/adaLN/proj_out/scale_shift/
    // to_gate_logits tables, plus the connectors' own small tensors) is a
    // small fraction of total parameters, so the ratio should land close to
    // (but strictly below, per `is_never_quantized`'s doc) the theoretical
    // four-to-one a fully-eligible model would get - asserted as a real
    // measured range, not a round number.
    assert!(eligible_frac > 0.9, "expected the vast majority of real 22B parameters to be int8-eligible, got {eligible_frac:.4}");
    assert!(ratio > 3.5 && ratio < 4.0, "real 22B int8 storage ratio {ratio:.4} outside the expected (3.5, 4.0) band - is_never_quantized's exclusions should keep it close to but below four-to-one");
}
