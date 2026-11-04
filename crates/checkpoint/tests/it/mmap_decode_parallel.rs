// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The spec for the streaming reader's dtype decode: fanning the conversion
//! out across cores must not move a single bit, at any tensor length.
//!
//! Swedish Embedded AB implements checkpoint loading and streaming
//! weight-import paths for its clients. If your team needs expertise in
//! on-device model loading then you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! `MmapSafetensors::tensor_f32` decodes through a chunked, host-parallel
//! converter. The oracle here is a plain serial loop written out in this
//! file, deliberately NOT `safetensors::parse`: that now shares the very
//! implementation under test, so comparing against it would be comparing a
//! thing to itself.
//!
//! The property is EQUALITY OF BITS, not closeness. Every conversion is
//! exact (a widening of f16/bf16, or an integer that fits), so any difference
//! at all means a real defect: a dropped ragged tail, an off-by-one chunk
//! base, or two threads writing the same slot. Tensor lengths are chosen to
//! straddle the chunk size in both directions - shorter than one chunk, an
//! exact multiple of it, and a multiple plus a ragged remainder - because a
//! chunked decoder that mishandles the tail passes every length that happens
//! to divide evenly.

use checkpoint::mmap::MmapSafetensors;

/// The chunk width the parallel decoder uses internally. The interesting
/// lengths are the ones around it.
const CHUNK: usize = 1 << 16;

fn f16_bits(x: f32) -> u16 {
    // Only exactly-representable values are used below, so this narrow
    // encoder does not need subnormal or rounding handling.
    let b = x.to_bits();
    let sign = ((b >> 31) & 1) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = ((b >> 13) & 0x3ff) as u16;
    if x == 0.0 {
        return sign << 15;
    }
    (sign << 15) | ((exp as u16) << 10) | mant
}

/// Deterministic values that are exact in EVERY dtype under test, so the
/// comparison can be on bits rather than a tolerance.
///
/// `m * 2^e` with `m` in `1..=127` (seven significand bits) and `e` in
/// `0..=7`: exact in bf16 (eight significand bits - the tightest here), in
/// f16 and f32, and an integer, so the I32/I64 arms are lossless too. A
/// plain `i % k` ramp would NOT do: its period would divide the decoder's
/// 65536-element chunk width, and a whole-chunk indexing error would then
/// reproduce the expected values exactly and pass. This sequence's period is
/// 1016, which does not divide 65536, so a shifted chunk shows up.
fn value(i: usize) -> f32 {
    let m = ((i % 127) + 1) as f32;
    let e = ((i / 127) % 8) as i32;
    let v = m * (1i32 << e) as f32;
    if i.is_multiple_of(3) {
        -v
    } else {
        v
    }
}

/// Build a one-tensor safetensors file of `n` elements in `dtype`, plus the
/// serial reference decoding of it.
fn build(dtype: &str, n: usize) -> (std::path::PathBuf, Vec<f32>) {
    let mut blob = Vec::new();
    let mut want = Vec::with_capacity(n);
    for i in 0..n {
        let v = value(i);
        want.push(v);
        match dtype {
            "F32" => blob.extend_from_slice(&v.to_le_bytes()),
            "F16" => blob.extend_from_slice(&f16_bits(v).to_le_bytes()),
            "BF16" => blob.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes()),
            "I32" => blob.extend_from_slice(&(v as i32).to_le_bytes()),
            "I64" => blob.extend_from_slice(&(v as i64).to_le_bytes()),
            "U8" => blob.push((i % 256) as u8),
            other => panic!("unhandled dtype {other}"),
        }
    }
    if dtype == "U8" {
        // U8 cannot carry `value`'s signed range; its reference is the byte.
        want = (0..n).map(|i| (i % 256) as f32).collect();
    }
    let header = serde_json::json!({
        "w": {"dtype": dtype, "shape": [n], "data_offsets": [0, blob.len()]},
    });
    let hbytes = serde_json::to_vec(&header).unwrap();
    let mut file = (hbytes.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&hbytes);
    file.extend_from_slice(&blob);

    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let c = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("checkpoint-mmap-decode-{}-{c}.safetensors", std::process::id()));
    std::fs::write(&path, &file).unwrap();
    (path, want)
}

#[test]
fn parallel_decode_matches_a_serial_reference_for_every_dtype_and_length() {
    // Shorter than a chunk; exactly one chunk; several whole chunks; and
    // whole chunks plus a ragged remainder - the tail case a chunked decoder
    // gets wrong while passing every evenly-divisible length.
    let lengths = [1usize, 1000, CHUNK, CHUNK + 1, 3 * CHUNK, 3 * CHUNK + 777];
    for dtype in ["F32", "F16", "BF16", "I32", "I64", "U8"] {
        for n in lengths {
            let (path, want) = build(dtype, n);
            let m = MmapSafetensors::open(&path).unwrap();
            assert_eq!(m.numel("w"), Some(n), "{dtype} n={n}: numel");
            let got = m.tensor_f32("w").unwrap_or_else(|| panic!("{dtype} n={n}: no tensor"));
            assert_eq!(got.len(), n, "{dtype} n={n}: length");
            // Bits, not a tolerance: every conversion here is exact.
            assert_eq!(got, want, "{dtype} n={n}: parallel decode must equal the serial reference");
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// The chunked accessor must agree with the whole-tensor one element for
/// element, including when the caller's chunk size is unrelated to the
/// decoder's internal one.
#[test]
fn chunked_reads_agree_with_the_whole_tensor_decode() {
    let n = 3 * CHUNK + 777;
    let (path, want) = build("BF16", n);
    let m = MmapSafetensors::open(&path).unwrap();

    for max_elems in [1usize, 999, CHUNK, n] {
        let mut got = vec![0f32; n];
        let mut pieces = 0usize;
        assert!(
            m.with_tensor_chunks("w", max_elems, &mut |off, d| {
                got[off as usize..off as usize + d.len()].copy_from_slice(d);
                pieces += 1;
            }),
            "max_elems={max_elems}: chunked read refused"
        );
        assert!(pieces >= 1);
        assert_eq!(got, want, "max_elems={max_elems}: chunked decode must equal the serial reference");
    }
    let _ = std::fs::remove_file(&path);
}
