// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Fixtures for the packed-tensor tests: deterministic byte streams and encoders.

/// Deterministic pseudo-random bytes (a fixed LCG: the tests need a repeatable
/// stream, not entropy).
pub fn bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 33) as u8
        })
        .collect()
}

/// FP8 E4M3FN bytes with the NaN codes (0x7F, 0xFF) replaced, so comparisons are exact.
pub fn e4m3_bytes(seed: u64, n: usize) -> Vec<u8> {
    bytes(seed, n).into_iter().map(|b| if b & 0x7F == 0x7F { b & 0xFE } else { b }).collect()
}

pub fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|&v| half::bf16::from_f32(v).to_le_bytes()).collect()
}

pub fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}
