// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! T5's relative-position **bucket table** — integer index math, computed once
//! on the host.
//!
//! This is not arithmetic that belongs on an accelerator (AGENTS "host math
//! does not run on the accelerator"): it produces a `[t*t]` table of `u32`
//! bucket ids that is uploaded once and then *gathered* on the device by the
//! `embed` kernel, exactly as `sam2::hostpe` precomputes a positional encoding.
//! It is O(t²) integer work run once at model construction, never per forward.
//!
//! brain's existing `rel_shift` kernel is the **Transformer-XL** relative-
//! position shift (`crates/nemotron`'s Conformer attention): a pure reindex of
//! an already-computed `[rows, q, p]` score slab. T5 is a different mechanism —
//! a learned `[num_buckets, heads]` embedding gathered through a *logarithmic
//! bucketing* of `key - query` and added to the scores — so `rel_shift` is not
//! reusable here and nothing was re-implemented.
//!
//! The formula is Mesh-TensorFlow's, as carried into
//! `transformers.models.t5.modeling_t5.T5Attention._relative_position_bucket`
//! with `bidirectional=True` (an encoder):
//!
//! ```text
//! half      = num_buckets / 2
//! max_exact = half / 2
//! rel       = key - query                       (memory - query position)
//! bucket    = (rel > 0 ? half : 0)
//!           + if |rel| < max_exact { |rel| }
//!             else { min(max_exact + trunc(ln(|rel|/max_exact)
//!                                          / ln(max_distance/max_exact)
//!                                          * (half - max_exact)),
//!                        half - 1) }
//! ```
//!
//! The logarithmic branch is evaluated in **f32**, matching the reference (the
//! division is a torch f32 tensor op); the truncation is toward zero, as
//! `Tensor::to(torch.long)` is. `tests/parity.rs` gates the whole table against
//! the dumped `relative_position_bucket` golden, so this is checked, not
//! asserted.

/// The `[t*t]` bucket table in row-major `(query, key)` order:
/// `out[i*t + j]` is the bucket of key `j` seen from query `i`.
pub fn buckets(t: u32, num_buckets: u32, max_distance: u32) -> Vec<u32> {
    let half = num_buckets / 2;
    let max_exact = half / 2;
    let denom = (max_distance as f64 / max_exact as f64).ln() as f32;
    let span = (half - max_exact) as f32;
    let mut out = vec![0u32; (t as usize) * (t as usize)];
    for i in 0..t {
        for j in 0..t {
            let dir = if j > i { half } else { 0 };
            let a = i.abs_diff(j);
            let off = if a < max_exact {
                a
            } else {
                let big = (a as f32 / max_exact as f32).ln() / denom * span;
                (max_exact + big as u32).min(half - 1)
            };
            out[(i * t + j) as usize] = dir + off;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t5_bucket_table_structure() {
        let t = 128u32;
        let b = buckets(t, 32, 128);
        assert_eq!(b.len(), (t * t) as usize);
        // the diagonal is bucket 0, and |rel| < max_exact (=8) is the identity
        for i in 0..t {
            assert_eq!(b[(i * t + i) as usize], 0);
        }
        assert_eq!(b[(10 * t + 13) as usize], 16 + 3, "key ahead by 3");
        assert_eq!(b[(13 * t + 10) as usize], 3, "key behind by 3");
        // every bucket except the structurally unreachable `half` (rel > 0 with
        // |rel| == 0 cannot happen) must appear at t=128
        let mut seen = [false; 32];
        for &v in &b {
            seen[v as usize] = true;
        }
        for (k, &s) in seen.iter().enumerate() {
            assert_eq!(s, k != 16, "bucket {k} reachability");
        }
        // the far end saturates at half-1 / num_buckets-1
        assert_eq!(b[127], 31, "query 0, key 127");
        assert_eq!(b[(127 * t) as usize], 15, "query 127, key 0");
    }
}
