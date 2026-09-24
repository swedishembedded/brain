// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Putative descriptor matches between two images: nearest neighbours under
//! Lowe's ratio test (IJCV 2004, §7.1), kept only when mutual.
//!
//! Descriptors are unit-norm RootSIFT, so `|a-b|² = 2 - 2·a·b` and the nearest
//! neighbour is the largest dot product.

use crate::sift::DESC;

/// For each descriptor of `a`, its best and second-best dot products in `b`
/// and the index of the best.
fn best_two(a: &[f32], b: &[f32]) -> Vec<(usize, f32, f32)> {
    let nb = b.len() / DESC;
    backend_cpu::par::map(a.len() / DESC, |i| {
        let da = &a[i * DESC..i * DESC + DESC];
        let (mut bi, mut b1, mut b2) = (usize::MAX, -1.0f32, -1.0f32);
        for j in 0..nb {
            let db = &b[j * DESC..j * DESC + DESC];
            let mut s = 0.0f32;
            for k in 0..DESC {
                s += da[k] * db[k];
            }
            if s > b1 {
                b2 = b1;
                b1 = s;
                bi = j;
            } else if s > b2 {
                b2 = s;
            }
        }
        (bi, b1, b2)
    })
}

/// Mutual nearest-neighbour matches passing the ratio test at `ratio` (on
/// Euclidean distance), as index pairs `(in a, in b)`.
pub fn match_descriptors(a: &[f32], b: &[f32], ratio: f32) -> Vec<(usize, usize)> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let ab = best_two(a, b);
    let ba = best_two(b, a);
    let dist = |s: f32| (2.0 - 2.0 * s).max(0.0).sqrt();
    ab.iter()
        .enumerate()
        .filter_map(|(i, &(j, s1, s2))| {
            if j == usize::MAX || ba[j].0 != i {
                return None;
            }
            (dist(s1) < ratio * dist(s2)).then_some((i, j))
        })
        .collect()
}
