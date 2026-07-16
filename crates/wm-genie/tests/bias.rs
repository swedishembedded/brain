// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Position-bias helpers: ALiBi structure + ContinuousPositionBias invariants.
use wm_genie::bias::{alibi_bias, cpb_bias, CpbLayer};

#[test]
fn alibi_has_expected_structure() {
    let (heads, t) = (8usize, 6usize);
    let b = alibi_bias(heads, t);
    for h in 0..heads {
        for i in 0..t {
            // zero on the diagonal
            assert_eq!(b[(h*t+i)*t+i], 0.0);
            for j in 0..t {
                // symmetric and non-positive
                assert_eq!(b[(h*t+i)*t+j], b[(h*t+j)*t+i]);
                assert!(b[(h*t+i)*t+j] <= 0.0);
            }
            // strictly more negative as |i-j| grows (slope>0)
            if i >= 2 {
                assert!(b[(h*t+i)*t+0] < b[(h*t+i)*t+1]);
            }
        }
    }
    // head 0 has the largest slope (steepest); head 7 the smallest.
    assert!(b[(0*t+0)*t+(t-1)] < b[(7*t+0)*t+(t-1)]);
}

fn rand(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| { s = s.wrapping_add(0x9E3779B97F4A7C15); let mut z=s;
        z=(z^(z>>30)).wrapping_mul(0xBF58476D1CE4E5B9); z=(z^(z>>27)).wrapping_mul(0x94D049BB133111EB);
        ((( (z^(z>>31))>>40) as f32)/(1u64<<24) as f32 - 0.5)*2.0 }).collect()
}

#[test]
fn cpb_diagonal_is_constant_and_shaped() {
    // rel_pos on the diagonal (i==i) is always (0,0), so every self-bias
    // bias[h,p,p] must be identical across p — a real invariant of the MLP.
    let (h, w, heads, dim) = (3usize, 3usize, 8usize, 16usize);
    let hw = h*w;
    let net = vec![
        CpbLayer { w: rand(1, dim*2), b: rand(2, dim), in_dim: 2, out_dim: dim },
        CpbLayer { w: rand(3, dim*dim), b: rand(4, dim), in_dim: dim, out_dim: dim },
        CpbLayer { w: rand(5, heads*dim), b: rand(6, heads), in_dim: dim, out_dim: heads },
    ];
    let b = cpb_bias(&net, h, w, heads);
    assert_eq!(b.len(), heads*hw*hw);
    for hd in 0..heads {
        let d0 = b[(hd*hw+0)*hw+0];
        for p in 1..hw {
            assert!((b[(hd*hw+p)*hw+p] - d0).abs() < 1e-5, "cpb diagonal not constant");
        }
    }
    // symmetric relative positions p1->p2 vs p2->p1 differ (rel_pos negates,
    // and the MLP is not even) — sanity that it's not degenerate/all-equal.
    assert!((b[(0*hw+0)*hw+1] - b[(0*hw+1)*hw+0]).abs() > 1e-6);
}
