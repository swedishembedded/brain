// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Correctness gate for [`audio::conv::fold_weight_norm`], hoisted out of
//! `crates/minimaxmusic3/src/vocoder.rs` (a previously-paid-for bug class:
//! `weight_norm`'s `dim=0` is dim 0 of the STORED tensor, which for
//! `ConvTranspose1d` is `Cin`, not `Cout` - getting that axis wrong is
//! silent, not a shape error, since `Cin` and `Cout` are often both plausible
//! dimensions).

use audio::conv::fold_weight_norm;

#[test]
fn matches_pytorch_dim0_formula() {
    // d0=2 rows, rest=3: v = [[3,4,0],[0,0,5]] -> ||v[0]||=5, ||v[1]||=5.
    let v = [3.0f32, 4.0, 0.0, 0.0, 0.0, 5.0];
    let g = [2.0f32, 10.0];
    let out = fold_weight_norm(&g, &v, 2);
    // row0: g/||v0|| * v0 = (2/5)*[3,4,0] = [1.2, 1.6, 0.0]
    // row1: g/||v1|| * v1 = (10/5)*[0,0,5] = [0,0,10]
    assert!((out[0] - 1.2).abs() < 1e-6);
    assert!((out[1] - 1.6).abs() < 1e-6);
    assert!((out[2] - 0.0).abs() < 1e-6);
    assert!((out[5] - 10.0).abs() < 1e-6);
}

#[test]
#[should_panic(expected = "weight_g")]
fn mismatched_d0_panics() {
    let v = [1.0f32; 6];
    let g = [1.0f32; 3];
    fold_weight_norm(&g, &v, 2);
}
