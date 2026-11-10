// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: every primitive of the training graph has a backward equal to central finite differences of its own
//! forward in `f64` (block tolerance 1e-4).

use mirage::train::ops::Seg;
use mirage::train::seq::AttnShape;
use mirage::vision::ops::Map;

use super::fd::{check, noise};

#[test]
fn elementwise_ops() {
    let (a, b) = (noise(12, 1, 1.0), noise(12, 2, 1.0));
    check("add", &[a.clone(), b.clone()], &|t, i| t.add(i[0], i[1]));
    check("mul", &[a.clone(), b.clone()], &|t, i| t.mul(i[0], i[1]));
    check("scale", &[a.clone()], &|t, i| t.scale(i[0], 0.37));
    check("bias_add", &[a.clone(), noise(4, 3, 1.0)], &|t, i| t.bias_add(i[0], i[1]));
    check("chan_mul", &[a.clone(), noise(3, 4, 1.0)], &|t, i| t.chan_mul(i[0], i[1]));
    check("scalar_mul", &[a.clone(), noise(1, 5, 1.0)], &|t, i| t.scalar_mul(i[0], i[1]));
    check("sigmoid", &[a.clone()], &|t, i| t.sigmoid(i[0]));
    check("silu", &[a.clone()], &|t, i| t.silu(i[0]));
    check("gelu_tanh", &[a], &|t, i| t.gelu_tanh(i[0]));
}

#[test]
fn grad_scale_is_the_identity_forward_and_scales_only_the_gradient() {
    let mut t = mirage::train::tape::Tape::<f64>::new();
    let x = t.leaf(vec![1.0, -2.0, 3.0]);
    let y = t.grad_scale(x, 0.1);
    assert_eq!(t.val(y), t.val(x));
    let loss = t.sum_weighted(y, 1.0);
    let g = t.backward(loss);
    assert_eq!(g.get(x).unwrap(), &[0.1, 0.1, 0.1]);
}

#[test]
fn dense_and_norm_ops() {
    check("linear", &[noise(15, 6, 1.0), noise(20, 7, 1.0)], &|t, i| t.linear(i[0], i[1], 3, 5, 4));
    check("linear_nt", &[noise(15, 8, 1.0), noise(20, 9, 1.0)], &|t, i| t.linear_nt(i[0], i[1], 3, 5, 4));
    check("zc_norm", &[noise(18, 10, 1.0), noise(6, 11, 0.5)], &|t, i| t.zc_norm(i[0], 6, i[1]));
    check("rms_unit", &[noise(18, 12, 1.0)], &|t, i| t.rms_unit(i[0], 6));
    check("mask_rows", &[noise(12, 13, 1.0)], &|t, i| t.mask_rows(i[0], 4, &[true, false, true]));
    check("softmax_rows", &[noise(12, 14, 1.0)], &|t, i| t.softmax_rows(i[0], 4));
    check("row_dot", &[noise(12, 15, 1.0), noise(12, 16, 1.0)], &|t, i| t.row_dot(i[0], i[1], 4));
    check("row_mul", &[noise(3, 17, 1.0), noise(12, 18, 1.0)], &|t, i| t.row_mul(i[0], i[1], 4));
}

#[test]
fn data_movement_ops() {
    let segs = vec![Seg::copy(0, 4, 2), Seg { dst: 2, src: 0, len: 3, w: 0.5 }, Seg { dst: 3, src: 1, len: 2, w: -1.5 }];
    check("plan", &[noise(8, 19, 1.0)], &move |t, i| t.plan(i[0], 6, segs.clone()));
    check("gather_rows", &[noise(12, 20, 1.0)], &|t, i| t.gather_rows(i[0], 3, &[3, 0, 3]));
    check("concat", &[noise(5, 21, 1.0), noise(3, 22, 1.0)], &|t, i| t.concat(i[0], i[1]));
    check("slice", &[noise(9, 23, 1.0)], &|t, i| t.slice(i[0], 2, 5));
    check("upsample2", &[noise(12, 24, 1.0)], &|t, i| t.upsample2(i[0], Map { h: 2, w: 3, c: 2 }).0);
    check("permute_cols", &[noise(12, 25, 1.0)], &|t, i| t.permute_cols(i[0], 4, &[2, 0, 3, 1]));
}

#[test]
fn convolutions() {
    let m = Map { h: 5, w: 6, c: 3 };
    for (sy, sx) in [(1, 1), (2, 2), (2, 1)] {
        check(&format!("depthwise stride {sy}x{sx}"), &[noise(90, 26, 1.0), noise(27, 27, 1.0)], &move |t, i| t.depthwise(i[0], m, i[1], 3, 3, sy, sx).0);
        check(&format!("conv_dense stride {sy}x{sx}"), &[noise(90, 28, 1.0), noise(3 * 3 * 3 * 4, 29, 1.0)], &move |t, i| t.conv_dense(i[0], m, i[1], 3, 3, 4, sy, sx).0);
    }
    check("depthwise 1xk", &[noise(18, 30, 1.0), noise(15, 31, 1.0)], &|t, i| t.depthwise(i[0], Map { h: 1, w: 6, c: 3 }, i[1], 1, 5, 1, 1).0);
}

#[test]
fn channel_mixing_ops() {
    check("walsh", &[noise(24, 32, 1.0)], &|t, i| t.walsh(i[0], 8));
    check("band", &[noise(24, 33, 1.0), noise(8 * 5, 34, 1.0)], &|t, i| t.band(i[0], 8, i[1], 5));
    check("band wider than the row", &[noise(8, 35, 1.0), noise(4 * 9, 36, 1.0)], &|t, i| t.band(i[0], 4, i[1], 9));
    let valid = [true, true, false, true, false];
    check("grn_scale", &[noise(20, 37, 1.0), noise(4, 38, 1.0)], &move |t, i| t.grn_scale(i[0], 4, &valid, i[1]));
}

#[test]
fn walsh_is_its_own_inverse() {
    let mut t = mirage::train::tape::Tape::<f64>::new();
    let x = t.leaf(noise(16, 39, 1.0));
    let y = t.walsh(x, 16);
    let z = t.walsh(y, 16);
    let worst = t.val(x).iter().zip(t.val(z)).map(|(a, b)| (a - b).abs()).fold(0f64, f64::max);
    assert!(worst < 1e-12, "the Walsh transform applied twice moved a value by {worst}");
}

#[test]
fn sequence_ops() {
    let angles = noise(3 * 2, 40, 1.5);
    check("rope", &[noise(3 * 2 * 4, 41, 1.0)], &move |t, i| t.rope(i[0], 2, 4, &angles));
    for causal in [false, true] {
        // Two query heads over one key head (grouped), keys and values of different widths.
        let s = AttnShape { tq: 4, tk: 4, hq: 2, hk: 1, qk: 3, vh: 2, causal };
        check(&format!("attention causal={causal}"), &[noise(4 * 2 * 3, 42, 1.0), noise(4 * 3, 43, 1.0), noise(4 * 2, 44, 1.0)], &move |t, i| t.attention(i[0], i[1], i[2], s));
    }
    let cross = AttnShape { tq: 3, tk: 5, hq: 2, hk: 2, qk: 2, vh: 2, causal: false };
    check("cross attention", &[noise(3 * 2 * 2, 45, 1.0), noise(5 * 2 * 2, 46, 1.0), noise(5 * 2 * 2, 47, 1.0)], &move |t, i| t.attention(i[0], i[1], i[2], cross));
    check("causal_taps", &[noise(7 * 3, 48, 1.0), noise(4 * 3, 49, 1.0)], &|t, i| t.causal_taps(i[0], i[1], 3, 3));
    check("causal_taps dilation 1", &[noise(5 * 3, 50, 1.0), noise(3 * 3, 51, 1.0)], &|t, i| t.causal_taps(i[0], i[1], 3, 1));
    check("sinkhorn", &[noise(2 * 16, 52, 0.7)], &|t, i| t.sinkhorn(i[0], 4, 20));
    check("row_mix res", &[noise(2 * 9, 53, 1.0), noise(2 * 3 * 2, 54, 1.0)], &|t, i| t.row_mix(i[0], i[1], 3, 3, 2));
    check("row_mix pre", &[noise(2 * 3, 55, 1.0), noise(2 * 3 * 2, 56, 1.0)], &|t, i| t.row_mix(i[0], i[1], 1, 3, 2));
    check("row_mix post", &[noise(2 * 3, 57, 1.0), noise(2 * 2, 58, 1.0)], &|t, i| t.row_mix(i[0], i[1], 3, 1, 2));
    check("kron", &[noise(2 * 8, 59, 1.0), noise(4, 60, 1.0), noise(16, 61, 1.0)], &|t, i| t.kron(i[0], i[1], i[2], 2, 4));
}

#[test]
fn sinkhorn_rows_and_columns_sum_to_one() {
    let mut t = mirage::train::tape::Tape::<f64>::new();
    let x = t.leaf(noise(16, 62, 0.7));
    let y = t.sinkhorn(x, 4, 20);
    let m = t.val(y);
    for i in 0..4 {
        let row: f64 = m[i * 4..(i + 1) * 4].iter().sum();
        let col: f64 = (0..4).map(|r| m[r * 4 + i]).sum();
        assert!((row - 1.0).abs() < 1e-6 && (col - 1.0).abs() < 1e-6, "row {row} column {col}");
    }
}
