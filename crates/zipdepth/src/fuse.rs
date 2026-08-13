// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! RepVGG reparameterization: collapse `QARepBlock`'s three branches into one
//! biased 3x3 conv.
//!
//! Train-time the block is
//!     y = relu( BN3(conv3x3(x)) + BN1(conv1x1(x)) [+ x] )
//! and inference-time it is
//!     y = relu( conv3x3(x, k, b) )
//! for a single `(k, b)` derived from the branches. Same function, one third of
//! the dispatches — this is what takes the released model from 6.79M parameters
//! to the 6.1M headline figure, and it is why the checkpoint stores the LARGER,
//! unfused form.
//!
//! Purely host-side weight arithmetic, so it needs no kernel and no device. It
//! reuses `vision::fold_bn` per branch — the same fold `Conv`'s eval path and the
//! ONNX exporter already apply, so all three agree by construction rather than by
//! three independent derivations of the same formula.
//!
//! ⚠️ The identity branch has **no BN** (`architecture.py:89-108`), unlike
//! canonical RepVGG. So it contributes a kernel but **no bias term**. Adding one
//! is the obvious mistake here and it is silent: the model still runs, still has
//! the right shapes, and is simply wrong.

use vision::fold_bn;

/// A conv+BN branch's raw (unfolded) tensors.
pub struct Branch<'a> {
    pub weight: &'a [f32],
    pub gamma: &'a [f32],
    pub beta: &'a [f32],
    pub run_mean: &'a [f32],
    pub run_var: &'a [f32],
}

/// Fuse `QARepBlock` into `(kernel[cout, cin/groups, 3, 3], bias[cout])`.
///
/// `has_identity` must be `cin == cout && stride == 1` — the reference's own
/// condition (`architecture.py:89`).
pub fn fuse_qarep(
    b3: &Branch,
    b1: &Branch,
    cin: usize,
    cout: usize,
    groups: usize,
    has_identity: bool,
) -> (Vec<f32>, Vec<f32>) {
    let cin_g = cin / groups;
    assert_eq!(b3.weight.len(), cout * cin_g * 9, "3x3 branch weight shape");
    assert_eq!(b1.weight.len(), cout * cin_g, "1x1 branch weight shape");

    // Each branch folds its own BN first — the same fold as everywhere else.
    let (w3, bias3) = fold_bn(b3.weight, b3.gamma, b3.beta, b3.run_mean, b3.run_var, cout);
    let (w1, bias1) = fold_bn(b1.weight, b1.gamma, b1.beta, b1.run_mean, b1.run_var, cout);

    // The 1x1 kernel sits at the CENTRE of a zero-padded 3x3 (F.pad(k1,[1,1,1,1])).
    let mut kernel = w3;
    for o in 0..cout {
        for i in 0..cin_g {
            kernel[(o * cin_g + i) * 9 + 4] += w1[o * cin_g + i];
        }
    }
    // Identity: a 3x3 whose centre tap is 1 on the channel that maps to itself.
    // `k_id[i, i % (cin/groups), 1, 1] = 1` — for groups=1 that is i==i (a plain
    // identity); for depthwise (groups==cin) cin/groups==1 so it is [i, 0, 1, 1].
    if has_identity {
        assert_eq!(cin, cout, "identity branch requires cin == cout");
        for i in 0..cout {
            kernel[(i * cin_g + (i % cin_g)) * 9 + 4] += 1.0;
        }
    }
    // NO identity bias term — that branch has no BN.
    let bias: Vec<f32> = bias3.iter().zip(&bias1).map(|(a, b)| a + b).collect();
    (kernel, bias)
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::rng::Lcg;

    /// Host reference: `conv2d(x, k, b)` at stride 1, pad 1, NCHW, grouped.
    fn conv3x3(x: &[f32], k: &[f32], b: Option<&[f32]>, cin: usize, cout: usize, h: usize, w: usize, groups: usize) -> Vec<f32> {
        let cin_g = cin / groups;
        let cout_g = cout / groups;
        let mut y = vec![0.0f32; cout * h * w];
        for co in 0..cout {
            let g = co / cout_g;
            for oh in 0..h {
                for ow in 0..w {
                    let mut acc = b.map_or(0.0, |bb| bb[co]);
                    for cl in 0..cin_g {
                        let ci = g * cin_g + cl;
                        for kh in 0..3usize {
                            for kw in 0..3usize {
                                let ih = oh as i32 + kh as i32 - 1;
                                let iw = ow as i32 + kw as i32 - 1;
                                if ih >= 0 && iw >= 0 && (ih as usize) < h && (iw as usize) < w {
                                    acc += x[(ci * h + ih as usize) * w + iw as usize]
                                        * k[((co * cin_g + cl) * 3 + kh) * 3 + kw];
                                }
                            }
                        }
                    }
                    y[(co * h + oh) * w + ow] = acc;
                }
            }
        }
        y
    }
    fn conv1x1(x: &[f32], k: &[f32], b: &[f32], cin: usize, cout: usize, hw: usize, groups: usize) -> Vec<f32> {
        let cin_g = cin / groups;
        let cout_g = cout / groups;
        let mut y = vec![0.0f32; cout * hw];
        for co in 0..cout {
            let g = co / cout_g;
            for i in 0..hw {
                let mut acc = b[co];
                for cl in 0..cin_g {
                    acc += x[(g * cin_g + cl) * hw + i] * k[co * cin_g + cl];
                }
                y[co * hw + i] = acc;
            }
        }
        y
    }

    /// run_var must be positive.
    fn rvar(seed: u64, n: usize) -> Vec<f32> {
        Lcg::new(seed).vec(n).iter().map(|v| v.abs() + 0.1).collect()
    }

    /// THE property: the fused conv computes exactly what the three branches did.
    /// Checked against an independent host convolution, for all x — which is what
    /// makes it a fuse rather than an approximation.
    fn assert_fuse_equivalent(cin: usize, cout: usize, groups: usize, has_identity: bool) {
        let (h, w) = (5usize, 4usize);
        let cin_g = cin / groups;
        let x = Lcg::new(1).vec(cin * h * w);
        let (w3, g3, b3, m3, v3) = (Lcg::new(2).vec(cout * cin_g * 9), Lcg::new(3).vec(cout), Lcg::new(4).vec(cout), Lcg::new(5).vec(cout), rvar(6, cout));
        let (w1, g1, b1, m1, v1) = (Lcg::new(7).vec(cout * cin_g), Lcg::new(8).vec(cout), Lcg::new(9).vec(cout), Lcg::new(10).vec(cout), rvar(11, cout));
        let br3 = Branch { weight: &w3, gamma: &g3, beta: &b3, run_mean: &m3, run_var: &v3 };
        let br1 = Branch { weight: &w1, gamma: &g1, beta: &b1, run_mean: &m1, run_var: &v1 };

        // Reference: the unfused three-branch forward (pre-ReLU; ReLU is outside).
        let (fw3, fb3) = fold_bn(&w3, &g3, &b3, &m3, &v3, cout);
        let (fw1, fb1) = fold_bn(&w1, &g1, &b1, &m1, &v1, cout);
        let y3 = conv3x3(&x, &fw3, Some(&fb3), cin, cout, h, w, groups);
        let y1 = conv1x1(&x, &fw1, &fb1, cin, cout, h * w, groups);
        let mut want: Vec<f32> = y3.iter().zip(&y1).map(|(a, b)| a + b).collect();
        if has_identity {
            for i in 0..want.len() {
                want[i] += x[i];
            }
        }

        let (k, b) = fuse_qarep(&br3, &br1, cin, cout, groups, has_identity);
        let got = conv3x3(&x, &k, Some(&b), cin, cout, h, w, groups);
        for i in 0..want.len() {
            assert!(
                (got[i] - want[i]).abs() < 1e-4,
                "cin={cin} cout={cout} g={groups} id={has_identity} at {i}: fused {} != branches {}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn fused_conv_equals_the_three_branches() {
        assert_fuse_equivalent(8, 8, 1, true); // the residual case
        assert_fuse_equivalent(8, 8, 1, false);
        assert_fuse_equivalent(4, 8, 1, false); // downsample: cin != cout, no identity
        assert_fuse_equivalent(8, 8, 4, true); // grouped + identity
        assert_fuse_equivalent(6, 6, 6, true); // depthwise: cin/groups == 1
    }

    /// The identity branch contributes a KERNEL but no BIAS, because it has no BN.
    /// If someone "fixes" that by adding one, this fires.
    #[test]
    fn identity_branch_adds_no_bias() {
        let (cin, cout) = (4usize, 4usize);
        let z = vec![0.0f32; cout];
        let one = vec![1.0f32; cout];
        // Zero weights everywhere, zero BN shift -> the ONLY contribution is identity.
        let w3 = vec![0.0f32; cout * cin * 9];
        let w1 = vec![0.0f32; cout * cin];
        let br3 = Branch { weight: &w3, gamma: &z, beta: &z, run_mean: &z, run_var: &one };
        let br1 = Branch { weight: &w1, gamma: &z, beta: &z, run_mean: &z, run_var: &one };

        let (k, b) = fuse_qarep(&br3, &br1, cin, cout, 1, true);
        assert!(b.iter().all(|v| *v == 0.0), "identity must contribute no bias, got {b:?}");
        // ...and the kernel is exactly the identity: centre tap 1 on the diagonal.
        for o in 0..cout {
            for i in 0..cin {
                for t in 0..9 {
                    let want = if o == i && t == 4 { 1.0 } else { 0.0 };
                    assert_eq!(k[(o * cin + i) * 9 + t], want, "k[{o},{i},{t}]");
                }
            }
        }
    }
}
