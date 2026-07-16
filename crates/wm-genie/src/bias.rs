// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Position biases fed to the STBlock attention, both precomputed host-side
//! (one-time per resolution, tiny): the temporal ALiBi and the spatial
//! ContinuousPositionBias MLP. Layout matches the kernels: `[heads, i, j]`.

/// ALiBi slopes (lucidrains / GenieRedux `AlibiPositionalBias._get_slopes`).
/// Exact for a power-of-two head count (GenieRedux uses 8); the non-pow2
/// interleave fallback is included for completeness.
fn alibi_slopes(heads: usize) -> Vec<f32> {
    fn pow2(n: usize) -> Vec<f32> {
        let start = 2f64.powf(-(2f64.powf(-((n as f64).log2() - 3.0))));
        (0..n).map(|i| (start * start.powi(i as i32)) as f32).collect()
    }
    if (heads & (heads - 1)) == 0 {
        return pow2(heads);
    }
    let closest = 1usize << (heads as f64).log2().floor() as usize;
    let mut s = pow2(closest);
    let extra = pow2(2 * closest);
    s.extend(extra.iter().step_by(2).take(heads - closest));
    s
}

/// Temporal ALiBi bias `[heads, t, t]`: `bias[h,i,j] = -slope_h * |i - j|`.
/// (The causal `j>i` mask is applied by `attn_scores_causal_bias`, so those
/// entries are overwritten downstream; here they hold the symmetric value.)
pub fn alibi_bias(heads: usize, t: usize) -> Vec<f32> {
    let slopes = alibi_slopes(heads);
    let mut b = vec![0.0f32; heads * t * t];
    for h in 0..heads {
        for i in 0..t {
            for j in 0..t {
                b[(h * t + i) * t + j] = -slopes[h] * (i as f32 - j as f32).abs();
            }
        }
    }
    b
}

/// One linear layer of the ContinuousPositionBias MLP: `w` is `[out,in]`
/// row-major (torch `nn.Linear`), `b` is `[out]`.
pub struct CpbLayer {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub in_dim: usize,
    pub out_dim: usize,
}

const LEAKY_SLOPE: f32 = 0.1;

/// Spatial ContinuousPositionBias `[heads, hw, hw]` for an `h×w` grid.
///
/// Relative positions `grid[i]-grid[j]` (2-D), log-distance transformed
/// (`sign·log(|·|+1)`), then the MLP: `Linear→LeakyReLU(0.1)` for every layer
/// except the last (`Linear→heads`). Matches `ContinuousPositionBias.forward`
/// (num_dims=2). The final layer's `out_dim` must equal `heads`.
pub fn cpb_bias(net: &[CpbLayer], h: usize, w: usize, heads: usize) -> Vec<f32> {
    let hw = h * w;
    // relative positions, log-distance transformed: rel[(p1,p2)] = (dr, dc).
    let coord = |p: usize| ((p / w) as f32, (p % w) as f32);
    let mut x = vec![0.0f32; hw * hw * 2];
    for p1 in 0..hw {
        let (r1, c1) = coord(p1);
        for p2 in 0..hw {
            let (r2, c2) = coord(p2);
            let ld = |v: f32| v.signum() * (v.abs() + 1.0).ln();
            let base = (p1 * hw + p2) * 2;
            x[base] = ld(r1 - r2);
            x[base + 1] = ld(c1 - c2);
        }
    }
    // MLP over the (hw*hw) rows.
    let rows = hw * hw;
    let mut cur = x;
    let mut cur_dim = 2usize;
    for (li, layer) in net.iter().enumerate() {
        assert_eq!(layer.in_dim, cur_dim, "cpb layer {li} in_dim mismatch");
        let mut out = vec![0.0f32; rows * layer.out_dim];
        for r in 0..rows {
            for o in 0..layer.out_dim {
                let mut acc = layer.b[o];
                for k in 0..cur_dim {
                    acc += cur[r * cur_dim + k] * layer.w[o * cur_dim + k];
                }
                // LeakyReLU on every layer except the final one.
                if li + 1 < net.len() && acc < 0.0 {
                    acc *= LEAKY_SLOPE;
                }
                out[r * layer.out_dim + o] = acc;
            }
        }
        cur = out;
        cur_dim = layer.out_dim;
    }
    assert_eq!(cur_dim, heads, "cpb final out_dim must equal heads");
    // cur is [hw*hw, heads] -> [heads, hw, hw]
    let mut bias = vec![0.0f32; heads * hw * hw];
    for p1 in 0..hw {
        for p2 in 0..hw {
            for h_ in 0..heads {
                bias[(h_ * hw + p1) * hw + p2] = cur[(p1 * hw + p2) * heads + h_];
            }
        }
    }
    bias
}
