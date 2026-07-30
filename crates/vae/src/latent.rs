// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FLUX.2 latent boundary ops: 2×2 pixel-unshuffle packing + frozen (eval-mode,
//! affine-free) BatchNorm normalization between the conv VAE and the DiT.
//!
//! Mirrors the BFL `AutoEncoder.encode/decode` wrapper: the posterior **mean**
//! `[C, H, W]` is rearranged `c (i pi) (j pj) → (c pi pj) i j` (packed channel
//! `= c·4 + pi·2 + pj`) and normalized per packed channel with the checkpoint's
//! `bn.running_{mean,var}` (`(z − μ)/√(σ² + eps)`, eps 1e-4); decode inverts.
//!
//! These run once per encode/decode on a `[4C, H/2, W/2]` tensor — cold
//! boundary ops, not a hot path — so host math is correct here (they are
//! model-layout transforms, not shared numeric primitives).

/// Pack a latent mean `[c, h, w]` (h, w even) into the normalized DiT latent
/// `[4c, h/2, w/2]`: 2×2 pixel-unshuffle (packed channel `= ci·4 + pi·2 + pj`,
/// `pi` = row, `pj` = col within the 2×2 patch), then per-channel
/// `(z − bn_mean)/√(bn_var + eps)`.
pub fn pack(
    mean: &[f32],
    c: usize,
    h: usize,
    w: usize,
    bn_mean: &[f32],
    bn_var: &[f32],
    eps: f32,
) -> Vec<f32> {
    assert_eq!(mean.len(), c * h * w, "pack: mean len != c*h*w");
    assert!(h % 2 == 0 && w % 2 == 0, "pack: h/w must be even ({h}x{w})");
    assert_eq!(bn_mean.len(), 4 * c, "pack: bn_mean len != 4c");
    assert_eq!(bn_var.len(), 4 * c, "pack: bn_var len != 4c");
    let (oh, ow) = (h / 2, w / 2);
    let mut out = vec![0.0f32; 4 * c * oh * ow];
    for ci in 0..c {
        for pi in 0..2 {
            for pj in 0..2 {
                let oc = ci * 4 + pi * 2 + pj;
                let s = (bn_var[oc] + eps).sqrt();
                let m = bn_mean[oc];
                for i in 0..oh {
                    for j in 0..ow {
                        let v = mean[(ci * h + 2 * i + pi) * w + 2 * j + pj];
                        out[(oc * oh + i) * ow + j] = (v - m) / s;
                    }
                }
            }
        }
    }
    out
}

/// Inverse of [`pack`]: a normalized DiT latent `[4c, h/2, w/2]` → the latent
/// mean `[c, h, w]` the conv decoder consumes. Per packed channel
/// `z·√(bn_var + eps) + bn_mean`, then 2×2 pixel-shuffle. `c, h, w` are the
/// **unpacked** dims (as in [`pack`]).
pub fn unpack(
    z: &[f32],
    c: usize,
    h: usize,
    w: usize,
    bn_mean: &[f32],
    bn_var: &[f32],
    eps: f32,
) -> Vec<f32> {
    assert!(h % 2 == 0 && w % 2 == 0, "unpack: h/w must be even ({h}x{w})");
    let (oh, ow) = (h / 2, w / 2);
    assert_eq!(z.len(), 4 * c * oh * ow, "unpack: z len != 4c*(h/2)*(w/2)");
    assert_eq!(bn_mean.len(), 4 * c, "unpack: bn_mean len != 4c");
    assert_eq!(bn_var.len(), 4 * c, "unpack: bn_var len != 4c");
    let mut out = vec![0.0f32; c * h * w];
    for ci in 0..c {
        for pi in 0..2 {
            for pj in 0..2 {
                let oc = ci * 4 + pi * 2 + pj;
                let s = (bn_var[oc] + eps).sqrt();
                let m = bn_mean[oc];
                for i in 0..oh {
                    for j in 0..ow {
                        out[(ci * h + 2 * i + pi) * w + 2 * j + pj] =
                            z[(oc * oh + i) * ow + j] * s + m;
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-computed 1×2×2 case pinning the channel-order convention: packed
    /// channel `= c·4 + pi·2 + pj` with `pi` the row, `pj` the col.
    #[test]
    fn pack_channel_order() {
        // mean[0] = [[1, 2], [3, 4]] → 2×2 patch (i=0, j=0).
        let mean = [1.0, 2.0, 3.0, 4.0];
        let bn_mean = [0.5, 0.0, -1.0, 2.0];
        let bn_var = [3.0, 8.0, 0.0, 15.0];
        let eps = 1.0; // √(var+eps) = [2, 3, 1, 4]
        let z = pack(&mean, 1, 2, 2, &bn_mean, &bn_var, eps);
        // oc0 (pi=0,pj=0) ← 1: (1−0.5)/2   oc1 (pi=0,pj=1) ← 2: (2−0)/3
        // oc2 (pi=1,pj=0) ← 3: (3+1)/1     oc3 (pi=1,pj=1) ← 4: (4−2)/4
        assert_eq!(z, vec![0.25, 2.0 / 3.0, 4.0, 0.5]);
        let back = unpack(&z, 1, 2, 2, &bn_mean, &bn_var, eps);
        assert_eq!(back, mean);
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let (c, h, w) = (3, 4, 6);
        // Deterministic pseudo-random values; no shared RNG dependency needed
        // for a layout test.
        let mean: Vec<f32> = (0..c * h * w).map(|i| ((i * 2654435761usize) % 1000) as f32 / 500.0 - 1.0).collect();
        let bn_mean: Vec<f32> = (0..4 * c).map(|i| i as f32 * 0.01 - 0.05).collect();
        let bn_var: Vec<f32> = (0..4 * c).map(|i| 0.5 + i as f32 * 0.1).collect();
        let eps = 1e-4;
        let z = pack(&mean, c, h, w, &bn_mean, &bn_var, eps);
        assert_eq!(z.len(), 4 * c * (h / 2) * (w / 2));
        let back = unpack(&z, c, h, w, &bn_mean, &bn_var, eps);
        for (a, b) in mean.iter().zip(&back) {
            assert!((a - b).abs() < 1e-6, "roundtrip {a} vs {b}");
        }
    }
}
