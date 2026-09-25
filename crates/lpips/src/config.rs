// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The fixed LPIPS v0.1 / AlexNet architecture. Nothing here is a knob: there
//! is one released AlexNet trunk (torchvision's `alexnet-owt-7be5be79`) and
//! one set of v0.1 linear heads for it, and every number below is read off
//! the reference sources (`torchvision.models.alexnet`,
//! `lpips/pretrained_networks.py`, `lpips/lpips.py`).

/// One of AlexNet's five convolutions: `features.{index}` in torchvision's
/// `nn.Sequential`, always followed by a ReLU, which is where LPIPS taps it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrunkConv {
    /// Position in `alexnet().features` - the checkpoint's own tensor prefix.
    pub index: u32,
    pub cin: u32,
    pub cout: u32,
    pub k: u32,
    pub stride: u32,
    pub pad: u32,
    /// A `MaxPool2d(3, 2)` stands in front of this convolution.
    pub pooled: bool,
}

/// `features[0..12]`: conv-ReLU, pool, conv-ReLU, pool, then three conv-ReLUs.
/// The two pools sit between the taps (`slice2` and `slice3` of
/// `lpips.pretrained_networks.alexnet` open with them), so each tap is a
/// post-ReLU, pre-pool map.
pub const TRUNK: [TrunkConv; 5] = [
    TrunkConv { index: 0, cin: 3, cout: 64, k: 11, stride: 4, pad: 2, pooled: false },
    TrunkConv { index: 3, cin: 64, cout: 192, k: 5, stride: 1, pad: 2, pooled: true },
    TrunkConv { index: 6, cin: 192, cout: 384, k: 3, stride: 1, pad: 1, pooled: true },
    TrunkConv { index: 8, cin: 384, cout: 256, k: 3, stride: 1, pad: 1, pooled: false },
    TrunkConv { index: 10, cin: 256, cout: 256, k: 3, stride: 1, pad: 1, pooled: false },
];

/// The `MaxPool2d(kernel_size=3, stride=2)` between taps (no padding, floor).
pub const POOL_K: u32 = 3;
pub const POOL_STRIDE: u32 = 2;

/// `ScalingLayer`: the input, in `[-1, 1]`, becomes `(x - SHIFT) / SCALE`.
pub const SHIFT: [f32; 3] = [-0.030, -0.088, -0.188];
pub const SCALE: [f32; 3] = [0.458, 0.448, 0.450];

/// `normalize_tensor`'s epsilon, added to the channel norm:
/// `x / (sqrt(sum_c x^2) + EPS)`.
pub const EPS: f64 = 1e-10;

/// The trunk's checkpoint names for convolution `i`.
pub fn trunk_weight(i: usize) -> String {
    format!("features.{}.weight", TRUNK[i].index)
}
pub fn trunk_bias(i: usize) -> String {
    format!("features.{}.bias", TRUNK[i].index)
}
/// The v0.1 head for tap `i`: `NetLinLayer`'s `nn.Sequential(Dropout,
/// Conv2d(C, 1, 1, bias=False))`, so the convolution is child `1`.
pub fn head_weight(i: usize) -> String {
    format!("lin{i}.model.1.weight")
}

/// The smallest input side the trunk accepts: below it the second pool has
/// nothing to pool (conv1 must leave at least 7 rows so pool1 leaves 3 for
/// pool2).
pub const MIN_SIDE: u32 = 31;

/// The spatial size of every tap for an `h x w` input.
pub fn tap_sizes(h: u32, w: u32) -> [(u32, u32); 5] {
    let conv = |s: u32, c: &TrunkConv| (s + 2 * c.pad - c.k) / c.stride + 1;
    let pool = |s: u32| (s - POOL_K) / POOL_STRIDE + 1;
    let mut out = [(0, 0); 5];
    let (mut ch, mut cw) = (h, w);
    for (i, c) in TRUNK.iter().enumerate() {
        if c.pooled {
            (ch, cw) = (pool(ch), pool(cw));
        }
        (ch, cw) = (conv(ch, c), conv(cw, c));
        out[i] = (ch, cw);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The standard 224x224 ImageNet input gives torchvision's documented
    /// 55 / 27 / 13 / 13 / 13 feature grids, and the channel chain closes.
    #[test]
    fn the_trunk_matches_torchvision_alexnet() {
        assert_eq!(tap_sizes(224, 224), [(55, 55), (27, 27), (13, 13), (13, 13), (13, 13)]);
        for w in TRUNK.windows(2) {
            assert_eq!(w[0].cout, w[1].cin);
        }
        let s = tap_sizes(MIN_SIDE, MIN_SIDE);
        assert_eq!(s[4], (1, 1), "the smallest accepted side still leaves every tap non-empty");
    }
}
