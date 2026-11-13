// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The fixed LPIPS v0.1 architectures. Nothing here is a knob: [`Trunk`] names the two released trunks (torchvision's
//! `alexnet-owt-7be5be79` and `vgg16-397923af`), each with the one set of v0.1 linear heads trained for it, and every
//! number below is read off the reference sources (`torchvision.models.alexnet` and `vgg16`,
//! `lpips/pretrained_networks.py`, `lpips/lpips.py`).

/// One trunk convolution: `features.{index}` in torchvision's `nn.Sequential`, always followed by a ReLU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrunkConv {
    /// Position in `features` - the checkpoint's own tensor prefix.
    pub index: u32,
    pub cin: u32,
    pub cout: u32,
    pub k: u32,
    pub stride: u32,
    pub pad: u32,
    /// A max-pool stands in front of this convolution.
    pub pooled: bool,
}

impl TrunkConv {
    /// The checkpoint's name of this convolution's weight.
    pub fn weight(&self) -> String {
        format!("features.{}.weight", self.index)
    }
    /// The checkpoint's name of this convolution's bias.
    pub fn bias(&self) -> String {
        format!("features.{}.bias", self.index)
    }
}

/// AlexNet `features[0..12]`: conv-ReLU, pool, conv-ReLU, pool, then three conv-ReLUs.
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

const fn vgg(index: u32, cin: u32, cout: u32, pooled: bool) -> TrunkConv {
    TrunkConv { index, cin, cout, k: 3, stride: 1, pad: 1, pooled }
}

/// VGG16 `features[0..30]`: five blocks of 3x3 convolutions (2, 2, 3, 3, 3), a `MaxPool2d(2, 2)` in front of every
/// block but the first. `lpips.pretrained_networks.vgg16` slices `features` at `[4, 9, 16, 23, 30]`, so each tap is
/// the last ReLU of a block (`relu1_2`, `relu2_2`, `relu3_3`, `relu4_3`, `relu5_3`), before the next pool.
pub const VGG16_TRUNK: [TrunkConv; 13] = [
    vgg(0, 3, 64, false),
    vgg(2, 64, 64, false),
    vgg(5, 64, 128, true),
    vgg(7, 128, 128, false),
    vgg(10, 128, 256, true),
    vgg(12, 256, 256, false),
    vgg(14, 256, 256, false),
    vgg(17, 256, 512, true),
    vgg(19, 512, 512, false),
    vgg(21, 512, 512, false),
    vgg(24, 512, 512, true),
    vgg(26, 512, 512, false),
    vgg(28, 512, 512, false),
];

/// Which released LPIPS v0.1 network: the trunk the features come from, and with it the heads, the pooling and the
/// taps. Both feed five taps to five `1x1` heads, so the metric's shape is the same and only the trunk differs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trunk {
    /// `net='alex'`, LPIPS's default and the one the metric is conventionally reported with.
    Alex,
    /// `net='vgg'`, the one LPIPS's perceptual losses train against (taming-transformers, LlamaGen, CodeFormer).
    Vgg16,
}

impl Trunk {
    /// The trunk's convolutions in order, each with the pool (if any) in front of it.
    pub fn convs(self) -> &'static [TrunkConv] {
        match self {
            Trunk::Alex => &TRUNK,
            Trunk::Vgg16 => &VGG16_TRUNK,
        }
    }

    /// The index in [`Self::convs`] of the convolution each of the five taps follows.
    pub fn tap_convs(self) -> [usize; 5] {
        match self {
            Trunk::Alex => [0, 1, 2, 3, 4],
            Trunk::Vgg16 => [1, 3, 6, 9, 12],
        }
    }

    /// The channel count of each tap, which is its head's input width.
    pub fn tap_channels(self) -> [u32; 5] {
        self.tap_convs().map(|i| self.convs()[i].cout)
    }

    /// `(kernel, stride)` of the `MaxPool2d` in front of a pooled convolution (no padding, floor).
    pub fn pool(self) -> (u32, u32) {
        match self {
            Trunk::Alex => (POOL_K, POOL_STRIDE),
            Trunk::Vgg16 => (2, 2),
        }
    }

    /// The smallest input side the trunk accepts: below it a tap or a pool has nothing left to read.
    pub fn min_side(self) -> u32 {
        match self {
            Trunk::Alex => MIN_SIDE,
            // Four stride-2 pools to a 1x1 last tap.
            Trunk::Vgg16 => 16,
        }
    }

    /// The name the metric and its checkpoint files go by (`lpips`' `net=`).
    pub fn name(self) -> &'static str {
        match self {
            Trunk::Alex => "alex",
            Trunk::Vgg16 => "vgg",
        }
    }

    /// The spatial size of every tap for an `h x w` input.
    pub fn tap_sizes(self, h: u32, w: u32) -> [(u32, u32); 5] {
        let (pk, ps) = self.pool();
        let conv = |s: u32, c: &TrunkConv| (s + 2 * c.pad - c.k) / c.stride + 1;
        let pool = |s: u32| (s - pk) / ps + 1;
        let taps = self.tap_convs();
        let mut out = [(0, 0); 5];
        let (mut ch, mut cw) = (h, w);
        for (i, c) in self.convs().iter().enumerate() {
            if c.pooled {
                (ch, cw) = (pool(ch), pool(cw));
            }
            (ch, cw) = (conv(ch, c), conv(cw, c));
            if let Some(t) = taps.iter().position(|&x| x == i) {
                out[t] = (ch, cw);
            }
        }
        out
    }
}

/// AlexNet's `MaxPool2d(kernel_size=3, stride=2)` between taps (no padding, floor).
pub const POOL_K: u32 = 3;
pub const POOL_STRIDE: u32 = 2;

/// `ScalingLayer`: the input, in `[-1, 1]`, becomes `(x - SHIFT) / SCALE`.
pub const SHIFT: [f32; 3] = [-0.030, -0.088, -0.188];
pub const SCALE: [f32; 3] = [0.458, 0.448, 0.450];

/// `normalize_tensor`'s epsilon, added to the channel norm:
/// `x / (sqrt(sum_c x^2) + EPS)`.
pub const EPS: f64 = 1e-10;

/// The v0.1 head for tap `i`: `NetLinLayer`'s `nn.Sequential(Dropout,
/// Conv2d(C, 1, 1, bias=False))`, so the convolution is child `1`.
pub fn head_weight(i: usize) -> String {
    format!("lin{i}.model.1.weight")
}

/// The smallest input side AlexNet accepts: below it the second pool has
/// nothing to pool (conv1 must leave at least 7 rows so pool1 leaves 3 for
/// pool2).
pub const MIN_SIDE: u32 = 31;

#[cfg(test)]
mod tests {
    use super::*;

    /// The standard 224x224 ImageNet input gives torchvision's documented
    /// 55 / 27 / 13 / 13 / 13 feature grids, and the channel chain closes.
    #[test]
    fn the_trunk_matches_torchvision_alexnet() {
        let tap_sizes = |h, w| Trunk::Alex.tap_sizes(h, w);
        assert_eq!(tap_sizes(224, 224), [(55, 55), (27, 27), (13, 13), (13, 13), (13, 13)]);
        for w in TRUNK.windows(2) {
            assert_eq!(w[0].cout, w[1].cin);
        }
        let s = tap_sizes(MIN_SIDE, MIN_SIDE);
        assert_eq!(s[4], (1, 1), "the smallest accepted side still leaves every tap non-empty");
    }

    /// VGG16 at 224x224 gives torchvision's 224 / 112 / 56 / 28 / 14 grids, the channel chain closes (a pool changes
    /// no channel count), every tap follows the last convolution of a block, and the smallest side leaves a 1x1 tap.
    #[test]
    fn the_vgg16_trunk_matches_torchvision() {
        let t = Trunk::Vgg16;
        assert_eq!(t.tap_sizes(224, 224), [(224, 224), (112, 112), (56, 56), (28, 28), (14, 14)]);
        assert_eq!(t.tap_channels(), [64, 128, 256, 512, 512]);
        for w in VGG16_TRUNK.windows(2) {
            assert_eq!(w[0].cout, w[1].cin);
        }
        assert_eq!(VGG16_TRUNK.iter().filter(|c| c.pooled).count(), 4);
        assert_eq!(t.tap_sizes(t.min_side(), t.min_side())[4], (1, 1));
        assert_eq!(Trunk::Alex.tap_channels(), [64, 192, 384, 256, 256]);
    }
}
