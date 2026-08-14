// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The reference configuration, exactly as the insightface `antelopev2` release
//! defines it. Every number here was read off the released ONNX graph or the
//! insightface source vendored into `tools/goldens/scrfd_dump_reference.py`;
//! none is a guess, and the parity goldens' `manifest.json` records the same
//! values.

/// Preprocessing constants of a `cv2.dnn.blobFromImage` call: `(bgr_u8 → RGB −
/// mean) / std`, NCHW.
///
/// **This is the DETECTOR's pair and nothing else's**, and that is the trap:
/// SCRFD divides by 128.0 while ArcFace divides by 127.5. They look like the
/// same "normalise to [-1,1]" and are not - using the embedder's std here shifts
/// every activation by 0.4 %, which is exactly the kind of error that produces
/// plausible boxes. The two crates therefore each own their constants; neither
/// imports the other's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Preprocess {
    pub mean: f32,
    pub std: f32,
    /// Whether the source is BGR and must be swapped to RGB (`swapRB=True`).
    pub swap_rb: bool,
}

/// SCRFD-10GF with keypoints (`scrfd_10g_bnkps.onnx`).
///
/// BN-folded (every conv is biased, ReLU follows directly). The residual
/// backbone is a plain ResNet: `conv → relu → conv → (+shortcut) → relu`, with
/// the strided blocks' shortcut being `AveragePool(2, 2) → conv1x1` rather than a
/// strided 1x1 - the "ResNet-D" downsample, and getting it wrong silently
/// shifts every feature by half a pixel.
#[derive(Clone, Debug, PartialEq)]
pub struct ScrfdConfig {
    /// Square detector input (640).
    pub image_size: u32,
    /// Stem conv widths: `[28, 28, 56]`, then a 2×2/2 max-pool.
    pub stem_channels: [u32; 3],
    /// Blocks per backbone stage: `[3, 4, 2, 3]`.
    pub layers: [u32; 4],
    /// Output channels per backbone stage: `[56, 88, 88, 224]`.
    pub channels: [u32; 4],
    /// Whether stage `i`'s first block strides (stage 0 does not - the stem's
    /// max-pool already halved it).
    pub stage_stride2: [bool; 4],
    /// FPN/PAFPN width (56).
    pub neck_channels: u32,
    /// Head stem width (80) and its depth (3 convs).
    pub head_channels: u32,
    pub head_depth: u32,
    /// Output strides, coarse-to-fine order as the graph emits them.
    pub strides: [u32; 3],
    /// Anchors per location (2).
    pub num_anchors: u32,
    pub det_thresh: f32,
    pub nms_thresh: f32,
    pub pre: Preprocess,
}

impl Default for ScrfdConfig {
    fn default() -> Self {
        ScrfdConfig::scrfd_10g_bnkps()
    }
}

impl ScrfdConfig {
    pub fn scrfd_10g_bnkps() -> ScrfdConfig {
        ScrfdConfig {
            image_size: 640,
            stem_channels: [28, 28, 56],
            layers: [3, 4, 2, 3],
            channels: [56, 88, 88, 224],
            stage_stride2: [false, true, true, true],
            neck_channels: 56,
            head_channels: 80,
            head_depth: 3,
            strides: [8, 16, 32],
            num_anchors: 2,
            det_thresh: 0.5,
            nms_thresh: 0.4,
            pre: Preprocess { mean: 127.5, std: 128.0, swap_rb: true },
        }
    }

    /// Input channels of backbone stage `s` (stem output for stage 0).
    pub fn stage_in_c(&self, s: usize) -> u32 {
        if s == 0 {
            self.stem_channels[2]
        } else {
            self.channels[s - 1]
        }
    }

    /// Feature-map side at the *output* of backbone stage `s`.
    /// Stem = /2 (conv) then /2 (max-pool) = /4; stages 1..3 halve again.
    pub fn stage_out_hw(&self, s: usize) -> u32 {
        let mut hw = self.image_size / 4;
        for i in 1..=s {
            if self.stage_stride2[i] {
                hw /= 2;
            }
        }
        hw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scrfd_feature_sides_match_the_dumped_golden_shapes() {
        let c = ScrfdConfig::scrfd_10g_bnkps();
        // goldens: c2 160, c3 80, c4 40, c5 20 at a 640 input.
        assert_eq!((0..4).map(|s| c.stage_out_hw(s)).collect::<Vec<_>>(), vec![160, 80, 40, 20]);
        assert_eq!((0..4).map(|s| c.stage_in_c(s)).collect::<Vec<_>>(), vec![56, 56, 88, 88]);
        // one head row per (anchor, location)
        let rows: Vec<u32> =
            c.strides.iter().map(|&s| (c.image_size / s).pow(2) * c.num_anchors).collect();
        assert_eq!(rows, vec![12800, 3200, 800]);
    }

    /// The detector's `std` is 128.0. The embedder's is 127.5 and lives in the
    /// other crate; a pass that "unifies" them is a silent regression.
    #[test]
    fn the_detector_normalisation_is_the_reference_one() {
        let p = ScrfdConfig::scrfd_10g_bnkps().pre;
        assert_eq!((p.mean, p.std, p.swap_rb), (127.5, 128.0, true));
    }
}
