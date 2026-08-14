// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The reference configuration, exactly as the insightface `antelopev2` release
//! defines it. Every number here was read off the released ONNX graph or the
//! insightface source vendored into `tools/goldens/arcface_dump_reference.py`;
//! none is a guess, and the parity goldens' `manifest.json` records the same
//! values.

/// Preprocessing constants of a `cv2.dnn.blobFromImage` call: `(bgr_u8 → RGB −
/// mean) / std`, NCHW.
///
/// **This is the EMBEDDER's pair and nothing else's**, and that is the trap:
/// ArcFace divides by 127.5 while SCRFD divides by 128.0. They look like the
/// same "normalise to [-1,1]" and are not - using the detector's std here shifts
/// every activation by 0.4 %, which is exactly the kind of error that produces
/// plausible embeddings. The two crates therefore each own their constants;
/// neither imports the other's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Preprocess {
    pub mean: f32,
    pub std: f32,
    /// Whether the source is BGR and must be swapped to RGB (`swapRB=True`).
    pub swap_rb: bool,
}

/// IResNet-100 (`glintr100.onnx`) - the ArcFace embedding backbone.
///
/// The released graph has its BatchNorms **folded into the convolutions**: every
/// conv carries a bias and a PReLU sits directly after it. Only three BatchNorms
/// survive as real nodes - each block's entry `bn1`, the final `bn2`, and the
/// `features` BN after `fc`. So a port that models the torch `iresnet` module
/// literally (conv → bn → prelu) would need weights that are not in the file.
///
/// There is **no L2-normalisation in the graph**: the 512-d output is raw
/// (‖e‖ ≈ 15–20). Consumers normalise for cosine.
#[derive(Clone, Debug, PartialEq)]
pub struct ArcFaceConfig {
    /// Square input side (112).
    pub image_size: u32,
    /// Residual blocks per stage. IResNet-100 = `[3, 13, 30, 3]` = 49 blocks,
    /// which the importer asserts against the graph's 49 `Add` nodes.
    pub layers: [u32; 4],
    /// Output channels per stage: `[64, 128, 256, 512]`.
    pub channels: [u32; 4],
    /// Stem output channels (64) - also stage 0's input width.
    pub stem_channels: u32,
    /// Embedding width (512).
    pub embedding: u32,
    pub pre: Preprocess,
}

impl Default for ArcFaceConfig {
    fn default() -> Self {
        ArcFaceConfig::iresnet100()
    }
}

impl ArcFaceConfig {
    /// `glintr100.onnx` as released in antelopev2.
    pub fn iresnet100() -> ArcFaceConfig {
        ArcFaceConfig {
            image_size: 112,
            layers: [3, 13, 30, 3],
            channels: [64, 128, 256, 512],
            stem_channels: 64,
            embedding: 512,
            pre: Preprocess { mean: 127.5, std: 127.5, swap_rb: true },
        }
    }

    /// Total residual blocks (49 for IResNet-100).
    pub fn n_blocks(&self) -> u32 {
        self.layers.iter().sum()
    }

    /// Spatial side at the input of stage `s` (112, 56, 28, 14) - every stage's
    /// first block strides by 2, including stage 0.
    pub fn stage_in_hw(&self, s: usize) -> u32 {
        self.image_size >> s
    }

    /// Channels entering stage `s`: the stem for stage 0, else the previous
    /// stage's output.
    pub fn stage_in_c(&self, s: usize) -> u32 {
        if s == 0 {
            self.stem_channels
        } else {
            self.channels[s - 1]
        }
    }

    /// The flattened `bn2` width feeding `fc` (512 · 7 · 7 = 25088).
    pub fn flatten(&self) -> u32 {
        let hw = self.image_size >> 4;
        self.channels[3] * hw * hw
    }
}

/// The canonical ArcFace 5-point destination template at 112×112
/// (`insightface.utils.face_align.arcface_dst`), as `(x, y)` pairs in
/// left-eye / right-eye / nose / left-mouth / right-mouth order.
///
/// These are the exact float32 values in the reference; they are the target of
/// the similarity fit, so a rounded copy moves every aligned pixel.
pub const ARCFACE_DST_112: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iresnet100_geometry_matches_the_dumped_golden_shapes() {
        let c = ArcFaceConfig::iresnet100();
        assert_eq!(c.n_blocks(), 49, "49 residual Add nodes in glintr100.onnx");
        // stage input sides: the golden taps are 112, 56, 28, 14 and outputs
        // 56, 28, 14, 7.
        assert_eq!(
            (0..4).map(|s| c.stage_in_hw(s)).collect::<Vec<_>>(),
            vec![112, 56, 28, 14]
        );
        assert_eq!((0..4).map(|s| c.stage_in_c(s)).collect::<Vec<_>>(), vec![64, 64, 128, 256]);
        assert_eq!(c.flatten(), 25088);
    }

    /// The embedder's `std` is 127.5. The detector's is 128.0 and lives in the
    /// other crate; a pass that "unifies" them is a silent regression.
    #[test]
    fn the_embedder_normalisation_is_the_reference_one() {
        let p = ArcFaceConfig::iresnet100().pre;
        assert_eq!((p.mean, p.std, p.swap_rb), (127.5, 127.5, true));
    }
}
