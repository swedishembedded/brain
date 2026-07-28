// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Moondream 3 (preview) configuration — values from the released `config.py`.
//! The 3.1 release ships no modeling code; the preview is the architecture
//! reference (identical hyperparameters assumed).

/// SigLIP-style ViT vision encoder + overlap multi-crop.
#[derive(Clone, Debug, PartialEq)]
pub struct VisionConfig {
    pub dim: u32,            // enc_dim 1152
    pub patch: u32,          // enc_patch_size 14
    pub n_layers: u32,       // enc_n_layers 27
    pub ff_dim: u32,         // enc_ff_dim 4304
    pub n_heads: u32,        // enc_n_heads 16 (head_dim 72)
    pub crop_size: u32,      // 378 → 27×27 = 729 patches
    pub max_crops: u32,      // 12
    pub overlap_margin: u32, // 4 patches
}

impl VisionConfig {
    pub fn head_dim(&self) -> u32 {
        self.dim / self.n_heads
    }
    /// Patch grid side (`crop_size / patch`). 27.
    pub fn grid(&self) -> u32 {
        self.crop_size / self.patch
    }
    /// Patches per crop (`grid²`). 729.
    pub fn patches_per_crop(&self) -> u32 {
        self.grid() * self.grid()
    }
    /// Flattened per-patch vector (`3 · patch²`). 588.
    pub fn patch_vec(&self) -> u32 {
        3 * self.patch * self.patch
    }
}

/// Sparse-MoE FFN (top-k GeGLU-shift experts) for the deep decoder layers.
#[derive(Clone, Debug, PartialEq)]
pub struct MoeConfig {
    pub num_experts: u32, // 64
    pub start_layer: u32, // 4 (layers 0..3 dense, 4..23 MoE)
    pub top_k: u32,       // 8
    pub inner_dim: u32,   // expert GeGLU inner 1024
}

/// Full Moondream 3 configuration.
#[derive(Clone, Debug)]
pub struct MoondreamConfig {
    pub dim: u32,         // text dim 2048
    pub ff_dim: u32,      // dense FFN 8192
    pub n_layers: u32,    // 24
    pub vocab: u32,       // 51200
    pub n_heads: u32,     // 32 (full MHA, no GQA)
    pub head_dim: u32,    // 64
    pub prefix_attn: u32, // 730 = 1 (bos) + 729 image tokens (bidirectional)
    pub rot_dim: u32,     // partial-RoPE rotated channels 32
    pub rope_theta: f32,  // 1.5e6
    pub proj_inner: u32,  // connector hidden 8192
    pub proj_out: u32,    // connector out (= text dim) 2048
    pub vision: VisionConfig,
    pub moe: MoeConfig,
}

impl MoondreamConfig {
    /// The Moondream 3 preview configuration.
    pub fn preview() -> MoondreamConfig {
        MoondreamConfig {
            dim: 2048,
            ff_dim: 8192,
            n_layers: 24,
            vocab: 51200,
            n_heads: 32,
            head_dim: 64,
            prefix_attn: 730,
            rot_dim: 32,
            rope_theta: 1_500_000.0,
            proj_inner: 8192,
            proj_out: 2048,
            vision: VisionConfig {
                dim: 1152,
                patch: 14,
                n_layers: 27,
                ff_dim: 4304,
                n_heads: 16,
                crop_size: 378,
                max_crops: 12,
                overlap_margin: 4,
            },
            moe: MoeConfig { num_experts: 64, start_layer: 4, top_k: 8, inner_dim: 1024 },
        }
    }

    /// True if decoder layer `l` uses the MoE FFN (else a dense FFN).
    pub fn is_moe_layer(&self, l: u32) -> bool {
        l >= self.moe.start_layer
    }

    /// Connector input width: global‖local channel-concat of the ViT features
    /// (`2 · vision.dim`). 2304.
    pub fn connector_in(&self) -> u32 {
        2 * self.vision.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_dims() {
        let c = MoondreamConfig::preview();
        assert_eq!(c.head_dim, 64);
        assert_eq!(c.n_heads * c.head_dim, c.dim); // full MHA, 32×64 = 2048
        assert_eq!(c.connector_in(), 2304); // 2 × 1152
        assert_eq!(c.prefix_attn, 1 + c.vision.patches_per_crop()); // 730
    }

    #[test]
    fn vision_dims() {
        let v = MoondreamConfig::preview().vision;
        assert_eq!(v.head_dim(), 72); // 1152 / 16
        assert_eq!(v.grid(), 27); // 378 / 14
        assert_eq!(v.patches_per_crop(), 729);
        assert_eq!(v.patch_vec(), 588); // 3 · 14²
    }

    #[test]
    fn moe_layer_split() {
        let c = MoondreamConfig::preview();
        assert!(!c.is_moe_layer(0) && !c.is_moe_layer(3));
        assert!(c.is_moe_layer(4) && c.is_moe_layer(23));
        assert_eq!(c.moe.num_experts, 64);
        assert_eq!(c.moe.top_k, 8);
    }
}
