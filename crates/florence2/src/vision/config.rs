// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DaViT (Dual-Attention Vision Transformer) stage/block schedule - the
//! per-stage dimensions, feature-map sizes and block counts every later
//! DaViT module (patch embed, window attention, channel attention) is
//! parameterized by. Modeled on `sam2::config`'s per-stage block-table
//! shape: derive the whole schedule once from the stage-level arrays a
//! checkpoint's `config.json` ships, rather than each block computing its
//! own dimensions ad hoc.
//!
//! ## Block-level structure this schedule feeds (verified against the real
//! `microsoft/Florence-2-base` reference implementation, not the paper -
//! several details below aren't documented anywhere else)
//!
//! Each DaViT stage is: one patch-embed downsample, then `depth` PAIRS of
//! (spatial block, channel block) run in sequence - `depths=[1,1,9,1]`
//! means stage 2 alone runs 9 spatial+channel pairs (18 blocks), not 9
//! blocks total.
//!
//! Every spatial AND channel block carries **two extra residual depthwise
//! 3x3 convolutions** the block schedule alone doesn't show: one immediately
//! before its attention (`x = x + dwconv3x3(x)`, no norm) and one
//! immediately before its MLP - the reference's `conv_at_attn`/`conv_at_ffn`
//! flags, both `true` for every Florence-2 stage. This is a real per-block
//! addition, distinct from the once-per-stage conditional positional
//! encoding pattern `fastvlm::encoder::repcpe` uses - a DaViT block needs
//! its own depthwise-conv residual pair, not a single stage-level one.
//!
//! Channel attention's softmax scale is `1/sqrt(N)` where `N` is the
//! spatial TOKEN COUNT (the contraction length once q/k are transposed so
//! attention runs over channel-groups instead of tokens) - **not**
//! `1/sqrt(head_dim)` the way every other attention variant in this repo
//! scales. Getting this scale from the wrong dimension is silent: shapes
//! still agree, only the numbers are wrong.
//!
//! Window attention is the standard variant (`1/sqrt(head_dim)`), padded to
//! a `window_size` multiple - `crate::model::vit::WindowPlan::padded`'s
//! exact shape (zero-pad-to-multiple with a sentinel row), not `::new`'s
//! unpadded assumption.

/// One stage's overlapping-patch-embed downsample spec.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DavitPatchSpec {
    pub kernel: u32,
    pub stride: u32,
    pub padding: u32,
    /// `true`: LayerNorm(dim_in) before the conv (stages 1..3). `false`:
    /// LayerNorm(dim_out) after the conv (stage 0, over raw pixels - the
    /// reference skips the pre-norm branch entirely for a 4D pixel input,
    /// which is exactly what `pre_norm=false` selects here).
    pub pre_norm: bool,
}

/// One DaViT stage's full shape: input/output channel width, attention
/// head/group counts, block-pair depth, patch-embed spec, and the
/// feature-map size on each side of the downsample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DavitStageSpec {
    pub dim_in: u32,
    pub dim_out: u32,
    /// Spatial (window) attention head count.
    pub num_heads: u32,
    /// Channel attention group count - a separate config array from
    /// `num_heads` in the reference, though Florence-2-base's checkpoint
    /// happens to set them equal at every stage.
    pub num_groups: u32,
    /// Number of (spatial, channel) block PAIRS - 2x this many blocks run.
    pub depth: u32,
    pub patch: DavitPatchSpec,
    pub in_hw: (u32, u32),
    pub out_hw: (u32, u32),
}

impl DavitStageSpec {
    /// Token count entering this stage's blocks (after the patch embed).
    pub fn tokens(&self) -> u32 {
        self.out_hw.0 * self.out_hw.1
    }
}

/// The full DaViT schedule: window size (shared across every stage) and
/// the per-stage table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DavitConfig {
    pub window_size: u32,
    pub stages: Vec<DavitStageSpec>,
}

/// `floor((x + 2*pad - kernel) / stride) + 1` - standard conv output size.
fn conv_out(x: u32, kernel: u32, stride: u32, pad: u32) -> u32 {
    (x + 2 * pad - kernel) / stride + 1
}

impl DavitConfig {
    /// Build the schedule from a checkpoint's `vision_config` arrays
    /// (`dim_embed`, `num_heads`, `num_groups`, `depths`, `patch_size`,
    /// `patch_stride`, `patch_padding`, `patch_prenorm`), all the same
    /// length (one entry per stage), plus the input image side length
    /// (Florence-2-base: 768, square images only - the reference asserts
    /// `h == w` after the vision tower's unpooled forward).
    #[allow(clippy::too_many_arguments)]
    pub fn from_arrays(
        image_size: u32,
        dim_embed: &[u32],
        num_heads: &[u32],
        num_groups: &[u32],
        depths: &[u32],
        patch_size: &[u32],
        patch_stride: &[u32],
        patch_padding: &[u32],
        patch_prenorm: &[bool],
        window_size: u32,
    ) -> DavitConfig {
        let n = dim_embed.len();
        assert!(
            [num_heads.len(), num_groups.len(), depths.len(), patch_size.len(), patch_stride.len(), patch_padding.len(), patch_prenorm.len()]
                .iter()
                .all(|&l| l == n),
            "DaViT stage arrays must all have the same length"
        );

        let mut stages = Vec::with_capacity(n);
        let mut hw = (image_size, image_size);
        let mut dim_in = 3u32; // RGB input to stage 0
        for i in 0..n {
            let patch = DavitPatchSpec { kernel: patch_size[i], stride: patch_stride[i], padding: patch_padding[i], pre_norm: patch_prenorm[i] };
            let out_hw = (conv_out(hw.0, patch.kernel, patch.stride, patch.padding), conv_out(hw.1, patch.kernel, patch.stride, patch.padding));
            stages.push(DavitStageSpec {
                dim_in,
                dim_out: dim_embed[i],
                num_heads: num_heads[i],
                num_groups: num_groups[i],
                depth: depths[i],
                patch,
                in_hw: hw,
                out_hw,
            });
            hw = out_hw;
            dim_in = dim_embed[i];
        }
        DavitConfig { window_size, stages }
    }

    /// `microsoft/Florence-2-base`'s real `vision_config`, verified against
    /// the checkpoint's own `config.json` (not the DaViT paper's defaults,
    /// which differ - e.g. the paper's example config uses 3 stages of
    /// depth [1,1,3,1] scaled differently).
    pub fn florence2_base() -> DavitConfig {
        DavitConfig::from_arrays(
            768,
            &[128, 256, 512, 1024],
            &[4, 8, 16, 32],
            &[4, 8, 16, 32],
            &[1, 1, 9, 1],
            &[7, 3, 3, 3],
            &[4, 2, 2, 2],
            &[3, 1, 1, 1],
            &[false, true, true, true],
            12,
        )
    }

    /// Final stage's output token count and channel width - the
    /// `forward_features_unpool` output shape a caller projects/pools from.
    pub fn final_tokens_and_dim(&self) -> (u32, u32) {
        let last = self.stages.last().expect("DaViT config must have at least one stage");
        (last.tokens(), last.dim_out)
    }
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Cross-checked against the real forward pass: `_encode_image` produces
    /// `image_seq_length=577 = 1 (spatial_avg_pool) + 576 (24x24 spatial)`,
    /// exactly what `preprocessor_config.json` declares - so stage 3 must
    /// resolve to a 24x24 grid at dim 1024.
    #[test]
    fn florence2_base_stage_shapes_match_the_real_577_token_output() {
        let cfg = DavitConfig::florence2_base();
        assert_eq!(cfg.stages.len(), 4);

        let expect = [
            // (dim_in, dim_out, out_hw)
            (3u32, 128u32, (192u32, 192u32)),
            (128, 256, (96, 96)),
            (256, 512, (48, 48)),
            (512, 1024, (24, 24)),
        ];
        for (stage, (dim_in, dim_out, out_hw)) in cfg.stages.iter().zip(expect) {
            assert_eq!(stage.dim_in, dim_in);
            assert_eq!(stage.dim_out, dim_out);
            assert_eq!(stage.out_hw, out_hw);
        }

        let (tokens, dim) = cfg.final_tokens_and_dim();
        assert_eq!(tokens, 576, "24x24 grid feeding the +1 pooled token to reach image_seq_length=577");
        assert_eq!(dim, 1024);
    }

    #[test]
    fn florence2_base_block_pair_depths_and_window_size() {
        let cfg = DavitConfig::florence2_base();
        let depths: Vec<u32> = cfg.stages.iter().map(|s| s.depth).collect();
        assert_eq!(depths, vec![1, 1, 9, 1]);
        assert_eq!(cfg.window_size, 12);
        // Stage 2 alone contributes 9 (spatial, channel) pairs = 18 blocks.
        assert_eq!(cfg.stages[2].depth * 2, 18);
    }

    #[test]
    fn stage0_patch_embed_is_post_norm_over_raw_pixels() {
        let cfg = DavitConfig::florence2_base();
        assert!(!cfg.stages[0].patch.pre_norm, "stage 0 takes raw pixels, norm runs after the conv");
        assert!(cfg.stages[1..].iter().all(|s| s.patch.pre_norm), "stages 1..3 norm their input sequence before the conv");
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn mismatched_array_lengths_panics() {
        DavitConfig::from_arrays(768, &[128, 256], &[4], &[4, 8], &[1, 1], &[7, 3], &[4, 2], &[3, 1], &[false, true], 12);
    }
}
