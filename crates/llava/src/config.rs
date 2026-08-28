// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LLaVA-1.5-13B configuration: a CLIP-L/14@336 vision tower, a two-layer
//! `mlp2x_gelu` projector, and Vicuna-1.5-13B's decoder - `ClipVisionConfig`
//! and `QwenConfig` presets composed, not restated.

use clip::config::ClipVisionConfig;
use qwen3::config::QwenConfig;

/// Which tokens of the vision tower's tapped hidden state feed the
/// projector. LLaVA-1.5 ships `mm_vision_select_feature = "patch"` - the
/// class token is dropped, keeping only the 576 patch tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectFeature {
    /// Drop the class token: `native_patches()` tokens (576 for CLIP-L336).
    Patch,
    /// Keep the class token too: `n_positions()` tokens (577).
    ClsPatch,
}

/// LLaVA-1.5's full configuration: the vision tower, the decoder, and the
/// two knobs upstream's `LlavaMetaModel`/`CLIPVisionTower` apply between them.
#[derive(Clone, Debug)]
pub struct LlavaConfig {
    pub vision: ClipVisionConfig,
    pub decoder: QwenConfig,
    /// `mm_vision_select_layer` - always `-2` (the penultimate CLIP block) for
    /// every released LLaVA-1.5 checkpoint. Kept as a signed field (matching
    /// upstream's own negative-indexed config key) rather than baked into
    /// [`Self::vision_tap_layer`], so a config parsed from a real
    /// `config.json` that ever set it to `-1` is representable, not silently
    /// coerced.
    pub select_layer: i32,
    pub select_feature: SelectFeature,
    /// LLaVA's `IMAGE_TOKEN_INDEX` sentinel spliced into the text stream -
    /// out of vocabulary range on purpose, so it can never collide with a
    /// real token id. See `crate::prompt`.
    pub image_token_index: i32,
}

impl LlavaConfig {
    /// `liuhaotian/llava-v1.5-13b`: CLIP-L/14@336 vision tower (24x1024, 16
    /// heads, MLP 4096, 577 positions, quick-GELU),
    /// `mm_vision_select_layer=-2`, `mm_vision_select_feature="patch"`,
    /// `mm_projector_type="mlp2x_gelu"`, Vicuna-1.5-13B decoder (LLaMA-2-13B:
    /// 40 layers, `d_model` 5120, plain MHA, `d_ff` 13824, untied head).
    pub fn llava_1_5_13b() -> LlavaConfig {
        LlavaConfig {
            vision: ClipVisionConfig::clip_l336(),
            decoder: QwenConfig::llama2_13b(),
            select_layer: -2,
            select_feature: SelectFeature::Patch,
            image_token_index: -200,
        }
    }

    /// The 0-based CLIP block [`Self::select_layer`] resolves to.
    /// `-2` -> `vision.penultimate_layer()` (`layers - 2`); `-1` -> the last
    /// block (`layers - 1`). No released checkpoint uses anything else, so
    /// any other value is a config this reader has not seen and panics by
    /// name rather than silently picking a nearby layer.
    pub fn vision_tap_layer(&self) -> u32 {
        match self.select_layer {
            -2 => self.vision.penultimate_layer(),
            -1 => self.vision.layers() - 1,
            other => panic!("llava: mm_vision_select_layer {other} is not implemented (only -1/-2 are)"),
        }
    }

    /// How many image tokens one image contributes to the decoder's sequence
    /// (576 for `Patch`, 577 for `ClsPatch` on CLIP-L336).
    pub fn n_visual_tokens(&self) -> u32 {
        match self.select_feature {
            SelectFeature::Patch => self.vision.native_patches(),
            SelectFeature::ClsPatch => self.vision.n_positions(),
        }
    }

    /// The projector's input width - the vision tower's hidden size (1024).
    pub fn projector_in(&self) -> u32 {
        self.vision.d_model()
    }

    /// The projector's output width - the decoder's hidden size (5120), which
    /// `mlp2x_gelu` also uses as its (single) hidden layer's width.
    pub fn projector_out(&self) -> u32 {
        self.decoder.d_model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llava_1_5_13b_composes_the_verified_presets() {
        let c = LlavaConfig::llava_1_5_13b();
        assert_eq!(c.vision, ClipVisionConfig::clip_l336());
        assert_eq!(c.decoder.d_model, 5120);
        assert_eq!(c.decoder.n_layers, 40);
        assert_eq!(c.select_layer, -2);
        assert_eq!(c.select_feature, SelectFeature::Patch);
        assert_eq!(c.image_token_index, -200);
    }

    #[test]
    fn vision_tap_layer_resolves_the_penultimate_block() {
        let c = LlavaConfig::llava_1_5_13b();
        assert_eq!(c.vision_tap_layer(), 22, "24 layers: -2 -> block 22");
        let mut last = c.clone();
        last.select_layer = -1;
        assert_eq!(last.vision_tap_layer(), 23);
    }

    #[test]
    #[should_panic(expected = "mm_vision_select_layer -3")]
    fn vision_tap_layer_rejects_an_unimplemented_selector() {
        let mut c = LlavaConfig::llava_1_5_13b();
        c.select_layer = -3;
        c.vision_tap_layer();
    }

    #[test]
    fn select_feature_controls_the_visual_token_count() {
        let mut c = LlavaConfig::llava_1_5_13b();
        assert_eq!(c.n_visual_tokens(), 576, "patch-only drops the class token");
        c.select_feature = SelectFeature::ClsPatch;
        assert_eq!(c.n_visual_tokens(), 577);
    }

    #[test]
    fn projector_dims_bridge_vision_and_decoder() {
        let c = LlavaConfig::llava_1_5_13b();
        assert_eq!(c.projector_in(), 1024);
        assert_eq!(c.projector_out(), 5120);
    }
}
