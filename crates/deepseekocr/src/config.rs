// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The composite's shape: three sub-configs plus the projector's own two
//! numbers, and the invariants that tie them together.
//!
//! Nothing is re-declared. `sam1::SamViTConfig`, `clip::config::ClipVisionConfig`
//! and `deepseekv2::DeepseekV2Config` each already wrap the corresponding
//! `crates/gguf` struct, so a checkpoint fact has exactly one home and this
//! struct only owns what sits BETWEEN them:
//!
//! * `projector_in  = compressor_out + clip_width` -- the concat's width.
//! * `projector_out = decoder d_model` -- the residual stream it splices into.
//!
//! ## The real-scale invariant, and the one fixture that breaks it
//!
//! At real scale the compressor's output width and CLIP's width are **both
//! 1024**, and the compressor output IS the CLIP patch token -- verbatim, with no
//! adapter. [`DeepseekOcrConfig::check_real_scale_shaped`] asserts exactly that.
//!
//! The checkpoint-free golden fixture deliberately breaks the equality (11 vs
//! 14), because it is that equality which makes a swapped concat order
//! arithmetically invisible at real scale. Breaking it leaves a width gap
//! between the compressor and CLIP, which the fixture bridges with a widening
//! `Linear(c_out, clip_width)` that has **no real-scale analogue**. That linear
//! is modelled here as [`DeepseekOcrConfig::patch_bypass`] -- off by default,
//! settable only by a fixture, and refused by `check_real_scale_shaped`.

use clip::config::{ClipVisionConfig, TextAct};
use deepseekv2::DeepseekV2Config;
use sam1::SamViTConfig;

/// Parameter name of the projector's weight (`mm.model.fc` in the mmproj).
pub const PROJECTOR_W: &str = "projector.weight";
/// Parameter name of the projector's bias.
pub const PROJECTOR_B: &str = "projector.bias";
/// Parameter name of the fixture-only widening bridge's weight.
pub const BYPASS_W: &str = "patch_bypass.weight";
/// Parameter name of the fixture-only widening bridge's bias.
pub const BYPASS_B: &str = "patch_bypass.bias";
/// The learned `[d_model]` row every `crate::rows::Src::Newline` row of the
/// image block carries.
///
/// Spelled exactly as `gguf::deepseek_ocr_vision`'s loader emits it, so the
/// mmproj import needs no rename for it (the projector's `vision.projector.fc.*`
/// does, because [`PROJECTOR_W`] predates that loader).
pub const IMAGE_NEWLINE: &str = "vision.image_newline";
/// The learned `[d_model]` row the single `crate::rows::Src::Separator` row
/// carries. Same naming rule as [`IMAGE_NEWLINE`].
pub const VIEW_SEPARATOR: &str = "vision.view_separator";

/// The whole DeepSeek-OCR pipeline's shape.
#[derive(Clone, Debug, PartialEq)]
pub struct DeepseekOcrConfig {
    /// SAM ViT-B tower **including** the conv neck and the two stride-2
    /// compressor convs -- `gguf::deepseek_ocr_vision::SamConfig` groups all of
    /// them under the SAM prefix, so `sam1` owns the whole front half.
    pub sam: SamViTConfig,
    /// CLIP-L/14, driven through `clip::model::PatchSource::Tokens`.
    pub clip: ClipVisionConfig,
    /// The DeepSeek-V2-family MHA decoder.
    pub decoder: DeepseekV2Config,
    /// **Fixture only.** Insert a widening `Linear(compressor_out, clip_width)`
    /// between the compressor output and CLIP's patch tokens. The real model has
    /// no such tensor; see this module's header.
    pub patch_bypass: bool,
}

impl DeepseekOcrConfig {
    /// The compressor's output token grid `(h, w)` -- CLIP's patch grid.
    pub fn token_grid(&self) -> (u32, u32) {
        self.sam.compress_grid()
    }

    /// Image tokens produced by one view (`token_grid().0 * token_grid().1`).
    pub fn image_tokens(&self) -> u32 {
        let (h, w) = self.token_grid();
        h * w
    }

    /// The compressor's output channel width.
    pub fn compressor_out(&self) -> u32 {
        self.sam.compress_out
    }

    pub fn clip_width(&self) -> u32 {
        self.clip.d_model()
    }

    /// The projector's input width -- the concat `[clip_spatial, compressor_flat]`.
    pub fn projector_in(&self) -> u32 {
        self.clip_width() + self.compressor_out()
    }

    /// The projector's output width -- the decoder's residual stream.
    pub fn projector_out(&self) -> u32 {
        self.decoder.d_model()
    }

    /// The glue stage's parameters (projector, the two learned image-block rows,
    /// plus the fixture bridge when present) and their element counts.
    ///
    /// [`IMAGE_NEWLINE`] and [`VIEW_SEPARATOR`] are `[d_model]` vectors the
    /// mmproj ships and [`crate::layout::RowGather`] consumes. They are declared
    /// unconditionally, because they are real tensors of the real model and the
    /// import must cover them; a config whose layout never places a newline row
    /// simply never reads them (the checkpoint-free golden fixture is exactly
    /// that case, and supplies them as zeros - see `tests/tiny_ref.rs`).
    pub fn glue_param_list(&self) -> Vec<(String, usize)> {
        let (pin, pout) = (self.projector_in() as usize, self.projector_out() as usize);
        let mut v = vec![
            (PROJECTOR_W.to_string(), pout * pin),
            (PROJECTOR_B.to_string(), pout),
            (IMAGE_NEWLINE.to_string(), pout),
            (VIEW_SEPARATOR.to_string(), pout),
        ];
        if self.patch_bypass {
            let (c, w) = (self.compressor_out() as usize, self.clip_width() as usize);
            v.push((BYPASS_W.to_string(), w * c));
            v.push((BYPASS_B.to_string(), w));
        }
        v
    }

    /// Every invariant that must hold for ANY config, fixture or real.
    ///
    /// Panics with the numbers in scope. Called by [`crate::DeepEncoder::new`],
    /// so a mis-shaped config fails at construction rather than as a wrong
    /// number many layers later.
    pub fn check(&self) {
        self.sam.check_bindable();
        // CLIP runs at the compressor's grid, not at its own native one; the
        // learned position table is resampled onto it (`clip::ClipVision`).
        let (gh, gw) = self.token_grid();
        assert!(gh > 0 && gw > 0, "compressor collapsed the grid to {gh}x{gw}");
        assert_eq!(
            self.clip.n_positions(),
            self.clip.native_patches() + 1,
            "CLIP's position table must be 1 + (image_size/patch)^2 rows"
        );
        // Without the bridge the compressor output IS the patch token, so the
        // two widths must already agree.
        assert!(
            self.patch_bypass || self.compressor_out() == self.clip_width(),
            "compressor_out {} != clip_width {} and no patch_bypass bridge is configured",
            self.compressor_out(),
            self.clip_width()
        );
        assert_eq!(
            self.projector_out(),
            self.decoder.d_model(),
            "the projector must land on the decoder's residual width"
        );
    }

    /// [`Self::check`] plus the two things only a **real-scale** config may
    /// claim: the compressor output is CLIP's patch token verbatim, and there is
    /// therefore no bridging linear.
    pub fn check_real_scale_shaped(&self) {
        self.check();
        assert!(!self.patch_bypass, "patch_bypass is fixture-only plumbing and has no real-scale analogue");
        assert_eq!(
            self.compressor_out(),
            self.clip_width(),
            "at real scale the compressor output IS the CLIP patch token (both 1024)"
        );
    }

    /// The real DeepSeek-OCR, at the SAM tower's native 1024² input.
    pub fn deepseek_ocr(block_size: u32) -> DeepseekOcrConfig {
        DeepseekOcrConfig {
            sam: SamViTConfig::deepseek_ocr(),
            clip: ClipVisionConfig::deepseek_ocr(),
            decoder: DeepseekV2Config::deepseek_ocr(block_size),
            patch_bypass: false,
        }
    }

    /// The checkpoint-free golden fixture's config.
    ///
    /// Every number is the corresponding `params.tiny` entry of
    /// `testdata/deepseek-ocr/manifest-tiny.json`, and
    /// `tests/tiny_ref.rs::tiny_reference_stage_parity` asserts each of them
    /// against the dump's own tensor SHAPES before it compares a single value --
    /// a fixture regenerated at other dims fails loudly there instead of
    /// silently comparing the wrong tensors.
    pub fn tiny() -> DeepseekOcrConfig {
        DeepseekOcrConfig {
            sam: SamViTConfig::tiny(),
            clip: ClipVisionConfig {
                shape: gguf::deepseek_ocr_vision::ClipConfig {
                    d_model: 14,
                    n_layers: 2,
                    n_heads: 2,
                    ffn_hidden: 20,
                    patch_size: 5,
                    image_size: 15, // native grid 3x3
                    n_positions: 10,
                    layer_norm_eps: 1e-5,
                },
                act: TextAct::QuickGelu,
            },
            decoder: DeepseekV2Config::tiny(),
            // c_out 11 != clip_width 14 -- the concat-order gate. See the header.
            patch_bypass: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real preset is what a production path may claim, and it satisfies the
    /// invariant the fixture deliberately breaks.
    #[test]
    fn the_real_preset_is_real_scale_shaped() {
        let c = DeepseekOcrConfig::deepseek_ocr(8192);
        c.check_real_scale_shaped();
        assert_eq!(c.token_grid(), (16, 16), "the compressor quarters a 64x64 grid");
        assert_eq!(c.image_tokens(), 256);
        assert_eq!((c.compressor_out(), c.clip_width()), (1024, 1024));
        assert_eq!(c.projector_in(), 2048);
        assert_eq!(c.projector_out(), 1280);
        // CLIP-L/14@224's own grid is exactly the compressor's, so the position
        // resample is the identity at real scale.
        assert_eq!(c.clip.native_grid(), 16);
    }

    /// The tiny fixture is legal (it carries the bridge) but must be REFUSED as
    /// real-scale shaped -- that refusal is what keeps the production path honest.
    #[test]
    fn the_tiny_fixture_is_legal_but_not_real_scale_shaped() {
        let c = DeepseekOcrConfig::tiny();
        c.check();
        assert_eq!(c.token_grid(), (4, 2), "distinct extents: an h/w swap must not cancel");
        assert_eq!(c.image_tokens(), 8);
        assert_ne!(c.compressor_out(), c.clip_width(), "the concat-order gate needs distinct widths");
        assert_eq!(c.projector_in(), 25);
        assert_eq!(c.projector_out(), 12);
        assert_ne!(c.clip.native_grid(), c.token_grid().0, "the position resample must be real");
        let e = std::panic::catch_unwind(|| c.check_real_scale_shaped()).unwrap_err();
        let msg = e.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| e.downcast_ref::<String>().cloned()).unwrap_or_default();
        assert!(msg.contains("patch_bypass"), "expected the bridge to be refused, got {msg:?}");
    }

    /// A config that drops the bridge WITHOUT equalising the widths is refused
    /// by the general check, not only by the real-scale one -- otherwise the
    /// encoder would silently feed 11-wide tokens to a 14-wide tower.
    #[test]
    fn dropping_the_bridge_at_unequal_widths_is_refused() {
        let c = DeepseekOcrConfig { patch_bypass: false, ..DeepseekOcrConfig::tiny() };
        let e = std::panic::catch_unwind(|| c.check()).unwrap_err();
        let msg = e.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(msg.contains("no patch_bypass bridge"), "got {msg:?}");
    }

    #[test]
    fn glue_manifest_covers_the_bridge_only_when_configured() {
        let names = |c: &DeepseekOcrConfig| -> Vec<String> { c.glue_param_list().into_iter().map(|(n, _)| n).collect() };
        assert_eq!(names(&DeepseekOcrConfig::deepseek_ocr(4096)), vec![PROJECTOR_W, PROJECTOR_B, IMAGE_NEWLINE, VIEW_SEPARATOR]);
        assert_eq!(names(&DeepseekOcrConfig::tiny()), vec![PROJECTOR_W, PROJECTOR_B, IMAGE_NEWLINE, VIEW_SEPARATOR, BYPASS_W, BYPASS_B]);
        let tiny = DeepseekOcrConfig::tiny();
        let sizes: Vec<usize> = tiny.glue_param_list().into_iter().map(|(_, n)| n).collect();
        assert_eq!(sizes, vec![12 * 25, 12, 12, 12, 14 * 11, 14]);
    }

    /// The two learned image-block rows are declared at the DECODER's width and
    /// under the names the mmproj loader itself emits -- both halves matter: a
    /// row of the wrong width would be spliced into the residual stream anyway,
    /// and a renamed one would be silently absent from the import.
    #[test]
    fn the_two_learned_rows_are_declared_at_the_decoders_width_under_the_gguf_names() {
        let c = DeepseekOcrConfig::deepseek_ocr(4096);
        let glue = c.glue_param_list();
        let get = |n: &str| glue.iter().find(|(k, _)| k == n).map(|(_, v)| *v);
        assert_eq!(get(IMAGE_NEWLINE), Some(c.projector_out() as usize));
        assert_eq!(get(VIEW_SEPARATOR), Some(c.projector_out() as usize));
        assert_eq!(c.projector_out(), 1280);
        // Byte-identical to what `gguf::deepseek_ocr_vision::param_list()` emits
        // for these two entries, so the mmproj import needs no rename for them.
        // (Spelled out rather than derived: building a whole
        // `DeepseekOcrVisionConfig` here to read two strings off it would test
        // the constructor, not the agreement.)
        assert_eq!((IMAGE_NEWLINE, VIEW_SEPARATOR), ("vision.image_newline", "vision.view_separator"));
    }
}
