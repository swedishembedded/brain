// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! BiSeNet(`num_class=19`) - the ONE published shape (facexlib's
//! `parsing_bisenet.pth`). Channel widths and block counts are fixed to the
//! reference architecture (`ResNet18` backbone: 2 `BasicBlock`s per stage,
//! channels `[64,128,256,512]`), not derived from a checkpoint - there is
//! only one released variant, so nothing here is a config KNOB the way
//! `YoloConfig`/`Flux1Config` are for their families.

#[derive(Clone, Copy, Debug)]
pub struct BiSeNetConfig {
    pub num_class: u32,
    /// The square input side every released checkpoint and every caller in
    /// this workspace uses (`crate::align`'s FFHQ-512 template).
    pub input_size: u32,
}

/// `ResNet18`'s 4 stages: `(out_channels, num_blocks)`, first block of stages
/// 2-4 downsamples (stride 2); stage 1 does not (it follows the stem's own
/// stride-4 stem+maxpool).
pub const RESNET18_STAGES: [(u32, u32); 4] = [(64, 2), (128, 2), (256, 2), (512, 2)];

/// `ContextPath`'s two `AttentionRefinementModule`s: `(in_channels,
/// out_channels)`, applied to stage-3 (`arm16`) and stage-4 (`arm32`) output.
pub const ARM16: (u32, u32) = (256, 128);
pub const ARM32: (u32, u32) = (512, 128);
/// `FeatureFusionModule(in_chan=256, out_chan=256)` - `feat_res8` (128ch,
/// ContextPath's un-refined stage-2/"1/8" output) concatenated with
/// `feat_cp8` (128ch, the refined context-path output at the same
/// resolution).
pub const FFM: (u32, u32) = (256, 256);
/// `BiSeNetOutput(in_chan=256, mid_chan=256, num_class)` - the ONLY output
/// head this crate builds; `conv_out16`/`conv_out32` (upstream's two
/// auxiliary, training-only heads) are never read by the reference
/// inference path (`pipeline_flux.py` takes only `[0]` of BiSeNet's forward
/// return) and are not imported.
pub const OUTPUT_MID: u32 = 256;

impl BiSeNetConfig {
    pub fn bisenet() -> BiSeNetConfig {
        BiSeNetConfig { num_class: 19, input_size: 512 }
    }
}
