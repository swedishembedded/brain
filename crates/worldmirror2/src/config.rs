// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! WorldMirror-2 model shape + the full parameter layout.
//!
//! `MirrorConfig::param_list` enumerates every tensor with the reference
//! checkpoint's own `state_dict` names (the ZipDepth precedent): the importer
//! is then a strict 1:1 name copy with zero translation tables, and the layout
//! is gated device-free against the committed safetensors header
//! (`tests/t0_param_layout.rs`).
//!
//! The one aliasing quirk: the reference shares a single 2D-RoPE `periods`
//! buffer across all 48 trunk blocks (safetensors dedups the other 47 into
//! `__metadata__` aliases), so exactly one tensor exists —
//! `…frame_blocks.0.attn.rope.periods` — and the trunk reads it once.

/// (state_dict name, shape). Shapes follow PyTorch conventions:
/// Linear `[out, in]`, Conv2d `[out, in, kh, kw]`, ConvTranspose2d
/// `[in, out, kh, kw]`, LayerNorm/bias `[n]`.
pub type Param = (String, Vec<usize>);

#[derive(Clone, Debug)]
pub struct MirrorConfig {
    /// Trunk levels (frame+global block pairs) and DINOv2 depth.
    pub depth: usize,
    /// Trunk / DINOv2 embedding dim.
    pub dim: usize,
    /// Attention heads (trunk, DINOv2, camera head).
    pub heads: usize,
    /// MLP expansion factor.
    pub mlp_ratio: usize,
    /// Patch size (DINOv2 ViT-L/14).
    pub patch: usize,
    /// Native training resolution (informational; inference is size-driven).
    pub img: usize,
    /// DINOv2 register tokens.
    pub reg_tokens: usize,
    /// Trunk levels tapped for the heads (frame‖global concat → 2*dim).
    pub tap_levels: [usize; 4],
    /// DPT per-scale projection channels.
    pub dpt_proj: [usize; 4],
    /// DPT fusion feature width.
    pub dpt_feat: usize,
    /// Camera-head refinement blocks.
    pub cam_blocks: usize,
    /// Camera parameter vector length: [t(3), quat_xyzw(4), fov_v, fov_u].
    pub cam_params: usize,
}

impl Default for MirrorConfig {
    fn default() -> Self {
        MirrorConfig {
            depth: 24,
            dim: 1024,
            heads: 16,
            mlp_ratio: 4,
            patch: 14,
            img: 518,
            reg_tokens: 4,
            tap_levels: [4, 11, 17, 23],
            dpt_proj: [256, 512, 1024, 1024],
            dpt_feat: 256,
            cam_blocks: 4,
            cam_params: 9,
        }
    }
}

fn linear(out: &mut Vec<Param>, name: &str, o: usize, i: usize) {
    out.push((format!("{name}.weight"), vec![o, i]));
    out.push((format!("{name}.bias"), vec![o]));
}

/// LayerNorm / any (weight, bias) pair of width `n`.
fn ln(out: &mut Vec<Param>, name: &str, n: usize) {
    out.push((format!("{name}.weight"), vec![n]));
    out.push((format!("{name}.bias"), vec![n]));
}

fn conv(out: &mut Vec<Param>, name: &str, o: usize, i: usize, k: usize, bias: bool) {
    out.push((format!("{name}.weight"), vec![o, i, k, k]));
    if bias {
        out.push((format!("{name}.bias"), vec![o]));
    }
}

impl MirrorConfig {
    pub fn head_dim(&self) -> usize {
        self.dim / self.heads
    }

    /// Patch tokens per frame at the native resolution (37×37 = 1369 at 518).
    pub fn patches(&self) -> usize {
        (self.img / self.patch) * (self.img / self.patch)
    }

    /// Pre-norm ViT block `name` at width `d`: norm1/qkv[/qk-norm]/proj/ls1,
    /// norm2/mlp/ls2. `qk_norm` adds the per-head-dim LayerNorms the trunk
    /// blocks have (DINOv2 and camera-head blocks do not).
    fn vit_block(&self, out: &mut Vec<Param>, name: &str, d: usize, qk_norm: bool) {
        let hd = d / self.heads * self.heads; // d is always heads-divisible here
        debug_assert_eq!(hd, d);
        ln(out, &format!("{name}.norm1"), d);
        linear(out, &format!("{name}.attn.qkv"), 3 * d, d);
        if qk_norm {
            ln(out, &format!("{name}.attn.q_norm"), d / self.heads);
            ln(out, &format!("{name}.attn.k_norm"), d / self.heads);
        }
        linear(out, &format!("{name}.attn.proj"), d, d);
        out.push((format!("{name}.ls1.gamma"), vec![d]));
        ln(out, &format!("{name}.norm2"), d);
        linear(out, &format!("{name}.mlp.fc1"), self.mlp_ratio * d, d);
        linear(out, &format!("{name}.mlp.fc2"), d, self.mlp_ratio * d);
        out.push((format!("{name}.ls2.gamma"), vec![d]));
    }

    /// One DPT dense head under `prefix`: token norm, per-scale 1×1 projects,
    /// resize layers (ConvT k4s4 / ConvT k2s2 / identity / Conv k3s2),
    /// scratch RN convs + 4 fusion blocks, and the 2-stage output conv ending
    /// in `out_ch` channels. `input_merger` adds the GS head's RGB encoder.
    fn dpt_head(&self, out: &mut Vec<Param>, prefix: &str, out_ch: usize, input_merger: bool) {
        let f = self.dpt_feat; // 256
        let tap = 2 * self.dim; // frame‖global concat
        ln(out, &format!("{prefix}.norm"), tap);
        for (i, &c) in self.dpt_proj.iter().enumerate() {
            conv(out, &format!("{prefix}.projects.{i}"), c, tap, 1, true);
        }
        // ConvTranspose2d weight layout is [in, out, k, k]; in == out per scale.
        out.push((format!("{prefix}.resize_layers.0.weight"), vec![self.dpt_proj[0], self.dpt_proj[0], 4, 4]));
        out.push((format!("{prefix}.resize_layers.0.bias"), vec![self.dpt_proj[0]]));
        out.push((format!("{prefix}.resize_layers.1.weight"), vec![self.dpt_proj[1], self.dpt_proj[1], 2, 2]));
        out.push((format!("{prefix}.resize_layers.1.bias"), vec![self.dpt_proj[1]]));
        // resize_layers.2 is Identity (no params).
        conv(out, &format!("{prefix}.resize_layers.3"), self.dpt_proj[3], self.dpt_proj[3], 3, true);
        for (i, &c) in self.dpt_proj.iter().enumerate() {
            out.push((format!("{prefix}.scratch.layer{}_rn.weight", i + 1), vec![f, c, 3, 3]));
        }
        for r in 1..=4 {
            // refinenet4 (deepest) has no resConfUnit1: nothing to fuse into it.
            if r != 4 {
                conv(out, &format!("{prefix}.scratch.refinenet{r}.resConfUnit1.conv1"), f, f, 3, true);
                conv(out, &format!("{prefix}.scratch.refinenet{r}.resConfUnit1.conv2"), f, f, 3, true);
            }
            conv(out, &format!("{prefix}.scratch.refinenet{r}.resConfUnit2.conv1"), f, f, 3, true);
            conv(out, &format!("{prefix}.scratch.refinenet{r}.resConfUnit2.conv2"), f, f, 3, true);
            conv(out, &format!("{prefix}.scratch.refinenet{r}.out_conv"), f, f, 1, true);
        }
        conv(out, &format!("{prefix}.scratch.output_conv1"), f / 2, f, 3, true);
        conv(out, &format!("{prefix}.scratch.output_conv2.0"), f / 8, f / 2, 3, true);
        conv(out, &format!("{prefix}.scratch.output_conv2.2"), out_ch, f / 8, 1, true);
        if input_merger {
            conv(out, &format!("{prefix}.input_merger.0"), f / 2, 3, 7, true);
        }
    }

    /// Every parameter of the model, checkpoint-verbatim names. 1545 tensors
    /// for the default config, gated against the committed header fixture.
    pub fn param_list(&self) -> Vec<Param> {
        let mut p: Vec<Param> = Vec::new();
        let d = self.dim;
        let vgt = "visual_geometry_transformer";

        // ---- learnable frame tokens (index 0 = frame 0, index 1 = rest) ----
        p.push((format!("{vgt}.cam_token"), vec![1, 2, 1, d]));
        p.push((format!("{vgt}.reg_token"), vec![1, 2, self.reg_tokens, d]));

        // ---- prior/condition encoders (zero-input path still allocates) ----
        linear(&mut p, &format!("{vgt}.pose_embed.0"), d, 7);
        linear(&mut p, &format!("{vgt}.pose_embed.2"), d, d);
        linear(&mut p, &format!("{vgt}.ray_embed.0"), d, 4);
        linear(&mut p, &format!("{vgt}.ray_embed.2"), d, d);
        let pp = self.patch * self.patch; // 196: PixelUnshuffle(14) of 1ch depth
        linear(&mut p, &format!("{vgt}.depth_embed.proj.2.fc1"), self.mlp_ratio * d, pp);
        linear(&mut p, &format!("{vgt}.depth_embed.proj.2.fc2"), d, self.mlp_ratio * d);

        // ---- patch_embed: full DINOv2 ViT-L/14-reg ----
        let pe = format!("{vgt}.patch_embed");
        p.push((format!("{pe}.cls_token"), vec![1, 1, d]));
        p.push((format!("{pe}.register_tokens"), vec![1, self.reg_tokens, d]));
        p.push((format!("{pe}.mask_token"), vec![1, d])); // unused at inference
        p.push((format!("{pe}.pos_embed"), vec![1, 1 + self.patches(), d]));
        conv(&mut p, &format!("{pe}.patch_embed.proj"), d, 3, self.patch, true);
        for b in 0..self.depth {
            self.vit_block(&mut p, &format!("{pe}.blocks.{b}"), d, false);
        }
        ln(&mut p, &format!("{pe}.norm"), d);

        // ---- trunk: alternating frame/global blocks ----
        for b in 0..self.depth {
            self.vit_block(&mut p, &format!("{vgt}.frame_blocks.{b}"), d, true);
            if b == 0 {
                // The single stored RoPE periods buffer (aliased by all blocks).
                p.push((
                    format!("{vgt}.frame_blocks.0.attn.rope.periods"),
                    vec![self.head_dim() / 4],
                ));
            }
            self.vit_block(&mut p, &format!("{vgt}.global_blocks.{b}"), d, true);
        }

        // ---- camera head (iterative refinement over the cam token) ----
        let cd = 2 * d; // 2048: frame‖global tap width
        p.push(("cam_head.init_token".into(), vec![1, 1, self.cam_params]));
        linear(&mut p, "cam_head.param_embed", cd, self.cam_params);
        ln(&mut p, "cam_head.token_norm", cd);
        ln(&mut p, "cam_head.out_norm", cd);
        linear(&mut p, "cam_head.adapt_norm_gen.1", 3 * cd, cd);
        for b in 0..self.cam_blocks {
            self.vit_block(&mut p, &format!("cam_head.refine_net.{b}"), cd, false);
        }
        linear(&mut p, "cam_head.param_predictor.fc1", d, cd);
        linear(&mut p, "cam_head.param_predictor.fc2", self.cam_params, d);

        // ---- dense heads ----
        self.dpt_head(&mut p, "depth_head", 3, false); // depth + conf + mask
        self.dpt_head(&mut p, "pts_head", 4, false); // xyz + conf
        self.dpt_head(&mut p, "norm_head", 4, false); // normal + conf
        self.dpt_head(&mut p, "gs_head", 3, true); // gs_depth + conf + mask

        // ---- gaussian parameter conv (on the GS head's 128ch feature) ----
        let f = self.dpt_feat;
        p.push(("gs_renderer.gs_head.0.weight".into(), vec![f, f / 2, 3, 3])); // no bias
        // 12 = quat(4) + scale(3) + opacity(1) + sh_dc(3) + weight(1)
        p.push(("gs_renderer.gs_head.2.weight".into(), vec![12, f, 1, 1]));
        p.push(("gs_renderer.gs_head.2.bias".into(), vec![12]));

        p
    }
}
