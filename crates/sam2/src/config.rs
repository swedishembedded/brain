// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SAM 2 image-path configuration, transcribed 1:1 from the reference
//! `sam2/configs/sam2.1/sam2.1_hiera_{l,t}.yaml` plus the constructor defaults
//! `sam2.modeling.sam2_base.SAM2Base._build_sam_heads` bakes in
//! (`num_multimask_outputs=3`, `mlp_dim=2048`, `num_heads=8`,
//! `iou_head_depth=3`, `mask_in_chans=16`, ...).
//!
//! The per-block Hiera table is DERIVED here, by the same loop the reference
//! runs, rather than transcribed: the "window size lags by a block" rule and the
//! `dim_mul`/`head_mul` schedule are exactly the places a hand-typed table goes
//! wrong. [`Sam2Config::blocks`] is checked against the reference's own dumped
//! table in `tests/parity.rs`.

/// One Hiera `MultiScaleBlock`, fully resolved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockSpec {
    pub index: u32,
    /// Input channels.
    pub dim: u32,
    /// Output channels (`2*dim` at a stage boundary).
    pub dim_out: u32,
    pub num_heads: u32,
    /// 0 = global attention over the whole token grid.
    pub window_size: u32,
    /// `MaxPool2d(2, 2)` on the attention QUERY (and on the residual shortcut).
    pub q_pool: bool,
    /// Token grid entering the block.
    pub in_hw: (u32, u32),
    /// Token grid leaving it (halved iff `q_pool`).
    pub out_hw: (u32, u32),
}

impl BlockSpec {
    pub fn head_dim(&self) -> u32 {
        self.dim_out / self.num_heads
    }
    /// Window size the attention OUTPUT is reassembled at — halved by `q_pool`
    /// ("the window size lags by a block": block 2 partitions at 8 on a 256²
    /// grid and reassembles at 4 on a 128² one).
    pub fn out_window(&self) -> u32 {
        if self.q_pool {
            self.window_size / 2
        } else {
            self.window_size
        }
    }
    /// Does the reference's `window_partition` have to zero-pad this block's
    /// token grid? True for hiera-tiny's 14/7 windows on 64²/32² grids, false
    /// for every hiera-large block at 1024.
    pub fn needs_pad(&self) -> bool {
        self.window_size > 0
            && (!self.in_hw.0.is_multiple_of(self.window_size) || !self.in_hw.1.is_multiple_of(self.window_size))
    }
    /// Padded grid the windows tile (`== in_hw` when [`Self::needs_pad`] is false).
    pub fn pad_hw(&self) -> (u32, u32) {
        pad_to(self.in_hw, self.window_size)
    }
    /// Padded grid the attention OUTPUT tiles.
    pub fn out_pad_hw(&self) -> (u32, u32) {
        pad_to(self.out_hw, self.out_window())
    }
}

fn pad_to(hw: (u32, u32), win: u32) -> (u32, u32) {
    if win == 0 {
        return hw;
    }
    (hw.0.div_ceil(win) * win, hw.1.div_ceil(win) * win)
}

/// The image path of SAM 2.1. Video memory (`memory_attention`,
/// `memory_encoder`, object-pointer temporal encoding) is deliberately absent —
/// see `docs/imaging/plan.md` open decision 3.
#[derive(Clone, Debug)]
pub struct Sam2Config {
    // ---- image / trunk ----
    pub image_size: u32,
    pub backbone_stride: u32,
    pub patch_kernel: u32,
    pub patch_stride: u32,
    pub patch_pad: u32,
    pub embed_dim: u32,
    pub num_heads: u32,
    pub stages: Vec<u32>,
    pub global_att_blocks: Vec<u32>,
    pub window_pos_embed_bkg_spatial_size: (u32, u32),
    pub window_spec: Vec<u32>,
    /// Number of stage boundaries that pool the query (reference default 3).
    pub q_pool: usize,
    pub q_stride: u32,
    pub dim_mul: u32,
    pub head_mul: u32,
    pub mlp_ratio: u32,
    pub trunk_eps: f32,
    // ---- neck ----
    pub d_model: u32,
    pub backbone_channel_list: Vec<u32>,
    pub fpn_top_down_levels: Vec<u32>,
    pub scalp: u32,
    pub pos_sine_num_pos_feats: u32,
    pub pos_sine_temperature: f32,
    // ---- prompt encoder / mask decoder ----
    pub mask_in_chans: u32,
    pub transformer_depth: u32,
    pub transformer_heads: u32,
    pub transformer_mlp_dim: u32,
    pub attention_downsample_rate: u32,
    pub num_multimask_outputs: u32,
    pub iou_head_depth: u32,
    pub iou_head_hidden_dim: u32,
    pub use_high_res_features: bool,
    pub iou_prediction_use_sigmoid: bool,
    pub pred_obj_scores: bool,
    pub pred_obj_scores_mlp: bool,
    pub use_multimask_token_for_obj_ptr: bool,
    pub fixed_no_obj_ptr: bool,
    pub use_obj_ptrs_in_encoder: bool,
    pub use_mlp_for_obj_ptr_proj: bool,
    /// `nn.LayerNorm` default; the trunk's own norms use [`Self::trunk_eps`].
    pub ln_eps: f32,
    /// `LayerNorm2d` (ConvNeXt / SAM 2 spelling).
    pub ln2d_eps: f32,
    /// ImageNet normalisation SAM 2 preprocesses with.
    pub pixel_mean: [f32; 3],
    pub pixel_std: [f32; 3],
}

impl Sam2Config {
    /// `sam2.1_hiera_l.yaml`.
    pub fn hiera_large() -> Sam2Config {
        Sam2Config {
            embed_dim: 144,
            num_heads: 2,
            stages: vec![2, 6, 36, 4],
            global_att_blocks: vec![23, 33, 43],
            window_pos_embed_bkg_spatial_size: (7, 7),
            window_spec: vec![8, 4, 16, 8],
            backbone_channel_list: vec![1152, 576, 288, 144],
            ..Sam2Config::common()
        }
    }

    /// `sam2.1_hiera_t.yaml` — note it does NOT override `window_spec`, so it
    /// keeps the `Hiera` constructor default `[8, 4, 14, 7]`, which is the
    /// configuration whose windows do not divide the token grid.
    pub fn hiera_tiny() -> Sam2Config {
        Sam2Config {
            embed_dim: 96,
            num_heads: 1,
            stages: vec![1, 2, 7, 2],
            global_att_blocks: vec![5, 7, 9],
            window_pos_embed_bkg_spatial_size: (7, 7),
            window_spec: vec![8, 4, 14, 7],
            backbone_channel_list: vec![768, 384, 192, 96],
            ..Sam2Config::common()
        }
    }

    /// Everything the two share (and every SAM 2.1 variant shares).
    fn common() -> Sam2Config {
        Sam2Config {
            image_size: 1024,
            backbone_stride: 16,
            patch_kernel: 7,
            patch_stride: 4,
            patch_pad: 3,
            embed_dim: 0,
            num_heads: 0,
            stages: Vec::new(),
            global_att_blocks: Vec::new(),
            window_pos_embed_bkg_spatial_size: (7, 7),
            window_spec: Vec::new(),
            q_pool: 3,
            q_stride: 2,
            dim_mul: 2,
            head_mul: 2,
            mlp_ratio: 4,
            trunk_eps: 1e-6,
            d_model: 256,
            backbone_channel_list: Vec::new(),
            fpn_top_down_levels: vec![2, 3],
            scalp: 1,
            pos_sine_num_pos_feats: 256,
            pos_sine_temperature: 10000.0,
            mask_in_chans: 16,
            transformer_depth: 2,
            transformer_heads: 8,
            transformer_mlp_dim: 2048,
            attention_downsample_rate: 2,
            num_multimask_outputs: 3,
            iou_head_depth: 3,
            iou_head_hidden_dim: 256,
            use_high_res_features: true,
            iou_prediction_use_sigmoid: true,
            pred_obj_scores: true,
            pred_obj_scores_mlp: true,
            use_multimask_token_for_obj_ptr: true,
            fixed_no_obj_ptr: true,
            use_obj_ptrs_in_encoder: true,
            use_mlp_for_obj_ptr_proj: true,
            ln_eps: 1e-5,
            ln2d_eps: 1e-6,
            // The workspace's ONE copy of the ImageNet statistics
            // (`imaging::color`), not a re-typed literal: `depth` and `mirror`
            // each declared them byte-identically before they were hoisted.
            pixel_mean: imaging::IMAGENET_MEAN,
            pixel_std: imaging::IMAGENET_STD,
        }
    }

    pub fn depth(&self) -> u32 {
        self.stages.iter().sum()
    }

    /// `[sum(stages[..i]) - 1 for i in 1..=len]` — the LAST block index of each
    /// stage.
    pub fn stage_ends(&self) -> Vec<u32> {
        (1..=self.stages.len()).map(|i| self.stages[..i].iter().sum::<u32>() - 1).collect()
    }

    /// `[e + 1 for e in stage_ends[..-1]][:q_pool]` — the FIRST block of each
    /// new stage, which is also the block whose `dim != dim_out`.
    pub fn q_pool_blocks(&self) -> Vec<u32> {
        let ends = self.stage_ends();
        ends[..ends.len() - 1].iter().map(|e| e + 1).take(self.q_pool).collect()
    }

    /// Token grid the trunk starts at (`image_size / patch_stride`).
    pub fn trunk_grid(&self) -> u32 {
        self.image_size / self.patch_stride
    }

    /// Feature-map side of the SAM image embedding (`image_size /
    /// backbone_stride`, 64 at 1024).
    pub fn image_embedding_size(&self) -> u32 {
        self.image_size / self.backbone_stride
    }

    pub fn num_mask_tokens(&self) -> u32 {
        self.num_multimask_outputs + 1
    }

    /// The per-block table, derived by the reference's own loop.
    pub fn blocks(&self) -> Vec<BlockSpec> {
        let ends = self.stage_ends();
        let qpb = self.q_pool_blocks();
        let mut embed_dim = self.embed_dim;
        let mut num_heads = self.num_heads;
        let mut cur_stage = 1usize;
        let mut h = self.trunk_grid();
        let mut out = Vec::with_capacity(self.depth() as usize);
        for i in 0..self.depth() {
            let mut dim_out = embed_dim;
            // The window size lags by a block: it is read BEFORE the stage
            // counter advances, so the first block of a new stage still uses the
            // previous stage's window (and reassembles at half of it).
            let mut window_size = self.window_spec[cur_stage - 1];
            if self.global_att_blocks.contains(&i) {
                window_size = 0;
            }
            if i > 0 && ends.contains(&(i - 1)) {
                dim_out = embed_dim * self.dim_mul;
                num_heads *= self.head_mul;
                cur_stage += 1;
            }
            let q_pool = qpb.contains(&i);
            let out_h = if q_pool { h / self.q_stride } else { h };
            out.push(BlockSpec {
                index: i,
                dim: embed_dim,
                dim_out,
                num_heads,
                window_size,
                q_pool,
                in_hw: (h, h),
                out_hw: (out_h, out_h),
            });
            embed_dim = dim_out;
            h = out_h;
        }
        out
    }

    /// `trunk.channel_list` — the neck's `backbone_channel_list`, i.e. the stage
    /// outputs in REVERSE resolution order.
    pub fn trunk_channel_list(&self) -> Vec<u32> {
        let b = self.blocks();
        self.stage_ends().iter().rev().map(|&e| b[e as usize].dim_out).collect()
    }

    // -----------------------------------------------------------------------
    // canonical tensor manifest
    // -----------------------------------------------------------------------

    /// Every image-path tensor this model expects, with its exact checkpoint
    /// name and shape. Import validates both directions against it.
    pub fn tensor_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let mut v: Vec<(String, Vec<usize>)> = Vec::new();
        let d = self.d_model as usize;
        let mut push = |n: String, s: Vec<usize>| v.push((n, s));

        // ---- trunk ----
        let e = self.embed_dim as usize;
        let (pw, ph) = self.window_pos_embed_bkg_spatial_size;
        push("image_encoder.trunk.pos_embed".into(), vec![1, e, pw as usize, ph as usize]);
        let w0 = self.window_spec[0] as usize;
        push("image_encoder.trunk.pos_embed_window".into(), vec![1, e, w0, w0]);
        let k = self.patch_kernel as usize;
        push("image_encoder.trunk.patch_embed.proj.weight".into(), vec![e, 3, k, k]);
        push("image_encoder.trunk.patch_embed.proj.bias".into(), vec![e]);
        for b in self.blocks() {
            let p = format!("image_encoder.trunk.blocks.{}", b.index);
            let (di, dd) = (b.dim as usize, b.dim_out as usize);
            let m = dd * self.mlp_ratio as usize;
            push(format!("{p}.norm1.weight"), vec![di]);
            push(format!("{p}.norm1.bias"), vec![di]);
            push(format!("{p}.attn.qkv.weight"), vec![3 * dd, di]);
            push(format!("{p}.attn.qkv.bias"), vec![3 * dd]);
            push(format!("{p}.attn.proj.weight"), vec![dd, dd]);
            push(format!("{p}.attn.proj.bias"), vec![dd]);
            push(format!("{p}.norm2.weight"), vec![dd]);
            push(format!("{p}.norm2.bias"), vec![dd]);
            push(format!("{p}.mlp.layers.0.weight"), vec![m, dd]);
            push(format!("{p}.mlp.layers.0.bias"), vec![m]);
            push(format!("{p}.mlp.layers.1.weight"), vec![dd, m]);
            push(format!("{p}.mlp.layers.1.bias"), vec![dd]);
            if di != dd {
                push(format!("{p}.proj.weight"), vec![dd, di]);
                push(format!("{p}.proj.bias"), vec![dd]);
            }
        }

        // ---- neck: convs[j] is applied to trunk LEVEL (n-j) ----
        for (j, &c) in self.backbone_channel_list.iter().enumerate() {
            push(format!("image_encoder.neck.convs.{j}.conv.weight"), vec![d, c as usize, 1, 1]);
            push(format!("image_encoder.neck.convs.{j}.conv.bias"), vec![d]);
        }

        // ---- prompt encoder ----
        push("sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix".into(), vec![2, d / 2]);
        for i in 0..4 {
            push(format!("sam_prompt_encoder.point_embeddings.{i}.weight"), vec![1, d]);
        }
        push("sam_prompt_encoder.not_a_point_embed.weight".into(), vec![1, d]);
        let mc = self.mask_in_chans as usize;
        push("sam_prompt_encoder.mask_downscaling.0.weight".into(), vec![mc / 4, 1, 2, 2]);
        push("sam_prompt_encoder.mask_downscaling.0.bias".into(), vec![mc / 4]);
        push("sam_prompt_encoder.mask_downscaling.1.weight".into(), vec![mc / 4]);
        push("sam_prompt_encoder.mask_downscaling.1.bias".into(), vec![mc / 4]);
        push("sam_prompt_encoder.mask_downscaling.3.weight".into(), vec![mc, mc / 4, 2, 2]);
        push("sam_prompt_encoder.mask_downscaling.3.bias".into(), vec![mc]);
        push("sam_prompt_encoder.mask_downscaling.4.weight".into(), vec![mc]);
        push("sam_prompt_encoder.mask_downscaling.4.bias".into(), vec![mc]);
        push("sam_prompt_encoder.mask_downscaling.6.weight".into(), vec![d, mc, 1, 1]);
        push("sam_prompt_encoder.mask_downscaling.6.bias".into(), vec![d]);
        push("sam_prompt_encoder.no_mask_embed.weight".into(), vec![1, d]);

        // ---- two-way transformer ----
        let internal = d / self.attention_downsample_rate as usize;
        let mlp = self.transformer_mlp_dim as usize;
        for l in 0..self.transformer_depth {
            let p = format!("sam_mask_decoder.transformer.layers.{l}");
            for (attn, io) in [
                ("self_attn", d),
                ("cross_attn_token_to_image", internal),
                ("cross_attn_image_to_token", internal),
            ] {
                for proj in ["q_proj", "k_proj", "v_proj"] {
                    push(format!("{p}.{attn}.{proj}.weight"), vec![io, d]);
                    push(format!("{p}.{attn}.{proj}.bias"), vec![io]);
                }
                push(format!("{p}.{attn}.out_proj.weight"), vec![d, io]);
                push(format!("{p}.{attn}.out_proj.bias"), vec![d]);
            }
            for n in 1..=4 {
                push(format!("{p}.norm{n}.weight"), vec![d]);
                push(format!("{p}.norm{n}.bias"), vec![d]);
            }
            push(format!("{p}.mlp.layers.0.weight"), vec![mlp, d]);
            push(format!("{p}.mlp.layers.0.bias"), vec![mlp]);
            push(format!("{p}.mlp.layers.1.weight"), vec![d, mlp]);
            push(format!("{p}.mlp.layers.1.bias"), vec![d]);
        }
        {
            let p = "sam_mask_decoder.transformer.final_attn_token_to_image";
            for proj in ["q_proj", "k_proj", "v_proj"] {
                push(format!("{p}.{proj}.weight"), vec![internal, d]);
                push(format!("{p}.{proj}.bias"), vec![internal]);
            }
            push(format!("{p}.out_proj.weight"), vec![d, internal]);
            push(format!("{p}.out_proj.bias"), vec![d]);
        }
        push("sam_mask_decoder.transformer.norm_final_attn.weight".into(), vec![d]);
        push("sam_mask_decoder.transformer.norm_final_attn.bias".into(), vec![d]);

        // ---- mask decoder heads ----
        let nmt = self.num_mask_tokens() as usize;
        push("sam_mask_decoder.iou_token.weight".into(), vec![1, d]);
        push("sam_mask_decoder.mask_tokens.weight".into(), vec![nmt, d]);
        if self.pred_obj_scores {
            push("sam_mask_decoder.obj_score_token.weight".into(), vec![1, d]);
        }
        push("sam_mask_decoder.output_upscaling.0.weight".into(), vec![d, d / 4, 2, 2]);
        push("sam_mask_decoder.output_upscaling.0.bias".into(), vec![d / 4]);
        push("sam_mask_decoder.output_upscaling.1.weight".into(), vec![d / 4]);
        push("sam_mask_decoder.output_upscaling.1.bias".into(), vec![d / 4]);
        push("sam_mask_decoder.output_upscaling.3.weight".into(), vec![d / 4, d / 8, 2, 2]);
        push("sam_mask_decoder.output_upscaling.3.bias".into(), vec![d / 8]);
        if self.use_high_res_features {
            push("sam_mask_decoder.conv_s0.weight".into(), vec![d / 8, d, 1, 1]);
            push("sam_mask_decoder.conv_s0.bias".into(), vec![d / 8]);
            push("sam_mask_decoder.conv_s1.weight".into(), vec![d / 4, d, 1, 1]);
            push("sam_mask_decoder.conv_s1.bias".into(), vec![d / 4]);
        }
        for i in 0..nmt {
            let p = format!("sam_mask_decoder.output_hypernetworks_mlps.{i}");
            push(format!("{p}.layers.0.weight"), vec![d, d]);
            push(format!("{p}.layers.0.bias"), vec![d]);
            push(format!("{p}.layers.1.weight"), vec![d, d]);
            push(format!("{p}.layers.1.bias"), vec![d]);
            push(format!("{p}.layers.2.weight"), vec![d / 8, d]);
            push(format!("{p}.layers.2.bias"), vec![d / 8]);
        }
        mlp_manifest(&mut v, "sam_mask_decoder.iou_prediction_head", d, self.iou_head_hidden_dim as usize, nmt, self.iou_head_depth);
        if self.pred_obj_scores && self.pred_obj_scores_mlp {
            mlp_manifest(&mut v, "sam_mask_decoder.pred_obj_score_head", d, d, 1, 3);
        }
        if self.use_obj_ptrs_in_encoder {
            if self.use_mlp_for_obj_ptr_proj {
                mlp_manifest(&mut v, "obj_ptr_proj", d, d, d, 3);
            } else {
                v.push(("obj_ptr_proj.weight".into(), vec![d, d]));
                v.push(("obj_ptr_proj.bias".into(), vec![d]));
            }
        }
        if self.fixed_no_obj_ptr {
            v.push(("no_obj_ptr".into(), vec![1, d]));
        }
        v
    }
}

/// The `sam2_utils.MLP` name/shape series: `layers.{i}.{weight,bias}` with
/// `num_layers - 1` hidden layers of `hidden` and a final `out`.
fn mlp_manifest(v: &mut Vec<(String, Vec<usize>)>, prefix: &str, input: usize, hidden: usize, out: usize, layers: u32) {
    let mut fan_in = input;
    for i in 0..layers {
        let fan_out = if i + 1 == layers { out } else { hidden };
        v.push((format!("{prefix}.layers.{i}.weight"), vec![fan_out, fan_in]));
        v.push((format!("{prefix}.layers.{i}.bias"), vec![fan_out]));
        fan_in = fan_out;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_block_table_matches_the_reference_schedule() {
        let cfg = Sam2Config::hiera_large();
        let b = cfg.blocks();
        assert_eq!(b.len(), 48);
        assert_eq!(cfg.stage_ends(), vec![1, 7, 43, 47]);
        assert_eq!(cfg.q_pool_blocks(), vec![2, 8, 44]);
        assert_eq!(cfg.trunk_channel_list(), vec![1152, 576, 288, 144]);
        // block 2: dim 144 -> 288, heads 4, partitions at 8 on 256^2 and
        // reassembles at 4 on 128^2.
        assert_eq!(b[2].dim, 144);
        assert_eq!(b[2].dim_out, 288);
        assert_eq!(b[2].num_heads, 4);
        assert_eq!(b[2].window_size, 8);
        assert_eq!(b[2].out_window(), 4);
        assert_eq!(b[2].in_hw, (256, 256));
        assert_eq!(b[2].out_hw, (128, 128));
        // global-attention blocks
        for i in [23u32, 33, 43] {
            assert_eq!(b[i as usize].window_size, 0);
        }
        // no window padding anywhere at 1024
        assert!(b.iter().all(|s| !s.needs_pad()));
    }

    #[test]
    fn tiny_needs_window_padding() {
        let cfg = Sam2Config::hiera_tiny();
        let b = cfg.blocks();
        assert_eq!(b.len(), 12);
        assert_eq!(cfg.stage_ends(), vec![0, 2, 9, 11]);
        assert_eq!(cfg.q_pool_blocks(), vec![1, 3, 10]);
        assert_eq!(cfg.trunk_channel_list(), vec![768, 384, 192, 96]);
        let padded: Vec<u32> = b.iter().filter(|s| s.needs_pad()).map(|s| s.index).collect();
        assert_eq!(padded, vec![4, 6, 8, 10, 11]);
        assert_eq!(b[10].pad_hw(), (70, 70));
        assert_eq!(b[10].out_pad_hw(), (35, 35));
        assert_eq!(b[10].out_hw, (32, 32));
    }

    #[test]
    fn manifest_counts_the_image_path_tensors() {
        assert_eq!(Sam2Config::hiera_large().tensor_manifest().len(), 749);
        assert_eq!(Sam2Config::hiera_tiny().tensor_manifest().len(), 317);
    }
}
