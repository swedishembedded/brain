// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `UNet2DConditionModel` configuration, plus the canonical brain-side
//! tensor manifest the importer validates against in **both** directions.
//!
//! Every number here was read off `unet/config.json` in the released
//! `stable-diffusion-xl-base-1.0` checkpoint and is re-asserted by the parity
//! test against `testdata/sdxl/manifest.json`.
//!
//! ## The one config field that is not what it says
//! diffusers' `attention_head_dim: [5, 10, 20]` is **not** a head dimension —
//! `UNet2DConditionModel.__init__` does `num_attention_heads =
//! num_attention_heads or attention_head_dim`, and SDXL ships
//! `num_attention_heads: null`. So those are HEAD COUNTS, and the real head
//! dim is `block_out_channels[i] / heads[i]` = 64 at every level. Reading the
//! field literally gives 5 heads of dim 5 and a forward that runs, produces the
//! right shapes, and is wrong everywhere.

/// What a down block does with the text conditioning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockKind {
    /// `DownBlock2D` / `UpBlock2D` — resnets only.
    Plain,
    /// `CrossAttnDownBlock2D` / `CrossAttnUpBlock2D` — resnet then a spatial
    /// transformer, per layer.
    CrossAttn,
}

/// One `UNet2DConditionModel`.
#[derive(Clone, Debug, PartialEq)]
pub struct UNetConfig {
    pub in_channels: u32,
    pub out_channels: u32,
    /// Per-level channel width, coarsest LAST (`[320, 640, 1280]`).
    pub block_out_channels: Vec<u32>,
    pub down_block_types: Vec<BlockKind>,
    pub up_block_types: Vec<BlockKind>,
    /// Resnets per down block; an up block has this many **plus one**.
    pub layers_per_block: u32,
    /// `BasicTransformerBlock`s inside each level's spatial transformer.
    pub transformer_layers_per_block: Vec<u32>,
    /// Attention HEAD COUNT per level — see the module header.
    pub attention_heads: Vec<u32>,
    pub cross_attention_dim: u32,
    pub norm_num_groups: u32,
    /// GroupNorm eps in the resnets and in `conv_norm_out` (1e-5).
    /// The spatial transformer's own `norm` uses [`TRANSFORMER_NORM_EPS`].
    pub norm_eps: f32,
    /// `time_embedding` width (`block_out_channels[0] * 4` = 1280).
    pub time_embed_dim: u32,
    /// Sinusoid width per micro-conditioning value (256).
    pub addition_time_embed_dim: u32,
    /// `add_embedding.linear_1` input width: `pooled_dim + 6 * 256` = 2816.
    pub projection_class_embeddings_input_dim: u32,
    /// `cos` first then `sin` (SDXL: true).
    pub flip_sin_to_cos: bool,
    /// `downscale_freq_shift` (SDXL: 0).
    pub freq_shift: f32,
    /// `proj_in`/`proj_out` are `nn.Linear` (true) or 1x1 convs (false).
    /// SDXL is true; the flag exists because SD 1.5 is false and the tensor
    /// SHAPES differ ([C, C] vs [C, C, 1, 1]), so a wrong value fails at import
    /// rather than silently.
    pub use_linear_projection: bool,
}

/// GroupNorm eps inside `Transformer2DModel` — hardcoded `1e-6` in diffusers'
/// `_init_continuous_input`, and NOT the config's `norm_eps` (1e-5) that the
/// resnets use. One `Builder` records both, so the epsilon is switched at the
/// boundary; getting it wrong is a ~1e-5 relative error that survives a loose
/// cosine gate.
pub const TRANSFORMER_NORM_EPS: f32 = 1e-6;

impl UNetConfig {
    /// `stabilityai/stable-diffusion-xl-base-1.0`, `unet/config.json`.
    pub fn sdxl_base() -> UNetConfig {
        UNetConfig {
            in_channels: 4,
            out_channels: 4,
            block_out_channels: vec![320, 640, 1280],
            down_block_types: vec![BlockKind::Plain, BlockKind::CrossAttn, BlockKind::CrossAttn],
            up_block_types: vec![BlockKind::CrossAttn, BlockKind::CrossAttn, BlockKind::Plain],
            layers_per_block: 2,
            transformer_layers_per_block: vec![1, 2, 10],
            attention_heads: vec![5, 10, 20],
            cross_attention_dim: 2048,
            norm_num_groups: 32,
            norm_eps: 1e-5,
            time_embed_dim: 1280,
            addition_time_embed_dim: 256,
            projection_class_embeddings_input_dim: 2816,
            flip_sin_to_cos: true,
            freq_shift: 0.0,
            use_linear_projection: true,
        }
    }

    /// A deliberately tiny variant for smoke tests: same graph shape, 2 levels,
    /// distinct dims everywhere so a transposed or swapped axis cannot hide.
    /// `heads` deliberately does NOT divide into the "obvious" 64.
    pub fn tiny() -> UNetConfig {
        UNetConfig {
            in_channels: 4,
            out_channels: 4,
            block_out_channels: vec![32, 64],
            down_block_types: vec![BlockKind::Plain, BlockKind::CrossAttn],
            up_block_types: vec![BlockKind::CrossAttn, BlockKind::Plain],
            layers_per_block: 1,
            transformer_layers_per_block: vec![1, 2],
            attention_heads: vec![2, 4],
            cross_attention_dim: 24,
            norm_num_groups: 8,
            norm_eps: 1e-5,
            time_embed_dim: 128,
            addition_time_embed_dim: 16,
            projection_class_embeddings_input_dim: 8 + 6 * 16,
            flip_sin_to_cos: true,
            freq_shift: 0.0,
            use_linear_projection: true,
        }
    }

    pub fn levels(&self) -> usize {
        self.block_out_channels.len()
    }

    /// `block_out_channels[i] / attention_heads[i]` — 64 at every SDXL level.
    pub fn head_dim(&self, level: usize) -> u32 {
        let c = self.block_out_channels[level];
        let h = self.attention_heads[level];
        assert_eq!(c % h, 0, "level {level}: {c} channels is not divisible by {h} heads");
        c / h
    }

    /// The pooled-text width `add_embedding` expects, derived rather than
    /// configured: `projection_class_embeddings_input_dim - 6 * addition_time_embed_dim`.
    /// 1280 for SDXL (OpenCLIP-bigG's projection output).
    pub fn pooled_dim(&self) -> u32 {
        self.projection_class_embeddings_input_dim - N_TIME_IDS * self.addition_time_embed_dim
    }

    /// Down-block level `i`'s output channel count (its downsampler, when
    /// present, keeps the width).
    fn down_out(&self, i: usize) -> u32 {
        self.block_out_channels[i]
    }

    /// The `prev_output_channel` an up block receives from the level above it.
    /// diffusers' `reversed_block_out_channels[max(i-1, 0)]`.
    fn up_prev(&self, i: usize) -> u32 {
        let rev: Vec<u32> = self.block_out_channels.iter().rev().copied().collect();
        rev[i.saturating_sub(1)]
    }

    /// Up-block level `i`'s own output channel count.
    fn up_out(&self, i: usize) -> u32 {
        self.block_out_channels[self.levels() - 1 - i]
    }

    /// The skip channel widths popped by up block `i`, in the order the block
    /// consumes them (LAST-pushed first). Derived from the down-block push
    /// order — see [`UNetConfig::skip_stack`].
    pub fn up_skips(&self, i: usize) -> Vec<u32> {
        let stack = self.skip_stack();
        let n = (self.layers_per_block + 1) as usize;
        // Up block i consumes the top n entries remaining after blocks 0..i.
        let end = stack.len() - i * n;
        let mut v = stack[end - n..end].to_vec();
        v.reverse();
        v
    }

    /// The full down-path skip stack, in PUSH order: `conv_in`, then every
    /// down-block resnet output and every downsampler output.
    ///
    /// This is the single most error-prone part of a UNet port — the up path
    /// pops `layers_per_block + 1` entries per level, and an off-by-one shifts
    /// every concat by one resolution while still type-checking. Deriving it
    /// once here (and asserting the total) is what keeps `model.rs` from
    /// re-deriving it.
    pub fn skip_stack(&self) -> Vec<u32> {
        let mut v = vec![self.block_out_channels[0]]; // conv_in
        for i in 0..self.levels() {
            for _ in 0..self.layers_per_block {
                v.push(self.down_out(i));
            }
            if i + 1 < self.levels() {
                v.push(self.down_out(i)); // downsampler
            }
        }
        v
    }

    /// `(channels, h, w)` of every entry in [`UNetConfig::skip_stack`], at a
    /// `h x w` latent - the spatial companion to its channel widths, walking
    /// the SAME down-path loop so the two can only ever agree.
    ///
    /// Not derivable from `skip_stack()`'s channels alone: two of SDXL's
    /// levels both carry 320-channel entries, so the spatial size has to come
    /// from re-walking the loop (`conv_in` and the `layers_per_block` resnets
    /// of level `i` sit at `h >> i`; the downsampler that ends level `i` is
    /// already at `h >> (i+1)`).
    ///
    /// A ControlNet's residuals are produced at exactly these sites plus one
    /// more for the mid block, which is why
    /// `ControlNetConfig::residual_shapes` (crates/controlnet) builds on this
    /// rather than re-walking the loop a second time.
    pub fn skip_shapes(&self, h: u32, w: u32) -> Vec<(u32, u32, u32)> {
        let levels = self.levels();
        let mut v = vec![(self.block_out_channels[0], h, w)];
        let (mut ch, mut cw) = (h, w);
        for i in 0..levels {
            let cout = self.block_out_channels[i];
            for _ in 0..self.layers_per_block {
                v.push((cout, ch, cw));
            }
            if i + 1 < levels {
                ch /= 2;
                cw /= 2;
                v.push((cout, ch, cw));
            }
        }
        v
    }

    /// Canonical brain-side tensor manifest: `(name, shape)` for every
    /// parameter the graph binds, in a stable order.
    ///
    /// Three fusions relative to the checkpoint, all done on the host at import
    /// time so every device matmul reads a whole buffer (porting playbook §2):
    ///   * `attn1.{to_q,to_k,to_v}` -> `attn1.qkv` `[3C, C]`;
    ///   * `attn2.{to_k,to_v}`      -> `attn2.kv`  `[2C, cross_dim]`;
    ///   * the GEGLU `ff.net.0.proj` `[2I, C]` is SPLIT into `ff.hidden` and
    ///     `ff.gate`, each `[I, C]` — the opposite move, and it is the right
    ///     one: `chunk(2, dim=-1)` splits every ROW, so the two halves are
    ///     interleaved in the fused buffer and no elementwise kernel can read
    ///     them as contiguous operands.
    pub fn tensor_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let mut v: Vec<(String, Vec<usize>)> = Vec::new();
        let c0 = self.block_out_channels[0] as usize;
        let te = self.time_embed_dim as usize;
        let lin = |v: &mut Vec<(String, Vec<usize>)>, p: &str, out: usize, inp: usize| {
            v.push((format!("{p}.weight"), vec![out, inp]));
            v.push((format!("{p}.bias"), vec![out]));
        };

        lin(&mut v, "time_embedding.linear_1", te, c0);
        lin(&mut v, "time_embedding.linear_2", te, te);
        lin(&mut v, "add_embedding.linear_1", te, self.projection_class_embeddings_input_dim as usize);
        lin(&mut v, "add_embedding.linear_2", te, te);
        v.push(("conv_in.weight".into(), vec![c0, self.in_channels as usize, 3, 3]));
        v.push(("conv_in.bias".into(), vec![c0]));

        let resnet = |v: &mut Vec<(String, Vec<usize>)>, p: &str, cin: usize, cout: usize| {
            v.push((format!("{p}.norm1.weight"), vec![cin]));
            v.push((format!("{p}.norm1.bias"), vec![cin]));
            v.push((format!("{p}.conv1.weight"), vec![cout, cin, 3, 3]));
            v.push((format!("{p}.conv1.bias"), vec![cout]));
            v.push((format!("{p}.time_emb_proj.weight"), vec![cout, te]));
            v.push((format!("{p}.time_emb_proj.bias"), vec![cout]));
            v.push((format!("{p}.norm2.weight"), vec![cout]));
            v.push((format!("{p}.norm2.bias"), vec![cout]));
            v.push((format!("{p}.conv2.weight"), vec![cout, cout, 3, 3]));
            v.push((format!("{p}.conv2.bias"), vec![cout]));
            if cin != cout {
                v.push((format!("{p}.conv_shortcut.weight"), vec![cout, cin, 1, 1]));
                v.push((format!("{p}.conv_shortcut.bias"), vec![cout]));
            }
        };

        let proj_shape = |c: usize| {
            if self.use_linear_projection {
                vec![c, c]
            } else {
                vec![c, c, 1, 1]
            }
        };
        let transformer = |v: &mut Vec<(String, Vec<usize>)>, p: &str, level: usize| {
            let c = self.block_out_channels[level] as usize;
            let x = self.cross_attention_dim as usize;
            let inner = 4 * c; // GEGLU inner dim: ff_mult = 4
            v.push((format!("{p}.norm.weight"), vec![c]));
            v.push((format!("{p}.norm.bias"), vec![c]));
            v.push((format!("{p}.proj_in.weight"), proj_shape(c)));
            v.push((format!("{p}.proj_in.bias"), vec![c]));
            for k in 0..self.transformer_layers_per_block[level] {
                let b = format!("{p}.transformer_blocks.{k}");
                v.push((format!("{b}.norm1.weight"), vec![c]));
                v.push((format!("{b}.norm1.bias"), vec![c]));
                v.push((format!("{b}.attn1.qkv.weight"), vec![3 * c, c]));
                v.push((format!("{b}.attn1.to_out.weight"), vec![c, c]));
                v.push((format!("{b}.attn1.to_out.bias"), vec![c]));
                v.push((format!("{b}.norm2.weight"), vec![c]));
                v.push((format!("{b}.norm2.bias"), vec![c]));
                v.push((format!("{b}.attn2.to_q.weight"), vec![c, c]));
                v.push((format!("{b}.attn2.kv.weight"), vec![2 * c, x]));
                v.push((format!("{b}.attn2.to_out.weight"), vec![c, c]));
                v.push((format!("{b}.attn2.to_out.bias"), vec![c]));
                v.push((format!("{b}.norm3.weight"), vec![c]));
                v.push((format!("{b}.norm3.bias"), vec![c]));
                v.push((format!("{b}.ff.hidden.weight"), vec![inner, c]));
                v.push((format!("{b}.ff.hidden.bias"), vec![inner]));
                v.push((format!("{b}.ff.gate.weight"), vec![inner, c]));
                v.push((format!("{b}.ff.gate.bias"), vec![inner]));
                v.push((format!("{b}.ff.out.weight"), vec![c, inner]));
                v.push((format!("{b}.ff.out.bias"), vec![c]));
            }
            v.push((format!("{p}.proj_out.weight"), proj_shape(c)));
            v.push((format!("{p}.proj_out.bias"), vec![c]));
        };

        // ---- down ----
        let mut prev = c0;
        for i in 0..self.levels() {
            let cout = self.down_out(i) as usize;
            for j in 0..self.layers_per_block {
                let cin = if j == 0 { prev } else { cout };
                resnet(&mut v, &format!("down_blocks.{i}.resnets.{j}"), cin, cout);
                if self.down_block_types[i] == BlockKind::CrossAttn {
                    transformer(&mut v, &format!("down_blocks.{i}.attentions.{j}"), i);
                }
            }
            if i + 1 < self.levels() {
                v.push((
                    format!("down_blocks.{i}.downsamplers.0.conv.weight"),
                    vec![cout, cout, 3, 3],
                ));
                v.push((format!("down_blocks.{i}.downsamplers.0.conv.bias"), vec![cout]));
            }
            prev = cout;
        }

        // ---- mid ----
        let cmid = *self.block_out_channels.last().expect("levels >= 1") as usize;
        resnet(&mut v, "mid_block.resnets.0", cmid, cmid);
        transformer(&mut v, "mid_block.attentions.0", self.levels() - 1);
        resnet(&mut v, "mid_block.resnets.1", cmid, cmid);

        // ---- up ----
        for i in 0..self.levels() {
            let level = self.levels() - 1 - i;
            let cout = self.up_out(i) as usize;
            let prev_out = self.up_prev(i) as usize;
            let skips = self.up_skips(i);
            for (j, &skip) in skips.iter().enumerate() {
                let res_skip = skip as usize;
                // diffusers `CrossAttnUpBlock2D`: `resnet_in_channels =
                // prev_output_channel if i == 0 else out_channels`. Only the
                // FIRST resnet of an up block sees the coarser level's width;
                // the rest see their own. Inverting this is invisible at
                // up_blocks.0 (where the two are both 1280) and fails at
                // up_blocks.1 (1280 vs 640) — which is exactly how it was
                // caught, by the importer's shape check.
                let cin = if j == 0 { prev_out } else { cout };
                resnet(&mut v, &format!("up_blocks.{i}.resnets.{j}"), cin + res_skip, cout);
                if self.up_block_types[i] == BlockKind::CrossAttn {
                    transformer(&mut v, &format!("up_blocks.{i}.attentions.{j}"), level);
                }
            }
            if i + 1 < self.levels() {
                v.push((format!("up_blocks.{i}.upsamplers.0.conv.weight"), vec![cout, cout, 3, 3]));
                v.push((format!("up_blocks.{i}.upsamplers.0.conv.bias"), vec![cout]));
            }
        }

        v.push(("conv_norm_out.weight".into(), vec![c0]));
        v.push(("conv_norm_out.bias".into(), vec![c0]));
        v.push(("conv_out.weight".into(), vec![self.out_channels as usize, c0, 3, 3]));
        v.push(("conv_out.bias".into(), vec![self.out_channels as usize]));
        v
    }
}

/// The six SDXL micro-conditioning values:
/// `(original_h, original_w, crop_top, crop_left, target_h, target_w)`.
pub const N_TIME_IDS: u32 = 6;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sdxl_head_dim_is_64_everywhere() {
        let c = UNetConfig::sdxl_base();
        for l in 0..c.levels() {
            assert_eq!(c.head_dim(l), 64, "level {l}");
        }
    }

    #[test]
    fn sdxl_pooled_dim_is_openclip_bigg() {
        assert_eq!(UNetConfig::sdxl_base().pooled_dim(), 1280);
    }

    /// The skip stack is `1 + levels*layers + (levels-1)` = 9 for SDXL, and
    /// the up path must consume exactly all of it.
    #[test]
    fn skip_stack_is_exactly_consumed() {
        for c in [UNetConfig::sdxl_base(), UNetConfig::tiny()] {
            let stack = c.skip_stack();
            let per_up = (c.layers_per_block + 1) as usize;
            assert_eq!(stack.len(), c.levels() * per_up, "{stack:?}");
            let mut seen = 0;
            for i in 0..c.levels() {
                assert_eq!(c.up_skips(i).len(), per_up);
                seen += per_up;
            }
            assert_eq!(seen, stack.len());
        }
        assert_eq!(UNetConfig::sdxl_base().skip_stack(), vec![320, 320, 320, 320, 640, 640, 640, 1280, 1280]);
        // Up block 0 pops the last three, most-recent first.
        assert_eq!(UNetConfig::sdxl_base().up_skips(0), vec![1280, 1280, 640]);
        assert_eq!(UNetConfig::sdxl_base().up_skips(1), vec![640, 640, 320]);
        assert_eq!(UNetConfig::sdxl_base().up_skips(2), vec![320, 320, 320]);
    }

    /// The up-path resnet input widths, read straight off the released
    /// checkpoint's `up_blocks.*.resnets.*.conv1.weight` shapes. This is the
    /// arithmetic that is easy to invert and impossible to notice at
    /// `up_blocks.0`, where `prev_output_channel == out_channels == 1280`.
    #[test]
    fn sdxl_up_resnet_input_widths_match_the_checkpoint() {
        let m = UNetConfig::sdxl_base().tensor_manifest();
        let want = [
            ("up_blocks.0.resnets.0.conv1.weight", 2560),
            ("up_blocks.0.resnets.1.conv1.weight", 2560),
            ("up_blocks.0.resnets.2.conv1.weight", 1920),
            ("up_blocks.1.resnets.0.conv1.weight", 1920),
            ("up_blocks.1.resnets.1.conv1.weight", 1280),
            ("up_blocks.1.resnets.2.conv1.weight", 960),
            ("up_blocks.2.resnets.0.conv1.weight", 960),
            ("up_blocks.2.resnets.1.conv1.weight", 640),
            ("up_blocks.2.resnets.2.conv1.weight", 640),
        ];
        for (name, cin) in want {
            let (_, shape) = m.iter().find(|(n, _)| n == name).unwrap_or_else(|| panic!("{name}"));
            assert_eq!(shape[1], cin, "{name} shape {shape:?}");
        }
    }

    /// The SDXL manifest must be exactly the released checkpoint's tensor
    /// count once the three fusions are accounted for:
    ///   1680 source
    ///     - 2 per fused attn1 qkv (3 -> 1)   : 61 transformers x layers
    ///     - 1 per fused attn2 kv  (2 -> 1)
    ///     + 2 per split GEGLU proj (1 -> 2 weights, 1 -> 2 biases)
    #[test]
    fn sdxl_manifest_matches_the_checkpoint_count() {
        let c = UNetConfig::sdxl_base();
        let m = c.tensor_manifest();
        // 70 transformer blocks in SDXL: down 1x2 + 2x2... counted from config.
        let mut tb = 0usize;
        for i in 0..c.levels() {
            if c.down_block_types[i] == BlockKind::CrossAttn {
                tb += (c.layers_per_block * c.transformer_layers_per_block[i]) as usize;
            }
        }
        tb += c.transformer_layers_per_block[c.levels() - 1] as usize; // mid
        for i in 0..c.levels() {
            let level = c.levels() - 1 - i;
            if c.up_block_types[i] == BlockKind::CrossAttn {
                tb += ((c.layers_per_block + 1) * c.transformer_layers_per_block[level]) as usize;
            }
        }
        assert_eq!(tb, 70, "SDXL has 70 BasicTransformerBlocks");
        // 1680 - 2*tb (qkv) - 1*tb (kv) + 2*tb (geglu split) = 1680 - tb.
        assert_eq!(m.len(), 1680 - tb, "manifest {} tensors", m.len());
        let names: std::collections::HashSet<&str> = m.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names.len(), m.len(), "duplicate names in the manifest");
    }
}
