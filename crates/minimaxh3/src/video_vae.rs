// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniMax-H3's video VAE: a **causal 3D CNN encoder** (6 down-stages,
//! `CausalConv3d` throughout) and a **non-causal ViT decoder** (36 full-
//! self-attention transformer blocks, partial 3-axis RoPE, no conv layers at
//! all - patchify/unpatchify are pure reshapes). Ported from the real
//! installed `diffusers==0.40.0`
//! `diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py`
//! (`AutoencoderKLMiniMaxH3` + its submodules), read in full - every
//! convention below cites the exact class/method it came from rather than a
//! secondhand writeup.
//!
//! Follows `crate::vocoder`'s eager Ctx/kernels-array dispatch style (every
//! op below immediately builds, submits and reads back through a fresh
//! [`gpu_core::DeviceBuffer`], no deferred graph) rather than
//! `vae::blocks3d::Builder3d`'s recorded-graph/`feat_cache` idiom - per the
//! roadmap ledger's own explicit warning, this checkpoint's causal conv is
//! genuinely NOT Wan/LTXV's chunked `feat_cache` scheme: `_encode_clip` reads
//! ONE independent temporal chunk from scratch every time (zero-pad causal,
//! no cross-chunk conv state at all - temporal continuity across a whole
//! video comes entirely from the DECODE-side cross-fade blend, never from a
//! carried encoder feature cache). Reusing `Builder3d::conv_cached`'s
//! `feat_cache` machinery here would silently implement the WRONG model.
//!
//! # Conventions this port is NOT free to guess (confirmed from source)
//!
//! * **Causal conv = zero-pad temporal (one-sided, prepend only) + REFLECT
//!   spatial (symmetric or asymmetric, never zero)** - two independent pad
//!   kinds on the SAME conv. `crates/kernels/wgsl/conv3d.wgsl`'s own `pt`
//!   parameter already implements exactly the one-sided zero temporal pad
//!   this checkpoint wants (no doubling needed: MiniMax-H3's own
//!   `temporal_padding` field, e.g. 2 for `kernel_size=3`, IS the kernel's
//!   `pt` value directly - unlike the Wan-VAE precedent's `padding=1`-style
//!   config that needs doubling, see `vae::blocks3d::Conv3d`'s own doc for
//!   that DIFFERENT convention). Spatial reflect pad has no existing kernel
//!   ([`crates::kernels::PAD2D_REFLECT`], a new one this port adds - see its
//!   own doc for why zero-pad `pad2d.wgsl`/`conv3d.wgsl`'s built-in `ph`/`pw`
//!   cannot be reused for it) applied BEFORE the conv, `ph=pw=0` at the
//!   `conv3d` call itself.
//! * **GroupNorm is "t-isolated"**: statistics never mix across frames
//!   (`MiniMaxH3VideoGroupNorm` folds T into the batch axis). This port
//!   reuses the EXISTING plain-image GroupNorm kernels
//!   (`gn_part`/`gn_stats2`/`gn_apply`, already batch-generic at `N`) by
//!   looping over T frames and extracting/placing each one as a genuinely
//!   contiguous `[C,H,W]` slice via the existing `concat_split`/`chan_place`
//!   kernels (`vae::blocks3d::Builder3d::time_slice`/`::time_place`'s own
//!   trick, at N=1 per frame) - NOT a new fused kernel, since the channel-
//!   major `[C,T,H,W]` layout this crate shares with `vae::blocks3d` does not
//!   match `gn_*`'s assumed `[N,C,H,W]` contiguous-per-image layout without
//!   this reshape.
//! * **The ViT decoder's RoPE has an extra `2*pi` factor** the DiT core's own
//!   `crate::rope` does NOT carry (`MiniMaxH3VideoRotaryPosEmbed.forward`:
//!   `angles = 2*pi*position_ids*inv_freq`, vs `crate::rope::build_tables`'s
//!   bare `pos*freq`) - genuinely different modules, not reusable as-is
//!   despite the shared "3-axis partial RoPE" shape. See
//!   [`decoder_rope_tables`].
//! * **Register/cls token positions are the coordinate origin**
//!   (`position_ids` all-zero for the 4 register tokens + 1 all-zero cls
//!   token, i.e. UNROTATED), and the cls token's own embedding is a literal
//!   zero vector, never a learned parameter.
//! * **SwiGLU's chunk order**: `diffusers.models.activations.SwiGLU.forward`
//!   splits its fused `proj` output into `(value, gate) =
//!   chunk(2,dim=-1)` (FIRST half unactivated, SECOND half through `SiLU`),
//!   `y = value * silu(gate)` - the SAME convention
//!   `crate::block::BlockWeights`'s own `fc1_value`/`fc1_gate` split already
//!   uses for the DiT core's fused `mlp.fc1`, reused verbatim here for the
//!   decoder's `ff.net.0.proj`.
//! * **`_decode`'s outer chunk/pad/blend arithmetic** (mirrors `_encode`'s
//!   own `clip_length`/`token_drop` chunking) is implemented per
//!   `AutoencoderKLMiniMaxH3._decode`/`_encode` exactly, including the
//!   `num_chunks` formula the roadmap ledger's own research pass flagged as
//!   "traced by hand, not run" - [`decode`]'s own tests exercise it at a
//!   non-degenerate multi-chunk size, not only the single-chunk case.
//!
//! * **Spatial tiling is part of the model, not a memory optimization.**
//!   `use_tiling` is `True` in the reference's own `__init__`, so the tiled
//!   path is the SHIPPED one, and [`encode_clip`]/[`decode_clip`] default to
//!   it here too. It cannot be treated as an optional chunking of the same
//!   computation: the ViT decoder's rotary coordinates are normalized to
//!   `[-1, 1)` across whatever grid it is handed, so the 256x256-pixel /
//!   16x16-latent tile size is baked into what the 36 attention layers were
//!   trained to read. Handing the decoder a whole large frame instead
//!   stretches every positional relationship by the frame-to-tile ratio
//!   while the per-token `proj_out` still paints a hard 16x16 pixel block -
//!   a regular patch grid, not blur. Below 256 pixels on both axes it makes
//!   no difference at all (`_split_tiles` returns a single full-size tile
//!   when `tile_size >= length`), which is exactly why the failure is easy
//!   to miss: the untiled path is correct at small canvases and
//!   progressively wrong above 256 pixels, the opposite of the usual
//!   "approximation degrades gracefully" intuition. See [`split_tiles`] and
//!   [`stitch_tiles`] for the three details of the algorithm that a
//!   plausible-looking implementation gets wrong.
//!
//! Swedish Embedded AB implements this video variational autoencoder port
//! for its clients. If your team needs expertise in porting causal 3D CNN /
//! ViT video codecs to new inference stacks, you can procure our services by
//! sending an email to info@swedishembedded.com.

use gpu_core::{f, DeviceBuffer, Gpu};
use vae::blocks::Tensors;

const K_CONV3D: usize = 0;
const K_SILU: usize = 1;
const K_ADD2: usize = 2;
const K_PAD2D_REFLECT: usize = 3;
const K_CONCAT_SPLIT: usize = 4;
const K_CHAN_PLACE: usize = 5;
const K_GN_PART: usize = 6;
const K_GN_STATS2: usize = 7;
const K_GN_APPLY: usize = 8;
const K_MATMUL: usize = 9;
const K_BIAS_ADD: usize = 10;
const K_RMSNORM_EPS: usize = 11;
const K_ROPE2D_PARTIAL: usize = 12;
const K_ATTN_SCORES_QK: usize = 13;
const K_ATTN_SOFTMAX_BIDIR: usize = 14;
const K_ATTN_APPLY_FULL: usize = 15;
const K_SILU_MUL: usize = 16;
const K_GATE_ROW: usize = 17;
const K_LAYERNORM_ROWS: usize = 18;
/// The 128x128 register-tiled GEMM `linear` picks via `model::block::
/// pick_gemm` once a shape clears the measured tile-fill crossover - the
/// same fix `crate::block::linear`'s own doc explains in full (plain
/// `matmul` is a naive one-thread-per-output kernel with a fully
/// uncoalesced weight-row access pattern on a real GPU; this crate's own
/// DiT already made this switch). The ViT decoder's per-tile token count is
/// bounded by the fixed 256x256-pixel tile size regardless of output
/// canvas, so this is a straightforward speed fix here, not the memory-wall
/// fix flash attention was for the DiT's unbounded packed sequence.
const K_MATMUL_REG3: usize = 19;

pub const KERNELS: [(&str, &str); 20] = [
    ("conv3d", kernels::CONV3D),
    ("silu", kernels::SILU),
    ("add2", kernels::ADD2),
    ("pad2d_reflect", kernels::PAD2D_REFLECT),
    ("concat_split", kernels::CONCAT_SPLIT),
    ("chan_place", kernels::CHAN_PLACE),
    ("gn_part", kernels::GN_PART),
    ("gn_stats2", kernels::GN_STATS2),
    ("gn_apply", kernels::GN_APPLY),
    ("matmul", kernels::MATMUL),
    ("bias_add", kernels::BIAS_ADD),
    ("rmsnorm_eps", kernels::RMSNORM_EPS),
    ("rope2d_partial", kernels::ROPE2D_PARTIAL),
    ("attn_scores_qk", kernels::ATTN_SCORES_QK),
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
    ("attn_apply_full", kernels::ATTN_APPLY_FULL),
    ("silu_mul", kernels::SILU_MUL),
    ("gate_row", kernels::GATE_ROW),
    ("layernorm_rows", kernels::LAYERNORM_ROWS),
    ("matmul_reg3", kernels::MATMUL_REG3),
];

/// The two-stage GroupNorm reduction's threads-per-group - matching
/// `vae::blocks::GN_P`'s own value (not importable, that constant is
/// private to that crate's `Builder`).
const GN_P: u32 = 64;

/// The 3 rotary axes (t, h, w) - `MiniMaxH3VideoRotaryPosEmbed`'s own
/// `num_axes` default.
const NUM_AXES: u32 = 3;

/// One open device + the shared kernel table, matching `crate::vocoder::Ctx`
/// / `crate::block::Ctx`'s own role.
pub struct Ctx {
    pub gpu: Gpu,
}

impl Ctx {
    pub fn new(device: Option<&str>) -> Ctx {
        Ctx { gpu: Gpu::open(device, &KERNELS) }
    }
}

// ==================================================================
// Config
// ==================================================================

/// `AutoencoderKLMiniMaxH3.__init__`'s field-for-field shape surface, split
/// into the encoder's own CNN geometry and the decoder's own ViT geometry
/// (the two submodules share only `latent_channels` and the compression
/// ratios the encoder's own downsample factors determine).
#[derive(Clone, Debug, PartialEq)]
pub struct VideoVaeConfig {
    pub in_channels: u32,
    pub out_channels: u32,
    pub latent_channels: u32,
    pub block_out_channels: [u32; 6],
    pub layers_per_block: u32,
    pub spatial_downsample_factors: [u32; 6],
    pub temporal_downsample_factors: [u32; 6],
    pub norm_num_groups: u32,
    pub norm_eps: f32,
    pub decoder_num_layers: u32,
    pub decoder_num_attention_heads: u32,
    pub decoder_attention_head_dim: u32,
    pub decoder_num_register_tokens: u32,
    pub decoder_ffn_mult: u32,
    pub decoder_rope_theta: f32,
    pub decoder_rope_dim_ratio: f32,
    pub decoder_norm_eps: f32,
    /// Pixel frames per encoder chunk (`_encode`'s own `clip_length`).
    pub clip_length: u32,
    /// Trailing latent frames dropped once per whole encode
    /// (`AutoencoderKLMiniMaxH3.config.token_drop`).
    pub token_drop: u32,
    /// `AutoencoderKLMiniMaxH3.use_tiling` - `True` in the reference's own
    /// `__init__`, i.e. tiling is the SHIPPED path, not an opt-in. These five
    /// fields are plain instance attributes there, not `config.json` entries,
    /// so their values come from the constructor body rather than the
    /// checkpoint.
    pub use_tiling: bool,
    /// `tile_sample_min_height` - the tile height in PIXEL space.
    pub tile_sample_min_height: u32,
    /// `tile_sample_min_width` - the tile width in PIXEL space.
    pub tile_sample_min_width: u32,
    /// `tile_sample_min_overlap_height` - the MINIMUM overlap between two
    /// vertically adjacent tiles, in pixels. The actual overlap is widened
    /// from here so the tiles cover the frame exactly, see [`split_tiles`].
    pub tile_sample_min_overlap_height: u32,
    /// `tile_sample_min_overlap_width` - the same, horizontally.
    pub tile_sample_min_overlap_width: u32,
}

impl Default for VideoVaeConfig {
    fn default() -> VideoVaeConfig {
        VideoVaeConfig::real()
    }
}

impl VideoVaeConfig {
    /// The real `MiniMaxAI/MiniMax-H3` `vae/config.json` numbers, read
    /// directly from `AutoencoderKLMiniMaxH3.__init__`'s own defaults
    /// (`autoencoder_kl_minimax_h3.py`).
    pub fn real() -> VideoVaeConfig {
        VideoVaeConfig {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 24,
            block_out_channels: [128, 256, 256, 512, 512, 1024],
            layers_per_block: 2,
            spatial_downsample_factors: [2, 2, 2, 2, 1, 1],
            temporal_downsample_factors: [1, 2, 2, 1, 1, 1],
            norm_num_groups: 32,
            norm_eps: 1e-6,
            decoder_num_layers: 36,
            decoder_num_attention_heads: 32,
            decoder_attention_head_dim: 64,
            decoder_num_register_tokens: 4,
            decoder_ffn_mult: 4,
            decoder_rope_theta: 100.0,
            decoder_rope_dim_ratio: 0.75,
            decoder_norm_eps: 1e-5,
            clip_length: 17,
            token_drop: 3,
            use_tiling: true,
            tile_sample_min_height: 256,
            tile_sample_min_width: 256,
            tile_sample_min_overlap_height: 64,
            tile_sample_min_overlap_width: 64,
        }
    }

    /// A small config for weight-free smoke tests and the real-weight
    /// numeric parity harness, at the REAL config's own proportions.
    /// `block_out_channels` keeps the real 1:2:2:4:4:8 stage-width ratio
    /// (scaled to a `norm_num_groups=2`-divisible base); `clip_length`/
    /// `token_drop` and BOTH downsample-factor tuples are kept LITERAL
    /// (structural constants, not widths - shrinking them would change
    /// which stages carry a downsample module and desync the encoder/
    /// decoder's shared `patch_size`/`patch_size_t` wiring from the real
    /// config's own shape, same reasoning as
    /// `H3TransformerConfig::tiny()` keeping `patch_size` literal). The five
    /// tiling fields are kept LITERAL for the same reason: they are measured
    /// in PIXELS against `spatial_compression_ratio` (which this config keeps
    /// literal at 16), so shrinking them would produce a tile geometry the
    /// reference never uses and the parity golden could not gate.
    pub fn tiny() -> VideoVaeConfig {
        VideoVaeConfig {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 4,
            block_out_channels: [8, 16, 16, 32, 32, 64],
            layers_per_block: 2,
            spatial_downsample_factors: [2, 2, 2, 2, 1, 1],
            temporal_downsample_factors: [1, 2, 2, 1, 1, 1],
            norm_num_groups: 2,
            norm_eps: 1e-6,
            decoder_num_layers: 2,
            decoder_num_attention_heads: 2,
            decoder_attention_head_dim: 8,
            decoder_num_register_tokens: 4,
            decoder_ffn_mult: 4,
            decoder_rope_theta: 100.0,
            decoder_rope_dim_ratio: 0.75,
            decoder_norm_eps: 1e-5,
            clip_length: 17,
            token_drop: 3,
            use_tiling: true,
            tile_sample_min_height: 256,
            tile_sample_min_width: 256,
            tile_sample_min_overlap_height: 64,
            tile_sample_min_overlap_width: 64,
        }
    }

    pub fn spatial_compression_ratio(&self) -> u32 {
        self.spatial_downsample_factors.iter().product()
    }

    pub fn temporal_compression_ratio(&self) -> u32 {
        self.temporal_downsample_factors.iter().product()
    }

    /// The decoder ViT's residual-stream width (`num_attention_heads *
    /// attention_head_dim`).
    pub fn decoder_dim(&self) -> u32 {
        self.decoder_num_attention_heads * self.decoder_attention_head_dim
    }

    /// `MiniMaxH3VideoRotaryPosEmbed`'s own `dim` - `int(attention_head_dim *
    /// rope_dim_ratio)`, the number of `attention_head_dim` channels per head
    /// that get rotated (the rest pass through unrotated).
    pub fn decoder_rope_dim(&self) -> u32 {
        (self.decoder_attention_head_dim as f32 * self.decoder_rope_dim_ratio) as u32
    }

    /// Rotary frequencies PER AXIS - `decoder_rope_dim() / (2*NUM_AXES)`.
    pub fn decoder_rope_freqs_per_axis(&self) -> u32 {
        self.decoder_rope_dim() / (2 * NUM_AXES)
    }

    pub fn decoder_ffn_dim(&self) -> u32 {
        self.decoder_dim() * self.decoder_ffn_mult
    }

    /// `proj_out`'s output width - one pixel block per token
    /// (`out_channels * patch_size_t * patch_size * patch_size`, `patch_size
    /// = spatial_compression_ratio`, `patch_size_t = temporal_compression_ratio`).
    pub fn decoder_out_patch(&self) -> u32 {
        self.out_channels * self.temporal_compression_ratio() * self.spatial_compression_ratio() * self.spatial_compression_ratio()
    }

    /// `ceil(clip_length / temporal_compression_ratio)` - latent frames one
    /// encoded clip produces before `token_drop`.
    pub fn tokens_chunk_size(&self) -> u32 {
        self.clip_length.div_ceil(self.temporal_compression_ratio())
    }

    /// `(-clip_length) % temporal_compression_ratio`, computed in unsigned
    /// arithmetic (Python's `%` on a negative left operand and a positive
    /// right one is already non-negative, in `[0, temporal_compression_ratio)`).
    pub fn frame_pre_padding(&self) -> u32 {
        let r = self.temporal_compression_ratio();
        (r - self.clip_length % r) % r
    }

    /// `(-token_drop) % tokens_chunk_size`, same unsigned-modulo reasoning as
    /// [`Self::frame_pre_padding`].
    pub fn token_overlap(&self) -> u32 {
        let n = self.tokens_chunk_size();
        (n - self.token_drop % n) % n
    }

    /// `max(token_overlap * temporal_compression_ratio - frame_pre_padding, 0)`.
    pub fn frame_overlap(&self) -> u32 {
        (self.token_overlap() * self.temporal_compression_ratio()).saturating_sub(self.frame_pre_padding())
    }

    /// `block_in_channels[i]` - `block_out_channels[0]` for stage 0, else
    /// `block_out_channels[i-1]` (`MiniMaxH3VideoEncoder3d.__init__`'s own
    /// `(block_out_channels[0],) + block_out_channels[:-1]`).
    fn block_in(&self, i: usize) -> u32 {
        if i == 0 {
            self.block_out_channels[0]
        } else {
            self.block_out_channels[i - 1]
        }
    }

    /// Every tensor [`encode_clip`]/[`decode_clip`] read, named after the
    /// reference module's own attribute paths (`state_dict()`'s own keys -
    /// see `tools/minimaxh3_video_vae_dump_reference.py`'s matching dump),
    /// so an import needs no renaming.
    pub fn tensor_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let mut m: Vec<(String, Vec<usize>)> = Vec::new();
        let put = |m: &mut Vec<(String, Vec<usize>)>, name: String, shape: Vec<usize>| m.push((name, shape));

        // ---- encoder ----
        put(&mut m, "encoder.conv_in.weight".into(), vec![self.block_out_channels[0] as usize, self.in_channels as usize, 3, 3, 3]);
        put(&mut m, "encoder.conv_in.bias".into(), vec![self.block_out_channels[0] as usize]);

        for i in 0..6usize {
            let cout = self.block_out_channels[i];
            for j in 0..self.layers_per_block as usize {
                let rin = if j == 0 { self.block_in(i) } else { cout };
                let p = format!("encoder.down_blocks.{i}.resnets.{j}");
                put(&mut m, format!("{p}.norm1.weight"), vec![rin as usize]);
                put(&mut m, format!("{p}.norm1.bias"), vec![rin as usize]);
                put(&mut m, format!("{p}.conv1.weight"), vec![cout as usize, rin as usize, 3, 3, 3]);
                put(&mut m, format!("{p}.conv1.bias"), vec![cout as usize]);
                put(&mut m, format!("{p}.norm2.weight"), vec![cout as usize]);
                put(&mut m, format!("{p}.norm2.bias"), vec![cout as usize]);
                put(&mut m, format!("{p}.conv2.weight"), vec![cout as usize, cout as usize, 3, 3, 3]);
                put(&mut m, format!("{p}.conv2.bias"), vec![cout as usize]);
                if rin != cout {
                    put(&mut m, format!("{p}.conv_shortcut.weight"), vec![cout as usize, rin as usize, 1, 1, 1]);
                    put(&mut m, format!("{p}.conv_shortcut.bias"), vec![cout as usize]);
                }
            }
            if self.temporal_downsample_factors[i] * self.spatial_downsample_factors[i] > 1 {
                let p = format!("encoder.down_blocks.{i}.downsamplers.0.conv");
                put(&mut m, format!("{p}.weight"), vec![cout as usize, cout as usize, 3, 3, 3]);
                put(&mut m, format!("{p}.bias"), vec![cout as usize]);
            }
        }
        put(&mut m, "encoder.norm_out.weight".into(), vec![self.block_out_channels[5] as usize]);
        put(&mut m, "encoder.norm_out.bias".into(), vec![self.block_out_channels[5] as usize]);
        put(&mut m, "encoder.conv_out.weight".into(), vec![2 * self.latent_channels as usize, self.block_out_channels[5] as usize, 3, 3, 3]);
        put(&mut m, "encoder.conv_out.bias".into(), vec![2 * self.latent_channels as usize]);

        put(&mut m, "quant_conv.weight".into(), vec![2 * self.latent_channels as usize, 2 * self.latent_channels as usize, 1, 1, 1]);
        put(&mut m, "quant_conv.bias".into(), vec![2 * self.latent_channels as usize]);
        put(&mut m, "post_quant_conv.weight".into(), vec![self.latent_channels as usize, self.latent_channels as usize, 1, 1, 1]);
        put(&mut m, "post_quant_conv.bias".into(), vec![self.latent_channels as usize]);

        // ---- decoder ----
        let dim = self.decoder_dim();
        let ffn = self.decoder_ffn_dim();
        let out_patch = self.decoder_out_patch();
        put(&mut m, "decoder.proj_in.weight".into(), vec![dim as usize, self.latent_channels as usize]);
        put(&mut m, "decoder.proj_in.bias".into(), vec![dim as usize]);
        put(&mut m, "decoder.register_tokens".into(), vec![1, self.decoder_num_register_tokens as usize, dim as usize]);
        for i in 0..self.decoder_num_layers as usize {
            let p = format!("decoder.transformer_blocks.{i}");
            put(&mut m, format!("{p}.norm1.weight"), vec![dim as usize]);
            for proj in ["to_q", "to_k", "to_v", "to_out.0"] {
                put(&mut m, format!("{p}.attn.{proj}.weight"), vec![dim as usize, dim as usize]);
                put(&mut m, format!("{p}.attn.{proj}.bias"), vec![dim as usize]);
            }
            put(&mut m, format!("{p}.scale1"), vec![dim as usize]);
            put(&mut m, format!("{p}.norm2.weight"), vec![dim as usize]);
            put(&mut m, format!("{p}.ff.net.0.proj.weight"), vec![2 * ffn as usize, dim as usize]);
            put(&mut m, format!("{p}.ff.net.0.proj.bias"), vec![2 * ffn as usize]);
            put(&mut m, format!("{p}.ff.net.2.weight"), vec![dim as usize, ffn as usize]);
            put(&mut m, format!("{p}.ff.net.2.bias"), vec![dim as usize]);
            put(&mut m, format!("{p}.scale2"), vec![dim as usize]);
        }
        put(&mut m, "decoder.norm_out.weight".into(), vec![dim as usize]);
        put(&mut m, "decoder.norm_out.bias".into(), vec![dim as usize]);
        put(&mut m, "decoder.proj_out.weight".into(), vec![out_patch as usize, dim as usize]);
        put(&mut m, "decoder.proj_out.bias".into(), vec![out_patch as usize]);
        m
    }
}

// ==================================================================
// Host-side [C,T,H,W] (channel-major) tensor reshaping helpers - encode/
// decode's outer clip-chunking is entirely a host-side concatenate/slice/
// blend over these, between device calls to encode_clip/decode_clip.
// ==================================================================

/// `x[:, t0:t0+n]` of a channel-major `[c,t,h,w]` host buffer.
fn slice_frames_host(x: &[f32], c: u32, t: u32, h: u32, w: u32, t0: u32, n: u32) -> Vec<f32> {
    assert!(t0 + n <= t, "slice_frames_host: [{t0},{}) of {t} frames", t0 + n);
    let hw = (h * w) as usize;
    let mut out = Vec::with_capacity(c as usize * n as usize * hw);
    for ci in 0..c as usize {
        let base = ci * t as usize * hw + t0 as usize * hw;
        out.extend_from_slice(&x[base..base + n as usize * hw]);
    }
    out
}

/// Append `x`'s own last frame `pad` times (`x[:, -1:].repeat(1,1,pad,1,1)`).
fn repeat_last_frame_host(x: &[f32], c: u32, t: u32, h: u32, w: u32, pad: u32) -> Vec<f32> {
    let hw = (h * w) as usize;
    let mut out = Vec::with_capacity(c as usize * (t + pad) as usize * hw);
    for ci in 0..c as usize {
        let base = ci * t as usize * hw;
        out.extend_from_slice(&x[base..base + t as usize * hw]);
        let last = &x[base + (t as usize - 1) * hw..base + t as usize * hw];
        for _ in 0..pad {
            out.extend_from_slice(last);
        }
    }
    out
}

/// `torch.cat([...], dim=2)` over a list of same-`(c,h,w)` channel-major
/// clips - the multi-chunk `_encode`/`_decode` concatenation.
fn concat_frames_host(chunks: &[(Vec<f32>, u32)], c: u32, h: u32, w: u32) -> (Vec<f32>, u32) {
    let ttot: u32 = chunks.iter().map(|(_, t)| *t).sum();
    let hw = (h * w) as usize;
    let mut out = vec![0f32; c as usize * ttot as usize * hw];
    for ci in 0..c as usize {
        let mut toff = 0usize;
        for (data, tn) in chunks {
            let tn = *tn as usize;
            let src = ci * tn * hw;
            let dst = ci * ttot as usize * hw + toff * hw;
            out[dst..dst + tn * hw].copy_from_slice(&data[src..src + tn * hw]);
            toff += tn;
        }
    }
    (out, ttot)
}

/// `AutoencoderKLMiniMaxH3._blend(a, b, blend_extent, dim=-3)`: returns a
/// buffer THE SAME LENGTH AS `b` (`bt` frames), with `b`'s own leading
/// `min(blend_extent,at,bt)` frames replaced by a linear cross-fade of `a`'s
/// TRAILING frames and `b`'s own leading frames; the rest of `b` is
/// unchanged. NOT a concatenation that grows the frame count - `_decode`
/// uses this to smooth a chunk boundary within the new chunk's own frame
/// range, not to splice `a`'s frames in front of `b`.
fn blend_frames_host(a: &[f32], at: u32, b: &[f32], bt: u32, c: u32, h: u32, w: u32, blend_extent: u32) -> Vec<f32> {
    let be = blend_extent.min(at).min(bt);
    let hw = (h * w) as usize;
    let mut out = b.to_vec();
    for ci in 0..c as usize {
        let a_base = ci * at as usize * hw;
        let b_base = ci * bt as usize * hw;
        for k in 0..be as usize {
            let wa = 1.0 - (k as f32) / (be as f32);
            let wb = (k as f32) / (be as f32);
            let a_off = a_base + (at as usize - be as usize + k) * hw;
            let b_off = b_base + k * hw;
            for p in 0..hw {
                out[b_off + p] = a[a_off + p] * wa + b[b_off + p] * wb;
            }
        }
    }
    out
}

// ==================================================================
// Spatial tiling - `_split_tiles` / `_blend` / `_stitch_tiles` and the
// tiled halves of `_decode_clip` / `_encode_clip`.
//
// The reference lays fixed-size tiles over the frame and blends the
// overlaps. It matters that this is not a memory optimization that could be
// skipped: the ViT decoder's rotary coordinates are normalized across
// whatever grid it is handed, so the tile size is part of the MODEL, and
// handing it a whole large frame is a different (wrong) computation, not a
// less-chunked version of the same one. See this module's own doc.
// ==================================================================

/// A channel-major `[c,t,h,w]` host tensor - the spatial tiling's working
/// unit. `t` is carried through untouched: tiling is purely spatial, the
/// temporal chunking is the separate outer `_encode`/`_decode` machinery.
#[derive(Clone)]
struct Tile {
    data: Vec<f32>,
    c: u32,
    t: u32,
    h: u32,
    w: u32,
}

impl Tile {
    fn new(data: Vec<f32>, c: u32, t: u32, h: u32, w: u32) -> Tile {
        assert_eq!(data.len(), (c * t * h * w) as usize, "Tile::new: {} values for [{c},{t},{h},{w}]", data.len());
        Tile { data, c, t, h, w }
    }
}

/// `AutoencoderKLMiniMaxH3._split_tiles(length, tile_size, min_overlap)`,
/// returning `(tile_start_indices, tile_lengths, overlaps)`.
///
/// Three properties of this are easy to get wrong from the name alone, so
/// they are spelled out:
///
/// * **There is no partial edge tile.** Every returned length is exactly
///   `tile_size`; the slack is absorbed by WIDENING the overlaps, not by
///   truncating the last tile. So the last tile ends exactly at `length`
///   and the decoder sees an identically-shaped grid for every tile - which
///   is the whole point, given its rotary coordinates are normalized to the
///   grid it is given.
/// * **The tile COUNT is grown until the union can cover `length` at the
///   minimum overlap**, so it can exceed `ceil(length / tile_size)`. At
///   512 pixels with 256-pixel tiles that gives THREE tiles, not two.
/// * **The slack is distributed round-robin in whole `ratio` steps**
///   (`remaining // ratio` increments of `ratio`), which is what keeps every
///   tile boundary latent-aligned. The division truncates; the reference
///   does not correct for a remainder, and neither does this.
///
/// The degenerate `tile_size >= length` case returns ONE tile of the full
/// `length` (not of `tile_size`) with no overlaps - the reason tiling is a
/// no-op at or below 256 pixels rather than a different computation.
fn split_tiles(length: u32, tile_size: u32, min_overlap: u32, ratio: u32) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    if tile_size >= length {
        return (vec![0], vec![length], Vec::new());
    }
    let mut num_tiles = length.div_ceil(tile_size);
    while (tile_size as i64) * (num_tiles as i64) - (min_overlap as i64) * (num_tiles as i64 - 1) - (length as i64) < 0 {
        num_tiles += 1;
    }
    let n_ov = (num_tiles - 1) as usize;
    let mut overlaps = vec![min_overlap; n_ov];
    let remaining = tile_size * num_tiles - overlaps.iter().sum::<u32>() - length;
    for i in 0..(remaining / ratio) {
        overlaps[(i as usize) % n_ov] += ratio;
    }
    let mut starts = vec![0u32];
    for &ov in overlaps.iter() {
        starts.push(starts[starts.len() - 1] + tile_size - ov);
    }
    (starts, vec![tile_size; num_tiles as usize], overlaps)
}

/// `_blend(a, b, blend_extent, dim=-2)` - the HEIGHT axis. Returns a tensor
/// with `b`'s OWN shape, whose leading `min(a.h, b.h, blend_extent)` rows
/// are a linear cross-fade from `a`'s TRAILING rows into `b`'s leading rows.
/// Same "overwrite `b`'s head, never grow" semantics as
/// [`blend_frames_host`]'s temporal version.
fn blend_h(a: &Tile, b: &Tile, blend_extent: u32) -> Tile {
    let be = blend_extent.min(a.h).min(b.h);
    let mut out = b.data.clone();
    for ci in 0..a.c {
        for ti in 0..a.t {
            for k in 0..be {
                let wa = 1.0 - (k as f32) / (be as f32);
                let wb = (k as f32) / (be as f32);
                let a_row = (((ci * a.t + ti) * a.h + (a.h - be + k)) * a.w) as usize;
                let b_row = (((ci * b.t + ti) * b.h + k) * b.w) as usize;
                for x in 0..b.w as usize {
                    out[b_row + x] = a.data[a_row + x] * wa + b.data[b_row + x] * wb;
                }
            }
        }
    }
    Tile::new(out, b.c, b.t, b.h, b.w)
}

/// `_blend(a, b, blend_extent, dim=-1)` - the WIDTH axis, same semantics as
/// [`blend_h`].
fn blend_w(a: &Tile, b: &Tile, blend_extent: u32) -> Tile {
    let be = blend_extent.min(a.w).min(b.w);
    let mut out = b.data.clone();
    for ci in 0..a.c {
        for ti in 0..a.t {
            for hi in 0..b.h {
                let a_row = (((ci * a.t + ti) * a.h + hi) * a.w) as usize;
                let b_row = (((ci * b.t + ti) * b.h + hi) * b.w) as usize;
                for k in 0..be as usize {
                    let wa = 1.0 - (k as f32) / (be as f32);
                    let wb = (k as f32) / (be as f32);
                    out[b_row + k] = a.data[a_row + (a.w - be) as usize + k] * wa + b.data[b_row + k] * wb;
                }
            }
        }
    }
    Tile::new(out, b.c, b.t, b.h, b.w)
}

/// `x[..., :keep, :]` - keep the leading `keep` rows.
fn trim_h(x: &Tile, keep: u32) -> Tile {
    assert!(keep <= x.h, "trim_h: keep {keep} > h {}", x.h);
    let mut out = Vec::with_capacity((x.c * x.t * keep * x.w) as usize);
    for ci in 0..x.c {
        for ti in 0..x.t {
            let base = (((ci * x.t + ti) * x.h) * x.w) as usize;
            out.extend_from_slice(&x.data[base..base + (keep * x.w) as usize]);
        }
    }
    Tile::new(out, x.c, x.t, keep, x.w)
}

/// `x[..., :, :keep]` - keep the leading `keep` columns.
fn trim_w(x: &Tile, keep: u32) -> Tile {
    assert!(keep <= x.w, "trim_w: keep {keep} > w {}", x.w);
    let mut out = Vec::with_capacity((x.c * x.t * x.h * keep) as usize);
    for ci in 0..x.c {
        for ti in 0..x.t {
            for hi in 0..x.h {
                let base = (((ci * x.t + ti) * x.h + hi) * x.w) as usize;
                out.extend_from_slice(&x.data[base..base + keep as usize]);
            }
        }
    }
    Tile::new(out, x.c, x.t, x.h, keep)
}

/// `torch.cat(row, dim=-1)` over same-`(c,t,h)` tiles.
fn cat_w(row: &[Tile]) -> Tile {
    let first = &row[0];
    let wtot: u32 = row.iter().map(|t| t.w).sum();
    let mut out = vec![0f32; (first.c * first.t * first.h * wtot) as usize];
    for ci in 0..first.c {
        for ti in 0..first.t {
            for hi in 0..first.h {
                let mut woff = 0u32;
                for tile in row {
                    let src = (((ci * tile.t + ti) * tile.h + hi) * tile.w) as usize;
                    let dst = (((ci * first.t + ti) * first.h + hi) * wtot + woff) as usize;
                    out[dst..dst + tile.w as usize].copy_from_slice(&tile.data[src..src + tile.w as usize]);
                    woff += tile.w;
                }
            }
        }
    }
    Tile::new(out, first.c, first.t, first.h, wtot)
}

/// `torch.cat(rows, dim=-2)` over same-`(c,t,w)` tiles.
fn cat_h(rows: &[Tile]) -> Tile {
    let first = &rows[0];
    let htot: u32 = rows.iter().map(|t| t.h).sum();
    let mut out = vec![0f32; (first.c * first.t * htot * first.w) as usize];
    for ci in 0..first.c {
        for ti in 0..first.t {
            let mut hoff = 0u32;
            for tile in rows {
                let src = (((ci * tile.t + ti) * tile.h) * tile.w) as usize;
                let dst = (((ci * first.t + ti) * htot + hoff) * first.w) as usize;
                let n = (tile.h * tile.w) as usize;
                out[dst..dst + n].copy_from_slice(&tile.data[src..src + n]);
                hoff += tile.h;
            }
        }
    }
    Tile::new(out, first.c, first.t, htot, first.w)
}

/// `_stitch_tiles(tiles, height_overlaps, width_overlaps)`.
///
/// The detail worth stating, because the natural implementation gets it
/// wrong: **both blends read the ORIGINAL neighbouring tiles**, not the
/// already-blended-and-trimmed results being accumulated. `tiles[i-1][j]`
/// and `tiles[i][j-1]` are indexed straight out of the input grid, so in a
/// corner region (where a tile overlaps both the tile above and the tile to
/// its left) the vertical cross-fade runs first against the RAW tile above,
/// and the horizontal cross-fade then runs against the RAW tile to the
/// left, which has itself had no vertical blend applied. That asymmetry is
/// the reference's behaviour and is reproduced here rather than "improved".
fn stitch_tiles(tiles: &[Vec<Tile>], height_overlaps: &[u32], width_overlaps: &[u32]) -> Tile {
    let mut result_rows: Vec<Tile> = Vec::with_capacity(tiles.len());
    for (i, row) in tiles.iter().enumerate() {
        let mut result_row: Vec<Tile> = Vec::with_capacity(row.len());
        for (j, tile) in row.iter().enumerate() {
            let mut cur = tile.clone();
            if i > 0 {
                cur = blend_h(&tiles[i - 1][j], &cur, height_overlaps[i - 1]);
            }
            if j > 0 {
                cur = blend_w(&row[j - 1], &cur, width_overlaps[j - 1]);
            }
            if i < tiles.len() - 1 {
                cur = trim_h(&cur, cur.h - height_overlaps[i]);
            }
            if j < row.len() - 1 {
                cur = trim_w(&cur, cur.w - width_overlaps[j]);
            }
            result_row.push(cur);
        }
        result_rows.push(cat_w(&result_row));
    }
    cat_h(&result_rows)
}

/// `x[..., y0:y0+th, x0:x0+tw]` of a channel-major `[c,t,h,w]` host buffer.
fn spatial_window_host(x: &[f32], c: u32, t: u32, h: u32, w: u32, y0: u32, th: u32, x0: u32, tw: u32) -> Vec<f32> {
    assert!(y0 + th <= h && x0 + tw <= w, "spatial_window_host: [{y0},{}) x [{x0},{}) of {h}x{w}", y0 + th, x0 + tw);
    let mut out = Vec::with_capacity((c * t * th * tw) as usize);
    for ci in 0..c {
        for ti in 0..t {
            for hi in y0..y0 + th {
                let base = (((ci * t + ti) * h + hi) * w + x0) as usize;
                out.extend_from_slice(&x[base..base + tw as usize]);
            }
        }
    }
    out
}

// ==================================================================
// Device ops shared by the encoder and the decoder
// ==================================================================

fn get<'a>(t: &'a Tensors, name: &str) -> &'a (Vec<usize>, Vec<f32>) {
    t.get(name).unwrap_or_else(|| panic!("minimaxh3 video_vae: missing tensor {name:?}"))
}

fn upload(cx: &Ctx, t: &Tensors, name: &str) -> DeviceBuffer {
    cx.gpu.storage_init(name, &get(t, name).1)
}

/// One `conv3d` dispatch: zero-pad `pt` (one-sided, causal) / `ph`,`pw`
/// (symmetric) - the kernel's own built-in padding, see this module's own
/// doc for why the causal temporal pad needs no doubling here.
#[allow(clippy::too_many_arguments)]
fn conv3d(cx: &Ctx, wgt: &DeviceBuffer, bias: &DeviceBuffer, x: &DeviceBuffer, cin: u32, t: u32, h: u32, w: u32, cout: u32, kt: u32, kh: u32, kw: u32, st: u32, sh: u32, sw: u32, pt: u32, ph: u32, pw: u32) -> (DeviceBuffer, u32, u32, u32) {
    let to = (t + pt - kt) / st + 1;
    let ho = (h + 2 * ph - kh) / sh + 1;
    let wo = (w + 2 * pw - kw) / sw + 1;
    let y = cx.gpu.storage((cout * to * ho * wo) as u64);
    let params = [1, cin, t, h, w, cout, kt, kh, kw, st, sh, sw, pt, ph, pw, 1, to, ho, wo];
    cx.gpu.submit(&[], &[cx.gpu.step(K_CONV3D, &[x, wgt, bias, &y], &params, cout * to * ho * wo)]);
    (y, to, ho, wo)
}

/// Reflect-pad the H,W axes of a channel-major `[c,t,h,w]` device buffer by
/// `(l,r,top,bot)` via [`kernels::PAD2D_REFLECT`] (`img = c*t`, this
/// kernel's own "collapsed batch" convention - see its own doc).
fn pad_reflect_hw(cx: &Ctx, x: &DeviceBuffer, c: u32, t: u32, h: u32, w: u32, l: u32, r: u32, top: u32, bot: u32) -> (DeviceBuffer, u32, u32) {
    let hp = h + top + bot;
    let wp = w + l + r;
    let total = c * t * hp * wp;
    let y = cx.gpu.storage(total as u64);
    cx.gpu.submit(&[], &[cx.gpu.step(K_PAD2D_REFLECT, &[x, &y], &[total, h, w, l, r, top, bot], total)]);
    (y, hp, wp)
}

/// `MiniMaxH3VideoCausalConv3d` at spatial padding `sp` (reflect, symmetric,
/// applied before the conv) and temporal padding `tp` (zero, the causal
/// `conv3d.wgsl` built-in, one-sided).
#[allow(clippy::too_many_arguments)]
fn causal_conv3d(cx: &Ctx, tensors: &Tensors, prefix: &str, cin: u32, t: u32, h: u32, w: u32, cout: u32, k: u32, stride: (u32, u32, u32), sp: u32, tp: u32, x: &DeviceBuffer) -> (DeviceBuffer, u32, u32, u32) {
    let wgt = upload(cx, tensors, &format!("{prefix}.weight"));
    let bias = upload(cx, tensors, &format!("{prefix}.bias"));
    let (xin, hh, ww) = if sp > 0 { pad_reflect_hw(cx, x, cin, t, h, w, sp, sp, sp, sp) } else { (x.clone(), h, w) };
    conv3d(cx, &wgt, &bias, &xin, cin, t, hh, ww, cout, k, k, k, stride.0, stride.1, stride.2, tp, 0, 0)
}

/// `MiniMaxH3VideoDownsample3d`: asymmetric reflect pad `(0,1,0,1)` (right/
/// bottom only) when `spatial_stride==2`, then a causal `k=3` conv with
/// temporal padding 2 (always - the constructor's own fixed
/// `temporal_padding=2`) and NO further spatial padding (already applied).
fn downsample_conv3d(cx: &Ctx, tensors: &Tensors, prefix: &str, c: u32, t: u32, h: u32, w: u32, temporal_stride: u32, spatial_stride: u32, x: &DeviceBuffer) -> (DeviceBuffer, u32, u32, u32) {
    let wgt = upload(cx, tensors, &format!("{prefix}.weight"));
    let bias = upload(cx, tensors, &format!("{prefix}.bias"));
    let (xin, hh, ww) = if spatial_stride == 2 { pad_reflect_hw(cx, x, c, t, h, w, 0, 1, 0, 1) } else { (x.clone(), h, w) };
    conv3d(cx, &wgt, &bias, &xin, c, t, hh, ww, c, 3, 3, 3, temporal_stride, spatial_stride, spatial_stride, 2, 0, 0)
}

fn silu3d(cx: &Ctx, x: &DeviceBuffer, n: u32) -> DeviceBuffer {
    let y = cx.gpu.storage(n as u64);
    cx.gpu.submit(&[], &[cx.gpu.step(K_SILU, &[x, &y], &[n], n)]);
    y
}

fn add3d(cx: &Ctx, a: &DeviceBuffer, b: &DeviceBuffer, n: u32) -> DeviceBuffer {
    let y = cx.gpu.storage(n as u64);
    cx.gpu.submit(&[], &[cx.gpu.step(K_ADD2, &[a, b, &y], &[n], n)]);
    y
}

/// One frame `x[:, t0:t0+1]` of a channel-major `[c,t,h,w]` device buffer,
/// as a contiguous `[c,h,w]` buffer (`concat_split`'s own "collapsed
/// [N,total,inner]" view, `N=c, total=t, inner=h*w` - `vae::blocks3d::
/// Builder3d::time_slice`'s exact trick, reimplemented eagerly here since
/// this module does not share that builder's device/kernel-slot context).
fn frame_slice(cx: &Ctx, x: &DeviceBuffer, c: u32, t: u32, h: u32, w: u32, t0: u32) -> DeviceBuffer {
    let y = cx.gpu.storage((c * h * w) as u64);
    cx.gpu.submit(&[], &[cx.gpu.step(K_CONCAT_SPLIT, &[x, &y], &[c, t, 1, t0, h, w], c * h * w)]);
    y
}

/// Write a `[c,h,w]` frame into `dst[:, t0:t0+1]` of a `[c,t_tot,h,w]`
/// buffer - `frame_slice`'s inverse (`vae::blocks3d::Builder3d::time_place`'s
/// exact trick).
fn frame_place(cx: &Ctx, dst: &DeviceBuffer, src: &DeviceBuffer, c: u32, t_tot: u32, h: u32, w: u32, t0: u32) {
    cx.gpu.submit(&[], &[cx.gpu.step(K_CHAN_PLACE, &[src, dst], &[c, t_tot, 1, t0, h, w], c * h * w)]);
}

/// `MiniMaxH3VideoGroupNorm`: per-frame (t-isolated) GroupNorm, by looping
/// over `t` frames and running the existing plain-image GroupNorm kernels
/// (`gn_part`/`gn_stats2`/`gn_apply`, already `N`-generic) at `N=1` on each
/// genuinely contiguous frame slice - see this module's own doc for why this
/// reshape is needed rather than dispatching `gn_*` directly over `[c,t,h,w]`.
fn group_norm_t_isolated(cx: &Ctx, tensors: &Tensors, prefix: &str, c: u32, t: u32, h: u32, w: u32, groups: u32, eps: f32, x: &DeviceBuffer) -> DeviceBuffer {
    let (_, gamma) = get(tensors, &format!("{prefix}.weight"));
    let (_, beta) = get(tensors, &format!("{prefix}.bias"));
    let mut gb = gamma.clone();
    gb.extend_from_slice(beta);
    let gbdev = cx.gpu.storage_init(&format!("{prefix}.gb"), &gb);

    let y = cx.gpu.storage((c * t * h * w) as u64);
    for t0 in 0..t {
        let frame = frame_slice(cx, x, c, t, h, w, t0);
        let part = cx.gpu.storage((2 * groups * GN_P) as u64);
        cx.gpu.submit(&[], &[cx.gpu.step(K_GN_PART, &[&frame, &part], &[1, c, h, w, groups, GN_P], groups * GN_P)]);
        let stats = cx.gpu.storage((2 * groups) as u64);
        cx.gpu.submit(&[], &[cx.gpu.step(K_GN_STATS2, &[&part, &stats], &[1, c, h, w, groups, GN_P, f(eps)], groups)]);
        let normed = cx.gpu.storage((c * h * w) as u64);
        cx.gpu.submit(&[], &[cx.gpu.step(K_GN_APPLY, &[&frame, &stats, &gbdev, &normed], &[1, c, h, w, groups], c * h * w)]);
        frame_place(cx, &y, &normed, c, t, h, w, t0);
    }
    y
}

/// `MiniMaxH3VideoResnetBlock3d.forward`: `silu(norm1(x)) -> conv1 ->
/// silu(norm2) -> conv2`, residual (through `conv_shortcut` when the channel
/// count changes, the raw input otherwise).
#[allow(clippy::too_many_arguments)]
fn resnet_block3d(cx: &Ctx, tensors: &Tensors, prefix: &str, cin: u32, cout: u32, t: u32, h: u32, w: u32, groups: u32, eps: f32, x: &DeviceBuffer) -> (DeviceBuffer, u32, u32, u32) {
    let n1 = group_norm_t_isolated(cx, tensors, &format!("{prefix}.norm1"), cin, t, h, w, groups, eps, x);
    let s1 = silu3d(cx, &n1, cin * t * h * w);
    let (c1, t1, h1, w1) = causal_conv3d(cx, tensors, &format!("{prefix}.conv1"), cin, t, h, w, cout, 3, (1, 1, 1), 1, 2, &s1);
    let n2 = group_norm_t_isolated(cx, tensors, &format!("{prefix}.norm2"), cout, t1, h1, w1, groups, eps, &c1);
    let s2 = silu3d(cx, &n2, cout * t1 * h1 * w1);
    let (c2, t2, h2, w2) = causal_conv3d(cx, tensors, &format!("{prefix}.conv2"), cout, t1, h1, w1, cout, 3, (1, 1, 1), 1, 2, &s2);
    let residual = if cin != cout { causal_conv3d(cx, tensors, &format!("{prefix}.conv_shortcut"), cin, t, h, w, cout, 1, (1, 1, 1), 0, 0, x).0 } else { x.clone() };
    (add3d(cx, &residual, &c2, cout * t2 * h2 * w2), t2, h2, w2)
}

// ==================================================================
// Encoder: MiniMaxH3VideoEncoder3d + quant_conv, one clip (untiled)
// ==================================================================

/// `_encode_clip`'s `use_tiling=False` body: `quant_conv(encoder(x))` over
/// ONE pixel clip, `x` a channel-major `[in_channels,t,h,w]` host buffer.
/// Returns the flat moments buffer `[2*latent_channels,t',h',w']` plus its
/// shape. Takes an already-open [`Ctx`] so a tiled encode opens ONE device
/// for the whole tile grid.
fn encode_clip_untiled(cx: &Ctx, cfg: &VideoVaeConfig, tensors: &Tensors, x: &[f32], t: u32, h: u32, w: u32) -> (Vec<f32>, u32, u32, u32) {
    assert_eq!(x.len(), (cfg.in_channels * t * h * w) as usize, "encode_clip: input length {} != in_channels*t*h*w", x.len());
    let xin = cx.gpu.storage_init("video_vae.pixels", x);

    let (mut cur, mut ct, mut ch, mut cw) = causal_conv3d(cx, tensors, "encoder.conv_in", cfg.in_channels, t, h, w, cfg.block_out_channels[0], 3, (1, 1, 1), 1, 2, &xin);
    let mut cur_c = cfg.block_out_channels[0];

    for i in 0..6usize {
        let cout = cfg.block_out_channels[i];
        for j in 0..cfg.layers_per_block as usize {
            let p = format!("encoder.down_blocks.{i}.resnets.{j}");
            let (y, t2, h2, w2) = resnet_block3d(cx, tensors, &p, cur_c, cout, ct, ch, cw, cfg.norm_num_groups, cfg.norm_eps, &cur);
            cur = y;
            ct = t2;
            ch = h2;
            cw = w2;
            cur_c = cout;
            // Same reclaim-point requirement as the ViT decoder's per-layer
            // loop (see `decode_clip_untiled`'s own comment): each resnet
            // block uploads its own fresh weight buffers, never reused
            // across blocks, so they must be proven finished with before
            // the next block's buffers are allocated.
            cx.gpu.poll_wait();
        }
        if cfg.temporal_downsample_factors[i] * cfg.spatial_downsample_factors[i] > 1 {
            let p = format!("encoder.down_blocks.{i}.downsamplers.0.conv");
            let (y, t2, h2, w2) = downsample_conv3d(cx, tensors, &p, cur_c, ct, ch, cw, cfg.temporal_downsample_factors[i], cfg.spatial_downsample_factors[i], &cur);
            cur = y;
            ct = t2;
            ch = h2;
            cw = w2;
            cx.gpu.poll_wait();
        }
    }

    let n = group_norm_t_isolated(cx, tensors, "encoder.norm_out", cur_c, ct, ch, cw, cfg.norm_num_groups, cfg.norm_eps, &cur);
    let s = silu3d(cx, &n, cur_c * ct * ch * cw);
    let (out, t3, h3, w3) = causal_conv3d(cx, tensors, "encoder.conv_out", cur_c, ct, ch, cw, 2 * cfg.latent_channels, 3, (1, 1, 1), 1, 2, &s);
    let (moments, t4, h4, w4) = causal_conv3d(cx, tensors, "quant_conv", 2 * cfg.latent_channels, t3, h3, w3, 2 * cfg.latent_channels, 1, (1, 1, 1), 0, 0, &out);

    let flat = cx.gpu.read(&moments, (2 * cfg.latent_channels * t4 * h4 * w4) as usize);
    (flat, t4, h4, w4)
}

/// `_encode_clip`: one temporal clip, spatially tiled when
/// `cfg.use_tiling` (the reference's own shipped default).
///
/// Tiles are laid out directly in pixel space here - unlike [`decode_clip`],
/// where they are laid out in pixel space and then mapped back onto the
/// latent grid - and the overlaps handed to [`stitch_tiles`] are converted
/// to LATENT units (`overlap / spatial_compression_ratio`) because what is
/// being stitched is the encoder's latent output, not pixels.
pub fn encode_clip(cfg: &VideoVaeConfig, tensors: &Tensors, device: Option<&str>, x: &[f32], t: u32, h: u32, w: u32) -> (Vec<f32>, u32, u32, u32) {
    let cx = Ctx::new(device);
    encode_clip_in(&cx, cfg, tensors, x, t, h, w)
}

fn encode_clip_in(cx: &Ctx, cfg: &VideoVaeConfig, tensors: &Tensors, x: &[f32], t: u32, h: u32, w: u32) -> (Vec<f32>, u32, u32, u32) {
    if !cfg.use_tiling {
        return encode_clip_untiled(cx, cfg, tensors, x, t, h, w);
    }
    let ratio = cfg.spatial_compression_ratio();
    let (y_idx, y_len, y_ov) = split_tiles(h, cfg.tile_sample_min_height, cfg.tile_sample_min_overlap_height, ratio);
    let (x_idx, x_len, x_ov) = split_tiles(w, cfg.tile_sample_min_width, cfg.tile_sample_min_overlap_width, ratio);
    // One full-size tile is the untiled computation exactly (`_split_tiles`
    // returns `[0], [length], []`), so take the cheap path rather than
    // round-tripping the whole clip through a one-element stitch.
    if y_idx.len() == 1 && x_idx.len() == 1 {
        return encode_clip_untiled(cx, cfg, tensors, x, t, h, w);
    }

    let mut rows: Vec<Vec<Tile>> = Vec::with_capacity(y_idx.len());
    for (&i_pos, &i_len) in y_idx.iter().zip(y_len.iter()) {
        let mut row: Vec<Tile> = Vec::with_capacity(x_idx.len());
        for (&j_pos, &j_len) in x_idx.iter().zip(x_len.iter()) {
            let sub = spatial_window_host(x, cfg.in_channels, t, h, w, i_pos, i_len, j_pos, j_len);
            let (m, mt, mh, mw) = encode_clip_untiled(cx, cfg, tensors, &sub, t, i_len, j_len);
            row.push(Tile::new(m, 2 * cfg.latent_channels, mt, mh, mw));
            // Belt-and-suspenders alongside `encode_clip_untiled`'s own
            // per-block poll_wait: reclaim this tile's buffers before the
            // next tile's encoder pass starts allocating.
            cx.gpu.poll_wait();
        }
        rows.push(row);
    }
    let lat_y_ov: Vec<u32> = y_ov.iter().map(|o| o / ratio).collect();
    let lat_x_ov: Vec<u32> = x_ov.iter().map(|o| o / ratio).collect();
    let out = stitch_tiles(&rows, &lat_y_ov, &lat_x_ov);
    (out.data, out.t, out.h, out.w)
}

/// `_encode`: chunk `x` into `clip_length`-pixel-frame clips (repeat-padding
/// the tail up to a whole multiple when needed), encode each independently
/// (NO cross-chunk state - see this module's own doc), concatenate the
/// moments along T, then drop `token_drop` trailing latent frames once from
/// the WHOLE concatenated sequence. `num_frames==1` bypasses chunking
/// entirely (`_encode`'s own documented special case - repeating a lone
/// image to fill a clip is not this model's training-time conditioning).
pub fn encode(cfg: &VideoVaeConfig, tensors: &Tensors, device: Option<&str>, x: &[f32], num_frames: u32, h: u32, w: u32) -> (Vec<f32>, u32, u32, u32) {
    let cx = Ctx::new(device);
    if num_frames == 1 {
        return encode_clip_in(&cx, cfg, tensors, x, 1, h, w);
    }
    let clip_length = cfg.clip_length;
    let pad = (clip_length - num_frames % clip_length) % clip_length;
    let (xpad, total) = if pad > 0 { (repeat_last_frame_host(x, cfg.in_channels, num_frames, h, w, pad), num_frames + pad) } else { (x.to_vec(), num_frames) };

    let mut chunks: Vec<(Vec<f32>, u32)> = Vec::new();
    let mut oh = 0u32;
    let mut ow = 0u32;
    let mut oc = 0u32;
    for i in 0..(total / clip_length) {
        let clip = slice_frames_host(&xpad, cfg.in_channels, total, h, w, i * clip_length, clip_length);
        let (moments, ct, ch, cw) = encode_clip_in(&cx, cfg, tensors, &clip, clip_length, h, w);
        oh = ch;
        ow = cw;
        oc = 2 * cfg.latent_channels;
        chunks.push((moments, ct));
        // Same reclaim-point requirement as `decode`'s own chunk loop (see
        // its comment) - each clip's encoder activations must be proven
        // finished before the next clip's are allocated, or they pile up
        // across chunks on the wgpu backend.
        cx.gpu.poll_wait();
    }
    let (mut moments, mut t) = concat_frames_host(&chunks, oc, oh, ow);
    if cfg.token_drop > 0 {
        assert!(cfg.token_drop < t, "encode: token_drop {} >= encoded {t} latent frames", cfg.token_drop);
        moments = slice_frames_host(&moments, oc, t, oh, ow, 0, t - cfg.token_drop);
        t -= cfg.token_drop;
    }
    (moments, t, oh, ow)
}

/// `DiagonalGaussianDistribution.mode()` for `moments` as a channel-major
/// `[2*latent_channels,t,h,w]` buffer: `chunk(2,dim=1)`'s first half (the
/// channel axis is OUTERMOST in this layout, so this is a contiguous prefix
/// slice, not a strided gather).
pub fn posterior_mode(moments: &[f32], latent_channels: u32, t: u32, h: u32, w: u32) -> Vec<f32> {
    let n = (latent_channels * t * h * w) as usize;
    assert_eq!(moments.len(), 2 * n, "posterior_mode: moments length {} != 2*latent_channels*t*h*w", moments.len());
    moments[..n].to_vec()
}

// ==================================================================
// Decoder: MiniMaxH3VideoViTDecoder3d + post_quant_conv, one clip (untiled)
// ==================================================================

fn linear(cx: &Ctx, x: &DeviceBuffer, w: &DeviceBuffer, bias: Option<&DeviceBuffer>, m: u32, k: u32, n: u32) -> DeviceBuffer {
    let y = cx.gpu.storage((m * n) as u64);
    let (mm, threads) = model::block::pick_gemm(m as usize, n as usize, K_MATMUL, K_MATMUL_REG3, false);
    cx.gpu.submit(&[], &[cx.gpu.step(mm, &[x, w, &y], &[m, k, n], threads)]);
    if let Some(b) = bias {
        cx.gpu.submit(&[], &[cx.gpu.step(K_BIAS_ADD, &[&y, b], &[m, n], m * n)]);
    }
    y
}

fn rmsnorm(cx: &Ctx, x: &DeviceBuffer, w: &DeviceBuffer, rows: u32, d: u32, eps: f32) -> DeviceBuffer {
    let y = cx.gpu.storage((rows * d) as u64);
    cx.gpu.submit(&[], &[cx.gpu.step(K_RMSNORM_EPS, &[x, w, &y], &[d, rows, f(eps)], rows)]);
    y
}

fn layernorm_rows(cx: &Ctx, x: &DeviceBuffer, gamma: &DeviceBuffer, beta: &DeviceBuffer, rows: u32, d: u32, eps: f32) -> DeviceBuffer {
    let y = cx.gpu.storage((rows * d) as u64);
    cx.gpu.submit(&[], &[cx.gpu.step(K_LAYERNORM_ROWS, &[x, gamma, beta, &y], &[d, rows, f(eps)], rows * 64)]);
    y
}

/// In-place partial RoPE on `buf` (`[rows, heads*head_dim]`), against
/// precomputed `[rows, half]` `cos`/`sin` tables. Rotates the leading
/// `2*half` channels of every head; the rest pass through unchanged.
fn rope_partial(cx: &Ctx, buf: &DeviceBuffer, cos: &DeviceBuffer, sin: &DeviceBuffer, rows: u32, heads: u32, head_dim: u32, half: u32) {
    let total = rows * heads * half;
    let row_stride = heads * head_dim;
    cx.gpu.submit(&[], &[cx.gpu.step(K_ROPE2D_PARTIAL, &[buf, cos, sin], &[rows, heads, half, row_stride, 0, rows, f(1.0), head_dim], total)]);
}

/// Full (bidirectional, unmasked) self-attention - `dispatch_attention_fn`
/// with `attn_mask=None` (`MiniMaxH3VideoAttnProcessor`'s own shape).
fn attention(cx: &Ctx, q: &DeviceBuffer, k: &DeviceBuffer, v: &DeviceBuffer, seq_len: u32, heads: u32, head_dim: u32) -> DeviceBuffer {
    let inner = heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    // `heads * seq_len * seq_len` is the score-matrix element count, and it
    // does not fit in a u32 for a long sequence: at the real config's 32 heads
    // it wraps past seq_len 11585. Computed in u64 and checked, so an
    // oversized decode fails with this message instead of silently allocating
    // a wrapped-around buffer and dispatching a wrapped-around thread count.
    //
    // With `use_tiling` on - the reference's own shipped default, and this
    // port's - every tile is a 16x16 LATENT window whatever the canvas, so
    // the sequence is `t * 256 + num_register_tokens + 1` (1797 at the real
    // decoder's 7-latent-frame chunk) and this ceiling is three orders of
    // magnitude away. Reaching it therefore means tiling was bypassed
    // (`use_tiling = false`) on a frame the reference would have split, not
    // that the limit is too low: the fix is to leave tiling on, never to
    // widen this.
    let scores_len = heads as u64 * seq_len as u64 * seq_len as u64;
    assert!(
        scores_len <= u32::MAX as u64,
        "video_vae::attention: {heads} heads x {seq_len}^2 attention scores ({scores_len} elements) overflows the u32 \
         dispatch space. Only the UNTILED decode path can build a sequence this long - with `use_tiling` on, the \
         reference's own default, `_split_tiles` bounds every tile to a 16x16 latent grid. Set `use_tiling` rather \
         than raising this limit."
    );
    let scores = cx.gpu.storage(scores_len);
    cx.gpu.submit(&[], &[cx.gpu.step(K_ATTN_SCORES_QK, &[q, k, &scores], &[1, heads, seq_len, head_dim, inner, 0, f(scale)], heads * seq_len * seq_len)]);
    let probs = cx.gpu.storage(scores_len);
    cx.gpu.submit(&[], &[cx.gpu.step(K_ATTN_SOFTMAX_BIDIR, &[&scores, &probs], &[1, heads, seq_len], heads * seq_len)]);
    let out = cx.gpu.storage((seq_len * inner) as u64);
    cx.gpu.submit(&[], &[cx.gpu.step(K_ATTN_APPLY_FULL, &[&probs, v, &out], &[1, heads, seq_len, head_dim, inner, inner], heads * seq_len * head_dim)]);
    out
}

/// `SwiGLU.forward`: `(value, gate) = proj(x).chunk(2,-1); value * silu(gate)`.
/// `value`/`gate` are `fc1`'s two output-feature halves, split at import time
/// (matching `crate::block::BlockWeights`'s own convention).
fn swiglu(cx: &Ctx, value: &DeviceBuffer, gate: &DeviceBuffer, total: u32) -> DeviceBuffer {
    let y = cx.gpu.storage(total as u64);
    cx.gpu.submit(&[], &[cx.gpu.step(K_SILU_MUL, &[gate, value, &y], &[total], total)]);
    y
}

/// `y[r,:] = x[r,:] + scale[:] * h[r,:]` - the decoder block's LayerScale-
/// style residual gate (`scale1`/`scale2`, ONE shared `[dim]` vector
/// broadcast over every row), via `gate_row`'s own `rows_per_cond` knob set
/// to `rows` (a single condition group covering every row, `k = r/rows = 0`
/// for every valid `r`) - NOT `crate::block`'s own `gated_residual`, which
/// hardcodes `rows_per_cond=1` for AdaLN's per-ROW gate.
fn layerscale_residual(cx: &Ctx, x: &DeviceBuffer, scale: &DeviceBuffer, h: &DeviceBuffer, rows: u32, d: u32) -> DeviceBuffer {
    let y = cx.gpu.storage((rows * d) as u64);
    cx.gpu.submit(&[], &[cx.gpu.step(K_GATE_ROW, &[x, scale, h, &y], &[rows, d, rows], rows * d)]);
    y
}

/// One `MiniMaxH3VideoTransformerBlock.forward`.
#[allow(clippy::too_many_arguments)]
fn decoder_block_forward(cx: &Ctx, tensors: &Tensors, prefix: &str, x: &DeviceBuffer, cos: &DeviceBuffer, sin: &DeviceBuffer, ones_hd: &DeviceBuffer, cfg: &VideoVaeConfig, seq_len: u32) -> DeviceBuffer {
    let dim = cfg.decoder_dim();
    let heads = cfg.decoder_num_attention_heads;
    let hd = cfg.decoder_attention_head_dim;
    let ffn = cfg.decoder_ffn_dim();
    let half = cfg.decoder_rope_dim() / 2;
    let eps = cfg.decoder_norm_eps;

    let wq = upload(cx, tensors, &format!("{prefix}.attn.to_q.weight"));
    let bq = upload(cx, tensors, &format!("{prefix}.attn.to_q.bias"));
    let wk = upload(cx, tensors, &format!("{prefix}.attn.to_k.weight"));
    let bk = upload(cx, tensors, &format!("{prefix}.attn.to_k.bias"));
    let wv = upload(cx, tensors, &format!("{prefix}.attn.to_v.weight"));
    let bv = upload(cx, tensors, &format!("{prefix}.attn.to_v.bias"));
    let wo = upload(cx, tensors, &format!("{prefix}.attn.to_out.0.weight"));
    let bo = upload(cx, tensors, &format!("{prefix}.attn.to_out.0.bias"));
    let norm1 = upload(cx, tensors, &format!("{prefix}.norm1.weight"));
    let norm2 = upload(cx, tensors, &format!("{prefix}.norm2.weight"));
    let scale1 = upload(cx, tensors, &format!("{prefix}.scale1"));
    let scale2 = upload(cx, tensors, &format!("{prefix}.scale2"));

    let (fc1_w_shape, fc1_w) = get(tensors, &format!("{prefix}.ff.net.0.proj.weight"));
    let (_, fc1_b) = get(tensors, &format!("{prefix}.ff.net.0.proj.bias"));
    assert_eq!(fc1_w_shape, &vec![2 * ffn as usize, dim as usize], "{prefix}.ff.net.0.proj.weight");
    let half_w = (ffn * dim) as usize;
    let half_b = ffn as usize;
    let fc1_value_w = cx.gpu.storage_init(&format!("{prefix}.ff.net.0.proj.weight.value"), &fc1_w[..half_w]);
    let fc1_gate_w = cx.gpu.storage_init(&format!("{prefix}.ff.net.0.proj.weight.gate"), &fc1_w[half_w..2 * half_w]);
    let fc1_value_b = cx.gpu.storage_init(&format!("{prefix}.ff.net.0.proj.bias.value"), &fc1_b[..half_b]);
    let fc1_gate_b = cx.gpu.storage_init(&format!("{prefix}.ff.net.0.proj.bias.gate"), &fc1_b[half_b..2 * half_b]);
    let fc2_w = upload(cx, tensors, &format!("{prefix}.ff.net.2.weight"));
    let fc2_b = upload(cx, tensors, &format!("{prefix}.ff.net.2.bias"));

    let n1 = rmsnorm(cx, x, &norm1, seq_len, dim, eps);
    let q = linear(cx, &n1, &wq, Some(&bq), seq_len, dim, dim);
    let k = linear(cx, &n1, &wk, Some(&bk), seq_len, dim, dim);
    let v = linear(cx, &n1, &wv, Some(&bv), seq_len, dim, dim);
    let qn = rmsnorm(cx, &q, ones_hd, seq_len * heads, hd, eps);
    let kn = rmsnorm(cx, &k, ones_hd, seq_len * heads, hd, eps);
    rope_partial(cx, &qn, cos, sin, seq_len, heads, hd, half);
    rope_partial(cx, &kn, cos, sin, seq_len, heads, hd, half);
    let attn = attention(cx, &qn, &kn, &v, seq_len, heads, hd);
    let attn_out = linear(cx, &attn, &wo, Some(&bo), seq_len, dim, dim);
    let h1 = layerscale_residual(cx, x, &scale1, &attn_out, seq_len, dim);

    let n2 = rmsnorm(cx, &h1, &norm2, seq_len, dim, eps);
    let value = linear(cx, &n2, &fc1_value_w, Some(&fc1_value_b), seq_len, dim, ffn);
    let gate = linear(cx, &n2, &fc1_gate_w, Some(&fc1_gate_b), seq_len, dim, ffn);
    let act = swiglu(cx, &value, &gate, seq_len * ffn);
    let ff_out = linear(cx, &act, &fc2_w, Some(&fc2_b), seq_len, ffn, dim);
    layerscale_residual(cx, &h1, &scale2, &ff_out, seq_len, dim)
}

/// `MiniMaxH3VideoRotaryPosEmbed.forward`, host-side: `cos`/`sin` tables
/// `[seq_len, half]` (`half = decoder_rope_dim()/2`), with the `2*pi` factor
/// this module (unlike `crate::rope`) carries. `position_ids` is row-major
/// `[seq_len,3]` (t,h,w) coordinates, already in `[-1,1)` for patch rows and
/// `(0,0,0)` for the register/cls suffix rows (the caller's job, see
/// [`decode_clip`]).
fn decoder_rope_tables(cfg: &VideoVaeConfig, position_ids: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let seq_len = position_ids.len() / 3;
    let fpa = cfg.decoder_rope_freqs_per_axis() as usize;
    let half = 3 * fpa;
    let inv_freq: Vec<f64> = (0..fpa).map(|k| (cfg.decoder_rope_theta as f64).powf(-(k as f64) / fpa as f64)).collect();
    let mut cos = vec![0f32; seq_len * half];
    let mut sin = vec![0f32; seq_len * half];
    for row in 0..seq_len {
        for axis in 0..3usize {
            let pos = position_ids[row * 3 + axis] as f64;
            for (k, &freq) in inv_freq.iter().enumerate() {
                let angle = 2.0 * std::f64::consts::PI * pos * freq;
                let col = axis * fpa + k;
                cos[row * half + col] = angle.cos() as f32;
                sin[row * half + col] = angle.sin() as f32;
            }
        }
    }
    (cos, sin)
}

/// `hidden_states.view(B,F,H,W,out_c,pt,ps,ps).permute(0,4,1,5,2,6,3,7)
/// .reshape(B,out_c,F*pt,H*ps,W*ps)`, host-side (pure reshape, no learned
/// weights). `proj` is `[num_patches, out_patch]` row-major, `num_patches =
/// num_frames*height*width` in `(t,h,w)`-nested (t slowest) row order.
fn unpatchify(proj: &[f32], num_frames: u32, height: u32, width: u32, out_channels: u32, patch_size_t: u32, patch_size: u32) -> Vec<f32> {
    let ft = num_frames * patch_size_t;
    let fh = height * patch_size;
    let fw = width * patch_size;
    let mut out = vec![0f32; (out_channels * ft * fh * fw) as usize];
    let row_len = (out_channels * patch_size_t * patch_size * patch_size) as usize;
    for t in 0..num_frames {
        for h in 0..height {
            for w in 0..width {
                let row_idx = (t * height + h) * width + w;
                let row = &proj[row_idx as usize * row_len..(row_idx as usize + 1) * row_len];
                for oc in 0..out_channels {
                    for dt in 0..patch_size_t {
                        for dh in 0..patch_size {
                            for dw in 0..patch_size {
                                let src = (((oc * patch_size_t + dt) * patch_size + dh) * patch_size + dw) as usize;
                                let ot = t * patch_size_t + dt;
                                let oh = h * patch_size + dh;
                                let ow = w * patch_size + dw;
                                let dst = (((oc * ft + ot) * fh + oh) * fw + ow) as usize;
                                out[dst] = row[src];
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// `_decode_clip`'s `use_tiling=False` body: `decoder(post_quant_conv(z))`
/// over ONE latent clip. `z` a channel-major `[latent_channels,t,h,w]` host
/// buffer. Returns the pixel buffer `[out_channels, t*patch_size_t,
/// h*patch_size, w*patch_size]` plus its shape. Takes an already-open
/// [`Ctx`] so a tiled decode opens ONE device for the whole tile grid.
fn decode_clip_untiled(cx: &Ctx, cfg: &VideoVaeConfig, tensors: &Tensors, z: &[f32], t: u32, h: u32, w: u32) -> (Vec<f32>, u32, u32, u32) {
    assert_eq!(z.len(), (cfg.latent_channels * t * h * w) as usize, "decode_clip: input length {} != latent_channels*t*h*w", z.len());
    let zin = cx.gpu.storage_init("video_vae.z", z);
    let (pz, t1, h1, w1) = causal_conv3d(cx, tensors, "post_quant_conv", cfg.latent_channels, t, h, w, cfg.latent_channels, 1, (1, 1, 1), 0, 0, &zin);
    let pz_host = cx.gpu.read(&pz, (cfg.latent_channels * t1 * h1 * w1) as usize);

    // Tokenize: [C,T,H,W] channel-major -> [num_patches, C] row-major
    // (`(B,C,F,H,W).permute(0,2,3,4,1).reshape(B,F*H*W,C)`).
    let num_patches = t1 * h1 * w1;
    let mut tokens = vec![0f32; (num_patches * cfg.latent_channels) as usize];
    for c in 0..cfg.latent_channels {
        for tt in 0..t1 {
            for hh in 0..h1 {
                for ww in 0..w1 {
                    let src = (((c * t1 + tt) * h1 + hh) * w1 + ww) as usize;
                    let pidx = ((tt * h1 + hh) * w1 + ww) as usize;
                    tokens[pidx * cfg.latent_channels as usize + c as usize] = pz_host[src];
                }
            }
        }
    }

    let dim = cfg.decoder_dim();
    let proj_in_w = upload(cx, tensors, "decoder.proj_in.weight");
    let proj_in_b = upload(cx, tensors, "decoder.proj_in.bias");
    let tok_buf = cx.gpu.storage_init("video_vae.tokens", &tokens);
    let embedded = linear(cx, &tok_buf, &proj_in_w, Some(&proj_in_b), num_patches, cfg.latent_channels, dim);
    let embedded_host = cx.gpu.read(&embedded, (num_patches * dim) as usize);

    let (_, register) = get(tensors, "decoder.register_tokens");
    let num_reg = cfg.decoder_num_register_tokens;
    let seq_len = num_patches + num_reg + 1;
    let mut full = embedded_host;
    full.extend_from_slice(register);
    full.extend(std::iter::repeat_n(0f32, dim as usize)); // the literal-zero cls token

    // Position ids: patch rows get their (t,h,w) voxel-center coordinate in
    // [-1,1); register/cls rows stay at the origin (unrotated), matching
    // `position_ids.new_zeros((...))`'s all-zero suffix.
    let mut position_ids = vec![0f32; (seq_len * 3) as usize];
    for tt in 0..t1 {
        for hh in 0..h1 {
            for ww in 0..w1 {
                let pidx = ((tt * h1 + hh) * w1 + ww) as usize;
                position_ids[pidx * 3] = 2.0 * ((tt as f32 + 0.5) / t1 as f32) - 1.0;
                position_ids[pidx * 3 + 1] = 2.0 * ((hh as f32 + 0.5) / h1 as f32) - 1.0;
                position_ids[pidx * 3 + 2] = 2.0 * ((ww as f32 + 0.5) / w1 as f32) - 1.0;
            }
        }
    }
    let (cos, sin) = decoder_rope_tables(cfg, &position_ids);
    let cos_dev = cx.gpu.storage_init("video_vae.rope_cos", &cos);
    let sin_dev = cx.gpu.storage_init("video_vae.rope_sin", &sin);
    let ones_hd = cx.gpu.storage_init("video_vae.qknorm_ones", &vec![1f32; cfg.decoder_attention_head_dim as usize]);

    let mut hbuf = cx.gpu.storage_init("video_vae.h0", &full);
    for i in 0..cfg.decoder_num_layers as usize {
        let p = format!("decoder.transformer_blocks.{i}");
        hbuf = decoder_block_forward(cx, tensors, &p, &hbuf, &cos_dev, &sin_dev, &ones_hd, cfg, seq_len);
        // Same reclaim-point requirement as `H3Transformer::forward_streaming`'s
        // per-block loop: `decoder_block_forward` uploads roughly 18 fresh
        // device buffers per call (`upload`/`storage_init`, never reused
        // across layers), and dropping a `DeviceBuffer` does not itself free
        // VRAM on the wgpu backend until a poll proves the GPU is done with
        // it. Without this, one call's 36 layers alone hold onto ~10GB of
        // never-reclaimed weight buffers (dim=2048/ffn=8192 per layer) plus
        // activations - and `decode_clip_in`'s tile loop calls this function
        // once per tile (4 times for a 2x2 grid), so a real decode OOM'd
        // well before the outer per-chunk `poll_wait` in `decode`/`encode`
        // ever got a chance to run.
        cx.gpu.poll_wait();
    }

    let norm_out_w = upload(cx, tensors, "decoder.norm_out.weight");
    let norm_out_b = upload(cx, tensors, "decoder.norm_out.bias");
    let normed = layernorm_rows(cx, &hbuf, &norm_out_w, &norm_out_b, seq_len, dim, cfg.decoder_norm_eps);
    let proj_out_w = upload(cx, tensors, "decoder.proj_out.weight");
    let proj_out_b = upload(cx, tensors, "decoder.proj_out.bias");
    let out_patch = cfg.decoder_out_patch();
    let full_out = linear(cx, &normed, &proj_out_w, Some(&proj_out_b), seq_len, dim, out_patch);
    let full_out_host = cx.gpu.read(&full_out, (seq_len * out_patch) as usize);
    let patch_rows = &full_out_host[..(num_patches * out_patch) as usize];

    let ps = cfg.spatial_compression_ratio();
    let pst = cfg.temporal_compression_ratio();
    let pixels = unpatchify(patch_rows, t1, h1, w1, cfg.out_channels, pst, ps);
    (pixels, t1 * pst, h1 * ps, w1 * ps)
}

/// `_decode_clip`: one temporal clip, spatially tiled when `cfg.use_tiling`
/// (the reference's own shipped default).
///
/// The tile grid is laid out in PIXEL space and then mapped back onto the
/// latent grid (`z[..., i_pos/ratio : i_pos/ratio + i_len/ratio, ...]`) -
/// the opposite direction from [`encode_clip`], and the reason
/// [`split_tiles`] is called with the pixel height `h * ratio` rather than
/// with `h`. Because [`split_tiles`] keeps every boundary latent-aligned,
/// those divisions are exact. [`stitch_tiles`] is then given the PIXEL
/// overlaps unconverted, since what is being stitched here is pixels.
///
/// This is what bounds the ViT decoder's sequence length: every tile is a
/// 256x256-pixel / 16x16-latent window whatever the canvas, so the sequence
/// stays at `t * 256 + num_register_tokens + 1` instead of growing with the
/// frame area. See [`attention`]'s own overflow assert.
pub fn decode_clip(cfg: &VideoVaeConfig, tensors: &Tensors, device: Option<&str>, z: &[f32], t: u32, h: u32, w: u32) -> (Vec<f32>, u32, u32, u32) {
    let cx = Ctx::new(device);
    decode_clip_in(&cx, cfg, tensors, z, t, h, w)
}

fn decode_clip_in(cx: &Ctx, cfg: &VideoVaeConfig, tensors: &Tensors, z: &[f32], t: u32, h: u32, w: u32) -> (Vec<f32>, u32, u32, u32) {
    if !cfg.use_tiling {
        return decode_clip_untiled(cx, cfg, tensors, z, t, h, w);
    }
    let ratio = cfg.spatial_compression_ratio();
    let height = h * ratio;
    let width = w * ratio;
    let (y_idx, y_len, y_ov) = split_tiles(height, cfg.tile_sample_min_height, cfg.tile_sample_min_overlap_height, ratio);
    let (x_idx, x_len, x_ov) = split_tiles(width, cfg.tile_sample_min_width, cfg.tile_sample_min_overlap_width, ratio);
    if y_idx.len() == 1 && x_idx.len() == 1 {
        return decode_clip_untiled(cx, cfg, tensors, z, t, h, w);
    }

    let mut rows: Vec<Vec<Tile>> = Vec::with_capacity(y_idx.len());
    for (&i_pos, &i_len) in y_idx.iter().zip(y_len.iter()) {
        let mut row: Vec<Tile> = Vec::with_capacity(x_idx.len());
        for (&j_pos, &j_len) in x_idx.iter().zip(x_len.iter()) {
            let sub = spatial_window_host(z, cfg.latent_channels, t, h, w, i_pos / ratio, i_len / ratio, j_pos / ratio, j_len / ratio);
            let (p, pt, ph, pw) = decode_clip_untiled(cx, cfg, tensors, &sub, t, i_len / ratio, j_len / ratio);
            row.push(Tile::new(p, cfg.out_channels, pt, ph, pw));
            // Belt-and-suspenders alongside `decode_clip_untiled`'s own
            // per-layer poll_wait: reclaim this tile's buffers before the
            // next tile's 36-layer forward pass starts allocating.
            cx.gpu.poll_wait();
        }
        rows.push(row);
    }
    let out = stitch_tiles(&rows, &y_ov, &x_ov);
    (out.data, out.t, out.h, out.w)
}

/// `_decode`: mirrors `_encode`'s chunking. Decodes `tokens_chunk_size +
/// token_overlap` latent frames per internal chunk and linearly cross-fades
/// `frame_overlap` pixel frames between consecutive chunks, per
/// `AutoencoderKLMiniMaxH3._decode` exactly (including its own `num_chunks`
/// formula - see this module's own doc for the degenerate-`num_chunks`
/// caveat this reproduces rather than works around).
pub fn decode(cfg: &VideoVaeConfig, tensors: &Tensors, device: Option<&str>, z: &[f32], nt: u32, h: u32, w: u32) -> (Vec<f32>, u32, u32, u32) {
    let c = cfg.latent_channels;
    let tokens_chunk_size = cfg.tokens_chunk_size();
    let token_drop = cfg.token_drop;
    let temporal_ratio = cfg.temporal_compression_ratio();
    let chunk_num_frames = tokens_chunk_size * temporal_ratio;
    let token_overlap = cfg.token_overlap();
    let frame_pre_padding = cfg.frame_pre_padding();
    let frame_overlap = cfg.frame_overlap();
    let ps = cfg.spatial_compression_ratio();
    let cx = Ctx::new(device);

    let num_tokens = nt + token_drop;
    let pad_tokens = (tokens_chunk_size - num_tokens % tokens_chunk_size) % tokens_chunk_size;
    let num_chunks: u32 = ((num_tokens + pad_tokens) / tokens_chunk_size).saturating_sub(u32::from(token_drop > 0));

    let (zpad, zt) = if pad_tokens > 0 { (repeat_last_frame_host(z, c, nt, h, w, pad_tokens), nt + pad_tokens) } else { (z.to_vec(), nt) };

    let out_channels = cfg.out_channels;
    let (fh, fw) = (h * ps, w * ps);
    let mut decoded_chunks: Vec<(Vec<f32>, u32)> = Vec::new();
    let mut overlap: Option<(Vec<f32>, u32)> = None;

    for i in 0..num_chunks {
        let start = i * tokens_chunk_size;
        let take = (tokens_chunk_size + token_overlap).min(zt - start);
        let zc = slice_frames_host(&zpad, c, zt, h, w, start, take);
        let (clip_pixels, cf, ch_, cw_) = decode_clip_in(&cx, cfg, tensors, &zc, take, h, w);
        assert_eq!((ch_, cw_), (fh, fw), "decode: clip spatial shape drifted");
        // Same reclaim-point requirement as `H3Transformer::forward_streaming`'s
        // per-block loop (see that function's own doc): on the wgpu backend,
        // dropping a `DeviceBuffer` does not itself free VRAM until a poll
        // proves the GPU is done with it. A chunked loop that only submits
        // and reads back at the very end of each `decode_clip_in` call never
        // triggers that proof between chunks, so each clip's 36-layer ViT
        // decoder activations pile up - measured directly on a P40: without
        // this call, a 124-frame/384x384 decode (num_chunks in the
        // single-digit range here) climbed from ~0.4GB to ~22GB in ~10
        // seconds, one multi-GB jump per chunk, until it OOM'd; with it,
        // each chunk's buffers are reclaimed before the next chunk's clip is
        // decoded.
        cx.gpu.poll_wait();

        for j in 0..(1 + u32::from(token_drop > 0)) {
            let frame_start = j * chunk_num_frames;
            if frame_start >= cf {
                continue;
            }
            let take_frames = chunk_num_frames.min(cf - frame_start);
            let mut chunk = slice_frames_host(&clip_pixels, out_channels, cf, ch_, cw_, frame_start, take_frames);
            let mut chunk_t = take_frames;
            if frame_pre_padding > 0 {
                assert!(frame_pre_padding < chunk_t, "decode: frame_pre_padding {frame_pre_padding} >= chunk frame count {chunk_t}");
                chunk = slice_frames_host(&chunk, out_channels, chunk_t, ch_, cw_, frame_pre_padding, chunk_t - frame_pre_padding);
                chunk_t -= frame_pre_padding;
            }
            if j == 0 {
                if let Some((odata, ot)) = overlap.take() {
                    let blended = blend_frames_host(&odata, ot, &chunk, chunk_t, out_channels, fh, fw, frame_overlap);
                    decoded_chunks.push((blended, chunk_t));
                } else {
                    decoded_chunks.push((chunk, chunk_t));
                }
            } else {
                overlap = Some((chunk, chunk_t));
            }
        }
    }
    if let Some(last) = overlap {
        decoded_chunks.push(last);
    }

    let (mut dec, mut dt) = concat_frames_host(&decoded_chunks, out_channels, fh, fw);

    if pad_tokens > 0 {
        let intra_tail = cfg.clip_length % temporal_ratio;
        let num_tokens_before_pad = zt - pad_tokens;
        let mut pad_frames = 0u32;
        for k in 0..pad_tokens {
            pad_frames += if intra_tail != 0 && (num_tokens_before_pad + k).is_multiple_of(tokens_chunk_size) { intra_tail } else { temporal_ratio };
        }
        assert!(pad_frames <= dt, "decode: pad_frames {pad_frames} exceeds decoded {dt} frames");
        dec = slice_frames_host(&dec, out_channels, dt, fh, fw, 0, dt - pad_frames);
        dt -= pad_frames;
    }
    (dec, dt, fh, fw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_manifest_shapes_match_the_real_config_layout() {
        let cfg = VideoVaeConfig::real();
        let m = cfg.tensor_manifest();
        let names: std::collections::HashSet<&str> = m.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names.len(), m.len(), "duplicate tensor name in the manifest");

        let get = |n: &str| m.iter().find(|(k, _)| k == n).unwrap().1.clone();
        assert_eq!(get("encoder.conv_in.weight"), vec![128, 3, 3, 3, 3]);
        assert_eq!(get("encoder.conv_out.weight"), vec![48, 1024, 3, 3, 3]);
        assert_eq!(get("quant_conv.weight"), vec![48, 48, 1, 1, 1]);
        assert_eq!(get("post_quant_conv.weight"), vec![24, 24, 1, 1, 1]);
        // Stage 0: in==out==128, no conv_shortcut on resnet 0.
        assert!(!names.contains("encoder.down_blocks.0.resnets.0.conv_shortcut.weight"));
        // Stage 1 resnet 0: 128->256, HAS conv_shortcut.
        assert_eq!(get("encoder.down_blocks.1.resnets.0.conv_shortcut.weight"), vec![256, 128, 1, 1, 1]);
        // Downsample present stages 0-3, absent 4-5.
        assert!(names.contains("encoder.down_blocks.0.downsamplers.0.conv.weight"));
        assert!(names.contains("encoder.down_blocks.3.downsamplers.0.conv.weight"));
        assert!(!names.contains("encoder.down_blocks.4.downsamplers.0.conv.weight"));
        assert!(!names.contains("encoder.down_blocks.5.downsamplers.0.conv.weight"));

        assert_eq!(get("decoder.proj_in.weight"), vec![2048, 24]);
        assert_eq!(get("decoder.register_tokens"), vec![1, 4, 2048]);
        assert_eq!(get("decoder.transformer_blocks.0.attn.to_q.weight"), vec![2048, 2048]);
        assert_eq!(get("decoder.transformer_blocks.0.ff.net.0.proj.weight"), vec![16384, 2048]);
        assert_eq!(get("decoder.transformer_blocks.0.ff.net.2.weight"), vec![2048, 8192]);
        assert!(!names.contains("decoder.transformer_blocks.0.attn.norm_q.weight"), "norm_q is elementwise_affine=False - no weight tensor");
        assert_eq!(get("decoder.proj_out.weight"), vec![3072, 2048], "3*4*16*16 output patch");
        assert!(!names.contains("decoder.transformer_blocks.36.norm1.weight"), "only 36 layers (0..=35)");
    }

    #[test]
    fn derived_chunk_geometry_matches_the_real_config() {
        let cfg = VideoVaeConfig::real();
        assert_eq!(cfg.spatial_compression_ratio(), 16);
        assert_eq!(cfg.temporal_compression_ratio(), 4);
        assert_eq!(cfg.tokens_chunk_size(), 5);
        assert_eq!(cfg.frame_pre_padding(), 3);
        assert_eq!(cfg.token_overlap(), 2);
        assert_eq!(cfg.frame_overlap(), 5);
        assert_eq!(cfg.decoder_out_patch(), 3072);
    }

    fn rand_tensors(cfg: &VideoVaeConfig, seed: u64) -> Tensors {
        let mut rng = data::rng::Lcg::new(seed);
        cfg.tensor_manifest()
            .into_iter()
            .map(|(name, shape)| {
                let n: usize = shape.iter().product();
                (name, (shape, rng.vec_scaled(n, 0.2)))
            })
            .collect()
    }

    /// Weight-free tiny-config smoke test: a single 17-frame clip through
    /// `encode_clip` -> `posterior_mode` -> `decode_clip`, checking every
    /// stage's shape against the hand-traced chunk geometry (mirrors the
    /// roadmap ledger's own worked example, at tiny widths) and that the
    /// round-tripped pixels are finite and not trivially all-zero.
    #[test]
    fn tiny_config_single_clip_round_trip_is_finite_and_the_right_shape() {
        let cfg = VideoVaeConfig::tiny();
        let tensors = rand_tensors(&cfg, 11);
        let (h, w) = (32u32, 32u32);
        let t = cfg.clip_length;

        let mut rng = data::rng::Lcg::new(5);
        let pixels = rng.vec_scaled((cfg.in_channels * t * h * w) as usize, 0.3);

        let (moments, mt, mh, mw) = encode_clip(&cfg, &tensors, Some("cpu"), &pixels, t, h, w);
        // Matches the ledger's own worked trace shape (T: 17->17->9->5->5->5->5, H,W: 32->16->8->4->2->2->2).
        assert_eq!((mt, mh, mw), (5, 2, 2));
        assert_eq!(moments.len(), (2 * cfg.latent_channels * mt * mh * mw) as usize);
        assert!(moments.iter().all(|v| v.is_finite()));

        let z = posterior_mode(&moments, cfg.latent_channels, mt, mh, mw);
        assert_eq!(z.len(), (cfg.latent_channels * mt * mh * mw) as usize);

        let (pix, pt, ph, pw) = decode_clip(&cfg, &tensors, Some("cpu"), &z, mt, mh, mw);
        assert_eq!((pt, ph, pw), (mt * cfg.temporal_compression_ratio(), mh * cfg.spatial_compression_ratio(), mw * cfg.spatial_compression_ratio()));
        assert_eq!(pix.len(), (cfg.out_channels * pt * ph * pw) as usize);
        assert!(pix.iter().all(|v| v.is_finite()), "decode_clip output must be finite");
        assert!(pix.iter().any(|&v| v != 0.0), "decode_clip output must not be trivially all-zero");
    }

    /// The outer `encode`/`decode` chunk orchestration, weight-free, at TWO
    /// full clips (`num_frames = 2*clip_length = 34`) so `decode`'s own
    /// `num_chunks` formula lands non-degenerate (`num_chunks=1`, not the
    /// `num_chunks=0` edge case a single clip's worth of latent hits - see
    /// this module's own doc). Checks the round trip is finite, shaped
    /// consistently with the input frame count, and exercises the blend
    /// path (`frame_overlap=5 > 0` at the real geometry this tiny config
    /// keeps literal).
    #[test]
    fn tiny_config_multi_clip_round_trip_is_finite_and_consistently_shaped() {
        let cfg = VideoVaeConfig::tiny();
        let tensors = rand_tensors(&cfg, 13);
        let (h, w) = (32u32, 32u32);
        let num_frames = 2 * cfg.clip_length;

        let mut rng = data::rng::Lcg::new(6);
        let pixels = rng.vec_scaled((cfg.in_channels * num_frames * h * w) as usize, 0.3);

        let (moments, mt, mh, mw) = encode(&cfg, &tensors, Some("cpu"), &pixels, num_frames, h, w);
        // Two clips' worth of tokens_chunk_size(5) minus one token_drop(3): 2*5-3=7.
        assert_eq!(mt, 7);
        assert!(moments.iter().all(|v| v.is_finite()));

        let z = posterior_mode(&moments, cfg.latent_channels, mt, mh, mw);
        let (pix, pt, ph, pw) = decode(&cfg, &tensors, Some("cpu"), &z, mt, mh, mw);
        assert_eq!((ph, pw), (mh * cfg.spatial_compression_ratio(), mw * cfg.spatial_compression_ratio()));
        assert!(pt > 0, "decode must produce at least one pixel frame");
        assert_eq!(pix.len(), (cfg.out_channels * pt * ph * pw) as usize);
        assert!(pix.iter().all(|v| v.is_finite()), "decode output must be finite");
        assert!(pix.iter().any(|&v| v != 0.0), "decode output must not be trivially all-zero");
    }

    /// Real NUMERIC parity against `tools/minimaxh3_video_vae_dump_reference.py`'s
    /// golden - the ACTUAL installed `diffusers==0.40.0`
    /// `AutoencoderKLMiniMaxH3`, run for real at the tiny config
    /// (`use_tiling=False`) and dumped. Two rungs: a single-clip round trip
    /// through `encode_clip`/`decode_clip` directly (no outer chunking), and
    /// a two-clip round trip through the outer `encode`/`decode` chunk
    /// orchestration at a NON-degenerate `num_chunks` (see this module's own
    /// doc for the `num_chunks==0` edge case the dumper's own self-
    /// validation confirmed is real - the real reference itself raises on a
    /// single clip's worth of latent, which is why this golden's outer-
    /// chunking rung uses two clips instead).
    #[test]
    fn tiny_config_matches_the_real_reference_numerically() {
        let fixture_dir = brain_testutil::testdata_path("golden/minimaxh3/video_vae_tiny");
        let fixture_file = fixture_dir.join("minimaxh3_video_vae_tiny.safetensors");
        if !fixture_file.is_file() {
            brain_testutil::skip(&format!("{} not found - run tools/minimaxh3_video_vae_dump_reference.py --out {}", fixture_file.display(), fixture_dir.display()));
            return;
        }

        let cfg = VideoVaeConfig::tiny();

        let Some(src) = brain_testutil::golden::Source::open(&fixture_dir, "tools/minimaxh3_video_vae_dump_reference.py") else {
            return;
        };
        let ok = src.require(&[
            ("in_channels", cfg.in_channels as i64),
            ("out_channels", cfg.out_channels as i64),
            ("latent_channels", cfg.latent_channels as i64),
            ("layers_per_block", cfg.layers_per_block as i64),
            ("norm_num_groups", cfg.norm_num_groups as i64),
            ("decoder_num_layers", cfg.decoder_num_layers as i64),
            ("decoder_num_attention_heads", cfg.decoder_num_attention_heads as i64),
            ("decoder_attention_head_dim", cfg.decoder_attention_head_dim as i64),
            ("decoder_num_register_tokens", cfg.decoder_num_register_tokens as i64),
            ("decoder_ffn_mult", cfg.decoder_ffn_mult as i64),
            ("clip_length", cfg.clip_length as i64),
            ("token_drop", cfg.token_drop as i64),
            ("rope_theta_x1000", (cfg.decoder_rope_theta * 1000.0) as i64),
            ("rope_dim_ratio_x1000", (cfg.decoder_rope_dim_ratio * 1000.0) as i64),
        ]);
        if !ok {
            return;
        }

        let raw = checkpoint::safetensors::read(fixture_file.to_str().expect("fixture path is valid UTF-8")).expect("read golden fixture");
        let fx: std::collections::HashMap<String, checkpoint::safetensors::StTensor> = raw.into_iter().map(|t| (t.name.clone(), t)).collect();
        let get = |name: &str| -> &[f32] { &fx.get(name).unwrap_or_else(|| panic!("golden fixture tap {name:?} missing")).data };
        let shape = |name: &str| -> &[usize] { &fx.get(name).unwrap().shape };

        let mut tensors: Tensors = std::collections::HashMap::new();
        for (name, t) in &fx {
            if !name.starts_with("input_") && !name.starts_with("tap_") {
                tensors.insert(name.clone(), (t.shape.clone(), t.data.clone()));
            }
        }

        let (h, w) = (32u32, 32u32);
        let mut r = brain_testutil::parity::Report::new(0.9999);

        // ---- rung 1: single-clip encode_clip/decode_clip, no outer chunking ----
        let (moments1, mt1, mh1, mw1) = encode_clip(&cfg, &tensors, Some("cpu"), get("input_x1"), cfg.clip_length, h, w);
        assert_eq!([2 * cfg.latent_channels as usize, mt1 as usize, mh1 as usize, mw1 as usize], *shape("tap_clip_moments"));
        r.check("tap_clip_moments", &moments1, get("tap_clip_moments"));

        let z1 = posterior_mode(&moments1, cfg.latent_channels, mt1, mh1, mw1);
        r.check("input_z1 (mode of tap_clip_moments)", &z1, get("input_z1"));

        let (pixels1, pt1, ph1, pw1) = decode_clip(&cfg, &tensors, Some("cpu"), get("input_z1"), mt1, mh1, mw1);
        assert_eq!([cfg.out_channels as usize, pt1 as usize, ph1 as usize, pw1 as usize], *shape("tap_clip_pixels"));
        r.check("tap_clip_pixels", &pixels1, get("tap_clip_pixels"));

        // ---- rung 2: two-clip outer encode()/decode() chunk orchestration ----
        let num_frames2 = 2 * cfg.clip_length;
        let (moments2, mt2, mh2, mw2) = encode(&cfg, &tensors, Some("cpu"), get("input_x2"), num_frames2, h, w);
        let z2 = posterior_mode(&moments2, cfg.latent_channels, mt2, mh2, mw2);
        assert_eq!([cfg.latent_channels as usize, mt2 as usize, mh2 as usize, mw2 as usize], *shape("tap_multi_latent"));
        r.check("tap_multi_latent", &z2, get("tap_multi_latent"));

        let (pixels2, pt2, ph2, pw2) = decode(&cfg, &tensors, Some("cpu"), get("tap_multi_latent"), mt2, mh2, mw2);
        assert_eq!([cfg.out_channels as usize, pt2 as usize, ph2 as usize, pw2 as usize], *shape("tap_multi_pixels"));
        r.check("tap_multi_pixels", &pixels2, get("tap_multi_pixels"));

        // ---- rung 3: TILED _decode_clip at 512x768 (a 3x4 tile grid) ----
        // The tile grid this lands on has UNEVEN width overlaps
        // ([96,80,80]) and even height overlaps ([128,128]), so a height/
        // width mix-up or an "all overlaps equal" simplification cannot
        // pass. The dumper self-validates that the reference's own tiled and
        // untiled results differ here (max abs 2.6e-2), so this rung cannot
        // pass vacuously against the untiled path.
        assert!(cfg.use_tiling, "tiny() must ship the reference's own use_tiling default");
        let ratio = cfg.spatial_compression_ratio();
        let dec_shape = shape("input_z_tiled").to_vec();
        let (zt, zh, zw) = (dec_shape[1] as u32, dec_shape[2] as u32, dec_shape[3] as u32);
        assert!(zh * ratio > cfg.tile_sample_min_height && zw * ratio > cfg.tile_sample_min_width, "rung 3 canvas {}x{} is not above the tile size - it would not tile", zh * ratio, zw * ratio);
        let (tiled_pixels, tpt, tph, tpw) = decode_clip(&cfg, &tensors, Some("cpu"), get("input_z_tiled"), zt, zh, zw);
        assert_eq!([cfg.out_channels as usize, tpt as usize, tph as usize, tpw as usize], *shape("tap_tiled_dec_pixels"));
        r.check("tap_tiled_dec_pixels (3x4 tile grid)", &tiled_pixels, get("tap_tiled_dec_pixels"));

        // Vacuity guard: decode the SAME latent through this port's untiled
        // path and require it to disagree with the reference's tiled output
        // by far more than the float32 noise the tiled path clears it by.
        // Without this, a tiling implementation that quietly degenerated to
        // "decode the whole frame" would still pass the rung above at any
        // canvas where the two happened to be close, which is exactly the
        // failure mode - a confident, structured, wrong output - that this
        // whole pass exists to rule out.
        let mut untiled_cfg = cfg.clone();
        untiled_cfg.use_tiling = false;
        let (untiled_pixels, ..) = decode_clip(&untiled_cfg, &tensors, Some("cpu"), get("input_z_tiled"), zt, zh, zw);
        let want = get("tap_tiled_dec_pixels");
        let untiled_err = untiled_pixels.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let tiled_err = tiled_pixels.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("  vacuity guard: untiled max_abs vs the tiled reference {untiled_err:.3e}, tiled {tiled_err:.3e}");
        assert!(untiled_err > 1e-3, "the untiled decode agrees with the TILED reference to {untiled_err:.3e} - rung 3 would pass vacuously");
        assert!(untiled_err > 1000.0 * tiled_err, "the tiled path is not meaningfully closer than the untiled one ({tiled_err:.3e} vs {untiled_err:.3e})");

        // ---- rung 4: TILED _encode_clip at 384x384 (a 2x2 tile grid) ----
        // The other tiling direction: tiles are laid out in pixel space and
        // the LATENT output is stitched, so the overlaps are converted by
        // `/ spatial_compression_ratio` where rung 3 uses them as-is.
        let enc_shape = shape("input_x_tiled").to_vec();
        let (xt, xh, xw) = (enc_shape[1] as u32, enc_shape[2] as u32, enc_shape[3] as u32);
        assert!(xh > cfg.tile_sample_min_height && xw > cfg.tile_sample_min_width, "rung 4 canvas {xh}x{xw} is not above the tile size - it would not tile");
        let (tiled_moments, tmt, tmh, tmw) = encode_clip(&cfg, &tensors, Some("cpu"), get("input_x_tiled"), xt, xh, xw);
        assert_eq!([2 * cfg.latent_channels as usize, tmt as usize, tmh as usize, tmw as usize], *shape("tap_tiled_enc_moments"));
        r.check("tap_tiled_enc_moments (2x2 tile grid)", &tiled_moments, get("tap_tiled_enc_moments"));

        r.finish("minimaxh3 video vae tiny-config forward vs real reference");
    }

    /// Tiling is what BOUNDS the ViT decoder's sequence length, so the
    /// `u32` score-count ceiling in [`attention`] is unreachable on the
    /// tiled path at any canvas - including the released 768x1344 one,
    /// where the untiled path overflowed it.
    ///
    /// This is the property that makes that assert a real invariant rather
    /// than an arbitrary limit: a tile is a 16x16 latent grid whatever the
    /// canvas, so the sequence depends only on the temporal chunk size, and
    /// the canvas cancels out entirely. If this ever fails, the tiling has
    /// stopped bounding the sequence and the fix is there, not here.
    #[test]
    fn tiling_bounds_the_vit_sequence_below_the_u32_score_ceiling_at_every_canvas() {
        let cfg = VideoVaeConfig::real();
        let ratio = cfg.spatial_compression_ratio();
        let heads = cfg.decoder_num_attention_heads as u64;
        // The longest latent clip `decode` ever hands one `decode_clip`.
        let clip_latent_frames = cfg.tokens_chunk_size() + cfg.token_overlap();
        let tile_latent = cfg.tile_sample_min_height / ratio;
        assert_eq!(tile_latent, 16, "a 256-pixel tile is a 16x16 latent grid at ratio {ratio}");

        // 1344 is the released canvas's long edge; 4096 is far past
        // anything this model is run at.
        for canvas in [256u32, 384, 512, 768, 1024, 1344, 4096] {
            let latent = canvas / ratio;
            let (starts, lens, _) = split_tiles(canvas, cfg.tile_sample_min_width, cfg.tile_sample_min_overlap_width, ratio);
            // Every tile is at most one tile_size wide, so the per-tile
            // latent grid never exceeds 16 on an axis however large the
            // canvas gets.
            let max_tile_latent = lens.iter().map(|l| l / ratio).max().expect("at least one tile");
            assert!(max_tile_latent <= tile_latent, "canvas {canvas}: a tile spans {max_tile_latent} latents, more than the {tile_latent} a 256-pixel tile should");

            let seq = clip_latent_frames as u64 * (max_tile_latent as u64).pow(2) + cfg.decoder_num_register_tokens as u64 + 1;
            let scores = heads * seq * seq;
            assert!(scores <= u32::MAX as u64, "canvas {canvas}: tiled seq_len {seq} still overflows the u32 score space ({scores})");

            // And the untiled path at the same canvas, to show the ceiling
            // is a real one that tiling is what avoids - not a limit that
            // was simply set high enough.
            let untiled_seq = clip_latent_frames as u64 * (latent as u64).pow(2) + cfg.decoder_num_register_tokens as u64 + 1;
            let untiled_scores = heads * untiled_seq * untiled_seq;
            println!("  canvas {canvas:>5}: {} tiles, tiled seq {seq} ({scores} scores), untiled seq {untiled_seq} ({untiled_scores} scores){}", starts.len(), if untiled_scores > u32::MAX as u64 { " OVERFLOWS" } else { "" });
        }

        // The released canvas is the one section 2 of the audit measured as
        // overflowing untiled: 48x84 latents, 28229 tokens. Assert that the
        // untiled path there really would overflow, so this test cannot
        // quietly become vacuous if the config's widths change.
        let untiled_released = clip_latent_frames as u64 * (768 / ratio) as u64 * (1344 / ratio) as u64 + cfg.decoder_num_register_tokens as u64 + 1;
        assert!(heads * untiled_released * untiled_released > u32::MAX as u64, "the untiled 768x1344 decode no longer overflows - this test's premise is stale");
    }

    /// REAL-WEIGHT numeric parity for the spatial tiling: one tiled
    /// `_decode_clip` over a 384x384 canvas (a 24x24 latent grid, a 2x2 tile
    /// grid) through the actual `MiniMaxAI/MiniMax-H3` video VAE, against
    /// the real installed `diffusers==0.40.0` decoding the same latent,
    /// dumped by `tools/minimaxh3_video_vae_real_tiled_dump_reference.py`.
    ///
    /// This is a different claim from
    /// [`tiny_config_matches_the_real_reference_numerically`]'s tiled rungs,
    /// which pin the tiling ALGORITHM at random weights. What only real
    /// weights can show is that the algorithm composes correctly with the
    /// real 36-layer ViT decoder at its real widths - that a real tile is
    /// genuinely a 16x16 latent grid whose `[-1, 1)` rotary coordinates are
    /// the ones the decoder was trained on. The tiny config cannot see that:
    /// its `decoder_attention_head_dim: 8` leaves one RoPE frequency per
    /// axis, where `theta^0 = 1` hides any frequency-ladder error.
    ///
    /// The golden carries the reference's UNTILED decode of the same latent
    /// too, purely as a vacuity guard, so this rung cannot pass against an
    /// implementation that quietly decoded the whole frame.
    #[test]
    fn tiled_decode_matches_the_real_reference_numerically_with_real_weights() {
        let Ok(root) = std::env::var("BRAIN_MINIMAXH3_DIR") else {
            brain_testutil::skip("BRAIN_MINIMAXH3_DIR not set - no local MiniMax-H3 checkout to decode from");
            return;
        };
        let fixture_dir = brain_testutil::testdata_path("golden/minimaxh3/video_vae_real_tiled");
        let fixture_file = fixture_dir.join("minimaxh3_video_vae_real_tiled.safetensors");
        if !fixture_file.is_file() {
            brain_testutil::skip(&format!(
                "{} not found - run BRAIN_MINIMAXH3_DIR={root} tools/minimaxh3_video_vae_real_tiled_dump_reference.py --out {}",
                fixture_file.display(),
                fixture_dir.display()
            ));
            return;
        }

        let cfg = VideoVaeConfig::real();
        let dir = format!("{root}/vae");
        let tensors = crate::import::import_video_vae(&dir, &cfg).unwrap_or_else(|e| panic!("import_video_vae: {e}"));

        let raw = checkpoint::safetensors::read(fixture_file.to_str().expect("fixture path is valid UTF-8")).expect("read golden fixture");
        let fx: std::collections::HashMap<String, checkpoint::safetensors::StTensor> = raw.into_iter().map(|t| (t.name.clone(), t)).collect();
        let get = |name: &str| -> &[f32] { &fx.get(name).unwrap_or_else(|| panic!("golden tap {name:?} missing")).data };
        let shape = |name: &str| -> &[usize] { &fx.get(name).unwrap().shape };

        let zs = shape("input_z_real_tiled").to_vec();
        let (zt, zh, zw) = (zs[1] as u32, zs[2] as u32, zs[3] as u32);
        let ratio = cfg.spatial_compression_ratio();
        let (canvas_h, canvas_w) = (zh * ratio, zw * ratio);
        assert!(cfg.use_tiling, "real() must ship the reference's own use_tiling default");
        let (ty, ..) = split_tiles(canvas_h, cfg.tile_sample_min_height, cfg.tile_sample_min_overlap_height, ratio);
        let (tx, ..) = split_tiles(canvas_w, cfg.tile_sample_min_width, cfg.tile_sample_min_overlap_width, ratio);
        assert!(ty.len() > 1 && tx.len() > 1, "the golden's {canvas_h}x{canvas_w} canvas is not multi-tile ({}x{})", ty.len(), tx.len());

        let device = std::env::var("BRAIN_MINIMAXH3_TEST_DEVICE").unwrap_or_else(|_| "cpu".to_string());
        eprintln!("decoding a real-weight {canvas_h}x{canvas_w} canvas as a {}x{} tile grid on device={device:?} ...", ty.len(), tx.len());
        let t0 = std::time::Instant::now();
        let (pixels, pt, ph, pw) = decode_clip(&cfg, &tensors, Some(&device), get("input_z_real_tiled"), zt, zh, zw);
        eprintln!("  decoded in {:.1}s", t0.elapsed().as_secs_f32());
        assert_eq!([cfg.out_channels as usize, pt as usize, ph as usize, pw as usize], *shape("tap_real_tiled_pixels"));

        let mut r = brain_testutil::parity::Report::new(0.9999);
        r.check("tap_real_tiled_pixels (2x2 tile grid, real weights)", &pixels, get("tap_real_tiled_pixels"));

        // Vacuity guard, against the reference's OWN untiled decode of the
        // same latent: this port's tiled result must be far closer to the
        // tiled reference than the untiled reference is.
        let tiled_ref = get("tap_real_tiled_pixels");
        let untiled_ref = get("tap_real_untiled_pixels");
        let sep = untiled_ref.iter().zip(tiled_ref).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let err = pixels.iter().zip(tiled_ref).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("  vacuity guard: the reference's own untiled decode differs from its tiled one by max_abs {sep:.3e}; this port's tiled decode by {err:.3e}");
        assert!(sep > 1e-2, "the reference's tiled and untiled decodes agree to {sep:.3e} at real weights - this rung would pass vacuously");
        assert!(err < sep / 100.0, "this port's tiled decode ({err:.3e}) is not decisively closer to the tiled reference than the untiled one is ({sep:.3e})");

        r.finish("minimaxh3 video vae real-weight tiled decode vs real reference");
    }

    /// `split_tiles` against the REAL reference's own `_split_tiles`, called
    /// directly at a spread of lengths and dumped as integers by
    /// `tools/minimaxh3_video_vae_dump_reference.py`.
    ///
    /// This is a separate rung from the stitched-output goldens on purpose:
    /// the tile layout is the part of the algorithm with the most room for a
    /// plausible-but-wrong answer (tile count, slack distribution, edge
    /// handling), and comparing it as DATA localizes a mismatch to the
    /// layout instead of leaving it smeared across a blended pixel buffer.
    #[test]
    fn split_tiles_matches_the_real_references_own_tile_layout() {
        let fixture_dir = brain_testutil::testdata_path("golden/minimaxh3/video_vae_tiny");
        let fixture_file = fixture_dir.join("minimaxh3_video_vae_tiny.safetensors");
        if !fixture_file.is_file() {
            brain_testutil::skip(&format!("{} not found - run tools/minimaxh3_video_vae_dump_reference.py --out {}", fixture_file.display(), fixture_dir.display()));
            return;
        }
        let raw = checkpoint::safetensors::read(fixture_file.to_str().expect("fixture path is valid UTF-8")).expect("read golden fixture");
        let fx: std::collections::HashMap<String, checkpoint::safetensors::StTensor> = raw.into_iter().map(|t| (t.name.clone(), t)).collect();

        let ratio = VideoVaeConfig::real().spatial_compression_ratio();
        let probes: Vec<(u32, u32, u32)> = fx
            .keys()
            .filter_map(|k| k.strip_prefix("tap_split_starts_"))
            .map(|s| {
                let p: Vec<u32> = s.split('_').map(|v| v.parse().expect("probe key is three integers")).collect();
                (p[0], p[1], p[2])
            })
            .collect();
        assert!(!probes.is_empty(), "golden has no tap_split_* probes - regenerate it with the current dumper");

        let mut multi_tile = 0usize;
        for (length, tile, min_ov) in probes {
            let key = format!("{length}_{tile}_{min_ov}");
            let want = |what: &str| -> Vec<u32> { fx[&format!("tap_split_{what}_{key}")].data.iter().map(|v| *v as u32).collect() };
            let (starts, lens, ovs) = split_tiles(length, tile, min_ov, ratio);
            assert_eq!(starts, want("starts"), "_split_tiles({length},{tile},{min_ov}) start indices");
            assert_eq!(lens, want("lengths"), "_split_tiles({length},{tile},{min_ov}) tile lengths");
            assert_eq!(ovs, want("overlaps"), "_split_tiles({length},{tile},{min_ov}) overlaps");
            if starts.len() > 1 {
                multi_tile += 1;
                // Every tile is FULL size and the union covers `length`
                // exactly - there is no partial edge tile, the slack lives
                // in the widened overlaps.
                assert!(lens.iter().all(|&l| l == tile), "_split_tiles({length}) produced a partial edge tile");
                assert_eq!(starts[starts.len() - 1] + tile, length, "_split_tiles({length}) does not end exactly at the length");
                assert!(starts.iter().all(|s| s.is_multiple_of(ratio)), "_split_tiles({length}) produced a start that is not latent-aligned");
            }
        }
        assert!(multi_tile >= 4, "only {multi_tile} multi-tile probes - the golden is not covering the tiling regime");
    }
}
