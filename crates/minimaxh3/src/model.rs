// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniMax-H3's full DiT core: config, RoPE, the token refiner, the
//! `num_layers` block stack, the two input projections, `context_embedder`,
//! `norm_out` and the two output heads, assembled per
//! `MiniMaxH3Transformer3DModel.forward` (`transformer_minimax_h3.py`).
//!
//! **Packed-sequence LAYOUT is out of scope here** (a later phase's job):
//! [`H3Transformer::forward`] takes
//! `position_ids`/`token_tags`/`timestep_indices`/`video_indices`/
//! `audio_indices`/`text_indices` as explicit arguments, exactly matching the
//! reference's own `forward` signature - it does not build the packed layout
//! itself either.
//!
//! **Batch axis**: the reference's own doc states the batch axis is "a pure
//! replication axis" - every batch item shares the same structural tensors
//! and is an independent attention document. This port's [`H3Transformer::
//! forward`] takes exactly one packed sequence (batch size 1); the natural
//! extension to `bsz > 1` is tiling the per-row structural tensors
//! (`position_ids`/`token_tags`/`timestep_indices`/the three index arrays)
//! `bsz` times and widening `attn_scores_qk`/`attn_softmax_bidir`/
//! `attn_apply_full`'s own `bsz` parameter (already plumbed through
//! `crate::block::block_forward`'s attention call at `bsz=1`) - a
//! straightforward follow-up left to whichever phase needs real multi-item
//! batches, not a structural gap in this phase's design.
//!
//! Swedish Embedded AB implements this diffusion transformer assembly for
//! its clients. If your team needs expertise in porting large diffusion
//! models to new inference stacks, you can procure our services by sending
//! an email to info@swedishembedded.com.

use gpu_core::DeviceBuffer;
use model::hostmath::{linear_rows, silu_slice, timestep_embedding};
use vae::blocks::Tensors;

use crate::block::{self, BlockWeights, Ctx, RefinerBlockWeights};
use crate::config::H3TransformerConfig;
use crate::rope;

/// `MiniMax-H3` sinusoidal timestep embedding's `max_period` - the reference
/// `Timesteps`/`get_timestep_embedding` default (`model_channels=freq_dim`,
/// `flip_sin_to_cos=True`, `downscale_freq_shift=0`; nothing in
/// `transformer_minimax_h3.py` overrides `max_period`).
const TIME_MAX_PERIOD: f64 = 10000.0;

/// One packed sequence's forward inputs - every tensor
/// `MiniMaxH3Transformer3DModel.forward` itself takes, minus the batch axis
/// (see this module's own doc).
pub struct PackedInputs<'a> {
    /// `[num_video_tokens, video_patch_dim]`, ordered to match `video_indices`.
    pub hidden_states: &'a [f32],
    /// `[num_audio_tokens, audio_in_channels]`, ordered to match `audio_indices`.
    pub audio_hidden_states: &'a [f32],
    /// `[num_text_tokens, text_dim]`, ordered to match `text_indices`.
    pub encoder_hidden_states: &'a [f32],
    /// `[num_timesteps]` - the DISTINCT timestep values present, in `[0,1]`.
    pub timestep: &'a [f32],
    /// `[seq_len]` - index into `timestep` per packed row.
    pub timestep_indices: &'a [u32],
    /// `[seq_len]` - modality per packed row (`config::TAG_{VIDEO,TEXT,AUDIO}`).
    pub token_tags: &'a [u32],
    /// `[seq_len, 3]` row-major `(t, h, w)` rotary coordinates.
    pub position_ids: &'a [f32],
    pub video_indices: &'a [u32],
    pub audio_indices: &'a [u32],
    pub text_indices: &'a [u32],
}

/// `MiniMaxH3TransformerOutput`, minus the batch axis.
pub struct H3Output {
    /// `[num_video_tokens, video_patch_dim]`, in `video_indices`' order.
    pub video: Vec<f32>,
    /// `[num_audio_tokens, audio_in_channels]`, in `audio_indices`' order.
    pub audio: Vec<f32>,
}

/// This port's own per-row AdaLN table address, `token_tags[row] *
/// num_timesteps + timestep_indices[row]` - MODALITY-major, timestep-minor,
/// deliberately NOT the reference's `timestep_indices*3 + token_tags`
/// (timestep-major). See `crate::block`'s own doc for why: this order is a
/// contiguous host slice of the checkpoint's own fused `adaln_proj.linear`
/// weight, so no interleaving kernel is needed to build the per-block table
/// this indexes into.
pub fn adaln_indices(token_tags: &[u32], timestep_indices: &[u32], num_timesteps: usize) -> Vec<u32> {
    assert_eq!(token_tags.len(), timestep_indices.len(), "adaln_indices: token_tags/timestep_indices length mismatch");
    token_tags.iter().zip(timestep_indices).map(|(&tag, &ti)| tag * num_timesteps as u32 + ti).collect()
}

/// `Linear(in,out).forward` on a SINGLE row, host-side: `y = w @ x + b`,
/// `w` is `[out, in]` row-major.
fn linear1(x: &[f32], w: &[f32], b: &[f32], inn: usize, out: usize) -> Vec<f32> {
    let mut y = linear_rows(x, w, 1, inn, out);
    for (yi, bi) in y.iter_mut().zip(b) {
        *yi += bi;
    }
    y
}

/// `time_proj(timestep) -> time_embedder` for every DISTINCT timestep -
/// `[num_timesteps, time_embed_dim]`, row-major. Host-side: this is a
/// `(num_timesteps,)`-sized computation shared by every block, orders of
/// magnitude smaller than the per-row device work the rest of the forward
/// does (`dit::timestep`'s own precedent for host-side scalar conditioning).
fn build_temb(cfg: &H3TransformerConfig, timestep: &[f32], w0: &[f32], b0: &[f32], w2: &[f32], b2: &[f32]) -> Vec<f32> {
    let mut temb = Vec::with_capacity(timestep.len() * cfg.time_embed_dim as usize);
    for &t in timestep {
        let sinusoid = timestep_embedding(t, cfg.freq_dim as usize, true, 0.0, TIME_MAX_PERIOD);
        let h0 = silu_slice(&linear1(&sinusoid, w0, b0, cfg.freq_dim as usize, cfg.time_embed_hidden_dim as usize));
        let out = linear1(&h0, w2, b2, cfg.time_embed_hidden_dim as usize, cfg.time_embed_dim as usize);
        temb.extend(out);
    }
    temb
}

/// The resident, device-ready form of one loaded H3 DiT core.
pub struct H3Transformer {
    cfg: H3TransformerConfig,
    ctx: Ctx,

    proj_in_w: DeviceBuffer,
    proj_in_b: DeviceBuffer,
    audio_proj_in_w: DeviceBuffer,
    audio_proj_in_b: DeviceBuffer,
    context_embedder_w: DeviceBuffer,
    context_embedder_b: DeviceBuffer,

    time_w0: Vec<f32>,
    time_b0: Vec<f32>,
    time_w2: Vec<f32>,
    time_b2: Vec<f32>,

    refiner_blocks: Vec<RefinerBlockWeights>,
    refiner_final_norm: DeviceBuffer,

    blocks: Vec<BlockWeights>,

    norm_out_norm: DeviceBuffer,
    /// `norm_out.linear.weight`, `[2*hidden, time_embed_dim]` row-major
    /// (host; split at forward time into `shift`/`scale` row halves,
    /// mirroring `crate::block::adaln_tables`'s own per-forward host
    /// projection).
    norm_out_linear_w: Vec<f32>,
    norm_out_linear_b: Vec<f32>,

    proj_out_w: DeviceBuffer,
    proj_out_b: DeviceBuffer,
    audio_proj_out_w: DeviceBuffer,
    audio_proj_out_b: DeviceBuffer,
}

impl H3Transformer {
    /// Load every weight `forward` needs from a flat tensor map (a
    /// `safetensors` dump loaded via `checkpoint::safetensors::read`, or any
    /// other `Tensors` source), named after the reference module's own
    /// attribute paths (`proj_in.weight`, `transformer_blocks.{i}.attn.
    /// to_q.weight`, `token_refiner.refiner_blocks.{i}.norm1.weight`, ...) -
    /// see `tools/minimaxh3_dit_dump_reference.py` for the dump that
    /// produces exactly this naming.
    pub fn load(tensors: &Tensors, cfg: H3TransformerConfig, device: Option<&str>) -> H3Transformer {
        let ctx = Ctx::new(device);
        let get = |name: &str| -> &(Vec<usize>, Vec<f32>) { tensors.get(name).unwrap_or_else(|| panic!("minimaxh3 model: missing tensor {name:?}")) };
        let dev = |name: &str| ctx.upload(&get(name).1);
        let host = |name: &str| get(name).1.clone();

        let (hidden, ffn) = (cfg.hidden_size, cfg.ffn_dim);

        // Split the fused SwiGLU projection's two output-feature halves into
        // separate contiguous buffers (see `block::BlockWeights`'s own doc).
        let load_fc1 = |prefix: &str| -> (DeviceBuffer, DeviceBuffer) {
            let (shape, data) = get(&format!("{prefix}.ff.net.0.proj.weight"));
            assert_eq!(shape, &vec![2 * ffn as usize, hidden as usize], "{prefix}.ff.net.0.proj.weight");
            let half = (ffn * hidden) as usize;
            (ctx.upload(&data[..half]), ctx.upload(&data[half..2 * half]))
        };
        let load_attn = |prefix: &str| -> (DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer) {
            (dev(&format!("{prefix}.to_q.weight")), dev(&format!("{prefix}.to_k.weight")), dev(&format!("{prefix}.to_v.weight")), dev(&format!("{prefix}.norm_q.weight")), dev(&format!("{prefix}.norm_k.weight")), dev(&format!("{prefix}.to_out.0.weight")))
        };

        let refiner_blocks: Vec<RefinerBlockWeights> = (0..cfg.num_refiner_layers)
            .map(|i| {
                let p = format!("token_refiner.refiner_blocks.{i}");
                let (wq, wk, wv, norm_q, norm_k, wo) = load_attn(&format!("{p}.attn"));
                let (fc1_value, fc1_gate) = load_fc1(&p);
                RefinerBlockWeights { wq, wk, wv, wo, norm_q, norm_k, norm1: dev(&format!("{p}.norm1.weight")), norm2: dev(&format!("{p}.norm2.weight")), fc1_value, fc1_gate, fc2: dev(&format!("{p}.ff.net.2.weight")) }
            })
            .collect();

        let blocks: Vec<BlockWeights> = (0..cfg.num_layers)
            .map(|i| {
                let p = format!("transformer_blocks.{i}");
                let (wq, wk, wv, norm_q, norm_k, wo) = load_attn(&format!("{p}.attn"));
                let (fc1_value, fc1_gate) = load_fc1(&p);
                BlockWeights {
                    wq,
                    wk,
                    wv,
                    wo,
                    norm_q,
                    norm_k,
                    norm1: dev(&format!("{p}.norm1.weight")),
                    norm2: dev(&format!("{p}.norm2.weight")),
                    fc1_value,
                    fc1_gate,
                    fc2: dev(&format!("{p}.ff.net.2.weight")),
                    adaln_w: host(&format!("{p}.adaln_proj.linear.weight")),
                    adaln_b: host(&format!("{p}.adaln_proj.linear.bias")),
                }
            })
            .collect();

        H3Transformer {
            proj_in_w: dev("proj_in.weight"),
            proj_in_b: dev("proj_in.bias"),
            audio_proj_in_w: dev("audio_proj_in.weight"),
            audio_proj_in_b: dev("audio_proj_in.bias"),
            context_embedder_w: dev("context_embedder.weight"),
            context_embedder_b: dev("context_embedder.bias"),
            time_w0: host("time_embedder.linear_1.weight"),
            time_b0: host("time_embedder.linear_1.bias"),
            time_w2: host("time_embedder.linear_2.weight"),
            time_b2: host("time_embedder.linear_2.bias"),
            refiner_blocks,
            refiner_final_norm: dev("token_refiner.final_norm.weight"),
            blocks,
            norm_out_norm: dev("norm_out.norm.weight"),
            norm_out_linear_w: host("norm_out.linear.weight"),
            norm_out_linear_b: host("norm_out.linear.bias"),
            proj_out_w: dev("proj_out.weight"),
            proj_out_b: dev("proj_out.bias"),
            audio_proj_out_w: dev("audio_proj_out.weight"),
            audio_proj_out_b: dev("audio_proj_out.bias"),
            cfg,
            ctx,
        }
    }

    pub fn config(&self) -> &H3TransformerConfig {
        &self.cfg
    }

    /// `MiniMaxH3Transformer3DModel.forward`, at batch size 1 (see this
    /// module's own doc).
    pub fn forward(&self, inp: &PackedInputs) -> H3Output {
        self.forward_full(inp).0
    }

    /// [`Self::forward`], plus the intermediate parity taps
    /// `tools/minimaxh3_dit_dump_reference.py` also dumps (post-token-refiner
    /// text stream, post-RoPE `cos`/`sin`, block 0's packed input, block 0's
    /// attention output, block 0's full output) - the numeric parity test's
    /// own entry point, so a mismatch localizes to the first tap it breaks
    /// rather than only showing up in the final output.
    pub fn forward_with_taps(&self, inp: &PackedInputs) -> (H3Output, H3Taps) {
        self.forward_full(inp)
    }

    fn forward_full(&self, inp: &PackedInputs) -> (H3Output, H3Taps) {
        let cfg = &self.cfg;
        let cx = &self.ctx;
        let hidden = cfg.hidden_size;
        let seq_len = (inp.position_ids.len() / 3) as u32;
        assert_eq!(inp.token_tags.len(), seq_len as usize, "forward: token_tags must be [seq_len]");
        assert_eq!(inp.timestep_indices.len(), seq_len as usize, "forward: timestep_indices must be [seq_len]");
        let num_video = inp.video_indices.len() as u32;
        let num_audio = inp.audio_indices.len() as u32;
        let num_text = inp.text_indices.len() as u32;
        let num_timesteps = inp.timestep.len();

        // 1. Per-modality input projections (video/audio directly; text
        // through the token refiner).
        let video_in = cx.upload(inp.hidden_states);
        let video_embeds = block::linear(cx, &video_in, &self.proj_in_w, Some(&self.proj_in_b), num_video, cfg.video_patch_dim(), hidden);
        let audio_in = cx.upload(inp.audio_hidden_states);
        let audio_embeds = block::linear(cx, &audio_in, &self.audio_proj_in_w, Some(&self.audio_proj_in_b), num_audio, cfg.audio_in_channels, hidden);
        let text_in = cx.upload(inp.encoder_hidden_states);
        let mut text_embeds = block::linear(cx, &text_in, &self.context_embedder_w, Some(&self.context_embedder_b), num_text, cfg.text_dim, hidden);
        for w in &self.refiner_blocks {
            text_embeds = block::refiner_block_forward(cx, w, &text_embeds, cfg, num_text);
        }
        let text_embeds = block::rmsnorm(cx, &text_embeds, &self.refiner_final_norm, num_text, hidden, cfg.final_norm_eps);
        let tap_refiner_out = cx.gpu.read(&text_embeds, (num_text * hidden) as usize);

        // 2. Scatter every modality's projected rows into the packed
        // sequence buffer (`index_copy` in the reference).
        let hidden_states = cx.gpu.storage((seq_len * hidden) as u64);
        cx.gpu.write_f32(&hidden_states, &vec![0f32; (seq_len * hidden) as usize]);
        let text_idx = cx.upload_u32(inp.text_indices);
        let video_idx = cx.upload_u32(inp.video_indices);
        let audio_idx = cx.upload_u32(inp.audio_indices);
        block::row_scatter(cx, &text_idx, &text_embeds, &hidden_states, num_text, hidden, seq_len);
        block::row_scatter(cx, &video_idx, &video_embeds, &hidden_states, num_video, hidden, seq_len);
        block::row_scatter(cx, &audio_idx, &audio_embeds, &hidden_states, num_audio, hidden, seq_len);

        // 3. RoPE tables and the shared timestep embedding.
        let rope_tables = rope::build_tables(cfg, inp.position_ids);
        let cos = cx.upload(&rope_tables.cos);
        let sin = cx.upload(&rope_tables.sin);
        let temb = build_temb(cfg, inp.timestep, &self.time_w0, &self.time_b0, &self.time_w2, &self.time_b2);
        let temb_silu = silu_slice(&temb);

        // 4. Row -> this port's own AdaLN table address (see `adaln_indices`'
        // own doc for why this differs from the reference's row order).
        let adaln_idx = adaln_indices(inp.token_tags, inp.timestep_indices, num_timesteps);
        let adaln_idx_dev = cx.upload_u32(&adaln_idx);

        // 5. The block stack.
        let tap_block0_input = cx.gpu.read(&hidden_states, (seq_len * hidden) as usize);
        let mut h = hidden_states;
        let mut tap_block0_attn_out: Vec<f32> = Vec::new();
        let mut tap_block0_out: Vec<f32> = Vec::new();
        for (i, w) in self.blocks.iter().enumerate() {
            let (out, attn_out) = block::block_forward(cx, w, &h, &adaln_idx_dev, &cos, &sin, &temb_silu, num_timesteps, cfg, seq_len);
            if i == 0 {
                tap_block0_attn_out = cx.gpu.read(&attn_out, (seq_len * hidden) as usize);
                tap_block0_out = cx.gpu.read(&out, (seq_len * hidden) as usize);
            }
            h = out;
        }

        // 6. norm_out (per-TIMESTEP shift+scale only, no gate, no modality
        // axis - see `crate::block`'s own module doc) then the two output
        // heads, run over every row, rows of each modality selected after.
        let shift_w = &self.norm_out_linear_w[..(hidden * cfg.time_embed_dim) as usize];
        let scale_w = &self.norm_out_linear_w[(hidden * cfg.time_embed_dim) as usize..];
        let shift_b = &self.norm_out_linear_b[..hidden as usize];
        let scale_b = &self.norm_out_linear_b[hidden as usize..];
        let shift_tbl = linear1_rows(&temb_silu, shift_w, shift_b, num_timesteps, cfg.time_embed_dim as usize, hidden as usize);
        let scale_tbl = linear1_rows(&temb_silu, scale_w, scale_b, num_timesteps, cfg.time_embed_dim as usize, hidden as usize);
        let shift_dev = cx.upload(&shift_tbl);
        let scale_dev = cx.upload(&scale_tbl);
        let ts_idx_dev = cx.upload_u32(inp.timestep_indices);
        let shift_g = block::gather_rows(cx, &ts_idx_dev, &shift_dev, seq_len, hidden);
        let scale_g = block::gather_rows(cx, &ts_idx_dev, &scale_dev, seq_len, hidden);
        let normed = block::rmsnorm(cx, &h, &self.norm_out_norm, seq_len, hidden, cfg.final_norm_eps);
        let modulated = block::modulate(cx, &normed, &scale_g, &shift_g, seq_len * hidden);

        let video_patch_dim = cfg.video_patch_dim();
        let video_full = block::linear(cx, &modulated, &self.proj_out_w, Some(&self.proj_out_b), seq_len, hidden, video_patch_dim);
        let audio_full = block::linear(cx, &modulated, &self.audio_proj_out_w, Some(&self.audio_proj_out_b), seq_len, hidden, cfg.audio_in_channels);
        let video_out = block::gather_rows(cx, &video_idx, &video_full, num_video, video_patch_dim);
        let audio_out = block::gather_rows(cx, &audio_idx, &audio_full, num_audio, cfg.audio_in_channels);

        let output = H3Output { video: cx.gpu.read(&video_out, (num_video * video_patch_dim) as usize), audio: cx.gpu.read(&audio_out, (num_audio * cfg.audio_in_channels) as usize) };
        let taps = H3Taps { refiner_out: tap_refiner_out, rope_cos: rope_tables.cos, rope_sin: rope_tables.sin, block0_input: tap_block0_input, block0_attn_out: tap_block0_attn_out, block0_out: tap_block0_out };
        (output, taps)
    }
}

/// Intermediate parity taps for [`H3Transformer::forward_with_taps`] - see
/// `tools/minimaxh3_dit_dump_reference.py`'s own matching set of taps.
pub struct H3Taps {
    /// `[num_text_tokens, hidden]` - the token refiner's own output, before
    /// scatter into the packed sequence.
    pub refiner_out: Vec<f32>,
    /// `[seq_len, half]` each - `crate::rope::build_tables`'s output.
    pub rope_cos: Vec<f32>,
    pub rope_sin: Vec<f32>,
    /// `[seq_len, hidden]` - the packed sequence buffer block 0 reads (after
    /// every modality's projection, refiner and scatter).
    pub block0_input: Vec<f32>,
    /// `[seq_len, hidden]` - block 0's attention sub-block output (post
    /// `to_out`, pre gated residual).
    pub block0_attn_out: Vec<f32>,
    /// `[seq_len, hidden]` - block 0's full output.
    pub block0_out: Vec<f32>,
}

/// [`linear_rows`] over MULTIPLE rows, with a per-output-feature bias added -
/// `norm_out`'s shift/scale tables need this shape (`linear1` above is
/// deliberately single-row, for the per-timestep temb build).
fn linear1_rows(x: &[f32], w: &[f32], b: &[f32], rows: usize, inn: usize, out: usize) -> Vec<f32> {
    let mut y = linear_rows(x, w, rows, inn, out);
    for row in y.chunks_mut(out) {
        for (yi, bi) in row.iter_mut().zip(b) {
            *yi += bi;
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{TAG_AUDIO, TAG_TEXT, TAG_VIDEO};

    #[test]
    fn adaln_indices_are_modality_major() {
        // 2 distinct timesteps, rows: video@t0, text@t1, audio@t0.
        let idx = adaln_indices(&[TAG_VIDEO, TAG_TEXT, TAG_AUDIO], &[0, 1, 0], 2);
        // video block starts at row 0*2=0 -> video@t0 = row 0
        // text  block starts at row 1*2=2 -> text@t1  = row 2+1=3
        // audio block starts at row 2*2=4 -> audio@t0 = row 4
        assert_eq!(idx, vec![0, 3, 4]);
    }

    fn rand_tensors(cfg: &H3TransformerConfig, seed: u64) -> Tensors {
        let mut rng = data::rng::Lcg::new(seed);
        let mut t: Tensors = std::collections::HashMap::new();
        let mut put = |name: &str, shape: Vec<usize>| {
            let n: usize = shape.iter().product();
            t.insert(name.to_string(), (shape, rng.vec_scaled(n, 0.2)));
        };
        let (hidden, inner, hd, ffn, te) = (cfg.hidden_size as usize, cfg.inner_dim() as usize, cfg.attention_head_dim as usize, cfg.ffn_dim as usize, cfg.time_embed_dim as usize);

        let put_attn = |name: &mut dyn FnMut(&str, Vec<usize>), prefix: &str| {
            name(&format!("{prefix}.to_q.weight"), vec![inner, hidden]);
            name(&format!("{prefix}.to_k.weight"), vec![inner, hidden]);
            name(&format!("{prefix}.to_v.weight"), vec![inner, hidden]);
            name(&format!("{prefix}.norm_q.weight"), vec![hd]);
            name(&format!("{prefix}.norm_k.weight"), vec![hd]);
            name(&format!("{prefix}.to_out.0.weight"), vec![hidden, inner]);
        };
        let put_ff = |name: &mut dyn FnMut(&str, Vec<usize>), prefix: &str| {
            name(&format!("{prefix}.ff.net.0.proj.weight"), vec![2 * ffn, hidden]);
            name(&format!("{prefix}.ff.net.2.weight"), vec![hidden, ffn]);
        };

        for i in 0..cfg.num_refiner_layers {
            let p = format!("token_refiner.refiner_blocks.{i}");
            put_attn(&mut put, &format!("{p}.attn"));
            put_ff(&mut put, &p);
            put(&format!("{p}.norm1.weight"), vec![hidden]);
            put(&format!("{p}.norm2.weight"), vec![hidden]);
        }
        put("token_refiner.final_norm.weight", vec![hidden]);

        for i in 0..cfg.num_layers {
            let p = format!("transformer_blocks.{i}");
            put_attn(&mut put, &format!("{p}.attn"));
            put_ff(&mut put, &p);
            put(&format!("{p}.norm1.weight"), vec![hidden]);
            put(&format!("{p}.norm2.weight"), vec![hidden]);
            put(&format!("{p}.adaln_proj.linear.weight"), vec![18 * hidden, te]);
            put(&format!("{p}.adaln_proj.linear.bias"), vec![18 * hidden]);
        }

        put("proj_in.weight", vec![hidden, cfg.video_patch_dim() as usize]);
        put("proj_in.bias", vec![hidden]);
        put("audio_proj_in.weight", vec![hidden, cfg.audio_in_channels as usize]);
        put("audio_proj_in.bias", vec![hidden]);
        put("context_embedder.weight", vec![hidden, cfg.text_dim as usize]);
        put("context_embedder.bias", vec![hidden]);
        put("time_embedder.linear_1.weight", vec![cfg.time_embed_hidden_dim as usize, cfg.freq_dim as usize]);
        put("time_embedder.linear_1.bias", vec![cfg.time_embed_hidden_dim as usize]);
        put("time_embedder.linear_2.weight", vec![te, cfg.time_embed_hidden_dim as usize]);
        put("time_embedder.linear_2.bias", vec![te]);
        put("norm_out.norm.weight", vec![hidden]);
        put("norm_out.linear.weight", vec![2 * hidden, te]);
        put("norm_out.linear.bias", vec![2 * hidden]);
        put("proj_out.weight", vec![cfg.video_patch_dim() as usize, hidden]);
        put("proj_out.bias", vec![cfg.video_patch_dim() as usize]);
        put("audio_proj_out.weight", vec![cfg.audio_in_channels as usize, hidden]);
        put("audio_proj_out.bias", vec![cfg.audio_in_channels as usize]);
        t
    }

    /// Weight-free tiny-config smoke test (porting.md's own ladder rung,
    /// before any real/golden weights exist): builds a full `H3Transformer`
    /// at [`H3TransformerConfig::tiny`] with seeded random weights, runs one
    /// forward over a small synthetic packed sequence spanning all three
    /// modalities and TWO distinct timesteps (to genuinely exercise per-row
    /// AdaLN indexing, not just a single shared modulation vector), and
    /// asserts the outputs are finite, correctly shaped, and not trivially
    /// all-zero.
    #[test]
    fn tiny_config_forward_is_finite_and_the_right_shape() {
        let cfg = H3TransformerConfig::tiny();
        let tensors = rand_tensors(&cfg, 3);
        let model = H3Transformer::load(&tensors, cfg, Some("cpu"));
        let cfg = model.config();

        let num_video = 4u32;
        let num_audio = 2u32;
        let num_text = 3u32;
        let seq_len = num_text + num_audio + num_video;

        let mut rng = data::rng::Lcg::new(9);
        let hidden_states = rng.vec_scaled((num_video * cfg.video_patch_dim()) as usize, 0.3);
        let audio_hidden_states = rng.vec_scaled((num_audio * cfg.audio_in_channels) as usize, 0.3);
        let encoder_hidden_states = rng.vec_scaled((num_text * cfg.text_dim) as usize, 0.3);

        let text_indices: Vec<u32> = (0..num_text).collect();
        let audio_indices: Vec<u32> = (num_text..num_text + num_audio).collect();
        let video_indices: Vec<u32> = (num_text + num_audio..seq_len).collect();

        let mut token_tags = vec![0u32; seq_len as usize];
        let mut timestep_indices = vec![0u32; seq_len as usize];
        for &i in &text_indices {
            token_tags[i as usize] = TAG_TEXT;
        }
        for (n, &i) in audio_indices.iter().enumerate() {
            token_tags[i as usize] = TAG_AUDIO;
            timestep_indices[i as usize] = (n % 2) as u32; // exercise both timesteps
        }
        for (n, &i) in video_indices.iter().enumerate() {
            token_tags[i as usize] = TAG_VIDEO;
            timestep_indices[i as usize] = (n % 2) as u32;
        }

        let mut position_ids = vec![0f32; seq_len as usize * 3];
        for r in 0..seq_len as usize {
            position_ids[r * 3] = r as f32;
            position_ids[r * 3 + 1] = (r % 3) as f32;
            position_ids[r * 3 + 2] = (r % 2) as f32;
        }

        let timestep = vec![0.2f32, 0.8f32];

        let inp = PackedInputs {
            hidden_states: &hidden_states,
            audio_hidden_states: &audio_hidden_states,
            encoder_hidden_states: &encoder_hidden_states,
            timestep: &timestep,
            timestep_indices: &timestep_indices,
            token_tags: &token_tags,
            position_ids: &position_ids,
            video_indices: &video_indices,
            audio_indices: &audio_indices,
            text_indices: &text_indices,
        };

        let out = model.forward(&inp);
        assert_eq!(out.video.len(), (num_video * cfg.video_patch_dim()) as usize);
        assert_eq!(out.audio.len(), (num_audio * cfg.audio_in_channels) as usize);
        assert!(out.video.iter().all(|v| v.is_finite()), "video output must be finite: {:?}", out.video);
        assert!(out.audio.iter().all(|v| v.is_finite()), "audio output must be finite: {:?}", out.audio);
        assert!(out.video.iter().any(|&v| v != 0.0), "video output must not be trivially all-zero");
        assert!(out.audio.iter().any(|&v| v != 0.0), "audio output must not be trivially all-zero");
    }

    /// Real NUMERIC parity against `tools/minimaxh3_dit_dump_reference.py`'s
    /// golden - the ACTUAL installed `diffusers==0.40.0`
    /// `MiniMaxH3Transformer3DModel`, run for real at the tiny config and
    /// dumped, matching porting.md's rung-3 ("single-forward parity: replay
    /// the hooked reference inputs through the full model"). Five taps plus
    /// the two final outputs, so a mismatch localizes to the first tap it
    /// breaks rather than only showing up in the final cosine.
    #[test]
    fn tiny_config_matches_the_real_reference_numerically() {
        let fixture_dir = brain_testutil::testdata_path("golden/minimaxh3/dit_tiny");
        let fixture_file = fixture_dir.join("minimaxh3_dit_tiny.safetensors");
        if !fixture_file.is_file() {
            brain_testutil::skip(&format!("{} not found - run tools/minimaxh3_dit_dump_reference.py --out {}", fixture_file.display(), fixture_dir.display()));
            return;
        }

        let cfg = H3TransformerConfig::tiny();

        let Some(src) = brain_testutil::golden::Source::open(&fixture_dir, "tools/minimaxh3_dit_dump_reference.py") else {
            return;
        };
        let ok = src.require(&[
            ("num_attention_heads", cfg.num_attention_heads as i64),
            ("attention_head_dim", cfg.attention_head_dim as i64),
            ("hidden_size", cfg.hidden_size as i64),
            ("num_layers", cfg.num_layers as i64),
            ("num_refiner_layers", cfg.num_refiner_layers as i64),
            ("ffn_dim", cfg.ffn_dim as i64),
            ("in_channels", cfg.in_channels as i64),
            ("audio_in_channels", cfg.audio_in_channels as i64),
            ("text_dim", cfg.text_dim as i64),
            ("freq_dim", cfg.freq_dim as i64),
            ("time_embed_hidden_dim", cfg.time_embed_hidden_dim as i64),
            ("time_embed_dim", cfg.time_embed_dim as i64),
            ("rope_freq_dim", cfg.rope_freq_dim as i64),
            ("rope_theta_x1000", (cfg.rope_theta * 1000.0) as i64),
        ]);
        if !ok {
            return;
        }

        let raw = checkpoint::safetensors::read(fixture_file.to_str().expect("fixture path is valid UTF-8")).expect("read golden fixture");
        let fx: std::collections::HashMap<String, checkpoint::safetensors::StTensor> = raw.into_iter().map(|t| (t.name.clone(), t)).collect();
        let get = |name: &str| -> &[f32] { &fx.get(name).unwrap_or_else(|| panic!("golden fixture tap {name:?} missing")).data };
        let get_u32 = |name: &str| -> Vec<u32> { get(name).iter().map(|&v| v.round() as u32).collect() };

        // Every non-input/output/tap entry is a weight, named after the
        // reference module's own attribute paths - `H3Transformer::load`
        // reads them directly with no renaming (see its own doc).
        let mut tensors: Tensors = std::collections::HashMap::new();
        for (name, t) in &fx {
            if !name.starts_with("input_") && !name.starts_with("output_") && !name.starts_with("tap_") {
                tensors.insert(name.clone(), (t.shape.clone(), t.data.clone()));
            }
        }

        let model = H3Transformer::load(&tensors, cfg, Some("cpu"));

        let inp = PackedInputs {
            hidden_states: get("input_hidden_states"),
            audio_hidden_states: get("input_audio_hidden_states"),
            encoder_hidden_states: get("input_encoder_hidden_states"),
            timestep: get("input_timestep"),
            timestep_indices: &get_u32("input_timestep_indices"),
            token_tags: &get_u32("input_token_tags"),
            position_ids: get("input_position_ids"),
            video_indices: &get_u32("input_video_indices"),
            audio_indices: &get_u32("input_audio_indices"),
            text_indices: &get_u32("input_text_indices"),
        };

        let (out, taps) = model.forward_with_taps(&inp);

        // The reference's own `rope.forward` doubles each row's `half`
        // angles to `2*half` before taking cos/sin (the `rotate_half`
        // convention - see `crate::rope`'s own doc for why this port stores
        // only the `half`-wide table). Take the first `half` columns of each
        // row of the golden's `2*half`-wide dump so both sides compare the
        // exact same angles.
        let seq_len = get("input_position_ids").len() / 3;
        let half = taps.rope_cos.len() / seq_len;
        let first_half_cols = |full: &[f32]| -> Vec<f32> { (0..seq_len).flat_map(|r| full[r * 2 * half..r * 2 * half + half].to_vec()).collect() };

        let mut r = brain_testutil::parity::Report::new(0.9999);
        r.check("tap_refiner_out", &taps.refiner_out, get("tap_refiner_out"));
        r.check("tap_rope_cos", &taps.rope_cos, &first_half_cols(get("tap_rope_cos")));
        r.check("tap_rope_sin", &taps.rope_sin, &first_half_cols(get("tap_rope_sin")));
        r.check("tap_block0_input", &taps.block0_input, get("tap_block0_input"));
        r.check("tap_block0_attn_out", &taps.block0_attn_out, get("tap_block0_attn_out"));
        r.check("tap_block0_out", &taps.block0_out, get("tap_block0_out"));
        r.check("output_video", &out.video, get("output_video"));
        r.check("output_audio", &out.audio, get("output_audio"));
        r.finish("minimaxh3 DiT core tiny-config forward vs real reference");
    }
}
