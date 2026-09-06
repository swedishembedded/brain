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
#[cfg(test)]
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
    /// Load every weight `forward` needs from a [`checkpoint::TensorSource`]
    /// (a streaming [`checkpoint::weightio::WeightReader`], or the eager
    /// `Tensors` map every existing caller already had - it implements the
    /// trait, so this widened signature is a strict superset, not a breaking
    /// change), named after the reference module's own attribute paths
    /// (`proj_in.weight`, `transformer_blocks.{i}.attn.to_q.weight`,
    /// `token_refiner.refiner_blocks.{i}.norm1.weight`, ...) - see
    /// `tools/minimaxh3_dit_dump_reference.py` for the dump that produces
    /// exactly this naming.
    ///
    /// **Streams, does not materialize the whole checkpoint first**: this
    /// model is ~33B params (bf16 on disk), the same eager-`read_model_dir`
    /// mistake `qwen3vl::Qwen3Vl::from_hf` used to make would cost ~132GB
    /// just for the source before this function uploads a single byte (that
    /// exact mistake measurably OOMed a 150GB container cap on the
    /// text encoder before it was fixed the same way). Each tensor is
    /// fetched, uploaded,
    /// and `advise_drop`-ped before the next is read - peak host cost from
    /// THIS function is one tensor's transient plus whatever `ctx.upload`
    /// leaves resident (currently always fp32 - `Ctx::upload` has no
    /// reduced-precision tier yet, unlike `qwen3::model::Weight::upload`;
    /// see the roadmap's own recorded gap for that follow-up).
    pub fn load(tensors: &dyn checkpoint::TensorSource, cfg: H3TransformerConfig, device: Option<&str>) -> H3Transformer {
        let ctx = Ctx::new(device);
        let dev = |name: &str| block::load_dev(tensors, &ctx, name);
        let host = |name: &str| block::load_host(tensors, name);
        let (hidden, ffn) = (cfg.hidden_size, cfg.ffn_dim);

        let refiner_blocks: Vec<RefinerBlockWeights> = (0..cfg.num_refiner_layers as usize).map(|i| block::load_refiner_block(tensors, &ctx, i, hidden, ffn)).collect();
        let blocks: Vec<BlockWeights> = (0..cfg.num_layers as usize).map(|i| block::load_block(tensors, &ctx, i, hidden, ffn)).collect();

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

    /// [`Self::forward_with_taps`] WITHOUT ever building a resident
    /// [`H3Transformer`] - no `Vec<BlockWeights>` for all `num_layers`
    /// blocks is ever alive at once. Loads (via
    /// [`block::load_block_streaming`]) one block's weights into a set of
    /// device buffers reused across every block AND every denoise step,
    /// runs it, then overwrites those same buffers with the next block's
    /// weights.
    ///
    /// This exists for exactly one reason: a real-weight VALIDATION run
    /// that only checks a handful of tap points (block 0, a middle block,
    /// the last block, the final output) has no business holding the whole
    /// ~132GB fp32 model resident to reach them - peak host cost here is
    /// ONE block's weights (~2.6GB at real dimensions) plus the small
    /// always-resident pieces (input/output projections, refiner, time
    /// embedder - hundreds of MB), not the whole model. See
    /// `model::tests::dit_matches_the_real_reference_numerically_layer_by_layer`,
    /// the caller this was built for.
    ///
    /// Deliberately a SEPARATE implementation from [`Self::forward_full`]
    /// rather than a refactor of it into one shared code path: the two
    /// differ only in where block weights come from (already-resident
    /// `&self.blocks` vs. freshly streamed-and-dropped per index), but
    /// unifying that behind one abstraction over "owned vs. borrowed
    /// `BlockWeights`" was judged a bigger, riskier change to the
    /// already-working resident-serving path than the two implementations'
    /// duplication costs - if `forward_full`'s math ever changes, this
    /// function's own doc (this comment) is the reminder to check whether
    /// the same change belongs here too.
    /// [`Self::forward_streaming_with_taps`] without the taps - `pipeline::
    /// generate`'s own real (non-"hot"/resident) denoise loop calls this once
    /// per step, since nothing there consumes intermediate taps and there is
    /// no resident model to call [`Self::forward`] on. This is what lets
    /// `device` name a real GPU for a real generation at all: the eager
    /// [`Self::load`] path this replaces needs every block's weights
    /// resident at once (~132GB fp32), which fits in this host's RAM but no
    /// single GPU's VRAM on this box; streaming caps the per-step footprint
    /// at one block's weights (~2.6GB at real dimensions) regardless of
    /// which device runs the compute.
    ///
    /// Takes an ALREADY-OPEN `&Ctx`, never a device string - a caller
    /// driving a real multi-step denoise loop must open the device ONCE
    /// outside the loop and reuse it every step. A real-weight 384x384/
    /// 16-step generation OOM'd a 24GB P40 when this instead called
    /// `Ctx::new(device)` internally on every step (opening 16 separate wgpu
    /// devices in sequence, with no guarantee the previous one's resources
    /// were reclaimed before the next opened) - the fix is structural, not a
    /// tighter memory budget: one device, reused.
    pub fn forward_streaming(tensors: &dyn checkpoint::TensorSource, cfg: &H3TransformerConfig, ctx: &Ctx, inp: &PackedInputs) -> H3Output {
        Self::forward_streaming_with_taps(tensors, cfg, ctx, inp).0
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_streaming_with_taps(tensors: &dyn checkpoint::TensorSource, cfg: &H3TransformerConfig, ctx: &Ctx, inp: &PackedInputs) -> (H3Output, H3Taps) {
        let cx = ctx;
        let dev = |name: &str| block::load_dev(tensors, ctx, name);
        let host = |name: &str| block::load_host(tensors, name);
        let hidden = cfg.hidden_size;

        let seq_len = (inp.position_ids.len() / 3) as u32;
        assert_eq!(inp.token_tags.len(), seq_len as usize, "forward_streaming: token_tags must be [seq_len]");
        assert_eq!(inp.timestep_indices.len(), seq_len as usize, "forward_streaming: timestep_indices must be [seq_len]");
        let num_video = inp.video_indices.len() as u32;
        let num_audio = inp.audio_indices.len() as u32;
        let num_text = inp.text_indices.len() as u32;
        let num_timesteps = inp.timestep.len();

        // 1. Per-modality input projections (video/audio directly; text
        // through the token refiner) - small, always-resident weights.
        let proj_in_w = dev("proj_in.weight");
        let proj_in_b = dev("proj_in.bias");
        let audio_proj_in_w = dev("audio_proj_in.weight");
        let audio_proj_in_b = dev("audio_proj_in.bias");
        let context_embedder_w = dev("context_embedder.weight");
        let context_embedder_b = dev("context_embedder.bias");
        let refiner_blocks: Vec<RefinerBlockWeights> = (0..cfg.num_refiner_layers as usize).map(|i| block::load_refiner_block(tensors, ctx, i, hidden, cfg.ffn_dim)).collect();
        let refiner_final_norm = dev("token_refiner.final_norm.weight");

        let video_in = cx.upload(inp.hidden_states);
        let video_embeds = block::linear(cx, &video_in, &proj_in_w, Some(&proj_in_b), num_video, cfg.video_patch_dim(), hidden);
        let audio_in = cx.upload(inp.audio_hidden_states);
        let audio_embeds = block::linear(cx, &audio_in, &audio_proj_in_w, Some(&audio_proj_in_b), num_audio, cfg.audio_in_channels, hidden);
        let text_in = cx.upload(inp.encoder_hidden_states);
        let mut text_embeds = block::linear(cx, &text_in, &context_embedder_w, Some(&context_embedder_b), num_text, cfg.text_dim, hidden);
        for w in &refiner_blocks {
            text_embeds = block::refiner_block_forward(cx, w, &text_embeds, cfg, num_text);
        }
        let text_embeds = block::rmsnorm(cx, &text_embeds, &refiner_final_norm, num_text, hidden, cfg.final_norm_eps);
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
        let time_w0 = host("time_embedder.linear_1.weight");
        let time_b0 = host("time_embedder.linear_1.bias");
        let time_w2 = host("time_embedder.linear_2.weight");
        let time_b2 = host("time_embedder.linear_2.bias");
        let temb = build_temb(cfg, inp.timestep, &time_w0, &time_b0, &time_w2, &time_b2);
        let temb_silu = silu_slice(&temb);

        // 4. Row -> this port's own AdaLN table address.
        let adaln_idx = adaln_indices(inp.token_tags, inp.timestep_indices, num_timesteps);
        let adaln_idx_dev = cx.upload_u32(&adaln_idx);

        // 5. The block stack - ONE block's weights resident at a time.
        let tap_block0_input = cx.gpu.read(&hidden_states, (seq_len * hidden) as usize);
        let n_layers = cfg.num_layers as usize;
        let mid_block_index = n_layers / 2;
        let last_block_index = n_layers - 1;
        let mut h = hidden_states;
        let mut tap_block0_attn_out: Vec<f32> = Vec::new();
        let mut tap_block0_out: Vec<f32> = Vec::new();
        let mut tap_mid_block_out: Vec<f32> = Vec::new();
        let mut tap_last_block_out: Vec<f32> = Vec::new();
        for i in 0..n_layers {
            let w = block::load_block_streaming(tensors, ctx, i, hidden, cfg.ffn_dim);
            let (out, attn_out) = block::block_forward(cx, &w, &h, &adaln_idx_dev, &cos, &sin, &temb_silu, num_timesteps, cfg, seq_len);
            if i == 0 {
                tap_block0_attn_out = cx.gpu.read(&attn_out, (seq_len * hidden) as usize);
                tap_block0_out = cx.gpu.read(&out, (seq_len * hidden) as usize);
            }
            if i == mid_block_index {
                tap_mid_block_out = cx.gpu.read(&out, (seq_len * hidden) as usize);
            }
            if i == last_block_index {
                tap_last_block_out = cx.gpu.read(&out, (seq_len * hidden) as usize);
            }
            h = out;
            // `w`'s device buffers are now REUSED across blocks (see
            // `block::load_block_streaming`'s doc), not freshly allocated
            // and dropped each iteration, so this call is no longer about
            // reclaiming per-block buffers - it still matters for `out`/
            // `attn_out`/`h`'s own allocations (one fresh buffer per block
            // for the block's output) and for the intermediate scratch
            // buffers `block_forward` allocates internally, which are NOT
            // pooled and must still be proven-finished before wgpu can
            // recycle them. `poll_wait` is a no-op on the CPU backend
            // (`backend_cpu::CpuBackend::poll_wait`, `HashMap`-backed
            // buffers need no such proof), so this costs nothing there.
            cx.gpu.poll_wait();
        }

        // 6. norm_out (per-TIMESTEP shift+scale only, no gate, no modality
        // axis) then the two output heads, run over every row, rows of each
        // modality selected after.
        let norm_out_norm = dev("norm_out.norm.weight");
        let norm_out_linear_w = host("norm_out.linear.weight");
        let norm_out_linear_b = host("norm_out.linear.bias");
        let proj_out_w = dev("proj_out.weight");
        let proj_out_b = dev("proj_out.bias");
        let audio_proj_out_w = dev("audio_proj_out.weight");
        let audio_proj_out_b = dev("audio_proj_out.bias");

        let shift_w = &norm_out_linear_w[..(hidden * cfg.time_embed_dim) as usize];
        let scale_w = &norm_out_linear_w[(hidden * cfg.time_embed_dim) as usize..];
        let shift_b = &norm_out_linear_b[..hidden as usize];
        let scale_b = &norm_out_linear_b[hidden as usize..];
        let shift_tbl = linear1_rows(&temb_silu, shift_w, shift_b, num_timesteps, cfg.time_embed_dim as usize, hidden as usize);
        let scale_tbl = linear1_rows(&temb_silu, scale_w, scale_b, num_timesteps, cfg.time_embed_dim as usize, hidden as usize);
        let shift_dev = cx.upload(&shift_tbl);
        let scale_dev = cx.upload(&scale_tbl);
        let ts_idx_dev = cx.upload_u32(inp.timestep_indices);
        let shift_g = block::gather_rows(cx, &ts_idx_dev, &shift_dev, seq_len, hidden);
        let scale_g = block::gather_rows(cx, &ts_idx_dev, &scale_dev, seq_len, hidden);
        let normed = block::rmsnorm(cx, &h, &norm_out_norm, seq_len, hidden, cfg.final_norm_eps);
        let modulated = block::modulate(cx, &normed, &scale_g, &shift_g, seq_len * hidden);

        let video_patch_dim = cfg.video_patch_dim();
        let video_full = block::linear(cx, &modulated, &proj_out_w, Some(&proj_out_b), seq_len, hidden, video_patch_dim);
        let audio_full = block::linear(cx, &modulated, &audio_proj_out_w, Some(&audio_proj_out_b), seq_len, hidden, cfg.audio_in_channels);
        let video_out = block::gather_rows(cx, &video_idx, &video_full, num_video, video_patch_dim);
        let audio_out = block::gather_rows(cx, &audio_idx, &audio_full, num_audio, cfg.audio_in_channels);

        let output = H3Output { video: cx.gpu.read(&video_out, (num_video * video_patch_dim) as usize), audio: cx.gpu.read(&audio_out, (num_audio * cfg.audio_in_channels) as usize) };

        // Same reasoning as the per-block poll_wait above, but for the
        // "outer" buffers allocated once per call (norm_out/proj_out
        // weights, shift/scale tables, gather results) rather than once
        // per block: this function now runs once per denoising step, so
        // without a final reclaim point these residual buffers compound
        // step over step even though the per-block leak is already
        // handled above (measured: a 384x384/16-step generation still
        // OOM'd, later than before the per-block fix but not fixed).
        cx.gpu.poll_wait();

        let taps = H3Taps {
            refiner_out: tap_refiner_out,
            rope_cos: rope_tables.cos,
            rope_sin: rope_tables.sin,
            temb: temb.clone(),
            block0_input: tap_block0_input,
            block0_attn_out: tap_block0_attn_out,
            block0_out: tap_block0_out,
            mid_block_out: tap_mid_block_out,
            mid_block_index,
            last_block_out: tap_last_block_out,
        };
        (output, taps)
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
        let mid_block_index = self.blocks.len() / 2;
        let last_block_index = self.blocks.len() - 1;
        let mut h = hidden_states;
        let mut tap_block0_attn_out: Vec<f32> = Vec::new();
        let mut tap_block0_out: Vec<f32> = Vec::new();
        let mut tap_mid_block_out: Vec<f32> = Vec::new();
        let mut tap_last_block_out: Vec<f32> = Vec::new();
        for (i, w) in self.blocks.iter().enumerate() {
            let (out, attn_out) = block::block_forward(cx, w, &h, &adaln_idx_dev, &cos, &sin, &temb_silu, num_timesteps, cfg, seq_len);
            if i == 0 {
                tap_block0_attn_out = cx.gpu.read(&attn_out, (seq_len * hidden) as usize);
                tap_block0_out = cx.gpu.read(&out, (seq_len * hidden) as usize);
            }
            if i == mid_block_index {
                tap_mid_block_out = cx.gpu.read(&out, (seq_len * hidden) as usize);
            }
            if i == last_block_index {
                tap_last_block_out = cx.gpu.read(&out, (seq_len * hidden) as usize);
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
        let taps = H3Taps {
            refiner_out: tap_refiner_out,
            rope_cos: rope_tables.cos,
            rope_sin: rope_tables.sin,
            temb: temb.clone(),
            block0_input: tap_block0_input,
            block0_attn_out: tap_block0_attn_out,
            block0_out: tap_block0_out,
            mid_block_out: tap_mid_block_out,
            mid_block_index,
            last_block_out: tap_last_block_out,
        };
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
    /// `[num_timesteps, time_embed_dim]` - `time_embedder`'s own raw output
    /// (before the SiLU this module's own caller applies - matches the
    /// reference's `register_forward_hook` on `model.time_embedder` itself,
    /// which fires on that module's return value, not on whatever the
    /// caller does with it afterward).
    pub temb: Vec<f32>,
    /// `[seq_len, hidden]` - the packed sequence buffer block 0 reads (after
    /// every modality's projection, refiner and scatter).
    pub block0_input: Vec<f32>,
    /// `[seq_len, hidden]` - block 0's attention sub-block output (post
    /// `to_out`, pre gated residual).
    pub block0_attn_out: Vec<f32>,
    /// `[seq_len, hidden]` - block 0's full output.
    pub block0_out: Vec<f32>,
    /// `[seq_len, hidden]` - output of block `mid_block_index` (0-indexed,
    /// `num_layers/2`) - catches a bug that only manifests after several
    /// blocks' worth of accumulated state (e.g. a per-block AdaLN indexing
    /// bug), which a block-0-only tap cannot.
    pub mid_block_out: Vec<f32>,
    pub mid_block_index: usize,
    /// `[seq_len, hidden]` - output of the LAST block (`num_layers - 1`).
    pub last_block_out: Vec<f32>,
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

    /// The REAL checkpoint's DiT, real weights, against a live real-weight
    /// `diffusers` reference run - `tools/minimaxh3_dit_real_dump_reference.py`,
    /// NOT `minimaxh3_dit_dump_reference.py` (that one uses tiny dims and
    /// small RANDOM weights, which validates the architecture code but
    /// structurally cannot catch a bug that only shows up on real,
    /// structured weights - a wrong transpose, a wrong axis order, a wrong
    /// tensor-name-to-role mapping all look "fine" when every element is
    /// equally arbitrary). Checked at block 0, a MIDDLE block, and the LAST
    /// block (not just end-to-end) - exactly the audio VAE's own precedent
    /// (`import::decode_matches_the_real_reference_numerically`) applied to
    /// the DiT for the first time.
    ///
    /// Real DiT weight loading (~66GB bf16 -> ~132GB fp32, streamed) is the
    /// expensive part here (~10-20 minutes) - the forward pass itself runs
    /// over a 9-row synthetic sequence, seconds not minutes.
    #[test]
    fn dit_matches_the_real_reference_numerically_layer_by_layer() {
        let Ok(root) = std::env::var("BRAIN_MINIMAXH3_DIR") else {
            brain_testutil::skip("BRAIN_MINIMAXH3_DIR not set - no local MiniMax-H3 checkout to load from");
            return;
        };
        let paths = crate::caps::Paths::resolve(&root);
        if !checkpoint::safetensors::has_model_weights(std::path::Path::new(&paths.dit)) {
            brain_testutil::skip(&format!("{} has no complete weight set yet - checkpoint not (yet) downloaded", paths.dit));
            return;
        }

        let fixture_dir = brain_testutil::testdata_path("golden/minimaxh3/dit_real");
        let fixture_file = fixture_dir.join("minimaxh3_dit_real.safetensors");
        if !fixture_file.is_file() {
            brain_testutil::skip(&format!(
                "{} not found - run tools/minimaxh3_dit_real_dump_reference.py --checkpoint {} --out {}",
                fixture_file.display(),
                paths.dit,
                fixture_dir.display()
            ));
            return;
        }

        let cfg = H3TransformerConfig::real();

        // Golden/checkpoint pairing, proven rather than assumed - see
        // brain_testutil::golden's own module doc.
        let Some(src) = brain_testutil::golden::Source::open(&fixture_dir, "tools/minimaxh3_dit_real_dump_reference.py") else {
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

        let fx = checkpoint::safetensors::read(fixture_file.to_str().expect("fixture path is valid UTF-8")).expect("read golden fixture");
        let get = |name: &str| -> &checkpoint::safetensors::StTensor {
            fx.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("golden fixture tap {name:?} missing"))
        };
        let get_u32 = |name: &str| -> Vec<u32> { get(name).data.iter().map(|&f| f.round() as u32).collect() };

        let hidden_states = get("input_hidden_states").data.clone();
        let audio_hidden_states = get("input_audio_hidden_states").data.clone();
        let encoder_hidden_states = get("input_encoder_hidden_states").data.clone();
        let timestep = get("input_timestep").data.clone();
        let timestep_indices = get_u32("input_timestep_indices");
        let token_tags = get_u32("input_token_tags");
        let position_ids = get("input_position_ids").data.clone();
        let video_indices = get_u32("input_video_indices");
        let audio_indices = get_u32("input_audio_indices");
        let text_indices = get_u32("input_text_indices");

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

        // Defaults to "cpu" (this crate's own CI-safe device) - overridable
        // via `BRAIN_MINIMAXH3_TEST_DEVICE` (e.g. "vulkan") for a manual
        // real-weight parity check on an actual GPU, since
        // `forward_streaming` had never been run on real hardware other than
        // CPU before `pipeline::generate` started calling it there too.
        let device_str = std::env::var("BRAIN_MINIMAXH3_TEST_DEVICE").unwrap_or_else(|_| "cpu".to_string());
        eprintln!("running real DiT one block at a time on device={device_str:?} (never all 50 blocks resident - see H3Transformer::forward_streaming_with_taps's own doc) ...");
        let t0 = std::time::Instant::now();
        let reader = crate::caps::open_dit_reader(&paths.dit).unwrap_or_else(|e| panic!("open_dit_reader: {e}"));
        let ctx = crate::block::Ctx::new(Some(&device_str));
        let (out, taps) = H3Transformer::forward_streaming_with_taps(&reader, &cfg, &ctx, &inp);
        eprintln!("  done in {:.1}s", t0.elapsed().as_secs_f32());

        assert_eq!(taps.mid_block_index, 25, "golden was dumped at mid_block=25; H3TransformerConfig::real()'s own 50 layers must still land on 25");

        // porting.md's own floor: "cosine >= 0.9999 for networks" at the
        // stage-parity rung. This is stage-by-stage AND real-weight, so it
        // should land far tighter than that floor (float32-noise level, the
        // audio VAE's own real-weight precedent landed at max_abs ~4e-6) -
        // the floor is a refusal threshold, not a target.
        // Same `2*half`-wide reference doubling as `tiny_config_matches_the_
        // real_reference_numerically` below - see `crate::rope`'s own doc.
        let seq_len = position_ids.len() / 3;
        let half = taps.rope_cos.len() / seq_len;
        let first_half_cols = |full: &[f32]| -> Vec<f32> { (0..seq_len).flat_map(|r| full[r * 2 * half..r * 2 * half + half].to_vec()).collect() };

        let mut r = brain_testutil::parity::Report::new(0.9999);
        r.check("refiner_out", &taps.refiner_out, &get("tap_refiner_out").data);
        r.check("rope_cos", &taps.rope_cos, &first_half_cols(&get("tap_rope_cos").data));
        r.check("rope_sin", &taps.rope_sin, &first_half_cols(&get("tap_rope_sin").data));
        r.check("temb", &taps.temb, &get("tap_temb").data);
        r.check("block0_input", &taps.block0_input, &get("tap_block0_input").data);
        r.check("block0_attn_out", &taps.block0_attn_out, &get("tap_block0_attn_out").data);
        r.check("block0_out", &taps.block0_out, &get("tap_block0_out").data);
        r.check("block25_out (mid)", &taps.mid_block_out, &get("tap_block25_out").data);
        r.check("block49_out (last)", &taps.last_block_out, &get("tap_block49_out").data);
        r.check("output_video", &out.video, &get("output_video").data);
        r.check("output_audio", &out.audio, &get("output_audio").data);
        r.finish("minimaxh3 DiT forward vs REAL-WEIGHT reference, layer by layer");
    }

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
