// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniMax-H3's full DiT core: config, RoPE, the token refiner, the
//! `num_layers` block stack, the two input projections, `context_embedder`,
//! `norm_out` and the two output heads, assembled per
//! `MiniMaxH3Transformer3DModel.forward` (`transformer_minimax_h3.py`).
//!
//! **Packed-sequence LAYOUT is not built here** - [`crate::pipeline`]'s
//! `build_packed_sequence` owns it:
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
//! straightforward follow-up when real multi-item batches are needed, not a
//! structural gap in this design.
//!
//! Swedish Embedded AB implements this diffusion transformer assembly for
//! its clients. If your team needs expertise in porting large diffusion
//! models to new inference stacks, you can procure our services by sending
//! an email to info@swedishembedded.com.

use std::borrow::Cow;
use std::sync::Mutex;

use gpu_core::select::Dtype;
use gpu_core::DeviceBuffer;
use model::hostmath::{linear_rows, silu_slice, timestep_embedding};
use model::ops::Weight;
use model::shard::LayerBytes;
use model::StreamPlan;
#[cfg(test)]
use vae::blocks::Tensors;

use crate::adaln::{AdalnCache, AdalnPlan};
use crate::block::{self, BlockWeights, Ctx, RefinerBlockWeights};
use crate::config::H3TransformerConfig;
use crate::dit_shard::StreamingDit;
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
pub(crate) fn linear1(x: &[f32], w: &[f32], b: &[f32], inn: usize, out: usize) -> Vec<f32> {
    let mut y = linear_rows(x, w, 1, inn, out);
    for (yi, bi) in y.iter_mut().zip(b) {
        *yi += bi;
    }
    y
}

/// `time_embedder(time_proj(t))` for ONE raw timestep scalar, host-side -
/// `time_w0`/`time_b0`/`time_w2`/`time_b2` are `time_embedder.linear_1`/`linear_2`'s
/// weights. Returns the RAW `linear_2` output: `time_embedder` itself ends
/// there (the reference's `TimestepEmbedding` has no trailing activation), and
/// the SiLU every consumer wants belongs to the consumer - the AdaLN
/// projections apply `silu(temb)` themselves, exactly once. Applying it here
/// as well double-activated every block's modulation. `pub(crate)`:
/// [`crate::adaln`] evaluates it at every timestep value it caches.
pub(crate) fn temb_row(cfg: &H3TransformerConfig, t: f32, w0: &[f32], b0: &[f32], w2: &[f32], b2: &[f32]) -> Vec<f32> {
    let sinusoid = timestep_embedding(t, cfg.freq_dim as usize, true, 0.0, TIME_MAX_PERIOD);
    let h0 = silu_slice(&linear1(&sinusoid, w0, b0, cfg.freq_dim as usize, cfg.time_embed_hidden_dim as usize));
    linear1(&h0, w2, b2, cfg.time_embed_hidden_dim as usize, cfg.time_embed_dim as usize)
}

/// `time_proj(timestep) -> time_embedder` for every DISTINCT timestep -
/// `[num_timesteps, time_embed_dim]`, row-major. Host-side: this is a
/// `(num_timesteps,)`-sized computation shared by every block, orders of
/// magnitude smaller than the per-row device work the rest of the forward
/// does (`dit::timestep`'s own precedent for host-side scalar conditioning).
fn build_temb(cfg: &H3TransformerConfig, timestep: &[f32], w0: &[f32], b0: &[f32], w2: &[f32], b2: &[f32]) -> Vec<f32> {
    let mut temb = Vec::with_capacity(timestep.len() * cfg.time_embed_dim as usize);
    for &t in timestep {
        temb.extend(temb_row(cfg, t, w0, b0, w2, b2));
    }
    temb
}

/// What the embed stage reads besides its blocks: the three per-modality
/// input projections and the token refiner. The projections stay fp32 -
/// they are a sliver of the model and carry its inputs - while the refiner's
/// full-width blocks take the stack's tier.
#[derive(Clone)]
pub struct EmbedWeights {
    proj_in_w: Weight,
    proj_in_b: DeviceBuffer,
    audio_proj_in_w: Weight,
    audio_proj_in_b: DeviceBuffer,
    context_embedder_w: Weight,
    context_embedder_b: DeviceBuffer,
    refiner_blocks: Vec<RefinerBlockWeights>,
    refiner_final_norm: DeviceBuffer,
}

impl EmbedWeights {
    fn load(tensors: &dyn checkpoint::TensorSource, cx: &Ctx, cfg: &H3TransformerConfig, tier: Dtype) -> EmbedWeights {
        let hidden = cfg.hidden_size;
        let dev = |name: &str| block::load_dev(tensors, cx, name);
        let lin = |name: &str, k: u32| block::load_linear(tensors, cx, name, hidden, k, Dtype::F32);
        EmbedWeights {
            proj_in_w: lin("proj_in.weight", cfg.video_patch_dim()),
            proj_in_b: dev("proj_in.bias"),
            audio_proj_in_w: lin("audio_proj_in.weight", cfg.audio_in_channels),
            audio_proj_in_b: dev("audio_proj_in.bias"),
            context_embedder_w: lin("context_embedder.weight", cfg.text_dim),
            context_embedder_b: dev("context_embedder.bias"),
            refiner_blocks: (0..cfg.num_refiner_layers as usize).map(|i| block::load_refiner_block(tensors, cx, i, cfg, tier)).collect(),
            refiner_final_norm: dev("token_refiner.final_norm.weight"),
        }
    }
}

/// What the head stage reads after the stack: `norm_out` (its projection on
/// the host, split per forward into shift/scale) and the two output heads,
/// fp32 for the same reason as [`EmbedWeights`]' projections.
#[derive(Clone)]
pub struct HeadWeights {
    norm_out_norm: DeviceBuffer,
    /// `norm_out.linear.weight`, `[2*hidden, time_embed_dim]` row-major.
    norm_out_linear_w: Vec<f32>,
    norm_out_linear_b: Vec<f32>,
    proj_out_w: Weight,
    proj_out_b: DeviceBuffer,
    audio_proj_out_w: Weight,
    audio_proj_out_b: DeviceBuffer,
}

impl HeadWeights {
    fn load(tensors: &dyn checkpoint::TensorSource, cx: &Ctx, cfg: &H3TransformerConfig) -> HeadWeights {
        let hidden = cfg.hidden_size;
        let dev = |name: &str| block::load_dev(tensors, cx, name);
        HeadWeights {
            norm_out_norm: dev("norm_out.norm.weight"),
            norm_out_linear_w: block::load_host(tensors, "norm_out.linear.weight"),
            norm_out_linear_b: block::load_host(tensors, "norm_out.linear.bias"),
            proj_out_w: block::load_linear(tensors, cx, "proj_out.weight", cfg.video_patch_dim(), hidden, Dtype::F32),
            proj_out_b: dev("proj_out.bias"),
            audio_proj_out_w: block::load_linear(tensors, cx, "audio_proj_out.weight", cfg.audio_in_channels, hidden, Dtype::F32),
            audio_proj_out_b: dev("audio_proj_out.bias"),
        }
    }
}

/// `time_embedder`'s two linears, host-side (see [`build_temb`]).
#[derive(Clone)]
pub struct TimeWeights {
    w0: Vec<f32>,
    b0: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

impl TimeWeights {
    pub(crate) fn load(tensors: &dyn checkpoint::TensorSource) -> TimeWeights {
        let host = |name: &str| block::load_host(tensors, name);
        TimeWeights { w0: host("time_embedder.linear_1.weight"), b0: host("time_embedder.linear_1.bias"), w2: host("time_embedder.linear_2.weight"), b2: host("time_embedder.linear_2.bias") }
    }

    /// `time_embedder(time_proj(t))` for one timestep (pre-SiLU).
    pub(crate) fn temb(&self, cfg: &H3TransformerConfig, t: f32) -> Vec<f32> {
        temb_row(cfg, t, &self.w0, &self.b0, &self.w2, &self.b2)
    }
}

/// Where one forward finds its weights. The resident [`H3Transformer`] lends
/// what it holds; [`Streamed`] reads each piece as the forward reaches it, so
/// at most one block's weights are on a card at a time.
trait StageWeights {
    fn time(&self) -> Cow<'_, TimeWeights>;
    /// Every block's AdaLN modulation at each of `timesteps`
    /// ([`crate::adaln`]): the projection itself is never a block weight.
    fn adaln(&self, time: &TimeWeights, timesteps: &[f32]) -> Result<AdalnPlan, String>;
    fn embed(&self, cx: &Ctx) -> Cow<'_, EmbedWeights>;
    fn block(&self, i: usize, cx: &Ctx) -> Cow<'_, BlockWeights>;
    fn head(&self, cx: &Ctx) -> Cow<'_, HeadWeights>;
}

/// [`StageWeights`] read off the checkpoint at fp32 as the forward goes.
struct Streamed<'a> {
    tensors: &'a dyn checkpoint::TensorSource,
    cfg: &'a H3TransformerConfig,
    dit: &'a StreamingDit,
}

impl StageWeights for Streamed<'_> {
    fn time(&self) -> Cow<'_, TimeWeights> {
        Cow::Owned(TimeWeights::load(self.tensors))
    }
    /// A generation prepared its whole schedule up front
    /// ([`StreamingDit::prepare`]), so this is a cache hit that reads no
    /// tensor; a caller that did not still gets the right rows, at the cost
    /// of reading each block's `adaln_proj` here.
    fn adaln(&self, time: &TimeWeights, timesteps: &[f32]) -> Result<AdalnPlan, String> {
        self.dit.prepare_with(self.tensors, self.cfg, time, timesteps)
    }
    fn embed(&self, cx: &Ctx) -> Cow<'_, EmbedWeights> {
        Cow::Owned(EmbedWeights::load(self.tensors, cx, self.cfg, Dtype::F32))
    }
    fn block(&self, i: usize, cx: &Ctx) -> Cow<'_, BlockWeights> {
        Cow::Owned(block::load_block_streaming(self.tensors, cx, i, self.cfg))
    }
    fn head(&self, cx: &Ctx) -> Cow<'_, HeadWeights> {
        Cow::Owned(HeadWeights::load(self.tensors, cx, self.cfg))
    }
}

/// The DiT held resident for every forward it serves, its block stack placed
/// over as many cards as hold it - the shape a denoise loop wants, since every
/// step reads every weight. [`H3Transformer::forward_streaming`] is the
/// placement for a machine whose cards cannot hold it.
pub struct H3Transformer {
    cfg: H3TransformerConfig,
    plan: StreamPlan,
    stages: Vec<Ctx>,
    time: TimeWeights,
    /// Every block's `adaln_proj` output by timestep value - the projection
    /// itself is never held ([`crate::adaln`]). Filled by [`Self::prepare`].
    adaln: Mutex<AdalnCache>,
    embed: EmbedWeights,
    blocks: Vec<BlockWeights>,
    head: HeadWeights,
}

impl StageWeights for H3Transformer {
    fn time(&self) -> Cow<'_, TimeWeights> {
        Cow::Borrowed(&self.time)
    }
    fn adaln(&self, _time: &TimeWeights, timesteps: &[f32]) -> Result<AdalnPlan, String> {
        self.adaln.lock().unwrap_or_else(|p| p.into_inner()).cached(timesteps)
    }
    fn embed(&self, _cx: &Ctx) -> Cow<'_, EmbedWeights> {
        Cow::Borrowed(&self.embed)
    }
    fn block(&self, i: usize, _cx: &Ctx) -> Cow<'_, BlockWeights> {
        Cow::Borrowed(&self.blocks[i])
    }
    fn head(&self, _cx: &Ctx) -> Cow<'_, HeadWeights> {
        Cow::Borrowed(&self.head)
    }
}

/// Device bytes of one block's packed linears at `tier` plus its norms.
fn block_bytes(cfg: &H3TransformerConfig, tier: Dtype) -> u64 {
    let (hidden, inner, ffn) = (cfg.hidden_size as usize, cfg.inner_dim() as usize, cfg.ffn_dim as usize);
    let linears = 3 * Weight::bytes_for(tier, inner, hidden) + Weight::bytes_for(tier, hidden, inner) + 2 * Weight::bytes_for(tier, ffn, hidden) + Weight::bytes_for(tier, hidden, ffn);
    linears + (2 * hidden + 2 * cfg.attention_head_dim as usize) as u64 * 4
}

/// Peak transient device bytes one block's forward allocates over `rows`
/// packed rows: the residual and its normed/modulated copies and six gathered
/// AdaLN rows at `hidden`, q/k/v and the attention output at `inner`, and the
/// SwiGLU's value, gate and product at `ffn`.
pub fn block_workspace_bytes(cfg: &H3TransformerConfig, rows: u32) -> u64 {
    let (hidden, inner, ffn) = (cfg.hidden_size as u64, cfg.inner_dim() as u64, cfg.ffn_dim as u64);
    rows as u64 * (12 * hidden + 6 * inner + 3 * ffn) * 4
}

/// Peak transient device bytes one card's share of a forward allocates over
/// `rows` packed rows: [`block_workspace_bytes`] plus what the stage holds
/// beside its blocks - the packed sequence the embed stage builds and its
/// fill staging, and the residual a block reads while the next one's is
/// written.
pub fn forward_workspace_bytes(cfg: &H3TransformerConfig, rows: u32) -> u64 {
    block_workspace_bytes(cfg, rows) + rows as u64 * 3 * cfg.hidden_size as u64 * 4
}

/// The resident model's per-card cost at `tier`: the block stack, the embed
/// stage's projections and refiner, and the head stage's output projections.
pub fn resident_cost(cfg: &H3TransformerConfig, tier: Dtype) -> LayerBytes {
    let hidden = cfg.hidden_size as usize;
    let fp32 = |n: usize, k: usize| Weight::bytes_for(Dtype::F32, n, k) + n as u64 * 4;
    let embed = fp32(hidden, cfg.video_patch_dim() as usize) + fp32(hidden, cfg.audio_in_channels as usize) + fp32(hidden, cfg.text_dim as usize) + cfg.num_refiner_layers as u64 * block_bytes(cfg, tier);
    let head = fp32(cfg.video_patch_dim() as usize, hidden) + fp32(cfg.audio_in_channels as usize, hidden) + hidden as u64 * 4;
    LayerBytes { per_layer: vec![block_bytes(cfg, tier); cfg.num_layers as usize], embed, head }
}

impl H3Transformer {
    /// Hold the DiT resident at linear tier `tier`, over the fewest cards
    /// `device` may schedule that hold it while keeping `rows` packed rows of
    /// forward workspace free on each. `Err` when no combination of them
    /// does, or when the cards cannot run `tier` - the caller streams instead
    /// ([`Self::forward_streaming`]).
    pub fn load(tensors: &dyn checkpoint::TensorSource, cfg: H3TransformerConfig, device: Option<&str>, tier: Dtype, rows: u32) -> Result<H3Transformer, String> {
        let cards = StreamPlan::cards(device, forward_workspace_bytes(&cfg, rows));
        let cost = resident_cost(&cfg, tier);
        let plan = StreamPlan::fit(&cost, &cards).ok_or_else(|| format!("minimaxh3: the DiT at {tier:?} is {} bytes, which no combination of the schedulable cards holds with {rows}-row workspace free (usable bytes {cards:?})", cost.total()))?;
        H3Transformer::load_on(tensors, cfg, plan, device, tier)
    }

    /// [`Self::load`] over the caller's own plan - what a test pinning the
    /// placement (one stage on the CPU, a forced split) uses.
    pub fn load_on(tensors: &dyn checkpoint::TensorSource, cfg: H3TransformerConfig, plan: StreamPlan, device: Option<&str>, tier: Dtype) -> Result<H3Transformer, String> {
        tracing::info!(plan = %plan.describe(), ?tier, "minimaxh3 resident DiT placement");
        let mut stages = Vec::with_capacity(plan.n_stages());
        let mut blocks = Vec::with_capacity(cfg.num_layers as usize);
        let mut embed = None;
        let mut head = None;
        for (s, shard) in plan.stages().iter().enumerate() {
            let cx = plan.open_stage(s, || Ctx::new(device))?;
            for i in shard.start..shard.end {
                let w = block::load_block(tensors, &cx, i, &cfg, tier);
                if w.wq.dtype() != tier {
                    return Err(format!("minimaxh3: {tier:?} weights landed as {:?} on stage {s} - this card cannot run the requested tier", w.wq.dtype()));
                }
                blocks.push(w);
            }
            if shard.embed {
                embed = Some(EmbedWeights::load(tensors, &cx, &cfg, tier));
            }
            if shard.head {
                head = Some(HeadWeights::load(tensors, &cx, &cfg));
            }
            stages.push(cx);
        }
        let embed = embed.ok_or("minimaxh3: the plan has no embed stage")?;
        let head = head.ok_or("minimaxh3: the plan has no head stage")?;
        Ok(H3Transformer { time: TimeWeights::load(tensors), adaln: Mutex::new(AdalnCache::new(crate::adaln::DEFAULT_CAPACITY)), cfg, plan, stages, embed, blocks, head })
    }

    pub fn config(&self) -> &H3TransformerConfig {
        &self.cfg
    }

    /// Cache the AdaLN modulation for every timestep in `values` from
    /// `tensors`' `adaln_proj` - the checkpoint [`Self::load`] read, which
    /// never holds the projection. A generation prepares its whole schedule
    /// before the first step and keeps the returned plan alive for the
    /// request: the cache never evicts a row a live plan holds, so every
    /// forward of that request finds its rows.
    pub fn prepare(&self, tensors: &dyn checkpoint::TensorSource, values: &[f32]) -> Result<AdalnPlan, String> {
        self.adaln.lock().unwrap_or_else(|p| p.into_inner()).prepare(tensors, &self.cfg, &self.time, values)
    }

    /// `MiniMaxH3Transformer3DModel.forward`, at batch size 1 (see this
    /// module's own doc).
    pub fn forward(&self, inp: &PackedInputs) -> H3Output {
        forward_core(&self.cfg, &self.plan, &|s| &self.stages[s], self, inp, false).0
    }

    /// [`Self::forward`], plus the intermediate parity taps
    /// `tools/goldens/minimaxh3_dit_dump_reference.py` also dumps (post-token-refiner
    /// text stream, post-RoPE `cos`/`sin`, block 0's packed input, block 0's
    /// attention output, block 0's full output) - the numeric parity test's
    /// own entry point, so a mismatch localizes to the first tap it breaks
    /// rather than only showing up in the final output.
    pub fn forward_with_taps(&self, inp: &PackedInputs) -> (H3Output, H3Taps) {
        forward_core(&self.cfg, &self.plan, &|s| &self.stages[s], self, inp, true)
    }

    /// One forward WITHOUT a resident model: each stage of `dit`'s plan
    /// reads its own blocks off `tensors` one at a time, into device buffers
    /// reused across every block AND every denoise step
    /// ([`block::load_block_streaming`]), so a card never holds more than one
    /// block's fp32 weights (~2.6 GB at the real dimensions).
    ///
    /// Takes an ALREADY-OPEN [`StreamingDit`], never a device string - a
    /// caller driving a real multi-step denoise loop must open the devices
    /// ONCE outside the loop and reuse them every step. A real-weight 384x384/
    /// 16-step generation OOM'd a 24GB P40 when this instead opened a fresh
    /// device per step: wgpu gives no guarantee the previous one's resources
    /// are reclaimed before the next opens.
    pub fn forward_streaming(tensors: &dyn checkpoint::TensorSource, cfg: &H3TransformerConfig, dit: &StreamingDit, inp: &PackedInputs) -> H3Output {
        forward_core(cfg, dit.plan(), &|s| dit.stage(s), &Streamed { tensors, cfg, dit }, inp, false).0
    }

    /// [`Self::forward_streaming`] with [`Self::forward_with_taps`]'s taps.
    pub fn forward_streaming_with_taps(tensors: &dyn checkpoint::TensorSource, cfg: &H3TransformerConfig, dit: &StreamingDit, inp: &PackedInputs) -> (H3Output, H3Taps) {
        forward_core(cfg, dit.plan(), &|s| dit.stage(s), &Streamed { tensors, cfg, dit }, inp, true)
    }
}

/// The scratch arenas a forward's two repeating sequences replay: the block
/// stack's and the token refiner's (a different row count, so its own slots).
const BLOCK_ARENA: &str = "minimaxh3 block";
const REFINER_ARENA: &str = "minimaxh3 refiner";

/// The one forward body: the block stack runs across `plan`'s stages, one
/// contiguous block range per card. Only the residual stream crosses a stage
/// boundary, host-staged (`gpu.read` on one stage, `Ctx::upload` on the
/// next): an f32 round trip through host memory is lossless, so a split plan
/// computes bit-identical numbers to the single-stage one and changes only
/// where the work happens.
///
/// `taps` reads the parity taps back ([`H3Taps`]); a production forward
/// passes `false` and gets them empty, since each is a full-sequence
/// readback that holds its block's buffers past the block.
fn forward_core<'c>(cfg: &H3TransformerConfig, plan: &StreamPlan, stage: &dyn Fn(usize) -> &'c Ctx, weights: &dyn StageWeights, inp: &PackedInputs, taps: bool) -> (H3Output, H3Taps) {
    let hidden = cfg.hidden_size;
    let seq_len = (inp.position_ids.len() / 3) as u32;
    assert_eq!(inp.token_tags.len(), seq_len as usize, "forward: token_tags must be [seq_len]");
    assert_eq!(inp.timestep_indices.len(), seq_len as usize, "forward: timestep_indices must be [seq_len]");
    let num_video = inp.video_indices.len() as u32;
    let num_audio = inp.audio_indices.len() as u32;
    let num_text = inp.text_indices.len() as u32;
    let num_timesteps = inp.timestep.len();

    // Everything computed on the HOST is stage-independent and computed once:
    // the RoPE tables and the timestep embedding are functions of the layout
    // and the step's scalars, not of any card's state. Each stage uploads its
    // own copy of the device-side tables it needs.
    let rope_tables = rope::build_tables(cfg, inp.position_ids);
    let time = weights.time();
    let temb = build_temb(cfg, inp.timestep, &time.w0, &time.b0, &time.w2, &time.b2);
    let temb_silu = silu_slice(&temb);
    let modulation = weights.adaln(&time, inp.timestep).unwrap_or_else(|e| panic!("minimaxh3 forward: {e}"));
    // Row -> this port's own AdaLN table address.
    let adaln_idx = adaln_indices(inp.token_tags, inp.timestep_indices, num_timesteps);

    let n_layers = cfg.num_layers as usize;
    let mid_block_index = n_layers / 2;
    let last_block_index = n_layers - 1;
    let mut tap_refiner_out: Vec<f32> = Vec::new();
    let mut tap_block0_input: Vec<f32> = Vec::new();
    let mut tap_block0_attn_out: Vec<f32> = Vec::new();
    let mut tap_block0_out: Vec<f32> = Vec::new();
    let mut tap_mid_block_out: Vec<f32> = Vec::new();
    let mut tap_last_block_out: Vec<f32> = Vec::new();
    // The residual stream between two stages, host-staged - the ONLY value
    // that crosses a cut.
    let mut carry: Option<Vec<f32>> = None;
    let mut output: Option<H3Output> = None;

    for (s, shard) in plan.stages().iter().enumerate() {
        let cx = stage(s);
        // A stage with no blocks of its own (a plan may put the endpoint
        // weights alone on a card) never uploads the block tables.
        let block_tables = (shard.end > shard.start).then(|| (cx.upload(&rope_tables.cos), cx.upload(&rope_tables.sin), cx.upload_u32(&adaln_idx)));
        // The modality row indices are read by the input scatter (embed
        // stage) and the output gather (head stage); a middle stage never
        // touches them.
        let modality_idx = (shard.embed || shard.head).then(|| (cx.upload_u32(inp.video_indices), cx.upload_u32(inp.audio_indices)));

        let mut h = if shard.embed {
            // 1. Per-modality input projections (video/audio directly; text
            // through the token refiner).
            let e = weights.embed(cx);
            let video_in = cx.upload(inp.hidden_states);
            let video_embeds = block::linear(cx, &video_in, &e.proj_in_w, Some(&e.proj_in_b), num_video);
            let audio_in = cx.upload(inp.audio_hidden_states);
            let audio_embeds = block::linear(cx, &audio_in, &e.audio_proj_in_w, Some(&e.audio_proj_in_b), num_audio);
            let text_in = cx.upload(inp.encoder_hidden_states);
            let mut text_embeds = block::linear(cx, &text_in, &e.context_embedder_w, Some(&e.context_embedder_b), num_text);
            for w in &e.refiner_blocks {
                text_embeds = {
                    let _arena = cx.gpu.scratch_scope_in(REFINER_ARENA);
                    block::refiner_block_forward(cx, w, &text_embeds, cfg, num_text)
                };
                cx.gpu.poll_wait();
            }
            let text_embeds = block::rmsnorm(cx, &text_embeds, &e.refiner_final_norm, num_text, hidden, cfg.final_norm_eps);
            if taps {
                tap_refiner_out = cx.gpu.read(&text_embeds, (num_text * hidden) as usize);
            }

            // 2. Scatter every modality's projected rows into the packed
            // sequence buffer (`index_copy` in the reference).
            let hidden_states = cx.gpu.storage((seq_len * hidden) as u64);
            cx.gpu.write_f32(&hidden_states, &vec![0f32; (seq_len * hidden) as usize]);
            let text_idx = cx.upload_u32(inp.text_indices);
            let (video_idx, audio_idx) = modality_idx.as_ref().expect("the embed stage uploads the modality indices");
            block::row_scatter(cx, &text_idx, &text_embeds, &hidden_states, num_text, hidden, seq_len);
            block::row_scatter(cx, video_idx, &video_embeds, &hidden_states, num_video, hidden, seq_len);
            block::row_scatter(cx, audio_idx, &audio_embeds, &hidden_states, num_audio, hidden, seq_len);
            hidden_states
        } else {
            cx.upload(&carry.take().expect("forward: a non-embed stage needs the previous stage's residual"))
        };

        // 3. This stage's own slice of the block stack.
        if taps && shard.owns(0) {
            tap_block0_input = cx.gpu.read(&h, (seq_len * hidden) as usize);
        }
        for i in shard.start..shard.end {
            let (cos, sin, adaln_idx_dev) = block_tables.as_ref().expect("a stage with blocks uploads its own RoPE/AdaLN tables");
            let w = weights.block(i, cx);
            // Every block allocates the same activations in the same order;
            // the arena hands each block the previous block's buffers instead
            // of a fresh multi-hundred-MB driver allocation per op. The
            // `poll_wait` below drains the block before the next scope opens.
            let out = {
                let _arena = cx.gpu.scratch_scope_in(BLOCK_ARENA);
                let rows = modulation.block_rows(i, inp.timestep).unwrap_or_else(|e| panic!("minimaxh3 forward: {e}"));
                let (out, attn_out) = block::block_forward(cx, &w, &h, adaln_idx_dev, cos, sin, &rows, cfg, seq_len);
                if taps && i == 0 {
                    tap_block0_attn_out = cx.gpu.read(&attn_out, (seq_len * hidden) as usize);
                    tap_block0_out = cx.gpu.read(&out, (seq_len * hidden) as usize);
                }
                out
            };
            if taps && i == mid_block_index {
                tap_mid_block_out = cx.gpu.read(&out, (seq_len * hidden) as usize);
            }
            if taps && i == last_block_index {
                tap_last_block_out = cx.gpu.read(&out, (seq_len * hidden) as usize);
            }
            h = out;
            // Every block's outputs and the scratch `block_forward`
            // allocates are fresh buffers; proving them finished here lets
            // wgpu recycle them before the next block allocates its own. A
            // no-op on the CPU backend.
            cx.gpu.poll_wait();
        }

        if !shard.head {
            // Hand the residual to the next card through host memory -
            // lossless for f32, so the split changes placement, not math.
            carry = Some(cx.gpu.read(&h, (seq_len * hidden) as usize));
            cx.gpu.poll_wait();
            continue;
        }

        // 4. norm_out (per-TIMESTEP shift+scale only, no gate, no modality
        // axis) then the two output heads, run over every row, rows of each
        // modality selected after.
        let head = weights.head(cx);
        let shift_w = &head.norm_out_linear_w[..(hidden * cfg.time_embed_dim) as usize];
        let scale_w = &head.norm_out_linear_w[(hidden * cfg.time_embed_dim) as usize..];
        let shift_b = &head.norm_out_linear_b[..hidden as usize];
        let scale_b = &head.norm_out_linear_b[hidden as usize..];
        let shift_tbl = linear1_rows(&temb_silu, shift_w, shift_b, num_timesteps, cfg.time_embed_dim as usize, hidden as usize);
        let scale_tbl = linear1_rows(&temb_silu, scale_w, scale_b, num_timesteps, cfg.time_embed_dim as usize, hidden as usize);
        let shift_dev = cx.upload(&shift_tbl);
        let scale_dev = cx.upload(&scale_tbl);
        let ts_idx_dev = cx.upload_u32(inp.timestep_indices);
        let shift_g = block::gather_rows(cx, &ts_idx_dev, &shift_dev, seq_len, hidden);
        let scale_g = block::gather_rows(cx, &ts_idx_dev, &scale_dev, seq_len, hidden);
        let normed = block::rmsnorm(cx, &h, &head.norm_out_norm, seq_len, hidden, cfg.final_norm_eps);
        let modulated = block::modulate(cx, &normed, &scale_g, &shift_g, seq_len * hidden);

        let video_patch_dim = cfg.video_patch_dim();
        let video_full = block::linear(cx, &modulated, &head.proj_out_w, Some(&head.proj_out_b), seq_len);
        let audio_full = block::linear(cx, &modulated, &head.audio_proj_out_w, Some(&head.audio_proj_out_b), seq_len);
        let (video_idx, audio_idx) = modality_idx.as_ref().expect("the head stage uploads the modality indices");
        let video_out = block::gather_rows(cx, video_idx, &video_full, num_video, video_patch_dim);
        let audio_out = block::gather_rows(cx, audio_idx, &audio_full, num_audio, cfg.audio_in_channels);

        output = Some(H3Output { video: cx.gpu.read(&video_out, (num_video * video_patch_dim) as usize), audio: cx.gpu.read(&audio_out, (num_audio * cfg.audio_in_channels) as usize) });
        // The once-per-forward buffers (head weights when streamed, tables,
        // gathers) are reclaimed here, or they compound step over step.
        cx.gpu.poll_wait();
    }

    let output = output.expect("forward: the plan must have a head stage");
    let taps = H3Taps {
        refiner_out: tap_refiner_out,
        rope_cos: rope_tables.cos,
        rope_sin: rope_tables.sin,
        temb,
        block0_input: tap_block0_input,
        block0_attn_out: tap_block0_attn_out,
        block0_out: tap_block0_out,
        mid_block_out: tap_mid_block_out,
        mid_block_index,
        last_block_out: tap_last_block_out,
    };
    (output, taps)
}

/// Intermediate parity taps for [`H3Transformer::forward_with_taps`] - see
/// `tools/goldens/minimaxh3_dit_dump_reference.py`'s own matching set of taps.
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
    /// `diffusers` reference run - `tools/goldens/minimaxh3_dit_real_dump_reference.py`,
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
                "{} not found - run tools/goldens/minimaxh3_dit_real_dump_reference.py --checkpoint {} --out {}",
                fixture_file.display(),
                paths.dit,
                fixture_dir.display()
            ));
            return;
        }

        let cfg = H3TransformerConfig::real();

        // Golden/checkpoint pairing, proven rather than assumed - see
        // brain_testutil::golden's own module doc.
        let Some(src) = brain_testutil::golden::Source::open(&fixture_dir, "tools/goldens/minimaxh3_dit_real_dump_reference.py") else {
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
        // Whatever placement this machine supports - one stage on a one-card
        // or CPU run, one stage per card on a multi-card one. The reference
        // tolerances below are the SAME either way: the split carries only
        // the residual, host-staged, and an f32 round trip is lossless
        // (`model::shard::set_stream_shard(false)` forces the single-stage arm for a
        // side-by-side check).
        let dit = crate::dit_shard::StreamingDit::open(&cfg, Some(&device_str));
        eprintln!("  block placement: {}", dit.plan().describe());
        let (out, taps) = H3Transformer::forward_streaming_with_taps(&reader, &cfg, &dit, &inp);
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
        let model = H3Transformer::load_on(&tensors, cfg, StreamPlan::single(cfg.num_layers as usize), Some("cpu"), Dtype::F32).expect("a resident tiny model");
        let cfg = model.config();

        let num_video = 4u32;
        let num_audio = 2u32;
        let num_text = 3u32;
        let seq_len = num_text + num_audio + num_video;

        let mut rng = data::rng::Lcg::new(data::rng::random_seed());
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

        let _plan = model.prepare(&tensors, &timestep).expect("prepare");
        let out = model.forward(&inp);
        assert_eq!(out.video.len(), (num_video * cfg.video_patch_dim()) as usize);
        assert_eq!(out.audio.len(), (num_audio * cfg.audio_in_channels) as usize);
        assert!(out.video.iter().all(|v| v.is_finite()), "video output must be finite: {:?}", out.video);
        assert!(out.audio.iter().all(|v| v.is_finite()), "audio output must be finite: {:?}", out.audio);
        assert!(out.video.iter().any(|&v| v != 0.0), "video output must not be trivially all-zero");
        assert!(out.audio.iter().any(|&v| v != 0.0), "audio output must not be trivially all-zero");
    }

    /// One small packed request - three text rows, two audio, four video,
    /// two timesteps - with random inputs, owned so a test can borrow it as
    /// [`PackedInputs`] as often as it likes.
    struct Packed {
        hidden_states: Vec<f32>,
        audio_hidden_states: Vec<f32>,
        encoder_hidden_states: Vec<f32>,
        timestep: Vec<f32>,
        timestep_indices: Vec<u32>,
        token_tags: Vec<u32>,
        position_ids: Vec<f32>,
        video_indices: Vec<u32>,
        audio_indices: Vec<u32>,
        text_indices: Vec<u32>,
    }

    impl Packed {
        fn random(cfg: &H3TransformerConfig) -> Packed {
            let (num_video, num_audio, num_text) = (4u32, 2u32, 3u32);
            let seq_len = num_text + num_audio + num_video;
            let mut rng = data::rng::Lcg::new(data::rng::random_seed());
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
                timestep_indices[i as usize] = (n % 2) as u32;
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
            Packed {
                hidden_states: rng.vec_scaled((num_video * cfg.video_patch_dim()) as usize, 0.3),
                audio_hidden_states: rng.vec_scaled((num_audio * cfg.audio_in_channels) as usize, 0.3),
                encoder_hidden_states: rng.vec_scaled((num_text * cfg.text_dim) as usize, 0.3),
                timestep: vec![0.2, 0.8],
                timestep_indices,
                token_tags,
                position_ids,
                video_indices,
                audio_indices,
                text_indices,
            }
        }

        fn inputs(&self) -> PackedInputs<'_> {
            PackedInputs {
                hidden_states: &self.hidden_states,
                audio_hidden_states: &self.audio_hidden_states,
                encoder_hidden_states: &self.encoder_hidden_states,
                timestep: &self.timestep,
                timestep_indices: &self.timestep_indices,
                token_tags: &self.token_tags,
                position_ids: &self.position_ids,
                video_indices: &self.video_indices,
                audio_indices: &self.audio_indices,
                text_indices: &self.text_indices,
            }
        }
    }

    /// A model held resident runs the same forward as one streamed block by
    /// block - the same weights, the same arithmetic, the same bits.
    #[test]
    fn adaln_proj_is_read_once_per_schedule_and_never_held_by_a_forward() {
        use crate::adaln::Counting;
        use crate::dit_shard::StreamingDit;

        let cfg = H3TransformerConfig::tiny();
        let src = Counting::new(rand_tensors(&cfg, 23));
        let model = H3Transformer::load_on(&src, cfg, StreamPlan::single(cfg.num_layers as usize), Some("cpu"), Dtype::F32).expect("a resident tiny model");
        assert!(src.take_matching("adaln_proj").is_empty(), "loading the resident model must not read adaln_proj");
        let packed = Packed::random(&cfg);
        let inp = packed.inputs();
        let plan = model.prepare(&src, inp.timestep).expect("prepare");
        assert_eq!(src.take_matching("adaln_proj").len(), 2 * cfg.num_layers as usize, "one weight + one bias read per block for the whole schedule");

        let dit = StreamingDit::open_single(&cfg, Some("cpu"));
        let _streamed_plan = dit.prepare(&src, &cfg, inp.timestep).expect("prepare the streaming DiT");
        src.take_matching("");
        for _ in 0..2 {
            let resident = model.forward(&inp);
            let streamed = H3Transformer::forward_streaming(&src, &cfg, &dit, &inp);
            assert!(src.take_matching("adaln_proj").is_empty(), "a prepared step must not re-read adaln_proj");
            assert_eq!(resident.video, streamed.video, "resident and streaming forwards must agree");
            assert_eq!(resident.audio, streamed.audio);
        }
        drop(plan);
    }

    #[test]
    fn a_resident_model_computes_what_the_streamed_forward_computes() {
        use crate::dit_shard::StreamingDit;

        let cfg = H3TransformerConfig::tiny();
        let tensors = rand_tensors(&cfg, data::rng::random_seed());
        let packed = Packed::random(&cfg);
        let streamed = H3Transformer::forward_streaming(&tensors, &cfg, &StreamingDit::open_single(&cfg, Some("cpu")), &packed.inputs());
        let resident = H3Transformer::load_on(&tensors, cfg, StreamPlan::single(cfg.num_layers as usize), Some("cpu"), Dtype::F32).expect("a resident tiny model");
        let _plan = resident.prepare(&tensors, packed.inputs().timestep).expect("prepare");
        let out = resident.forward(&packed.inputs());
        assert_eq!(out.video, streamed.video);
        assert_eq!(out.audio, streamed.audio);
        assert!(out.video.iter().any(|&v| v != 0.0), "the comparison must not be trivially all-zero");
    }

    /// The int8 tier a generation holds its DiT at stays close to the fp32
    /// model it quantizes - on a device that runs the packed dot product.
    /// The widths are the tiny config's widened to whole quantization groups.
    #[test]
    fn an_int8_resident_model_tracks_the_fp32_one() {
        let cfg = H3TransformerConfig { hidden_size: 64, text_dim: 32, ..H3TransformerConfig::tiny() };
        if !Ctx::new(None).ops.caps().numeric.int8_dot {
            brain_testutil::skip("the ambient device has no packed int8 dot product");
            return;
        }
        let tensors = rand_tensors(&cfg, data::rng::random_seed());
        let packed = Packed::random(&cfg);
        let plan = || StreamPlan::single(cfg.num_layers as usize);
        let run = |tier: Dtype| {
            let m = H3Transformer::load_on(&tensors, cfg, plan(), None, tier).expect("a resident tiny model");
            let _plan = m.prepare(&tensors, packed.inputs().timestep).expect("prepare");
            m.forward(&packed.inputs())
        };
        let (fp32, int8) = (run(Dtype::F32), run(Dtype::I8));
        let cosine = |a: &[f32], b: &[f32]| {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let norm = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            dot / (norm(a) * norm(b))
        };
        assert!(cosine(&int8.video, &fp32.video) > 0.99, "video cosine {}", cosine(&int8.video, &fp32.video));
        assert!(cosine(&int8.audio, &fp32.audio) > 0.99, "audio cosine {}", cosine(&int8.audio, &fp32.audio));
    }

    /// Splitting the block stack over two stages must change WHERE the work
    /// happens and nothing else. Exact equality, not a tolerance: the only
    /// value crossing a stage boundary is the residual, carried host-staged
    /// as f32, and every block still sees the same inputs and runs the same
    /// dispatches - so a single differing bit here would mean the split lost
    /// or reordered something real, which is worth failing on rather than
    /// widening a tolerance for.
    ///
    /// Both stages run on the same ambient device (`Shard::ANY_GPU`), so this
    /// gates the HANDOFF on any machine, GPU-less runners included; a real
    /// two-card run additionally exercises the same code path with the stages
    /// on different cards.
    #[test]
    fn a_two_stage_split_is_numerically_identical_to_the_single_stage_path() {
        use crate::dit_shard::{shard_cost, StreamingDit};
        use model::{Shard, StreamPlan};

        let cfg = H3TransformerConfig::tiny();
        assert!(cfg.num_layers >= 2, "a two-stage split needs at least two blocks");
        let tensors = rand_tensors(&cfg, 17);
        let packed = Packed::random(&cfg);
        let inp = packed.inputs();

        let one = StreamingDit::open_single(&cfg, Some("cpu"));
        assert_eq!(one.plan().n_stages(), 1);
        let single = H3Transformer::forward_streaming(&tensors, &cfg, &one, &inp);

        let split_plan = StreamPlan::balanced(&shard_cost(&cfg), &[Shard::ANY_GPU, Shard::ANY_GPU]);
        assert_eq!(split_plan.n_stages(), 2, "the tiny config's two blocks must give two stages");
        assert!(split_plan.stages()[0].embed && !split_plan.stages()[0].head, "stage 0 owns the input projections only");
        assert!(split_plan.stages()[1].head && !split_plan.stages()[1].embed, "the last stage owns the output heads only");
        let two = StreamingDit::open_with_plan(split_plan, Some("cpu"));
        let split = H3Transformer::forward_streaming(&tensors, &cfg, &two, &inp);

        assert_eq!(split.video, single.video, "a split plan must compute bit-identical video output");
        assert_eq!(split.audio, single.audio, "a split plan must compute bit-identical audio output");
        assert!(single.video.iter().any(|&v| v != 0.0), "the comparison must not be trivially all-zero");
    }

    /// Real NUMERIC parity against `tools/goldens/minimaxh3_dit_dump_reference.py`'s
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
            brain_testutil::skip(&format!("{} not found - run tools/goldens/minimaxh3_dit_dump_reference.py --out {}", fixture_file.display(), fixture_dir.display()));
            return;
        }

        let cfg = H3TransformerConfig::tiny();

        let Some(src) = brain_testutil::golden::Source::open(&fixture_dir, "tools/goldens/minimaxh3_dit_dump_reference.py") else {
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

        let model = H3Transformer::load_on(&tensors, cfg, StreamPlan::single(cfg.num_layers as usize), Some("cpu"), Dtype::F32).expect("a resident tiny model");

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

        let _plan = model.prepare(&tensors, inp.timestep).expect("prepare");
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
