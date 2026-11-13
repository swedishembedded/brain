// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for cutting autoregressive
// generation cost for its clients. If your team needs expertise in
// transformer inference caching then you can procure our services by sending
// an email to info@swedishembedded.com.

//! Incremental (cached) ST-transformer forward.
//!
//! Every stage of an STBlock is causal in time: spatial attention and the
//! feed-forwards act within one frame, the PEG convolutions pad `(2, 0)` in
//! time, and the temporal attention is masked `j > i` with an ALiBi bias that
//! depends only on `i - j`. So the activations of frame `f` at every layer are
//! a function of frames `0..=f` alone, and a pass over frames `0..t` followed by
//! a pass over frame `t` equals one pass over `0..=t`. [`StCache`] keeps what a
//! later frame reads from earlier ones:
//!
//! * per layer, the QK-normalized keys and the values of the temporal attention
//!   for every processed frame and location (`frames * heads * head_dim` floats
//!   each, per location);
//! * per layer, the inputs of the last two frames at each of the two PEG sites
//!   (the 3x3x3 causal convolution reaches two frames back).
//!
//! Extending by `t_new` frames then costs `t_new` frames of projections, PEG,
//! spatial attention and feed-forward, plus temporal attention of those frames
//! against the cached keys; see [`cached_forward_flops`] for the count.
//!
//! Everything, the temporal attention included, runs on the same kernels as
//! the full forward (`attn_scores_causal_bias`, `attn_softmax_bidir`,
//! `attn_apply_bidir`), so the result is bit-identical to it. The existing
//! causal-bias kernels are square (queries and keys share one length), so the
//! temporal attention is dispatched over all `t` frames with zero query rows for
//! the cached ones: `O(t^2)` score entries per location and head instead of
//! `O(t * t_new)`, a cost that is small next to the projections (the useful
//! work [`cached_forward_flops`] counts) and is the price of not adding a
//! rectangular kernel.
//!
//! The cache lives on the device. Per layer, one buffer holds the keys and
//! values of every processed frame (`[frame, location, 2 * inner]`) and one
//! per PEG site holds the site's input of every processed frame, channels
//! first (`[frame, dim, location]`). A call appends the new frames in place
//! with device copies (`copy_cols`), grows a buffer by doubling when it is
//! full (a device copy again), and builds the packed query/key/value operand
//! of the attention from the cache with device copies too. The new keys and
//! values never visit the host, and neither do the cached ones: what a call
//! uploads is its own frames' activations, the layer weights and, only when
//! the frame count outgrows it, the ALiBi table, none of which depend on how
//! many frames are cached (`extending_the_*_cache_uploads_no_more_per_frame_as_it_grows`
//! holds this with [`Gpu::uploaded_bytes`]). A [`StCache`] belongs to the
//! device that built it.
//!
//! Lengths are the frame count alone: [`StCache::mark`] records it and
//! [`StCache::rollback`] restores it, because rows past the count are never
//! read and are overwritten by the next append.
//!
//! Memory: the cache grows by `layers * (2 * hw * inner + 2 * hw * dim)`
//! floats per frame (up to the doubling slack), so the caller bounds it by
//! bounding the frames it keeps (the world model's window). Absolute positions
//! enter the dynamics model through its positional embedding, so a cache is
//! valid only for a growing prefix: dropping the oldest frame requires
//! [`StCache::clear`].

use crate::import::{DynamicsConfig, TokenizerConfig};
use crate::{
    add, attn_forward, attn_project, bias, biased_attn, dynamics_embed, dynamics_logits, ff_inner, final_norm,
    geglu_forward, indices_to_latents, k, matmul, to_pixels, DynamicsWeights, PegWeights, StBlockWeights,
    StTransformerWeights, TokenizerWeights, ATTN_SCORE_SCALE,
};
use gpu_core::{DeviceBuffer, Gpu, Step};

/// A device word index or extent: the kernels address with 32 bits.
fn w32(n: usize) -> u32 {
    u32::try_from(n).expect("the cache outgrew 32-bit word addressing")
}

/// `copy_cols`: `rows` rows of `cnt` words from `src` (row stride `src_stride`,
/// first word `src_off`) to `dst` (`dst_stride`, `dst_off`), on the device.
#[allow(clippy::too_many_arguments)]
fn copy_cols(
    gpu: &Gpu, src: &DeviceBuffer, dst: &DeviceBuffer, rows: usize, cnt: usize, src_stride: usize, src_off: usize,
    dst_stride: usize, dst_off: usize,
) -> Step {
    let params = [w32(rows), w32(cnt), w32(src_stride), w32(src_off), w32(dst_stride), w32(dst_off)];
    gpu.step(k::COPY_COLS, &[src, dst], &params, w32(rows * cnt))
}

/// A device matrix of `width`-word rows that grows by doubling; rows are
/// written in place by the caller's copies and only the first `used` of them,
/// which the caller tracks, mean anything.
#[derive(Default)]
struct DeviceRows {
    buf: Option<DeviceBuffer>,
    cap_rows: usize,
    width: usize,
}

impl DeviceRows {
    fn new(width: usize) -> DeviceRows {
        DeviceRows { buf: None, cap_rows: 0, width }
    }

    /// Room for `rows` rows, keeping the existing contents (device copy).
    fn reserve(&mut self, gpu: &Gpu, rows: usize) {
        if rows <= self.cap_rows {
            return;
        }
        let cap_rows = rows.max(2 * self.cap_rows);
        let fresh = gpu.storage((cap_rows * self.width) as u64);
        if let Some(old) = &self.buf {
            let words = self.cap_rows * self.width;
            gpu.submit(&[], &[copy_cols(gpu, old, &fresh, 1, words, words, 0, words, 0)]);
        }
        self.buf = Some(fresh);
        self.cap_rows = cap_rows;
    }

    fn buf(&self) -> &DeviceBuffer {
        self.buf.as_ref().expect("reserve() before use")
    }
}

#[derive(Default)]
struct LayerCache {
    /// Inputs of every processed frame at the spatial / temporal PEG sites,
    /// one row per frame: `[dim, hw]` channels first. The convolution reads
    /// the last two.
    sp_hist: DeviceRows,
    tp_hist: DeviceRows,
    /// Temporal-attention keys (QK-normalized) and values, one row per frame
    /// and location: `[k (inner) | v (inner)]`, location-minor within a frame.
    kv: DeviceRows,
}

/// Per-layer temporal key/value and PEG-history cache of one ST transformer,
/// resident on the device that built it.
#[derive(Default)]
pub struct StCache {
    layers: Vec<LayerCache>,
    frames: usize,
    heads: usize,
    inner: usize,
    /// `alibi_bias(heads, alibi_cap)`: the temporal bias of every frame count
    /// up to `alibi_cap` is its top-left corner.
    alibi: Option<DeviceBuffer>,
    alibi_cap: usize,
}

/// A point to return to with [`StCache::rollback`].
pub struct StCacheMark {
    frames: usize,
}

impl StCache {
    pub fn new() -> StCache {
        StCache::default()
    }

    /// Frames processed so far.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Forget everything (after the frame window moved), releasing the device
    /// buffers.
    pub fn clear(&mut self) {
        *self = StCache::default();
    }

    pub fn mark(&self) -> StCacheMark {
        StCacheMark { frames: self.frames }
    }

    /// Drop every frame processed since `mark`. The mark must come from this
    /// cache at a frame count not above the current one.
    pub fn rollback(&mut self, mark: &StCacheMark) {
        assert!(mark.frames <= self.frames, "mark is ahead of the cache");
        self.frames = mark.frames;
    }

    /// The `[heads, total, total]` temporal bias as its own device buffer,
    /// cut from the table. The table is rebuilt on the host (doubling its
    /// side) only when `total` exceeds it, so a growing cache pays that
    /// upload once per doubling, not once per frame.
    fn temporal_bias(&mut self, gpu: &Gpu, total: usize) -> DeviceBuffer {
        if total > self.alibi_cap {
            self.alibi_cap = total.max(2 * self.alibi_cap).max(MIN_ALIBI_FRAMES);
            self.alibi = Some(gpu.storage_init("alibi", &bias::alibi_bias(self.heads, self.alibi_cap)));
        }
        let (table, cap) = (self.alibi.as_ref().expect("table built above"), self.alibi_cap);
        let out = gpu.storage((self.heads * total * total) as u64);
        let cuts: Vec<Step> = (0..self.heads)
            .map(|h| copy_cols(gpu, table, &out, total, total, cap, h * cap * cap, total, h * total * total))
            .collect();
        gpu.submit(&[], &cuts);
        out
    }
}

/// Smallest side of the ALiBi table, so the first frames of a session do not
/// each rebuild it.
const MIN_ALIBI_FRAMES: usize = 16;

/// Geometry shared by one extend call.
struct Geo {
    h: u32,
    w: u32,
    dim: u32,
    heads: u32,
    head_dim: u32,
    t_new: u32,
    /// Frames already cached (absolute index of the first new frame).
    t0: usize,
}

impl Geo {
    fn hw(&self) -> usize {
        (self.h * self.w) as usize
    }
    fn inner(&self) -> usize {
        (self.heads * self.head_dim) as usize
    }
}

/// PEG over `[history ++ new]` frames, returning the convolution of the new
/// frames only, and appending the new input frames to the device history. The
/// kernel computes each output from the same 27 taps whatever the video length,
/// so the result equals that of the full video. The history (the last two
/// frames) is read from, and the new frames written to, the device buffer; only
/// the new frames cross the host boundary, each way.
fn peg_with_history(gpu: &Gpu, hist: &mut DeviceRows, x: &[f32], w: &PegWeights, g: &Geo) -> Vec<f32> {
    let (hw, dim, t_new) = (g.hw(), g.dim as usize, g.t_new as usize);
    let hist_frames = g.t0.min(2);
    let te = hist_frames + t_new;
    hist.reserve(gpu, g.t0 + t_new);

    // New frames channels first: [dim, t_new, hw].
    let mut cf = vec![0.0f32; x.len()];
    for i in 0..t_new {
        for l in 0..hw {
            for d in 0..dim {
                cf[(d * t_new + i) * hw + l] = x[(i * hw + l) * dim + d];
            }
        }
    }
    let xn = gpu.storage_init("x", &cf);
    let ext = gpu.storage((dim * te * hw) as u64);
    let mut steps = Vec::new();
    for tt in 0..hist_frames {
        let from = (g.t0 - hist_frames + tt) * dim * hw;
        steps.push(copy_cols(gpu, hist.buf(), &ext, dim, hw, hw, from, te * hw, tt * hw));
    }
    steps.push(copy_cols(gpu, &xn, &ext, dim, t_new * hw, t_new * hw, 0, te * hw, hist_frames * hw));
    for i in 0..t_new {
        steps.push(copy_cols(gpu, &xn, hist.buf(), dim, hw, t_new * hw, i * hw, hw, (g.t0 + i) * dim * hw));
    }

    let wtb = gpu.storage_init("wt", &w.dsconv);
    let bb = gpu.storage_init("b", &w.bias);
    let yb = gpu.storage((dim * te * hw) as u64);
    let params = [1, g.dim, w32(te), g.h, g.w, 3, 1, 2];
    steps.push(gpu.step(k::DWCONV3D, &[&ext, &wtb, &bb, &yb], &params, w32(dim * te * hw)));
    let new_out = gpu.storage((dim * t_new * hw) as u64);
    steps.push(copy_cols(gpu, &yb, &new_out, dim, t_new * hw, te * hw, hist_frames * hw, t_new * hw, 0));
    gpu.submit(&[], &steps);

    let cf_out = gpu.read(&new_out, dim * t_new * hw);
    let mut out = vec![0.0f32; x.len()];
    for d in 0..dim {
        for i in 0..t_new {
            for l in 0..hw {
                out[(i * hw + l) * dim + d] = cf_out[(d * t_new + i) * hw + l];
            }
        }
    }
    out
}

fn spatial_half(gpu: &Gpu, lc: &mut LayerCache, blk: &StBlockWeights, x: &[f32], g: &Geo, sbias: &[f32]) -> Vec<f32> {
    let hw = g.hw() as u32;
    let p = peg_with_history(gpu, &mut lc.sp_hist, x, &blk.spatial_peg, g);
    let xs = add(x, &p);
    let a = attn_forward(gpu, &xs, g.t_new, hw, g.dim, g.heads, g.head_dim, &blk.spatial_attn, sbias, false);
    let xs = add(&xs, &a);
    let f = geglu_forward(gpu, &xs, g.t_new * hw, g.dim, ff_inner(g.dim), &blk.spatial_ff);
    add(&xs, &f)
}

/// Temporal attention of the new frames against the cached keys and values,
/// in the accumulation order of the causal-bias kernels. `tbias` is the
/// `[heads, total, total]` ALiBi bias.
fn temporal_half(gpu: &Gpu, lc: &mut LayerCache, blk: &StBlockWeights, x: &[f32], g: &Geo, tbias: &DeviceBuffer, total: usize) -> Vec<f32> {
    let (hw, inner) = (g.hw(), g.inner());
    let (t_new, rows) = (g.t_new as usize, g.t_new as usize * g.hw());
    let p = peg_with_history(gpu, &mut lc.tp_hist, x, &blk.temporal_peg, g);
    let xs = add(x, &p);

    let xb = gpu.storage_init("x", &xs);
    let (qn, kn, v) = attn_project(gpu, &xb, rows as u32, g.dim, g.heads, g.head_dim, &blk.temporal_attn);
    lc.kv.reserve(gpu, total * hw);
    let kv = lc.kv.buf();

    // Attention on the device through the very kernels the full forward uses
    // (`attn_scores_causal_bias`, `attn_softmax_bidir`, `attn_apply_bidir`):
    // one batch entry per location, the cached keys/values of all `total`
    // frames plus the new queries. Row i of a causal attention reads only rows
    // 0..=i, so the query rows of the cached frames (zero here) cannot change
    // the new frames' rows and the result equals the full pass bit for bit.
    // The packed operand is `[location, frame, q | k | v]`; its keys/values come
    // from the cache after the new frames were appended to it.
    let (att, row3, kv2) = (biased_attn(), 3 * inner, 2 * inner);
    let packed = gpu.storage((hw * total * row3) as u64);
    let (b, t, heads) = (hw as u32, total as u32, g.heads);
    let scores = gpu.storage((b * heads * t * t) as u64);
    let probs = gpu.storage((b * heads * t * t) as u64);
    let ao_all = gpu.storage((hw * total * inner) as u64);
    let ao = gpu.storage((rows * inner) as u64);
    let mut steps = vec![
        copy_cols(gpu, &kn, kv, rows, inner, inner, 0, kv2, g.t0 * hw * kv2),
        copy_cols(gpu, &v, kv, rows, inner, inner, 0, kv2, g.t0 * hw * kv2 + inner),
    ];
    for i in 0..t_new {
        steps.push(copy_cols(gpu, &qn, &packed, hw, inner, inner, i * hw * inner, total * row3, (g.t0 + i) * row3));
    }
    for j in 0..total {
        steps.push(copy_cols(gpu, kv, &packed, hw, kv2, kv2, j * hw * kv2, total * row3, j * row3 + inner));
    }
    steps.push(att.step_scores(gpu, b, heads, t, g.head_dim, ATTN_SCORE_SCALE, true, &packed, tbias, &scores));
    steps.push(att.step_softmax(gpu, b, heads, t, &scores, &probs));
    steps.push(att.step_apply(gpu, b, heads, t, g.head_dim, &probs, &packed, &ao_all));
    for i in 0..t_new {
        steps.push(copy_cols(gpu, &ao_all, &ao, hw, inner, total * inner, (g.t0 + i) * inner, inner, i * hw * inner));
    }
    gpu.submit(&[&packed], &steps);

    let wo = gpu.storage_init("wo", &blk.temporal_attn.to_out);
    let y = gpu.read(&matmul(gpu, &ao, &wo, rows as u32, inner as u32, g.dim), rows * g.dim as usize);
    let xt = add(&xs, &y);
    let f = geglu_forward(gpu, &xt, rows as u32, g.dim, ff_inner(g.dim), &blk.temporal_ff);
    add(&xt, &f)
}

/// Run `t_new` further frames `x_new` (`[t_new, h, w, dim]`, batch 1) through
/// the block stack and the closing norm, given the frames already in `cache`;
/// returns the output of the new frames and appends them to the cache. Equal to
/// the last `t_new` frames of [`crate::sttransformer_forward`] over the whole
/// video with causal PEGs. `temporal_first` selects the block order as there.
#[allow(clippy::too_many_arguments)]
pub fn sttransformer_extend(
    gpu: &Gpu, cache: &mut StCache, x_new: &[f32], t_new: u32, h: u32, w: u32, dim: u32,
    heads: u32, head_dim: u32, wts: &StTransformerWeights, spatial_bias: &[f32], temporal_first: bool,
) -> Vec<f32> {
    let g = Geo { h, w, dim, heads, head_dim, t_new, t0: cache.frames };
    assert_eq!(x_new.len(), t_new as usize * g.hw() * dim as usize, "x_new must be [t_new, h, w, dim]");
    if cache.layers.is_empty() {
        let (peg_row, kv_row) = (dim as usize * g.hw(), 2 * g.inner());
        cache.layers = wts
            .layers
            .iter()
            .map(|_| LayerCache { sp_hist: DeviceRows::new(peg_row), tp_hist: DeviceRows::new(peg_row), kv: DeviceRows::new(kv_row) })
            .collect();
        cache.heads = heads as usize;
        cache.inner = g.inner();
    }
    assert_eq!(cache.layers.len(), wts.layers.len(), "cache was built for a different layer count");
    assert_eq!(cache.heads, heads as usize, "cache was built for a different head count");
    assert_eq!(cache.inner, g.inner(), "cache was built for a different attention width");
    let total = g.t0 + t_new as usize;
    let tbias = cache.temporal_bias(gpu, total);

    let mut cur = x_new.to_vec();
    for (blk, lc) in wts.layers.iter().zip(cache.layers.iter_mut()) {
        cur = if temporal_first {
            let t = temporal_half(gpu, lc, blk, &cur, &g, &tbias, total);
            spatial_half(gpu, lc, blk, &t, &g, spatial_bias)
        } else {
            let s = spatial_half(gpu, lc, blk, &cur, &g, spatial_bias);
            temporal_half(gpu, lc, blk, &s, &g, &tbias, total)
        };
    }
    cache.frames = total;
    final_norm(gpu, &cur, &wts.norm_out_gamma, t_new * h * w, dim)
}

/// ContinuousPositionBias of the dynamics transformer for an `h x w` grid; it
/// depends on the grid only, so a caller computes it once.
pub fn dynamics_spatial_bias(dw: &DynamicsWeights, dc: &DynamicsConfig, h: u32, w: u32) -> Vec<f32> {
    bias::cpb_bias(&dw.cpb_net, h as usize, w as usize, dc.heads as usize)
}

/// Dynamics logits `[t_new*h*w, n_codes]` for `t_new` further frames of token
/// ids (`ids_new`, `[t_new, h, w]`) with their one-hot `actions_new`
/// (`[t_new, num_actions]`), given the frames already in `cache`. Equal to the
/// last `t_new * h*w` rows of [`crate::dynamics_forward`] over the whole
/// sequence.
#[allow(clippy::too_many_arguments)]
pub fn dynamics_forward_extend(
    gpu: &Gpu, cache: &mut StCache, ids_new: &[u32], actions_new: &[f32], dw: &DynamicsWeights,
    dc: &DynamicsConfig, h: u32, w: u32, spatial_bias: &[f32],
) -> Vec<f32> {
    let hw = (h * w) as usize;
    assert!(ids_new.len() % hw == 0 && !ids_new.is_empty(), "ids_new must be whole frames");
    let t_new = (ids_new.len() / hw) as u32;
    let vta = dynamics_embed(
        gpu, ids_new, actions_new, dw, 1, t_new, h, w, dc.dim, dc.n_codes, dc.code_dim,
        dc.num_actions, cache.frames as u32,
    );
    let enc = sttransformer_extend(
        gpu, cache, &vta, t_new, h, w, dc.dim2, dc.heads, dc.head_dim, &dw.transformer, spatial_bias, false,
    );
    dynamics_logits(gpu, &enc, dw, t_new * h * w, dc.dim2, dc.n_codes)
}

/// Tokenizer decoder over further token frames, given the frames already in
/// `cache`: returns the new frames as pixels, frame-major `[t_new, c, H, W]`.
/// The first frame of a video ever decoded uses the first-frame `to_pixels`,
/// as in [`crate::decode_indices`].
#[allow(clippy::too_many_arguments)]
pub fn decode_extend(
    gpu: &Gpu, cache: &mut StCache, indices_new: &[u32], tw: &TokenizerWeights, tc: &TokenizerConfig,
    h: u32, w: u32, spatial_bias: &[f32],
) -> Vec<f32> {
    let hw = (h * w) as usize;
    assert!(indices_new.len() % hw == 0 && !indices_new.is_empty(), "indices_new must be whole frames");
    let t_new = indices_new.len() / hw;
    let t0 = cache.frames;
    let latents = indices_to_latents(gpu, indices_new, &tw.vq, hw as u32 * t_new as u32, tc.dim, tc.code_dim);
    let dec = sttransformer_extend(
        gpu, cache, &latents, t_new as u32, h, w, tc.dim, tc.heads, tc.head_dim, &tw.decoder, spatial_bias, true,
    );
    let frame_toklen = hw * tc.dim as usize;
    let mut pixels = Vec::new();
    for i in 0..t_new {
        let tp = if t0 + i == 0 { &tw.to_pixels_first } else { &tw.to_pixels_rest };
        pixels.extend(to_pixels(gpu, &dec[i * frame_toklen..(i + 1) * frame_toklen], tp, 1, 1, h, w, tc.dim, tc.channels, tc.patch));
    }
    pixels
}

/// FLOPs (a multiply-add counts 2) of running `new_frames` dynamics frames
/// through the transformer, token embedding and logits head when `cached_frames`
/// frames are already in the cache; `hw` tokens per frame. With
/// `cached_frames = 0` this is the full pass over `new_frames` frames, so the
/// work a cache saves per generated frame is
/// `cached_forward_flops(0, t + 1) - cached_forward_flops(t, 1)` for a context of `t` frames
/// (the full path also re-runs the `t` context frames).
pub fn cached_forward_flops(dc: &DynamicsConfig, hw: usize, cached_frames: usize, new_frames: usize) -> u64 {
    let (d2, inner, ffi) = (dc.dim2 as u64, (dc.heads * dc.head_dim) as u64, ff_inner(dc.dim2) as u64);
    let tokens = (hw * new_frames) as u64;
    let hw = hw as u64;
    // per token per STBlock half: q,k,v,out projections, GEGLU (x, gate, out), PEG taps
    let half = 2 * (4 * d2 * inner + 3 * d2 * ffi + 27 * d2);
    // spatial attention: scores + weighted sum over the frame's hw keys
    let spatial = 2 * 2 * hw * inner;
    // temporal attention of frame g reads g + 1 keys (causal): 2 * 2 * (g+1) * inner per token
    let temporal_keys: u64 = (cached_frames..cached_frames + new_frames).map(|g| g as u64 + 1).sum();
    let temporal = hw * 2 * 2 * inner * temporal_keys;
    let per_layer = tokens * (2 * half + spatial) + temporal;
    let embed = tokens * 2 * dc.code_dim as u64 * dc.dim as u64;
    let head = tokens * 2 * d2 * dc.n_codes as u64;
    dc.layers as u64 * per_layer + embed + head
}
