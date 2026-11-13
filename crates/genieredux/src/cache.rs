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
//! rectangular kernel. The keys and values live on the host between calls and
//! are uploaded with each temporal attention.
//!
//! Memory: the cache grows by `layers * 2 * hw * inner` floats per frame, so
//! the caller bounds it by bounding the frames it keeps (the world model's
//! window). Absolute positions enter the dynamics model through its positional
//! embedding, so a cache is valid only for a growing prefix: dropping the
//! oldest frame requires [`StCache::clear`].

use crate::import::{DynamicsConfig, TokenizerConfig};
use crate::{
    add, attn_forward, attn_project, bias, biased_attn, dynamics_embed, dynamics_logits, ff_inner, final_norm,
    geglu_forward, indices_to_latents, matmul, peg_forward_w, to_pixels, DynamicsWeights,
    PegWeights, StBlockWeights, StTransformerWeights, TokenizerWeights, ATTN_SCORE_SCALE,
};
use gpu_core::Gpu;

#[derive(Clone, Default)]
struct LayerCache {
    /// Inputs of the last <= 2 frames at the spatial / temporal PEG sites,
    /// `[frames, hw, dim]` each.
    sp_hist: Vec<f32>,
    tp_hist: Vec<f32>,
    /// Temporal-attention keys (QK-normalized) and values per location, each
    /// `frames * inner`.
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

/// Per-layer temporal key/value and PEG-history cache of one ST transformer.
#[derive(Clone, Default)]
pub struct StCache {
    layers: Vec<LayerCache>,
    frames: usize,
    inner: usize,
}

/// A point to return to with [`StCache::rollback`].
pub struct StCacheMark {
    frames: usize,
    hist: Vec<(Vec<f32>, Vec<f32>)>,
}

impl StCache {
    pub fn new() -> StCache {
        StCache::default()
    }

    /// Frames processed so far.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Forget everything (after the frame window moved).
    pub fn clear(&mut self) {
        *self = StCache::default();
    }

    pub fn mark(&self) -> StCacheMark {
        StCacheMark {
            frames: self.frames,
            hist: self.layers.iter().map(|l| (l.sp_hist.clone(), l.tp_hist.clone())).collect(),
        }
    }

    /// Drop every frame processed since `mark`. The mark must come from this
    /// cache at a frame count not above the current one.
    pub fn rollback(&mut self, mark: &StCacheMark) {
        assert!(mark.frames <= self.frames, "mark is ahead of the cache");
        if mark.frames == 0 {
            self.clear();
            return;
        }
        for (l, (sp, tp)) in self.layers.iter_mut().zip(&mark.hist) {
            l.sp_hist = sp.clone();
            l.tp_hist = tp.clone();
            for v in l.k.iter_mut().chain(l.v.iter_mut()) {
                v.truncate(mark.frames * self.inner);
            }
        }
        self.frames = mark.frames;
    }
}

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
/// frames only, and keeping the last two input frames as the new history. The
/// kernel computes each output from the same 27 taps whatever the video length,
/// so the result equals that of the full video.
fn peg_with_history(gpu: &Gpu, hist: &mut Vec<f32>, x: &[f32], w: &PegWeights, g: &Geo) -> Vec<f32> {
    let frame = g.hw() * g.dim as usize;
    let hist_frames = hist.len() / frame;
    let mut ext = std::mem::take(hist);
    ext.extend_from_slice(x);
    let te = hist_frames + g.t_new as usize;
    let conv = peg_forward_w(gpu, &ext, w, 1, te as u32, g.h, g.w, g.dim, true);
    let keep = te.min(2);
    *hist = ext[(te - keep) * frame..].to_vec();
    conv[hist_frames * frame..].to_vec()
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
/// in the accumulation order of the causal-bias kernels.
fn temporal_half(gpu: &Gpu, lc: &mut LayerCache, blk: &StBlockWeights, x: &[f32], g: &Geo, tbias: &[f32], total: usize) -> Vec<f32> {
    let (hw, inner) = (g.hw(), g.inner());
    let rows = g.t_new as usize * hw;
    let p = peg_with_history(gpu, &mut lc.tp_hist, x, &blk.temporal_peg, g);
    let xs = add(x, &p);

    let xb = gpu.storage_init("x", &xs);
    let (qn, kn, v) = attn_project(gpu, &xb, rows as u32, g.dim, g.heads, g.head_dim, &blk.temporal_attn);
    let (qh, kh, vh) = (gpu.read(&qn, rows * inner), gpu.read(&kn, rows * inner), gpu.read(&v, rows * inner));
    if lc.k.is_empty() {
        lc.k = vec![Vec::new(); hw];
        lc.v = vec![Vec::new(); hw];
    }
    for i in 0..g.t_new as usize {
        for l in 0..hw {
            let row = (i * hw + l) * inner;
            lc.k[l].extend_from_slice(&kh[row..row + inner]);
            lc.v[l].extend_from_slice(&vh[row..row + inner]);
        }
    }

    // Attention on the device through the very kernels the full forward uses
    // (`attn_scores_causal_bias`, `attn_softmax_bidir`, `attn_apply_bidir`):
    // one batch entry per location, the cached keys/values of all `total`
    // frames plus the new queries. Row i of a causal attention reads only rows
    // 0..=i, so the query rows of the cached frames (zero here) cannot change
    // the new frames' rows and the result equals the full pass bit for bit.
    let att = biased_attn();
    let mut packed = vec![0.0f32; hw * total * 3 * inner];
    for l in 0..hw {
        for j in 0..total {
            let dst = (l * total + j) * 3 * inner;
            if j >= g.t0 {
                let src = ((j - g.t0) * hw + l) * inner;
                packed[dst..dst + inner].copy_from_slice(&qh[src..src + inner]);
            }
            packed[dst + inner..dst + 2 * inner].copy_from_slice(&lc.k[l][j * inner..(j + 1) * inner]);
            packed[dst + 2 * inner..dst + 3 * inner].copy_from_slice(&lc.v[l][j * inner..(j + 1) * inner]);
        }
    }
    let (b, t, heads) = (hw as u32, total as u32, g.heads);
    let (packed_b, bias_b) = (gpu.storage_init("qkv", &packed), gpu.storage_init("bias", tbias));
    let scores = gpu.storage((b * heads * t * t) as u64);
    let probs = gpu.storage((b * heads * t * t) as u64);
    let ao_all = gpu.storage((hw * total * inner) as u64);
    gpu.submit(&[], &[
        att.step_scores(gpu, b, heads, t, g.head_dim, ATTN_SCORE_SCALE, true, &packed_b, &bias_b, &scores),
        att.step_softmax(gpu, b, heads, t, &scores, &probs),
        att.step_apply(gpu, b, heads, t, g.head_dim, &probs, &packed_b, &ao_all),
    ]);
    let ao_all = gpu.read(&ao_all, hw * total * inner);
    let mut ao = vec![0.0f32; rows * inner];
    for i in 0..g.t_new as usize {
        for l in 0..hw {
            let src = (l * total + g.t0 + i) * inner;
            ao[(i * hw + l) * inner..(i * hw + l + 1) * inner].copy_from_slice(&ao_all[src..src + inner]);
        }
    }

    let aob = gpu.storage_init("ao", &ao);
    let wo = gpu.storage_init("wo", &blk.temporal_attn.to_out);
    let y = gpu.read(&matmul(gpu, &aob, &wo, rows as u32, inner as u32, g.dim), rows * g.dim as usize);
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
        cache.layers = vec![LayerCache::default(); wts.layers.len()];
        cache.inner = g.inner();
    }
    assert_eq!(cache.layers.len(), wts.layers.len(), "cache was built for a different layer count");
    let total = g.t0 + t_new as usize;
    let tbias = bias::alibi_bias(heads as usize, total);

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
