// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Wan-VAE: a 3D **causal** autoencoder at a (4, 8, 8) stride.
//!
//! Encodes `1 + 4k` video frames into `1 + k` latent frames and back. The block
//! primitives live in [`vae::blocks3d`] (they are not Wan-specific - any 3D
//! causal VAE is built from the same conv / channel-L2-norm / SiLU / per-frame
//! attention / resample set); what lives here is the **schedule**, the tensor
//! names, and the chunked causal driver, exactly as `crates/vae`'s own
//! `decoder.rs` owns the diffusers `AutoencoderKL` schedule over the 2D blocks.
//!
//! # Everything runs in temporal chunks, and it has to
//!
//! Upstream never runs this model over a whole clip. `encode` feeds it 1 frame
//! and then 4 at a time; `decode` feeds it ONE latent frame at a time; and each
//! `CausalConv3d` receives the previous chunk's last two input frames as
//! `cache_x`, concatenated on time and subtracted from its low pad. That is
//! what makes the chunked result equal to a whole-clip causal forward, and it is
//! also what makes 81 frames at 480p fit at all - the level-0 activation for a
//! whole clip is over 12 GB, past any binding limit, while a 4-frame chunk is
//! under a gigabyte.
//!
//! The whole clip is nevertheless recorded as ONE graph: the cache is an SSA
//! buffer handed from one chunk's sub-graph to the next, so there is no device
//! state, no readback between chunks, and one submit for the clip.
//!
//! # The three states of `upsample3d`
//!
//! The decoder's temporal upsample is the one place the cache is more than "the
//! last two frames". Its slot holds one of three things and each selects
//! different arithmetic:
//!
//! | slot | what runs |
//! |---|---|
//! | empty (first chunk) | **no** temporal conv at all - the chunk passes through, and the slot is marked `Rep` |
//! | `Rep` | the temporal conv against an all-zero history: frame 0 is deliberately dropped from the temporal receptive field |
//! | frames | the temporal conv against the real cached frames |
//!
//! Only the first is an initialisation case. `Rep` is a *lasting* semantic
//! choice - it is why the whole-clip equivalent of this layer convolves
//! `x[:, :, 1:]` and not `x`.
//!
//! # Chunk-size invariance holds for encode, NOT for decode
//!
//! The encoder's `downsample3d` carries one frame of history and consumes
//! stride-2 windows at even positions, so splitting the clip as (1,4,4) or
//! (1,8) gives the same answer - `crates/wan/tests/vae_parity.rs` asserts that
//! BIT-EXACTLY, and it is the cheapest possible gate on the whole cache
//! mechanism. The decoder's `Rep` state breaks the same property (a 2-frame
//! first-conv chunk zero-fills two history slots where two 1-frame chunks
//! zero-fill one), so [`WanVaeDecoder`] hardcodes upstream's one-latent-frame
//! chunking rather than exposing a knob that would be silently wrong.

use gpu_core::{DeviceBuffer, Gpu, Step};
use vae::blocks::Tensors;
use vae::blocks3d::{Builder3d, CacheSlot, Conv3d, FeatCache, T3, KERNELS};

/// `BRAIN_WAN_VAE_TAPS=1` records every block output for parity debugging
/// (pins buffers, so it also disables activation reuse).
fn taps_enabled() -> bool {
    std::env::var("BRAIN_WAN_VAE_TAPS").is_ok()
}

/// Architecture + latent normalisation of the Wan-VAE.
///
/// `temperal_downsample` keeps upstream's misspelling on purpose: it is the
/// key name in the shipped `vae/config.json`, and a "corrected" spelling here
/// would not match the file it is read from.
#[derive(Clone, Debug, PartialEq)]
pub struct WanVaeConfig {
    pub base_dim: u32,
    pub z_dim: u32,
    pub dim_mult: Vec<u32>,
    pub num_res_blocks: u32,
    pub temperal_downsample: Vec<bool>,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

/// One entry of the encoder's / decoder's flat block list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Block {
    Res { cin: u32, cout: u32 },
    /// Spatial (and optionally temporal) downsample.
    Down { dim: u32, temporal: bool },
    /// Temporal (optional) then spatial upsample; output width is `dim / 2`.
    Up { dim: u32, temporal: bool },
}

impl Default for WanVaeConfig {
    fn default() -> Self {
        Self::wan21()
    }
}

impl WanVaeConfig {
    /// The Wan2.1 / Wan2.2 VAE, transcribed from the shipped
    /// `vae/config.json` (`AutoencoderKLWan`) and cross-checked against
    /// `_video_vae`'s hardcoded `cfg` in the reference.
    pub fn wan21() -> WanVaeConfig {
        WanVaeConfig {
            base_dim: 96,
            z_dim: 16,
            dim_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            temperal_downsample: vec![false, true, true],
            latents_mean: vec![
                -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134,
                -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
            ],
            latents_std: vec![
                2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526,
                2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
            ],
        }
    }

    /// Channel widths the encoder steps through: `dim * [1, *dim_mult]`.
    fn enc_dims(&self) -> Vec<u32> {
        std::iter::once(1u32)
            .chain(self.dim_mult.iter().copied())
            .map(|u| self.base_dim * u)
            .collect()
    }

    /// Channel widths the decoder steps through: `dim * [dim_mult[-1],
    /// *reversed(dim_mult)]`.
    fn dec_dims(&self) -> Vec<u32> {
        std::iter::once(*self.dim_mult.last().unwrap())
            .chain(self.dim_mult.iter().rev().copied())
            .map(|u| self.base_dim * u)
            .collect()
    }

    /// Latent frames for `frames` video frames - the `1 + 4k -> 1 + k` rule.
    /// `None` when `frames` is not `1 + 4k`, which the causal first-frame
    /// special case makes a hard requirement rather than a rounding.
    pub fn latent_frames(&self, frames: u32) -> Option<u32> {
        ((frames >= 1) && (frames - 1).is_multiple_of(4)).then_some(1 + (frames - 1) / 4)
    }

    /// Upstream's encode chunking: one frame, then four at a time.
    pub fn encode_chunks(&self, frames: u32) -> Vec<u32> {
        assert!(self.latent_frames(frames).is_some(), "{frames} frames is not 1+4k");
        std::iter::once(1).chain(std::iter::repeat_n(4, ((frames - 1) / 4) as usize)).collect()
    }

    fn enc_blocks(&self) -> Vec<Block> {
        let dims = self.enc_dims();
        let mut out = Vec::new();
        for i in 0..dims.len() - 1 {
            let (mut cin, cout) = (dims[i], dims[i + 1]);
            for _ in 0..self.num_res_blocks {
                out.push(Block::Res { cin, cout });
                cin = cout;
            }
            if i != self.dim_mult.len() - 1 {
                out.push(Block::Down { dim: cout, temporal: self.temperal_downsample[i] });
            }
        }
        out
    }

    fn dec_blocks(&self) -> Vec<Block> {
        let dims = self.dec_dims();
        // `temperal_upsample = temperal_downsample[::-1]`.
        let tup: Vec<bool> = self.temperal_downsample.iter().rev().copied().collect();
        let mut out = Vec::new();
        for i in 0..dims.len() - 1 {
            let (in_dim, cout) = (dims[i], dims[i + 1]);
            // Upstream halves the input width on every level but the first,
            // because the preceding `Resample` emitted `dim // 2` channels.
            let mut cin = if i == 0 { in_dim } else { in_dim / 2 };
            for _ in 0..self.num_res_blocks + 1 {
                out.push(Block::Res { cin, cout });
                cin = cout;
            }
            if i != self.dim_mult.len() - 1 {
                out.push(Block::Up { dim: cout, temporal: tup[i] });
            }
        }
        out
    }

    /// Every tensor this model reads, with its checkpoint shape, in the
    /// reference (`Wan2.1_VAE.pth`) name space. Derived from the config, so an
    /// importer can count it against a real checkpoint in both directions.
    pub fn tensor_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let mut m: Vec<(String, Vec<usize>)> = Vec::new();
        let z = self.z_dim as usize;
        let conv = |m: &mut Vec<(String, Vec<usize>)>, p: &str, shape: Vec<usize>| {
            let cout = shape[0];
            m.push((format!("{p}.weight"), shape));
            m.push((format!("{p}.bias"), vec![cout]));
        };
        let res = |m: &mut Vec<_>, p: &str, cin: u32, cout: u32| {
            let (ci, co) = (cin as usize, cout as usize);
            m.push((format!("{p}.residual.0.gamma"), vec![ci, 1, 1, 1]));
            conv(m, &format!("{p}.residual.2"), vec![co, ci, 3, 3, 3]);
            m.push((format!("{p}.residual.3.gamma"), vec![co, 1, 1, 1]));
            conv(m, &format!("{p}.residual.6"), vec![co, co, 3, 3, 3]);
            if cin != cout {
                conv(m, &format!("{p}.shortcut"), vec![co, ci, 1, 1, 1]);
            }
        };
        let attn = |m: &mut Vec<_>, p: &str, c: u32| {
            let c = c as usize;
            m.push((format!("{p}.norm.gamma"), vec![c, 1, 1]));
            conv(m, &format!("{p}.to_qkv"), vec![3 * c, c, 1, 1]);
            conv(m, &format!("{p}.proj"), vec![c, c, 1, 1]);
        };

        // ---- encoder ----
        let enc_last = *self.enc_dims().last().unwrap();
        conv(&mut m, "encoder.conv1", vec![self.enc_dims()[0] as usize, 3, 3, 3, 3]);
        for (i, b) in self.enc_blocks().iter().enumerate() {
            let p = format!("encoder.downsamples.{i}");
            match *b {
                Block::Res { cin, cout } => res(&mut m, &p, cin, cout),
                Block::Down { dim, temporal } => {
                    let d = dim as usize;
                    conv(&mut m, &format!("{p}.resample.1"), vec![d, d, 3, 3]);
                    if temporal {
                        conv(&mut m, &format!("{p}.time_conv"), vec![d, d, 3, 1, 1]);
                    }
                }
                Block::Up { .. } => unreachable!("encoder has no upsample"),
            }
        }
        res(&mut m, "encoder.middle.0", enc_last, enc_last);
        attn(&mut m, "encoder.middle.1", enc_last);
        res(&mut m, "encoder.middle.2", enc_last, enc_last);
        m.push(("encoder.head.0.gamma".into(), vec![enc_last as usize, 1, 1, 1]));
        conv(&mut m, "encoder.head.2", vec![2 * z, enc_last as usize, 3, 3, 3]);

        // ---- the two pointwise convs either side of the latent ----
        conv(&mut m, "conv1", vec![2 * z, 2 * z, 1, 1, 1]);
        conv(&mut m, "conv2", vec![z, z, 1, 1, 1]);

        // ---- decoder ----
        let dec_first = self.dec_dims()[0];
        let dec_last = *self.dec_dims().last().unwrap();
        conv(&mut m, "decoder.conv1", vec![dec_first as usize, z, 3, 3, 3]);
        res(&mut m, "decoder.middle.0", dec_first, dec_first);
        attn(&mut m, "decoder.middle.1", dec_first);
        res(&mut m, "decoder.middle.2", dec_first, dec_first);
        for (i, b) in self.dec_blocks().iter().enumerate() {
            let p = format!("decoder.upsamples.{i}");
            match *b {
                Block::Res { cin, cout } => res(&mut m, &p, cin, cout),
                Block::Up { dim, temporal } => {
                    let d = dim as usize;
                    conv(&mut m, &format!("{p}.resample.1"), vec![d / 2, d, 3, 3]);
                    if temporal {
                        conv(&mut m, &format!("{p}.time_conv"), vec![2 * d, d, 3, 1, 1]);
                    }
                }
                Block::Down { .. } => unreachable!("decoder has no downsample"),
            }
        }
        m.push(("decoder.head.0.gamma".into(), vec![dec_last as usize, 1, 1, 1]));
        conv(&mut m, "decoder.head.2", vec![3, dec_last as usize, 3, 3, 3]);
        m
    }
}

// ---------------------------------------------------------------- blocks

/// `ResidualBlock`: `x + conv6(silu(norm3(conv2(silu(norm0(x))))))`, over a
/// `shortcut` 1x1x1 conv when the width changes.
///
/// The `residual.N` leaf names are the reference `nn.Sequential`'s positional
/// indices (`0` norm, `2` conv, `3` norm, `6` conv - `1`/`4` are SiLU and `5`
/// is a Dropout), and both convs consume a `feat_cache` slot. The shortcut does
/// NOT: it is applied outside the loop upstream, and being pointwise it has no
/// temporal pad to subtract from anyway.
fn residual(
    b: &mut Builder3d,
    prefix: &str,
    cin: u32,
    cout: u32,
    x: &T3,
    cache: &mut FeatCache,
) -> T3 {
    let (skip, skip_owned) = if cin != cout {
        (b.conv(&format!("{prefix}.shortcut"), cout, Conv3d::point(), x), true)
    } else {
        (x.clone(), false)
    };
    let n0 = b.rms_norm(&format!("{prefix}.residual.0"), x);
    let s0 = b.silu(&n0);
    b.free(n0);
    let c0 = b.conv_cached(&format!("{prefix}.residual.2"), cout, Conv3d::causal3(), &s0, cache);
    b.free(s0);
    let n1 = b.rms_norm(&format!("{prefix}.residual.3"), &c0);
    b.free(c0);
    let s1 = b.silu(&n1);
    b.free(n1);
    let c1 = b.conv_cached(&format!("{prefix}.residual.6"), cout, Conv3d::causal3(), &s1, cache);
    b.free(s1);
    let out = b.add(&c1, &skip);
    b.free(c1);
    if skip_owned {
        b.free(skip);
    }
    b.tap(prefix, &out);
    out
}

/// `Resample(mode='downsample2d'|'downsample3d')`: the asymmetric-padded
/// stride-2 spatial conv, then (3D only) the stride-2 temporal conv.
///
/// The spatial half reproduces `nn.ZeroPad2d((0,1,0,1))` + `Conv2d(3, stride=2,
/// padding=0)` the same way `vae::blocks::conv_down` does: force the output to
/// `h/2 x w/2` with zero pad, and let the kernel's bounds checks supply the
/// missing right/bottom column as zeros.
fn downsample(
    b: &mut Builder3d,
    prefix: &str,
    dim: u32,
    temporal: bool,
    x: &T3,
    cache: &mut FeatCache,
) -> T3 {
    let spec = Conv3d { kt: 1, kh: 3, kw: 3, st: 1, sh: 2, sw: 2, pt: 0, ph: 0, pw: 0 };
    let y = b.conv_sized(&format!("{prefix}.resample.1"), dim, spec, (x.t, x.h / 2, x.w / 2), x);
    if !temporal {
        return y;
    }
    let (idx, slot) = cache.claim();
    // Only the previous chunk's LAST frame is ever read (upstream stores the
    // whole first chunk and then slices `[-1:]`), so that is what is stored.
    let last = b.time_slice(&y, y.t - 1, 1);
    match slot {
        CacheSlot::Empty => {
            // First chunk: no temporal conv at all, and the chunk is the seed
            // of the running sequence.
            cache.set(idx, CacheSlot::Frames(last));
            y
        }
        CacheSlot::Rep => unreachable!("downsample3d never stores the Rep sentinel"),
        CacheSlot::Frames(prev) => {
            let hist = b.time_slice(&prev, prev.t - 1, 1);
            let xin = b.time_cat(&hist, &y);
            b.free(hist);
            b.free(prev);
            b.free(y);
            let out = b.conv(&format!("{prefix}.time_conv"), dim, Conv3d::time_down(), &xin);
            b.free(xin);
            cache.set(idx, CacheSlot::Frames(last));
            out
        }
    }
}

/// `Resample(mode='upsample2d'|'upsample3d')`: (3D only) the causal temporal
/// conv doubling the frame count, then the per-frame nearest-2x + conv.
///
/// See this module's header for the three cache states; all three are here.
fn upsample(
    b: &mut Builder3d,
    prefix: &str,
    dim: u32,
    temporal: bool,
    x: &T3,
    cache: &mut FeatCache,
) -> T3 {
    let mut cur = x.clone();
    let mut cur_owned = false;
    if temporal {
        let (idx, slot) = cache.claim();
        match slot {
            CacheSlot::Empty => cache.set(idx, CacheSlot::Rep),
            slot => {
                let keep = vae::blocks3d::CACHE_T.min(x.t);
                let mut cache_x = b.time_slice(x, x.t - keep, keep);
                if cache_x.t < vae::blocks3d::CACHE_T {
                    // A one-frame chunk: pair it with the previous cache's last
                    // frame, or - in the `Rep` state - with an explicit zero
                    // frame, which is the history the NEXT chunk convolves
                    // against.
                    let head = match &slot {
                        CacheSlot::Frames(prev) => b.time_slice(prev, prev.t - 1, 1),
                        _ => b.zeros(x.c, 1, x.h, x.w),
                    };
                    let joined = b.time_cat(&head, &cache_x);
                    if matches!(slot, CacheSlot::Frames(_)) {
                        b.free(head);
                    }
                    b.free(cache_x);
                    cache_x = joined;
                }
                let mut spec = Conv3d::time_causal();
                let xin = match &slot {
                    CacheSlot::Frames(prev) => {
                        assert!(
                            prev.t <= spec.pt,
                            "upsample3d cache of {} frames exceeds pt={}",
                            prev.t,
                            spec.pt
                        );
                        spec.pt -= prev.t;
                        Some(b.time_cat(prev, x))
                    }
                    _ => None,
                };
                let z = b.conv(
                    &format!("{prefix}.time_conv"),
                    2 * dim,
                    spec,
                    xin.as_ref().unwrap_or(x),
                );
                if let Some(xin) = xin {
                    b.free(xin);
                }
                if let CacheSlot::Frames(prev) = slot {
                    b.free(prev);
                }
                cache.set(idx, CacheSlot::Frames(cache_x));
                cur = b.time_interleave(&z);
                cur_owned = true;
                b.free(z);
            }
        }
    }
    let up = b.upsample2(&cur);
    if cur_owned {
        b.free(cur);
    }
    let out = b.conv(&format!("{prefix}.resample.1"), dim / 2, Conv3d::spatial(3, 1, 1), &up);
    b.free(up);
    out
}

/// One encoder chunk: `conv1 -> downsamples -> middle -> head`.
fn encoder_chunk(b: &mut Builder3d, cfg: &WanVaeConfig, x: &T3, cache: &mut FeatCache) -> T3 {
    cache.rewind();
    let dims = cfg.enc_dims();
    let mut cur = b.conv_cached("encoder.conv1", dims[0], Conv3d::causal3(), x, cache);
    b.tap("encoder.conv1", &cur);
    for (i, blk) in cfg.enc_blocks().iter().enumerate() {
        let p = format!("encoder.downsamples.{i}");
        let next = match *blk {
            Block::Res { cin, cout } => residual(b, &p, cin, cout, &cur, cache),
            Block::Down { dim, temporal } => downsample(b, &p, dim, temporal, &cur, cache),
            Block::Up { .. } => unreachable!(),
        };
        b.free(cur);
        cur = next;
        b.tap(&p, &cur);
    }
    let last = *dims.last().unwrap();
    let next = residual(b, "encoder.middle.0", last, last, &cur, cache);
    b.free(cur);
    cur = next;
    let next = b.attn(
        "encoder.middle.1.norm",
        "encoder.middle.1.to_qkv",
        "encoder.middle.1.proj",
        &cur,
    );
    b.free(cur);
    cur = next;
    let next = residual(b, "encoder.middle.2", last, last, &cur, cache);
    b.free(cur);
    cur = next;
    b.tap("encoder.middle", &cur);

    let n = b.rms_norm("encoder.head.0", &cur);
    b.free(cur);
    let s = b.silu(&n);
    b.free(n);
    let out = b.conv_cached("encoder.head.2", 2 * cfg.z_dim, Conv3d::causal3(), &s, cache);
    b.free(s);
    out
}

/// One decoder chunk: `conv1 -> middle -> upsamples -> head`.
fn decoder_chunk(b: &mut Builder3d, cfg: &WanVaeConfig, x: &T3, cache: &mut FeatCache) -> T3 {
    cache.rewind();
    let dims = cfg.dec_dims();
    let first = dims[0];
    let mut cur = b.conv_cached("decoder.conv1", first, Conv3d::causal3(), x, cache);
    b.tap("decoder.conv1", &cur);
    let next = residual(b, "decoder.middle.0", first, first, &cur, cache);
    b.free(cur);
    cur = next;
    let next = b.attn(
        "decoder.middle.1.norm",
        "decoder.middle.1.to_qkv",
        "decoder.middle.1.proj",
        &cur,
    );
    b.free(cur);
    cur = next;
    let next = residual(b, "decoder.middle.2", first, first, &cur, cache);
    b.free(cur);
    cur = next;
    b.tap("decoder.middle", &cur);
    for (i, blk) in cfg.dec_blocks().iter().enumerate() {
        let p = format!("decoder.upsamples.{i}");
        let next = match *blk {
            Block::Res { cin, cout } => residual(b, &p, cin, cout, &cur, cache),
            Block::Up { dim, temporal } => upsample(b, &p, dim, temporal, &cur, cache),
            Block::Down { .. } => unreachable!(),
        };
        b.free(cur);
        cur = next;
        b.tap(&p, &cur);
    }
    let n = b.rms_norm("decoder.head.0", &cur);
    b.free(cur);
    let s = b.silu(&n);
    b.free(n);
    let out = b.conv_cached("decoder.head.2", 3, Conv3d::causal3(), &s, cache);
    b.free(s);
    out
}

/// Slots to allocate per chunk. Upstream sizes the map with `count_conv3d`,
/// which over-counts (it includes the shortcut convs, which never claim a
/// slot); a fixed generous bound is the same thing without walking the tree,
/// and [`FeatCache::claim`] asserts on overflow.
const CACHE_SLOTS: usize = 128;

fn new_gpu(device: Option<&str>) -> Gpu {
    match device {
        Some("cpu") => Gpu::new_cpu(&KERNELS),
        Some("gpu") | Some("wgpu") => Gpu::new_wgpu(&KERNELS),
        _ => Gpu::new(&KERNELS),
    }
}

/// Read one of a graph's named stage buffers.
fn read_named(gpu: &Gpu, v: &[(String, DeviceBuffer, usize)], name: &str) -> Option<Vec<f32>> {
    v.iter().find(|(n, _, _)| n == name).map(|(_, b, l)| gpu.read(b, *l))
}

/// The encode graph for a fixed clip size, with weights resident.
pub struct WanVaeEncoder {
    gpu: Gpu,
    steps: Vec<Step>,
    x_in: DeviceBuffer,
    in_len: usize,
    out: DeviceBuffer,
    out_len: usize,
    stages: Vec<(String, DeviceBuffer, usize)>,
    taps: Vec<(String, DeviceBuffer, usize)>,
    latent_frames: u32,
}

impl WanVaeEncoder {
    /// Build the encode graph for a `[3, frames, h, w]` clip.
    ///
    /// `chunks` is the temporal split, in input frames per chunk. It must start
    /// at 1 (the causal first frame is its own chunk upstream) and sum to
    /// `frames`; every later chunk must be a multiple of 4 so the two temporal
    /// downsamples land on whole windows. [`WanVaeConfig::encode_chunks`] is
    /// upstream's plan; the result does not depend on which valid plan is used,
    /// which is what `vae_parity.rs` asserts.
    pub fn build(
        cfg: &WanVaeConfig,
        tensors: &Tensors,
        chunks: &[u32],
        h: u32,
        w: u32,
        device: Option<&str>,
    ) -> WanVaeEncoder {
        let frames: u32 = chunks.iter().sum();
        assert_eq!(chunks.first().copied(), Some(1), "the first encode chunk must be 1 frame");
        assert!(
            chunks[1..].iter().all(|c| c.is_multiple_of(4)),
            "later encode chunks must be multiples of 4"
        );
        let lat_t = cfg.latent_frames(frames).expect("frames must be 1+4k");
        let ds = 1 << (cfg.dim_mult.len() - 1);
        assert!(h.is_multiple_of(ds) && w.is_multiple_of(ds), "{h}x{w} is not a multiple of {ds}");
        let (lh, lw) = (h / ds, w / ds);

        let gpu = new_gpu(device);
        let mut b = Builder3d::new(&gpu, tensors, taps_enabled());
        let x_in = gpu.storage(3u64 * frames as u64 * h as u64 * w as u64);
        let video = T3 { buf: x_in.clone(), c: 3, t: frames, h, w };

        let zc2 = 2 * cfg.z_dim;
        let enc_out = T3 { buf: gpu.storage(zc2 as u64 * lat_t as u64 * lh as u64 * lw as u64), c: zc2, t: lat_t, h: lh, w: lw };
        let mut cache = FeatCache::new(CACHE_SLOTS);
        let (mut t0, mut lt0) = (0u32, 0u32);
        for &n in chunks {
            let chunk = b.time_slice(&video, t0, n);
            let y = encoder_chunk(&mut b, cfg, &chunk, &mut cache);
            b.free(chunk);
            b.time_place(&y, &enc_out.buf, lat_t, lt0);
            lt0 += y.t;
            t0 += n;
            b.free(y);
        }
        assert_eq!(lt0, lat_t, "chunked encode produced {lt0} latent frames, expected {lat_t}");

        let moments = b.conv("conv1", zc2, Conv3d::point(), &enc_out);
        let mu = b.chan_slice(&moments, 0, cfg.z_dim);
        let log_var = b.chan_slice(&moments, cfg.z_dim, cfg.z_dim);
        // `(mu - mean) * (1/std)`, in upstream's order: the subtraction first,
        // then a multiply by the f32 reciprocal upstream also precomputes.
        let neg_mean: Vec<f32> = cfg.latents_mean.iter().map(|v| -v).collect();
        let inv_std: Vec<f32> = cfg.latents_std.iter().map(|v| 1.0f32 / v).collect();
        let centred = b.add_chan("latents_neg_mean", &neg_mean, &mu);
        let latent = b.scale_chan("latents_inv_std", &inv_std, &centred);
        b.free(centred);

        let stages = vec![
            ("enc_out".to_string(), enc_out.buf.clone(), enc_out.len() as usize),
            ("moments".to_string(), moments.buf.clone(), moments.len() as usize),
            ("mu".to_string(), mu.buf.clone(), mu.len() as usize),
            ("log_var".to_string(), log_var.buf.clone(), log_var.len() as usize),
        ];
        let out_len = latent.len() as usize;
        let (steps, taps) = b.finish();
        WanVaeEncoder {
            gpu,
            steps,
            x_in,
            in_len: 3 * (frames * h * w) as usize,
            out: latent.buf,
            out_len,
            stages,
            taps,
            latent_frames: lat_t,
        }
    }

    /// Encode `[3, frames, h, w]` (row-major) into the NORMALISED latent
    /// `[z_dim, 1+k, h/8, w/8]`.
    pub fn encode(&self, video: &[f32]) -> Vec<f32> {
        assert_eq!(video.len(), self.in_len, "encode: {} values, expected {}", video.len(), self.in_len);
        self.gpu.write_f32(&self.x_in, video);
        self.gpu.submit(&[], &self.steps);
        self.gpu.read(&self.out, self.out_len)
    }

    /// A boundary tensor of the last [`WanVaeEncoder::encode`]: `enc_out`,
    /// `moments`, `mu` or `log_var`.
    pub fn read_stage(&self, name: &str) -> Option<Vec<f32>> {
        read_named(&self.gpu, &self.stages, name)
    }

    /// A per-block tap (only recorded under `BRAIN_WAN_VAE_TAPS`).
    pub fn read_tap(&self, name: &str) -> Option<Vec<f32>> {
        read_named(&self.gpu, &self.taps, name)
    }

    /// Latent frames this graph produces.
    pub fn latent_frames(&self) -> u32 {
        self.latent_frames
    }
}

/// The decode graph for a fixed latent size, with weights resident.
pub struct WanVaeDecoder {
    gpu: Gpu,
    steps: Vec<Step>,
    z_in: DeviceBuffer,
    in_len: usize,
    out: DeviceBuffer,
    out_len: usize,
    stages: Vec<(String, DeviceBuffer, usize)>,
    taps: Vec<(String, DeviceBuffer, usize)>,
    frames: u32,
}

impl WanVaeDecoder {
    /// Build the decode graph for a `[z_dim, lat_t, lh, lw]` latent.
    ///
    /// The chunking is upstream's and is NOT a parameter: one latent frame per
    /// chunk. See this module's header for why a larger chunk is a different
    /// model here, not an optimisation.
    pub fn build(
        cfg: &WanVaeConfig,
        tensors: &Tensors,
        lat_t: u32,
        lh: u32,
        lw: u32,
        device: Option<&str>,
    ) -> WanVaeDecoder {
        assert!(lat_t >= 1, "a latent needs at least one frame");
        let frames = 1 + 4 * (lat_t - 1);
        let up = 1 << (cfg.dim_mult.len() - 1);

        let gpu = new_gpu(device);
        let mut b = Builder3d::new(&gpu, tensors, taps_enabled());
        let z_in = gpu.storage(cfg.z_dim as u64 * lat_t as u64 * lh as u64 * lw as u64);
        let z = T3 { buf: z_in.clone(), c: cfg.z_dim, t: lat_t, h: lh, w: lw };

        // `z / (1/std) + mean`. The division is done on the host against the
        // same f32 reciprocal upstream holds, so this is a multiply by
        // `1/(1/std)` rather than by `std` - within one ulp of upstream's
        // per-element divide, and the alternative (`* std`) is a different
        // number by the same amount.
        let recip: Vec<f32> = cfg.latents_std.iter().map(|v| 1.0f32 / (1.0f32 / v)).collect();
        let scaled = b.scale_chan("latents_std_recip", &recip, &z);
        let denorm = b.add_chan("latents_mean", &cfg.latents_mean, &scaled);
        b.free(scaled);
        let x = b.conv("conv2", cfg.z_dim, Conv3d::point(), &denorm);

        let out = T3 {
            buf: gpu.storage(3u64 * frames as u64 * (lh * up) as u64 * (lw * up) as u64),
            c: 3,
            t: frames,
            h: lh * up,
            w: lw * up,
        };
        let mut cache = FeatCache::new(CACHE_SLOTS);
        let mut t0 = 0u32;
        for i in 0..lat_t {
            let chunk = b.time_slice(&x, i, 1);
            let y = decoder_chunk(&mut b, cfg, &chunk, &mut cache);
            b.free(chunk);
            b.time_place(&y, &out.buf, frames, t0);
            t0 += y.t;
            b.free(y);
        }
        assert_eq!(t0, frames, "chunked decode produced {t0} frames, expected {frames}");

        let stages = vec![
            ("z_denorm".to_string(), denorm.buf.clone(), denorm.len() as usize),
            ("dec_conv2".to_string(), x.buf.clone(), x.len() as usize),
        ];
        let out_len = out.len() as usize;
        let (steps, taps) = b.finish();
        WanVaeDecoder {
            gpu,
            steps,
            z_in,
            in_len: (cfg.z_dim * lat_t * lh * lw) as usize,
            out: out.buf,
            out_len,
            stages,
            taps,
            frames,
        }
    }

    /// Decode a NORMALISED latent `[z_dim, lat_t, lh, lw]` into
    /// `[3, 1+4(lat_t-1), lh*8, lw*8]`. No clamp is applied - upstream clamps
    /// to `[-1, 1]` outside the model, in `WanVAE.decode`.
    pub fn decode(&self, latent: &[f32]) -> Vec<f32> {
        assert_eq!(latent.len(), self.in_len, "decode: {} values, expected {}", latent.len(), self.in_len);
        self.gpu.write_f32(&self.z_in, latent);
        self.gpu.submit(&[], &self.steps);
        self.gpu.read(&self.out, self.out_len)
    }

    /// A boundary tensor of the last [`WanVaeDecoder::decode`]: `z_denorm` or
    /// `dec_conv2`.
    pub fn read_stage(&self, name: &str) -> Option<Vec<f32>> {
        read_named(&self.gpu, &self.stages, name)
    }

    /// A per-block tap (only recorded under `BRAIN_WAN_VAE_TAPS`).
    pub fn read_tap(&self, name: &str) -> Option<Vec<f32>> {
        read_named(&self.gpu, &self.taps, name)
    }

    /// Video frames this graph produces.
    pub fn frames(&self) -> u32 {
        self.frames
    }

    /// The device this graph was built on, for a caller that submits it itself.
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// The recorded decode graph - every chunk's dispatches, in order. A
    /// profiler groups these by kernel kind; nothing else should need them.
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_matches_the_reference_module_list() {
        let cfg = WanVaeConfig::wan21();
        // 11 encoder blocks: res,res,down2d | res,res,down3d | res,res,down3d | res,res
        let enc = cfg.enc_blocks();
        assert_eq!(enc.len(), 11);
        assert_eq!(enc[0], Block::Res { cin: 96, cout: 96 });
        assert_eq!(enc[2], Block::Down { dim: 96, temporal: false });
        assert_eq!(enc[3], Block::Res { cin: 96, cout: 192 });
        assert_eq!(enc[5], Block::Down { dim: 192, temporal: true });
        assert_eq!(enc[8], Block::Down { dim: 384, temporal: true });
        assert_eq!(enc[10], Block::Res { cin: 384, cout: 384 });
        // 15 decoder blocks: 3 res + up, three times, then 3 res.
        let dec = cfg.dec_blocks();
        assert_eq!(dec.len(), 15);
        assert_eq!(dec[0], Block::Res { cin: 384, cout: 384 });
        assert_eq!(dec[3], Block::Up { dim: 384, temporal: true });
        assert_eq!(dec[4], Block::Res { cin: 192, cout: 384 });
        assert_eq!(dec[7], Block::Up { dim: 384, temporal: true });
        assert_eq!(dec[8], Block::Res { cin: 192, cout: 192 });
        assert_eq!(dec[11], Block::Up { dim: 192, temporal: false });
        assert_eq!(dec[12], Block::Res { cin: 96, cout: 96 });
    }

    #[test]
    fn manifest_counts_the_shipped_checkpoint() {
        // 194 tensors is what both `Wan2.1_VAE.pth` and the diffusers export
        // ship; a schedule that drifts by one block changes this number.
        let m = WanVaeConfig::wan21().tensor_manifest();
        assert_eq!(m.len(), 194, "manifest has {} tensors", m.len());
        let names: std::collections::HashSet<&str> = m.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names.len(), m.len(), "duplicate tensor name in the manifest");
        assert!(names.contains("encoder.downsamples.5.time_conv.weight"));
        assert!(names.contains("decoder.upsamples.7.time_conv.weight"));
        assert!(!names.contains("decoder.upsamples.11.time_conv.weight"));
        assert!(names.contains("decoder.upsamples.4.shortcut.weight"));
    }

    #[test]
    fn latent_frame_rule() {
        let cfg = WanVaeConfig::wan21();
        assert_eq!(cfg.latent_frames(81), Some(21));
        assert_eq!(cfg.latent_frames(1), Some(1));
        assert_eq!(cfg.latent_frames(9), Some(3));
        assert_eq!(cfg.latent_frames(80), None);
        assert_eq!(cfg.encode_chunks(9), vec![1, 4, 4]);
    }
}
