// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Tiled encode and decode for the 2-D image autoencoder: one device graph at
//! a time over an overlapping cover, so peak VRAM is bounded by the TILE and
//! not by the image.
//!
//! Swedish Embedded AB implements memory-bounded image-model inference for its
//! clients. If your team needs expertise in running high-resolution generative
//! pipelines inside a fixed VRAM budget, you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! # The problem this closes
//!
//! A `VaeDecoder`'s activations scale with the OUTPUT image, not with the
//! checkpoint: at FLUX.2's channel schedule one live buffer at each resolution
//! level costs [`crate::level_bytes_per_pixel`] = 928 bytes per output pixel,
//! and the graph keeps about eleven of them alive. A 1024x1024 frame is
//! therefore 10.9 GiB and fits a 24 GiB card with room; a 2048x2048 one is
//! 40.8 GiB and cannot fit at all. Attention is not the ceiling in this
//! pipeline - the DiT's is already linear in token count - so the conv VAE is
//! where a high-resolution run dies, and it dies in the LAST stage, after
//! every denoise step has been paid for.
//!
//! # What is exact and what is not
//!
//! A cover that yields ONE tile is the whole-image path, bit for bit
//! ([`crate::tiling2d::TilePlan2d`] returns a single ramp-free interval for an
//! axis that fits), which is what makes the automatic threshold safe: below it
//! nothing changes. A cover that genuinely splits is an APPROXIMATION of the
//! whole-image result and this module does not pretend otherwise. Three
//! reasons, all structural:
//!
//! * the conv stack's receptive field is wide - summing every 3x3 conv at the
//!   resolution it runs at, the radius is ~130 output pixels - so a tile's
//!   borders saw less context than the whole image would have given them, and
//!   at a 512-pixel tile that leaves only a modest exact interior;
//! * the mid-block self-attention is GLOBAL over whatever extent it is handed,
//!   so a tile attends only within itself (measured small here: see below);
//! * GroupNorm normalises over the extent it is handed, which is the big one
//!   and is why [`Tiling::global_gn`] exists.
//!
//! The seam is a weighted average of two tiles that disagree, and the
//! trapezoidal ramp makes that a gradient rather than an edge.
//!
//! # Measured, on the real FLUX.2 VAE
//!
//! A 1024x1024 decode of an in-distribution latent (a synthetic scene put
//! through this same VAE's encoder), nine 512-pixel tiles at a 128-pixel
//! overlap, against the whole-image decode:
//!
//! | variant | cosine | rel_l2 | PSNR vs the whole-image decode |
//! |---|---|---|---|
//! | [`Tiling::global_gn`] (default) | 0.99950 | 0.036 | **38.5 dB** |
//! | per-tile norms (`diffusers`' own behaviour) | 0.99435 | 0.142 | 26.5 dB |
//!
//! For scale: this VAE's own round-trip reconstruction of that image is
//! 38.6 dB, so with synchronised statistics the tiling error is at the level
//! of the autoencoder's own, and without them it is twelve decibels worse -
//! per-tile brightness and contrast steps, which is exactly the artefact
//! naively tiled VAEs are known for. Dropping the mid-block attention instead
//! changes rel_l2 from 0.036 to 0.038, which is how we know it is not the term
//! that matters here.
//!
//! # Two passes, and what the first one buys
//!
//! With [`Tiling::global_gn`] the cover runs twice: once to learn what the
//! whole image's GroupNorm statistics are, once to apply them to every tile
//! (see [`GnStats`]). That is where the 12 dB above comes from, and it costs a
//! second pass - which is the right trade for a stage that runs ONCE per
//! generation, after tens of denoise steps. A cover that does not split skips
//! it entirely (its one tile already is the whole image), so nothing below the
//! threshold pays for it.
//!
//! # One graph per tile SHAPE, not per tile
//!
//! A tile's graph is built, used for every tile of that shape, and dropped
//! before the next shape's is built - the same "fresh resources per unit of
//! work" pattern `ltxv::vae3d::LtxVaeTiledDecoder` uses, and the reason peak
//! VRAM is one tile's rather than the image's. A `split_by_size` cover has at
//! most four distinct shapes (interior, short last row, short last column, and
//! their corner) however many tiles it has.

use gpu_core::Gpu;

use crate::blocks::{GnStats, Tensors};
use crate::config::VaeConfig;
use crate::decoder::{decoder_device_bytes, encoder_device_bytes, VaeDecoder, VaeEncoder};
use crate::tiling2d::{slice_src, Blender2d, Tile2d, TilePlan2d};

/// Tile geometry, in PIXELS of the full-resolution image - the unit an
/// operator thinks in, and the unit `diffusers` states its own defaults in.
/// Both are quantised down to whole latent cells before use
/// ([`Tiling::latent`]), because a tile whose pixel extent is not a whole
/// number of latent cells cannot be run by either graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tiling {
    pub tile_px: u32,
    pub overlap_px: u32,
    /// Run the cover TWICE, the first pass only to learn the whole image's
    /// GroupNorm statistics and the second to apply them to every tile (see
    /// [`GnStats`]). On by default, because without it a tiled decode has
    /// visible per-tile brightness steps that no overlap blend can remove -
    /// the deviation is a near-uniform shift across each tile's interior, not
    /// a seam. Costs a second pass over the cover.
    pub global_gn: bool,
}

impl Tiling {
    /// `diffusers`' own `AutoencoderKL` tiling defaults for this architecture:
    /// a 512-pixel tile (`tile_sample_min_size`) overlapping its neighbours by
    /// a quarter of that (`tile_overlap_factor = 0.25`).
    ///
    /// Transcribed rather than re-derived: 512 is the resolution this family
    /// of VAEs was trained at, so a tile of that size is the largest extent
    /// whose statistics the network has actually seen, and the quarter overlap
    /// is what upstream ships against the same receptive field. At FLUX.2's
    /// schedule one such tile predicts 3.4 GiB (measured on a P40: 3.2 GiB
    /// really allocated), where one 2048x2048 whole-image decode would want
    /// 40.8 GiB.
    pub const AUTO: Tiling = Tiling { tile_px: 512, overlap_px: 128, global_gn: true };

    pub fn new(tile_px: u32, overlap_px: u32) -> Tiling {
        assert!(tile_px > 0, "Tiling: tile must be non-zero");
        assert!(overlap_px < tile_px, "Tiling: overlap {overlap_px} must be < tile {tile_px}");
        Tiling { tile_px, overlap_px, global_gn: true }
    }

    /// Let every tile normalise over itself - the single-pass behaviour, and
    /// what `diffusers`' own `tiled_decode` does. Half the work and visibly
    /// worse; kept reachable so the two arms can be compared
    /// (`crates/vae/tests/tiled_parity.rs` measures the difference, which it
    /// could not do if the losing arm were unreachable).
    pub fn local_gn(self) -> Tiling {
        Tiling { global_gn: false, ..self }
    }

    /// `(tile, overlap)` in LATENT cells for a VAE whose spatial factor is
    /// `scale`. The tile is floored at `max(2, overlap + 1)` - the smallest
    /// legal argument to `split_by_size` - so a pathologically small pixel
    /// tile degrades to a coarse cover rather than to a panic.
    pub fn latent(&self, scale: u32) -> (usize, usize) {
        let scale = scale.max(1) as usize;
        let overlap = self.overlap_px as usize / scale;
        let tile = (self.tile_px as usize / scale).max(2).max(overlap + 1);
        (tile, overlap)
    }
}

/// The device budget one whole-image VAE graph may predict before the tiled
/// path takes over.
///
/// A card-sized constant, and it says so: 18 GiB is what leaves real headroom
/// on the 24 GiB Tesla P40s this repo's imaging work is measured on, once the
/// driver's own context and the pipeline's other resident parts are counted.
/// It is deliberately compared against [`crate::decoder_device_bytes_for_pixels`]
/// rather than against a pixel count, so the decision follows the CONFIG - a
/// VAE with a different channel schedule gets a different threshold for free,
/// which a hardcoded resolution could not do.
///
/// At FLUX.2's schedule this keeps every size the pipeline generates today on
/// the exact path (1024x1024 predicts 10.9 GiB, 1024x1536 15.9 GiB) and sends
/// 1536x1536 (23.4 GiB) and 2048x2048 (40.8 GiB) to the tiled one.
pub const WHOLE_GRAPH_MAX_BYTES: u64 = 18 << 30;

/// `BRAIN_VAE_TILE=1`/`0` forces tiling on/off; anything else (or unset) is
/// the [`WHOLE_GRAPH_MAX_BYTES`] policy below. Forcing it ON at a size that
/// already fits is the supported way to compare the two paths.
fn forced() -> Option<bool> {
    match std::env::var("BRAIN_VAE_TILE").ok().as_deref() {
        Some("1") | Some("on") | Some("true") => Some(true),
        Some("0") | Some("off") | Some("false") => Some(false),
        _ => None,
    }
}

/// Whether a decode producing `px` output pixels should take the tiled path.
pub fn should_tile_decode(cfg: &VaeConfig, px: u64) -> bool {
    forced().unwrap_or_else(|| crate::decoder_device_bytes_for_pixels(cfg, px) > WHOLE_GRAPH_MAX_BYTES)
}

/// Whether an encode reading `px` input pixels should take the tiled path.
pub fn should_tile_encode(cfg: &VaeConfig, px: u64) -> bool {
    forced().unwrap_or_else(|| crate::encoder_device_bytes_for_pixels(cfg, px) > WHOLE_GRAPH_MAX_BYTES)
}

/// Output pixels one [`Tiling::AUTO`] tile covers - the cap a tiled estimate
/// is bounded by. Quantised through [`Tiling::latent`] so it is the tile that
/// will really be built, not the pixel number it was asked for.
fn auto_tile_pixels(cfg: &VaeConfig) -> u64 {
    let (tile, _) = Tiling::AUTO.latent(cfg.upscale_factor());
    let side = tile as u64 * cfg.upscale_factor() as u64;
    side * side
}

/// Device bytes a TILED decode of a `lh x lw` latent holds: the largest single
/// tile's graph, since only one is resident at a time.
pub fn decoder_device_bytes_tiled(cfg: &VaeConfig, lh: u32, lw: u32, tiling: Tiling) -> u64 {
    let (tile, overlap) = tiling.latent(cfg.upscale_factor());
    // Scale 1: the cover's INPUT shapes are what a graph is built at, and
    // those are on the latent grid whatever the output scale is.
    let (th, tw) = TilePlan2d::decode(lh as usize, lw as usize, tile, overlap, 1).max_src_shape();
    decoder_device_bytes(cfg, th as u32, tw as u32)
}

/// Device bytes a TILED encode of an `h x w` image holds - see
/// [`decoder_device_bytes_tiled`].
pub fn encoder_device_bytes_tiled(cfg: &VaeConfig, h: u32, w: u32, tiling: Tiling) -> u64 {
    let scale = cfg.upscale_factor();
    let (tile, overlap) = tiling.latent(scale);
    let (th, tw) = TilePlan2d::encode((h / scale) as usize, (w / scale) as usize, tile, overlap, scale as usize)
        .max_src_shape();
    encoder_device_bytes(cfg, th as u32, tw as u32)
}

/// What a decode of `px` output pixels will REALLY cost on the device: the
/// whole-image figure below the threshold, one tile's above it.
///
/// This is the function a placement decision wants. Pricing a high-resolution
/// run with [`crate::decoder_device_bytes_for_pixels`] once tiling is engaged
/// over-reserves by the ratio of the image to the tile - which on a busy
/// machine turns a run the hardware can do into a refusal.
pub fn decoder_device_bytes_for_pixels_planned(cfg: &VaeConfig, px: u64) -> u64 {
    let px = if should_tile_decode(cfg, px) { px.min(auto_tile_pixels(cfg)) } else { px };
    crate::decoder_device_bytes_for_pixels(cfg, px)
}

/// [`decoder_device_bytes_for_pixels_planned`] for the encode direction.
pub fn encoder_device_bytes_for_pixels_planned(cfg: &VaeConfig, px: u64) -> u64 {
    let px = if should_tile_encode(cfg, px) { px.min(auto_tile_pixels(cfg)) } else { px };
    crate::encoder_device_bytes_for_pixels(cfg, px)
}

/// A tile's share of the image, for weighting its GroupNorm statistics.
///
/// The true per-group element count at a given norm is
/// `(channels/groups) * h * w` at that norm's own resolution level, which is
/// the tile's input area times a constant that is the same for every tile. A
/// weighted mean is unchanged by that constant, so the area is the weight.
fn tile_weight(tile: Tile2d<'_>) -> f64 {
    (tile.h.src_len() * tile.w.src_len()) as f64
}

/// Accumulates each tile's per-group GroupNorm `[mean, rstd]` into one set of
/// statistics for the whole image.
///
/// Means combine as a weighted mean; variances cannot, so each tile's is
/// turned back into a second moment (`var + mean^2`), combined, and turned
/// back at the end - the standard pooled-variance identity, in `f64` because
/// the subtraction `E[x^2] - E[x]^2` is where a naive `f32` accumulation loses
/// its significant digits.
///
/// One deliberate approximation: overlapping tiles count their shared region
/// twice, so the overlap is slightly over-weighted. The alternative is to
/// weight each tile by its blend mask, which needs the per-site resolution the
/// statistics deliberately do not carry, for a correction far below the
/// difference this whole mechanism is closing.
struct GlobalGn {
    /// Per norm site, per group: `(sum of w*mean, sum of w*(var + mean^2))`.
    acc: Vec<Vec<(f64, f64)>>,
    weight: f64,
    eps: f64,
}

impl GlobalGn {
    fn new(eps: f64) -> GlobalGn {
        GlobalGn { acc: Vec::new(), weight: 0.0, eps }
    }

    /// One tile's statistics, as [`VaeDecoder::read_gn_stats`] returns them.
    fn add(&mut self, stats: &[Vec<f32>], w: f64) {
        assert!(!stats.is_empty(), "tiled GroupNorm: the collecting graph recorded no norms");
        if self.acc.is_empty() {
            self.acc = stats.iter().map(|s| vec![(0.0, 0.0); s.len() / 2]).collect();
        }
        assert_eq!(self.acc.len(), stats.len(), "tiled GroupNorm: tile shapes disagree on how many norms the graph has");
        for (site, s) in self.acc.iter_mut().zip(stats) {
            assert_eq!(site.len() * 2, s.len(), "tiled GroupNorm: tile shapes disagree on the group count");
            for (j, a) in site.iter_mut().enumerate() {
                let (mean, rstd) = (s[2 * j] as f64, s[2 * j + 1] as f64);
                // `rstd = 1/sqrt(var + eps)`, so this inverts the kernel's own
                // last line. Floored at zero: a group of constant values has
                // `var = 0` and rounding can put the inversion just below it.
                let var = (1.0 / (rstd * rstd) - self.eps).max(0.0);
                a.0 += w * mean;
                a.1 += w * (var + mean * mean);
            }
        }
        self.weight += w;
    }

    /// The whole image's `[mean, rstd]` per group, per site - ready to hand
    /// straight back as [`GnStats::Inject`].
    fn finish(self) -> Vec<Vec<f32>> {
        assert!(self.weight > 0.0, "tiled GroupNorm: no tiles contributed statistics");
        self.acc
            .iter()
            .map(|site| {
                let mut out = Vec::with_capacity(site.len() * 2);
                for (sum, sq) in site {
                    let mean = sum / self.weight;
                    let var = (sq / self.weight - mean * mean).max(0.0);
                    out.push(mean as f32);
                    out.push((1.0 / (var + self.eps).sqrt()) as f32);
                }
                out
            })
            .collect()
    }
}

/// Decodes a `[latent_ch, lh, lw]` latent as an overlapping cover of tiles,
/// one device-resident graph at a time, blending the decoded pixel tiles with
/// trapezoidal masks. See this module's header for what that is and is not.
pub struct VaeTiledDecoder<'a> {
    gpu: &'a Gpu,
    cfg: VaeConfig,
    /// BORROWED, not owned: the tiled path needs the host weights across one
    /// graph build per distinct tile shape, and a caller that decodes several
    /// images against the same weights must not pay a host copy per image.
    tensors: &'a Tensors,
    lh: u32,
    lw: u32,
    plan: TilePlan2d,
    tiling: Tiling,
}

impl<'a> VaeTiledDecoder<'a> {
    /// A tiled decoder for a `[latent_ch, lh, lw]` latent under `tiling`.
    /// Constructing it costs no VRAM at all - no device graph exists until
    /// [`VaeTiledDecoder::decode`] runs.
    pub fn new(gpu: &'a Gpu, cfg: VaeConfig, tensors: &'a Tensors, lh: u32, lw: u32, tiling: Tiling) -> VaeTiledDecoder<'a> {
        assert!(lh >= 1 && lw >= 1, "a latent needs at least one cell per axis");
        let (tile, overlap) = tiling.latent(cfg.upscale_factor());
        let plan = TilePlan2d::decode(lh as usize, lw as usize, tile, overlap, cfg.upscale_factor() as usize);
        VaeTiledDecoder { gpu, cfg, tensors, lh, lw, plan, tiling }
    }

    /// The same under [`Tiling::AUTO`].
    pub fn auto(gpu: &'a Gpu, cfg: VaeConfig, tensors: &'a Tensors, lh: u32, lw: u32) -> VaeTiledDecoder<'a> {
        Self::new(gpu, cfg, tensors, lh, lw, Tiling::AUTO)
    }

    /// The tile cover this decoder will run.
    pub fn plan(&self) -> &TilePlan2d {
        &self.plan
    }

    /// Device bytes the largest of this cover's graphs will hold - what a
    /// placement decision reserves for a tiled decode.
    pub fn device_bytes(&self) -> u64 {
        decoder_device_bytes_tiled(&self.cfg, self.lh, self.lw, self.tiling)
    }

    /// Decode `[latent_ch * lh * lw]` into `[out_ch * lh*scale * lw*scale]`,
    /// one tile at a time. `on_tile(done, total)` is called after each tile so
    /// a caller can report progress on a multi-minute stage.
    /// `on_tile`'s `total` counts BOTH passes when
    /// [`Tiling::global_gn`] is on, so a progress bar built from it reaches
    /// the end exactly once.
    pub fn decode_with(&self, latent: &[f32], mut on_tile: impl FnMut(usize, usize)) -> Vec<f32> {
        let c = self.cfg.latent_channels as usize;
        let (lh, lw) = (self.lh as usize, self.lw as usize);
        assert_eq!(latent.len(), c * lh * lw, "tiled decode: {} values, expected {}", latent.len(), c * lh * lw);

        let tiles = self.plan.tiles();
        // A cover that does not split needs no synchronisation: its one tile
        // IS the whole image, so the local statistics are already the global
        // ones - and skipping the pass is what keeps a one-tile cover bit
        // identical to the untiled path rather than merely close to it.
        let sync = self.tiling.global_gn && tiles.len() > 1;
        let total = tiles.len() * (1 + usize::from(sync));
        let mut done = 0usize;

        // Pass 1 (only when asked): every tile, for its GroupNorm statistics
        // alone. The pixels it produces are discarded - they are the ones with
        // the per-tile normalisation this exists to remove.
        let global = sync.then(|| self.gn_stats_with(latent, |d| { done += d; on_tile(done, total) }));

        let mut blender = Blender2d::new(&self.plan, self.cfg.out_channels as usize);
        for ((th, tw), idxs) in self.plan.by_src_shape() {
            let gn = match &global {
                Some(v) => GnStats::Inject(v.clone()),
                None => GnStats::Local,
            };
            let dec =
                VaeDecoder::from_diffusers_on_gn(self.gpu, self.cfg.clone(), self.tensors, th as u32, tw as u32, gn);
            for i in idxs {
                let tile = tiles[i];
                blender.add(tile, &dec.decode(&slice_src(latent, c, (lh, lw), tile)));
                done += 1;
                on_tile(done, total);
            }
            // Explicit: this shape's ACTIVATION buffers must be gone before
            // the next shape's are allocated. That is the whole mechanism.
            drop(dec);
        }
        blender.finish()
    }

    /// [`VaeTiledDecoder::decode_with`] with no progress callback.
    pub fn decode(&self, latent: &[f32]) -> Vec<f32> {
        self.decode_with(latent, |_, _| {})
    }

    /// What the WHOLE image's GroupNorm statistics are, learned by running the
    /// cover once and pooling every tile's - pass 1 of a
    /// [`Tiling::global_gn`] decode, on its own.
    ///
    /// Public because it is the mechanism's own gate: these values can be
    /// compared directly against a whole-image graph's
    /// [`VaeDecoder::read_gn_stats`], which is a far sharper test of the
    /// pooling arithmetic than any image-space tolerance.
    pub fn gn_stats(&self, latent: &[f32]) -> Vec<Vec<f32>> {
        self.gn_stats_with(latent, |_| {})
    }

    fn gn_stats_with(&self, latent: &[f32], mut tick: impl FnMut(usize)) -> Vec<Vec<f32>> {
        let c = self.cfg.latent_channels as usize;
        let (lh, lw) = (self.lh as usize, self.lw as usize);
        let tiles = self.plan.tiles();
        let mut acc = GlobalGn::new(self.cfg.norm_eps as f64);
        for ((th, tw), idxs) in self.plan.by_src_shape() {
            let dec = VaeDecoder::from_diffusers_on_gn(
                self.gpu,
                self.cfg.clone(),
                self.tensors,
                th as u32,
                tw as u32,
                GnStats::Collect,
            );
            for i in idxs {
                let tile = tiles[i];
                dec.decode(&slice_src(latent, c, (lh, lw), tile));
                acc.add(&dec.read_gn_stats(), tile_weight(tile));
                tick(1);
            }
            drop(dec);
        }
        acc.finish()
    }
}

/// Encodes an `[in_ch, h, w]` image as an overlapping cover of tiles into the
/// moments `[2*latent_ch, h/scale, w/scale]`, blended on the LATENT grid.
///
/// The mirror of [`VaeTiledDecoder`]; the same caveats apply, plus one of its
/// own: the blend averages the posterior's log-variance channels as linearly
/// as it averages its mean. That is `diffusers`' behaviour too, and every
/// consumer in this workspace reads the mean
/// ([`VaeTiledEncoder::encode_mean`]).
pub struct VaeTiledEncoder<'a> {
    gpu: &'a Gpu,
    cfg: VaeConfig,
    tensors: &'a Tensors,
    h: u32,
    w: u32,
    plan: TilePlan2d,
    tiling: Tiling,
}

impl<'a> VaeTiledEncoder<'a> {
    /// A tiled encoder for an `[in_ch, h, w]` image (full-res, NOT latent
    /// size). `h` and `w` must be whole multiples of the VAE's spatial factor,
    /// which is what the untiled graph requires of them too.
    pub fn new(gpu: &'a Gpu, cfg: VaeConfig, tensors: &'a Tensors, h: u32, w: u32, tiling: Tiling) -> VaeTiledEncoder<'a> {
        let scale = cfg.upscale_factor();
        assert!(
            h.is_multiple_of(scale) && w.is_multiple_of(scale),
            "tiled encode: {h}x{w} is not a multiple of the VAE factor {scale}"
        );
        let (tile, overlap) = tiling.latent(scale);
        let plan = TilePlan2d::encode((h / scale) as usize, (w / scale) as usize, tile, overlap, scale as usize);
        VaeTiledEncoder { gpu, cfg, tensors, h, w, plan, tiling }
    }

    /// The same under [`Tiling::AUTO`].
    pub fn auto(gpu: &'a Gpu, cfg: VaeConfig, tensors: &'a Tensors, h: u32, w: u32) -> VaeTiledEncoder<'a> {
        Self::new(gpu, cfg, tensors, h, w, Tiling::AUTO)
    }

    pub fn plan(&self) -> &TilePlan2d {
        &self.plan
    }

    /// Device bytes the largest of this cover's graphs will hold.
    pub fn device_bytes(&self) -> u64 {
        encoder_device_bytes_tiled(&self.cfg, self.h, self.w, self.tiling)
    }

    /// Encode `[in_ch * h * w]` into the moments `[2*latent_ch * lh * lw]`,
    /// one tile at a time.
    pub fn encode_with(&self, image: &[f32], mut on_tile: impl FnMut(usize, usize)) -> Vec<f32> {
        let c = self.cfg.in_channels as usize;
        let (h, w) = (self.h as usize, self.w as usize);
        assert_eq!(image.len(), c * h * w, "tiled encode: {} values, expected {}", image.len(), c * h * w);

        let tiles = self.plan.tiles();
        // A cover that does not split needs no synchronisation: its one tile
        // IS the whole image, so the local statistics are already the global
        // ones - and skipping the pass is what keeps a one-tile cover bit
        // identical to the untiled path rather than merely close to it.
        let sync = self.tiling.global_gn && tiles.len() > 1;
        let total = tiles.len() * (1 + usize::from(sync));
        let mut done = 0usize;

        let global = sync.then(|| self.gn_stats_with(image, |d| { done += d; on_tile(done, total) }));

        let mut blender = Blender2d::new(&self.plan, 2 * self.cfg.latent_channels as usize);
        for ((th, tw), idxs) in self.plan.by_src_shape() {
            let gn = match &global {
                Some(v) => GnStats::Inject(v.clone()),
                None => GnStats::Local,
            };
            let enc =
                VaeEncoder::from_diffusers_on_gn(self.gpu, self.cfg.clone(), self.tensors, th as u32, tw as u32, gn);
            for i in idxs {
                let tile = tiles[i];
                blender.add(tile, &enc.encode(&slice_src(image, c, (h, w), tile)));
                done += 1;
                on_tile(done, total);
            }
            drop(enc);
        }
        blender.finish()
    }

    /// [`VaeTiledEncoder::encode_with`] with no progress callback.
    pub fn encode(&self, image: &[f32]) -> Vec<f32> {
        self.encode_with(image, |_, _| {})
    }

    /// The whole image's GroupNorm statistics - see
    /// [`VaeTiledDecoder::gn_stats`].
    pub fn gn_stats(&self, image: &[f32]) -> Vec<Vec<f32>> {
        self.gn_stats_with(image, |_| {})
    }

    fn gn_stats_with(&self, image: &[f32], mut tick: impl FnMut(usize)) -> Vec<Vec<f32>> {
        let c = self.cfg.in_channels as usize;
        let (h, w) = (self.h as usize, self.w as usize);
        let tiles = self.plan.tiles();
        let mut acc = GlobalGn::new(self.cfg.norm_eps as f64);
        for ((th, tw), idxs) in self.plan.by_src_shape() {
            let enc = VaeEncoder::from_diffusers_on_gn(
                self.gpu,
                self.cfg.clone(),
                self.tensors,
                th as u32,
                tw as u32,
                GnStats::Collect,
            );
            for i in idxs {
                let tile = tiles[i];
                enc.encode(&slice_src(image, c, (h, w), tile));
                acc.add(&enc.read_gn_stats(), tile_weight(tile));
                tick(1);
            }
            drop(enc);
        }
        acc.finish()
    }

    /// Encode and return only the posterior **mean** `[latent_ch * lh * lw]` -
    /// mirrors [`VaeEncoder::encode_mean`].
    pub fn encode_mean(&self, image: &[f32]) -> Vec<f32> {
        let m = self.encode(image);
        let (lh, lw) = self.plan.out_shape();
        m[..self.cfg.latent_channels as usize * lh * lw].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tile geometry a pixel-stated `Tiling` really runs at, on the VAE
    /// the pipeline uses (factor 8).
    #[test]
    fn the_auto_tiling_quantises_to_whole_latent_cells() {
        let cfg = VaeConfig::flux2();
        assert_eq!(Tiling::AUTO.latent(cfg.upscale_factor()), (64, 16));
        // A tile smaller than one latent cell cannot exist; it degrades to the
        // smallest legal split rather than panicking inside `split_by_size`.
        assert_eq!(Tiling::new(4, 2).latent(8), (2, 0));
    }

    /// The cost of a tiled decode must not know how big the image is.
    #[test]
    fn the_tiled_cost_is_the_same_at_every_resolution_past_one_tile() {
        let cfg = VaeConfig::flux2();
        let at = |lh, lw| decoder_device_bytes_tiled(&cfg, lh, lw, Tiling::AUTO);
        assert_eq!(at(256, 256), at(1024, 1024), "4x the latent, same tile");
        assert_eq!(at(256, 256), decoder_device_bytes(&cfg, 64, 64), "priced at one AUTO tile");
        // Below one tile there is nothing to bound: the plan is the image.
        assert_eq!(at(32, 48), decoder_device_bytes(&cfg, 32, 48));
    }
}
