// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device-side scene + rasterizer entry points: `render` is the tiled
//! pipeline (project → tile count → scan → emit → radix sort → tile ranges →
//! per-tile compositing → rgba8 pack), `render_naive_gpu` the per-pixel
//! oracle path for tests and tiny scenes.

use gpu_core::{f, DeviceBuffer, Gpu};

use crate::sort::{record_scan, record_sort_pairs, ScanScratch, SortScratch};
use crate::types::{Camera, Mode, RenderOpts, Splats};
use crate::Kernels;

pub const TILE: u32 = 16;

/// Device-resident gaussians (SoA, post-activation — see [`Splats`]).
pub struct GpuSplats {
    pub n: usize,
    pub means: DeviceBuffer,
    pub quats: DeviceBuffer,
    pub scales: DeviceBuffer,
    pub opacities: DeviceBuffer,
    pub colors: DeviceBuffer,
}

impl GpuSplats {
    pub fn upload(gpu: &Gpu, s: &Splats) -> GpuSplats {
        GpuSplats {
            n: s.len(),
            means: gpu.storage_init("splat.means", &s.means),
            quats: gpu.storage_init("splat.quats", &s.quats),
            scales: gpu.storage_init("splat.scales", &s.scales),
            opacities: gpu.storage_init("splat.opacities", &s.opacities),
            colors: gpu.storage_init("splat.colors", &s.colors),
        }
    }

    /// Zero-copy handoff of model-produced buffers living on the same `Gpu`.
    pub fn from_buffers(
        n: usize,
        means: DeviceBuffer,
        quats: DeviceBuffer,
        scales: DeviceBuffer,
        opacities: DeviceBuffer,
        colors: DeviceBuffer,
    ) -> GpuSplats {
        GpuSplats { n, means, quats, scales, opacities, colors }
    }
}

/// Pack the `splat_project` uniform: `[n W H aa | fx fy cx cy | near far eps2d
/// pad | viewmat rows]` — must match the WGSL Params field order.
fn project_params(n: usize, cam: &Camera, o: &RenderOpts) -> [u32; 24] {
    let v = cam.viewmat();
    let mut p = [0u32; 24];
    p[0] = n as u32;
    p[1] = cam.width;
    p[2] = cam.height;
    p[3] = o.antialiased as u32;
    p[4] = f(cam.fx);
    p[5] = f(cam.fy);
    p[6] = f(cam.cx);
    p[7] = f(cam.cy);
    p[8] = f(o.near);
    p[9] = f(o.far);
    p[10] = f(o.eps2d);
    p[11] = 0;
    for (i, val) in v.iter().enumerate() {
        p[12 + i] = f(*val);
    }
    p
}

/// What a tiled frame did (for HUD/telemetry).
#[derive(Clone, Copy, Debug, Default)]
pub struct RenderStats {
    /// Sorted (gaussian, tile) instances this frame.
    pub n_isects: usize,
    /// The instance buffer overflowed and the tail was dropped.
    pub clamped: bool,
}

/// Rasterizer with pre-allocated scratch sized at construction.
pub struct Renderer {
    ks: Kernels,
    max_n: usize,
    max_px: usize,
    max_tiles: usize,
    isect_cap: usize,
    pub proj: DeviceBuffer,
    pub img: DeviceBuffer,
    /// Per-pixel expected depth of the last [`Renderer::render`], written by
    /// the same compositing pass that produced `img` - see
    /// [`Renderer::read_depth`].
    pub depth: DeviceBuffer,
    packed: DeviceBuffer,
    counts: DeviceBuffer,
    count_scan: ScanScratch,
    keys_a: DeviceBuffer,
    vals_a: DeviceBuffer,
    keys_b: DeviceBuffer,
    vals_b: DeviceBuffer,
    sort: SortScratch,
    ranges: DeviceBuffer,
    /// Last render() state the backward replays.
    last: Option<(usize, bool, u32, u32)>, // (n_isects, vals_in_b, tiles_x, tiles_y)
    /// Grow the instance buffers when a frame needs more, instead of
    /// dropping its depth-latest splats - see [`Renderer::growable`].
    grow: bool,
}

impl Renderer {
    /// `isect_cap` bounds the sort working set; pass 0 for the default
    /// (`4*max_n`, capped at 16M). Overflow drops the depth-latest tail of one
    /// frame and reports `clamped`.
    pub fn new(gpu: &Gpu, ks: Kernels, max_n: usize, max_w: u32, max_h: u32, isect_cap: usize) -> Renderer {
        let max_px = (max_w * max_h) as usize;
        let max_tiles = (max_w.div_ceil(TILE) * max_h.div_ceil(TILE)) as usize;
        // Instances ≈ Σ tiles-per-gaussian: big screen-space gaussians touch
        // dozens of tiles each, so the floor matters more than the multiple.
        let cap = if isect_cap == 0 { (8 * max_n).clamp(1 << 20, 16 << 20) } else { isect_cap };
        Renderer {
            ks,
            max_n,
            max_px,
            max_tiles,
            isect_cap: cap,
            proj: gpu.storage(9 * max_n as u64),
            img: gpu.storage(4 * max_px as u64),
            depth: gpu.storage(max_px as u64),
            packed: gpu.storage(max_px as u64),
            counts: gpu.storage(max_n as u64),
            count_scan: ScanScratch::new(gpu, max_n),
            keys_a: gpu.storage(cap as u64),
            vals_a: gpu.storage(cap as u64),
            keys_b: gpu.storage(cap as u64),
            vals_b: gpu.storage(cap as u64),
            sort: SortScratch::new(gpu, cap),
            ranges: gpu.storage(2 * max_tiles as u64),
            last: None,
            grow: false,
        }
    }

    /// Never clamp: when a frame needs more (gaussian, tile) instances than
    /// the buffers hold, grow them (up to what one storage binding holds) and
    /// render all of it.
    ///
    /// For an optimizer, clamping is not a degraded frame but a wrong
    /// gradient: a dropped splat contributes to no pixel, so it gets no
    /// gradient at all and density control reads it as dead. A scene grown
    /// from a sparse cloud is a few thousand LARGE gaussians early on, each
    /// touching dozens of tiles, which is exactly when the default budget
    /// overflows.
    pub fn growable(mut self) -> Renderer {
        self.grow = true;
        self
    }

    /// Tiled render into `self.img` (+ packed rgba8). Two submissions with a
    /// 4-byte readback between them (the instance count that sizes the sort).
    pub fn render(&mut self, gpu: &Gpu, s: &GpuSplats, cam: &Camera, o: &RenderOpts) -> RenderStats {
        assert!(s.n <= self.max_n, "scene {} exceeds renderer max_n {}", s.n, self.max_n);
        let px = (cam.width * cam.height) as usize;
        assert!(px <= self.max_px);
        let tiles_x = cam.width.div_ceil(TILE);
        let tiles_y = cam.height.div_ceil(TILE);
        let n_tiles = (tiles_x * tiles_y) as usize;
        assert!(n_tiles <= self.max_tiles);
        // tile bits from the tile-id range; the rest of the 32-bit key is depth.
        let tile_bits = (n_tiles.next_power_of_two().trailing_zeros()).max(1);
        let depth_bits = 32 - tile_bits;

        // ---- pass 1: project + tile-count + scan → n_isects ----
        let mut steps = Vec::new();
        steps.push(gpu.step(
            self.ks.splat_project,
            &[&s.means, &s.quats, &s.scales, &s.opacities, &self.proj],
            &project_params(s.n, cam, o),
            s.n as u32,
        ));
        steps.push(gpu.step(
            self.ks.splat_tile_count,
            &[&self.proj, &self.counts],
            &[s.n as u32, tiles_x, tiles_y, TILE],
            s.n as u32,
        ));
        let ctotal = record_scan(gpu, &self.ks, &self.counts, s.n, &self.count_scan, &mut steps);
        gpu.submit(&[], &steps);
        let total = gpu.read(ctotal, 1)[0].to_bits() as usize;
        if self.grow && total > self.isect_cap {
            // keys and values are one word each, the largest single binding
            let ceiling = (gpu.max_storage_binding_bytes() / 4) as usize;
            let cap = (total + total / 4).min(ceiling);
            self.keys_a = gpu.storage(cap as u64);
            self.vals_a = gpu.storage(cap as u64);
            self.keys_b = gpu.storage(cap as u64);
            self.vals_b = gpu.storage(cap as u64);
            self.sort = SortScratch::new(gpu, cap);
            self.isect_cap = cap;
        }
        let clamped = total > self.isect_cap;
        let n_isects = total.min(self.isect_cap);

        // ---- pass 2: emit + sort + ranges + rasterize + pack ----
        let mut steps = Vec::new();
        steps.push(gpu.step(
            self.ks.splat_emit,
            &[&self.proj, &self.counts, &self.keys_a, &self.vals_a],
            &[s.n as u32, tiles_x, tiles_y, TILE, depth_bits, self.isect_cap as u32],
            s.n as u32,
        ));
        let mut vals_in_b = false;
        let (keys, vals) = if n_isects > 0 {
            let in_b = record_sort_pairs(
                gpu, &self.ks, &self.keys_a, &self.vals_a, &self.keys_b, &self.vals_b,
                n_isects, 32, &self.sort, &mut steps,
            );
            vals_in_b = in_b;
            if in_b { (&self.keys_b, &self.vals_b) } else { (&self.keys_a, &self.vals_a) }
        } else {
            (&self.keys_a, &self.vals_a)
        };
        if n_isects > 0 {
            steps.push(gpu.step(
                self.ks.splat_tile_ranges,
                &[keys, &self.ranges],
                &[n_isects as u32, depth_bits],
                n_isects as u32,
            ));
        }
        steps.push(gpu.step(
            self.ks.splat_rasterize,
            &[&self.proj, &s.colors, vals, &self.ranges, &self.img, &self.depth],
            &[
                cam.width,
                cam.height,
                tiles_x,
                tiles_y,
                (o.mode == Mode::Depth) as u32,
                f(o.bg[0]),
                f(o.bg[1]),
                f(o.bg[2]),
            ],
            (n_tiles * 64) as u32,
        ));
        steps.push(gpu.step(
            self.ks.splat_pack_rgba8,
            &[&self.img, &self.packed],
            &[px as u32],
            px as u32,
        ));
        gpu.submit(&[&self.ranges], &steps);
        self.last = Some((n_isects, vals_in_b, tiles_x, tiles_y));
        RenderStats { n_isects, clamped }
    }

    /// Read the packed frame back as tight RGB24 bytes (drops alpha).
    pub fn read_rgb24(&self, gpu: &Gpu, w: u32, h: u32) -> Vec<u8> {
        let px = (w * h) as usize;
        let packed = gpu.read(&self.packed, px);
        let mut out = Vec::with_capacity(px * 3);
        for v in packed {
            let bits = v.to_bits();
            out.push((bits & 0xff) as u8);
            out.push(((bits >> 8) & 0xff) as u8);
            out.push(((bits >> 16) & 0xff) as u8);
        }
        out
    }

    /// Read the RGBA f32 framebuffer (tests / headless render).
    pub fn read_rgba(&self, gpu: &Gpu, w: u32, h: u32) -> Vec<f32> {
        gpu.read(&self.img, (w * h) as usize * 4)
    }

    /// Read the per-pixel EXPECTED depth of the last [`Renderer::render`]:
    /// `(sum_i z_i alpha_i T_i) / A` in camera-space units, 0 where the frame
    /// has no geometry. Available from any render, not only `Mode::Depth` -
    /// depth supervision needs the colour and the depth of the same frame.
    pub fn read_depth(&self, gpu: &Gpu, w: u32, h: u32) -> Vec<f32> {
        gpu.read(&self.depth, (w * h) as usize)
    }

    /// Project + composite in **buffer order** (caller sorts by depth) and
    /// read back RGBA f32.
    pub fn render_naive_gpu(
        &self,
        gpu: &Gpu,
        s: &GpuSplats,
        cam: &Camera,
        o: &RenderOpts,
    ) -> Vec<f32> {
        assert!(s.n <= self.max_n);
        let px = (cam.width * cam.height) as usize;
        assert!(px <= self.max_px);
        let pp = project_params(s.n, cam, o);
        let project = gpu.step(
            self.ks.splat_project,
            &[&s.means, &s.quats, &s.scales, &s.opacities, &self.proj],
            &pp,
            s.n as u32,
        );
        let np = [
            s.n as u32,
            cam.width,
            cam.height,
            (o.mode == Mode::Depth) as u32,
            f(o.bg[0]),
            f(o.bg[1]),
            f(o.bg[2]),
        ];
        let naive = gpu.step(
            self.ks.splat_naive,
            &[&self.proj, &s.colors, &self.img],
            &np,
            px as u32,
        );
        gpu.submit(&[], &[project, naive]);
        gpu.read(&self.img, px * 4)
    }
}

/// Below this alpha a pixel's expected depth is not supervised.
///
/// Expected depth is the accumulated depth divided by the accumulated alpha,
/// so where a frame is nearly transparent the quotient is the ratio of two
/// numbers that are nearly zero - numerically meaningless in fp32, and
/// meaningless as geometry too, because there is nothing there to anchor. The
/// gradient is not actually amplified by the division (the weights of the
/// gaussians at such a pixel sum back to that same alpha), so this is a
/// numerical floor rather than a stability hack.
pub const MIN_DEPTH_ALPHA: f32 = 1e-2;

/// Split a per-pixel dL/d(EXPECTED depth) into the two channels
/// [`Renderer::render_bwd`] consumes, accumulating into both.
///
/// The forward reports `D = Dacc / A` with `Dacc = sum_i z_i alpha_i T_i` and
/// `A` the alpha output. So one upstream gradient reaches the rasterizer by
/// two routes - through the composite and through its normalizer - and a
/// caller that forgets the second gets a gradient that is wrong by exactly the
/// amount the geometry's opacity is about to change. Doing the split once,
/// here, is what keeps `splat::opt` and the gradient gate differentiating the
/// same function.
///
/// `v_dn`, `depth` and `ddepth` are `W*H`; `rgba` and `dimg` are `W*H*4`.
pub fn add_expected_depth_vjp(
    v_dn: &[f32],
    depth: &[f32],
    rgba: &[f32],
    dimg: &mut [f32],
    ddepth: &mut [f32],
) {
    assert_eq!(v_dn.len(), depth.len());
    assert_eq!(v_dn.len(), ddepth.len());
    assert_eq!(rgba.len(), 4 * v_dn.len());
    assert_eq!(dimg.len(), 4 * v_dn.len());
    for i in 0..v_dn.len() {
        let a = rgba[i * 4 + 3];
        if v_dn[i] == 0.0 || a < MIN_DEPTH_ALPHA {
            continue;
        }
        ddepth[i] += v_dn[i] / a;
        dimg[i * 4 + 3] -= v_dn[i] * depth[i] / a;
    }
}

/// Per-gaussian gradient buffers the backward accumulates into (caller
/// clears/consumes them; layouts match splat_project_bwd/splat_grad_reduce).
pub struct SplatGrads {
    pub d_gauss: DeviceBuffer,  // N*10: d_means(3), d_scales(3), d_quats(4)
    pub d_opac: DeviceBuffer,   // N
    pub d_colors: DeviceBuffer, // N*3
    /// N: sum of per-pixel 2D position gradient MAGNITUDES (AbsGS), the
    /// criterion density control splits on. Not a gradient - nothing consumes
    /// it in the chain rule - so it is accumulated alongside rather than
    /// inside `d_gauss`.
    pub d_absgrad: DeviceBuffer,
    /// N*2: the summed 2D position gradient, which is what upstream 3DGS
    /// splits on. Kept beside [`Self::d_absgrad`] so the difference between
    /// the two criteria is a measurement rather than a claim.
    pub d_sumgrad: DeviceBuffer,
}

impl SplatGrads {
    pub fn new(gpu: &Gpu, n: usize) -> SplatGrads {
        SplatGrads {
            d_gauss: gpu.storage(10 * n as u64),
            d_opac: gpu.storage(n as u64),
            d_colors: gpu.storage(3 * n as u64),
            d_absgrad: gpu.storage(n as u64),
            d_sumgrad: gpu.storage(2 * n as u64),
        }
    }
}

/// Words per gradient record in `recs`: one record per (tile, gaussian)
/// INSTANCE, written by `splat_bwd_tile_reduce.wgsl` and read by
/// `splat_bwd_keys.wgsl` / `splat_grad_reduce.wgsl` - five sigma partials,
/// the opacity partial, three weighted colour partials, the depth partial,
/// the summed per-pixel magnitude of the position partial, and the gaussian id
/// bitcast into the last slot as the sort key. Declared once here because the
/// host sizes the buffer and only the kernels know the stride -
/// `record_width` in `crates/kernels/tests/` gates the two against each other.
pub const RECORD_WORDS: usize = 12;

/// Channels per (instance, pixel) slot written by `splat_bwd_slots.wgsl`:
/// the record's eleven values, before the id.
pub const SLOT_CHANNELS: usize = 11;

/// Pixels per tile, the slot grid's inner dimension.
const TILE_PIXELS: usize = (TILE * TILE) as usize;

/// Bytes of slot grid one instance occupies while its band is reduced.
pub const fn slot_bytes_per_instance() -> u64 {
    (SLOT_CHANNELS * TILE_PIXELS * 4) as u64
}

/// Most slot bytes a band may use, whatever the binding limit allows: the
/// grid is transient, rewritten per band, and a smaller one is as fast.
const SLOT_BAND_BYTES: u64 = 256 << 20;

/// How many gradient records fit in ONE storage binding of `limit` bytes.
///
/// This is a real ceiling, not a tunable: `recs` is bound as a single
/// storage buffer, so a frame with more (tile, gaussian) instances than this
/// cannot be differentiated on the device at all.
pub fn max_records_for_binding(limit: u64) -> usize {
    (limit / (RECORD_WORDS * 4) as u64) as usize
}

/// The capacity to allocate for a pass needing `need` records, or an error
/// naming what the device refused.
///
/// A quarter of headroom keeps a fit whose instance count drifts upward from
/// reallocating every step, but the headroom is CLAMPED to the binding limit:
/// exceeding it to leave room to grow would fail a pass that would otherwise
/// have run.
pub fn record_capacity(need: usize, limit: u64) -> Result<usize, String> {
    let ceiling = max_records_for_binding(limit);
    if need > ceiling {
        return Err(format!(
            "one frame has {need} (tile, gaussian) instances, {:.1} GiB of gradient records, but \
             one storage binding on this device holds {ceiling} ({} MiB). The scene is too dense to \
             differentiate at this image size: thin it with `--prune` (voxel-merge duplicate \
             gaussians across overlapping views) or `--min-opacity`.",
            (need * RECORD_WORDS * 4) as f64 / (1u64 << 30) as f64,
            limit >> 20,
        ));
    }
    Ok((need + need / 4).min(ceiling))
}

/// The record capacity a [`BwdScratch`] starts with: `rec_cap`, or 8 per
/// gaussian when that is 0 - never more than ONE storage binding of `limit`
/// bytes holds.
///
/// The ceiling used to apply only when the buffers GREW, so a default start
/// sized per pixel (64 per pixel, 50.3M records at 1024x768, 2.2 GB) was
/// allocated unclamped past a 2 GiB binding and the first backward of any fit
/// at that size failed bind-group validation.
pub fn initial_record_capacity(max_n: usize, rec_cap: usize, limit: u64) -> usize {
    let want = if rec_cap == 0 { (8 * max_n).clamp(1 << 16, 16 << 20) } else { rec_cap };
    want.min(max_records_for_binding(limit))
}

/// Backward scratch. The record buffers start at `rec_cap` (default 8 per
/// gaussian) and [grow][BwdScratch::reserve_records] to the frame's instance
/// count; the slot grid is sized per band.
pub struct BwdScratch {
    recs: DeviceBuffer,
    rkeys_a: DeviceBuffer,
    rvals_a: DeviceBuffer,
    rkeys_b: DeviceBuffer,
    rvals_b: DeviceBuffer,
    rsort: SortScratch,
    granges: DeviceBuffer,
    pgrad: DeviceBuffer,
    /// The (instance, channel, pixel) grid one band of tiles writes and
    /// `splat_bwd_tile_reduce` sums; `slot_cap` instances.
    slots: DeviceBuffer,
    slot_cap: usize,
    /// A permanently zero `W*H` buffer bound as the depth-gradient input when
    /// the caller supplies none. The slots kernel always reads that binding,
    /// so an RGB-only backward needs something there that contributes
    /// nothing, and one zeroed allocation per scratch is cheaper than
    /// branching the kernel or carrying a second pipeline.
    no_depth: DeviceBuffer,
    rec_cap: usize,
    limit: Option<u64>,
}

impl BwdScratch {
    pub fn new(gpu: &Gpu, max_n: usize, max_px: usize, rec_cap: usize) -> BwdScratch {
        let limit = gpu.max_storage_binding_bytes();
        let cap = initial_record_capacity(max_n, rec_cap, limit);
        let mut s = BwdScratch {
            recs: gpu.storage(0),
            rkeys_a: gpu.storage(0),
            rvals_a: gpu.storage(0),
            rkeys_b: gpu.storage(0),
            rvals_b: gpu.storage(0),
            rsort: SortScratch::new(gpu, 1),
            granges: gpu.storage(2 * max_n as u64),
            pgrad: gpu.storage(10 * max_n as u64),
            slots: gpu.storage(0),
            slot_cap: 0,
            no_depth: gpu.storage(max_px as u64),
            rec_cap: 0,
            limit: None,
        };
        gpu.submit(&[&s.no_depth], &[]);
        s.alloc_records(gpu, cap);
        s
    }

    /// Size the record-keyed buffers for exactly `cap` records.
    fn alloc_records(&mut self, gpu: &Gpu, cap: usize) {
        self.recs = gpu.storage(RECORD_WORDS as u64 * cap as u64);
        self.rkeys_a = gpu.storage(cap as u64);
        self.rvals_a = gpu.storage(cap as u64);
        self.rkeys_b = gpu.storage(cap as u64);
        self.rvals_b = gpu.storage(cap as u64);
        self.rsort = SortScratch::new(gpu, cap);
        self.rec_cap = cap;
    }

    /// Make room for `need` gradient records, reallocating if the current
    /// buffers are too small - see [`record_capacity`] for the headroom and
    /// the device ceiling it is clamped to.
    ///
    /// Nothing is preserved across a grow, and nothing needs to be: the
    /// records for one `render_bwd` are written and consumed inside that one
    /// call.
    pub fn reserve_records(&mut self, gpu: &Gpu, need: usize) -> Result<(), String> {
        if need <= self.rec_cap {
            return Ok(());
        }
        let cap = record_capacity(need, self.limit(gpu))?;
        self.alloc_records(gpu, cap);
        Ok(())
    }

    /// Pretend the device's per-binding limit is `bytes`, so a test can force
    /// the banding path without building a scene big enough to reach the real
    /// one.
    pub fn with_binding_limit(mut self, bytes: u64) -> BwdScratch {
        self.limit = Some(bytes);
        self
    }

    fn limit(&self, gpu: &Gpu) -> u64 {
        self.limit.unwrap_or_else(|| gpu.max_storage_binding_bytes())
    }

    /// How many instances one band's slot grid may hold.
    fn band_instances(&self, gpu: &Gpu) -> usize {
        let bytes = self.limit(gpu).min(SLOT_BAND_BYTES);
        (bytes / slot_bytes_per_instance()) as usize
    }

    fn reserve_slots(&mut self, gpu: &Gpu, instances: usize) {
        if instances > self.slot_cap {
            self.slots = gpu.storage(instances as u64 * (SLOT_CHANNELS * TILE_PIXELS) as u64);
            self.slot_cap = instances;
        }
    }
}

/// Contiguous runs of tiles whose instances fit `cap` at a time, as
/// `(first tile, tile count, first instance, instance count)`, from the
/// per-tile instance ranges of the last render. Tiles are sorted by id, so a
/// run of tiles owns one contiguous run of instances.
///
/// A tile holding more than `cap` instances on its own gets bands of `cap`
/// of ITS instances each: every pixel of the tile replays the whole list in
/// every one of them and writes only that band's run, so the price of a
/// crowded tile (the far ground near a horizon) is repeated walks, not a
/// failed pass.
fn plan_bands(ranges: &[u32], cap: usize) -> Vec<(u32, u32, u32, u32)> {
    let cap = cap.max(1) as u32;
    let n_tiles = ranges.len() / 2;
    let mut out = Vec::new();
    // (first tile, first instance, one past the last instance) of the open band
    let mut open: Option<(usize, u32, u32)> = None;
    for t in 0..n_tiles {
        let (s, e) = (ranges[2 * t], ranges[2 * t + 1]);
        if e <= s {
            continue; // an empty tile belongs to whichever band surrounds it
        }
        if e - s > cap {
            if let Some((t0, k0, k1)) = open.take() {
                out.push((t0 as u32, (t - t0) as u32, k0, k1 - k0));
            }
            out.extend((s..e).step_by(cap as usize).map(|k| (t as u32, 1, k, cap.min(e - k))));
            continue;
        }
        match open {
            Some((t0, k0, _)) if e - k0 <= cap => open = Some((t0, k0, e)),
            Some((t0, k0, k1)) => {
                out.push((t0 as u32, (t - t0) as u32, k0, k1 - k0));
                open = Some((t, s, e));
            }
            None => open = Some((t, s, e)),
        }
    }
    if let Some((t0, k0, k1)) = open {
        out.push((t0 as u32, (n_tiles - t0) as u32, k0, k1 - k0));
    }
    out
}

impl Renderer {
    /// Differentiate the frame last [rendered][Self::render] against `dimg`
    /// (the whole frame's dL/dRGBA), accumulating into `grads`. Returns the
    /// number of gradient records the pass produced: one per (tile, gaussian)
    /// instance.
    ///
    /// `ddepth` (`W*H`, `None` = no depth supervision) is dL/d(**accumulated**
    /// depth `sum_i z_i alpha_i T_i`), the unnormalized composite. What the
    /// forward reports is the EXPECTED depth, that quantity divided by the
    /// alpha output, so a caller supervising expected depth must also put the
    /// normalizer's share into `dimg`'s alpha channel -
    /// [`add_expected_depth_vjp`] is the one place that does the split, and
    /// callers should use it rather than repeat it.
    ///
    /// The pass reduces each instance over its tile's pixels first (a fixed
    /// slot grid, band by band of tiles so the grid stays bounded), and only
    /// then sorts - instances, not pixels - by gaussian. Sorting one record
    /// per (pixel, gaussian) was 70% of the backward's device time on a
    /// trained 300k-gaussian scene.
    ///
    /// `Err` only when the frame has more records than one binding holds.
    #[allow(clippy::too_many_arguments)]
    pub fn render_bwd(
        &mut self,
        gpu: &Gpu,
        s: &GpuSplats,
        cam: &Camera,
        o: &RenderOpts,
        dimg: &DeviceBuffer,
        ddepth: Option<&DeviceBuffer>,
        scr: &mut BwdScratch,
        grads: &SplatGrads,
    ) -> Result<usize, String> {
        let (n_isects, vals_in_b, tiles_x, tiles_y) =
            self.last.expect("render() must run before render_bwd()");
        if n_isects == 0 {
            return Ok(0);
        }
        let vals = if vals_in_b { &self.vals_b } else { &self.vals_a };
        let n_tiles = (tiles_x * tiles_y) as usize;
        let ranges = gpu.read(&self.ranges, 2 * n_tiles);
        let ranges: Vec<u32> = ranges.iter().map(|v| v.to_bits()).collect();
        let bands = plan_bands(&ranges, scr.band_instances(gpu));
        scr.reserve_records(gpu, n_isects)?;
        scr.reserve_slots(gpu, bands.iter().map(|b| b.3 as usize).max().unwrap_or(0));

        // Stage timings, on request. Splitting the submissions costs a few
        // extra syncs, which is the price of finding out which stage is the
        // expensive one.
        let prof = std::env::var_os("BRAIN_SPLAT_PROFILE").is_some();
        let flush = |gpu: &Gpu, steps: &mut Vec<gpu_core::Step>, what: &'static str| {
            if !prof {
                return;
            }
            let t = std::time::Instant::now();
            let taken = std::mem::take(steps);
            gpu.submit(&[], &taken);
            gpu.read(&self.proj, 1);
            eprintln!("    bwd {what}: {:.1} ms", 1e3 * t.elapsed().as_secs_f64());
        };

        let mut steps = Vec::new();
        for &(tile0, tiles, k0, count) in &bands {
            if count == 0 {
                continue;
            }
            let threads = tiles * TILE * TILE;
            steps.push(gpu.step(
                self.ks.splat_bwd_slots,
                &[&self.proj, &s.colors, vals, &self.ranges, dimg, ddepth.unwrap_or(&scr.no_depth), &scr.slots],
                &[cam.width, cam.height, tiles_x, tile0, k0, k0 + count, threads, f(o.bg[0]), f(o.bg[1]), f(o.bg[2])],
                threads,
            ));
            steps.push(gpu.dispatch(
                self.ks.splat_bwd_tile_reduce,
                &[&scr.slots, vals, &scr.recs],
                &[count, k0],
                gpu_core::Dispatch::Workgroups(count),
            ));
        }
        flush(gpu, &mut steps, "slots + tile reduce");
        steps.push(gpu.step(
            self.ks.splat_bwd_keys,
            &[&scr.recs, &scr.rkeys_a, &scr.rvals_a],
            &[n_isects as u32],
            n_isects as u32,
        ));
        let key_bits = 32u32.min((s.n.next_power_of_two().trailing_zeros()).max(1) + 1);
        let in_b = record_sort_pairs(
            gpu, &self.ks, &scr.rkeys_a, &scr.rvals_a, &scr.rkeys_b, &scr.rvals_b,
            n_isects, key_bits, &scr.rsort, &mut steps,
        );
        flush(gpu, &mut steps, "sort instances by gaussian");
        let (skeys, svals) = if in_b { (&scr.rkeys_b, &scr.rvals_b) } else { (&scr.rkeys_a, &scr.rvals_a) };
        // segment ranges over gaussian ids (tile_ranges with depth_bits = 0)
        steps.push(gpu.step(
            self.ks.splat_tile_ranges,
            &[skeys, &scr.granges],
            &[n_isects as u32, 0],
            n_isects as u32,
        ));
        steps.push(gpu.step(
            self.ks.splat_grad_reduce,
            &[&scr.recs, svals, &scr.granges, &scr.pgrad, &grads.d_colors, &grads.d_absgrad, &grads.d_sumgrad],
            &[s.n as u32],
            s.n as u32,
        ));
        steps.push(gpu.step(
            self.ks.splat_project_bwd,
            &[&s.means, &s.quats, &s.scales, &self.proj, &scr.pgrad, &grads.d_gauss, &grads.d_opac],
            &project_params(s.n, cam, o),
            s.n as u32,
        ));
        let t = std::time::Instant::now();
        gpu.submit(&[&scr.granges, &scr.pgrad], &steps);
        if prof {
            gpu.read(&self.proj, 1);
            eprintln!("    bwd {} instances in {} band(s); ranges + reduce + project: {:.1} ms", n_isects, bands.len(), 1e3 * t.elapsed().as_secs_f64());
        }
        Ok(n_isects)
    }
}

/// Sort a host scene front-to-back for the given camera (the naive kernel
/// composites in buffer order). Returns a reordered copy.
/// Drop the alpha channel: interleaved RGBA f32 `[N*4]` -> interleaved RGB f32
/// `[N*3]`. The one implementation both the CLI's PPM writer and `caps::render`
/// use - [`Renderer::read_rgba`]'s natural output is RGBA, but every consumer
/// of a *rendered image* (a wire-format blob, an 8-bit quantizer) wants RGB.
pub fn rgba_to_rgb(rgba: &[f32]) -> Vec<f32> {
    rgba.chunks_exact(4).flat_map(|px| [px[0], px[1], px[2]]).collect()
}

pub fn sorted_by_depth(s: &Splats, cam: &Camera) -> Splats {
    let v = cam.viewmat();
    let mut order: Vec<usize> = (0..s.len()).collect();
    let depth = |i: usize| {
        let m = &s.means[i * 3..i * 3 + 3];
        v[8] * m[0] + v[9] * m[1] + v[10] * m[2] + v[11]
    };
    order.sort_by(|&a, &b| depth(a).total_cmp(&depth(b)));
    let mut out = Splats::default();
    for &i in &order {
        out.means.extend_from_slice(&s.means[i * 3..i * 3 + 3]);
        out.quats.extend_from_slice(&s.quats[i * 4..i * 4 + 4]);
        out.scales.extend_from_slice(&s.scales[i * 3..i * 3 + 3]);
        out.opacities.push(s.opacities[i]);
        out.colors.extend_from_slice(&s.colors[i * 3..i * 3 + 3]);
    }
    out
}
