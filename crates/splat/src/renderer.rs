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
        }
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
        record_scan(gpu, &self.ks, &self.counts, s.n, &self.count_scan, &mut steps);
        gpu.submit(&[], &steps);
        let total = gpu.read(self.count_scan.total(), 1)[0].to_bits() as usize;
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
            &[&self.proj, &s.colors, vals, &self.ranges, &self.img],
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

/// Per-gaussian gradient buffers the backward accumulates into (caller
/// clears/consumes them; layouts match splat_project_bwd/splat_grad_reduce).
pub struct SplatGrads {
    pub d_gauss: DeviceBuffer,  // N*10: d_means(3), d_scales(3), d_quats(4)
    pub d_opac: DeviceBuffer,   // N
    pub d_colors: DeviceBuffer, // N*3
}

impl SplatGrads {
    pub fn new(gpu: &Gpu, n: usize) -> SplatGrads {
        SplatGrads {
            d_gauss: gpu.storage(10 * n as u64),
            d_opac: gpu.storage(n as u64),
            d_colors: gpu.storage(3 * n as u64),
        }
    }
}

/// Backward scratch (record buffer sized `rec_cap`, default 64·px).
pub struct BwdScratch {
    counts_px: DeviceBuffer,
    px_scan: ScanScratch,
    recs: DeviceBuffer,
    rkeys_a: DeviceBuffer,
    rvals_a: DeviceBuffer,
    rkeys_b: DeviceBuffer,
    rvals_b: DeviceBuffer,
    rsort: SortScratch,
    granges: DeviceBuffer,
    pgrad: DeviceBuffer,
    rec_cap: usize,
}

impl BwdScratch {
    pub fn new(gpu: &Gpu, max_n: usize, max_px: usize, rec_cap: usize) -> BwdScratch {
        let cap = if rec_cap == 0 { (64 * max_px).clamp(1 << 20, 64 << 20) } else { rec_cap };
        BwdScratch {
            counts_px: gpu.storage(max_px as u64),
            px_scan: ScanScratch::new(gpu, max_px),
            recs: gpu.storage(10 * cap as u64),
            rkeys_a: gpu.storage(cap as u64),
            rvals_a: gpu.storage(cap as u64),
            rkeys_b: gpu.storage(cap as u64),
            rvals_b: gpu.storage(cap as u64),
            rsort: SortScratch::new(gpu, cap),
            granges: gpu.storage(2 * max_n as u64),
            pgrad: gpu.storage(9 * max_n as u64),
            rec_cap: cap,
        }
    }
}

impl Renderer {
    /// Backward through the LAST `render()` call: upstream RGBA image grads
    /// `dimg` (`W*H*4`) → accumulate parameter grads. Returns the gradient
    /// record count (panics if the record capacity would overflow — raise
    /// `rec_cap` or shrink the fit image).
    #[allow(clippy::too_many_arguments)]
    pub fn render_bwd(
        &mut self,
        gpu: &Gpu,
        s: &GpuSplats,
        cam: &Camera,
        o: &RenderOpts,
        dimg: &DeviceBuffer,
        scr: &BwdScratch,
        grads: &SplatGrads,
    ) -> usize {
        let (n_isects, vals_in_b, tiles_x, tiles_y) =
            self.last.expect("render() must run before render_bwd()");
        let _ = n_isects;
        let vals = if vals_in_b { &self.vals_b } else { &self.vals_a };
        let px = (cam.width * cam.height) as usize;

        // pass A: per-pixel record counts -> offsets + total
        let mut steps = Vec::new();
        steps.push(gpu.step(
            self.ks.splat_bwd_count,
            &[&self.proj, vals, &self.ranges, &scr.counts_px],
            &[cam.width, cam.height, tiles_x, tiles_y],
            px as u32,
        ));
        record_scan(gpu, &self.ks, &scr.counts_px, px, &scr.px_scan, &mut steps);
        gpu.submit(&[], &steps);
        let n_recs = gpu.read(scr.px_scan.total(), 1)[0].to_bits() as usize;
        assert!(
            n_recs <= scr.rec_cap,
            "gradient records {n_recs} exceed capacity {} — raise rec_cap",
            scr.rec_cap
        );
        if n_recs == 0 {
            return 0;
        }

        // pass B: emit records, sort by gaussian id, segment-reduce, project VJP
        let mut steps = Vec::new();
        steps.push(gpu.step(
            self.ks.splat_bwd_emit,
            &[&self.proj, &s.colors, vals, &self.ranges, dimg, &scr.counts_px, &scr.recs],
            &[
                cam.width, cam.height, tiles_x, tiles_y,
                f(o.bg[0]), f(o.bg[1]), f(o.bg[2]), 0,
            ],
            px as u32,
        ));
        steps.push(gpu.step(
            self.ks.splat_bwd_keys,
            &[&scr.recs, &scr.rkeys_a, &scr.rvals_a],
            &[n_recs as u32],
            n_recs as u32,
        ));
        let key_bits = 32u32.min((s.n.next_power_of_two().trailing_zeros()).max(1) + 1);
        let in_b = record_sort_pairs(
            gpu, &self.ks, &scr.rkeys_a, &scr.rvals_a, &scr.rkeys_b, &scr.rvals_b,
            n_recs, key_bits, &scr.rsort, &mut steps,
        );
        let (skeys, svals) = if in_b { (&scr.rkeys_b, &scr.rvals_b) } else { (&scr.rkeys_a, &scr.rvals_a) };
        // segment ranges over gaussian ids (tile_ranges with depth_bits = 0)
        steps.push(gpu.step(
            self.ks.splat_tile_ranges,
            &[skeys, &scr.granges],
            &[n_recs as u32, 0],
            n_recs as u32,
        ));
        steps.push(gpu.step(
            self.ks.splat_grad_reduce,
            &[&scr.recs, svals, &scr.granges, &scr.pgrad, &grads.d_colors],
            &[s.n as u32],
            s.n as u32,
        ));
        let v = cam.viewmat();
        let mut pp = [0u32; 24];
        pp[0] = s.n as u32;
        pp[1] = cam.width;
        pp[2] = cam.height;
        pp[3] = 0;
        pp[4] = f(cam.fx);
        pp[5] = f(cam.fy);
        pp[6] = f(cam.cx);
        pp[7] = f(cam.cy);
        pp[8] = f(o.near);
        pp[9] = f(o.far);
        pp[10] = f(o.eps2d);
        pp[11] = 0;
        for (i, val) in v.iter().enumerate() {
            pp[12 + i] = f(*val);
        }
        steps.push(gpu.step(
            self.ks.splat_project_bwd,
            &[&s.means, &s.quats, &s.scales, &self.proj, &scr.pgrad, &grads.d_gauss, &grads.d_opac],
            &pp,
            s.n as u32,
        ));
        gpu.submit(&[&scr.granges, &scr.pgrad], &steps);
        n_recs
    }
}

/// Sort a host scene front-to-back for the given camera (the naive kernel
/// composites in buffer order). Returns a reordered copy.
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
