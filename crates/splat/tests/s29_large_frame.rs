// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A frame with more (tile, gaussian) instances than a byte-addressed binding
//! holds is still differentiated where the device can address more.
//!
//! The gradient records are an `array<f32>` of `instances * 17` words, indexed
//! by `u32` elements, so the limit is 2^32 WORDS (16 GiB), not the 4 GiB a
//! `u32` BYTE offset reaches. A 251-photograph truck capture crashed at 69.5M
//! instances (4.4 GiB of records) on a 96 GB card because the scratch used
//! the byte figure.
//!
//! 1. `splat_grad_reduce` sums records whose word index is past 2^30 (byte
//!    offset past 4 GiB) exactly.
//! 2. A whole ray-rendered frame past the old record ceiling renders and
//!    differentiates: finite gradients, the front gaussians' non-zero.
//!
//! Swedish Embedded AB implements GPU training engines that stay correct at
//! the scale of real captures. If your team needs expertise in gaussian
//! splatting at scale then you can procure our services by sending an email to
//! info@swedishembedded.com.

use gpu_core::Gpu;
use splat::renderer::{max_records_for_binding, BwdScratch, GpuSplats, Renderer, SplatGrads, RAY_RECORD_WORDS};
use splat::types::{Camera, RenderOpts, Splats};
use splat::Kernels;

/// Records the old byte-addressed ceiling held, plus a margin.
fn past_the_byte_ceiling(g: &Gpu) -> usize {
    max_records_for_binding(g.max_storage_binding_bytes(), RAY_RECORD_WORDS) * 11 / 10
}

fn can_address(g: &Gpu, records: usize) -> bool {
    g.max_word_binding_bytes() >= (records * RAY_RECORD_WORDS * 4) as u64
}

#[test]
fn the_gradient_reduction_reads_records_past_four_gib() {
    let g = Gpu::new(splat::PIPELINES);
    let records = past_the_byte_ceiling(&g);
    if !can_address(&g, records) {
        return brain_testutil::skip_unavailable("this device cannot bind a buffer larger than its byte ceiling");
    }
    let ks = Kernels::at(0);
    let stride = RAY_RECORD_WORDS;
    // record j holds (j % 7 + 1) in every channel
    let recs = g.storage((records * stride) as u64);
    let chunk_records = 1 << 22;
    let mut j = 0;
    while j < records {
        let n = chunk_records.min(records - j);
        let data: Vec<f32> = (j..j + n).flat_map(|r| std::iter::repeat_n((r % 7 + 1) as f32, stride)).collect();
        g.write_f32_at(&recs, (j * stride) as u64, &data);
        j += n;
    }
    // 1000 gaussians owning equal runs, the last owning the remainder
    let n_gauss = 1000usize;
    let per = records / n_gauss;
    let offsets: Vec<u32> = (0..n_gauss).map(|i| (i * per) as u32).collect();
    let off = g.storage(n_gauss as u64);
    g.write(&off, &offsets);
    let pgrad = g.storage((13 * n_gauss) as u64);
    let (d_colors, absgrad, sumgrad) = (g.storage(3 * n_gauss as u64), g.storage(n_gauss as u64), g.storage(2 * n_gauss as u64));
    g.submit(&[&pgrad, &d_colors, &absgrad, &sumgrad], &[]);
    let step = g.step(
        ks.splat_grad_reduce,
        &[&recs, &off, &pgrad, &d_colors, &absgrad, &sumgrad],
        &[n_gauss as u32, stride as u32, 13, 13, 16, records as u32],
        n_gauss as u32,
    );
    g.submit(&[], &[step]);
    let out = g.read(&pgrad, 13 * n_gauss);
    for gi in [0, n_gauss / 2, n_gauss - 1] {
        let (lo, hi) = (gi * per, if gi + 1 == n_gauss { records } else { (gi + 1) * per });
        let want: u64 = (lo..hi).map(|r| (r % 7 + 1) as u64).sum();
        // fp32 sums of small integers are exact up to 2^24 per partial; allow
        // the rounding of the long serial sum
        let got = out[13 * gi] as f64;
        assert!((got - want as f64).abs() <= want as f64 * 1e-4, "gaussian {gi}: {got} vs {want}");
    }
}

#[test]
fn a_frame_past_the_old_record_ceiling_is_differentiated() {
    let g = Gpu::new(splat::PIPELINES);
    let want = past_the_byte_ceiling(&g);
    if !can_address(&g, want) {
        return brain_testutil::skip_unavailable("this device cannot bind a buffer larger than its byte ceiling");
    }
    let (w, h) = (1920u32, 1088u32);
    let tiles = (w.div_ceil(16) * h.div_ceil(16)) as usize;
    // every gaussian is far larger than the frame, so it touches every tile
    let n = want / tiles + 1;
    let mut s = Splats::default();
    for i in 0..n {
        let z = 3.0 + i as f32 * 1e-4;
        s.means.extend_from_slice(&[0.0, 0.0, z]);
        s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        s.scales.extend_from_slice(&[8.0; 3]);
        s.opacities.push(0.05);
        s.colors.extend_from_slice(&[0.2 + 0.6 * (i % 5) as f32 / 4.0, 0.5, 0.3]);
    }
    let cam = Camera::look_at([0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 60.0, w, h);
    let o = RenderOpts { ray: true, ..Default::default() };
    let mut ren = Renderer::new(&g, Kernels::at(0), n, w, h, 0).growable();
    let gs = GpuSplats::upload(&g, &s);
    ren.render(&g, &gs, &cam, &o);
    let grads = SplatGrads::new(&g, n);
    g.submit(&[&grads.d_gauss, &grads.d_opac, &grads.d_colors, &grads.d_absgrad, &grads.d_sumgrad], &[]);
    let dimg = g.storage_init("dimg", &vec![1e-3; (w * h * 4) as usize]);
    let mut scr = BwdScratch::new(&g, n, (w * h) as usize, 0);
    let records = ren.render_bwd_ray(&g, &gs, &cam, &o, &dimg, None, &mut scr, &grads, false).expect("a frame this dense differentiates");
    assert!(records.is_none());
    for (name, buf, len) in [("d_gauss", &grads.d_gauss, 10 * n), ("d_opac", &grads.d_opac, n), ("d_colors", &grads.d_colors, 3 * n)] {
        let v = g.read(buf, len);
        assert!(v.iter().all(|x| x.is_finite()), "{name} has a non-finite gradient");
    }
    let front = g.read(&grads.d_opac, 64);
    assert!(front.iter().any(|&x| x != 0.0), "the front gaussians received no gradient");
}
