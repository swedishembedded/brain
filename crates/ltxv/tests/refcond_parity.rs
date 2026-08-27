// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! IC-LoRA reference-video conditioning parity against
//! `tools/goldens/ltxv_refcond_dump_reference.py`, which is a LIVE run of the
//! official `ltx_core.conditioning.types.reference_video_cond.
//! VideoConditionByReferenceLatent`, its attention-strength wrapper, and
//! `ltx_pipelines.iclora_utils.downsample_mask_video_to_latent` - not a
//! transcription of them.
//!
//! This is pure geometry (token layout, RoPE position bounds, denoise/keyframe
//! markers, attention cross-mask), so it needs NO checkpoint and runs on any
//! machine. That matters here: the 22B DiT this conditioning eventually feeds is
//! a separately tracked real-weight gap, and this contract is exactly the part
//! that can be pinned down without it.
//!
//! Every gate asserts **cosine AND rel_l2**. Cosine alone is scale-invariant and
//! would pass a port that returned `k * golden`; the reference-position path
//! multiplies by `downscale_factor` and divides by `fps / S`, so a wrong scale
//! is precisely the defect most likely to occur here and precisely the one
//! cosine cannot see.

use ltxv::refcond::{
    append_reference_video_conditioning, dense_attention_mask, downsample_mask_to_latent,
    reference_pixel_extent, reference_video_positions,
};

// ------------------------------------------------------------------ metrics

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "cosine: length mismatch ({} vs {})", a.len(), b.len());
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        d += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    let den = na.sqrt() * nb.sqrt();
    if den <= 0.0 { 0.0 } else { d / den }
}

/// `||got - want||_2 / ||want||_2` - the scale-SENSITIVE half of every gate
/// here. When `want` is all zeros (a legitimate case: the keyframes mask over a
/// reference block is exactly zero), it degenerates to the absolute norm of the
/// error, which is the right answer for that case too.
fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "rel_l2: length mismatch ({} vs {})", got.len(), want.len());
    let (mut e, mut w) = (0.0f64, 0.0f64);
    for (x, y) in got.iter().zip(want) {
        e += (*x as f64 - *y as f64).powi(2);
        w += (*y as f64).powi(2);
    }
    if w <= 0.0 { e.sqrt() } else { (e / w).sqrt() }
}

/// Both halves asserted. `MIN_COS`/`MAX_REL` are tight because this is
/// deterministic geometry in f32 against f32 - the only spread is a handful of
/// ULPs from the reference doing the same divisions in a different order.
const MIN_COS: f64 = 0.999999;
const MAX_REL: f64 = 1e-5;

fn report(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: {} values vs {}", got.len(), want.len());
    let (c, r) = (cosine(got, want), rel_l2(got, want));
    eprintln!("{label}: cosine={c:.9} rel_l2={r:.3e} n={}", got.len());
    assert!(c >= MIN_COS, "{label}: cosine {c:.9} < {MIN_COS}");
    assert!(r <= MAX_REL, "{label}: rel_l2 {r:.3e} > {MAX_REL:.0e} - cosine can be 1.0 while the MAGNITUDE is wrong, which is exactly what this bound is for");
}

// ------------------------------------------------------------------ fixture

struct Fixture {
    t: Vec<checkpoint::safetensors::StTensor>,
}

impl Fixture {
    fn open(name: &str) -> Option<Fixture> {
        let p = brain_testutil::testdata(&format!("golden/ltxv/refcond/{name}.safetensors"));
        if !std::path::Path::new(&p).exists() {
            brain_testutil::skip(&format!("fixture {p} absent - run tools/goldens/ltxv_refcond_dump_reference.py"));
            return None;
        }
        Some(Fixture { t: checkpoint::safetensors::read(&p).expect("read golden") })
    }
    fn get(&self, name: &str) -> &[f32] {
        &self.t.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("no golden {name}")).data
    }
    fn shape(&self, name: &str) -> &[usize] {
        &self.t.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("no golden {name}")).shape
    }
}

/// name, target `(f,h,w)`, reference `(f,h,w)`, fps, downscale, temporal
/// scale, strength - the dumper's own case tuple.
type Case = (&'static str, (usize, usize, usize), (usize, usize, usize), f64, usize, usize, f32);

/// The dumper's own `CASES`, verbatim.
const CASES: &[Case] = &[
    ("plain", (3, 4, 6), (3, 4, 6), 24.0, 1, 1, 1.0),
    ("half_res_ref", (3, 8, 8), (3, 4, 4), 24.0, 2, 1, 1.0),
    ("temporal_x4", (9, 4, 4), (3, 4, 4), 24.0, 1, 4, 1.0),
    ("both_scales", (9, 8, 8), (3, 4, 4), 30.0, 2, 4, 1.0),
    ("partial_strength", (3, 4, 6), (3, 4, 6), 24.0, 1, 1, 0.7),
    ("scalar_attn", (3, 4, 6), (3, 4, 6), 24.0, 1, 1, 1.0),
    ("spatial_attn", (3, 4, 6), (3, 4, 6), 24.0, 1, 1, 1.0),
];

const CHANNELS: usize = 8;

/// Reference positions match the official item's own, across both scale knobs.
#[test]
fn ltxv_refcond_positions_match_reference() {
    let Some(fx) = Fixture::open("refcond") else { return };
    for &(name, _target, (rf, rh, rw), fps, down, temporal, _s) in CASES {
        let got = reference_video_positions(rf, rh, rw, fps, down, temporal);
        let want = fx.get(&format!("{name}.positions"));
        assert_eq!(fx.shape(&format!("{name}.positions")), [3, rf * rh * rw, 2], "{name}: golden position shape");
        report(&format!("{name}.positions"), &got, want);
    }
}

/// The whole appended block - positions, both masks, and the clean tail.
#[test]
fn ltxv_refcond_appended_block_matches_reference() {
    let Some(fx) = Fixture::open("refcond") else { return };
    for &(name, (tf, th, tw), (rf, rh, rw), fps, down, temporal, strength) in CASES {
        let (base_t, m) = (tf * th * tw, rf * rh * rw);
        // The base state the reference built with `create_initial_state`: the
        // target's own grid positions, and the causal first-latent-frame
        // keyframe marker `VideoLatentTools` sets unconditionally.
        let base_positions = ltxv::pipeline::real_pixel_positions(tf, th, tw, fps);
        let mut base_km = vec![0f32; base_t];
        base_km[..th * tw].fill(1.0);

        let ref_tokens = fx.get(&format!("{name}.clean_tail")).to_vec();
        assert_eq!(ref_tokens.len(), m * CHANNELS, "{name}: golden clean tail size");

        let out = append_reference_video_conditioning(
            base_t, &base_positions, &base_km, CHANNELS, fps, (rf, rh, rw), &ref_tokens, down, temporal, strength, None,
        );

        report(&format!("{name}.denoise_mask"), &out.denoise_mask, fx.get(&format!("{name}.denoise_mask")));
        report(&format!("{name}.clean_tail"), &out.clean[base_t * CHANNELS..], &ref_tokens);
        // Appended positions, sliced back out of the concatenated block.
        let total = base_t + m;
        let mut got_tail = Vec::with_capacity(3 * m * 2);
        for axis in 0..3 {
            got_tail.extend_from_slice(&out.positions[axis * total * 2 + base_t * 2..(axis + 1) * total * 2]);
        }
        report(&format!("{name}.appended_positions"), &got_tail, fx.get(&format!("{name}.positions")));

        // Structural invariants the reference's own docstrings state.
        assert_eq!(&out.keyframes_mask[..base_t], &base_km[..], "{name}: base keyframes mask must survive verbatim");
        assert!(out.keyframes_mask[base_t..].iter().all(|&v| v == 0.0), "{name}: reference tokens are never keyframe-marked");
        assert!(out.denoise_mask[..base_t].iter().all(|&v| v == 1.0), "{name}: base tokens stay fully denoised");
        assert!(out.clean[..base_t * CHANNELS].iter().all(|&v| v == 0.0), "{name}: base clean range is never written");
    }
}

/// The keyframes mask the reference emits, where it emitted one.
#[test]
fn ltxv_refcond_keyframes_mask_matches_reference() {
    let Some(fx) = Fixture::open("refcond") else { return };
    for &(name, (tf, th, tw), (rf, rh, rw), fps, down, temporal, strength) in CASES {
        let key = format!("{name}.keyframes_mask");
        if !fx.t.iter().any(|t| t.name == key) {
            continue;
        }
        let base_t = tf * th * tw;
        let base_positions = ltxv::pipeline::real_pixel_positions(tf, th, tw, fps);
        let mut base_km = vec![0f32; base_t];
        base_km[..th * tw].fill(1.0);
        let ref_tokens = fx.get(&format!("{name}.clean_tail")).to_vec();
        let out = append_reference_video_conditioning(
            base_t, &base_positions, &base_km, CHANNELS, fps, (rf, rh, rw), &ref_tokens, down, temporal, strength, None,
        );
        report(&key, &out.keyframes_mask, fx.get(&key));
    }
}

/// The factored `[M]` cross-mask reconstructs the reference's DENSE
/// `(N+M)^2` attention matrix exactly. This is the claim that licenses this
/// port to store `M` numbers instead of `(N+M)^2` - at a real 1280x704x121
/// clip that is the difference between kilobytes and hundreds of gigabytes.
#[test]
fn ltxv_refcond_factored_cross_mask_rebuilds_the_dense_reference_matrix() {
    let Some(fx) = Fixture::open("refcond") else { return };
    let (base_t, m) = (3 * 4 * 6, 3 * 4 * 6);
    let cross = fx.get("scalar_attn.cross_mask");
    assert_eq!(cross.len(), m, "golden cross mask length");
    let got = dense_attention_mask(base_t, cross);
    let want = fx.get("scalar_attn.dense_attention");
    assert_eq!(fx.shape("scalar_attn.dense_attention"), [base_t + m, base_t + m], "golden dense shape");
    report("scalar_attn.dense_attention", &got, want);
}

/// A per-region (spatial) cross-mask survives the append unchanged - this is
/// the dial that says WHICH part of the frame the reference governs.
#[test]
fn ltxv_refcond_spatial_cross_mask_is_carried_per_token() {
    let Some(fx) = Fixture::open("refcond") else { return };
    let (tf, th, tw) = (3usize, 4usize, 6usize);
    let (base_t, m) = (tf * th * tw, tf * th * tw);
    let cross = fx.get("spatial_attn.cross_mask");
    assert_eq!(cross.len(), m, "golden spatial cross mask length");
    let base_positions = ltxv::pipeline::real_pixel_positions(tf, th, tw, 24.0);
    let mut base_km = vec![0f32; base_t];
    base_km[..th * tw].fill(1.0);
    let ref_tokens = fx.get("spatial_attn.clean_tail").to_vec();
    let out = append_reference_video_conditioning(
        base_t, &base_positions, &base_km, CHANNELS, 24.0, (tf, th, tw), &ref_tokens, 1, 1, 1.0, Some(cross),
    );
    report("spatial_attn.cross_mask", out.cross_mask.as_deref().expect("cross mask carried"), cross);
}

/// `downsample_mask_video_to_latent`: area-average spatially, causal split
/// temporally. This is how a pixel-space mask video becomes the per-token
/// weights the gate above carries.
#[test]
fn ltxv_refcond_mask_downsample_matches_reference() {
    let Some(fx) = Fixture::open("mask_latent") else { return };
    // The dumper's own `MASK_CASES`, verbatim.
    let cases: &[(&str, usize, usize, usize, usize, usize, usize)] = &[
        ("m_basic", 9, 32, 32, 2, 4, 4),
        ("m_single_frame", 1, 16, 16, 1, 2, 2),
        ("m_deep", 17, 64, 32, 3, 8, 4),
        // Ratios that do NOT divide evenly. The area average is torch's
        // ADAPTIVE rule, which collapses to a plain box pool whenever the
        // ratio divides - and every ratio the VAE produces does, so the three
        // cases above cannot tell the two rules apart. Found by mutation: a
        // cell end that floors instead of ceiling survived this gate until
        // these two were added.
        ("m_ragged", 9, 50, 30, 2, 4, 4),
        ("m_ragged_deep", 17, 45, 45, 3, 8, 8),
    ];
    for &(name, fp, hp, wp, lf, lh, lw) in cases {
        let mask = fx.get(&format!("{name}.mask"));
        assert_eq!(mask.len(), fp * hp * wp, "{name}: golden mask size");
        let got = downsample_mask_to_latent(mask, fp, hp, wp, lf, lh, lw);
        report(&format!("{name}.latent_mask"), &got, fx.get(&format!("{name}.latent_mask")));
    }
}

/// The pixel extent a caller has to supply a mask video at, from the VAE's own
/// causal `8x` temporal and `32x32` spatial factors.
#[test]
fn ltxv_refcond_pixel_extent_follows_the_causal_vae() {
    assert_eq!(reference_pixel_extent((1, 4, 4)), (1, 128, 128), "a single latent frame covers ONE pixel frame (causal VAE)");
    assert_eq!(reference_pixel_extent((2, 4, 4)), (9, 128, 128), "each later latent frame covers 8");
    assert_eq!(reference_pixel_extent((3, 8, 4)), (17, 256, 128));
}
