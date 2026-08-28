// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Masked-conditioning parity against
//! `tools/goldens/ltxv_maskcond_dump_reference.py`, which is a LIVE run of the
//! official `ltx_core.conditioning.types.mask_cond.VideoConditionByMask` (and,
//! for the composed cases, `latent_cond.VideoConditionByLatentIndex` and
//! `components.noisers.GaussianNoiser`) - not a transcription of them.
//!
//! This is pure algebra over the latent state, so it needs NO checkpoint and
//! runs on any machine. That matters here for the same reason it does for
//! `refcond_parity.rs`: the 22B DiT this conditioning eventually feeds is a
//! separately tracked real-weight gap, and this contract is exactly the part
//! that can be pinned down without it.
//!
//! **Every gate asserts cosine AND rel_l2.** Cosine alone is scale-invariant
//! and blind to a uniform factor, which is precisely the defect shape this port
//! is exposed to: a dropped `clean * inv` term, a `1 - strength * m` instead of
//! `denoise * inv + (1 - strength) * m`, a mask applied once instead of twice.
//! The precedent is not hypothetical - on `refcond.rs` a dropped temporal
//! `clamp(min = 0)` scored cosine 0.999999884, ABOVE this file's own bound,
//! while rel_l2 4.8e-4 caught it. A cosine-only gate would have shipped it.

use ltxv::maskcond::{apply_video_condition_by_mask, noised_initial_latent, LatentMaskAccumulator, MaskSeqPolarity};
use ltxv::refcond::downsample_mask_to_latent;

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
    if den <= 0.0 { 1.0 } else { d / den }
}

/// `||got - want||_2 / ||want||_2` - the scale-SENSITIVE half of every gate
/// here. When `want` is all zeros (a legitimate case: the `all_one` case's
/// denoise mask is exactly zero) it degenerates to the absolute norm of the
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

/// Both halves asserted. Tight because this is deterministic f32 algebra
/// against f32 - the only spread is a handful of ULPs from the reference
/// evaluating the same two blends in a different association order.
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
    fn open() -> Option<Fixture> {
        let p = brain_testutil::testdata("golden/ltxv/maskcond/maskcond.safetensors");
        if !std::path::Path::new(&p).exists() {
            brain_testutil::skip(&format!("fixture {p} absent - run tools/goldens/ltxv_maskcond_dump_reference.py"));
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

/// name, latent `(f, h, w)`, strength, noise_scale - the dumper's own case
/// tuple, minus the fields the port does not take (the mask itself and the
/// base state are read out of the golden).
type Case = (&'static str, (usize, usize, usize), f32, f32);

/// The dumper's own `CASES`, verbatim.
const CASES: &[Case] = &[
    ("binary_half", (3, 4, 6), 1.0, 1.0),
    ("temporal_varying", (4, 4, 4), 1.0, 1.0),
    ("partial_strength", (3, 4, 6), 0.6, 1.0),
    ("fractional", (3, 4, 6), 1.0, 1.0),
    ("fractional_partial", (2, 4, 4), 0.35, 1.0),
    ("all_zero", (2, 4, 4), 1.0, 1.0),
    ("all_one", (2, 4, 4), 1.0, 1.0),
    ("over_existing", (3, 4, 6), 1.0, 1.0),
    ("after_latent_index", (3, 4, 6), 0.8, 1.0),
    ("partial_noise_scale", (2, 4, 4), 1.0, 0.7),
];

const CHANNELS: usize = 8;

/// The two tensors `VideoConditionByMask.apply_to` replaces, against the live
/// reference, on every case.
#[test]
fn ltxv_maskcond_clean_and_denoise_mask_match_reference() {
    let Some(fx) = Fixture::open() else { return };
    for &(name, (f, h, w), strength, _) in CASES {
        let n = f * h * w;
        let got = apply_video_condition_by_mask(
            fx.get(&format!("{name}.base_clean")),
            fx.get(&format!("{name}.base_denoise_mask")),
            fx.get(&format!("{name}.cond_tokens")),
            fx.get(&format!("{name}.mask_tokens")),
            CHANNELS,
            strength,
        );
        assert_eq!(fx.shape(&format!("{name}.clean")), [n, CHANNELS], "{name}: golden clean shape");
        report(&format!("{name}.clean"), &got.clean, fx.get(&format!("{name}.clean")));
        report(&format!("{name}.denoise_mask"), &got.denoise_mask, fx.get(&format!("{name}.denoise_mask")));
    }
}

/// The initial latent `GaussianNoiser` derives from that pair - the tensor the
/// denoise loop actually starts from, and where the mask's quadratic entry at
/// fractional values lives.
#[test]
fn ltxv_maskcond_initial_latent_matches_the_reference_noiser() {
    let Some(fx) = Fixture::open() else { return };
    for &(name, _, strength, noise_scale) in CASES {
        let c = apply_video_condition_by_mask(
            fx.get(&format!("{name}.base_clean")),
            fx.get(&format!("{name}.base_denoise_mask")),
            fx.get(&format!("{name}.cond_tokens")),
            fx.get(&format!("{name}.mask_tokens")),
            CHANNELS,
            strength,
        );
        let got = noised_initial_latent(
            fx.get(&format!("{name}.base_latent")),
            fx.get(&format!("{name}.noise")),
            &c.clean,
            &c.denoise_mask,
            CHANNELS,
            noise_scale,
        );
        report(&format!("{name}.initial_latent"), &got, fx.get(&format!("{name}.initial_latent")));
    }
}

/// A `strength = 1.0` conditioning position comes out of the noiser BIT-exactly
/// the conditioning latent. This is the whole claim that separates masked
/// conditioning from an IC-LoRA: preserved regions are identical, not merely
/// structurally similar.
#[test]
fn ltxv_maskcond_a_conditioned_region_is_bit_exact_not_merely_close() {
    let Some(fx) = Fixture::open() else { return };
    for &(name, _, strength, noise_scale) in CASES {
        if strength != 1.0 || noise_scale != 1.0 {
            continue;
        }
        let mask = fx.get(&format!("{name}.mask_tokens"));
        let cond = fx.get(&format!("{name}.cond_tokens"));
        let c = apply_video_condition_by_mask(fx.get(&format!("{name}.base_clean")), fx.get(&format!("{name}.base_denoise_mask")), cond, mask, CHANNELS, strength);
        let l = noised_initial_latent(fx.get(&format!("{name}.base_latent")), fx.get(&format!("{name}.noise")), &c.clean, &c.denoise_mask, CHANNELS, noise_scale);
        let mut pinned = 0usize;
        for (tok, &m) in mask.iter().enumerate() {
            if m != 1.0 {
                continue;
            }
            pinned += 1;
            assert_eq!(c.denoise_mask[tok], 0.0, "{name}: token {tok} is a strength-1.0 conditioning position and must be frozen");
            let off = tok * CHANNELS;
            assert_eq!(&l[off..off + CHANNELS], &cond[off..off + CHANNELS], "{name}: token {tok} must start BIT-exactly at the conditioning latent");
        }
        eprintln!("{name}: {pinned} of {} tokens pinned bit-exactly", mask.len());
    }
}

/// The `[N]` mask this port stores rebuilds the reference's `[N, C]` broadcast
/// EXACTLY. That is what licenses storing one weight per token instead of one
// perf-number: 128 is the checkpoint's own latent-channel count, not a measured runtime ratio
/// per token per channel - a 128x difference at the real checkpoint's latent
/// width, on the tensor a caller is most likely to hold several copies of.
#[test]
fn ltxv_maskcond_factored_mask_rebuilds_the_dense_per_channel_form() {
    let Some(fx) = Fixture::open() else { return };
    for &(name, (f, h, w), strength, _) in CASES {
        let n = f * h * w;
        let mask = fx.get(&format!("{name}.mask_tokens"));
        let (base_clean, cond) = (fx.get(&format!("{name}.base_clean")), fx.get(&format!("{name}.cond_tokens")));
        let got = apply_video_condition_by_mask(base_clean, fx.get(&format!("{name}.base_denoise_mask")), cond, mask, CHANNELS, strength);
        // The mask EXPANDED to one weight per token per channel - what torch
        // materialises when it broadcasts `[N, 1]` against `[N, C]` - and the
        // blend written against that expansion with no token axis left in it.
        let dense_mask: Vec<f32> = mask.iter().flat_map(|&m| std::iter::repeat_n(m, CHANNELS)).collect();
        assert_eq!(dense_mask.len(), n * CHANNELS, "{name}: expanded mask size");
        let dense: Vec<f32> = base_clean.iter().zip(cond).zip(&dense_mask).map(|((&b, &c), &m)| b * (1.0 - m) + c * m).collect();
        assert_eq!(got.clean, dense, "{name}: the factored [N] mask is not bit-identical to the dense [N, C] broadcast");
    }
}

/// The patchified mask IS the `[F, H, W]` latent mask flattened in C order, at
/// LTX-2.5's patch size. The golden dumper asserts this against the live
/// patchifier; this side asserts the port is entitled to rely on it, i.e. that
/// no permutation was quietly needed.
#[test]
fn ltxv_maskcond_mask_tokens_are_the_c_order_latent_mask() {
    let Some(fx) = Fixture::open() else { return };
    for &(name, (f, h, w), _, _) in CASES {
        let flat = fx.get(&format!("{name}.mask"));
        assert_eq!(fx.shape(&format!("{name}.mask")), [f, h, w], "{name}: golden mask shape");
        assert_eq!(flat, fx.get(&format!("{name}.mask_tokens")), "{name}: token order is not the C-order flatten");
    }
}

// ------------------------------------------- pixel space -> latent space

/// The streaming reduction is BIT-identical to the whole-buffer one it exists
/// to avoid allocating. Nothing is traded for holding one frame at a time.
#[test]
fn ltxv_maskcond_streaming_downsample_is_bit_identical_to_the_whole_buffer_form() {
    // `(f_pix, h_pix, w_pix, lat_f, lat_h, lat_w)`, including the causal VAE's
    // own ratios (8x temporal with a lone first frame, 32x spatial).
    // The last two ratios do NOT divide evenly. That matters: the area
    // average is torch's ADAPTIVE rule (cell `i` spans `[floor(i*H/O),
    // ceil((i+1)*H/O))`, so neighbouring cells overlap), and it degenerates to
    // a plain box pool exactly when the ratio divides - which every ratio the
    // VAE itself produces does. Testing only those hides a cell-boundary
    // defect entirely; a mutation that floors the cell end instead of ceiling
    // it survived this gate until these two were added.
    let cases: &[(usize, usize, usize, usize, usize, usize)] = &[
        (9, 64, 64, 2, 2, 2),
        (1, 32, 32, 1, 1, 1),
        (17, 96, 64, 3, 3, 2),
        (25, 128, 128, 4, 4, 4),
        (9, 50, 30, 2, 4, 4),
        (17, 45, 45, 3, 8, 8),
    ];
    let mut rng = data::rng::Rng::new(0x9e37_79b9);
    for &(fp, hp, wp, lf, lh, lw) in cases {
        let mask: Vec<f32> = (0..fp * hp * wp).map(|_| rng.next_f32()).collect();
        let want = downsample_mask_to_latent(&mask, fp, hp, wp, lf, lh, lw);
        let mut acc = LatentMaskAccumulator::new(fp, hp, wp, lf, lh, lw);
        for f in 0..fp {
            acc.push_frame(&mask[f * hp * wp..(f + 1) * hp * wp]);
        }
        let got = acc.finish();
        assert_eq!(got, want, "streaming reduction differs at {fp}x{hp}x{wp} -> {lf}x{lh}x{lw}");
        report(&format!("stream_{fp}x{hp}x{wp}"), &got, &want);
    }
}

/// The temporal rule is a MEAN over each latent frame's run, with the causal
/// first frame taken alone - not `any`, not `all`, not `max`. Those three are
/// all plausible and all wrong, and they agree with the mean on a hard mask
/// that does not move, which is why this checks a mask that DOES move: a
/// subject present for part of a latent frame's 8-frame run must land strictly
/// between 0 and 1, and at the exact fraction of frames it was present for.
#[test]
fn ltxv_maskcond_temporal_reduction_is_a_mean_not_any_all_or_max() {
    let (fp, hp, wp) = (17usize, 32usize, 32usize);
    let (lf, lh, lw) = (3usize, 1usize, 1usize);
    let mut mask = vec![0f32; fp * hp * wp];
    // Pixel frames 1..=8 are latent frame 1's run; mark 2 of the 8.
    // Pixel frames 9..=16 are latent frame 2's run; mark 6 of the 8.
    for f in [1usize, 2, 9, 10, 11, 12, 13, 14] {
        mask[f * hp * wp..(f + 1) * hp * wp].fill(1.0);
    }
    let got = downsample_mask_to_latent(&mask, fp, hp, wp, lf, lh, lw);
    let want = [0.0f32, 2.0 / 8.0, 6.0 / 8.0];
    report("temporal_mean", &got, &want);
    assert!(got[1] > 0.0 && got[1] < 1.0, "an `any`/`all`/`max` reduction would have produced 1 or 0 here, got {}", got[1]);
}

/// Inverting the polarity is an affine map, so it commutes with the area
/// average. Asserting that is what lets the flip live at the PIXEL, where the
/// manifest defines it, instead of being smuggled in after the reduction where
/// a reader would not find it.
#[test]
fn ltxv_maskcond_polarity_flip_commutes_with_the_reduction() {
    let (fp, hp, wp, lf, lh, lw) = (9usize, 64usize, 64usize, 2usize, 2usize, 2usize);
    let mut rng = data::rng::Rng::new(0x5f3a_1c2d);
    let object: Vec<f32> = (0..fp * hp * wp).map(|_| rng.next_f32()).collect();
    let flipped: Vec<f32> = object.iter().map(|&v| MaskSeqPolarity::ObjectWhite.to_conditioning(v)).collect();
    let first = downsample_mask_to_latent(&flipped, fp, hp, wp, lf, lh, lw);
    let after: Vec<f32> = downsample_mask_to_latent(&object, fp, hp, wp, lf, lh, lw).iter().map(|&v| 1.0 - v).collect();
    report("polarity_commutes", &first, &after);
}

// ------------------------------------------- the sam2-maskseq contract

/// A `brain/sam2-maskseq/1` directory end to end: both polarities land the
/// TRACKED OBJECT on the generated side, an occluded frame conditions fully
/// rather than regenerating everything, and a frame-count mismatch is refused.
///
/// This is the gate for the failure the format exists to prevent. Reading the
/// polarity backwards preserves the character and regenerates the whole set -
/// visually plausible, completely wrong, and silent.
#[test]
fn ltxv_maskcond_a_sam2_mask_sequence_reaches_the_reference_polarity() {
    let dir = std::env::temp_dir().join(format!("ltxv_maskseq_{}", std::process::id()));
    let (w, h, frames) = (64usize, 64usize, 9usize);
    let (lf, lh, lw) = (2usize, 2usize, 2usize);

    // The object occupies the LEFT half of every frame; frame 5 is occluded.
    let write = |polarity: &str, occlude: bool| {
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        let object_value: u8 = if polarity == "object=255" { 255 } else { 0 };
        let other: u8 = 255 - object_value;
        for f in 0..frames {
            let mut img = image::RgbImage::new(w as u32, h as u32);
            for y in 0..h {
                for x in 0..w {
                    let v = if x < w / 2 { object_value } else { other };
                    img.put_pixel(x as u32, y as u32, image::Rgb([v, v, v]));
                }
            }
            img.save(dir.join(format!("mask_{f:06}.png"))).expect("write png");
        }
        let per_frame: Vec<String> = (0..frames)
            .map(|f| format!("{{\"frame\":{f},\"object_score\":{},\"iou\":0.9,\"area_px\":1}}", if occlude && f == 5 { "-0.5" } else { "1.5" }))
            .collect();
        std::fs::write(
            dir.join("masks.json"),
            format!(
                "{{\"format\":\"brain/sam2-maskseq/1\",\"pattern\":\"mask_%06d.png\",\"frames\":{frames},\"width\":{w},\"height\":{h},\"fps\":24.0,\"polarity\":\"{polarity}\",\"binary\":true,\"threshold\":0.5,\"object_id\":0,\"per_frame\":[{}]}}",
                per_frame.join(",")
            ),
        )
        .expect("write manifest");
    };

    for polarity in ["object=255", "object=0"] {
        write(polarity, false);
        let m = ltxv::maskcond::read_mask_sequence(&dir, frames, w, h, lf, lh, lw).expect("read mask sequence");
        assert_eq!(m.len(), lf * lh * lw);
        // Left latent column = the tracked object = GENERATED = 0. Right
        // column = background = CONDITIONING = 1. Both polarities, same answer.
        for f in 0..lf {
            assert_eq!(m[f * lh * lw..(f + 1) * lh * lw], [0.0, 1.0, 0.0, 1.0], "{polarity}: the tracked object must land on the GENERATED side");
        }
    }

    // An occluded frame conditions fully, so its latent frame is pulled toward
    // 1 rather than left as an empty (= regenerate-everything) mask.
    write("object=255", true);
    let m = ltxv::maskcond::read_mask_sequence(&dir, frames, w, h, lf, lh, lw).expect("read mask sequence");
    assert_eq!(&m[..lh * lw], &[0.0, 1.0, 0.0, 1.0], "latent frame 0 covers pixel frame 0 alone and is not occluded");
    // Latent frame 1 averages pixel frames 1..=8, one of which is all-ones.
    for (i, &v) in m[lh * lw..].iter().enumerate() {
        let want = if i % 2 == 0 { 1.0 / 8.0 } else { 1.0 };
        assert!((v - want).abs() < 1e-6, "occluded frame must condition FULLY, not empty: token {i} is {v}, expected {want}");
    }

    // A mask sequence that does not cover the clip is refused outright.
    let e = ltxv::maskcond::read_mask_sequence(&dir, frames + 8, w, h, lf, lh, lw).expect_err("a frame-count mismatch must be refused");
    assert!(e.contains("frames"), "{e}");
    let e = ltxv::maskcond::read_mask_sequence(&dir, frames, w * 2, h, lf, lh, lw).expect_err("a resolution mismatch must be refused");
    assert!(e.contains("64x64") || e.contains("128x64"), "{e}");

    let _ = std::fs::remove_dir_all(&dir);
}
