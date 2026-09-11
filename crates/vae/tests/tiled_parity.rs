// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The 2-D tiled image VAE against the whole-image path it replaces.
//!
//! Swedish Embedded AB implements memory-bounded image-autoencoder inference
//! for its clients. If your team needs expertise in running a high-resolution
//! diffusion pipeline inside a fixed VRAM budget, you can procure our services
//! by sending an email to info@swedishembedded.com.
//!
//! Four claims, of which only the first two can be exact:
//!
//! 1. **A cover that yields ONE tile is the untiled path**, bit for bit. This
//!    is what makes the automatic threshold safe: every size below it decodes
//!    exactly as it did before tiling existed.
//! 2. **The estimate a placement decision reads is bounded by the tile**, not
//!    by the image, once tiling engages - and is unchanged below the
//!    threshold.
//! 3. **The pooled GroupNorm statistics are the whole image's**, which is what
//!    the second pass buys and the largest single term in how close a tiled
//!    decode gets.
//! 4. **A genuinely split cover reproduces the whole-image result closely**,
//!    but NOT exactly, and the test prints by how much rather than pretending
//!    otherwise.
//!
//! Everything here but the last test runs on the CPU backend at a tiny config:
//! the geometry, slicing, blending, pooling and divisor are size-independent,
//! so a 32x32 image exercises the same arithmetic a 2048x2048 one does without
//! needing a card. The deviation ITSELF is not size-independent - it depends on
//! the tile's size against the model's receptive field - so the number that
//! means something is measured on the real checkpoint at a real resolution, by
//! the `#[ignore]`d test at the end.

mod zeros;

use brain_testutil::parity::{compare, rel_l2};
use data::rng::Lcg;
use vae::blocks::{GnStats, Tensors};
use vae::tiled::Tiling;
use vae::{VaeConfig, VaeDecoder, VaeEncoder, VaeTiledDecoder, VaeTiledEncoder};

/// A deliberately small autoencoder with the same SHAPE as the real one - two
/// resolution levels (so the decoder upsamples once, factor 2), resnets, a
/// GroupNorm/SiLU head - so every block the real graph runs is exercised.
/// `attn` picks whether the mid block carries its global self-attention, which
/// is the term that makes a tiled result differ most from a whole-image one.
fn tiny(attn: bool) -> VaeConfig {
    VaeConfig {
        in_channels: 3,
        out_channels: 3,
        latent_channels: 4,
        block_out_channels: vec![8, 8],
        layers_per_block: 1,
        norm_num_groups: 4,
        norm_eps: 1e-6,
        mid_block_add_attention: attn,
        scaling_factor: 1.0,
        shift_factor: 0.0,
        use_quant_conv: true,
        use_post_quant_conv: true,
        patch_size: [1, 1],
        batch_norm_eps: 1e-4,
    }
}

/// The shapes `zeros` declares, filled with reproducible noise: a graph of
/// zero weights decodes zeros, which every tiling bug also does.
fn randomize(mut t: Tensors, seed: u64) -> Tensors {
    let mut rng = Lcg::new(seed);
    // Sorted, so the weights a given seed produces do not depend on the hash
    // order the map happens to iterate in.
    let mut names: Vec<String> = t.keys().cloned().collect();
    names.sort();
    for name in names {
        let e = t.get_mut(&name).expect("just listed");
        for v in e.1.iter_mut() {
            *v = rng.scaled(0.3);
        }
    }
    t
}

fn cpu() -> gpu_core::Gpu {
    vae::device(Some("cpu"))
}

// --------------------------------------------------------------- exactness

/// The property the automatic threshold rests on: a cover with one tile is
/// the whole-image path, to the bit. Anything less and "tiling is off below
/// the threshold" would not be a claim about the OUTPUT.
#[test]
fn a_single_tile_cover_decodes_bit_identically_to_the_untiled_path() {
    let cfg = tiny(true);
    let ts = randomize(zeros::decoder(&cfg), 7);
    let (lh, lw) = (8u32, 8u32);
    let z = Lcg::new(11).vec(cfg.latent_channels as usize * (lh * lw) as usize);

    let gpu = cpu();
    let whole = VaeDecoder::from_diffusers_on(&gpu, cfg.clone(), &ts, lh, lw).decode(&z);
    // A tile larger than the latent: `split_by_size` returns one ramp-free
    // interval covering the axis.
    let tiled = VaeTiledDecoder::new(&gpu, cfg.clone(), &ts, lh, lw, Tiling::new(64, 16));
    assert_eq!(tiled.plan().tiles().len(), 1, "expected an untiled cover");
    assert_eq!(tiled.decode(&z), whole, "a one-tile cover must be bit-identical");
}

#[test]
fn a_single_tile_cover_encodes_bit_identically_to_the_untiled_path() {
    let cfg = tiny(true);
    let ts = randomize(zeros::encoder(&cfg), 5);
    let (h, w) = (16u32, 16u32);
    let img = Lcg::new(3).vec(cfg.in_channels as usize * (h * w) as usize);

    let gpu = cpu();
    let whole = VaeEncoder::from_diffusers_on(&gpu, cfg.clone(), &ts, h, w).encode(&img);
    let tiled = VaeTiledEncoder::new(&gpu, cfg.clone(), &ts, h, w, Tiling::new(64, 16));
    assert_eq!(tiled.plan().tiles().len(), 1, "expected an untiled cover");
    assert_eq!(tiled.encode(&img), whole, "a one-tile cover must be bit-identical");
}

// ------------------------------------------------------------- tiled parity

/// A genuinely split cover against the whole-image decode, at
/// [`Tiling::AUTO`]'s own 1:4 overlap-to-tile ratio.
///
/// The floors are enforced minimums, not measurements to reproduce, and they
/// are deliberately loose: this fixture's tile is 16 latent cells against a
/// receptive field of the same order, which is far harsher than the real
/// VAE's 64-cell tile. What they are for is catching a broken cover - a
/// swapped axis, an unnormalised seam, a dropped tile, an unpooled norm - all
/// of which land orders of magnitude away. The number that describes the
/// SHIPPING path is measured on the real checkpoint by
/// `a_tiled_decode_holds_up_on_the_real_checkpoint`.
#[test]
fn a_tiled_decode_reproduces_the_untiled_decode() {
    for (attn, cos_floor, l2_ceiling) in [(false, 0.995, 0.10), (true, 0.995, 0.10)] {
        let cfg = tiny(attn);
        let ts = randomize(zeros::decoder(&cfg), 7);
        let (lh, lw) = (32u32, 32u32);
        let z = Lcg::new(11).vec(cfg.latent_channels as usize * (lh * lw) as usize);

        let gpu = cpu();
        let whole = VaeDecoder::from_diffusers_on(&gpu, cfg.clone(), &ts, lh, lw).decode(&z);
        let tiled = VaeTiledDecoder::new(&gpu, cfg.clone(), &ts, lh, lw, Tiling::new(32, 8));
        assert!(tiled.plan().tiles().len() >= 9, "expected a genuinely split cover, got {}", tiled.plan().tiles().len());
        let got = tiled.decode(&z);
        assert_eq!(got.len(), whole.len());

        let (cos, max_abs) = compare(&got, &whole);
        let l2 = rel_l2(&got, &whole);
        println!("decode tiles={} attn={attn}: cosine {cos:.9} rel_l2 {l2:.6} max_abs {max_abs:.6}", tiled.plan().tiles().len());
        assert!(cos > cos_floor, "attn={attn}: cosine {cos:.9} below floor {cos_floor}");
        assert!(l2 < l2_ceiling, "attn={attn}: rel_l2 {l2:.6} above ceiling {l2_ceiling}");
    }
}

/// The same in the encode direction, where the blend happens on the LATENT
/// grid - see [`a_tiled_decode_reproduces_the_untiled_decode`] for what the
/// floors do and do not mean.
#[test]
fn a_tiled_encode_reproduces_the_untiled_encode() {
    for (attn, cos_floor, l2_ceiling) in [(false, 0.995, 0.10), (true, 0.995, 0.10)] {
        let cfg = tiny(attn);
        let ts = randomize(zeros::encoder(&cfg), 5);
        let (h, w) = (64u32, 64u32);
        let img = Lcg::new(3).vec(cfg.in_channels as usize * (h * w) as usize);

        let gpu = cpu();
        let whole = VaeEncoder::from_diffusers_on(&gpu, cfg.clone(), &ts, h, w).encode(&img);
        let tiled = VaeTiledEncoder::new(&gpu, cfg.clone(), &ts, h, w, Tiling::new(32, 8));
        assert!(tiled.plan().tiles().len() >= 9, "expected a genuinely split cover, got {}", tiled.plan().tiles().len());
        let got = tiled.encode(&img);
        assert_eq!(got.len(), whole.len());

        let (cos, max_abs) = compare(&got, &whole);
        let l2 = rel_l2(&got, &whole);
        println!("encode tiles={} attn={attn}: cosine {cos:.9} rel_l2 {l2:.6} max_abs {max_abs:.6}", tiled.plan().tiles().len());
        assert!(cos > cos_floor, "attn={attn}: cosine {cos:.9} below floor {cos_floor}");
        assert!(l2 < l2_ceiling, "attn={attn}: rel_l2 {l2:.6} above ceiling {l2_ceiling}");
    }
}

// ------------------------------------------------- synchronised statistics

/// The sharpest available gate on the two-pass GroupNorm synchronisation, and
/// the one that does not need an image-space tolerance to interpret: the
/// statistics the cover pools must be the statistics the WHOLE image has.
///
/// They are not identical and cannot be, for three reasons that are worth
/// separating:
///
/// * overlapping tiles count their shared region twice, tilting the pooled
///   mean slightly toward the overlap;
/// * a tile's borders are convolved against zeros rather than against its
///   neighbour's content;
/// * the collecting pass itself runs with each tile's LOCAL statistics, so a
///   norm deep in the stack sees an activation map that already diverged at
///   the norms before it. Only a fixed point over repeated passes would close
///   that, and it is not worth a third pass.
///
/// So this asserts two things at the two ends of that: the FIRST norm, whose
/// input is a plain convolution of the tile's own content, must land very
/// close to the whole image's; and taken over every norm, pooling must recover
/// most of what a single tile's own statistics get wrong.
#[test]
fn the_pooled_statistics_are_the_whole_images_statistics() {
    let cfg = tiny(true);
    let ts = randomize(zeros::decoder(&cfg), 7);
    let (lh, lw) = (32u32, 32u32);
    let z = Lcg::new(11).vec(cfg.latent_channels as usize * (lh * lw) as usize);

    let gpu = cpu();
    let whole = VaeDecoder::from_diffusers_on_gn(&gpu, cfg.clone(), &ts, lh, lw, GnStats::Collect);
    whole.decode(&z);
    let want = whole.read_gn_stats();
    assert!(want.len() > 4, "expected a norm at every block, got {}", want.len());

    let tiled = VaeTiledDecoder::new(&gpu, cfg.clone(), &ts, lh, lw, Tiling::new(32, 8));
    let got = tiled.gn_stats(&z);
    assert_eq!(got.len(), want.len(), "the cover must find the same norms the whole graph has");

    // Per-tile spread: what one tile's own statistics would be wrong by.
    let per_tile: Vec<Vec<f32>> = tiled
        .plan()
        .tiles()
        .iter()
        .map(|t| {
            let d = VaeDecoder::from_diffusers_on_gn(
                &gpu,
                cfg.clone(),
                &ts,
                t.h.src_len() as u32,
                t.w.src_len() as u32,
                GnStats::Collect,
            );
            d.decode(&vae::tiling2d::slice_src(&z, cfg.latent_channels as usize, (lh as usize, lw as usize), *t));
            d.read_gn_stats().concat()
        })
        .collect();
    let flat_want = want.concat();
    let flat_got = got.concat();
    let worst = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let worst_tile = per_tile.iter().map(|t| worst(t, &flat_want)).fold(0.0f32, f32::max);
    let pooled = worst(&flat_got, &flat_want);
    // The first norm, on its own: no earlier norm has moved its input.
    let first = worst(&got[0], &want[0]);
    let first_tile = per_tile.iter().map(|t| worst(&t[..want[0].len()], &want[0])).fold(0.0f32, f32::max);
    let scale = want[0].iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("gn stats: pooled worst |delta| {pooled:.6} vs a single tile's {worst_tile:.6}; first norm {first:.6} vs {first_tile:.6} (values up to {scale:.3})");
    assert!(
        first < 0.05 * scale && first < first_tile / 3.0,
        "the first norm's pooled statistics must be the whole image's: {first:.6} against values up to {scale:.3}, \
         where one tile's own is {first_tile:.6}"
    );
    assert!(
        pooled < worst_tile / 2.0,
        "pooling must recover most of what a per-tile norm gets wrong: {pooled:.6} against {worst_tile:.6}"
    );
}

// ------------------------------------------------------------ the estimate

const MIB: f64 = (1u64 << 20) as f64;

/// The whole point of the cost variant: once tiling engages, what the graph
/// holds is one TILE's activations, so the number a placement decision reads
/// must stop tracking the image. Without this the planner refuses a run the
/// hardware can do.
#[test]
fn the_planned_estimate_is_bounded_by_the_tile_not_the_image() {
    let cfg = VaeConfig::flux2();
    let mut prev_untiled = 0u64;
    let mut planned = Vec::new();
    // 1 MP (a stock FLUX.2 frame) up to 64 MP (8192x8192).
    for mp in [1u64, 2, 4, 8, 16, 32, 64] {
        let px = mp * 1_000_000;
        let untiled = vae::decoder_device_bytes_for_pixels(&cfg, px);
        let plan = vae::decoder_device_bytes_for_pixels_planned(&cfg, px);
        println!("{mp:>3} MP: untiled {:>9.0} MiB  planned {:>9.0} MiB", untiled as f64 / MIB, plan as f64 / MIB);
        assert!(untiled > prev_untiled, "the untiled formula is the thing that grows");
        prev_untiled = untiled;
        assert!(plan <= untiled, "the planned figure may never exceed the untiled one");
        planned.push(plan);
    }
    let worst = *planned.iter().max().expect("non-empty");
    assert!(
        worst <= vae::tiled::WHOLE_GRAPH_MAX_BYTES,
        "the planned figure must stay inside the budget that decides to tile: {:.0} MiB",
        worst as f64 / MIB
    );
    // Bounded, not merely slower-growing: beyond the threshold the tile is the
    // same tile however large the image gets.
    assert_eq!(planned[4], planned[6], "16 MP and 64 MP must cost the same tiled decode");
    assert!(
        vae::encoder_device_bytes_for_pixels_planned(&cfg, 16_000_000)
            == vae::encoder_device_bytes_for_pixels_planned(&cfg, 64_000_000),
        "the encode estimate must be bounded the same way"
    );
}

/// Zero behaviour change where nothing needed to change: at the sizes FLUX.2
/// actually generates, the decision is "do not tile" and the number is the
/// one that was already gated in `tests/footprint.rs`.
#[test]
fn a_normal_resolution_run_is_untouched() {
    let cfg = VaeConfig::flux2();
    for (h, w) in [(512u64, 512u64), (768, 1024), (1024, 1024), (1024, 1536)] {
        let px = h * w;
        assert!(!vae::tiled::should_tile_decode(&cfg, px), "{h}x{w} must stay on the whole-image decode");
        assert!(!vae::tiled::should_tile_encode(&cfg, px), "{h}x{w} must stay on the whole-image encode");
        assert_eq!(
            vae::decoder_device_bytes_for_pixels_planned(&cfg, px),
            vae::decoder_device_bytes_for_pixels(&cfg, px),
            "{h}x{w}: the reservation must be the untiled one"
        );
        assert_eq!(
            vae::encoder_device_bytes_for_pixels_planned(&cfg, px),
            vae::encoder_device_bytes_for_pixels(&cfg, px)
        );
    }
    // ...and the sizes that motivated tiling do engage it.
    for (h, w) in [(2048u64, 2048u64), (1536, 2048), (4096, 4096)] {
        assert!(vae::tiled::should_tile_decode(&cfg, h * w), "{h}x{w} must tile");
    }
}

// ------------------------------------------------- the real thing, measured

/// What the shipping path actually costs in image quality, on the real FLUX.2
/// VAE at a real resolution and on an in-distribution latent.
///
/// Every number above is from a tiny fixture whose proportions are far harsher
/// than production's; this is the one that describes what a user sees. It
/// decodes a 1024x1024 latent - obtained by putting a synthetic scene through
/// this same VAE's encoder, so the latent is one the decoder was trained for -
/// as nine 512-pixel tiles, and holds the result to the whole-image decode.
///
/// It also gates the reason the second pass exists: with per-tile norms the
/// same cover is worse by more than a factor of two in relative L2, which on a
/// picture is the familiar per-tile brightness stepping.
///
/// Ignored by default - it needs the real checkpoint (`BRAIN_FLUX2_VAE`), a
/// card, and a couple of minutes. Run it after touching the cover, the pooling
/// or the blend.
#[test]
#[ignore = "needs BRAIN_FLUX2_VAE and a card; minutes. The quality number for the shipping path"]
fn a_tiled_decode_holds_up_on_the_real_checkpoint() {
    let Some(ts) = real_flux2_vae() else { return };
    let cfg = VaeConfig::flux2();
    let gpu = vae::device(Some("gpu"));
    let (h, w) = (1024u32, 1024u32);
    let (lh, lw) = (h / 8, w / 8);

    let img = scene(h as usize, w as usize);
    let z = VaeEncoder::from_diffusers_on(&gpu, cfg.clone(), &ts, h, w).encode_mean(&img, lh, lw);
    let whole = VaeDecoder::from_diffusers_on(&gpu, cfg.clone(), &ts, lh, lw).decode(&z);
    println!("this VAE's own round trip: {:.2} dB", psnr(&whole, &img));

    let mut result = Vec::new();
    for tiling in [Tiling::AUTO, Tiling::AUTO.local_gn()] {
        let tiled = VaeTiledDecoder::new(&gpu, cfg.clone(), &ts, lh, lw, tiling);
        assert_eq!(tiled.plan().tiles().len(), 9, "expected the 3x3 cover AUTO gives a 1024x1024 image");
        let got = tiled.decode(&z);
        let (cos, max_abs) = compare(&got, &whole);
        let l2 = rel_l2(&got, &whole);
        println!(
            "global_gn={}: cosine {cos:.9} rel_l2 {l2:.6} max_abs {max_abs:.6} psnr-vs-whole {:.2} dB",
            tiling.global_gn,
            psnr(&got, &whole)
        );
        result.push((cos, l2, psnr(&got, &whole)));
    }
    let (cos, l2, db) = result[0];
    assert!(cos > 0.999, "cosine {cos:.9}");
    assert!(l2 < 0.06, "rel_l2 {l2:.6}");
    assert!(db > 35.0, "a tiled decode must be within a decibel or two of the whole-image one: {db:.2} dB");
    assert!(
        result[1].1 > 2.0 * l2,
        "synchronising the norms must be worth a factor of two, or the second pass is not earning its cost: \
         rel_l2 {l2:.6} synchronised vs {:.6} per-tile",
        result[1].1
    );
}

/// A natural-ish RGB image in `[-1,1]`, CHW: smooth lighting, hard edges and
/// some texture. A VAE's latent space is only meaningful for images like the
/// ones it was trained on, so a deviation measured on noise says nothing about
/// what a picture would look like.
fn scene(h: usize, w: usize) -> Vec<f32> {
    let mut img = vec![0.0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let (fy, fx) = (y as f32 / h as f32, x as f32 / w as f32);
            let sky = 1.0 - 1.4 * fy;
            let sun = (-(((fx - 0.7).powi(2) + (fy - 0.2).powi(2)) * 30.0)).exp();
            let ground = f32::from(fy > 0.62);
            let tex = ((x as f32 * 0.35).sin() * (y as f32 * 0.27).cos()) * 0.08;
            let check = f32::from((x / 64 + y / 64) % 2 == 0) * 0.15;
            for (c, v) in [
                sky * 0.6 + sun * 0.9 - ground * 0.9 + tex + check,
                sky * 0.7 + sun * 0.6 - ground * 0.4 + tex,
                sky * 0.9 + sun * 0.2 - ground * 1.2 + tex - check,
            ]
            .into_iter()
            .enumerate()
            {
                img[(c * h + y) * w + x] = v.clamp(-1.0, 1.0);
            }
        }
    }
    img
}

/// Peak signal-to-noise ratio for images in `[-1,1]` (peak-to-peak 2).
fn psnr(a: &[f32], b: &[f32]) -> f64 {
    let mse: f64 = a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).powi(2)).sum::<f64>() / a.len() as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (4.0f64 / mse).log10()
}

/// The real VAE tensors, or `None` with a skip line - `BRAIN_FLUX2_VAE` names
/// either the diffusers `vae/` directory or the safetensors file itself, the
/// same two shapes `tests/flux2_parity.rs` accepts.
fn real_flux2_vae() -> Option<Tensors> {
    let Some(env) = std::env::var("BRAIN_FLUX2_VAE").ok().filter(|p| !p.is_empty()) else {
        brain_testutil::skip("set BRAIN_FLUX2_VAE to the FLUX.2 vae/ dir or safetensors file");
        return None;
    };
    let p = std::path::Path::new(&env);
    let path = if p.is_dir() { p.join("diffusion_pytorch_model.safetensors") } else { p.to_path_buf() };
    if !path.exists() {
        brain_testutil::skip(&format!("BRAIN_FLUX2_VAE={env} has no vae safetensors"));
        return None;
    }
    Some(
        checkpoint::safetensors::read(path.to_str().expect("utf-8 path"))
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
            .into_iter()
            .map(|t| (t.name, (t.shape, t.data)))
            .collect(),
    )
}
