// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Forward parity vs the real `HiFTGenerator.inference()` reference, dumped
//! by `tools/goldens/cosyvoice_dump_reference.py` (`hift_real_*`).
//!
//! ## The RNG-crossing gap and how this suite works around it
//!
//! `hift_real_meta.json`'s own `"gotcha"` field documents it: `SineGen2`
//! (the NSF harmonic source) draws `torch.randn_like(sine_waves)` from
//! PyTorch's global Mersenne-Twister RNG on every call, so the golden's
//! `magnitude`/`phase`/`waveform` are only reproducible by reseeding torch's
//! OWN RNG immediately before `hift.inference()` - which is exactly what the
//! dumper does (`torch.manual_seed(SEED)`, self-validated bit-exact by
//! calling twice). `crate::hift`'s own module doc records a second, narrower
//! finding: `SineGen2`'s OTHER random draw (`rand_ini`, the initial-phase
//! noise) is empirically provably inert at HiFT's real `upsample_scale=480`,
//! because the reference's own downsample interpolation discards it before
//! it can reach the output. So the ONLY draw that matters is that one
//! `randn_like(sine_waves)` call.
//!
//! Reimplementing PyTorch's Mersenne-Twister transform in Rust to reproduce
//! that draw bit-for-bit is out of scope for this milestone (the same
//! honest-gap call `crate::sampling`'s module doc makes for `ras_sampling`'s
//! RNG). Instead, this suite injects the EXACT noise values a real, reseeded
//! `hift.inference()` run actually consumed - captured by an ad-hoc script
//! (not part of `tools/goldens/`, not committed - ordinary Python + this
//! venv's `torch`, run once against the real `hift.pt` with
//! `torch.rand`/`torch.randn` monkeypatched to record their outputs) into
//! `testdata/golden/cosyvoice/hift_real_nsf_noise.f32` (`[1,30720,9]`) and
//! `hift_real_nsf_rand_ini.f32` (`[1,9]`, captured but UNUSED - see
//! `crate::hift`'s doc for why). The capture self-validated the same way the
//! official dumper does: reseed, rerun, assert the recorded draws AND the
//! resulting waveform are bit-exact (`hift_real_nsf_rng_meta.json`).
//!
//! This proves the conv-trunk + NSF-source + ISTFT MATH bit-exactly (given
//! the same noise, the two implementations must agree to floating-point
//! precision) without claiming this port's own RNG stream matches PyTorch's -
//! [`hift::forward_seeded`] (production) draws its own noise from
//! `data::rng::Rng`, exactly as undocumented-but-intentional a gap as
//! `crate::sampling`'s.
//!
//! **This suite's magnitude/phase/waveform rungs skip (not fail) when the
//! ad-hoc noise capture is absent** - it is not provisioned by
//! `make fetch/testdata` (a from-scratch capture against the real
//! checkpoint, not a tracked fixture), so a box that only has the official
//! goldens still runs the crate's `hift::tests` (tiny-config smoke +
//! `import_hift_pt`'s two-way tensor coverage against the real
//! checkpoint) but not this file's magnitude/phase/waveform comparisons.

use brain_testutil::{golden::Source, parity::Table, read_f32, testdata_path};
use cosyvoice::hift::{decode, f0_predictor_forward, forward_seeded, nsf_source_forward};
use cosyvoice::hift_config::HiftConfig;
use cosyvoice::hift_import::import_hift_pt;

const DUMPER: &str = "tools/goldens/cosyvoice_dump_reference.py";
const COS_FLOOR: f64 = 0.9999;
const REL_CEIL: f64 = 1e-3;

fn weights_dir() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("BRAIN_COSYVOICE_HIFT") {
        let p = std::path::PathBuf::from(p);
        return p.join("hift.pt").is_file().then_some(p);
    }
    let p = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../resources/cosyvoice/weights"));
    p.join("hift.pt").is_file().then_some(p)
}

#[test]
fn real_magnitude_phase_and_waveform_match_the_reference_given_the_same_nsf_noise() {
    let dir = testdata_path("golden/cosyvoice");
    let meta = dir.join("hift_real_meta.json");
    let cfg = HiftConfig::cosyvoice2();
    let Some(src) = Source::open_manifest(&meta, DUMPER) else { return };
    if !src.require(&[
        ("in_channels", cfg.in_channels as i64),
        ("base_channels", cfg.base_channels as i64),
        ("nb_harmonics", cfg.nb_harmonics as i64),
        ("sampling_rate", cfg.sampling_rate as i64),
        ("n_fft", cfg.n_fft as i64),
        ("hop_len", cfg.hop_len as i64),
    ]) {
        return;
    }
    let Some(wdir) = weights_dir() else {
        brain_testutil::skip("set BRAIN_COSYVOICE_HIFT to a directory containing hift.pt");
        return;
    };

    let Some(mel) = read_f32(dir.join("hift_real_speech_feat_in.f32")) else {
        brain_testutil::skip("hift_real_speech_feat_in.f32 absent - run the dumper");
        return;
    };
    let Some(want_mag) = read_f32(dir.join("hift_real_magnitude.f32")) else { return };
    let Some(want_phase) = read_f32(dir.join("hift_real_phase.f32")) else { return };
    let Some(want_wave) = read_f32(dir.join("hift_real_waveform.f32")) else { return };

    // The ad-hoc RNG capture (see this file's module doc) - not a
    // fetch-testdata-provisioned fixture, so this rung skips cleanly rather
    // than failing when it is absent.
    let Some(noise) = read_f32(dir.join("hift_real_nsf_noise.f32")) else {
        brain_testutil::skip(
            "hift_real_nsf_noise.f32 absent - this ad-hoc RNG capture is not part of \
             `make fetch/testdata`; see hift_parity.rs's module doc for how to regenerate it \
             against a real hift.pt",
        );
        return;
    };

    let t_mel = mel.len() / cfg.in_channels as usize;
    assert_eq!(t_mel * cfg.in_channels as usize, mel.len(), "mel is not a whole number of {}-channel frames", cfg.in_channels);

    let w = import_hift_pt(wdir.join("hift.pt").to_str().unwrap(), &cfg).unwrap_or_else(|e| panic!("import_hift_pt: {e}"));

    let f0 = f0_predictor_forward(&w.f0_predictor, &mel, cfg.in_channels as usize, cfg.f0_cond_channels as usize, t_mel);
    let excitation = nsf_source_forward(&f0, &cfg, &w, &noise);
    let out = decode(&mel, t_mel, &excitation, &cfg, &w);

    assert_eq!(out.magnitude.len(), want_mag.len(), "magnitude length");
    assert_eq!(out.phase.len(), want_phase.len(), "phase length");
    assert_eq!(out.waveform.len(), want_wave.len(), "waveform length");

    let mut table = Table::new(COS_FLOOR, REL_CEIL);
    table.check("hift_real_magnitude", &out.magnitude, &want_mag);
    table.check("hift_real_phase", &out.phase, &want_phase);
    table.check("hift_real_waveform", &out.waveform, &want_wave);
    table.print();
    table.assert_clean();
}

/// `forward_seeded` (the production entry point - own RNG, no injected
/// noise): an HONEST best-effort check, not a golden-parity gate. Asserts
/// what DOES transfer across the RNG boundary (finiteness, the audio-limit
/// clamp, same-seed determinism) rather than claiming bit-for-bit agreement
/// with a reference this port's RNG cannot reproduce - mirrors
/// `llm_parity.rs`'s `real_ar_generation_is_seed_deterministic_and_valid`.
#[test]
fn production_forward_seeded_is_deterministic_and_bounded() {
    let dir = testdata_path("golden/cosyvoice");
    let Some(mel) = read_f32(dir.join("hift_real_speech_feat_in.f32")) else {
        brain_testutil::skip("hift_real_speech_feat_in.f32 absent - run the dumper");
        return;
    };
    let Some(wdir) = weights_dir() else {
        brain_testutil::skip("set BRAIN_COSYVOICE_HIFT to a directory containing hift.pt");
        return;
    };

    let cfg = HiftConfig::cosyvoice2();
    let t_mel = mel.len() / cfg.in_channels as usize;
    let w = import_hift_pt(wdir.join("hift.pt").to_str().unwrap(), &cfg).unwrap_or_else(|e| panic!("import_hift_pt: {e}"));

    let a = forward_seeded(&mel, t_mel, &cfg, &w, 20240727);
    let b = forward_seeded(&mel, t_mel, &cfg, &w, 20240727);
    assert_eq!(a.waveform, b.waveform, "forward_seeded must be deterministic for a fixed seed");

    assert!(a.waveform.iter().all(|v| v.is_finite()), "waveform must be finite");
    assert!(a.waveform.iter().all(|&v| (-cfg.audio_limit..=cfg.audio_limit).contains(&v)), "waveform must respect the audio_limit clamp");
    assert!(a.f0.iter().all(|v| v.is_finite() && *v >= 0.0), "f0 must be finite and non-negative (abs())");

    let c = forward_seeded(&mel, t_mel, &cfg, &w, 1);
    assert_ne!(a.waveform, c.waveform, "a different seed's own NSF noise draw should change the waveform");
}
