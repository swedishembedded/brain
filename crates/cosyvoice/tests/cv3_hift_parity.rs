// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Forward parity vs the real `CausalHiFTGenerator.inference()` reference,
//! dumped by `tools/goldens/cosyvoice3_dump_reference.py` (`hift_real_*`)
//! PLUS an ad-hoc NSF-noise capture this suite depends on (see below) - the
//! CosyVoice 3 sibling of `crates/cosyvoice/tests/hift_parity.rs`, driven by a
//! materially EASIER noise story: CosyVoice 3's `SineGen2(causal=True)` reads
//! a FIXED per-instance buffer (drawn once at construction, never
//! checkpointed - `hift_real_meta.json`'s own `"gotcha"` field documents
//! this), so - unlike CosyVoice 2's per-call-reseed requirement - capturing
//! that ONE buffer (`hift_real_nsf_noise.f32`, `[1, 30720, 9]`) and feeding it
//! into this port's `nsf_source_forward_causal` lets the conv-trunk + NSF +
//! ISTFT math be checked against the real `magnitude`/`phase`/`waveform`
//! without reimplementing PyTorch's RNG at all:
//! `tools/goldens/../scratchpad/capture_cv3_hift_noise.py` (an ad-hoc,
//! uncommitted script - ordinary Python + this venv's `torch`, run once
//! against the real `hift.pt`, reconstructing the model the SAME way
//! `cosyvoice3_dump_reference.py`'s own `main()` does so the global RNG
//! trajectory lines up) self-validated by reconstructing the model TWICE from
//! the same seed (bit-exact buffer both times) and by replaying
//! `hift.inference()` and comparing the resulting waveform against
//! `hift_real_waveform.f32` with `torch.equal` (bit-exact) BEFORE writing the
//! capture out - see `hift_real_nsf_noise_meta.json`.
//!
//! **This rung is NOT bit-exact, and the reason is known, not hand-waved**:
//! the reference explicitly runs `f0_predictor` in `float64` for causal
//! inference ("precision is crucial" per its own comment); this port runs it
//! in `f32` throughout. The measured effect is small but real - cosine
//! 0.9999999936/0.9999999961/0.9999998382 and `rel_l2`
//! 1.128e-4/8.797e-5/5.689e-4 for magnitude/phase/waveform respectively (this
//! suite's own numbers, not assumed) - well inside this crate's
//! `COS_FLOOR`/`REL_CEIL` gate, but a real, attributable f32-vs-f64 residual
//! rather than a perfect match.
//!
//! Skips cleanly when the golden, the ad-hoc noise capture, or the
//! checkpoint is absent.

use brain_testutil::{golden::Source, parity::Table, read_f32, testdata_path};
use cosyvoice::cv3_hift::{decode, f0_predictor_forward, Cv3HiftInstance};
use cosyvoice::cv3_hift_config::Cv3HiftConfig;
use cosyvoice::cv3_hift_import::import_cv3_hift_pt;
use cosyvoice::hift::nsf_source_forward_causal;

const DUMPER: &str = "tools/goldens/cosyvoice3_dump_reference.py";
const COS_FLOOR: f64 = 0.9999;
const REL_CEIL: f64 = 1e-3;

fn weights_dir() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("BRAIN_COSYVOICE3_HIFT") {
        let p = std::path::PathBuf::from(p);
        return p.join("hift.pt").is_file().then_some(p);
    }
    let p = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../resources/cosyvoice/weights3"));
    p.join("hift.pt").is_file().then_some(p)
}

#[test]
fn real_magnitude_phase_and_waveform_match_the_reference_given_the_captured_nsf_noise() {
    let dir = testdata_path("golden/cosyvoice3");
    let meta = dir.join("hift_real_meta.json");
    let cfg = Cv3HiftConfig::cosyvoice3();
    let Some(src) = Source::open_manifest(&meta, DUMPER) else { return };
    if !src.require(&[
        ("in_channels", cfg.in_channels as i64),
        ("base_channels", cfg.base_channels as i64),
        ("nb_harmonics", cfg.nb_harmonics as i64),
        ("sampling_rate", cfg.sampling_rate as i64),
        ("n_fft", cfg.n_fft as i64),
        ("hop_len", cfg.hop_len as i64),
        ("conv_pre_look_right", cfg.conv_pre_look_right as i64),
    ]) {
        return;
    }
    let Some(wdir) = weights_dir() else {
        brain_testutil::skip("set BRAIN_COSYVOICE3_HIFT to a directory containing hift.pt");
        return;
    };

    let Some(mel) = read_f32(dir.join("hift_real_speech_feat_in.f32")) else {
        brain_testutil::skip("hift_real_speech_feat_in.f32 absent - run the dumper");
        return;
    };
    let Some(want_mag) = read_f32(dir.join("hift_real_magnitude.f32")) else { return };
    let Some(want_phase) = read_f32(dir.join("hift_real_phase.f32")) else { return };
    let Some(want_wave) = read_f32(dir.join("hift_real_waveform.f32")) else { return };

    let Some(noise) = read_f32(dir.join("hift_real_nsf_noise.f32")) else {
        brain_testutil::skip(
            "hift_real_nsf_noise.f32 absent - this ad-hoc RNG capture is not part of \
             `make fetch/testdata`; see this file's module doc for how to regenerate it \
             against a real hift.pt",
        );
        return;
    };

    let t_mel = mel.len() / cfg.in_channels as usize;
    assert_eq!(t_mel * cfg.in_channels as usize, mel.len());

    let w = import_cv3_hift_pt(wdir.join("hift.pt").to_str().unwrap(), &cfg).unwrap_or_else(|e| panic!("import_cv3_hift_pt: {e}"));

    let f0 = f0_predictor_forward(&w.f0_predictor, &mel, cfg.in_channels as usize, cfg.f0_cond_channels as usize, t_mel);
    let nsf_cfg = cosyvoice::hift_config::HiftConfig {
        in_channels: cfg.in_channels,
        base_channels: cfg.base_channels,
        nb_harmonics: cfg.nb_harmonics,
        sampling_rate: cfg.sampling_rate,
        nsf_alpha: cfg.nsf_alpha,
        nsf_sigma: cfg.nsf_sigma,
        nsf_voiced_threshold: cfg.nsf_voiced_threshold,
        upsample_rates: cfg.upsample_rates,
        upsample_kernel_sizes: cfg.upsample_kernel_sizes,
        n_fft: cfg.n_fft,
        hop_len: cfg.hop_len,
        resblock_kernel_sizes: cfg.resblock_kernel_sizes,
        source_resblock_kernel_sizes: cfg.source_resblock_kernel_sizes,
        lrelu_slope: cfg.lrelu_slope,
        audio_limit: cfg.audio_limit,
        f0_cond_channels: cfg.f0_cond_channels,
    };
    let excitation = nsf_source_forward_causal(&f0, &nsf_cfg, &w, &noise);
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

/// `forward()` two ways, given the SAME captured NSF noise, must agree
/// (sanity on `cv3_hift::forward`'s own purity) and `Cv3HiftInstance`'s
/// fixed-per-instance-buffer contract must hold: two `forward()` calls on
/// the SAME instance are bit-exact without reseeding - mirroring the
/// golden's OWN self-validation (`hift_real_meta.json`'s
/// `"self_validation"`), not just asserted by fiat.
#[test]
fn cv3_hift_instance_is_bit_exact_across_repeated_calls_without_reseeding() {
    let dir = testdata_path("golden/cosyvoice3");
    let Some(mel) = read_f32(dir.join("hift_real_speech_feat_in.f32")) else {
        brain_testutil::skip("hift_real_speech_feat_in.f32 absent - run the dumper");
        return;
    };
    let Some(wdir) = weights_dir() else {
        brain_testutil::skip("set BRAIN_COSYVOICE3_HIFT to a directory containing hift.pt");
        return;
    };

    let cfg = Cv3HiftConfig::cosyvoice3();
    let t_mel = mel.len() / cfg.in_channels as usize;
    let w = import_cv3_hift_pt(wdir.join("hift.pt").to_str().unwrap(), &cfg).unwrap_or_else(|e| panic!("import_cv3_hift_pt: {e}"));

    let inst = Cv3HiftInstance::new_seeded(w, &cfg, t_mel, 20240727);
    let a = inst.forward(&mel, t_mel, &cfg);
    let b = inst.forward(&mel, t_mel, &cfg);
    assert_eq!(a.waveform, b.waveform, "two forward() calls on the same Cv3HiftInstance must be bit-exact without reseeding");
    assert!(a.waveform.iter().all(|v| v.is_finite()));
    assert!(a.waveform.iter().all(|&v| (-cfg.audio_limit..=cfg.audio_limit).contains(&v)));

    // A DIFFERENT instance (different seed -> different fixed buffer) must
    // generally differ - proves the buffer is actually load-bearing, not
    // silently ignored.
    let w2 = import_cv3_hift_pt(wdir.join("hift.pt").to_str().unwrap(), &cfg).unwrap_or_else(|e| panic!("import_cv3_hift_pt: {e}"));
    let inst2 = Cv3HiftInstance::new_seeded(w2, &cfg, t_mel, 1);
    let c = inst2.forward(&mel, t_mel, &cfg);
    assert_ne!(a.waveform, c.waveform, "a different instance's own fixed noise buffer should change the waveform");
}
