// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end pipeline coverage for `crate::pipeline`, at three rungs:
//!
//! 1. `mel_frontend_matches_the_reference_mel` - `pipeline::extract_prompt_mel`
//!    (the 24 kHz prompt-mel glue, never independently checked against a
//!    golden until this test) vs the real `mel_real_*` dump - stage parity
//!    for a piece of glue code this milestone adds, not a component this
//!    port already proved.
//! 2. `spliced_flow_and_hift_reproduce_the_reference_given_golden_tokens_and_xvec` -
//!    a composed-pipeline regression check (this port's parity ladder's
//!    "composed-loop parity, with real weights" rung): feed the GOLDEN's own
//!    captured prompt/generated speech tokens, x-vector and prompt mel
//!    through THIS crate's `flow::forward`, then feed flow's own mel output
//!    straight into `hift::forward` (using the ad-hoc NSF-noise capture
//!    `hift_parity.rs` already documents) - both components were already
//!    parity-proven standalone; this proves the SEAM between them (flow's
//!    channel-major mel output needs no reshaping to become HiFT's input)
//!    still reproduces the reference end to end, without solving the LM/NSF
//!    RNG-crossing problem.
//! 3. `full_pipeline_produces_a_real_playable_wav_from_real_weights` - the
//!    actual milestone deliverable: `pipeline::generate()`, this crate's OWN
//!    sampling end to end, against the real reference clip
//!    (`resources/cosyvoice/source/asset/zero_shot_prompt.wav`) and a short
//!    target text, gated STRUCTURALLY (finite, bounded, non-silent, a
//!    plausible duration, deterministic given the same seed) rather than
//!    against a golden waveform - see `crate::pipeline`'s own module doc for
//!    why bit-exact end-to-end parity is not the right gate here.
//!
//! Skips cleanly when the goldens or the checkpoints are absent. Rung 3 runs
//! the flow decoder's UNet forward, which this crate's own docs already
//! record as impractically slow in a debug build - run with `--release`.

use std::path::PathBuf;

use brain_testutil::{golden::Source, parity::Table, read_f32, read_i32, testdata_path};
use cosyvoice::flow;
use cosyvoice::flow_config::FlowConfig;
use cosyvoice::flow_import::import_flow_pt;
use cosyvoice::hift;
use cosyvoice::hift_config::HiftConfig;
use cosyvoice::hift_import::import_hift_pt;
use cosyvoice::pipeline::{self, CosyVoicePaths, GenOpts};

const DUMPER: &str = "tools/goldens/cosyvoice_dump_reference.py";
const COS_FLOOR: f64 = 0.9999;
const REL_CEIL: f64 = 1e-3;

fn repo_dir(rel: &str) -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../..")).join(rel)
}

/// `env_var`, else the repo-relative `resources/cosyvoice/weights` -
/// the same fallback convention `llm_parity.rs`/`flow_parity.rs`/
/// `hift_parity.rs`/`campplus`'s and `s3tokenizer`'s own parity tests use.
fn weights_role_dir(env_var: &str, marker_file: &str) -> Option<PathBuf> {
    if let Ok(p) = std::env::var(env_var) {
        let p = PathBuf::from(p);
        return p.join(marker_file).exists().then_some(p);
    }
    let p = repo_dir("resources/cosyvoice/weights");
    p.join(marker_file).exists().then_some(p)
}

fn cosyvoice_blank_en_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BRAIN_COSYVOICE_TOKENIZER") {
        let p = PathBuf::from(p);
        return p.join("vocab.json").exists().then_some(p);
    }
    let p = repo_dir("resources/cosyvoice/weights/CosyVoice-BlankEN");
    p.join("vocab.json").exists().then_some(p)
}

fn all_paths() -> Option<CosyVoicePaths> {
    let llm = weights_role_dir("BRAIN_COSYVOICE_LLM", "llm.pt")?;
    let flow = weights_role_dir("BRAIN_COSYVOICE_FLOW", "flow.pt")?;
    let hift = weights_role_dir("BRAIN_COSYVOICE_HIFT", "hift.pt")?;
    let s3tokenizer = weights_role_dir("BRAIN_S3TOKENIZER_V2", s3tokenizer::import::RELEASE_FILE)?;
    let campplus = weights_role_dir("BRAIN_CAMPPLUS_DIR", campplus::import::RELEASE_FILE)?;
    let tokenizer = cosyvoice_blank_en_dir()?;
    Some(CosyVoicePaths {
        llm: llm.to_string_lossy().into_owned(),
        flow: flow.to_string_lossy().into_owned(),
        hift: hift.to_string_lossy().into_owned(),
        s3tokenizer: s3tokenizer.to_string_lossy().into_owned(),
        campplus: campplus.to_string_lossy().into_owned(),
        tokenizer: tokenizer.to_string_lossy().into_owned(),
    })
}

/// `[c, t]` channel-major -> `[t, c]` time-major (`mel_real_out.f32` is
/// dumped straight from `feat_extractor(wav)`'s own `(1, num_mels, T)`
/// layout, same convention `flow_parity.rs`'s own helper of this name uses).
fn transpose_ct_to_tc(x: &[f32], c: usize, t: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; c * t];
    for ci in 0..c {
        for ti in 0..t {
            y[ti * c + ci] = x[ci * t + ti];
        }
    }
    y
}

#[test]
fn mel_frontend_matches_the_reference_mel() {
    let dir = testdata_path("golden/cosyvoice");
    let meta = dir.join("mel_real_meta.json");
    let cfg = audio::mel::MelConfig::cosyvoice_24k();
    let Some(src) = Source::open_manifest(&meta, DUMPER) else { return };
    if !src.require(&[
        ("n_fft", cfg.n_fft as i64),
        ("num_mels", cfg.n_mels as i64),
        ("sampling_rate", cfg.sample_rate as i64),
        ("hop_size", cfg.hop as i64),
        ("win_size", cfg.win as i64),
        ("fmin", cfg.fmin as i64),
        ("fmax", cfg.fmax as i64),
    ]) {
        return;
    }
    let Some(samples) = read_f32(dir.join("mel_real_in.f32")) else { return };
    let Some(want_ct) = read_f32(dir.join("mel_real_out.f32")) else { return };

    let (got_tc, n_frames) = pipeline::extract_prompt_mel(&samples);
    assert_eq!(got_tc.len(), want_ct.len(), "mel length");
    let want_tc = transpose_ct_to_tc(&want_ct, cfg.n_mels, n_frames);

    let mut table = Table::new(COS_FLOOR, REL_CEIL);
    table.check("mel_real_out (time-major)", &got_tc, &want_tc);
    table.print();
    table.assert_clean();
}

#[test]
fn spliced_flow_and_hift_reproduce_the_reference_given_golden_tokens_and_xvec() {
    let dir = testdata_path("golden/cosyvoice");
    let flow_cfg = FlowConfig::cosyvoice2();
    let hift_cfg = HiftConfig::cosyvoice2();

    let Some(flow_src) = Source::open_manifest(&dir.join("flow_real_meta.json"), DUMPER) else { return };
    if !flow_src.require(&[("input_size", flow_cfg.input_size as i64), ("output_size", flow_cfg.output_size as i64), ("n_timesteps", flow_cfg.n_timesteps as i64)]) {
        return;
    }
    let Some(hift_src) = Source::open_manifest(&dir.join("hift_real_meta.json"), DUMPER) else { return };
    if !hift_src.require(&[("in_channels", hift_cfg.in_channels as i64), ("n_fft", hift_cfg.n_fft as i64), ("hop_len", hift_cfg.hop_len as i64)]) {
        return;
    }

    let Some(flow_dir) = weights_role_dir("BRAIN_COSYVOICE_FLOW", "flow.pt") else {
        brain_testutil::skip("set BRAIN_COSYVOICE_FLOW to a directory containing flow.pt");
        return;
    };
    let Some(hift_dir) = weights_role_dir("BRAIN_COSYVOICE_HIFT", "hift.pt") else {
        brain_testutil::skip("set BRAIN_COSYVOICE_HIFT to a directory containing hift.pt");
        return;
    };

    let Some(noise) = read_f32(dir.join("hift_real_nsf_noise.f32")) else {
        brain_testutil::skip(
            "hift_real_nsf_noise.f32 absent - this ad-hoc RNG capture is not part of \
             `make fetch/testdata`; see hift_parity.rs's module doc for how to regenerate it",
        );
        return;
    };
    let Some(want_wave) = read_f32(dir.join("hift_real_waveform.f32")) else { return };
    let Some(want_mel) = read_f32(dir.join("flow_real_mel_out.f32")) else { return };

    let prompt_tokens = read_i32(dir.join("s3tokenizer_real_tokens.i32")).expect("s3tokenizer_real_tokens.i32");
    let gen_tokens = read_i32(dir.join("llm_real_ar_tokens.i32")).expect("llm_real_ar_tokens.i32");
    let xvec = read_f32(dir.join("campplus_real_out.f32")).expect("campplus_real_out.f32");
    let mel_ct = read_f32(dir.join("mel_real_out.f32")).expect("mel_real_out.f32");
    let mel_len1 = mel_ct.len() / flow_cfg.output_size as usize;
    let prompt_feat_tc = transpose_ct_to_tc(&mel_ct, flow_cfg.output_size as usize, mel_len1);

    let flow_w = import_flow_pt(flow_dir.join("flow.pt").to_str().unwrap(), &flow_cfg).unwrap_or_else(|e| panic!("import_flow_pt: {e}"));
    let noise_buf = flow::rand_noise();
    let flow_out = flow::forward(&flow_w, &flow_cfg, &prompt_tokens, &gen_tokens, &xvec, &prompt_feat_tc, mel_len1, &noise_buf, flow_cfg.n_timesteps as usize);
    drop(flow_w);

    assert_eq!(flow_out.mel.len(), want_mel.len(), "flow mel length");
    let mut mel_table = Table::new(COS_FLOOR, REL_CEIL);
    mel_table.check("pipeline_flow_mel_out", &flow_out.mel, &want_mel);
    mel_table.print();
    mel_table.assert_clean();

    let t_mel = flow_out.mel.len() / hift_cfg.in_channels as usize;
    let hift_w = import_hift_pt(hift_dir.join("hift.pt").to_str().unwrap(), &hift_cfg).unwrap_or_else(|e| panic!("import_hift_pt: {e}"));
    let hift_out = hift::forward(&flow_out.mel, t_mel, &hift_cfg, &hift_w, &noise);

    assert_eq!(hift_out.waveform.len(), want_wave.len(), "waveform length");
    let mut wave_table = Table::new(COS_FLOOR, REL_CEIL);
    wave_table.check("pipeline_flow_to_hift_waveform", &hift_out.waveform, &want_wave);
    wave_table.print();
    wave_table.assert_clean();
}

#[test]
fn full_pipeline_produces_a_real_playable_wav_from_real_weights() {
    let Some(paths) = all_paths() else {
        brain_testutil::skip("real CosyVoice2/S3Tokenizer/CAM++ checkpoints not found - set BRAIN_COSYVOICE_{LLM,FLOW,HIFT}/BRAIN_S3TOKENIZER_V2/BRAIN_CAMPPLUS_DIR/BRAIN_COSYVOICE_TOKENIZER or fetch resources/cosyvoice");
        return;
    };
    let ref_wav = repo_dir("resources/cosyvoice/source/asset/zero_shot_prompt.wav");
    if !ref_wav.is_file() {
        brain_testutil::skip(&format!("reference clip not found at {}", ref_wav.display()));
        return;
    }

    let ref_text = "\u{5e0c}\u{671b}\u{4f60}\u{4ee5}\u{540e}\u{80fd}\u{591f}\u{505a}\u{7684}\u{6bd4}\u{6211}\u{8fd8}\u{597d}\u{5466}\u{3002}";
    let text = "\u{4f60}\u{597d}\u{ff0c}\u{5f88}\u{9ad8}\u{5174}\u{8ba4}\u{8bc6}\u{4f60}\u{3002}";
    let opts = GenOpts { seed: 42, ..GenOpts::default() };

    let out_a = pipeline::generate(&paths, &opts, text, ref_wav.to_str().unwrap(), ref_text).unwrap_or_else(|e| panic!("pipeline::generate: {e}"));

    assert!(!out_a.samples.is_empty(), "generated zero samples");
    assert!(out_a.samples.iter().all(|v| v.is_finite()), "waveform must be finite");
    assert!(out_a.samples.iter().all(|&v| v.abs() <= 1.0 + 1e-3), "waveform must stay within the audio_limit clamp");

    let rms = (out_a.samples.iter().map(|&v| v * v).sum::<f32>() / out_a.samples.len() as f32).sqrt();
    let duration_s = out_a.samples.len() as f32 / out_a.sample_rate as f32;
    println!("full_pipeline: {} samples @ {} Hz = {:.3} s, rms={:.5}", out_a.samples.len(), out_a.sample_rate, duration_s, rms);
    assert!(rms > 1e-4, "waveform RMS {rms} looks silent");
    assert!(duration_s > 0.05, "waveform duration {duration_s}s implausibly short for the requested text");

    let out_b = pipeline::generate(&paths, &opts, text, ref_wav.to_str().unwrap(), ref_text).unwrap_or_else(|e| panic!("pipeline::generate (rerun): {e}"));
    assert_eq!(out_a.samples.len(), out_b.samples.len(), "same seed must reproduce the same length");
    assert_eq!(out_a.samples, out_b.samples, "same seed + same inputs must reproduce the same waveform end to end");

    let out_path = std::env::temp_dir().join("cosyvoice_pipeline_e2e_smoke.wav");
    audio::wav::write(&out_path, &out_a.samples, out_a.sample_rate).unwrap_or_else(|e| panic!("write {}: {e}", out_path.display()));
    let reread = audio::wav::read(&out_path).unwrap_or_else(|e| panic!("read back {}: {e}", out_path.display()));
    assert_eq!(reread.sample_rate, out_a.sample_rate);
    assert_eq!(reread.samples.len(), out_a.samples.len(), "the written WAV must round-trip the same sample count");
    let _ = std::fs::remove_file(&out_path);
}
