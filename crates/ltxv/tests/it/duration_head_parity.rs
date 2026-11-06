// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LTX-2.5 duration-head parity against
//! `tools/goldens/ltxv_duration_head_dump_reference.py`:
//!
//! 1. [`ltxv_duration_head_matches_reference`] - every dumped stage
//!    (`video_proj`/`audio_proj`/`tokens`/`pooled`/`hidden`/`duration`),
//!    real weights, on the golden's own synthetic token sequences.
//! 2. [`ltxv_duration_head_import_covers_the_shipped_checkpoint`] - the
//!    importer against the REAL 15-tensor file, both directions.
//! 3. [`a_prompt_predicts_the_reference_clip_length_on_real_weights`] - the
//!    whole prediction (text encoder, connectors, head) on real prompts,
//!    against `tools/goldens/ltxv_duration_e2e_dump_reference.py`.
//!
//! Needs the real duration-head weights and the golden fixture; skips loudly
//! without them (`BRAIN_REQUIRE_FIXTURES=1` upgrades a skip to a failure,
//! same convention as every other parity suite in this repo).

use std::path::Path;

use ltxv::duration_head::{DurationHead, DurationHeadConfig};
use ltxv::import::import_duration_head;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "cosine: length mismatch ({} vs {})", a.len(), b.len());
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        d += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    let den = na.sqrt() * nb.sqrt();
    if den <= 0.0 {
        0.0
    } else {
        d / den
    }
}

fn max_abs(got: &[f32], want: &[f32]) -> f32 {
    got.iter().zip(want).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn report(label: &str, got: &[f32], want: &[f32], min_cos: f64) {
    assert_eq!(got.len(), want.len(), "{label}: {} values vs {}", got.len(), want.len());
    let (c, m) = (cosine(got, want), max_abs(got, want));
    eprintln!("{label}: cosine={c:.9}  max_abs={m:.3e}  n={}", got.len());
    assert!(c >= min_cos, "{label}: cosine {c:.9} < {min_cos}");
}

/// Row-wise agreement of a `[rows, dim]` encoding, summarized over runs of
/// rows that agree alike - so a divergence confined to the padding the
/// connector fills with registers reads apart from one in the prompt's rows.
fn report_rows(label: &str, got: &[f32], want: &[f32], dim: usize) {
    let rows: Vec<f64> = got.chunks(dim).zip(want.chunks(dim)).map(|(g, w)| cosine(g, w)).collect();
    let mut start = 0;
    for r in 1..=rows.len() {
        let band = |c: f64| (c >= 0.9999) as u8 + (c >= 0.99) as u8 + (c >= 0.9) as u8;
        if r == rows.len() || band(rows[r]) != band(rows[start]) {
            let run = &rows[start..r];
            let min = run.iter().copied().fold(f64::INFINITY, f64::min);
            eprintln!("{label}: rows {start}..{r} row cosine min {min:.6} mean {:.6}", run.iter().sum::<f64>() / run.len() as f64);
            start = r;
        }
    }
}

/// the model store's copy, else the repo-relative
/// `resources/ltxv/weights/model_patches/` the real file ships under.
fn weights_path() -> Option<String> {
    if let Some(p) = ltxv::spec::default_store_paths().and_then(|p| p.duration_head) {
        return Some(p);
    }
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../resources/ltxv/weights/model_patches/ltx-2.5-duration-head-bf16.safetensors"
    );
    Path::new(p).exists().then(|| p.to_string())
}

struct Fixture {
    t: Vec<checkpoint::safetensors::StTensor>,
}

impl Fixture {
    fn get(&self, name: &str) -> &[f32] {
        &self.t.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("no golden {name}")).data
    }
}

#[test]
fn ltxv_duration_head_matches_reference() {
    let Some(wp) = weights_path() else {
        brain_testutil::skip("no ltx-2.5-duration-head-bf16.safetensors in the model store");
        return;
    };
    let fx_path = brain_testutil::testdata("golden/ltxv/duration_head/duration_head.safetensors");
    if !Path::new(&fx_path).exists() {
        brain_testutil::skip(&format!("fixture {fx_path} absent - run tools/goldens/ltxv_duration_head_dump_reference.py"));
        return;
    }

    let cfg = DurationHeadConfig::ltx25();
    let raw = checkpoint::safetensors::read(&wp).expect("read real duration head weights");
    let w = import_duration_head(raw, &cfg).expect("import real duration head weights");
    let fx = Fixture { t: checkpoint::safetensors::read(&fx_path).expect("read golden") };

    let head = DurationHead::new(cfg, &w);
    let taps = head.forward_taps(Some(fx.get("video_tokens")), Some(fx.get("audio_tokens")));

    report("video_proj", taps.video_proj.as_deref().unwrap(), fx.get("video_proj"), 0.999999);
    report("audio_proj", taps.audio_proj.as_deref().unwrap(), fx.get("audio_proj"), 0.999999);
    report("tokens", &taps.tokens, fx.get("tokens"), 0.999999);
    report("pooled", &taps.pooled, fx.get("pooled"), 0.999999);
    report("hidden", &taps.hidden, fx.get("hidden"), 0.999999);

    let want_duration = fx.get("duration")[0];
    let rel = ((taps.duration - want_duration).abs() / want_duration.abs().max(1e-6)) as f64;
    eprintln!("duration: got={:.6} want={:.6} rel={rel:.3e}", taps.duration, want_duration);
    assert!(rel < 1e-5, "duration relative error {rel:.3e} too large ({} vs {want_duration})", taps.duration);
}

/// The importer against the REAL shipped file, both directions.
#[test]
fn ltxv_duration_head_import_covers_the_shipped_checkpoint() {
    let Some(wp) = weights_path() else {
        brain_testutil::skip("no ltx-2.5-duration-head-bf16.safetensors in the model store");
        return;
    };
    let cfg = DurationHeadConfig::ltx25();
    let raw = checkpoint::safetensors::read(&wp).expect("read real duration head weights");
    let n = raw.len();
    let manifest = cfg.tensor_manifest();
    let w = import_duration_head(raw, &cfg).expect("import real duration head weights");
    assert_eq!(n, 15, "shipped checkpoint has {n} tensors, expected 15");
    assert_eq!(w.len(), manifest.len());
}

/// The head as a generation that names no length runs it, against
/// `tools/goldens/ltxv_duration_e2e_dump_reference.py`: the real text
/// encoder and both real connectors feed the real head, for a short and a
/// long prompt, on every stream set a pipeline passes. Each stage is checked
/// on the reference's own input before the next - the text encoder's
/// features, the connectors on the reference's features, the whole
/// encodings, then the predictions - so a divergence is placed in one stage.
#[test]
#[ignore = "needs the real duration head, 22B DiT and Gemma-4 text encoder in the model store"]
fn a_prompt_predicts_the_reference_clip_length_on_real_weights() {
    use ltxv::config::Stream;
    let Some(paths) = ltxv::spec::default_store_paths() else {
        brain_testutil::skip("the model store does not resolve LTX-2.5");
        return;
    };
    let fx_path = brain_testutil::testdata("golden/ltxv/duration_e2e/duration_e2e.safetensors");
    if !Path::new(&fx_path).exists() {
        brain_testutil::skip(&format!("fixture {fx_path} absent - run tools/goldens/ltxv_duration_e2e_dump_reference.py"));
        return;
    }
    let o = ltxv::pipeline::GenOpts { dit_config: "ltx25_22b".into(), audio: true, ..ltxv::pipeline::GenOpts::default() };
    if !ltxv::pipeline::can_predict_frames(&paths, &o) {
        brain_testutil::skip("the model store lacks the duration head, the real DiT or the text encoder");
        return;
    }
    let fx = Fixture { t: checkpoint::safetensors::read(&fx_path).expect("read golden") };
    let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(Path::new(&fx_path).with_file_name("manifest.json")).expect("read manifest")).expect("parse manifest");
    let prompts = manifest["run"]["prompts"].as_array().expect("manifest lists its prompts");
    for (i, prompt) in prompts.iter().enumerate() {
        let prompt = prompt.as_str().expect("a prompt is a string");
        // The text encoder: the same tokens, the same features in the
        // prompt's own rows (the padding rows are the connector's to fill).
        let text = ltxv::pipeline::prompt_context(&paths, prompt, &o).expect("prompt_context");
        let want_mask = fx.get(&format!("prompt{i}_mask"));
        assert_eq!(text.valid, want_mask, "prompt {i}: the prompt is framed at other positions than the reference's ({} vs {} tokens)", text.valid.iter().sum::<f32>(), want_mask.iter().sum::<f32>());
        let masked = |rows: &[f32], dim: usize| rows.chunks(dim).zip(&text.valid).flat_map(|(r, &v)| r.iter().map(move |x| x * v)).collect::<Vec<f32>>();
        report(&format!("prompt {i} video features"), &masked(&text.cond, 4096), fx.get(&format!("prompt{i}_video_features")), 0.9999);
        report(&format!("prompt {i} audio features"), &masked(text.a_cond.as_deref().unwrap(), 2048), fx.get(&format!("prompt{i}_audio_features")), 0.9999);
        // The connectors alone, on the reference's own features.
        for (stream, key, dim) in [(Stream::Video, "video", 4096), (Stream::Audio, "audio", 2048)] {
            let routed = ltxv::pipeline::route_prompt(&paths, &o, stream, fx.get(&format!("prompt{i}_{key}_features")), want_mask).expect("route_prompt");
            let want = fx.get(&format!("prompt{i}_{key}_encoding"));
            report_rows(&format!("prompt {i} {key} connector on reference features"), &routed, want, dim);
            report(&format!("prompt {i} {key} connector on reference features"), &routed, want, 0.9999);
        }
        let (video, audio) = ltxv::pipeline::prompt_encodings(&paths, prompt, &o, &[Stream::Video, Stream::Audio]).expect("prompt_encodings");
        let (video, audio) = (video.unwrap(), audio.unwrap());
        let (want_v, want_a) = (fx.get(&format!("prompt{i}_video_encoding")), fx.get(&format!("prompt{i}_audio_encoding")));
        report_rows(&format!("prompt {i} video encoding"), &video, want_v, 4096);
        report_rows(&format!("prompt {i} audio encoding"), &audio, want_a, 2048);
        report(&format!("prompt {i} video encoding"), &video, want_v, 0.9999);
        report(&format!("prompt {i} audio encoding"), &audio, want_a, 0.9999);
        for (key, streams) in [("av", &[Stream::Video, Stream::Audio][..]), ("video", &[Stream::Video]), ("audio", &[Stream::Audio])] {
            let got = ltxv::pipeline::predict_seconds(&paths, prompt, &o, streams).expect("predict_seconds");
            let want = fx.get(&format!("prompt{i}_seconds_{key}"))[0] as f64;
            eprintln!("prompt {i} {key}: {got:.4} s, reference {want:.4} s");
            assert!((got - want).abs() / want < 0.02, "prompt {i} {key}: predicted {got:.4} s, the reference {want:.4} s");
        }
    }
}
