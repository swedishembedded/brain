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

/// `BRAIN_LTXV_DURATION_HEAD`, else the repo-relative
/// `resources/ltxv/weights/model_patches/` the real file ships under.
fn weights_path() -> Option<String> {
    if let Ok(p) = std::env::var("BRAIN_LTXV_DURATION_HEAD") {
        return (!p.is_empty() && Path::new(&p).exists()).then_some(p);
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
        brain_testutil::skip("set BRAIN_LTXV_DURATION_HEAD to ltx-2.5-duration-head-bf16.safetensors");
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
        brain_testutil::skip("set BRAIN_LTXV_DURATION_HEAD to ltx-2.5-duration-head-bf16.safetensors");
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
