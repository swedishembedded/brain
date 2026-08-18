// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LTX-2.5 NA ("diffusion") video VAE decoder parity, climbed in the order
//! failures localise:
//!
//! 1. [`ltxv_na_context_matches_reference`] - stages 1-4 (the deterministic
//!    context path), real weights, against
//!    `tools/goldens/ltxv_na_decoder_dump_reference.py`'s `na_context.
//!    safetensors` (stage4_input + final context taps).
//! 2. [`ltxv_na_diff_matches_reference`] - one stage-5 forward
//!    (`CombinedDiffusionNABlock` x8, single-step x0 prediction), real
//!    weights, against the same dumper's DECOUPLED `na_diff.safetensors` tap
//!    (see that script's module doc for why the diffusion tap uses a
//!    smaller synthetic context than the real chained (17,56,56) one).
//! 3. [`ltxv_na_decoder_import_covers_the_shipped_checkpoint`] - the
//!    importer against the REAL checkpoint file, both directions.
//!
//! Everything here needs the real weights and the golden fixture and skips
//! loudly without them (`BRAIN_REQUIRE_FIXTURES=1` upgrades a skip to a
//! failure, same convention as every other parity suite in this repo).

use std::path::Path;
use std::sync::OnceLock;

use brain_testutil::testdata;
use ltxv::na_decoder::{self, NaDecoderConfig};
use vae::blocks::Tensors;

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

/// The shipped NA decoder weights: `BRAIN_LTXV_NA_VAE`, else the
/// repo-relative
/// `resources/ltxv/weights/vae/ltx-2.5-video-vae-bf16.safetensors` - the
/// file paired with the NA decoder (as opposed to `-conv-bf16`, M2's own
/// conv decoder).
fn weights_path() -> Option<String> {
    if let Ok(p) = std::env::var("BRAIN_LTXV_NA_VAE") {
        return (!p.is_empty() && Path::new(&p).exists()).then_some(p);
    }
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../resources/ltxv/weights/vae/ltx-2.5-video-vae-bf16.safetensors");
    Path::new(p).exists().then(|| p.to_string())
}

struct Fixture {
    t: Vec<checkpoint::safetensors::StTensor>,
}

impl Fixture {
    fn get(&self, name: &str) -> &[f32] {
        &self.t.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("no golden {name}")).data
    }
    fn shape(&self, name: &str) -> &[usize] {
        &self.t.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("no golden {name}")).shape
    }
}

/// The imported weights, decoded ONCE for the whole test binary - the real
/// checkpoint is ~417M decoder parameters (~1.7GB as f32), the same "shared
/// `OnceLock`, never cloned" precedent `vae_parity.rs`'s own `WEIGHTS` sets.
static WEIGHTS: OnceLock<Option<Tensors>> = OnceLock::new();

fn weights() -> Option<&'static Tensors> {
    WEIGHTS
        .get_or_init(|| {
            let wp = weights_path()?;
            let cfg = NaDecoderConfig::ltx25();
            let raw = checkpoint::safetensors::read(&wp).ok()?;
            let n = raw.len();
            let w = na_decoder::import_na_decoder(raw, &cfg).ok()?;
            eprintln!("imported {n} source tensors -> {} canonical from {wp}", w.len());
            Some(w)
        })
        .as_ref()
}

fn setup(fixture_file: &str) -> Option<(&'static Tensors, Fixture)> {
    let fx = testdata(&format!("golden/ltxv/na_decoder/{fixture_file}"));
    if !Path::new(&fx).exists() {
        brain_testutil::skip(&format!("fixture {fx} absent - run tools/goldens/ltxv_na_decoder_dump_reference.py"));
        return None;
    }
    let Some(w) = weights() else {
        brain_testutil::skip("set BRAIN_LTXV_NA_VAE to ltx-2.5-video-vae-bf16.safetensors");
        return None;
    };
    let t = checkpoint::safetensors::read(&fx).expect("read golden");
    Some((w, Fixture { t }))
}

#[test]
fn ltxv_na_context_matches_reference() {
    let Some((w, fx)) = setup("na_context.safetensors") else { return };
    let cfg = NaDecoderConfig::ltx25();
    let gpu = na_decoder::open_device(None);

    let ls = fx.shape("latent").to_vec();
    let (t, h, wd) = (ls[1] as u32, ls[2] as u32, ls[3] as u32);
    let (stage4_input, t3, h3, w3) = na_decoder::forward_stages_1_to_3(&gpu, w, &cfg, fx.get("latent"), t, h, wd);
    report("stage4_input", &stage4_input, fx.get("stage4_input"), 0.999999);

    let (context, t4, h4, w4) = na_decoder::forward_stage_4(&gpu, w, &cfg, &stage4_input, t3, h3, w3);
    report("context", &context, fx.get("context"), 0.999999);

    let want_ctx_shape = fx.shape("context");
    assert_eq!([t4 as usize, h4 as usize, w4 as usize, cfg.context_channels() as usize], want_ctx_shape[..]);
}

#[test]
fn ltxv_na_diff_matches_reference() {
    let Some((w, fx)) = setup("na_diff.safetensors") else { return };
    let cfg = NaDecoderConfig::ltx25();
    let gpu = na_decoder::open_device(None);

    let cs = fx.shape("context").to_vec();
    let (t, h, wd) = (cs[0] as u32, cs[1] as u32, cs[2] as u32);
    let x0_pred = na_decoder::forward_diff(&gpu, w, &cfg, fx.get("context"), t, h, wd, fx.get("x_t"));
    report("x0_pred", &x0_pred, fx.get("x0_pred"), 0.999999);
}

/// The importer against the REAL shipped file, both directions - reuses the
/// process-wide cached [`weights`] for the OK case and
/// [`NaDecoderConfig::tensor_manifest`] for the negative-case stubs, same
/// pattern `vae_parity.rs`'s own coverage test uses.
#[test]
fn ltxv_na_decoder_import_covers_the_shipped_checkpoint() {
    let Some(w) = weights() else {
        brain_testutil::skip("set BRAIN_LTXV_NA_VAE to ltx-2.5-video-vae-bf16.safetensors");
        return;
    };
    let cfg = NaDecoderConfig::ltx25();
    let manifest = cfg.tensor_manifest();
    // Post-qkv-split: +4 tensors per attention block (16 det + 8 diff = 24).
    assert_eq!(w.len(), manifest.len() + 24 * 4, "shipped checkpoint imported to {} tensors", w.len());

    let stub = |skip: Option<&str>, add: Option<&str>| -> Vec<checkpoint::safetensors::StTensor> {
        let mut v: Vec<_> = manifest
            .iter()
            .filter(|(n, _)| Some(n.as_str()) != skip)
            .map(|(n, s)| checkpoint::safetensors::StTensor { name: n.clone(), shape: s.clone(), data: vec![0.0; s.iter().product()] })
            .collect();
        if let Some(a) = add {
            v.push(checkpoint::safetensors::StTensor { name: a.into(), shape: vec![1], data: vec![0.0] });
        }
        v
    };

    let e = na_decoder::import_na_decoder(stub(Some("decoder.norm_out.weight"), None), &cfg).unwrap_err();
    assert!(e.contains("decoder.norm_out.weight"), "{e}");

    let e = na_decoder::import_na_decoder(stub(None, Some("decoder.diff_blocks.99.norm1.weight")), &cfg).unwrap_err();
    assert!(e.contains("unused source tensors"), "{e}");
}
