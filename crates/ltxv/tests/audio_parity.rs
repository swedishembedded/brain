// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LTX-2.5 audio VAE + base vocoder parity, against
//! `tools/goldens/ltxv_audio_dump_reference.py`'s real-weight golden.
//!
//! Climbed in the order failures localise, mirroring `vae_parity.rs`'s shape:
//!
//! 1. [`ltxv_audio_encoder_matches_reference`] - `mel -> latent`.
//! 2. [`ltxv_audio_decoder_matches_reference`] - `latent -> recon_mel`,
//!    fed the GOLDEN's own latent (stage parity, not composed).
//! 3. [`ltxv_audio_vae_round_trip`] - `mel -> encode -> decode`, this crate's
//!    own composition of the two stages, against the golden's `recon_mel`.
//! 4. [`ltxv_vocoder_matches_reference`] - `recon_mel -> recon_wave`, fed the
//!    GOLDEN's own `recon_mel` (stage parity for the vocoder alone, not
//!    composed with the VAE decode above - a vocoder bug and a decoder bug
//!    must not be able to hide behind each other).
//! 5. [`ltxv_audio_vae_import_covers_the_shipped_checkpoint`] /
//!    [`ltxv_vocoder_import_covers_the_shipped_checkpoint`] - both importers
//!    against the REAL file, both directions.
//!
//! Everything here needs the real audio VAE weights (which also carry the
//! vocoder - one file, see `crate::import`'s module doc) and the golden
//! fixture, and skips loudly without them (`BRAIN_REQUIRE_FIXTURES=1`
//! upgrades a skip to a failure, same convention as every other parity suite
//! in this repo).

use std::path::Path;
use std::sync::OnceLock;

use brain_testutil::parity::Report;
use brain_testutil::testdata;
use ltxv::audio_vae::{self, AudioVaeConfig};
use ltxv::import::{import_audio_vae, import_vocoder};
use ltxv::vocoder::{self, VocoderConfig};
use vae::blocks::Tensors;

/// This port's established bar for real-weight parity.
const FLOOR: f64 = 0.999999;

// ---------------------------------------------------------- real fixtures

/// The shipped audio-VAE-and-vocoder weights: `BRAIN_LTXV_AUDIO_VAE`, else the
/// repo-relative `resources/ltxv/weights/vae/ltx-2.5-audio-vae-bf16.safetensors`.
fn weights_path() -> Option<String> {
    if let Ok(p) = std::env::var("BRAIN_LTXV_AUDIO_VAE") {
        return (!p.is_empty() && Path::new(&p).exists()).then_some(p);
    }
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../resources/ltxv/weights/vae/ltx-2.5-audio-vae-bf16.safetensors");
    Path::new(p).exists().then(|| p.to_string())
}

/// The imported audio-VAE and vocoder weights, decoded exactly ONCE for the
/// whole test binary (`OnceLock`) - same rationale `vae_parity.rs` states for
/// its own weight cache: several tests borrowing one shared copy instead of
/// each re-reading/re-importing the file.
static WEIGHTS: OnceLock<Option<(Tensors, Tensors)>> = OnceLock::new();

fn weights() -> Option<(&'static Tensors, &'static Tensors)> {
    WEIGHTS
        .get_or_init(|| {
            let wp = weights_path()?;
            // `StTensor` is not `Clone` (each holds its own owned `Vec<f32>`
            // data), and each import only needs its own tensor subset anyway
            // (see `crate::import`'s module doc) - two reads of the same file
            // rather than one read plus a full-file clone.
            let vae_w = import_audio_vae(checkpoint::safetensors::read(&wp).ok()?, &AudioVaeConfig::ltx25()).ok()?;
            let voc_w = import_vocoder(checkpoint::safetensors::read(&wp).ok()?, &VocoderConfig::ltx25()).ok()?;
            eprintln!("imported audio vae ({} tensors) + vocoder ({} tensors) from {wp}", vae_w.len(), voc_w.len());
            Some((vae_w, voc_w))
        })
        .as_ref()
        .map(|(a, b)| (a, b))
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

/// `(vae weights, vocoder weights, fixture)` or `None` with a loud skip.
fn setup() -> Option<((&'static Tensors, &'static Tensors), Fixture)> {
    let fx = testdata("golden/ltxv/audio/audio.safetensors");
    if !Path::new(&fx).exists() {
        brain_testutil::skip(&format!("fixture {fx} absent - run tools/goldens/ltxv_audio_dump_reference.py"));
        return None;
    }
    let Some(w) = weights() else {
        brain_testutil::skip("set BRAIN_LTXV_AUDIO_VAE to ltx-2.5-audio-vae-bf16.safetensors");
        return None;
    };
    let t = checkpoint::safetensors::read(&fx).expect("read golden");
    Some((w, Fixture { t }))
}

#[test]
fn ltxv_audio_encoder_matches_reference() {
    let Some(((vae_w, _), fx)) = setup() else { return };
    let cfg = AudioVaeConfig::ltx25();
    let mshape = fx.shape("mel").to_vec();
    let (t, fbins) = (mshape[1] as u32, mshape[2] as u32);
    let latent = audio_vae::encode(&cfg, vae_w, fx.get("mel"), t, fbins, None);

    let mut r = Report::new(FLOOR);
    r.check("latent", &latent, fx.get("latent"));
    r.finish("audio encoder");
}

#[test]
fn ltxv_audio_decoder_matches_reference() {
    let Some(((vae_w, _), fx)) = setup() else { return };
    let cfg = AudioVaeConfig::ltx25();
    let lshape = fx.shape("latent").to_vec();
    let (lt, lf) = (lshape[1] as u32, lshape[2] as u32);
    let recon_mel = audio_vae::decode(&cfg, vae_w, fx.get("latent"), lt, lf, None);

    let mut r = Report::new(FLOOR);
    r.check("recon_mel", &recon_mel, fx.get("recon_mel"));
    r.finish("audio decoder");
}

/// Encode then decode in one composed run, against the reference's own
/// `latent = encoder(mel); recon_mel = decoder(latent)` composition.
#[test]
fn ltxv_audio_vae_round_trip() {
    let Some(((vae_w, _), fx)) = setup() else { return };
    let cfg = AudioVaeConfig::ltx25();
    let mshape = fx.shape("mel").to_vec();
    let (t, fbins) = (mshape[1] as u32, mshape[2] as u32);
    let latent = audio_vae::encode(&cfg, vae_w, fx.get("mel"), t, fbins, None);
    let lshape = fx.shape("latent").to_vec();
    let (lt, lf) = (lshape[1] as u32, lshape[2] as u32);
    assert_eq!(latent.len(), fx.get("latent").len());
    let recon_mel = audio_vae::decode(&cfg, vae_w, &latent, lt, lf, None);

    let mut r = Report::new(FLOOR);
    r.check("latent (round trip)", &latent, fx.get("latent"));
    r.check("recon_mel (round trip)", &recon_mel, fx.get("recon_mel"));
    r.finish("audio vae round trip");
}

/// The base vocoder ALONE, fed the golden's own `recon_mel` (stage parity,
/// not composed with this crate's own VAE decode above).
#[test]
fn ltxv_vocoder_matches_reference() {
    let Some(((_, voc_w), fx)) = setup() else { return };
    let cfg = VocoderConfig::ltx25();
    let mshape = fx.shape("recon_mel").to_vec();
    let (channels, t, mel_bins) = (mshape[0] as u32, mshape[1] as u32, mshape[2] as u32);
    let wave = vocoder::synthesize(&cfg, voc_w, fx.get("recon_mel"), channels, t, mel_bins, None);

    let mut r = Report::new(FLOOR);
    r.check("recon_wave", &wave, fx.get("recon_wave"));
    r.finish("vocoder");
}

#[test]
fn ltxv_audio_vae_import_covers_the_shipped_checkpoint() {
    let Some((vae_w, _)) = weights() else {
        brain_testutil::skip("set BRAIN_LTXV_AUDIO_VAE to ltx-2.5-audio-vae-bf16.safetensors");
        return;
    };
    let cfg = AudioVaeConfig::ltx25();
    let manifest = cfg.tensor_manifest();
    assert_eq!(vae_w.len(), 102, "shipped audio vae imported to {} tensors, expected 102", vae_w.len());
    assert_eq!(vae_w.len(), manifest.len());
}

#[test]
fn ltxv_vocoder_import_covers_the_shipped_checkpoint() {
    let Some((_, voc_w)) = weights() else {
        brain_testutil::skip("set BRAIN_LTXV_AUDIO_VAE to ltx-2.5-audio-vae-bf16.safetensors");
        return;
    };
    let cfg = VocoderConfig::ltx25();
    let manifest = cfg.tensor_manifest();
    assert_eq!(voc_w.len(), 667, "shipped vocoder imported to {} tensors, expected 667", voc_w.len());
    assert_eq!(voc_w.len(), manifest.len());
}
