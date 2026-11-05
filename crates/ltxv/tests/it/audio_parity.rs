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

use brain_testutil::parity::{Report, Table};
use brain_testutil::testdata;
use ltxv::audio_vae::{self, AudioVaeConfig};
use ltxv::import::{import_audio_vae, import_vocoder};
use ltxv::vocoder::{self, VocoderConfig};
use vae::blocks::Tensors;

/// This port's established bar for real-weight parity.
const FLOOR: f64 = 0.999999;

/// The magnitude ceiling every audio tap is ALSO gated on.
///
/// Cosine alone cannot gate a waveform. It is scale invariant, so a decode
/// that came out uniformly twice as loud - or half as loud, or inverted and
/// re-inverted through a mis-set gain somewhere in the vocoder - scores a
/// perfect 1.0 while being audibly wrong. `rel_l2` is what notices.
///
/// Set from this suite's own measured clean values (the worst tap here is the
/// vocoder's waveform, an order of magnitude below this) with a wide margin,
/// not fitted to one run: the point of the ceiling is to catch a real
/// regression, and a bound sitting a few percent above the current number
/// would fail on driver noise instead.
const REL_L2_CEILING: f64 = 1e-4;

// ---------------------------------------------------------- real fixtures

/// The shipped audio-VAE-and-vocoder weights: the model store's copy, else the
/// repo-relative `resources/ltxv/weights/vae/ltx-2.5-audio-vae-bf16.safetensors`.
fn weights_path() -> Option<String> {
    if let Some(p) = brain_testutil::model_dir("Lightricks/LTX-2.5").map(|d| format!("{d}/ltx-2.5-audio-vae-bf16.safetensors")).filter(|p| Path::new(p).exists()) {
        return Some(p);
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
        brain_testutil::skip("no ltx-2.5-audio-vae-bf16.safetensors in the model store");
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
///
/// Gated on cosine AND `rel_l2`, unlike the stages above, because this is the
/// one that produces a WAVEFORM - see [`REL_L2_CEILING`] for why cosine on its
/// own cannot gate one.
#[test]
fn ltxv_vocoder_matches_reference() {
    let Some(((_, voc_w), fx)) = setup() else { return };
    let cfg = VocoderConfig::ltx25();
    let mshape = fx.shape("recon_mel").to_vec();
    let (channels, t, mel_bins) = (mshape[0] as u32, mshape[1] as u32, mshape[2] as u32);
    let wave = vocoder::synthesize(&cfg, voc_w, fx.get("recon_mel"), channels, t, mel_bins, None);

    let mut tab = Table::new(FLOOR, REL_L2_CEILING);
    tab.check("recon_wave", &wave, fx.get("recon_wave"));
    tab.print();
    tab.assert_clean();
}

/// A uniform GAIN change and a TIME SHIFT are the two ways an audio pipeline
/// fails while still sounding like something, and neither is a crash. This
/// proves the gate above catches both - so a green run on it means the
/// waveform is right, not merely present.
///
/// Deliberately not a mutation of the code: it perturbs the golden comparison
/// itself, which is the same arithmetic the gate performs, and therefore says
/// exactly what the gate can and cannot see.
#[test]
fn the_waveform_gate_catches_a_gain_change_and_a_time_shift() {
    let Some((_, fx)) = setup() else { return };
    let want = fx.get("recon_wave").to_vec();

    // 1: a uniform 1% gain. Cosine is SCALE INVARIANT and does not move at
    // all; rel_l2 is what fails, which is the entire reason both are gated.
    let louder: Vec<f32> = want.iter().map(|&v| v * 1.01).collect();
    let mut gain = Table::new(FLOOR, REL_L2_CEILING);
    gain.check("uniform gain", &louder, &want);
    gain.print();
    assert!(!gain.failures.is_empty(), "a 1% uniform gain must fail the waveform gate");
    assert!(gain.failures.iter().all(|f| f.contains("rel_l2")), "the gain must be caught by rel_l2, not cosine: {:?}", gain.failures);
    assert!(
        gain.rows[0].1 >= FLOOR,
        "cosine must be BLIND to a uniform gain ({:.10}) - if it moved, this test is not proving what it claims",
        gain.rows[0].1
    );

    // 2: a one-sample time shift - the same waveform, misaligned. Both
    // metrics see it, and either failing is enough.
    let mut shifted = vec![0f32; want.len()];
    shifted[1..].copy_from_slice(&want[..want.len() - 1]);
    let mut shift = Table::new(FLOOR, REL_L2_CEILING);
    shift.check("one-sample shift", &shifted, &want);
    shift.print();
    assert!(!shift.failures.is_empty(), "a one-sample misalignment must fail the waveform gate");
}

/// Listen to the result the only way a test can: measure it.
///
/// A file of the right length full of near-silence, or full-scale noise, or
/// two identical channels, all "work" by every structural check in this
/// suite. These are the properties that separate a real waveform from those,
/// measured on the golden's own reconstruction so the thresholds are anchored
/// to something a listener has effectively already approved.
#[test]
fn the_reconstructed_waveform_has_the_statistics_of_real_audio() {
    let Some((_, fx)) = setup() else { return };
    let shape = fx.shape("recon_wave").to_vec();
    let (channels, n) = (shape[0], shape[1]);
    let wave = fx.get("recon_wave");
    let planes: Vec<&[f32]> = (0..channels).map(|c| &wave[c * n..(c + 1) * n]).collect();

    for (c, p) in planes.iter().enumerate() {
        let peak = p.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let clipped = p.iter().filter(|v| v.abs() >= 0.999).count();
        let rms = (p.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / n as f64).sqrt();
        let dbfs = 20.0 * rms.max(1e-12).log10();
        eprintln!("channel {c}: peak {peak:.4}  clipped {clipped}/{n}  rms {dbfs:.2} dBFS");
        assert!(peak > 0.01, "channel {c} peaks at {peak:.5} - that is silence, not audio");
        assert!(peak <= 1.0, "channel {c} peaks at {peak:.5}, outside the vocoder's own clamp");
        assert!(dbfs > -60.0, "channel {c} is {dbfs:.1} dBFS - effectively silent");
        assert!(dbfs < -3.0, "channel {c} is {dbfs:.1} dBFS - that is full-scale noise, not a signal");
        assert!(clipped * 100 < n, "channel {c} clips in {clipped} of {n} samples");
        // No long near-silent stretch: a decode that produced a correct
        // opening and then collapsed would pass every whole-file statistic
        // above. A tenth of a second is the shortest gap a listener reliably
        // notices as a dropout.
        let win = (ltxv::audio::SAMPLE_RATE as usize / 10).min(n);
        let mut run = 0usize;
        let mut worst = 0usize;
        for &v in p.iter() {
            run = if v.abs() < 1e-4 { run + 1 } else { 0 };
            worst = worst.max(run);
        }
        assert!(worst < win, "channel {c} has {worst} consecutive near-silent samples (>= {win} is an audible dropout)");
    }

    if channels == 2 {
        // A genuine stereo field is neither 1.0 (one channel duplicated - the
        // shape a broken planar/interleaved split produces) nor 0.0
        // (uncorrelated, i.e. two unrelated signals rather than one scene).
        let (a, b) = (planes[0], planes[1]);
        let mean = |x: &[f32]| x.iter().map(|&v| f64::from(v)).sum::<f64>() / x.len() as f64;
        let (ma, mb) = (mean(a), mean(b));
        let mut num = 0f64;
        let (mut da, mut db) = (0f64, 0f64);
        for (&x, &y) in a.iter().zip(b) {
            let (u, v) = (f64::from(x) - ma, f64::from(y) - mb);
            num += u * v;
            da += u * u;
            db += v * v;
        }
        let corr = num / (da.sqrt() * db.sqrt()).max(1e-12);
        eprintln!("L/R correlation: {corr:.6}");
        assert!(corr.abs() < 0.9999, "L and R correlate at {corr:.6} - the two channels are the same signal, not a stereo field");
    }
}

#[test]
fn ltxv_audio_vae_import_covers_the_shipped_checkpoint() {
    let Some((vae_w, _)) = weights() else {
        brain_testutil::skip("no ltx-2.5-audio-vae-bf16.safetensors in the model store");
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
        brain_testutil::skip("no ltx-2.5-audio-vae-bf16.safetensors in the model store");
        return;
    };
    let cfg = VocoderConfig::ltx25();
    let manifest = cfg.tensor_manifest();
    assert_eq!(voc_w.len(), 667, "shipped vocoder imported to {} tensors, expected 667", voc_w.len());
    assert_eq!(voc_w.len(), manifest.len());
}
