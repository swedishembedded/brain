// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LTX-2.5's 48 kHz audio: the vocoder's bandwidth-extension (BWE) stage,
//! against `tools/goldens/ltxv_bwe_dump_reference.py`'s real-weight golden
//! (the official `VocoderWithBWE` and `decode_audio`).
//!
//! Climbed in the order failures localise:
//!
//! 1. [`ltxv_bwe_stages_match_reference`] - the golden's own 16 kHz base
//!    waveform through the causal checkpoint-basis STFT log-mel, the BWE
//!    generator's residual, the x3 Hann-sinc skip, and their clamped sum.
//! 2. [`ltxv_vocoder_with_bwe_matches_reference`] - `mel -> 48 kHz`, the base
//!    vocoder and the BWE composed.
//! 3. [`ltxv_audio_decode_chain_is_48k_and_matches_reference`] - a DiT audio
//!    latent all the way to the 48 kHz waveform through `ltxv::audio::decode`,
//!    the function a generation calls.
//! 4. [`ltxv_bwe_import_covers_the_shipped_checkpoint`] - the importer against
//!    the real file.
//!
//! Every waveform tap is gated on cosine AND `rel_l2`: cosine is scale
//! invariant, so a gain error would pass it.

use std::path::Path;
use std::sync::OnceLock;

use brain_testutil::parity::Table;
use brain_testutil::testdata;
use ltxv::audio::{self, VocoderWeights};
use ltxv::audio_vae::AudioVaeConfig;
use ltxv::import::{import_audio_vae, import_vocoder_with_bwe};
use ltxv::vocoder::{self, BweConfig, VocoderConfig};
use vae::blocks::Tensors;

const FLOOR: f64 = 0.999999;
/// Every WAVEFORM a listener gets (the 48 kHz output, the whole decode chain)
/// and the STFT front end: this port's audio bar, with an order of magnitude
/// of margin over the measured values.
const REL_L2_CEILING: f64 = 1e-4;
/// The BWE generator's RESIDUAL is gated looser, by measurement rather than
/// by taste: it is a high-band signal some 40 dB below the output it is
/// added to, so its relative error is taken against a small norm, and the
/// generator is sensitive to its input - fed the golden's own mel it lands
/// at rel_l2 ~1e-4 (the f32 rounding of ~100 sequential convolutions),
/// fed this port's mel (itself 3e-7 from the golden) at ~8e-4. Neither is
/// audible: the summed output stays at ~3e-5. 5e-3 keeps the residual's own
/// regressions visible (a wrong tap or a missing stage is O(1)) without
/// failing on that measured noise floor.
const RESIDUAL_FLOOR: f64 = 0.99999;
const RESIDUAL_REL_L2_CEILING: f64 = 5e-3;
/// The whole decode chain starts two stages further upstream (the audio VAE
/// decoder, whose mel is ~4e-7 from the golden's), and that difference
/// reaches the output through the same input-sensitive BWE generator:
/// measured 7.4e-5 on the CPU backend and 2.5e-4 on a P40, where every stage
/// fed the golden's own input stays under 1e-5. 1e-3 (-60 dB) is what this
/// gate is for - a composition error (a transposed latent, a wrong length or
/// gain) is O(1) - with headroom over the measured backend spread.
const CHAIN_FLOOR: f64 = 0.99999;
const CHAIN_REL_L2_CEILING: f64 = 1e-3;

/// The audio checkpoint from the model store (`Lightricks/LTX-2.5`).
fn weights_path() -> Option<String> {
    let p = format!("{}/ltx-2.5-audio-vae-bf16.safetensors", brain_testutil::model_dir("Lightricks/LTX-2.5")?);
    Path::new(&p).exists().then_some(p)
}

static WEIGHTS: OnceLock<Option<(Tensors, VocoderWeights)>> = OnceLock::new();

fn weights() -> Option<&'static (Tensors, VocoderWeights)> {
    WEIGHTS
        .get_or_init(|| {
            let wp = weights_path()?;
            let (voc_raw, vae_raw): (Vec<_>, Vec<_>) = checkpoint::safetensors::read(&wp).ok()?.into_iter().partition(|t| t.name.starts_with("vocoder."));
            let vae_w = import_audio_vae(vae_raw, &AudioVaeConfig::ltx25()).expect("import audio vae");
            let voc = import_vocoder_with_bwe(voc_raw).expect("import vocoder + bwe");
            Some((vae_w, voc))
        })
        .as_ref()
}

struct Fixture(Vec<checkpoint::safetensors::StTensor>);

impl Fixture {
    fn open(name: &str) -> Option<Fixture> {
        let fx = testdata(&format!("golden/ltxv/bwe/{name}"));
        if !Path::new(&fx).exists() {
            brain_testutil::skip(&format!("fixture {fx} absent - run tools/goldens/ltxv_bwe_dump_reference.py"));
            return None;
        }
        Some(Fixture(checkpoint::safetensors::read(&fx).expect("read golden")))
    }
    fn get(&self, name: &str) -> &[f32] {
        &self.0.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("no golden {name}")).data
    }
    fn shape(&self, name: &str) -> &[usize] {
        &self.0.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("no golden {name}")).shape
    }
}

fn setup(fixture: &str) -> Option<(&'static (Tensors, VocoderWeights), Fixture)> {
    let fx = Fixture::open(fixture)?;
    let Some(w) = weights() else {
        brain_testutil::skip("no ltx-2.5-audio-vae-bf16.safetensors in the model store (Lightricks/LTX-2.5)");
        return None;
    };
    Some((w, fx))
}

#[test]
fn ltxv_bwe_stages_match_reference() {
    let Some(((_, voc), fx)) = setup("bwe.safetensors") else { return };
    let base = fx.get("base_wave");
    let channels = fx.shape("base_wave")[0] as u32;
    let (out, taps) = vocoder::extend_bandwidth(&BweConfig::ltx25(), &voc.bwe, base, channels, None);

    // The generator alone, fed the GOLDEN's own mel, so a generator error
    // cannot hide behind (or be blamed on) the STFT front end.
    let cfg = BweConfig::ltx25();
    let mshape = fx.shape("bwe_mel").to_vec();
    let (m, frames) = (mshape[1], mshape[2]);
    let gm = fx.get("bwe_mel");
    let mut mel_t = vec![0f32; gm.len()];
    for c in 0..channels as usize {
        for mi in 0..m {
            for fi in 0..frames {
                mel_t[(c * frames + fi) * m + mi] = gm[(c * m + mi) * frames + fi];
            }
        }
    }
    let residual_alone = vocoder::synthesize(&cfg.generator, &voc.bwe.generator, &mel_t, channels, frames as u32, m as u32, None);

    let mut residual = Table::new(RESIDUAL_FLOOR, RESIDUAL_REL_L2_CEILING);
    residual.check("residual (golden mel in)", &residual_alone, fx.get("residual"));
    residual.check("residual", &taps.residual, fx.get("residual"));
    residual.print();
    residual.assert_clean();

    let mut tab = Table::new(FLOOR, REL_L2_CEILING);
    tab.check("bwe_mel", &taps.mel, fx.get("bwe_mel"));
    tab.check("skip", &taps.skip, fx.get("skip"));
    tab.check("out (48 kHz)", &out, fx.get("out"));
    tab.print();
    tab.assert_clean();
}

#[test]
fn ltxv_vocoder_with_bwe_matches_reference() {
    let Some(((_, voc), fx)) = setup("bwe.safetensors") else { return };
    let shape = fx.shape("mel_in").to_vec();
    let (channels, t, mel_bins) = (shape[0] as u32, shape[1] as u32, shape[2] as u32);
    let wave = vocoder::synthesize_with_bwe(&VocoderConfig::ltx25(), voc, &BweConfig::ltx25(), fx.get("mel_in"), channels, t, mel_bins, None);

    let mut tab = Table::new(FLOOR, REL_L2_CEILING);
    tab.check("mel -> 48 kHz", &wave, fx.get("out"));
    tab.print();
    tab.assert_clean();
}

#[test]
fn ltxv_audio_decode_chain_is_48k_and_matches_reference() {
    let Some(((vae_w, voc), fx)) = setup("chain.safetensors") else { return };
    let lshape = fx.shape("latent").to_vec();
    let (c, lt, f) = (lshape[0], lshape[1], lshape[2]);
    // The golden latent is channel-major `[c, t, f]`; the DiT hands
    // `audio::decode` a token-major `[t, c*f]` one.
    let g = fx.get("latent");
    let mut tokens = vec![0f32; lt * c * f];
    for t in 0..lt {
        for ci in 0..c {
            for fi in 0..f {
                tokens[t * c * f + ci * f + fi] = g[ci * lt * f + t * f + fi];
            }
        }
    }
    let clip = audio::decode(vae_w, voc, &tokens, lt, None);
    assert_eq!(clip.sample_rate, 48_000, "LTX-2.5 decodes 48 kHz audio");
    assert_eq!(clip.channels.len(), 2, "the model generates stereo");

    let got: Vec<f32> = clip.channels.concat();
    let want = fx.get("recon_wave_48k");
    assert_eq!(got.len(), want.len(), "the chain produced {} samples, the golden has {}", got.len(), want.len());
    let mut tab = Table::new(CHAIN_FLOOR, CHAIN_REL_L2_CEILING);
    tab.check("latent -> 48 kHz waveform", &got, want);
    tab.print();
    tab.assert_clean();
}

#[test]
fn ltxv_bwe_import_covers_the_shipped_checkpoint() {
    let Some((_, voc)) = weights() else {
        brain_testutil::skip("no ltx-2.5-audio-vae-bf16.safetensors in the model store (Lightricks/LTX-2.5)");
        return;
    };
    let m = BweConfig::ltx25().generator.tensor_manifest();
    assert_eq!(voc.bwe.generator.len(), m.len(), "every BWE generator tensor imported");
    for (name, shape) in &m {
        assert_eq!(&voc.bwe.generator[name].0, shape, "{name}");
    }
    let cfg = BweConfig::ltx25();
    let freqs = (cfg.n_fft / 2 + 1) as usize;
    assert_eq!(voc.bwe.mel_basis.len(), cfg.num_mels as usize * freqs);
    assert_eq!(voc.bwe.forward_basis.len(), 2 * freqs * cfg.n_fft as usize);
}
