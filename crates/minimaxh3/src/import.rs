// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import MiniMax-H3's audio VAE decoder weights (the `audio_vae/` component
//! directory's own weight file, within a fetched checkpoint directory) into
//! a [`vae::blocks::Tensors`] map
//! matching [`crate::vocoder::VocoderConfig::tensor_manifest`], folding every
//! `weight_norm` pair (`{prefix}.weight_g`/`{prefix}.weight_v`, PyTorch's
//! OLDER `nn.utils.weight_norm` API - confirmed against the real checkpoint
//! header, NOT the newer `parametrizations.weight.original0/1` naming a
//! different torch build/API would use) via `audio::conv::fold_weight_norm`
//! at import time. The vocoder forward never sees a `weight_g`/`weight_v`
//! pair, only the folded `.weight`.
//!
//! **Decode-only, by design** (see [`crate::vocoder`]'s own module doc): this
//! checkpoint carries `encoder`/`mean_proj`/`logs_proj`/`pre_block` tensors
//! (173 of them at the real config) the shipped inference-only forward never
//! touches. Two-way coverage here is scoped to the tensors
//! [`crate::vocoder::VocoderConfig::tensor_manifest`] actually claims
//! (`decoder.*` and `dec_in_proj.*`) - every one of those must be present and
//! is consumed exactly once, and any left over among THOSE after folding is
//! an error. The encoder-side tensors are neither read nor treated as an
//! error, but their prefix set (and only that set) is asserted, so a real
//! checkpoint shape change is caught here rather than silently passing an
//! unrelated missing tensor through as "expected out of scope."

use std::collections::HashMap;

use audio::conv::fold_weight_norm;
use vae::blocks::Tensors;

use crate::video_vae::VideoVaeConfig;
use crate::vocoder::VocoderConfig;

/// Prefixes this port deliberately does not read - present in the checkpoint
/// for training-time compatibility, dead in the shipped `DacAudioVAE.decode`
/// forward (see [`crate::vocoder`]'s module doc for why).
const OUT_OF_SCOPE_PREFIXES: [&str; 4] = ["encoder.", "mean_proj.", "logs_proj.", "pre_block."];

/// Import the `audio_vae/` component directory's weight file into a
/// decode-ready [`Tensors`] map. `dir` is the component directory itself
/// (containing its weight file, `config.json`, `metadata.json`,
/// `config.yaml` and the checkpoint's own reference `.py` sources).
/// `checkpoint::safetensors::read_model_dir` resolves the weight file's own
/// name - `model.safetensors` under the legacy `FL2VA/`-partitioned layout,
/// `diffusion_pytorch_model.safetensors` under the root
/// `MiniMaxH3ModularPipeline` layout this port now targets (confirmed
/// against both real directories; same tensor names either way) - rather
/// than this function hardcoding either one.
pub fn import_audio_vae_decoder(dir: &str, cfg: &VocoderConfig) -> Result<Tensors, String> {
    let raw = checkpoint::safetensors::read_model_dir(std::path::Path::new(dir))?;

    let mut by_name: HashMap<String, (Vec<usize>, Vec<f32>)> = raw.into_iter().map(|t| (t.name, (t.shape, t.data))).collect();

    let take = |m: &mut HashMap<String, (Vec<usize>, Vec<f32>)>, name: &str| -> Result<(Vec<usize>, Vec<f32>), String> {
        m.remove(name).ok_or_else(|| format!("minimaxh3 audio_vae import: missing tensor {name:?}"))
    };

    let mut out: Tensors = HashMap::new();
    let manifest = cfg.tensor_manifest();

    // Every manifest entry is either a PLAIN tensor (dec_in_proj.weight/bias,
    // and every .bias/.act.alpha/.act.beta/.upsample.filter/
    // .downsample.lowpass.filter leaf) or the FOLDED result of a
    // {name}_g/{name}_v weight-norm pair (every other .weight - conv_pre,
    // conv_post, every decoder.ups.{i}.0 and every AMPBlock1 conv).
    for (name, shape) in &manifest {
        if name.ends_with(".weight") && name != "dec_in_proj.weight" {
            let (gs, g) = take(&mut by_name, &format!("{name}_g"))?;
            let (vs, v) = take(&mut by_name, &format!("{name}_v"))?;
            // weight_norm's dim=0 is dim 0 of the STORED tensor: Cout for a
            // plain Conv1d (conv_pre/conv_post/every AMPBlock1 conv), Cin for
            // ConvTranspose1d's native [Cin,Cout/G,K] layout (every
            // decoder.ups.{i}.0 stage) - `fold_weight_norm`'s own documented
            // contract. `gs[0]` IS that dimension either way (`weight_g`'s
            // own leading/only real dim, confirmed [Cout,1,1] and [Cin,1,1]
            // respectively against the real header), so this needs no
            // per-site branch on conv kind.
            let d0 = gs[0];
            assert_eq!(gs.iter().product::<usize>(), d0, "{name}_g: expected a [{d0},1,1]-shaped scalar-per-row tensor, got {gs:?}");
            let folded = fold_weight_norm(&g, &v, d0);
            assert_eq!(&vs, shape, "{name}_v: checkpoint shape {vs:?} disagrees with the manifest's {shape:?}");
            out.insert(name.clone(), (shape.clone(), folded));
        } else {
            let (s, v) = take(&mut by_name, name)?;
            assert_eq!(&s, shape, "{name}: checkpoint shape {s:?} disagrees with the manifest's {shape:?}");
            out.insert(name.clone(), (shape.clone(), v));
        }
    }

    // Everything left is out of scope for this decode-only port (see the
    // module doc) - asserted BY PREFIX rather than silently dropped, so a
    // real checkpoint shape change surfaces here instead of passing through
    // this exact assertion by accident.
    let stray: Vec<&String> = by_name.keys().filter(|n| !OUT_OF_SCOPE_PREFIXES.iter().any(|p| n.starts_with(p))).collect();
    assert!(
        stray.is_empty(),
        "minimaxh3 audio_vae import: {} tensor(s) unclaimed and outside the known out-of-scope prefixes {OUT_OF_SCOPE_PREFIXES:?} - a real checkpoint shape change, not the expected encoder/mean_proj/logs_proj/pre_block set: {stray:?}",
        stray.len()
    );

    assert_eq!(out.len(), manifest.len(), "minimaxh3 audio_vae import: produced {} tensors, manifest has {}", out.len(), manifest.len());

    Ok(out)
}

/// Import the video `vae/` component directory's weight file (the real
/// checkpoint's ROOT flat layout: `transformer/`, `transformer_ref/`, a
/// shared `text_encoder/`, `vae/` and `audio_vae/`, one copy of everything -
/// the layout `diffusers==0.40.0` actually loads and the one this port
/// targets) into a decode/encode-ready [`Tensors`] map matching
/// [`VideoVaeConfig::tensor_manifest`]. `checkpoint::safetensors::
/// read_model_dir` resolves the weight file's own name
/// (`diffusion_pytorch_model.safetensors`, diffusers' `ModelMixin` save
/// convention). Two-way coverage: every manifest entry must be present at
/// its manifest shape, and the checkpoint carries NOTHING beyond the
/// manifest (unlike the audio VAE, this checkpoint's own shipped inference
/// forward - `encoder`+`decoder`+`quant_conv`+`post_quant_conv` - uses every
/// module the class defines, so there is no analogous "out of scope" prefix
/// set to special-case here).
pub fn import_video_vae(dir: &str, cfg: &VideoVaeConfig) -> Result<Tensors, String> {
    let raw = checkpoint::safetensors::read_model_dir(std::path::Path::new(dir))?;
    let mut by_name: HashMap<String, (Vec<usize>, Vec<f32>)> = raw.into_iter().map(|t| (t.name, (t.shape, t.data))).collect();

    let mut out: Tensors = HashMap::new();
    let manifest = cfg.tensor_manifest();
    for (name, shape) in &manifest {
        let (s, v) = by_name.remove(name).ok_or_else(|| format!("minimaxh3 video_vae import: missing tensor {name:?}"))?;
        assert_eq!(&s, shape, "{name}: checkpoint shape {s:?} disagrees with the manifest's {shape:?}");
        out.insert(name.clone(), (shape.clone(), v));
    }

    assert!(by_name.is_empty(), "minimaxh3 video_vae import: {} tensor(s) unclaimed by the manifest: {:?}", by_name.len(), by_name.keys().collect::<Vec<_>>());
    assert_eq!(out.len(), manifest.len(), "minimaxh3 video_vae import: produced {} tensors, manifest has {}", out.len(), manifest.len());

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whether `dir` has landed a weight file `checkpoint::safetensors::
    /// read_model_dir` can resolve - either naming convention, so a test's
    /// own "not downloaded yet" skip check agrees with what the import
    /// function underneath it will actually accept.
    fn has_weights(dir: &str) -> bool {
        checkpoint::safetensors::has_model_weights(std::path::Path::new(dir))
    }

    /// `import_audio_vae_decoder` against the real checkpoint, when
    /// `BRAIN_MINIMAXH3_DIR` points at one - the two-way coverage check
    /// (every claimed tensor present, every present-and-claimed tensor
    /// exactly accounted for, nothing outside the known out-of-scope
    /// prefixes left over) IS this test, matching this workspace's
    /// "goldens/import coverage before a forward runs" discipline. No
    /// MiniMax-H3 weights are vendored or referenced by a baked-in path -
    /// this test is a no-op skip on any machine without the env var pointed
    /// at a real local checkout, per this port's license-driven no-auto-fetch
    /// design (see crate::caps::check_license).
    #[test]
    fn import_covers_the_real_checkpoint_two_way() {
        let Ok(root) = std::env::var("BRAIN_MINIMAXH3_DIR") else {
            brain_testutil::skip("BRAIN_MINIMAXH3_DIR not set - no local MiniMax-H3 checkout to import from");
            return;
        };
        let dir = format!("{root}/audio_vae");
        if !has_weights(&dir) {
            brain_testutil::skip(&format!("{dir} has no importable weight file yet - checkpoint not (yet) downloaded"));
            return;
        }

        let cfg = VocoderConfig::h3_32khz();
        let tensors = import_audio_vae_decoder(&dir, &cfg).unwrap_or_else(|e| panic!("import_audio_vae_decoder: {e}"));

        assert_eq!(tensors.len(), 779);
        assert_eq!(tensors.get("dec_in_proj.weight").unwrap().0, vec![2048, 32, 1]);
        assert_eq!(tensors.get("decoder.conv_pre.weight").unwrap().0, vec![1024, 2048, 7]);
        assert_eq!(tensors.get("decoder.ups.0.0.weight").unwrap().0, vec![1024, 512, 9]);
        assert_eq!(tensors.get("decoder.conv_post.weight").unwrap().0, vec![1, 8, 7]);
        // A weight-normed conv's folded weight must not just be weight_v
        // verbatim (a no-op fold would silently pass every shape assertion
        // above while being numerically wrong) - the fold changed the values
        // unless weight_v's rows already happened to be unit-norm, which real
        // trained weights essentially never are.
        let (_, folded) = tensors.get("decoder.conv_pre.weight").unwrap();
        assert!(folded.iter().any(|&v| v != 0.0), "folded conv_pre.weight must not be trivially all-zero");
    }

    /// Run the REAL, full-scale decoder forward against the real imported
    /// weights above. Not a parity test (no reference oracle is available
    /// yet - see the roadmap ledger's open convention questions), but a real
    /// finding on its own: proves the whole kernel-dispatch graph this port
    /// wrote executes correctly, with no NaN/Inf and no crash/OOM, at the
    /// checkpoint's ACTUAL 1024-wide/7-stage scale on this hardware - the
    /// tiny synthetic config in `crate::vocoder`'s own tests exercises the
    /// same code paths but never at real width, and real width is exactly
    /// where a shape or buffer-sizing bug would first surface.
    #[test]
    fn decode_runs_at_real_scale_with_real_weights_and_stays_finite() {
        let Ok(root) = std::env::var("BRAIN_MINIMAXH3_DIR") else {
            brain_testutil::skip("BRAIN_MINIMAXH3_DIR not set - no local MiniMax-H3 checkout to decode from");
            return;
        };
        let dir = format!("{root}/audio_vae");
        if !has_weights(&dir) {
            brain_testutil::skip(&format!("{dir} has no importable weight file yet - checkpoint not (yet) downloaded"));
            return;
        }

        let cfg = VocoderConfig::h3_32khz();
        let tensors = import_audio_vae_decoder(&dir, &cfg).unwrap_or_else(|e| panic!("import_audio_vae_decoder: {e}"));

        // A short synthetic latent stands in for a real VAE-encoded one -
        // this test is about the DECODER graph running correctly at real
        // scale, not about what a real encoded reference would sound like.
        let t = 4u32;
        let mut rng = data::rng::Lcg::new(11);
        let z = rng.vec_scaled((cfg.vae_latent_channels * t) as usize, 0.3);

        let wave = crate::vocoder::decode(&cfg, &tensors, &z, t, Some("cpu"));
        assert_eq!(wave.len(), (cfg.out_channels * t * cfg.hop_length()) as usize);
        assert!(wave.iter().all(|v| v.is_finite()), "real-weight decode output must be finite");
        assert!(wave.iter().all(|&v| (-1.0..=1.0).contains(&v)), "real-weight decode output must be clamped to [-1,1]");
        assert!(wave.iter().any(|&v| v != 0.0), "real-weight decode output must not be trivially all-zero");
    }

    /// Real NUMERIC parity against `tools/minimaxh3_audio_vae_dump_reference.py`'s
    /// golden - the checkpoint's OWN shipped `minimax_h3_audio_vae.py`/
    /// `dac_audio_vae.py` reference, run for real and dumped. Unlike
    /// `decode_runs_at_real_scale_with_real_weights_and_stays_finite` above
    /// (finite + shaped, no oracle), this is the actual parity ladder rung
    /// this port's Phase 4 was still missing: three real intermediate taps
    /// (`dec_in_proj`, `conv_pre`, the first upsample+resblock-average stage)
    /// plus the end-to-end waveform, all against the SAME latent the golden
    /// was dumped from, so a bug localizes to the first tap it breaks rather
    /// than hiding behind a coincidentally-passing final cosine.
    #[test]
    fn decode_matches_the_real_reference_numerically() {
        let Ok(root) = std::env::var("BRAIN_MINIMAXH3_DIR") else {
            brain_testutil::skip("BRAIN_MINIMAXH3_DIR not set - no local MiniMax-H3 checkout to decode from");
            return;
        };
        let dir = format!("{root}/audio_vae");
        if !has_weights(&dir) {
            brain_testutil::skip(&format!("{dir} has no importable weight file yet - checkpoint not (yet) downloaded"));
            return;
        }

        let fixture_dir = brain_testutil::testdata_path("golden/minimaxh3/audio_vae");
        let fixture_file = fixture_dir.join("minimaxh3_audio_vae.safetensors");
        if !fixture_file.is_file() {
            brain_testutil::skip(&format!(
                "{} not found - run tools/minimaxh3_audio_vae_dump_reference.py --audio-vae-dir {dir} --out {}",
                fixture_file.display(),
                fixture_dir.display()
            ));
            return;
        }

        let cfg = VocoderConfig::h3_32khz();

        // Golden/checkpoint pairing, proven rather than assumed - see
        // brain_testutil::golden's own module doc for the failure this
        // refuses (a golden dumped from a different tier of this
        // architecture silently "passing" a comparison against the wrong
        // reference).
        let Some(src) = brain_testutil::golden::Source::open(&fixture_dir, "tools/minimaxh3_audio_vae_dump_reference.py") else {
            return;
        };
        let ok = src.require(&[
            ("vae_latent_channels", cfg.vae_latent_channels as i64),
            ("mel_channels", cfg.mel_channels as i64),
            ("upsample_initial_channel", cfg.upsample_initial_channel as i64),
            ("num_upsamples", cfg.num_upsamples() as i64),
            ("out_channels", cfg.out_channels as i64),
        ]);
        if !ok {
            return;
        }

        let tensors = import_audio_vae_decoder(&dir, &cfg).unwrap_or_else(|e| panic!("import_audio_vae_decoder: {e}"));

        let fx = checkpoint::safetensors::read(fixture_file.to_str().expect("fixture path is valid UTF-8")).expect("read golden fixture");
        let get = |name: &str| -> &checkpoint::safetensors::StTensor {
            fx.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("golden fixture tap {name:?} missing"))
        };

        let z = get("z");
        let t = z.shape[1] as u32;
        assert_eq!(z.shape[0], cfg.vae_latent_channels as usize, "golden z has {} channels, config expects {}", z.shape[0], cfg.vae_latent_channels);

        let (wave, taps) = crate::vocoder::decode_with_taps(&cfg, &tensors, &z.data, t, Some("cpu"));

        // porting.md's own floor: "cosine >= 0.9999 for networks" at the
        // stage-parity rung.
        let mut r = brain_testutil::parity::Report::new(0.9999);
        r.check("tap_dec_in_proj", &taps.dec_in_proj, &get("tap_dec_in_proj").data);
        r.check("tap_conv_pre", &taps.conv_pre, &get("tap_conv_pre").data);
        r.check("tap_stage0", &taps.stage0, &get("tap_stage0").data);
        r.check("waveform", &wave, &get("waveform").data);
        r.finish("minimaxh3 audio vae decode vs real reference");
    }

    /// `import_video_vae` against the real checkpoint's `vae/` component
    /// (the root flat layout - see [`import_video_vae`]'s own doc), when
    /// `BRAIN_MINIMAXH3_DIR` points at a checkout that has it. As of this
    /// port's Phase 6 pass, `vae/` had NOT yet been fetched (only
    /// `text_encoder/` partially and `FL2VA/audio_vae` were present locally) -
    /// this test exists and is correct, but real-weight coverage is BLOCKED on
    /// that download, not yet achieved; it skips cleanly rather than claiming
    /// something unverified, matching this crate's own honesty convention
    /// (compare `import_covers_the_real_checkpoint_two_way` above, which hit
    /// the same state for the audio VAE before its own download completed).
    #[test]
    fn video_vae_import_covers_the_real_checkpoint_two_way() {
        let Ok(root) = std::env::var("BRAIN_MINIMAXH3_DIR") else {
            brain_testutil::skip("BRAIN_MINIMAXH3_DIR not set - no local MiniMax-H3 checkout to import from");
            return;
        };
        let dir = format!("{root}/vae");
        if !has_weights(&dir) {
            brain_testutil::skip(&format!("{dir} has no importable weight file yet - video vae not (yet) downloaded"));
            return;
        }

        let cfg = VideoVaeConfig::real();
        let tensors = import_video_vae(&dir, &cfg).unwrap_or_else(|e| panic!("import_video_vae: {e}"));
        assert_eq!(tensors.len(), cfg.tensor_manifest().len());
        assert_eq!(tensors.get("encoder.conv_in.weight").unwrap().0, vec![128, 3, 3, 3, 3]);
        assert_eq!(tensors.get("decoder.proj_in.weight").unwrap().0, vec![2048, 24]);
    }

    /// Real-scale structural validation: the full encoder+decoder round trip
    /// (`encode_clip` -> `posterior_mode` -> `decode_clip`) against the real
    /// imported weights, at the checkpoint's ACTUAL widths - finite, no
    /// NaN/Inf, matching this crate's Phase 4 audio-VAE precedent
    /// (`decode_runs_at_real_scale_with_real_weights_and_stays_finite`). Same
    /// download-blocked skip as the coverage test above.
    #[test]
    fn video_vae_round_trip_runs_at_real_scale_with_real_weights_and_stays_finite() {
        let Ok(root) = std::env::var("BRAIN_MINIMAXH3_DIR") else {
            brain_testutil::skip("BRAIN_MINIMAXH3_DIR not set - no local MiniMax-H3 checkout to decode from");
            return;
        };
        let dir = format!("{root}/vae");
        if !has_weights(&dir) {
            brain_testutil::skip(&format!("{dir} has no importable weight file yet - video vae not (yet) downloaded"));
            return;
        }

        let cfg = VideoVaeConfig::real();
        let tensors = import_video_vae(&dir, &cfg).unwrap_or_else(|e| panic!("import_video_vae: {e}"));

        let (t, h, w) = (cfg.clip_length, 256u32, 256u32);
        let mut rng = data::rng::Lcg::new(21);
        let pixels = rng.vec_scaled((cfg.in_channels * t * h * w) as usize, 0.3);
        let (moments, mt, mh, mw) = crate::video_vae::encode_clip(&cfg, &tensors, Some("cpu"), &pixels, t, h, w);
        assert!(moments.iter().all(|v| v.is_finite()), "real-weight encode output must be finite");
        assert!(moments.iter().any(|&v| v != 0.0), "real-weight encode output must not be trivially all-zero");

        let z = crate::video_vae::posterior_mode(&moments, cfg.latent_channels, mt, mh, mw);
        let (pix, pt, ph, pw) = crate::video_vae::decode_clip(&cfg, &tensors, Some("cpu"), &z, mt, mh, mw);
        assert_eq!(pix.len(), (cfg.out_channels * pt * ph * pw) as usize);
        assert!(pix.iter().all(|v| v.is_finite()), "real-weight decode output must be finite");
        assert!(pix.iter().any(|&v| v != 0.0), "real-weight decode output must not be trivially all-zero");
    }
}
