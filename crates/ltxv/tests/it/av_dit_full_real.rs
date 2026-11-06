// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for verifying ported generative
// media pipelines against their reference implementations for its clients. If
// your team needs expertise in video diffusion model inference then you can
// procure our services by sending an email to info@swedishembedded.com.

//! The whole LTX-2.5 audio-visual DiT against the reference, at full depth.
//!
//! `tools/goldens/ltxv_real_av_dit_full_dump_reference.py` runs the official
//! `LTXModel` - all 48 blocks, both streams, built from the real bf16
//! transformer and run in fp32 - on one denoise step's inputs built the way
//! the reference pipeline builds them, against a real prompt's 1024-token
//! context. This test runs brain's production forward on the same step: the
//! int8 streamed, device-resident path a generation denoises through, reading
//! the store's Q8_0 checkpoint and routing the prompt through its own
//! connectors (`tools/goldens/ltxv_duration_e2e_dump_reference.py`'s features,
//! the connectors' input). What it measures is everything a generation's
//! velocity carries: the Q8_0 weights, int8 activations, both connectors,
//! both streams' prompt-side AdaLN and all 48 blocks.

use std::path::Path;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "cosine: length mismatch ({} vs {})", a.len(), b.len());
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        d += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    d / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE)
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let num: f64 = got.iter().zip(want).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = want.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(f64::MIN_POSITIVE)).sqrt()
}

struct Fixture(Vec<checkpoint::safetensors::StTensor>);

impl Fixture {
    fn read(path: &str) -> Fixture {
        Fixture(checkpoint::safetensors::read(path).unwrap_or_else(|e| panic!("reading {path}: {e}")))
    }
    fn has(&self, name: &str) -> bool {
        self.0.iter().any(|t| t.name == name)
    }
    fn get(&self, name: &str) -> &[f32] {
        &self.0.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("no golden {name}")).data
    }
}

/// One velocity of the production audio-visual forward against the
/// reference's, on every stream.
#[test]
#[ignore = "needs a GPU, the real LTX-2.5 Q8_0 DiT in the model store and both goldens"]
fn the_production_av_forward_matches_the_reference_at_full_depth() {
    let Some(paths) = ltxv::spec::default_store_paths() else {
        brain_testutil::skip("the model store does not resolve LTX-2.5");
        return;
    };
    let Some(dit) = paths.dit.clone() else {
        brain_testutil::skip("the model store has no LTX-2.5 DiT");
        return;
    };
    let (step_path, prompt_path) = (brain_testutil::testdata("golden/ltxv/av_dit_full/av_dit_full.safetensors"), brain_testutil::testdata("golden/ltxv/duration_e2e/duration_e2e.safetensors"));
    for p in [&step_path, &prompt_path] {
        if !Path::new(p).exists() {
            brain_testutil::skip(&format!("fixture {p} absent - run tools/goldens/ltxv_real_av_dit_full_dump_reference.py (and ltxv_duration_e2e_dump_reference.py)"));
            return;
        }
    }
    let (fx, prompt) = (Fixture::read(&step_path), Fixture::read(&prompt_path));

    let src = ltxv::gguf_src::LtxvGgufSource::open(&dit).unwrap_or_else(|e| panic!("opening {dit}: {e}"));
    let cfg = *src.config();
    let head = ltxv::dit::load_av_head_tensors_from_source(&src, &cfg);
    let (in_v, in_a) = (cfg.video.in_channels as usize, cfg.audio.in_channels as usize);
    let (v_latent, a_latent) = (fx.get("video.latent"), fx.get("audio.latent"));
    let (tv, ta) = (v_latent.len() / in_v, a_latent.len() / in_a);
    let sigma = fx.get("sigma")[0];
    let valid = prompt.get("prompt0_mask");
    let step = ltxv::dit::AvStreamedStep {
        perturbations: &ltxv::block::NO_PERTURBATIONS,
        reference_attention: None,
        v_latent,
        v_timesteps: fx.get("video.timesteps"),
        v_positions: fx.get("video.positions"),
        v_keyframes_mask: fx.get("video.keyframes_mask"),
        v_context: prompt.get("prompt0_video_features"),
        v_context_len: valid.len(),
        tv,
        v_sigma: sigma,
        v_context_valid: valid,
        a_latent,
        a_timesteps: fx.get("audio.timesteps"),
        a_positions: fx.get("audio.positions"),
        a_context: prompt.get("prompt0_audio_features"),
        a_context_len: valid.len(),
        ta,
        a_sigma: sigma,
        a_context_valid: valid,
    };
    // A one-block window: the forward streams every block through it, so the
    // test fits beside other work on the card.
    let session = ltxv::devres::AvDitSession::resident_with_slots(None, 1);
    let cache = ltxv::block::GenerationCache::default();
    let t0 = std::time::Instant::now();
    let (v, a) = ltxv::dit::av_forward_q_streamed_in(&session, &cfg, &src, &head, ltxv::block::QTier::Int8, &step, &cache);
    eprintln!("production AV forward, {} blocks, {tv} video + {ta} audio tokens: {:.1} s", cfg.video.num_layers, t0.elapsed().as_secs_f64());

    let mut worst = Vec::new();
    for (label, got, want, dim) in [("video velocity", &v, fx.get("video.velocity"), in_v), ("audio velocity", &a, fx.get("audio.velocity"), in_a)] {
        let (c, r) = (cosine(got, want), rel_l2(got, want));
        let rows: Vec<f64> = got.chunks(dim).zip(want.chunks(dim)).map(|(g, w)| cosine(g, w)).collect();
        let min = rows.iter().copied().fold(f64::INFINITY, f64::min);
        eprintln!("{label}: cosine {c:.6}, rel_l2 {r:.3e}, token cosine min {min:.4} mean {:.4}", rows.iter().sum::<f64>() / rows.len() as f64);
        eprintln!("{label}: per-token cosine {:?}", rows.iter().map(|x| (x * 1000.0).round() / 1000.0).collect::<Vec<_>>());
        worst.push((label, c));
    }
    // Measured 0.99991 (video) and 0.99997 (audio) against the bf16
    // reference: Q8_0 weights and int8 activations over 48 blocks.
    for (label, c) in worst {
        assert!(c >= 0.9995, "{label}: cosine {c:.6} against the reference");
    }
}

/// The same step through the eager fp32 forward - the reference definition
/// of brain's math, without the int8 path - with the reference's block
/// outputs after blocks 0, 11, 23, 35 and 47 compared, so a divergence is
/// placed in depth and apart from quantization. Holds the whole Q8_0
/// checkpoint dequantized to fp32 on the host (about 88 GB).
#[test]
#[ignore = "needs ~90 GB of host memory, the real LTX-2.5 Q8_0 DiT in the model store and both goldens; slow on the CPU"]
fn the_eager_av_forward_matches_the_reference_block_by_block() {
    let Some(dit) = ltxv::spec::default_store_paths().and_then(|p| p.dit) else {
        brain_testutil::skip("the model store has no LTX-2.5 DiT");
        return;
    };
    let (step_path, prompt_path) = (brain_testutil::testdata("golden/ltxv/av_dit_full/av_dit_full.safetensors"), brain_testutil::testdata("golden/ltxv/duration_e2e/duration_e2e.safetensors"));
    if !Path::new(&step_path).exists() || !Path::new(&prompt_path).exists() {
        brain_testutil::skip("the full-depth goldens are absent");
        return;
    }
    let (fx, prompt) = (Fixture::read(&step_path), Fixture::read(&prompt_path));
    let cfg = *ltxv::gguf_src::LtxvGgufSource::open(&dit).unwrap().config();
    let mg = checkpoint::gguf::MmapGguf::open(&dit).unwrap();
    let w: vae::blocks::Tensors = ltxv::dit::av_dit_tensor_manifest(&cfg)
        .into_iter()
        .map(|(name, shape)| {
            let data = mg.tensor(&name).unwrap_or_else(|| panic!("missing {name}")).unwrap_or_else(|e| panic!("{name}: {e}"));
            (name, (shape, data))
        })
        .collect();
    let model = ltxv::LtxAvDit::new(cfg, w, None);
    let (v_latent, a_latent) = (fx.get("video.latent"), fx.get("audio.latent"));
    let (tv, ta) = (v_latent.len() / cfg.video.in_channels as usize, a_latent.len() / cfg.audio.in_channels as usize);
    let (sigma, valid) = (fx.get("sigma")[0], prompt.get("prompt0_mask"));
    let t0 = std::time::Instant::now();
    #[rustfmt::skip]
    let taps = model.forward(
        v_latent, fx.get("video.timesteps"), fx.get("video.positions"), fx.get("video.keyframes_mask"), prompt.get("prompt0_video_features"), valid.len(), tv, sigma, valid,
        a_latent, fx.get("audio.timesteps"), fx.get("audio.positions"), prompt.get("prompt0_audio_features"), valid.len(), ta, sigma, valid,
    );
    eprintln!("eager fp32 AV forward: {:.1} s", t0.elapsed().as_secs_f64());
    eprintln!("video connector_out vs reference encoding: cosine {:.6}", cosine(&taps.video.connector_out, prompt.get("prompt0_video_encoding")));
    // Block 0's internals, when the golden was dumped with `--tap-block0`.
    for (name, got) in [
        ("b0.attn1", &taps.video.b0_attn1_out),
        ("b0.attn2", &taps.video.b0_attn2_out),
        ("b0.ff", &taps.video.b0_ff_out),
        ("b0.audio_attn1", &taps.audio.b0_attn1_out),
        ("b0.audio_attn2", &taps.audio.b0_attn2_out),
        ("b0.audio_ff", &taps.audio.b0_ff_out),
        ("b0.a2v", &taps.b0_a2v_out),
        ("b0.v2a", &taps.b0_v2a_out),
    ] {
        if fx.has(name) {
            eprintln!("{name}: cosine {:.6}", cosine(got, fx.get(name)));
        }
    }
    for b in (0..taps.video.block_out.len()).filter(|b| fx.has(&format!("video.block{b}"))) {
        let (v, a) = (&taps.video.block_out[b], &taps.audio.block_out[b]);
        eprintln!("block {b}: video cosine {:.6}, audio cosine {:.6}", cosine(v, fx.get(&format!("video.block{b}"))), cosine(a, fx.get(&format!("audio.block{b}"))));
    }
    let (cv, ca) = (cosine(&taps.video.out, fx.get("video.velocity")), cosine(&taps.audio.out, fx.get("audio.velocity")));
    eprintln!("velocity: video cosine {cv:.6}, audio cosine {ca:.6}");
    assert!(cv >= 0.999 && ca >= 0.999, "eager velocity: video {cv:.6}, audio {ca:.6}");
}
