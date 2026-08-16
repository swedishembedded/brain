// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Wan-VAE parity, climbed in the order failures localise.
//!
//! 1. [`wan_vae_encoder_is_chunk_size_invariant`] - **no weights, no fixture**,
//!    and it runs first because it is the one gate on the whole `feat_cache`
//!    mechanism that needs nothing but the code. Splitting a clip as (1,4,4) or
//!    (1,8) must give a BIT-IDENTICAL latent: every output element is summed
//!    over the same taps in the same order either way, so anything but zero
//!    difference is a real cache defect, not rounding. This is the test that
//!    would catch "correct at 9 frames, wrong at 81".
//! 2. [`wan_vae_decoder_tiny_smoke`] - the decoder's three `upsample3d` cache
//!    states exercised at toy dims, again weight-free.
//! 3. [`wan_vae_encoder_matches_reference`] / [`wan_vae_decoder_matches_reference`]
//!    - stage parity against `tools/goldens/wan_vae_dump_reference.py`.
//! 4. [`wan_vae_decoder_matches_the_unchunked_reference`] - the same decode
//!    against the dumper's INDEPENDENT whole-clip path.
//! 5. [`wan_vae_round_trip`] - encode then decode, composed.
//!
//! Everything from (3) on needs the real VAE weights and the golden fixture and
//! skips loudly without them.

use std::collections::HashMap;
use std::path::Path;

use brain_testutil::testdata;
use vae::blocks::Tensors;
use wan::import::import_vae;
use wan::vae3d::{WanVaeConfig, WanVaeDecoder, WanVaeEncoder};

// ------------------------------------------------------------------ metrics

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    model::hostmath::cosine(a, b) as f64
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (&x, &y) in got.iter().zip(want) {
        num += (x as f64 - y as f64).powi(2);
        den += (y as f64).powi(2);
    }
    (num / den).sqrt()
}

fn max_abs(got: &[f32], want: &[f32]) -> f32 {
    got.iter().zip(want).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max)
}

/// Report all three, always: cosine hides a scale error, rel_l2 hides a single
/// bad element, and max_abs hides a broad small bias.
fn report(label: &str, got: &[f32], want: &[f32], min_cos: f64, max_rel: f64) {
    assert_eq!(got.len(), want.len(), "{label}: {} values vs {}", got.len(), want.len());
    let (c, r, m) = (cosine(got, want), rel_l2(got, want), max_abs(got, want));
    eprintln!("{label}: cosine={c:.6}  rel_l2={r:.3e}  max_abs={m:.3e}");
    assert!(c >= min_cos, "{label}: cosine {c:.6} < {min_cos}");
    assert!(r <= max_rel, "{label}: rel_l2 {r:.3e} > {max_rel:.0e}");
}

// ------------------------------------------------------------- weight-free

/// A toy config: same topology, tiny widths, so a weight-free forward runs in
/// seconds while exercising every step kind (3D conv, temporal conv, both
/// resample modes, the channel-L2 norm, per-frame attention, the cache).
fn tiny_cfg() -> WanVaeConfig {
    WanVaeConfig {
        base_dim: 8,
        z_dim: 4,
        dim_mult: vec![1, 2, 4, 4],
        num_res_blocks: 1,
        temperal_downsample: vec![false, true, true],
        latents_mean: (0..4).map(|i| 0.1 * i as f32).collect(),
        latents_std: (0..4).map(|i| 1.0 + 0.25 * i as f32).collect(),
    }
}

/// Deterministic synthetic weights covering the manifest exactly - no
/// randomness crate, no file, and (by construction) no missing or extra name.
fn synthetic_weights(cfg: &WanVaeConfig) -> Tensors {
    let mut t: Tensors = HashMap::new();
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    for (name, shape) in cfg.tensor_manifest() {
        let n: usize = shape.iter().product();
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((state >> 33) as u32) as f32 / (1u64 << 31) as f32; // [0,1)
            v.push(0.2 * (u - 0.5));
        }
        // Norm gains near 1: a gain of ~0 would make every downstream
        // activation ~0 and hide a real difference behind an all-zero compare.
        if name.ends_with(".gamma") {
            for x in v.iter_mut() {
                *x += 1.0;
            }
        }
        t.insert(name, (shape, v));
    }
    t
}

fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % 97) as f32 / 97.0 - 0.5) * 1.7).collect()
}

/// The `feat_cache` gate. See the module header.
#[test]
fn wan_vae_encoder_is_chunk_size_invariant() {
    let cfg = tiny_cfg();
    let w = synthetic_weights(&cfg);
    let (frames, h, wd) = (9u32, 16u32, 16u32);
    let video = ramp((3 * frames * h * wd) as usize);

    let a = WanVaeEncoder::build(&cfg, &w, &cfg.encode_chunks(frames), h, wd, None);
    let b = WanVaeEncoder::build(&cfg, &w, &[1, 8], h, wd, None);
    assert_eq!(a.latent_frames(), 3);
    assert_eq!(b.latent_frames(), 3);
    let za = a.encode(&video);
    let zb = b.encode(&video);

    let m = max_abs(&za, &zb);
    eprintln!(
        "chunk invariance (1,4,4) vs (1,8): {} latent values, max_abs={m:.3e}, cosine={:.9}",
        za.len(),
        cosine(&za, &zb)
    );
    assert!(za.iter().all(|v| v.is_finite()), "latent has non-finite values");
    assert_eq!(m, 0.0, "chunked encode is not chunk-size invariant (max_abs {m:.3e})");
}

/// The decoder walks all three `upsample3d` cache states in a 3-latent-frame
/// decode: chunk 0 takes the empty->`Rep` path, chunk 1 the `Rep` path, chunk 2
/// the cached-frames path. Weight-free; it gates the bindings and the frame
/// bookkeeping, not the numbers.
#[test]
fn wan_vae_decoder_tiny_smoke() {
    let cfg = tiny_cfg();
    let w = synthetic_weights(&cfg);
    let d = WanVaeDecoder::build(&cfg, &w, 3, 2, 2, None);
    assert_eq!(d.frames(), 9);
    let out = d.decode(&ramp((cfg.z_dim * 3 * 2 * 2) as usize));
    assert_eq!(out.len(), (3 * 9 * 16 * 16) as usize);
    assert!(out.iter().all(|v| v.is_finite()), "decode produced non-finite values");
    // A collapsed graph (e.g. a cache that never fires) shows up as a constant.
    let (lo, hi) = out.iter().fold((f32::MAX, f32::MIN), |(l, h), &v| (l.min(v), h.max(v)));
    assert!(hi - lo > 1e-6, "decode output is constant ({lo}..{hi})");
}

// ---------------------------------------------------------- real fixtures

/// The shipped VAE weights: `BRAIN_WAN_VAE`, else the same file inside
/// whatever model store `BRAIN_MODELS_DIR` names. Both are variables rather
/// than a literal because a machine path baked into a test passes on exactly
/// one machine and skips silently on every other, which reads as "no fixture"
/// instead of "wrong path".
fn weights_path() -> Option<String> {
    if let Ok(p) = std::env::var("BRAIN_WAN_VAE") {
        return (!p.is_empty() && Path::new(&p).exists()).then_some(p);
    }
    let store = std::env::var("BRAIN_MODELS_DIR").ok()?;
    let p = Path::new(&store)
        .join("Wan-AI/Wan2.1-T2V-1.3B-Diffusers/vae/diffusion_pytorch_model.safetensors");
    p.exists().then(|| p.to_string_lossy().into_owned())
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

/// `(weights, fixture)` or `None` with a loud skip.
fn setup(frames: u32) -> Option<(Tensors, Fixture)> {
    let fx = testdata(&format!("golden/wan/vae/vae_t{frames}.safetensors"));
    if !Path::new(&fx).exists() {
        eprintln!("SKIP: fixture {fx} absent - run tools/goldens/wan_vae_dump_reference.py");
        return None;
    }
    let Some(wp) = weights_path() else {
        eprintln!("SKIP: set BRAIN_WAN_VAE to the Wan2.1 VAE safetensors");
        return None;
    };
    let cfg = WanVaeConfig::wan21();
    let raw = checkpoint::safetensors::read(&wp).expect("read VAE weights");
    let n = raw.len();
    let w = import_vae(raw, &cfg).expect("import VAE weights");
    eprintln!("imported {n} source tensors -> {} canonical from {wp}", w.len());
    let t = checkpoint::safetensors::read(&fx).expect("read golden");
    Some((w, Fixture { t }))
}

fn run_encoder(frames: u32) {
    let Some((w, fx)) = setup(frames) else { return };
    let cfg = WanVaeConfig::wan21();
    let vshape = fx.shape("video").to_vec();
    let (h, wd) = (vshape[2] as u32, vshape[3] as u32);
    let enc = WanVaeEncoder::build(&cfg, &w, &cfg.encode_chunks(frames), h, wd, None);
    let latent = enc.encode(fx.get("video"));

    report("enc_out", &enc.read_stage("enc_out").unwrap(), fx.get("enc_out"), 0.99999, 1e-4);
    report("moments", &enc.read_stage("moments").unwrap(), fx.get("moments"), 0.99999, 1e-4);
    report("mu", &enc.read_stage("mu").unwrap(), fx.get("mu"), 0.99999, 1e-4);
    report("log_var", &enc.read_stage("log_var").unwrap(), fx.get("log_var"), 0.99999, 1e-4);
    report("latent", &latent, fx.get("latent"), 0.99999, 1e-4);
}

fn run_decoder(frames: u32) {
    let Some((w, fx)) = setup(frames) else { return };
    let cfg = WanVaeConfig::wan21();
    let ls = fx.shape("latent").to_vec();
    let (lt, lh, lw) = (ls[1] as u32, ls[2] as u32, ls[3] as u32);
    let dec = WanVaeDecoder::build(&cfg, &w, lt, lh, lw, None);
    let recon = dec.decode(fx.get("latent"));
    assert_eq!(dec.frames(), frames);

    report("z_denorm", &dec.read_stage("z_denorm").unwrap(), fx.get("z_denorm"), 0.99999, 1e-4);
    report("dec_conv2", &dec.read_stage("dec_conv2").unwrap(), fx.get("dec_conv2"), 0.99999, 1e-4);
    report("recon (vs chunked reference)", &recon, fx.get("recon_chunked"), 0.99999, 1e-4);
}

#[test]
fn wan_vae_encoder_matches_reference() {
    run_encoder(9);
}

/// 17 frames is 5 encode chunks and 5 decode chunks - the multi-chunk case the
/// 9-frame fixture only just reaches. A cache bug that survives three chunks
/// rarely survives five.
#[test]
fn wan_vae_encoder_matches_reference_multichunk() {
    run_encoder(17);
}

#[test]
fn wan_vae_decoder_matches_reference() {
    run_decoder(9);
}

#[test]
fn wan_vae_decoder_matches_reference_multichunk() {
    run_decoder(17);
}

/// brain's chunked decode against the dumper's INDEPENDENT whole-clip path -
/// a different formulation of the same model (see the dumper's header), so it
/// gates the cache semantics rather than reproducing them.
#[test]
fn wan_vae_decoder_matches_the_unchunked_reference() {
    for frames in [9u32, 17] {
        let Some((w, fx)) = setup(frames) else { return };
        let cfg = WanVaeConfig::wan21();
        let ls = fx.shape("latent").to_vec();
        let dec =
            WanVaeDecoder::build(&cfg, &w, ls[1] as u32, ls[2] as u32, ls[3] as u32, None);
        let recon = dec.decode(fx.get("latent"));
        report(
            &format!("recon t={frames} (vs UNCHUNKED reference)"),
            &recon,
            fx.get("recon_unchunked"),
            0.99999,
            1e-4,
        );
    }
}

/// Encode then decode in one composed run, against the reference's own
/// composition of the same two stages.
#[test]
fn wan_vae_round_trip() {
    let frames = 9u32;
    let Some((w, fx)) = setup(frames) else { return };
    let cfg = WanVaeConfig::wan21();
    let vshape = fx.shape("video").to_vec();
    let (h, wd) = (vshape[2] as u32, vshape[3] as u32);
    let enc = WanVaeEncoder::build(&cfg, &w, &cfg.encode_chunks(frames), h, wd, None);
    let latent = enc.encode(fx.get("video"));
    let dec = WanVaeDecoder::build(&cfg, &w, enc.latent_frames(), h / 8, wd / 8, None);
    let recon = dec.decode(&latent);
    report("round trip", &recon, fx.get("recon_chunked"), 0.99999, 1e-4);

    // The reference clamps outside the model; check the clamped form too, since
    // that is what a pipeline actually writes to a file.
    let clamped: Vec<f32> = recon.iter().map(|v| v.clamp(-1.0, 1.0)).collect();
    report("round trip (clamped)", &clamped, fx.get("recon_clamped"), 0.99999, 1e-4);
}

/// The importer against the REAL shipped file, both directions.
#[test]
fn wan_vae_import_covers_the_shipped_checkpoint() {
    let Some(wp) = weights_path() else {
        eprintln!("SKIP: set BRAIN_WAN_VAE to the Wan2.1 VAE safetensors");
        return;
    };
    let cfg = WanVaeConfig::wan21();
    let raw = checkpoint::safetensors::read(&wp).expect("read VAE weights");
    assert_eq!(raw.len(), 194, "{wp} has {} tensors", raw.len());
    // `StTensor` is not `Clone`, and the negative cases only need the real
    // file's NAMES and SHAPES - the values are irrelevant to coverage.
    let meta: Vec<(String, Vec<usize>)> =
        raw.iter().map(|t| (t.name.clone(), t.shape.clone())).collect();
    let stub = |skip: Option<&str>, add: Option<&str>| {
        let mut v: Vec<checkpoint::safetensors::StTensor> = meta
            .iter()
            .filter(|(n, _)| Some(n.as_str()) != skip)
            .map(|(n, s)| checkpoint::safetensors::StTensor {
                name: n.clone(),
                shape: s.clone(),
                data: vec![0.0; s.iter().product()],
            })
            .collect();
        if let Some(a) = add {
            v.push(checkpoint::safetensors::StTensor {
                name: a.into(),
                shape: vec![1],
                data: vec![0.0],
            });
        }
        v
    };

    let w = import_vae(raw, &cfg).expect("import");
    assert_eq!(w.len(), cfg.tensor_manifest().len());
    drop(w);

    // Drop one source tensor: the error must name the CANONICAL tensor it fed.
    let e = import_vae(stub(Some("decoder.up_blocks.1.upsamplers.0.time_conv.weight"), None), &cfg)
        .unwrap_err();
    assert!(e.contains("decoder.upsamples.7.time_conv.weight"), "{e}");

    // Add one the model does not read: it must be reported, not ignored.
    let e = import_vae(stub(None, Some("encoder.down_blocks.2.time_conv.weight")), &cfg)
        .unwrap_err();
    assert!(e.contains("unused source tensors"), "{e}");
}
