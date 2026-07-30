// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Composed-loop parity: replay the reference run from its first captured
//! post-step latent — our DiT + Euler + schedule must land on the same
//! per-step latents, and the decoded image must match the reference output.
//! (The initial noise itself is torch-Philox and not reproduced; starting from
//! `latents_step0` removes RNG from the equation.)

use flux2::{position_ids, Flux2Config, Flux2Model};

fn testdata(rel: &str) -> String {
    let root = std::env::var("BRAIN_TESTDATA")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata").to_string());
    format!("{root}/{rel}")
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[test]
fn euler_loop_and_decode_match_reference() {
    let f_e2e = testdata("flux2/klein-4b/e2e.safetensors");
    let f_text = testdata("flux2/klein-4b/text.safetensors");
    if !std::path::Path::new(&f_e2e).exists() {
        eprintln!("SKIP: fixture {f_e2e} absent");
        return;
    }
    let Ok(dit_dir) = std::env::var("BRAIN_FLUX2_TRANSFORMER") else {
        eprintln!("SKIP: BRAIN_FLUX2_TRANSFORMER unset");
        return;
    };
    let Ok(vae_dir) = std::env::var("BRAIN_FLUX2_VAE") else {
        eprintln!("SKIP: BRAIN_FLUX2_VAE unset");
        return;
    };

    let e2e = checkpoint::safetensors::read(&f_e2e).unwrap();
    let text = checkpoint::safetensors::read(&f_text).unwrap();
    let get = |fx: &Vec<checkpoint::safetensors::StTensor>, name: &str| -> Vec<f32> {
        fx.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("golden {name}")).data.clone()
    };
    let ctx = get(&text, "ctx");
    let steps_lat: Vec<Vec<f32>> =
        (0..4).map(|i| get(&e2e, &format!("latents_step{i}"))).collect();
    let want_img = get(&e2e, "image"); // [512,512,3] in [0,1]

    // DiT
    let cfg = Flux2Config::klein_4b();
    let mut files: Vec<_> = std::fs::read_dir(&dit_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|q| q.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    let mut ts = Vec::new();
    for f in files {
        ts.extend(checkpoint::safetensors::read(f.to_str().unwrap()).unwrap());
    }
    let map = flux2::import_diffusers(ts, &cfg).unwrap();
    let gpu = gpu_core::testgpu::dev(flux2::KERNELS);
    let model = Flux2Model::new(&cfg, &map, gpu, 512 + 1024);

    // schedule: 512x512 -> 1024 tokens, 4 steps
    let sigmas = diffusion::scheduler::klein_sigmas(4, 1024);
    let ids = position_ids(512, 32, 32, &[]);

    // replay steps 1..4 from the captured post-step-0 latent
    let mut lat = steps_lat[0].clone();
    for i in 1..4 {
        let pred = model.forward(&lat, &ctx, sigmas[i], &ids, 1024);
        for (x, v) in lat.iter_mut().zip(&pred) {
            *x += (sigmas[i + 1] - sigmas[i]) * v;
        }
        let cos = cosine(&lat, &steps_lat[i]);
        eprintln!("step {i}: latent cosine={cos:.6}");
        assert!(cos >= 0.999, "step {i} latent cosine {cos:.6}");
    }

    // decode via the pipeline's VAE path (VAE-only Pipeline is not
    // constructible without TE weights, so use the vae crate directly —
    // the same calls Pipeline::decode_tokens makes)
    let vp = std::path::Path::new(&vae_dir);
    let vae_json = std::fs::read_to_string(vp.join("config.json")).unwrap();
    let vae_cfg = vae::VaeConfig::from_json(&serde_json::from_str(&vae_json).unwrap());
    let vts = checkpoint::safetensors::read(
        vp.join("diffusion_pytorch_model.safetensors").to_str().unwrap(),
    )
    .unwrap();
    let mut vmap = std::collections::HashMap::new();
    let (mut bn_mean, mut bn_var) = (Vec::new(), Vec::new());
    for t in vts {
        if t.name == "bn.running_mean" { bn_mean = t.data.clone(); }
        if t.name == "bn.running_var" { bn_var = t.data.clone(); }
        vmap.insert(t.name, (t.shape, t.data));
    }
    let (lh, lw) = (32usize, 32usize);
    let mut packed = vec![0.0f32; 128 * lh * lw];
    for c in 0..128 {
        for y in 0..lh {
            for x in 0..lw {
                packed[(c * lh + y) * lw + x] = lat[(y * lw + x) * 128 + c];
            }
        }
    }
    let unpacked = vae::latent::unpack(&packed, 32, lh * 2, lw * 2, &bn_mean, &bn_var, vae_cfg.batch_norm_eps);
    let dec = vae::VaeDecoder::from_diffusers(vae_cfg, &vmap, (lh * 2) as u32, (lw * 2) as u32, None);
    let chw = dec.decode(&unpacked);
    // reference image is [0,1] HWC; ours is [-1,1] CHW
    let n = 512 * 512;
    let mut got = vec![0.0f32; n * 3];
    for c in 0..3 {
        for i in 0..n {
            got[i * 3 + c] = (chw[c * n + i].clamp(-1.0, 1.0) + 1.0) * 0.5;
        }
    }
    let cos = cosine(&got, &want_img);
    let max_abs = got.iter().zip(&want_img).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max);
    eprintln!("decoded image: cosine={cos:.6} max_abs={max_abs:.4}");
    assert!(cos >= 0.999, "image cosine {cos:.6}");
}
