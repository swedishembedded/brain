// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DiT forward parity vs the diffusers reference, replaying the exact
//! transformer inputs captured by `tools/flux2_dump_reference.py` (forward
//! hooks during a real pipeline run): packed latents, text conditioning,
//! timestep, and the reference's own position ids.
//!
//! `dit.safetensors`: the plain t2i case (512×512 → 1024 image tokens).
//! `dit_edit.safetensors`: one reference image appended (2048 image tokens,
//! ids carrying the t=10 offset); the prediction golden covers all image rows.
//!
//! Env: `BRAIN_FLUX2_TRANSFORMER` = the klein-4B diffusers `transformer/` dir.
//! Skips without weights or fixtures.

use flux2::{Flux2Config, Flux2Model};

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

fn load_weights() -> Option<flux2::Tensors> {
    let Ok(dir) = std::env::var("BRAIN_FLUX2_TRANSFORMER") else {
        eprintln!("SKIP: BRAIN_FLUX2_TRANSFORMER unset");
        return None;
    };
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("transformer dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|q| q.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    let mut tensors = Vec::new();
    for f in files {
        tensors.extend(checkpoint::safetensors::read(f.to_str().unwrap()).unwrap());
    }
    Some(flux2::import_diffusers(tensors, &Flux2Config::klein_4b()).unwrap())
}

fn run_case(model: &Flux2Model, fixture: &str) {
    let fx = checkpoint::safetensors::read(fixture).expect("read fixture");
    let get = |name: &str| {
        fx.iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("golden {name}"))
    };
    let hs = get("hs");
    let ctx = get("ctx");
    let t = get("timestep").data[0];
    assert!((0.0..=1.0).contains(&t), "timestep {t} not in [0,1]");
    let img_ids = get("img_ids");
    let txt_ids = get("txt_ids");
    let want = &get("out").data;

    let n_img = hs.shape[1];
    // joint ids, text rows first — the reference's own ids, f32 → u32
    let mut ids: Vec<u32> = Vec::with_capacity((512 + n_img) * 4);
    ids.extend(txt_ids.data.iter().map(|&v| v as u32));
    ids.extend(img_ids.data.iter().map(|&v| v as u32));

    let got = model.forward(&hs.data, &ctx.data, t, &ids, n_img);
    assert_eq!(got.len(), want.len());
    let cos = cosine(&got, want);
    let max_abs = got
        .iter()
        .zip(want)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    eprintln!("{fixture}: cosine={cos:.6} max_abs={max_abs:.4}");
    assert!(cos >= 0.999, "cosine {cos:.6} < 0.999");
}

#[test]
fn dit_forward_matches_reference() {
    let f_t2i = testdata("flux2/klein-4b/dit.safetensors");
    let f_edit = testdata("flux2/klein-4b/dit_edit.safetensors");
    if !std::path::Path::new(&f_t2i).exists() {
        eprintln!("SKIP: fixture {f_t2i} absent");
        return;
    }
    let Some(map) = load_weights() else { return };
    let cfg = Flux2Config::klein_4b();
    let gpu = gpu_core::testgpu::dev(flux2::KERNELS);
    // sized for the edit case: 512 txt + 2048 img/ref tokens
    let model = Flux2Model::new(&cfg, &map, gpu, 512 + 2048);
    run_case(&model, &f_t2i);
    if std::path::Path::new(&f_edit).exists() {
        run_case(&model, &f_edit);
    }
}

#[test]
fn position_ids_match_reference_layout() {
    let f_t2i = testdata("flux2/klein-4b/dit.safetensors");
    if !std::path::Path::new(&f_t2i).exists() {
        eprintln!("SKIP: fixture {f_t2i} absent");
        return;
    }
    let fx = checkpoint::safetensors::read(&f_t2i).unwrap();
    let get = |name: &str| &fx.iter().find(|t| t.name == name).unwrap().data;
    let mut want: Vec<u32> = get("txt_ids").iter().map(|&v| v as u32).collect();
    want.extend(get("img_ids").iter().map(|&v| v as u32));
    // 512×512 → 32×32 latent tokens, no refs
    let ours = flux2::position_ids(512, 32, 32, &[]);
    assert_eq!(ours, want, "position id layout diverges from the reference");

    let f_edit = testdata("flux2/klein-4b/dit_edit.safetensors");
    if std::path::Path::new(&f_edit).exists() {
        let fx = checkpoint::safetensors::read(&f_edit).unwrap();
        let get = |name: &str| &fx.iter().find(|t| t.name == name).unwrap().data;
        let mut want: Vec<u32> = get("txt_ids").iter().map(|&v| v as u32).collect();
        want.extend(get("img_ids").iter().map(|&v| v as u32));
        let ours = flux2::position_ids(512, 32, 32, &[(32, 32)]);
        assert_eq!(ours, want, "edit position ids diverge (ref t-offset)");
    }
}
