// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Forward parity vs the ORIGINAL DIAMOND implementation: a tiny random-weight
//! InnerModel was run by scripts/parity-dump/diamond.py on a fixed input; brain
//! must reproduce its output within fp32 tolerance.
//!
//! Fixtures are NOT in git — regenerate with `make wm-fixtures` (needs torch).
//! The test SKIPS (with a message) when the fixture directory is absent.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use wm_diamond::{DiamondConfig, DiamondUNet, Tensors};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/diamond")
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

struct Fixture {
    manifest: serde_json::Value,
    dir: PathBuf,
}

impl Fixture {
    fn load() -> Option<Fixture> {
        let dir = fixture_dir();
        let m = dir.join("manifest.json");
        if !m.exists() {
            eprintln!("SKIP: {} absent — run `make wm-fixtures` (needs torch)", m.display());
            return None;
        }
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&m).unwrap()).unwrap();
        Some(Fixture { manifest, dir })
    }

    fn tensor(&self, section: &str, name: &str) -> (Vec<usize>, Vec<f32>) {
        let e = &self.manifest[section][name];
        assert!(!e.is_null(), "fixture missing {section}/{name}");
        let shape: Vec<usize> =
            e["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let data = read_f32(&self.dir.join(e["file"].as_str().unwrap()));
        assert_eq!(data.len(), shape.iter().product::<usize>().max(1), "{section}/{name}");
        (shape, data)
    }

    fn weights(&self) -> Tensors {
        let mut t: Tensors = HashMap::new();
        for (name, _) in self.manifest["weights"].as_object().unwrap() {
            let (shape, data) = self.tensor("weights", name);
            t.insert(name.clone(), (shape, data));
        }
        t
    }

    fn config(&self) -> DiamondConfig {
        let c = &self.manifest["config"];
        let u = |k: &str| c[k].as_u64().unwrap() as u32;
        DiamondConfig {
            img_channels: u("img_channels"),
            num_steps_conditioning: u("num_steps_conditioning"),
            cond_channels: u("cond_channels"),
            depths: c["depths"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect(),
            channels: c["channels"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect(),
            attn_depths: c["attn_depths"].as_array().unwrap().iter().map(|v| v.as_bool().unwrap()).collect(),
            num_actions: u("num_actions"),
            h: u("h"),
            w: u("w"),
            sigma_data: 0.5,
            sigma_offset_noise: 0.3,
        }
    }
}

#[test]
fn parity_tiny_unet_forward_matches_reference() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        eprintln!("SKIP: MOE_SKIP_GPU_TESTS set");
        return;
    }
    let Some(fx) = Fixture::load() else { return };
    let cfg = fx.config();
    let weights = fx.weights();

    // The fixture dumps every InnerModel parameter — the config's manifest
    // must cover them exactly (same full-coverage discipline as import).
    let expected = cfg.param_list();
    assert_eq!(
        expected.len(),
        weights.len(),
        "param_list vs fixture weight count"
    );
    for (name, shape) in &expected {
        let (got, _) = weights.get(name).unwrap_or_else(|| panic!("fixture lacks {name}"));
        assert_eq!(got, shape, "{name} shape");
    }

    let unet = DiamondUNet::new(cfg, &weights, Some("cpu"));

    let (_, noisy) = fx.tensor("inputs", "noisy");
    let (_, c_noise) = fx.tensor("inputs", "c_noise");
    let (_, obs) = fx.tensor("inputs", "obs");
    let (_, act) = fx.tensor("inputs", "act");
    let actions: Vec<u32> = act.iter().map(|&v| v as u32).collect();

    unet.set_context(&obs);
    let y = unet.forward(&noisy, c_noise[0], &actions);

    let (_, y_ref) = fx.tensor("output", "y");
    assert_eq!(y.len(), y_ref.len());
    let mut max_abs = 0.0f32;
    for (a, b) in y.iter().zip(&y_ref) {
        max_abs = max_abs.max((a - b).abs());
    }
    assert!(
        max_abs < 1e-4,
        "forward parity: max abs diff {max_abs} (tolerance 1e-4)"
    );
}

#[test]
fn parity_host_cond_path_matches_reference() {
    let Some(fx) = Fixture::load() else { return };
    let cfg = fx.config();
    let weights = fx.weights();
    let get = |n: &str| weights.get(n).unwrap().1.clone();
    let cond = wm_diamond::cond::CondNet {
        cond_channels: cfg.cond_channels as usize,
        num_steps_conditioning: cfg.num_steps_conditioning as usize,
        fourier_w: get("noise_emb.weight"),
        act_emb: get("act_emb.0.weight"),
        num_actions: cfg.num_actions as usize,
        mlp0_w: get("cond_proj.0.weight"),
        mlp0_b: get("cond_proj.0.bias"),
        mlp2_w: get("cond_proj.2.weight"),
        mlp2_b: get("cond_proj.2.bias"),
    };
    let (_, c_noise) = fx.tensor("inputs", "c_noise");
    let (_, act) = fx.tensor("inputs", "act");
    let actions: Vec<u32> = act.iter().map(|&v| v as u32).collect();
    let mine = cond.cond(c_noise[0], &actions);
    let (_, r) = fx.tensor("activations", "cond_proj");
    let max = mine.iter().zip(&r).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max < 1e-5, "cond path max abs diff {max}");
}

/// Localizes divergence: walks taps in graph order, reports the first module
/// whose output differs from the reference beyond 1e-4.
#[test]
fn parity_first_divergent_module_report() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let Some(fx) = Fixture::load() else { return };
    let cfg = fx.config();
    let weights = fx.weights();
    let unet = DiamondUNet::new(cfg, &weights, Some("cpu"));
    let (_, noisy) = fx.tensor("inputs", "noisy");
    let (_, c_noise) = fx.tensor("inputs", "c_noise");
    let (_, obs) = fx.tensor("inputs", "obs");
    let (_, act) = fx.tensor("inputs", "act");
    let actions: Vec<u32> = act.iter().map(|&v| v as u32).collect();
    unet.set_context(&obs);
    let _ = unet.forward(&noisy, c_noise[0], &actions);
    for name in unet.tap_names() {
        let mine = unet.read_tap(&name).unwrap();
        let e = &fx.manifest["activations"][&name];
        if e.is_null() {
            if let Ok(dir) = std::env::var("WM_DUMP_TAPS") {
                let mine = unet.read_tap(&name).unwrap();
                let bytes: Vec<u8> = mine.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::create_dir_all(&dir).unwrap();
                std::fs::write(format!("{dir}/{}.f32", name.replace('/', "_")), bytes).unwrap();
            }
            continue;
        }
        let (_, r) = fx.tensor("activations", &name);
        if mine.len() != r.len() {
            panic!("{name}: len {} vs ref {}", mine.len(), r.len());
        }
        let max = mine.iter().zip(&r).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("{name}: max abs {max:e}");
        if let Ok(dir) = std::env::var("WM_DUMP_TAPS") {
            let bytes: Vec<u8> = mine.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(format!("{dir}/{}.f32", name.replace('/', "_")), bytes).unwrap();
            continue; // dump everything, do not stop at the first divergence
        }
        assert!(max < 1e-4, "FIRST DIVERGENT MODULE: {name} (max abs {max})");
    }
}
