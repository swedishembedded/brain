// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The real-checkpoint fixture the parity tests share: the model directory in
//! the store, the goldens `tools/goldens/minicpmv_dump_reference.py` wrote,
//! and the identity check that pairs the two.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;

use minicpmv::MiniCpmVConfig;

pub const REFERENCE: &str = "openbmb/MiniCPM-V-4.6";
pub const DUMPER: &str = "tools/goldens/minicpmv_dump_reference.py";

pub struct Fixture {
    pub dir: PathBuf,
    pub golden: PathBuf,
    pub cfg: MiniCpmVConfig,
}

/// The checkpoint and its goldens, or `None` after naming what is absent.
pub fn fixture() -> Option<Fixture> {
    let Some(dir) = brain_testutil::model_dir(REFERENCE).map(PathBuf::from).filter(|d| d.join("config.json").is_file()) else {
        brain_testutil::skip(&format!("{REFERENCE} is not in the model store (brain pull {REFERENCE})"));
        return None;
    };
    let golden = brain_testutil::testdata_path("minicpmv");
    let src = brain_testutil::golden::Source::open(&golden, DUMPER)?;
    let cfg = MiniCpmVConfig::from_dir(&dir).expect("read the checkpoint config");
    let identity = [
        ("text_n_layers", cfg.text.n_layers as i64),
        ("text_d_model", cfg.text.d_model as i64),
        ("vocab_size", cfg.text.vocab as i64),
        ("vision_n_layers", cfg.vision.n_layers as i64),
        ("vision_d_model", cfg.vision.hidden as i64),
        ("insert_layer_id", cfg.vision.insert_layer_id as i64),
    ];
    if !src.require(&identity) {
        return None;
    }
    Some(Fixture { dir, golden, cfg })
}

impl Fixture {
    /// Every tensor of one golden file, by name.
    pub fn tensors(&self, rel: &str) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
        let path = self.golden.join(rel);
        let path = path.to_str().expect("UTF-8 golden path");
        checkpoint::safetensors::read(path)
            .unwrap_or_else(|e| panic!("golden {path}: {e} - re-run {DUMPER}"))
            .into_iter()
            .map(|t| (t.name, (t.shape, t.data)))
            .collect()
    }

    /// The manifest entry of one case.
    pub fn manifest(&self) -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(self.golden.join("manifest.json")).expect("manifest")).expect("manifest JSON")
    }
}

pub fn ids(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| *x as u32).collect()
}

pub fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) }).0 as u32
}

/// `head [vocab, d] * hidden [d]` on the host.
pub fn head_logits(head: &[f32], hidden: &[f32]) -> Vec<f32> {
    let d = hidden.len();
    head.chunks_exact(d).map(|row| row.iter().zip(hidden).map(|(a, b)| a * b).sum()).collect()
}

/// The checkpoint directory alone, for tests that need no goldens, or
/// `None` after naming what is absent.
pub fn checkpoint() -> Option<PathBuf> {
    let dir = brain_testutil::model_dir(REFERENCE).map(PathBuf::from).filter(|d| d.join("config.json").is_file());
    if dir.is_none() {
        brain_testutil::skip(&format!("{REFERENCE} is not in the model store (brain pull {REFERENCE})"));
    }
    dir
}
