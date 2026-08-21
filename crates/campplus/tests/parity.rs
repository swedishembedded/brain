// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Forward parity vs the real `campplus.onnx` reference, dumped by
//! `tools/goldens/cosyvoice_dump_reference.py` (self-validated there by
//! running onnxruntime twice on identical input and asserting bit-exact -
//! CAM++ has no second reference implementation in this repo, so this test IS
//! the parity gate).
//!
//! Skips cleanly when the golden or the checkpoint is absent.

use std::path::{Path, PathBuf};

use brain_testutil::{golden::Source, parity::Table, testdata_path};
use campplus::config::CampplusConfig;
use campplus::import::{import_dir, RELEASE_FILE};
use campplus::model::{Campplus, PIPELINES};
use gpu_core::Gpu;

const DUMPER: &str = "tools/goldens/cosyvoice_dump_reference.py";
const COS_FLOOR: f64 = 0.9999;
// Measured on the real checkpoint: cosine 1.0000000000, rel_l2 2.3e-6,
// max_abs 5.3e-6 (CPU backend, `brain-wgsl-cpu`). 1e-4 leaves ~40x headroom
// for backend/precision variance without being a floor that would wave
// through a real regression.
const REL_CEIL: f64 = 1e-4;

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The shipped `campplus.onnx`: `BRAIN_CAMPPLUS_DIR`, else the repo-relative
/// `resources/cosyvoice/weights/` - a variable rather than a literal machine
/// path so this test passes on any checkout that ran `resources/cosyvoice/fetch.py`,
/// not just the one it was written on.
fn weights_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BRAIN_CAMPPLUS_DIR") {
        let pb = PathBuf::from(p);
        return pb.join(RELEASE_FILE).is_file().then_some(pb);
    }
    let p = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../resources/cosyvoice/weights"));
    p.join(RELEASE_FILE).is_file().then_some(p)
}

#[test]
fn real_matches_the_onnxruntime_golden() {
    let dir = testdata_path("golden/cosyvoice");
    let meta = dir.join("campplus_real_meta.json");
    let Some(src) = Source::open_manifest(&meta, DUMPER) else { return };
    let cfg = CampplusConfig::campplus_v2();
    if !src.require(&[("spk_embed_dim", cfg.embedding_size as i64)]) {
        return;
    }
    let Some(wdir) = weights_dir() else {
        brain_testutil::skip(&format!("set BRAIN_CAMPPLUS_DIR to a directory containing {RELEASE_FILE}"));
        return;
    };

    let weights = import_dir(&wdir).unwrap_or_else(|e| panic!("import {}: {e}", wdir.display()));
    let input = read_f32(&dir.join("campplus_real_in.f32"));
    let want = read_f32(&dir.join("campplus_real_out.f32"));
    assert_eq!(
        input.len() as u32 % cfg.feat_dim,
        0,
        "campplus_real_in.f32 ({} values) is not a whole number of {}-dim frames",
        input.len(),
        cfg.feat_dim
    );
    let t = input.len() as u32 / cfg.feat_dim;

    let gpu = Gpu::new_cpu(PIPELINES);
    let m = Campplus::new(gpu, cfg, &weights);
    let got = m.forward(&input, t);

    assert_eq!(got.len(), want.len(), "x-vector length mismatch: got {} want {}", got.len(), want.len());
    let mut table = Table::new(COS_FLOOR, REL_CEIL);
    table.check("campplus_real", &got, &want);
    table.print();
    table.assert_clean();
}
