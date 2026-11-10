// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The decoder against teacher-forced logits from the numpy reference, and
//! the incremental path against the full recompute.

use std::sync::OnceLock;

use mirage::cact::CactFile;
use mirage::cactmap::CactModel;
use mirage::decoder::hadamard::{legacy_permutation, PERMUTATION_SEEDS};
use mirage::decoder::{Decoder, Positions};
use mirage::vision::weights::Weights;

use crate::common::{released_dir, RELEASED_ARCHITECTURE_JSON};
use crate::vision::{assert_stage, compare};

type Golden = std::collections::HashMap<String, Vec<f32>>;

fn decoder(test: &str) -> Option<&'static Decoder> {
    static CELL: OnceLock<Option<Decoder>> = OnceLock::new();
    let d = CELL.get_or_init(|| {
        let ck = mirage::import::load_dir(&released_dir()?, &mirage::import::ImportOptions::default()).ok()?;
        Some(Decoder::new(&ck.config, &Weights::new(&ck)))
    });
    if d.is_none() {
        brain_testutil::skip(&format!("{test}: the released checkpoint is absent (make testdata/fetch)"));
    }
    d.as_ref()
}

pub fn golden(test: &str, name: &str) -> Option<Golden> {
    let path = brain_testutil::testdata(&format!("golden/mirage/goldens/{name}.safetensors"));
    if !std::path::Path::new(&path).is_file() {
        brain_testutil::skip(&format!("{test}: testdata/golden/mirage/goldens/{name}.safetensors is absent (make testdata/fetch)"));
        return None;
    }
    Some(checkpoint::st::load_safetensors(&path).expect("golden loads").tensors)
}

#[test]
fn the_legacy_permutations_are_the_ones_in_the_deployment_file() {
    let Some(dir) = released_dir() else {
        brain_testutil::skip("hadamard permutations: the released checkpoint is absent");
        return;
    };
    let config = mirage::config::MirageConfig::from_json(RELEASED_ARCHITECTURE_JSON).unwrap();
    let model = CactModel::new(CactFile::open(&dir.join("mirage.cact")).unwrap(), config).unwrap();
    let (p1, p2) = model.hadamard_permutations().unwrap();
    assert_eq!(legacy_permutation(PERMUTATION_SEEDS.0, 512), p1);
    assert_eq!(legacy_permutation(PERMUTATION_SEEDS.1, 512), p2);
}

fn teacher_forced(d: &Decoder, g: &Golden) -> (Vec<u32>, Vec<f32>, usize) {
    let tokens: Vec<u32> = g["tokens"].iter().map(|&t| t as u32).collect();
    let rows = g["memory"].len() / 512;
    let cross = d.cross_memory(&g["memory"], rows);
    let logits = d.forward(&mut d.new_state(), &cross, &tokens, Positions::All);
    (tokens, logits, rows)
}

#[test]
fn teacher_forced_logits_match_the_reference() {
    let Some(d) = decoder("teacher-forced logits") else { return };
    for name in ["cats", "text"] {
        let Some(g) = golden("teacher-forced logits", name) else { return };
        let (tokens, logits, _) = teacher_forced(d, &g);
        let vocab = 8299;
        let want = &g["logits"];
        assert_eq!(logits.len(), tokens.len() * vocab);
        assert_stage(&format!("{name} logits"), &logits, want);
        let argmax = |row: &[f32]| row.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0;
        let agree = logits.chunks_exact(vocab).zip(want.chunks_exact(vocab)).filter(|(a, b)| argmax(a) == argmax(b)).count();
        eprintln!("{name}: argmax agreement {agree} of {}", tokens.len());
        assert_eq!(agree, tokens.len());
    }
}

#[test]
fn incremental_decoding_equals_the_full_recompute() {
    let Some(d) = decoder("incremental decoding") else { return };
    let Some(g) = golden("incremental decoding", "cats") else { return };
    let (tokens, full, rows) = teacher_forced(d, &g);
    let vocab = 8299;
    let cross = d.cross_memory(&g["memory"], rows);
    // A prompt chunk, then one token at a time.
    let mut st = d.new_state();
    let mut got = d.forward(&mut st, &cross, &tokens[..4], Positions::All);
    for t in &tokens[4..40] {
        got.extend(d.forward(&mut st, &cross, std::slice::from_ref(t), Positions::Last));
    }
    let (cos, worst) = compare(&got, &full[..40 * vocab]);
    eprintln!("incremental vs full: cosine {cos:.9}, max_abs {worst:.3e}");
    assert!(cos > 0.999999 && worst < 1e-2, "cosine {cos}, max_abs {worst}");
}
