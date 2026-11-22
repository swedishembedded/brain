// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements parity-proven ports of looped mixture-of-experts models for its clients. If your
// team needs expertise in serving a published NVFP4/MXFP8 release at its reference numerics then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: the published checkpoint, served as released (NVFP4 and MXFP8 expert banks read in place, bf16 dense
//! weights, f32 activations), reproduces the reference on three chat prompts at every one of its 48 visits, in its
//! router's choices and in its logits.
//!
//! The thresholds come from the reference's own numerical spread: the same dump run in bf16 (`real`, how the release
//! is normally run) and in f32 (`real-f32`, experts dequantized to f32) differ by up to 8e-2 of a visit's scale and
//! disagree on up to 4% of the router's choices. brain computes in f32, so it is held to the f32 run: at every visit
//! its distance from that run must be a hundredth of the bf16 run's, its router must agree with that run at least as
//! well as the bf16 run does, and its greedy token must be the f32 run's at every position.

use std::collections::HashMap;

use inkling2::{Inkling2, Probe};

const REPO: &str = "thinkingmachines/Inkling-2-Mini";

/// brain's distance from the f32 reference, as a fraction of the reference's own bf16-versus-f32 distance at the same
/// tensor. Measured on the CPU backend: at most a thousandth (every visit, the full logit rows), so this keeps a
/// tenfold margin.
const SPREAD_FRACTION: f32 = 0.01;

struct Recorder(HashMap<String, Vec<f32>>);

impl Probe for Recorder {
    fn wants(&self, key: &str) -> bool {
        key.ends_with(".in") || key.ends_with(".out") || key.ends_with(".router_indices")
    }
    fn record(&mut self, key: &str, values: Vec<f32>) {
        self.0.insert(key.to_string(), values);
    }
}

type Golden = HashMap<String, (Vec<usize>, Vec<f32>)>;

fn golden(dir: &std::path::Path) -> Option<Golden> {
    let path = dir.join("golden.safetensors");
    if !path.is_file() {
        return None;
    }
    Some(checkpoint::safetensors::read(&path.to_string_lossy()).expect("read golden").into_iter().map(|t| (t.name, (t.shape, t.data))).collect())
}

/// `|got - want|_inf / |want|_inf`.
fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(f32::MIN_POSITIVE, |m, v| m.max(v.abs()));
    got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) / scale
}

/// The fraction of `(token, expert)` choices two routers share.
fn agreement(a: &[f32], b: &[f32], k: usize) -> f32 {
    let same: usize = a.chunks(k).zip(b.chunks(k)).map(|(x, y)| x.iter().filter(|e| y.contains(e)).count()).sum();
    same as f32 / a.len() as f32
}

fn argmax(row: &[f32]) -> usize {
    row.iter().enumerate().fold((0, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0
}

#[test]
fn the_released_checkpoint_matches_the_reference_at_every_visit() {
    let Some(dir) = brain_testutil::model_dir(REPO).map(std::path::PathBuf::from).filter(|d| d.join("hf_quant_config.json").is_file()) else {
        brain_testutil::skip_unavailable(&format!("{REPO} is not in the model store"));
        return;
    };
    let (Some(bf16), Some(f32)) = (golden(&brain_testutil::testdata_path("golden/inkling2/real")), golden(&brain_testutil::testdata_path("golden/inkling2/real-f32"))) else {
        brain_testutil::skip("testdata/golden/inkling2/real and real-f32 absent (make testdata/fetch, or run tools/goldens/inkling2_dump_reference.py real and real-f32)");
        return;
    };
    let m = Inkling2::load(crate::tiny_parity::ops(), &dir).unwrap();
    let (k, v) = (m.config().top_k as usize, m.config().valid_vocab as usize);
    for p in 0..3 {
        let g = |gold: &Golden, key: &str| gold[&format!("p{p}.{key}")].1.clone();
        let tokens: Vec<u32> = g(&f32, "tokens").iter().map(|&t| t as u32).collect();
        assert_eq!(g(&f32, "tokens"), g(&bf16, "tokens"), "the two reference runs read the same prompt");
        let mut rec = Recorder(HashMap::new());
        let logits = m.forward_probed(&tokens, &mut rec).unwrap();
        let mut worst = 0.0f32;
        for visit in m.plan().visits() {
            for side in ["in", "out"] {
                let key = format!("visit{}.{side}", visit.index);
                let (want, spread) = (g(&f32, &key), rel_err(&g(&bf16, &key), &g(&f32, &key)));
                let e = rel_err(&rec.0[&key], &want);
                worst = worst.max(e / spread);
                assert!(e <= SPREAD_FRACTION * spread, "prompt {p} {key}: {e:.3e} from the f32 reference, whose bf16 run is {spread:.3e} from it");
            }
            let key = format!("visit{}.router_indices", visit.index);
            let (ours, theirs) = (agreement(&rec.0[&key], &g(&f32, &key), k), agreement(&g(&bf16, &key), &g(&f32, &key), k));
            assert!(ours >= theirs, "prompt {p} {key}: brain shares {ours:.3} of the f32 reference's choices, its bf16 run {theirs:.3}");
        }
        let t = tokens.len();
        for back in [1usize, 2, 4] {
            let key = format!("logits_row_m{back}");
            let row = &logits[(t - back) * v..(t - back + 1) * v];
            let (want, spread) = (g(&f32, &key), rel_err(&g(&bf16, &key), &g(&f32, &key)));
            let e = rel_err(row, &want);
            assert!(e <= SPREAD_FRACTION * spread, "prompt {p} {key}: {e:.3e} from the f32 reference, its bf16 run {spread:.3e}");
            assert_eq!(argmax(row), argmax(&want), "prompt {p} {key}: the greedy token");
            worst = worst.max(e / spread);
        }
        let (shape, top) = &f32[&format!("p{p}.logits_topk_indices")];
        let agree = (0..t).filter(|&i| argmax(&logits[i * v..(i + 1) * v]) == top[i * shape[1]] as usize).count();
        assert_eq!(agree, t, "prompt {p}: the greedy token at every position");
        eprintln!("prompt {p}: {t} tokens, worst distance {worst:.3} of the reference's own bf16 spread");
    }
}
