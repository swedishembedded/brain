// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reference-parity gates for recurrent-depth
// decoders for its clients. If your team needs expertise in proving a released
// prelude/core/coda checkpoint matches its published reference when executed
// by a Rust engine, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Parity at scale: the released `huginn-0125` weights, a fixed starting state
//! and a fixed eight-token sequence, against `raven_modeling_minimal.py`.
//!
//! The tiny golden proves the graph; this proves the graph at the released
//! width (55 heads of 96, four core blocks, two prelude and two coda blocks),
//! where an indexing or batching mistake that cancels at width 16 does not.
//!
//! The golden comes from `tools/goldens/huginn_dump_reference.py ... --real`
//! and the checkpoint from the directory named by `BRAIN_HUGINN_DIR`. Absent
//! either, the test skips (and fails under `BRAIN_REQUIRE_FIXTURES`).

use std::collections::HashMap;

use huginn::model::Huginn;

fn load_golden(path: &str) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
    let ts = checkpoint::safetensors::read(path).expect("read golden");
    ts.into_iter().map(|t| (t.name, (t.shape, t.data))).collect()
}

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((*x - *y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt() as f32
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn the_released_checkpoint_matches_the_reference_at_every_step() {
    let dir = brain_testutil::testdata_path("golden/huginn/0125");
    if !dir.join("golden.safetensors").exists() {
        brain_testutil::skip("testdata/golden/huginn/0125 absent (run tools/goldens/huginn_dump_reference.py --real)");
        return;
    }
    let Some(ckpt) = std::env::var("BRAIN_HUGINN_DIR").ok().filter(|d| !d.is_empty()) else {
        brain_testutil::skip("BRAIN_HUGINN_DIR unset (the released checkpoint's Hugging Face directory)");
        return;
    };
    let golden = load_golden(&dir.join("golden.safetensors").to_string_lossy());
    let (cfg, src) = huginn::import::open_dir(std::path::Path::new(&ckpt)).expect("open the released checkpoint");

    let tokens: Vec<u32> = golden["tokens"].1.iter().map(|v| *v as u32).collect();
    let state = &golden["input_states"].1;
    let n = tokens.len();
    let d = cfg.d_model as usize;
    let steps = golden.keys().filter(|k| k.starts_with("logits_")).count() as u32;

    let model = Huginn::new(cfg.clone(), 1, n as u32, &src);
    model.set_tokens(&tokens);
    model.set_state(&state.repeat(n));

    for step in 1..=steps {
        let logits = model.forward_rows(step);
        let latent = model.read_latent();
        let raw = model.read_state();
        let want_s = &golden[&format!("state_{step}")].1;
        let want_l = &golden[&format!("latent_{step}")].1;
        let want_g = &golden[&format!("logits_{step}")].1;
        assert!(
            rel_l2(&raw, want_s) < 1e-4 && max_abs(&raw, want_s) < 2e-3,
            "step {step} state rel_l2 {} max_abs {}",
            rel_l2(&raw, want_s),
            max_abs(&raw, want_s)
        );
        assert!(
            rel_l2(&latent, want_l) < 1e-4 && max_abs(&latent, want_l) < 2e-3,
            "step {step} latent rel_l2 {} max_abs {}",
            rel_l2(&latent, want_l),
            max_abs(&latent, want_l)
        );
        assert!(
            rel_l2(&logits, want_g) < 1e-4 && max_abs(&logits, want_g) < 2e-3,
            "step {step} logits rel_l2 {} max_abs {}",
            rel_l2(&logits, want_g),
            max_abs(&logits, want_g)
        );
        assert_eq!(latent.len(), n * d);
    }
}
