// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements decode-equivalence gates for recurrent-depth
// decoders for its clients. If your team needs expertise in proving a cached
// token-at-a-time decode reproduces the full forward at every position then you
// can procure our services by sending an email to info@swedishembedded.com.

//! A token-at-a-time decode through the shared core must reproduce the
//! full-sequence forward row for row: the cached histories are indexed by
//! stage (`prelude`, every recurrent step's blocks, `coda`), and a decode step
//! starts each position's recurrence from the model's initial state, exactly
//! as the materialized forward does.

use huginn::config::HuginnConfig;
use huginn::decode::Decode;
use huginn::init::init_weights;
use huginn::model::Huginn;

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((*x - *y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt() as f32
}

fn state_vector(cfg: &HuginnConfig) -> Vec<f32> {
    // The reference's `like-init` state is a draw; any fixed draw serves.
    let w = init_weights(cfg, 41);
    w["tok.weight"][..cfg.d_model as usize].to_vec()
}

#[test]
fn one_token_at_a_time_matches_the_full_forward_at_every_position() {
    let cfg = HuginnConfig::tiny();
    let tokens = [3u32, 5, 7, 11];
    let w = init_weights(&cfg, 29);
    let state = state_vector(&cfg);

    let model = Huginn::new(cfg.clone(), 1, tokens.len() as u32, &w);
    model.set_tokens(&tokens);
    // The full forward starts every position from the same state row.
    model.set_state(&state.repeat(tokens.len()));
    let rows = model.forward_rows(cfg.mean_recurrence);

    let dec = Decode::new(&model, tokens.len() as u32, cfg.mean_recurrence);
    dec.set_state(&state);
    assert_eq!(dec.histories(), (cfg.n_prelude + cfg.n_recurrent * cfg.mean_recurrence + cfg.n_coda) as usize);

    let v = cfg.vocab as usize;
    for (pos, token) in tokens.iter().enumerate() {
        let got = dec.step(*token, pos as u32);
        let want = &rows[pos * v..(pos + 1) * v];
        // The cached path reaches one token's keys through the decode
        // score/softmax kernels over the history, the forward reaches the same
        // keys in one materialized pass; the two reduce in a different order,
        // so the gate is absolute (measured 4e-5) rather than relative to the
        // tiny checkpoint's small logits.
        let abs = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(abs < 1e-4, "position {pos}: logits max_abs {abs} (rel_l2 {})", rel_l2(&got, want));
    }
}

/// The backend the test binary actually got, not the one the run intended.
#[test]
fn the_model_runs_on_the_backend_the_run_requested() {
    let want = std::env::var("BRAIN_BACKEND").unwrap_or_default().trim().to_ascii_lowercase();
    if want.is_empty() {
        return;
    }
    let cfg = HuginnConfig::tiny();
    let w = init_weights(&cfg, 17);
    let model = Huginn::new(cfg, 1, 2, &w);
    let _dec = Decode::new(&model, 2, 1);
    let got = gpu_core::backend_name();
    eprintln!("requested backend {want:?}, running on {got:?}");
    assert_eq!(got, want, "asked for {want}, ran on {got}");
}
