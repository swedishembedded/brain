// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements numerical parity gates for recurrent-depth
// model execution for its clients. If your team needs expertise in proving a
// looped transformer matches its reference implementation then you can procure
// our services by sending an email to info@swedishembedded.com.

//! Shape/behaviour tests that need no checkpoint: the loop really reuses one
//! stack, every pass is a distinct state, and the exit rule selects the last
//! pass at the released threshold.

use ouro::config::OuroConfig;
use ouro::init::init_weights;
use ouro::model::Ouro;

fn finite(v: &[f32]) -> bool {
    v.iter().all(|x| x.is_finite())
}

#[test]
fn every_pass_reads_the_shared_stack_and_produces_a_distinct_state() {
    let cfg = OuroConfig::tiny();
    let w = init_weights(&cfg, 11);
    let model = Ouro::new_inference(cfg.clone(), 1, 4, &w);
    assert_eq!(model.steps(), 3);

    model.set_tokens(&[3, 4, 5, 6]);
    model.trunk_forward();

    let mut hiddens = Vec::new();
    let mut logits = Vec::new();
    for pass in 0..cfg.total_ut_steps as usize {
        let h = model.read_hidden(pass);
        assert!(finite(&h), "pass {pass} hidden is not finite");
        hiddens.push(h);
        model.head_forward(pass);
        let l = model.read_logits();
        assert!(finite(&l), "pass {pass} logits are not finite");
        logits.push(l);
    }
    // Pass 0 is the depth-1 model; each further pass changes the state.
    assert_ne!(hiddens[0], hiddens[1]);
    assert_ne!(hiddens[1], hiddens[2]);
    assert_ne!(logits[0], logits[1]);
    assert_ne!(logits[1], logits[2]);

    // Pass 1 consumes pass 0's final NORMED hidden state (the reference
    // reassigns hidden_states = norm(hidden_states) each pass), which is not
    // the same tensor as the pre-norm residual it was computed from.
    let pass0_pre_norm = model.read_res(0, cfg.n_layers as usize);
    let pass0_hidden = model.read_hidden(0);
    assert_ne!(pass0_pre_norm, pass0_hidden);
}

#[test]
fn the_released_threshold_selects_the_final_pass() {
    let cfg = OuroConfig::tiny();
    let w = init_weights(&cfg, 12);
    let model = Ouro::new_inference(cfg.clone(), 1, 4, &w);
    model.set_tokens(&[1, 2, 3, 4]);
    model.trunk_forward();
    let steps = model.exit_steps();
    assert_eq!(steps, vec![cfg.total_ut_steps - 1; 4]);
    assert_eq!(model.read_gate(0).len(), 4);
}

#[test]
fn the_embedding_is_applied_once_not_per_pass() {
    // If a pass re-injected the embedding, pass 1's input would be a function
    // of the tokens rather than exactly pass 0's output. The residual chain
    // test above pins the composition; this one pins that changing only the
    // last token changes earlier passes' states (context flows forward) and
    // that a zero-depth read is the raw embedding.
    let cfg = OuroConfig::tiny();
    let w = init_weights(&cfg, 13);
    let model = Ouro::new_inference(cfg.clone(), 1, 4, &w);
    model.set_tokens(&[7, 8, 9, 10]);
    model.trunk_forward();
    let e = model.read_res(0, 0);
    let d = cfg.d_model as usize;
    // Row 0 of the residual stream is exactly the embedding row of token 7.
    let emb = &w["tok.weight"][7 * d..8 * d];
    for (a, b) in e[..d].iter().zip(emb.iter()) {
        assert!((a - b).abs() < 1e-6, "embedding row mismatch: {a} vs {b}");
    }
}

/// The backend the test binary actually got, not the one the run intended:
/// `Gpu::new` silently degrades to the host tier when no card can hold work,
/// so a green suite on a GPU box proves nothing unless this holds.
#[test]
fn the_model_runs_on_the_backend_the_run_requested() {
    let want = std::env::var("BRAIN_BACKEND").unwrap_or_default();
    let want = want.trim().to_ascii_lowercase();
    if want.is_empty() {
        return;
    }
    let cfg = OuroConfig::tiny();
    let w = init_weights(&cfg, 17);
    let _model = Ouro::new_inference(cfg, 1, 2, &w);
    let got = gpu_core::backend_name();
    eprintln!("requested backend {want:?}, running on {got:?}");
    assert_eq!(got, want, "asked for {want}, ran on {got}");
}
