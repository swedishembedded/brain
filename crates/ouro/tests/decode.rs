// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements decode-equivalence gates for recurrent-depth
// language models for its clients. If your team needs expertise in proving a
// cached recurrent pass equals the full forward then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The per-pass KV cache is the serving contract: a decode step at position
//! `p` must produce exactly what the full-sequence forward produced at row
//! `p`, for every pass, even though pass `p` attends to keys written by an
//! earlier pass's state at the same positions.

use ouro::config::OuroConfig;
use ouro::decode::Decode;
use ouro::init::init_weights;
use ouro::model::Ouro;

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((*x - *y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt() as f32
}

#[test]
fn cached_decode_matches_the_full_forward_at_every_position_and_pass() {
    let cfg = OuroConfig::tiny();
    let w = init_weights(&cfg, 21);
    let tokens: Vec<u32> = vec![5, 7, 11, 13, 17];
    let n = tokens.len() as u32;
    let (d, v) = (cfg.d_model as usize, cfg.vocab as usize);
    let t = cfg.total_ut_steps as usize;

    // Full-sequence reference: every pass's logits for every row.
    let base = Ouro::new_inference(cfg.clone(), 1, n, &w);
    base.set_tokens(&tokens);
    base.trunk_forward();
    let mut want_logits = Vec::new();
    let mut want_hidden = Vec::new();
    for p in 0..t {
        base.head_forward(p);
        want_logits.push(base.read_logits());
        want_hidden.push(base.read_hidden(p));
    }

    // One cached step per position, in order, with all passes sharing the
    // weights but not their KV histories.
    let dec = Decode::new(&base, n);
    for pos in 0..n as usize {
        let got = dec.step(tokens[pos], pos as u32);
        // The released threshold (1.0) selects the final pass.
        assert_eq!(dec.selected_pass() as usize, t - 1);
        let want = &want_logits[t - 1][pos * v..(pos + 1) * v];
        let rel = rel_l2(&got, want);
        assert!(rel < 2e-4, "position {pos}: logits rel_l2 {rel}");
        for p in 0..t {
            let hd = dec.read_hidden(p);
            let want_h = &want_hidden[p][pos * d..(pos + 1) * d];
            let rel = rel_l2(&hd, want_h);
            assert!(rel < 2e-4, "position {pos} pass {p}: hidden rel_l2 {rel}");
        }
    }
}
