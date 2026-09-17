// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Where a decision training step spends its time - per STAGE, and per kernel
//! kind within the two device passes.
//!
//! A profile of the device work alone cannot explain this model's step cost,
//! because the shape it runs at is tiny: a Banking77 state is a couple of dozen
//! tokens and a question is a handful of short slots, so the encoder's GEMMs
//! are small enough that a step can be dominated by things that are not
//! arithmetic at all - re-recording the dispatch tape, draining the queue
//! between the halves, or reading results back. So the first table here is the
//! HOST-side stage breakdown - wall minus device is the number to get BEFORE
//! ranking any kernel, because no per-kernel table can see the time the card
//! spent idle - and the per-kernel tables come second.
//!
//! Every timed region is `poll_wait`-bracketed; a bare submit only queues.
//!
//! Swedish Embedded AB implements profiling and optimization of on-device
//! neural network training for its clients. If your team needs expertise in
//! GPU kernel performance work, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! Usage: decide_bench [options] [reps] [state-tokens]

use std::time::Instant;

use decide::decide::{Decide, Limits, Request};
use decide::loss::{decision_loss, LossConfig};
use decide::primitives::{Opt, Question};

/// A Banking77 message, roughly the length of the median one.
const STATE: &str = "I still have not received the new card I ordered three weeks ago, \
                     and the tracking page has not changed since the day it shipped.";

const OPTIONS: &[&str] = &[
    "card arrival",
    "exchange rate",
    "card payment fee charged",
    "pin blocked",
    "top up by bank transfer charge",
    "cancel transfer",
    "declined card payment",
    "lost or stolen card",
];

const INSTRUCTIONS: &str = "which banking intent does this message express";

/// Mean and minimum of a set of samples, in milliseconds.
struct Stat {
    name: &'static str,
    min_ms: f64,
    mean_ms: f64,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n_opts: usize = args.first().and_then(|a| a.parse().ok()).unwrap_or(OPTIONS.len());
    let reps: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(10);
    let state_mult: usize = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(1);

    let tok_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/decide/tokenizer/tokenizer.json");
    let tok = match data::wordpiece::WordPiece::from_file(tok_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("SKIP: {tok_path}: {e}\n  run scripts/data/fetch-testdata.sh");
            return;
        }
    };

    // Random weights: the dispatch graph, and so the profile, depends on the
    // shapes alone. That keeps this runnable without the checkpoint.
    let cfg = decide::config::EncoderConfig::mini_lm_l6();
    let enc_init = decide::init::init_weights(&cfg, 1);
    let head_init = decide::init::init_head(&cfg, 2);
    let gpu = gpu_core::Gpu::new(decide::kern::PIPELINES);
    let roofs = gpu_core::roof::ensure(&gpu);
    let mut m = Decide::new_on(gpu, cfg, tok, Limits::default(), &enc_init, &head_init, true);

    // Real training never sees one length twice in a row: each example is a
    // different message, so the packed layout - and therefore every recorded
    // dispatch's params - changes every step. `state_mult` > 1 cycles a set of
    // lengths so the bench pays that cost instead of hiding it behind a shape
    // that repeats.
    let states: Vec<String> =
        (0..state_mult.max(1)).map(|i| STATE[..STATE.len() - i * 7].to_string()).collect();
    let state = states[0].clone();
    let q = Question::Choice {
        instructions: INSTRUCTIONS.to_string(),
        options: (0..n_opts).map(|i| Opt::new(OPTIONS[i % OPTIONS.len()])).collect(),
    };
    let qs = std::slice::from_ref(&q);
    let loss = LossConfig::cross_entropy();

    let req = m.pack_request(&state, qs).expect("pack");
    println!(
        "shape: {} packed rows ({} state + {} slot), {} spans, {} options",
        req.packed.ids.len(),
        req.state_rows,
        req.packed.ids.len() as u32 - req.state_rows,
        req.packed.spans.len(),
        req.cls_rows.len(),
    );

    // Warm up: the first call records every tape and compiles nothing else.
    for i in 0..states.len().max(2) {
        step(&mut m, &states[i % states.len()], qs, &loss);
    }

    let mut stats: Vec<Stat> = Vec::new();
    let mut whole = (f64::MAX, 0.0);
    for r in 0..reps {
        let state = &states[r % states.len()];
        let t0 = Instant::now();
        let s = timed_step(&mut m, state, qs, &loss);
        drain(&m);
        let total = t0.elapsed().as_secs_f64() * 1e3;
        whole.0 = whole.0.min(total);
        whole.1 += total / reps as f64;
        if stats.is_empty() {
            stats = s.iter().map(|&(n, _)| Stat { name: n, min_ms: f64::MAX, mean_ms: 0.0 }).collect();
        }
        for (st, &(_, ms)) in stats.iter_mut().zip(&s) {
            st.min_ms = st.min_ms.min(ms);
            st.mean_ms += ms / reps as f64;
        }
    }

    println!("\n=== train step: host stage breakdown, best of {reps} ===");
    println!("{:<34} {:>10} {:>10} {:>7}", "stage", "min ms", "mean ms", "% mean");
    println!("{}", "-".repeat(64));
    for st in &stats {
        println!(
            "{:<34} {:>10.3} {:>10.3} {:>6.1}%",
            st.name,
            st.min_ms,
            st.mean_ms,
            100.0 * st.mean_ms / whole.1
        );
    }
    println!("{:<34} {:>10.3} {:>10.3}", "WHOLE STEP", whole.0, whole.1);

    if let Some((h, mi, live)) = m.enc.gpu().step_cache_stats() {
        println!("\nstep cache: {h} hits, {mi} misses, {live} live entries");
    }

    println!(
        "\ndispatches: encoder fwd {}, encoder bwd {}, head fwd {}, head bwd {}",
        m.enc.fwd_steps().len(),
        m.enc.bwd_steps().len(),
        m.head.fwd_steps().len(),
        m.head.bwd_steps().len(),
    );

    let g = m.enc.gpu();
    for (label, steps) in [
        ("encoder forward", m.enc.fwd_steps()),
        ("encoder backward", m.enc.bwd_steps()),
    ] {
        gpu_core::profile::profile(g, label, steps, reps).print_top(roofs, 12);
    }
    let g = m.head.gpu();
    for (label, steps) in [("head forward", m.head.fwd_steps()), ("head backward", m.head.bwd_steps())] {
        gpu_core::profile::profile(g, label, steps, reps).print_top(roofs, 12);
    }
}

fn drain(m: &Decide) {
    m.enc.poll_wait();
    m.head.poll_wait();
}

fn step(m: &mut Decide, state: &str, qs: &[Question], loss: &LossConfig) {
    let ex = decide::Example { state, question: &qs[0], gold: 0 };
    m.train_step(&ex, loss, 2e-5, 1e-3).expect("step");
    drain(m);
}

/// One training step, broken at every point where the work changes character.
fn timed_step(m: &mut Decide, state: &str, qs: &[Question], loss: &LossConfig) -> Vec<(&'static str, f64)> {
    let mut out = Vec::new();
    let lap = |out: &mut Vec<_>, name, t: Instant| {
        out.push((name, t.elapsed().as_secs_f64() * 1e3));
        Instant::now()
    };

    let t = Instant::now();
    let req: Request = m.pack_request(state, qs).expect("pack");
    let t = lap(&mut out, "tokenize + pack (host)", t);

    m.enc.set_batch(&req.packed.ids, &req.packed.types, &req.packed.spans);
    m.enc.poll_wait();
    let t = lap(&mut out, "enc.set_batch (record + upload)", t);

    m.enc.forward();
    m.enc.poll_wait();
    let t = lap(&mut out, "enc.forward (device)", t);

    m.head.set_call(m.enc.hidden_buf(), Some(m.enc.seed_buf()), req.state_rows, &req.cls_rows);
    m.head.poll_wait();
    let t = lap(&mut out, "head.set_call (record)", t);

    let scores = m.head.forward();
    m.head.poll_wait();
    let t = lap(&mut out, "head.forward (device + read)", t);

    let (_, d_score) = decision_loss(&scores[..req.arity[0]], 0, loss);
    let t = lap(&mut out, "loss (host)", t);

    m.enc.zero_grads();
    m.head.zero_grads();
    drain(m);
    let t = lap(&mut out, "zero_grads", t);

    m.head.backward(m.enc.seed_buf(), &d_score);
    m.head.poll_wait();
    let t = lap(&mut out, "head.backward (device)", t);

    m.enc.backward_seeded();
    m.enc.poll_wait();
    let t = lap(&mut out, "enc.backward (device)", t);

    m.adamw(2e-5, 1e-3);
    drain(m);
    lap(&mut out, "adamw (device)", t);
    out
}
