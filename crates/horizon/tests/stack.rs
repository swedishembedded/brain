// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The stack backbone (`Backbone::Stack`) with Gated DeltaNet, attention and
//! hybrid mixers. Gradient checks live in `brain-gradcheck`; these hold the
//! behaviour a gradient check cannot see:
//!
//! - the gap-aware gates of the delta rule are what the specification says,
//!   padding included, and their backward matches finite differences of them;
//! - the device computes what the CPU computes (loss and every gradient);
//! - a subject's prediction does not depend on its batch neighbours, on empty
//!   slots, or on how many of them there are;
//! - a Gated DeltaNet stack forgets a history completely over a gap far longer
//!   than its time constants, because the decay is the exponential of the
//!   physical gap.

use data::rng::Rng;
use gpu_core::Gpu;
use horizon::batch::assemble;
use horizon::encode::{encode, Encoded};
use horizon::synthetic::drifting::{self, Gaps};
use horizon::vocab::{FitOptions, Vocab};
use horizon::{Backbone, Horizon, HorizonConfig, Mixer, StackConfig, PIPELINES};

const GAPS: Gaps = Gaps {
    last: (0.0, 2.0),
    between: (0.5, 3.0),
    visits: (1, 5),
};

/// The three stacks under test: all Gated DeltaNet, all attention, and the
/// 3:1 hybrid, each over `visits` visit slots.
fn stacks() -> [(&'static str, StackConfig); 3] {
    [
        ("gdn", StackConfig::new(Mixer::GatedDeltaNet, 2)),
        ("attention", StackConfig::new(Mixer::Attention, 2)),
        ("hybrid", StackConfig::new(Mixer::Hybrid, 4)),
    ]
}

fn population(n: usize, seed: u64) -> (Vec<horizon::timeline::Subject>, Vocab) {
    let (subjects, _) = drifting::population(n, seed, &GAPS, 6.0);
    let codes = vec![drifting::CODE.to_string()];
    let fit = FitOptions {
        knots: 9,
        min_count: 1,
    };
    let vocab = Vocab::fit(&subjects, &codes, &codes, &fit).unwrap();
    (subjects, vocab)
}

fn config(vocab: &Vocab, visits: u32, stack: StackConfig) -> HorizonConfig {
    let mut cfg = HorizonConfig::tiny(vocab.len(), 1);
    cfg.d_model = 16;
    cfg.n_heads = 2;
    cfg.max_tokens = 8;
    cfg.visits = visits;
    cfg.backbone = Backbone::Stack(stack);
    cfg
}

fn encoded(subjects: &[horizon::timeline::Subject], vocab: &Vocab, cfg: &HorizonConfig) -> Vec<Encoded> {
    subjects.iter().map(|s| encode(s, vocab, cfg)).collect()
}

#[test]
fn the_gap_gates_decay_by_physical_time_and_pass_padding_through() {
    let gpu = Gpu::new(PIPELINES);
    let (fwd, bwd) = (
        gpu.kernel_index("gdn_gap_gate").unwrap(),
        gpu.kernel_index("gdn_gap_gate_bwd").unwrap(),
    );
    // Two heads; rows: a visit, a longer gap, no gap, a padding row.
    let (rows, heads) = (4usize, 2usize);
    let b_pre = [0.3f32, -0.7, 1.1, 0.2, -0.4, 0.9, 0.5, 0.5];
    let dt = [0.5f32, 4.0, 0.0, -1.0];
    let rate = [0.2f32, -1.3];
    let wg = [0.7f32, -0.3, 0.2, 0.9, -0.5, 0.4, 0.8, -0.6];
    let wb = [-0.2f32, 0.5, 0.3, -0.8, 0.6, 0.1, 0.9, 0.7];
    let buf = |x: &[f32]| gpu.storage_init("t", x);
    let (b_pre_d, dt_d, rate_d) = (buf(&b_pre), buf(&dt), buf(&rate));
    let (g_d, beta_d) = (gpu.storage((rows * heads) as u64), gpu.storage((rows * heads) as u64));
    let n = (rows * heads) as u32;
    gpu.submit(
        &[],
        &[gpu.step(
            fwd,
            &[&b_pre_d, &dt_d, &rate_d, &g_d, &beta_d],
            &[rows as u32, heads as u32],
            n,
        )],
    );
    let (g, beta) = (gpu.read(&g_d, rows * heads), gpu.read(&beta_d, rows * heads));
    let softplus = |x: f64| x.max(0.0) + (-x.abs()).exp().ln_1p();
    let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
    let host = |b_pre: &[f32], rate: &[f32]| -> (Vec<f64>, Vec<f64>) {
        let mut g = vec![0.0; rows * heads];
        let mut beta = vec![0.0; rows * heads];
        for i in 0..rows * heads {
            let (r, h) = (i / heads, i % heads);
            if dt[r] >= 0.0 {
                g[i] = -softplus(rate[h] as f64) * dt[r] as f64;
                beta[i] = sigmoid(b_pre[i] as f64);
            }
        }
        (g, beta)
    };
    let (want_g, want_beta) = host(&b_pre, &rate);
    for i in 0..rows * heads {
        assert!((g[i] as f64 - want_g[i]).abs() < 1e-5, "g[{i}] {} vs {}", g[i], want_g[i]);
        assert!((beta[i] as f64 - want_beta[i]).abs() < 1e-6, "beta[{i}]");
    }
    // A longer gap decays more, in proportion; no gap and padding do not decay.
    assert!(g[2] < g[0] && g[0] < 0.0, "head 0: {} then {}", g[0], g[2]);
    assert!((g[2] / g[0] - 8.0).abs() < 1e-4, "decay is linear in the elapsed time");
    assert_eq!((g[4], g[6], beta[6], beta[7]), (0.0, 0.0, 0.0, 0.0));

    // The backward against central differences of the host forward, of the
    // loss sum(wg * g + wb * beta).
    let loss = |b_pre: &[f32], rate: &[f32]| -> f64 {
        let (g, beta) = host(b_pre, rate);
        (0..rows * heads).map(|i| wg[i] as f64 * g[i] + wb[i] as f64 * beta[i]).sum()
    };
    let (dg_d, db_d) = (buf(&wg), buf(&wb));
    let (d_b_pre_d, d_rate_d) = (gpu.storage(n as u64), gpu.storage(n as u64));
    gpu.submit(
        &[],
        &[gpu.step(
            bwd,
            &[&b_pre_d, &dt_d, &rate_d, &dg_d, &db_d, &d_b_pre_d, &d_rate_d],
            &[rows as u32, heads as u32],
            n,
        )],
    );
    let (d_b_pre, d_rate_part) = (gpu.read(&d_b_pre_d, rows * heads), gpu.read(&d_rate_d, rows * heads));
    let eps = 1e-3f32;
    for i in 0..rows * heads {
        let (mut up, mut down) = (b_pre, b_pre);
        up[i] += eps;
        down[i] -= eps;
        let fd = (loss(&up, &rate) - loss(&down, &rate)) / (2.0 * eps as f64);
        assert!((d_b_pre[i] as f64 - fd).abs() < 1e-4, "d b_pre[{i}] {} vs {fd}", d_b_pre[i]);
    }
    for h in 0..heads {
        let (mut up, mut down) = (rate, rate);
        up[h] += eps;
        down[h] -= eps;
        let fd = (loss(&b_pre, &up) - loss(&b_pre, &down)) / (2.0 * eps as f64);
        let summed: f64 = (0..rows).map(|r| d_rate_part[r * heads + h] as f64).sum();
        assert!((summed - fd).abs() < 1e-4, "d rate[{h}] {summed} vs {fd}");
    }
    assert_eq!((d_b_pre[6], d_b_pre[7], d_rate_part[6], d_rate_part[7]), (0.0, 0.0, 0.0, 0.0));
}

/// A CPU model and the selected device's, on the same weights and batch.
fn pair(stack: StackConfig, visits: u32) -> (Horizon, Horizon) {
    let (subjects, vocab) = population(400, 5);
    let cfg = config(&vocab, visits, stack);
    let enc = encoded(&subjects[..30], &vocab, &cfg);
    let refs: Vec<&Encoded> = enc.iter().collect();
    let hb = assemble(&cfg, &refs, 32, 0.4, &mut Rng::new(9));
    let init = horizon::init_weights(&cfg, 3);
    let cpu = Horizon::new_on(Gpu::new_cpu(PIPELINES), cfg.clone(), 32, &init);
    let dev = Horizon::new(cfg, 32, &init);
    for m in [&cpu, &dev] {
        m.set_batch(&hb);
        m.zero_grads();
    }
    (cpu, dev)
}

#[test]
fn the_selected_device_matches_the_cpu() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    // 3 visit slots fit one chunk; 20 span two.
    for (name, stack) in stacks() {
        for visits in [3, 20] {
            let (cpu, dev) = pair(stack, visits);
            println!("{name}, {visits} visit slots: device {}", dev.gpu.kind());
            let (a, b) = (cpu.forward(), dev.forward());
            assert!(
                (a - b).abs() <= 1e-4 * (1.0 + a.abs()),
                "{name} {visits}: loss {a} vs {b}"
            );
            cpu.backward();
            dev.backward();
            for (param, _) in cpu.ps.params.iter() {
                let (x, y) = (
                    cpu.ps.read_grad(&cpu.gpu, param),
                    dev.ps.read_grad(&dev.gpu, param),
                );
                let scale = x.iter().map(|v| v.abs()).fold(1e-6f32, f32::max);
                let worst = x.iter().zip(&y).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
                assert!(
                    worst <= 2e-3 * scale,
                    "{name} {visits}: gradient of {param} differs by {worst} (scale {scale})"
                );
            }
        }
    }
}

/// Log-hazards of each subject in `enc`, predicted with `slots` batch slots.
fn log_hazards(cfg: &HorizonConfig, init: &std::collections::HashMap<String, Vec<f32>>, enc: &[&Encoded], slots: u32) -> Vec<f32> {
    let model = Horizon::new(cfg.clone(), slots, init);
    model.set_batch(&assemble(cfg, enc, slots as usize, 0.0, &mut Rng::new(1)));
    model.forward();
    let all = model.read_log_hazards();
    let per = (cfg.pieces() * cfg.n_codes) as usize;
    all[..per * enc.len()].to_vec()
}

#[test]
fn a_subjects_prediction_does_not_depend_on_its_batch() {
    let (subjects, vocab) = population(60, 7);
    for (name, stack) in stacks() {
        let cfg = config(&vocab, 6, stack);
        let init = horizon::init_weights(&cfg, 4);
        let enc = encoded(&subjects, &vocab, &cfg);
        // Subjects with one visit and with many, so the slots in front of
        // them are unused to different degrees.
        let by_visits = |n: usize| enc.iter().find(|e| e.visits.len() == n).unwrap();
        let chosen = [by_visits(1), by_visits(2), by_visits(5), by_visits(3)];
        let per = (cfg.pieces() * cfg.n_codes) as usize;
        let together = log_hazards(&cfg, &init, &chosen, 7);
        let reversed = log_hazards(&cfg, &init, &[chosen[3], chosen[2], chosen[1], chosen[0]], 4);
        for (i, e) in chosen.iter().enumerate() {
            let alone = log_hazards(&cfg, &init, &[*e], 1);
            for (k, a) in alone.iter().enumerate() {
                let (t, r) = (together[i * per + k], reversed[(3 - i) * per + k]);
                assert!(
                    (a - t).abs() < 2e-4 && (a - r).abs() < 2e-4,
                    "{name}: subject {i} alone {a}, in a batch of 4 of 7 slots {t}, reversed in 4 slots {r}"
                );
            }
        }
    }
}

#[test]
fn a_gated_deltanet_stack_forgets_a_history_a_century_back() {
    let (subjects, vocab) = population(60, 9);
    let cfg = config(&vocab, 6, StackConfig::new(Mixer::GatedDeltaNet, 2));
    let init = horizon::init_weights(&cfg, 4);
    let enc = encoded(&subjects, &vocab, &cfg);
    let with_history = enc.iter().find(|e| e.visits.len() >= 3).unwrap();
    // The same subject whose last visit was a million years before entry
    // (every time constant of the initial rates is far shorter), and the
    // same subject with no history at all.
    let mut long_ago = with_history.clone();
    for v in &mut long_ago.visits {
        v.ago += 1e6;
    }
    let mut none = with_history.clone();
    none.visits.clear();
    let near = log_hazards(&cfg, &init, &[with_history], 1);
    let far = log_hazards(&cfg, &init, &[&long_ago], 1);
    let blank = log_hazards(&cfg, &init, &[&none], 1);
    let dist = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    assert!(dist(&far, &blank) < 1e-5, "a gap of a million units leaves {} of the history", dist(&far, &blank));
    assert!(dist(&near, &blank) > 1e-3, "the recent history must matter ({})", dist(&near, &blank));
}
