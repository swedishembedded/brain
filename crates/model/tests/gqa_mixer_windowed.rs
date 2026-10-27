// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context inference at a
// fixed memory budget for its clients. If your team needs expertise in
// bounded-memory model architectures then you can procure our services by
// sending an email to info@swedishembedded.com.

//! `gqa_mixer_fwd_windowed` and `gqa_mixer_bwd` on its saved probabilities: the
//! window must reach the output (a query sees only the last `window` keys) and the
//! backward must carry no gradient across it. Central finite differences of
//! `L = sum(ctx_gated * d_out)` against the analytic input gradients, at
//! pairwise-distinct dims and `b > 1`, and the window-equals-length case against
//! the plain causal mixer.

use data::rng::Lcg;
use gpu_core::Gpu;
use model::block::{KernelIds, UNREGISTERED};
use model::gqa_mixer::{gqa_mixer_bwd, gqa_mixer_fwd, gqa_mixer_fwd_windowed, GqaMixerGrads, GqaMixerIds, GqaMixerShape, GqaMixerWeights, SlidingWindow};

const KERNELS: &[(&str, &str)] = &[
    ("rmsnorm", kernels::RMSNORM),
    ("rms_inv", kernels::RMS_INV),
    ("rmsnorm_dx", kernels::RMSNORM_DX),
    ("rmsnorm_dw", kernels::RMSNORM_DW),
    ("concat_split", kernels::CONCAT_SPLIT),
    ("concat2", kernels::CONCAT2),
    ("sigmoid", kernels::SIGMOID),
    ("sigmoid_bwd", kernels::SIGMOID_BWD),
    ("mul", kernels::MUL),
    ("rope2d_partial", kernels::ROPE2D_PARTIAL),
    ("gqa_scores", kernels::GQA_SCORES),
    ("gqa_scores_win", kernels::GQA_SCORES_WIN),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("gqa_apply", kernels::GQA_APPLY),
    ("gqa_bwd_dscores", kernels::GQA_BWD_DSCORES),
    ("gqa_bwd_dv", kernels::GQA_BWD_DV),
    ("gqa_bwd_dq", kernels::GQA_BWD_DQ),
    ("gqa_bwd_dk", kernels::GQA_BWD_DK),
];

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

fn ids(g: &Gpu) -> GqaMixerIds {
    GqaMixerIds {
        kernels: KernelIds {
            rmsnorm: idx(g, "rmsnorm"),
            rms_inv: idx(g, "rms_inv"),
            rmsnorm_dx: idx(g, "rmsnorm_dx"),
            rmsnorm_dx_rows: UNREGISTERED,
            rmsnorm_dw: idx(g, "rmsnorm_dw"),
            rope: UNREGISTERED,
            rope_bwd: UNREGISTERED,
            gqa_scores: idx(g, "gqa_scores"),
            gqa_apply: idx(g, "gqa_apply"),
            attn_softmax: idx(g, "attn_softmax"),
            gqa_dscores: idx(g, "gqa_bwd_dscores"),
            gqa_dv: idx(g, "gqa_bwd_dv"),
            gqa_dq: idx(g, "gqa_bwd_dq"),
            gqa_dk: idx(g, "gqa_bwd_dk"),
            silu_mul: UNREGISTERED,
            silu_da: UNREGISTERED,
            silu_db: UNREGISTERED,
            rmsnorm_rows: UNREGISTERED,
        },
        concat_split: idx(g, "concat_split"),
        concat2: idx(g, "concat2"),
        sigmoid: idx(g, "sigmoid"),
        sigmoid_bwd: idx(g, "sigmoid_bwd"),
        mul: idx(g, "mul"),
        rope2d_partial: idx(g, "rope2d_partial"),
    }
}

#[derive(Clone)]
struct Inputs {
    q_full: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    d_out: Vec<f32>,
}

struct Run {
    out: Vec<f32>,
    d_q_full: Vec<f32>,
    d_k: Vec<f32>,
    d_v: Vec<f32>,
}

/// One forward (and, with `backward`, backward) of the mixer; `window` `None` is the plain causal mixer.
fn run(g: &Gpu, shape: &GqaMixerShape, x: &Inputs, tables: &(Vec<f32>, Vec<f32>), window: Option<u32>, backward: bool) -> Run {
    let n = shape.b * shape.t;
    let (qd, kvd, hd) = (shape.qd(), shape.kvd(), shape.head_dim);
    let ids = ids(g);
    let (q_full, k, v) = (g.storage_init("q_full", &x.q_full), g.storage_init("k", &x.k), g.storage_init("v", &x.v));
    let (q_norm, k_norm) = (g.storage_init("q_norm", &vec![1.0; hd as usize]), g.storage_init("k_norm", &vec![1.0; hd as usize]));
    let (cos, sin) = (g.storage_init("cos", &tables.0), g.storage_init("sin", &tables.1));
    let w = GqaMixerWeights { q_norm: &q_norm, k_norm: &k_norm, cos: &cos, sin: &sin };
    let (out, acts) = match window {
        Some(window) => gqa_mixer_fwd_windowed(g, &ids, SlidingWindow { scores_kernel: idx(g, "gqa_scores_win"), window }, shape, &w, &q_full, &k, &v, n, backward),
        None => gqa_mixer_fwd(g, &ids, shape, &w, &q_full, &k, &v, n, backward),
    };
    let mut r = Run { out: g.read(&out, (n * qd) as usize), d_q_full: vec![], d_k: vec![], d_v: vec![] };
    if backward {
        let d_out = g.storage_init("d_out", &x.d_out);
        let gw = GqaMixerGrads { q_norm: None, k_norm: None };
        let (dq, dk, dv) = gqa_mixer_bwd(g, &ids, shape, &w, &gw, &acts.expect("training saves activations"), &d_out, n);
        r.d_q_full = g.read(&dq, (n * 2 * qd) as usize);
        r.d_k = g.read(&dk, (n * kvd) as usize);
        r.d_v = g.read(&dv, (n * kvd) as usize);
    }
    r
}

fn draw(shape: &GqaMixerShape, rng: &mut Lcg) -> (Inputs, (Vec<f32>, Vec<f32>)) {
    let n = (shape.b * shape.t) as usize;
    let (qd, kvd, half) = (shape.qd() as usize, shape.kvd() as usize, shape.rotary_half as usize);
    let x = Inputs { q_full: rng.vec_scaled(n * 2 * qd, 1.0), k: rng.vec_scaled(n * kvd, 1.0), v: rng.vec_scaled(n * kvd, 1.0), d_out: rng.vec_scaled(n * qd, 1.0) };
    (x, (rng.vec_scaled(n * half, 1.0), rng.vec_scaled(n * half, 1.0)))
}

fn shape() -> GqaMixerShape {
    GqaMixerShape { b: 2, t: 19, n_heads: 6, n_kv_heads: 2, head_dim: 12, rotary_half: 5, rms_eps: 1e-6 }
}

/// A window as long as the sequence is the plain causal mask.
#[test]
fn a_window_covering_the_sequence_is_the_plain_causal_mixer() {
    let g = Gpu::new_cpu(KERNELS);
    let s = shape();
    let (x, tables) = draw(&s, &mut Lcg::new(0x77aa_0001)); // seed: fixed so a failure reproduces
    let plain = run(&g, &s, &x, &tables, None, true);
    let wide = run(&g, &s, &x, &tables, Some(s.t), true);
    for (name, a, b) in [("out", &plain.out, &wide.out), ("d_q_full", &plain.d_q_full, &wide.d_q_full), ("d_k", &plain.d_k, &wide.d_k), ("d_v", &plain.d_v, &wide.d_v)] {
        let worst = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(worst < 1e-5, "{name}: window = length differs from plain causal by {worst}");
    }
}

/// Finite differences through the windowed forward, and the window itself: a key further back than the window
/// from every query gets no gradient.
#[test]
fn windowed_backward_matches_finite_differences_and_carries_nothing_across_the_window() {
    let g = Gpu::new_cpu(KERNELS);
    let s = shape();
    let window = 6u32;
    let (x, tables) = draw(&s, &mut Lcg::new(0x77aa_0002)); // seed: fixed so a failure reproduces
    let an = run(&g, &s, &x, &tables, Some(window), true);
    let plain = run(&g, &s, &x, &tables, None, true);
    assert!(an.out.iter().zip(&plain.out).any(|(a, b)| (a - b).abs() > 1e-3), "the window must change the output at this length");

    let loss = |x: &Inputs| -> f64 { run(&g, &s, x, &tables, Some(window), false).out.iter().zip(&x.d_out).map(|(o, d)| f64::from(*o) * f64::from(*d)).sum() };
    let eps = 1e-2f32;
    let mut rng = Lcg::new(0x77aa_0003); // seed: fixed so a failure reproduces
    let mut live = 0;
    let mut probe = |name: &str, len: usize, analytic: &[f32], pick: &dyn Fn(&mut Inputs) -> &mut Vec<f32>| {
        for _ in 0..24 {
            let i = (rng.next_u64() % len as u64) as usize;
            let (mut plus, mut minus) = (x.clone(), x.clone());
            pick(&mut plus)[i] += eps;
            pick(&mut minus)[i] -= eps;
            let fd = ((loss(&plus) - loss(&minus)) / (2.0 * f64::from(eps))) as f32;
            assert!((fd - analytic[i]).abs() <= 2e-2 * analytic[i].abs().max(fd.abs()) + 2e-3, "{name}[{i}]: finite difference {fd} vs analytic {}", analytic[i]);
            live += usize::from(analytic[i].abs() > 1e-2);
        }
    };
    probe("q_full", x.q_full.len(), &an.d_q_full, &|m| &mut m.q_full);
    probe("k", x.k.len(), &an.d_k, &|m| &mut m.k);
    probe("v", x.v.len(), &an.d_v, &|m| &mut m.v);
    assert!(live >= 24, "only {live} of 72 sampled gradients were non-trivial: the check would pass on a dead backward");

    // Causality of the window in the other direction: moving the queries of rows `window..` leaves the outputs of
    // the first `window` rows untouched.
    let mut moved = x.clone();
    let qpd = 2 * s.qd() as usize;
    for q in window as usize..s.t as usize {
        moved.q_full[q * qpd..(q + 1) * qpd].iter_mut().for_each(|c| *c += 0.5);
    }
    let base = run(&g, &s, &x, &tables, Some(window), false);
    let after = run(&g, &s, &moved, &tables, Some(window), false);
    let first = |r: &Run| r.out[..window as usize * s.qd() as usize].to_vec();
    assert_eq!(first(&base), first(&after), "queries inside the first window must not see queries beyond it");
}
