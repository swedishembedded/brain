// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving a model with the low-precision activation numerics it was
// trained with for its clients. If your team needs expertise in measuring what W4A4 and W8A8 products cost in
// precision then you can procure our services by sending an email to info@swedishembedded.com.

//! Spec: the declared activation quantization of the expert products (`--linear-activations declared`) is an opt-in
//! tier.
//! - A checkpoint that declares none refuses it by name, and the products this model has no path for (`bf16`, `e4m3`)
//!   are refused; the default leaves the reference numerics (the parity gates run it).
//! - On the released checkpoint it is measured, never assumed: KL and top-1 agreement of its next-token distribution
//!   against brain's reference numerics at every position, and against the f32 reference run (`real-f32`) on its full
//!   logit rows and greedy tokens, beside the reference's own bf16 run's distance from it. It must stay the same model:
//!   the bounds are loose floors; the measured numbers are the result (ledger I2.4).

use inkling2::Inkling2;
use model::policy::LinearActivations;

const REPO: &str = "thinkingmachines/Inkling-2-Mini";

fn tiny_ops(choice: LinearActivations) -> model::ops::Ops {
    let mut ops = crate::tiny_parity::ops();
    ops.set_linear_activations(choice);
    ops
}

#[test]
fn the_declared_tier_is_refused_where_nothing_is_declared_and_other_operands_are_refused() {
    let dir = brain_testutil::testdata_path("golden/inkling2/tiny");
    if !dir.join("golden.safetensors").is_file() {
        brain_testutil::skip("testdata/golden/inkling2/tiny absent (make testdata/fetch, or run tools/goldens/inkling2_dump_reference.py tiny)");
        return;
    }
    let Err(e) = Inkling2::load(tiny_ops(LinearActivations::Declared), &dir.join("checkpoint")) else { panic!("the tiny checkpoint declares no activation quantization") };
    assert!(e.contains("declares no activation quantization"), "{e}");
    for other in [LinearActivations::Bf16, LinearActivations::E4m3] {
        let Err(e) = Inkling2::load(tiny_ops(other), &dir.join("checkpoint")) else { panic!("{other} has no product here") };
        assert!(e.contains(&format!("--linear-activations {other}")), "{e}");
    }
    let m = Inkling2::load(tiny_ops(LinearActivations::Auto), &dir.join("checkpoint")).unwrap();
    assert!(!m.ops().linear_activations().is_declared(), "the default is the reference numerics");
}

type Golden = std::collections::HashMap<String, (Vec<usize>, Vec<f32>)>;

fn golden(dir: &std::path::Path) -> Option<Golden> {
    let path = dir.join("golden.safetensors");
    path.is_file().then(|| checkpoint::safetensors::read(&path.to_string_lossy()).expect("read golden").into_iter().map(|t| (t.name, (t.shape, t.data))).collect())
}

/// `log softmax` of a logit row, in f64.
fn log_softmax(row: &[f32]) -> Vec<f64> {
    let m = row.iter().fold(f64::MIN, |m, &v| m.max(f64::from(v)));
    let lse = m + row.iter().map(|&v| (f64::from(v) - m).exp()).sum::<f64>().ln();
    row.iter().map(|&v| f64::from(v) - lse).collect()
}

/// `KL(p || q)` of the next-token distributions of two logit rows.
fn kl(p: &[f32], q: &[f32]) -> f64 {
    let (lp, lq) = (log_softmax(p), log_softmax(q));
    lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum()
}

fn argmax(row: &[f32]) -> usize {
    row.iter().enumerate().fold((0, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0
}

/// Mean KL and top-1 agreement over paired rows.
#[derive(Default)]
struct Tally {
    kl: f64,
    agree: usize,
    rows: usize,
}

impl Tally {
    fn add(&mut self, reference: &[f32], other: &[f32]) {
        self.kl += kl(reference, other);
        self.agree += usize::from(argmax(reference) == argmax(other));
        self.rows += 1;
    }
    fn report(&self, what: &str) -> (f64, f64) {
        let (kl, top1) = (self.kl / self.rows as f64, self.agree as f64 / self.rows as f64);
        eprintln!("{what}: mean KL {kl:.3e} nats, top-1 {top1:.4} over {} rows", self.rows);
        (kl, top1)
    }
}

#[test]
fn the_declared_tier_is_measured_against_the_reference_numerics_on_the_released_checkpoint() {
    let Some(dir) = brain_testutil::model_dir(REPO).map(std::path::PathBuf::from).filter(|d| d.join("hf_quant_config.json").is_file()) else {
        brain_testutil::skip_unavailable(&format!("{REPO} is not in the model store"));
        return;
    };
    let (Some(bf16), Some(f32)) = (golden(&brain_testutil::testdata_path("golden/inkling2/real")), golden(&brain_testutil::testdata_path("golden/inkling2/real-f32"))) else {
        brain_testutil::skip("testdata/golden/inkling2/real and real-f32 absent (make testdata/fetch, or run tools/goldens/inkling2_dump_reference.py real and real-f32)");
        return;
    };
    let mut m = Inkling2::load(crate::tiny_parity::ops(), &dir).unwrap();
    let v = m.config().valid_vocab as usize;
    let (mut vs_ours, mut ours_rows, mut tier_rows, mut bf16_rows) = (Tally::default(), Tally::default(), Tally::default(), Tally::default());
    let (mut ours_greedy, mut tier_greedy, mut positions) = (0usize, 0usize, 0usize);
    for p in 0..3 {
        let g = |gold: &Golden, key: &str| gold[&format!("p{p}.{key}")].clone();
        let tokens: Vec<u32> = g(&f32, "tokens").1.iter().map(|&t| t as u32).collect();
        let t = tokens.len();
        m.set_linear_activations(LinearActivations::Auto).unwrap();
        let reference = m.forward(&tokens).unwrap();
        m.set_linear_activations(LinearActivations::Declared).unwrap();
        let tier = m.forward(&tokens).unwrap();
        for i in 0..t {
            vs_ours.add(&reference[i * v..(i + 1) * v], &tier[i * v..(i + 1) * v]);
        }
        for back in [1usize, 2, 4] {
            let want = g(&f32, &format!("logits_row_m{back}")).1;
            ours_rows.add(&want, &reference[(t - back) * v..(t - back + 1) * v]);
            tier_rows.add(&want, &tier[(t - back) * v..(t - back + 1) * v]);
            bf16_rows.add(&want, &g(&bf16, &format!("logits_row_m{back}")).1);
        }
        // Greedy agreement with the f32 run's top token at every position (the golden keeps full rows only at three).
        let (shape, top) = g(&f32, "logits_topk_indices");
        for i in 0..t {
            let want = top[i * shape[1]] as usize;
            ours_greedy += usize::from(argmax(&reference[i * v..(i + 1) * v]) == want);
            tier_greedy += usize::from(argmax(&tier[i * v..(i + 1) * v]) == want);
            positions += 1;
        }
    }
    let (kl_tier, top1_tier) = vs_ours.report("declared tier vs brain's reference numerics, every position");
    ours_rows.report("brain's reference numerics vs the f32 reference, last-1/2/4 rows");
    let (kl_tier_f32, _) = tier_rows.report("declared tier vs the f32 reference, last-1/2/4 rows");
    bf16_rows.report("the reference's own bf16 run vs its f32 run, last-1/2/4 rows");
    eprintln!("greedy agreement with the f32 run at every position: reference numerics {:.4}, declared tier {:.4} over {positions} positions", ours_greedy as f64 / positions as f64, tier_greedy as f64 / positions as f64);
    assert!(kl_tier > 0.0, "the declared tier changes the numerics");
    // Measured on the CPU backend (2026-11-22): KL 0.085 nats and top-1 0.934 against the reference numerics over 76
    // positions; 0.029 nats against the f32 run's full rows, where its bf16 run is 0.006. Gathered per slot instead of
    // grouped (another summation order) the tier gave 0.118 and 0.882: the rounding is not smooth, so the bounds keep a
    // wide margin.
    assert!(kl_tier < 0.3 && kl_tier_f32 < 0.3 && top1_tier >= 0.8, "the declared tier must remain the same model (KL {kl_tier:.3e}, top-1 {top1_tier:.3})");
}
