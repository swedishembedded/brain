// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements low-rank adapter fine-tuning that trains exactly the projections it is asked to
// for its clients. If your team needs expertise in parameter-efficient fine-tuning then you can procure our services
// by sending an email to info@swedishembedded.com.

//! The LoRA contract, on the CPU backend in fp32:
//!
//! * the adapted forward is the base with `W + s * B A` in place of each adapted `W`, `s = alpha / rank`
//!   (`alpha / sqrt(rank)` under rsLoRA), `A` of `[rank, in]` and `B` of `[out, rank]`: the live adapter's logits
//!   equal a plain model's whose weights the test merges on the host from the formula;
//! * gradients reach exactly the configured projections (all seven by default, or the named subset), both factors
//!   of each, and nothing else: no other weight changes over training steps;
//! * a fresh adapter (`B = 0`) leaves the base's logits bit for bit, and its first gradient reaches only `B`
//!   (`dL/dA = s * B^T dL/dY X^T` is zero while `B` is).
//!
//! Each check is shown to fail on a mutant: the merge at the wrong scale or with `B A` transposed, and an adapter
//! built on a target set other than the configured one.

use std::collections::{BTreeSet, HashMap};

use data::rng::Lcg;
use model::adapter::device::is_adapter_param;
use model::adapter::kind::{LORA_A, LORA_B};
use model::{Batch, Model};
use qwen3::{LoraCfg, Qwen, QwenConfig};

const PROMPT: [u32; 8] = [3, 17, 5, 9, 22, 1, 14, 8];
/// Rank 3 is coprime with every width of the tiny config, so a transposed factor cannot hide in a square shape.
const RANK: u32 = 3;
const ALPHA: f32 = 6.0;

fn targets(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| s.to_string()).collect()
}

fn adapted(names: &[&str], rank_stabilized: bool) -> QwenConfig {
    QwenConfig { lora: Some(LoraCfg { rank_stabilized, ..LoraCfg::new(RANK, ALPHA, targets(names)) }), ..QwenConfig::tiny() }
}

/// Weights for `cfg` with both factors of every adapter drawn non-zero, so the adapter changes the forward and both
/// factors carry a gradient.
fn drawn_adapters(cfg: &QwenConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut w = qwen3::init_weights(cfg, seed);
    let mut rng = Lcg::new(seed ^ 0x5eed);
    for (name, v) in w.iter_mut() {
        if name.ends_with(LORA_A) || name.ends_with(LORA_B) {
            v.iter_mut().for_each(|x| *x = rng.scaled(0.3));
        }
    }
    w
}

/// The base weights of `w` with every adapter merged by the formula, computed here on the host: `W[o][i] +
/// scale * sum_k B[o][k] A[k][i]`.
fn merged_by_formula(cfg: &QwenConfig, w: &HashMap<String, Vec<f32>>, scale: f32, transposed: bool) -> HashMap<String, Vec<f32>> {
    let r = RANK as usize;
    let mut out: HashMap<String, Vec<f32>> = w.iter().filter(|(n, _)| !is_adapter_param(n)).map(|(n, v)| (n.clone(), v.clone())).collect();
    let plain = QwenConfig { lora: None, ..cfg.clone() };
    for (name, numel) in plain.param_list() {
        let (Some(a), Some(b)) = (w.get(&format!("{name}{LORA_A}")), w.get(&format!("{name}{LORA_B}"))) else { continue };
        let (inn, outn) = (a.len() / r, b.len() / r);
        assert_eq!(inn * outn, numel, "{name}: the factors fit the base");
        let base = out.get_mut(&name).unwrap();
        for o in 0..outn {
            for i in 0..inn {
                let ba: f64 = (0..r).map(|k| f64::from(b[o * r + k]) * f64::from(a[k * inn + i])).sum();
                // The transposed mutant reads B A as if it were laid out [in, out].
                let at = if transposed { (i * outn + o) % numel } else { o * inn + i };
                base[at] += scale * ba as f32;
            }
        }
    }
    out
}

fn max_rel(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
    got.iter().zip(want).fold(0f32, |m, (a, b)| m.max((a - b).abs())) / scale
}

/// The live adapter's logits against a plain model's whose weights are merged by the formula at `scale`.
fn live_against_formula(rank_stabilized: bool, scale: f32, transposed: bool) -> f32 {
    let cfg = adapted(&qwen3::finetune::LORA_TARGETS, rank_stabilized);
    let w = drawn_adapters(&cfg, 31);
    let live = Qwen::new(cfg.clone(), 1, 12, &w).logits_all(&PROMPT);
    let plain = Qwen::new(QwenConfig::tiny(), 1, 12, &merged_by_formula(&cfg, &w, scale, transposed)).logits_all(&PROMPT);
    let base = Qwen::new(QwenConfig::tiny(), 1, 12, &merged_by_formula(&cfg, &w, 0.0, false)).logits_all(&PROMPT);
    assert!(max_rel(&live, &base) > 1e-2, "the drawn adapters move the logits, so agreement with the formula means something");
    max_rel(&live, &plain)
}

/// Band of the merged-against-live comparison: the two compute `W x + s B (A x)` and `(W + s B A) x`, the same sum
/// in different association orders, so they agree within the binary32 agreement bound of the forward's longest chain
/// of rounded operations; the tiny config's (two layers of norms, projections of at most 64 terms, attention over 8
/// positions, a head of 23) is under 512. Measured: about 1e-6.
fn merge_band() -> f32 {
    brain_testutil::training::step_bound(512) as f32
}

#[test]
fn the_live_adapter_is_the_base_with_the_formula_merged_in() {
    let band = merge_band();
    let lora = live_against_formula(false, ALPHA / RANK as f32, false);
    assert!(lora <= band, "LoRA: live logits are {lora:e} off W + (alpha/rank) B A");
    let rs = live_against_formula(true, ALPHA / (RANK as f32).sqrt(), false);
    eprintln!("live against the formula: LoRA {lora:e}, rsLoRA {rs:e} (band {band:e})");
    assert!(rs <= band, "rsLoRA: live logits are {rs:e} off W + (alpha/sqrt(rank)) B A");
    for (what, off) in [("scale alpha instead of alpha/rank", live_against_formula(false, ALPHA, false)), ("B A transposed", live_against_formula(false, ALPHA / RANK as f32, true)), ("rsLoRA at alpha/rank", live_against_formula(true, ALPHA / RANK as f32, false))] {
        assert!(off > band, "mutant {what:?} survived: {off:e}");
        eprintln!("mutant {what:?} killed: live logits {off:e} off the mutant merge");
    }
}

/// The base tensors carrying an adapter in `m`, and whether every factor got a non-zero gradient from one step.
fn adapted_sites(m: &Qwen) -> (BTreeSet<String>, Vec<String>) {
    let trained = Model::optimized_params(m).expect("an adapter build names its trained tensors");
    assert!(trained.iter().all(|n| is_adapter_param(n)), "an adapter build trains adapter tensors only: {trained:?}");
    let sites = trained.iter().filter_map(|n| n.strip_suffix(LORA_A)).map(str::to_string).collect();
    let silent = trained.iter().filter(|n| m.read_grad(n).iter().all(|&g| g == 0.0)).cloned().collect();
    (sites, silent)
}

/// The sites a configuration names: each target projection of every layer.
fn expected_sites(cfg: &QwenConfig, names: &[&str]) -> BTreeSet<String> {
    let leaf = |n: &str| match n {
        "wq" | "wk" | "wv" | "wo" => format!("attn.{n}"),
        _ => format!("mlp.{n}"),
    };
    (0..cfg.n_layers).flat_map(|l| names.iter().map(move |n| format!("blocks.{l}.{}.weight", leaf(n)))).collect()
}

/// One forward and backward on a fixed batch, then `steps` AdamW steps; whether every non-adapter weight is still
/// bit for bit what it was.
fn train_steps(m: &Qwen, steps: u32) -> Option<String> {
    let x: Vec<u32> = (0..12).map(|i| (i * 7 + 1) % 23).collect();
    let y: Vec<u32> = (0..12).map(|i| (i * 5 + 2) % 23).collect();
    let base: Vec<(String, Vec<f32>)> = m.param_names().into_iter().filter(|n| !is_adapter_param(n)).map(|n| (n.clone(), m.read_weight(&n))).collect();
    Model::set_batch(m, Batch::Lm { tokens: &x, targets: &y });
    for step in 1..=steps {
        m.zero_grads();
        Model::forward(m);
        Model::backward(m);
        if step < steps {
            Model::adamw_step(m, step, 1e-2, 0.1, model::Adam::default(), Some(1.0), 1.0);
        }
    }
    m.poll_wait();
    base.into_iter().find(|(n, w)| m.read_weight(n).iter().zip(w).any(|(a, b)| a.to_bits() != b.to_bits())).map(|(n, _)| n)
}

/// Whether `m` adapts exactly `names` and every factor of every site received a gradient; `Err` names the first
/// difference.
fn trains_exactly(m: &Qwen, names: &[&str]) -> Result<(), String> {
    let (sites, silent) = adapted_sites(m);
    let want = expected_sites(&m.cfg, names);
    if sites != want {
        let missing: Vec<_> = want.difference(&sites).collect();
        let extra: Vec<_> = sites.difference(&want).collect();
        return Err(format!("adapted {} sites, configured {}: missing {missing:?}, extra {extra:?}", sites.len(), want.len()));
    }
    match silent.first() {
        Some(n) => Err(format!("{n} received no gradient")),
        None => Ok(()),
    }
}

#[test]
fn gradients_reach_exactly_the_configured_projections_and_nothing_else() {
    for names in [&qwen3::finetune::LORA_TARGETS[..], &["wq", "up"][..]] {
        let cfg = adapted(names, false);
        let m = Qwen::new(cfg.clone(), 1, 12, &drawn_adapters(&cfg, 7));
        if let Some(moved) = train_steps(&m, 4) {
            panic!("{names:?}: the frozen weight {moved} changed over the steps");
        }
        trains_exactly(&m, names).unwrap_or_else(|why| panic!("{names:?}: {why}"));
    }
    // Mutants: an adapter built on a different target set than the one configured.
    for (built, configured) in [(&["wq", "wk", "wv", "wo", "gate", "up"][..], &qwen3::finetune::LORA_TARGETS[..]), (&qwen3::finetune::LORA_TARGETS[..], &["wq", "up"][..])] {
        let cfg = adapted(built, false);
        let m = Qwen::new(cfg.clone(), 1, 12, &drawn_adapters(&cfg, 7));
        train_steps(&m, 1);
        let why = trains_exactly(&m, configured).expect_err("an adapter on the wrong target set must be caught");
        eprintln!("mutant \"target set {built:?} for configured {configured:?}\" killed: {why}");
    }
}

#[test]
fn a_fresh_adapter_is_the_base_and_its_first_gradient_reaches_only_b() {
    let cfg = adapted(&qwen3::finetune::LORA_TARGETS, false);
    let init = qwen3::init_weights(&cfg, 13);
    let m = Qwen::new(cfg.clone(), 1, 12, &init);
    let base: HashMap<String, Vec<f32>> = init.iter().filter(|(n, _)| !is_adapter_param(n)).map(|(n, v)| (n.clone(), v.clone())).collect();
    let (live, plain) = (m.logits_all(&PROMPT), Qwen::new(QwenConfig::tiny(), 1, 12, &base).logits_all(&PROMPT));
    assert!(live.iter().zip(&plain).all(|(a, b)| a.to_bits() == b.to_bits()), "a zero-B adapter changed the base's logits");
    train_steps(&m, 1);
    let trained = Model::optimized_params(&m).unwrap();
    for n in &trained {
        let g = m.read_grad(n);
        if n.ends_with(LORA_A) {
            assert!(g.iter().all(|&v| v == 0.0), "{n}: A received a gradient while B is zero");
        } else {
            assert!(g.iter().any(|&v| v != 0.0), "{n}: B received no gradient");
        }
    }
}
