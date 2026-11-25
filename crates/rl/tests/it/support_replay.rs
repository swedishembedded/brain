// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training language models without bias on rollouts sampled with
// top-k, top-p or min-p for its clients. If your team needs expertise in replaying a truncated sampler's support in a
// policy-gradient trainer then you can procure our services by sending an email to info@swedishembedded.com.

//! Support replay on real decoders (tiny Qwen3.5, and tiny Qwen3 with the `qwen3` feature), model-free:
//!
//! - the bias it removes: a rollout drawn from the model itself by brain's own truncated sampler
//!   (`model::serve::draw_row_supported`, `top_k` 5 and `top_p` 0.9) trains at step 0 with an importance ratio of 1 at
//!   every token when its support is replayed, and with a ratio equal to the support's mass (below 1) when the sampler's
//!   figure is taken against the trainer's whole-row log-probability;
//! - the gradient: the support-normalised loss of the clipped surrogate and of the score-centred estimator, its
//!   weights frozen and its log-probabilities renormalised over each position's support at every perturbed point,
//!   matches central finite differences on the LoRA adapters; the weights the objective installs are those of
//!   `model::head_ce::support_coefficients`.

use std::cell::{Cell, RefCell};

use data::rng::Rng;
use gradcheck::{directional_check, CheckModel};
use model::head_ce::{renormalise_over_support, support_coefficients};
use model::serve::{draw_key, draw_row_supported, Drawn, RowDraw};
use model::training;
use model::{Batch, Model, IGNORE};
use rl::objective::grpo_term::ClipBand;
use rl::objective::pg_batch::{install_sampler_ids, weigh, PgBatch, PgEstimator, PgSpec, SamplerSupport, SamplerTop};
use rl::objective::score_centring::{centred_term, CentringBand};

const T: usize = 12;
/// The prompt; the rest of the row is drawn by the sampler.
const PROMPT: usize = 4;
/// Support slots: the sampler's `top_k`.
const K: usize = 5;

/// One row drawn from `m` itself, token by token, by brain's own sampler with a cut (`top_k` K, `top_p` 0.9) at
/// temperature 1: the tokens, and at each drawn position the support, the sampler's figure within it and the
/// whole-row figure.
struct Drawn1 {
    tokens: Vec<u32>,
    supports: Vec<Vec<u32>>,
    sampler: Vec<f32>,
    whole: Vec<f32>,
}

fn draw_from<M: Model>(m: &M, seed: u64) -> Drawn1 {
    let vocab = model::ModelConfig::vocab(m.config()) as usize;
    let mut tokens: Vec<u32> = (0..PROMPT as u32).map(|t| (t * 7 + 3) % vocab as u32).collect();
    let (mut supports, mut sampler, mut whole) = (Vec::new(), Vec::new(), Vec::new());
    while tokens.len() < T {
        let logits = m.logits_all(&tokens).expect("a decoder scores its rows");
        let row = &logits[(tokens.len() - 1) * vocab..tokens.len() * vocab];
        let d = RowDraw { temp: 1.0, min_p: 0.0, top_p: 0.9, top_k: K, top: 0, support: K, key: draw_key(seed, tokens.len() as u64, 0) };
        let Drawn::Supported(s) = draw_row_supported(row, d) else { unreachable!("a cut draw that asks reports its support") };
        supports.push(s.support.expect("top_k K keeps at most K"));
        sampler.push(s.sampler);
        whole.push(s.whole);
        tokens.push(s.token);
    }
    Drawn1 { tokens, supports, sampler, whole }
}

/// The batch of one drawn row: targets at the drawn positions, the support of each laid out in K slots.
struct Laid {
    targets: Vec<u32>,
    advantage: Vec<f32>,
    ids: Vec<u32>,
    sampler: Vec<f32>,
    whole: Vec<f32>,
}

fn lay_out(d: &Drawn1) -> Laid {
    let mut l = Laid { targets: vec![IGNORE; T], advantage: vec![f32::NAN; T], ids: vec![IGNORE; T * K], sampler: vec![f32::NAN; T], whole: vec![0.0; T] };
    for (j, at) in (PROMPT - 1..T - 1).enumerate() {
        l.targets[at] = d.tokens[at + 1];
        l.advantage[at] = if j % 2 == 0 { 1.0 } else { -0.7 };
        l.ids[at * K..at * K + d.supports[j].len()].copy_from_slice(&d.supports[j]);
        l.sampler[at] = d.sampler[j];
        l.whole[at] = d.whole[j];
    }
    l
}

fn tiny_qwen35(seed: u64) -> qwen35::model::Qwen35 {
    use qwen35::config::{lora_cfg, Qwen35Config};
    let cfg = Qwen35Config { lora: Some(lora_cfg(2, 4.0)), ..Qwen35Config::tiny() };
    let init = qwen35::init::init_weights(&cfg, seed);
    qwen35::model::Qwen35::new_train_on(gpu_core::Gpu::new_cpu(qwen35::model::pipelines()), cfg, 1, T as u32, &init)
}

/// SPEC (the bias demonstration): at step 0 the policy is the sampler, so a draw whose support is replayed has an
/// importance ratio of exactly one at every token (to the rounding of the two arithmetics), while the sampler's own
/// figure against the trainer's whole-row log-probability gives a ratio equal to the mass the cut kept, below one.
#[test]
fn a_replayed_support_starts_at_ratio_one_and_the_whole_row_figure_does_not() {
    let mut m = tiny_qwen35(17);
    let objectives = training::require_token_objectives(&mut m).unwrap();
    objectives.enable_weighted_loss();
    objectives.enable_sparse_targets(K as u32).unwrap();
    let drawn = draw_from(&m, 5);
    let l = lay_out(&drawn);
    let spec = PgSpec::clipped(ClipBand::new(0.2, 0.2), 0.0);
    let replayed = PgBatch { targets: &l.targets, old_lp: &l.whole, ref_lp: None, advantage: &l.advantage, is_weight: None, top: None, support: Some(SamplerSupport { k: K, ids: &l.ids, logprobs: &l.sampler }) };
    Model::set_batch(&m, Batch::Lm { tokens: &drawn.tokens, targets: &l.targets });
    install_sampler_ids(&m, &replayed, &spec).unwrap();
    let _ = Model::forward(&m);
    let o = training::objectives(&m);
    let (new_lp, slots) = (o.batch_token_logprobs(), o.batch_sparse_target_logprobs().unwrap());
    let (_, terms) = weigh(&new_lp, Some(&slots), &replayed, &spec);
    assert_eq!(terms.policy_tokens, T - PROMPT);
    let off = (terms.ratio_min - 1.0).abs().max((terms.ratio_max - 1.0).abs());
    assert!(off < 2e-5, "replayed: the ratio at step 0 is one at every token: [{}, {}]", terms.ratio_min, terms.ratio_max);
    // Without replay: the sampler's figure, the distribution that drew the token, against the whole row.
    let biased = PgBatch { targets: &l.targets, old_lp: &l.sampler, ref_lp: None, advantage: &l.advantage, is_weight: None, top: None, support: None };
    let (_, terms) = weigh(&new_lp, None, &biased, &spec);
    assert!(terms.ratio_max < 0.999, "not replayed: the ratio sits below one, at the support's mass: max {}", terms.ratio_max);
    for at in (0..T).filter(|&at| l.targets[at] != IGNORE) {
        let ratio = (new_lp[at] - l.sampler[at]).exp();
        let mass = (l.whole[at] - l.sampler[at]).exp();
        assert!((ratio - mass).abs() < 2e-5 && mass < 1.0, "position {at}: ratio {ratio} against the support's mass {mass}");
    }
}

/// The estimator's weights over the policy's renormalised log-probabilities, frozen at the base point: per position the
/// label's `omega_y` and the support slots' `omega_j`.
#[derive(Clone)]
struct Frozen {
    label: Vec<f64>,
    slots: Vec<f64>,
}

struct Harness<M: Model> {
    m: M,
    tokens: Vec<u32>,
    l: Laid,
    /// The sampler's figures at the support's ids, renormalised over it (the score-centred estimator's `q`), and the
    /// same as whole-row figures (what a record's `behavior_top_logprobs` holds).
    top_lp: Vec<f32>,
    top_whole: Vec<f32>,
    spec: PgSpec,
    frozen: RefCell<Option<Frozen>>,
    forwarded: Cell<bool>,
}

const DENOMINATOR: f32 = 7.0;

impl<M: Model> Harness<M> {
    fn new(mut m: M, spec: PgSpec, seed: u64) -> Harness<M> {
        let objectives = training::require_token_objectives(&mut m).unwrap();
        objectives.enable_weighted_loss();
        objectives.enable_sparse_targets(K as u32).unwrap();
        let drawn = draw_from(&m, seed);
        // Off the invisible start (B = 0): every adapter tensor carries gradient, and the policy has left the sampler.
        let mut rng = Rng::new(seed ^ 0x77);
        for name in m.param_names().into_iter().filter(|n| n.contains(".lora_")) {
            let w: Vec<f32> = (0..m.read_weight(&name).len()).map(|_| 0.8 * (rng.next_f32() - 0.5)).collect();
            m.write_weight(&name, &w);
        }
        // The supports the gradient is checked over hold little of the row's mass (the label and two others), so the
        // whole-row softmax a wrong coefficient would leave in the gradient is not hidden by a support that is nearly the
        // row.
        let vocab = model::ModelConfig::vocab(m.config());
        let mut l = lay_out(&drawn);
        for at in (0..T).filter(|&at| l.targets[at] != IGNORE) {
            let y = l.targets[at];
            l.ids[at * K..(at + 1) * K].copy_from_slice(&[y, (y + 3) % vocab, (y + 11) % vocab, IGNORE, IGNORE]);
        }
        // The sampler's top-K is its support (a cut to the top K keeps at most K), with its figures renormalised over it,
        // a little off the policy's so the head and the tail both move.
        let mut top_lp = vec![f32::NAN; T * K];
        let h0 = Harness { m, tokens: drawn.tokens.clone(), l: Laid { targets: l.targets.clone(), advantage: l.advantage.clone(), ids: l.ids.clone(), sampler: l.sampler.clone(), whole: l.whole.clone() }, top_lp: top_lp.clone(), top_whole: vec![f32::NAN; T * K], spec, frozen: RefCell::new(None), forwarded: Cell::new(false) };
        h0.load();
        let _ = h0.m.forward();
        let o = training::objectives(&h0.m);
        let (lp, slots) = (o.batch_token_logprobs(), o.batch_sparse_target_logprobs().unwrap());
        for (n, at) in (0..T).filter(|&at| l.targets[at] != IGNORE).enumerate() {
            let norm = renormalise_over_support(lp[at], &slots[at * K..(at + 1) * K], &l.ids[at * K..(at + 1) * K]).unwrap();
            for j in (0..K).filter(|&j| l.ids[at * K + j] != IGNORE) {
                top_lp[at * K + j] = (norm.pi[j].ln() + [0.2, -0.15, 0.1][(n + j) % 3]).min(-1e-3) as f32;
            }
            l.sampler[at] = top_lp[at * K + l.ids[at * K..(at + 1) * K].iter().position(|&id| id == l.targets[at]).unwrap()];
            // The whole-row figure the sampler's implies: the trainer's support mass at the base point.
            l.whole[at] = (f64::from(l.sampler[at]) + norm.log_z.min(0.0)) as f32;
        }
        let top_whole: Vec<f32> = (0..T * K).map(|i| top_lp[i] + (l.whole[i / K] - l.sampler[i / K])).collect();
        let h = Harness { l, top_lp, top_whole, ..h0 };
        h.load();
        let _ = h.m.forward();
        let o = training::objectives(&h.m);
        let (lp, slots) = (o.batch_token_logprobs(), o.batch_sparse_target_logprobs().unwrap());
        let (installed, terms) = weigh(&lp, Some(&slots), &h.batch(), &h.spec);
        assert!(terms.finite && terms.policy_tokens == T - PROMPT, "{terms:?}");
        // The frozen weights, from the estimator's definition over the renormalised log-probabilities.
        let mut frozen = Frozen { label: vec![0.0; T], slots: vec![0.0; T * K] };
        for at in (0..T).filter(|&at| h.l.targets[at] != IGNORE) {
            let ids = &h.l.ids[at * K..(at + 1) * K];
            let norm = renormalise_over_support(lp[at], &slots[at * K..(at + 1) * K], ids).unwrap();
            match h.spec.estimator {
                PgEstimator::ClippedSurrogate => {
                    let term = rl::objective::grpo_term::token_term_weighted(h.l.sampler[at], norm.label as f32, h.l.advantage[at], h.spec.band, None, 0.0, 1.0);
                    frozen.label[at] = f64::from(term.weight);
                }
                PgEstimator::ScoreCentred(band) => {
                    let trainer: Vec<f32> = (0..K).map(|j| if ids[j] == IGNORE { 0.0 } else { (f64::from(slots[at * K + j]) - norm.log_z) as f32 }).collect();
                    let mut coefs = vec![0f32; K];
                    let term = centred_term(h.l.sampler[at], norm.label as f32, h.l.advantage[at], band, &h.top_lp[at * K..(at + 1) * K], &trainer, &mut coefs);
                    frozen.label[at] = f64::from(term.weight);
                    coefs.iter().enumerate().for_each(|(j, c)| frozen.slots[at * K + j] = f64::from(*c));
                }
            }
            let want = support_coefficients(&norm, frozen.label[at], &frozen.slots[at * K..(at + 1) * K]);
            let got = &installed.extra.as_ref().expect("supports take extra targets")[at * K..(at + 1) * K];
            for (j, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!((g - w).abs() <= 1e-5 * w.abs().max(1.0), "position {at} slot {j}: installed {g}, the definition's {w}");
            }
            assert!((f64::from(installed.label[at]) - frozen.label[at]).abs() <= 1e-5 * frozen.label[at].abs().max(1.0), "position {at}: label weight");
        }
        *h.frozen.borrow_mut() = Some(frozen);
        h
    }

    fn batch(&self) -> PgBatch<'_> {
        let centring = matches!(self.spec.estimator, PgEstimator::ScoreCentred(_));
        PgBatch {
            targets: &self.l.targets,
            old_lp: &self.l.whole,
            ref_lp: None,
            advantage: &self.l.advantage,
            is_weight: None,
            top: centring.then_some(SamplerTop { k: K, ids: &self.l.ids, logprobs: &self.top_whole }),
            support: Some(SamplerSupport { k: K, ids: &self.l.ids, logprobs: &self.l.sampler }),
        }
    }

    fn load(&self) {
        self.m.set_batch(Batch::Lm { tokens: &self.tokens, targets: &self.l.targets });
        training::objectives(&self.m).set_loss_denominator(DENOMINATOR);
        install_sampler_ids(&self.m, &self.batch(), &self.spec).unwrap();
    }
}

impl<M: Model> CheckModel for Harness<M> {
    fn param_names(&self) -> Vec<String> {
        self.m.param_names()
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        self.m.read_weight(name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        self.m.write_weight(name, data);
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        self.m.read_grad(name)
    }

    /// The support-normalised loss at the frozen weights, `sum omega_y (-log pi_S(y)) + sum omega_j (-log pi_S(s_j))`,
    /// renormalised at this point from the forward's own log-probabilities; it installs this point's coefficients for the
    /// backward.
    fn loss(&self) -> f32 {
        self.load();
        let _ = self.m.forward();
        self.forwarded.set(true);
        let o = training::objectives(&self.m);
        let (lp, slots) = (o.batch_token_logprobs(), o.batch_sparse_target_logprobs().unwrap());
        let frozen = self.frozen.borrow();
        let f = frozen.as_ref().expect("computed at construction");
        let (mut label_w, mut coefs, mut total) = (vec![0f32; T], vec![0f32; T * K], 0.0f64);
        for at in (0..T).filter(|&at| self.l.targets[at] != IGNORE) {
            let ids = &self.l.ids[at * K..(at + 1) * K];
            let norm = renormalise_over_support(lp[at], &slots[at * K..(at + 1) * K], ids).unwrap();
            let slot_w = &f.slots[at * K..(at + 1) * K];
            total -= f.label[at] * norm.label;
            for j in (0..K).filter(|&j| ids[j] != IGNORE) {
                total -= slot_w[j] * (f64::from(slots[at * K + j]) - norm.log_z);
            }
            label_w[at] = f.label[at] as f32;
            coefs[at * K..(at + 1) * K].copy_from_slice(&support_coefficients(&norm, f.label[at], slot_w));
        }
        o.set_loss_weights(&label_w);
        o.set_sparse_target_weights(&coefs).unwrap();
        (total / f64::from(DENOMINATOR)) as f32
    }

    fn zero_grads(&self) {
        self.m.zero_grads();
    }

    fn backward(&self) {
        let _ = self.loss();
        self.m.backward();
        self.m.poll_wait();
    }
}

fn check<M: Model>(h: &Harness<M>, what: &str) {
    let report = directional_check(h, 1.25e-3, 4, 0x5e9);
    let worst = report.checks.iter().fold((0.0f32, 0.0f32), |(a, r), c| (a.max(c.abs_err), r.max(c.rel_err)));
    eprintln!("{what}: {} adapter tensors checked, worst abs error {:e}, worst rel error {:e}", report.checks.len(), worst.0, worst.1);
    // Tighter than the workspace's (4e-3, 8e-2): these adapters' gradients are small, and at the workspace's floor a
    // coefficient that leaves a tenth of the whole-row softmax in the gradient passes on the tiny Qwen3.5.
    let failures = report.failures(2e-4, 2e-2);
    assert!(failures.is_empty(), "{what}: {:?}", failures.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>());
    assert!(report.dead_gradients().is_empty(), "{what}: dead gradients {:?}", report.dead_gradients());
    assert!(!report.checks.is_empty());
}

fn specs() -> [(PgSpec, &'static str); 2] {
    [
        (PgSpec::clipped(ClipBand::new(0.2, 0.2), 0.0), "clipped surrogate"),
        (PgSpec { band: ClipBand::new(0.2, 0.2), kl_beta: 0.0, estimator: PgEstimator::ScoreCentred(CentringBand { low: 0.6, high: 1.6 }) }, "score-centred"),
    ]
}

#[test]
fn qwen35_adapters_match_finite_differences_of_the_support_normalised_loss() {
    for (spec, what) in specs() {
        check(&Harness::new(tiny_qwen35(29), spec, 3), &format!("qwen35 {what} over the support"));
    }
}

#[cfg(feature = "qwen3")]
#[test]
fn qwen3_adapters_match_finite_differences_of_the_support_normalised_loss() {
    use qwen3::{Qwen, QwenConfig};
    for (spec, what) in specs() {
        let cfg = QwenConfig { lora: Some(qwen3::config::LoraCfg::new(2, 4.0, qwen3::finetune::default_lora_targets())), ..QwenConfig::tiny() };
        let init = <Qwen as Model>::init_weights(&cfg, 31);
        check(&Harness::new(Qwen::new(cfg, 1, T as u32, &init), spec, 4), &format!("qwen3 {what} over the support"));
    }
}
