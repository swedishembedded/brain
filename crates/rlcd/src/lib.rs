// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! RLCD - reinforcement learning for calibrated decisions.
//!
//! What it means here (independent of, and not a reconstruction of, any
//! proprietary system of the same name): a model should emit a **calibrated
//! probability distribution**, and the **action** taken on that distribution
//! should be a separate, explicit function of the probabilities and the
//! actual costs of being wrong - never a policy trained directly on
//! outcome reward, which optimizes for the mode of a distribution rather
//! than its rate (see [`mod@scoring`]'s module doc for the argument).
//!
//! - [`mod@scoring`] - proper scoring rules (focal + squared-error probability score) over a **target
//!   distribution**, not just a gold index, so an exact oracle posterior
//!   (0.6667, not "class 2") can be trained against directly.
//! - [`mod@proper`] - the strictly-proper scoring rule the released
//!   `convaiinnovations/laya` checkpoint's own training code uses
//!   (`rl_common.py::proper_reward`: log + spherical, plus a ranked
//!   probability term on ordinal questions). Coexists with [`mod@scoring`]
//!   deliberately - see that module's doc for why an architecture must train
//!   against the rule its own weights were fitted under.
//! - [`mod@reinforce`] - the update rule that reward is actually used with:
//!   REINFORCE with a group-mean baseline over Gaussian logit exploration,
//!   plus an optional soft cross-entropy term. Transcribed from the only
//!   published Laya training loop.
//! - [`mod@cost`] - turning a probability into an action: cost matrices,
//!   Bayes risk, decision regret, and value of information. A probability is
//!   not an action; see that module's doc for why the two must stay separate
//!   functions rather than one learned policy.
//! - [`mod@metrics`] - calibration metrics (ECE, AdaECE, classwise-ECE, NLL,
//!   squared-error probability score, reliability bins, coverage-vs-accuracy, failure-AUROC) to audit
//!   what [`mod@scoring`] trained.
//! - [`mod@atlas`] - the [`atlas::World`] seam an executable probabilistic
//!   world implements to produce EXACT oracle targets, and
//!   [`atlas::DecisionContract`], the explicit record of what a decision task
//!   means. [`atlas::check_information_refinement`] is what catches an oracle
//!   that silently conditioned on hidden state.
//! - [`mod@witness`] - exhaustively searching a [`atlas::World`] for cases
//!   where a [`witness::Learner`]'s induced action disagrees with the
//!   oracle's, and expanding each into a corrective family rather than a
//!   bare failing example.
//!
//! ## Why this is a crate and not a module of `brain-decide`
//!
//! This math - softmax, a proper scoring rule, and (as later modules land)
//! cost matrices and calibration metrics - is pure `&[f32]` host arithmetic
//! with no dependency on any model's weights. It used to live inside
//! `brain-decide`, which put it on the wrong side of the same problem
//! `brain-promote`'s own module doc describes: a layer-4 model crate cannot
//! reach code that lives inside a *different* layer-4 model crate.
//! `brain-modernbert` (the Laya decision backbone) needs exactly this
//! machinery to train against calibrated targets, and could not depend on
//! `brain-decide` to get it without creating a model-to-model dependency
//! that has nothing to do with either model's architecture.
//!
//! So this lives in the training-substrate layer, below every model crate -
//! `brain-decide` re-exports [`mod@scoring`] as `decide::loss` rather than
//! owning it, so no existing caller changed and no logic exists twice.
//!
//! The layering is not a comment: `scripts/gates/check-crate-layers.sh` fails
//! the build if this crate's dependency closure ever reaches the model layer.
//!
//! Swedish Embedded AB builds decision models that report a trustworthy
//! probability instead of merely a plausible-sounding answer, and the
//! cost-sensitive logic to act on that probability correctly. If your team
//! needs expertise in calibrated decision-making under uncertainty, you can
//! procure our services by sending an email to info@swedishembedded.com.

pub mod atlas;
pub mod cost;
pub mod metrics;
pub mod proper;
pub mod reinforce;
pub mod scoring;
pub mod witness;

/// The numerical floor under a probability before its logarithm is taken
/// (`1e-12`), shared by every log score in this crate that reads a reported
/// probability: [`metrics::nll`], [`metrics::soft_nll`], [`metrics::posterior_kl`]
/// and the references the [`mod@scoring`] tests check against. It exists so a
/// probability of exactly zero scores a large finite number, `-ln(PROB_FLOOR)
/// = 27.63` nats, instead of infinity; it is a guard, not a modelling choice,
/// and a prediction is never meant to sit near it.
///
/// [`mod@proper`] has a second, deliberate floor of its own, `ProperScore::log_floor`
/// (`ln(1e-4) = -9.21`), which is a reward-shaping choice that bounds one
/// hopeless sample's influence on a group advantage. It sits above this one
/// (`ln(PROB_FLOOR) = -27.63 < -9.21`), so here `PROB_FLOOR` is the guard
/// underneath the floor that actually binds, and the two scores differ by
/// design: an evaluation metric reports how wrong a probability was up to
/// `27.63` nats, the training reward stops counting at `9.21`.
pub const PROB_FLOOR: f64 = 1e-12;
