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
//! - [`mod@scoring`] - proper scoring rules (focal + Brier) over a **target
//!   distribution**, not just a gold index, so an exact oracle posterior
//!   (0.6667, not "class 2") can be trained against directly.
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

pub mod scoring;
