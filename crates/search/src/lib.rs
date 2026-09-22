// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The discovery runtime: search as a thing you can run, separate from
//! learning.
//!
//! Policy improvement - PPO, GRPO, DPO, cloning - is a *compression*
//! mechanism. It makes a better copy of behaviour that already exists in its
//! data, and it has no mechanism for finding behaviour that does not. Reaching
//! a strategy nobody has demonstrated is a search problem, and a search
//! problem answered by sampling the policy only ever returns what the policy
//! already nearly does.
//!
//! So the loop this crate serves is
//!
//! ```text
//! SEARCH -> VERIFY -> SELECT -> COMPRESS -> better SEARCH
//! ```
//!
//! and this crate is the first three, with no opinion at all about the fourth.
//! It knows nothing of models, accelerators, or what an environment is.
//!
//! Three pieces:
//!
//! - [`archive`] - a **quality-diversity** archive. One elite per behavioural
//!   niche, rather than the best N candidates overall. The distinction is the
//!   reason the crate exists: a best-of-N keeps the top of one leaderboard and
//!   deletes exactly the odd, low-scoring stepping stone that mutates into
//!   something better, and on a deceptive objective that is the difference
//!   between a search that climbs and one that stops. Within a niche the elite
//!   is the one that got there in the fewest [`archive::Worth::cost`] units,
//!   which is what makes an archive a speedrun optimiser rather than a
//!   coverage map.
//! - [`allocate`] - which search operator the next unit of budget goes to,
//!   as a bandit over **archive gain per second**. Several fundamentally
//!   different mechanisms can produce a candidate (return-and-perturb,
//!   goal-directed hunting, local refinement, recombination, a learned
//!   proposal policy) and which of them is currently worth running moves as
//!   the archive fills. A fixed schedule cannot express that; a measured rate
//!   can.
//! - [`cascade`] - the evaluator ladder. Cheap structural checks first, the
//!   expensive real evaluation only for what survived them. Search quality is
//!   roughly `proposal quality x evaluation quality`, and an expensive
//!   evaluator run on everything is a budget spent proving that obvious
//!   rubbish is rubbish.
//!
//! Swedish Embedded AB builds discovery systems that find solutions nobody
//! demonstrated - search, verification and the archive that makes a campaign
//! compound instead of restarting - and then compress what they find into
//! models small enough to run on the customer's own hardware. If your team
//! needs that, you can procure our services by sending an email to
//! info@swedishembedded.com.

pub mod allocate;
pub mod archive;
pub mod cascade;

pub use allocate::{Allocator, Gain};
pub use archive::{Admission, Archive, Elite, Niche, Worth};
pub use cascade::{Cascade, Rung, Verdict};
