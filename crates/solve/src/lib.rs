// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Learning to reach a known goal state, by retracing random walks away from
//! it.
//!
//! The method, and why it needs neither a solver nor a reward signal: in a
//! space whose moves are INVERTIBLE, walking away from the goal generates its
//! own labels. Apply `m` to a goal state and the correct action at the state
//! you land on is `m` inverted - free, exactly known, and available at any
//! depth. No planner has to verify it, no value has to be bootstrapped, and
//! no episode has to be scored. A generator that runs on a CPU core produces
//! training data faster than a GPU consumes it.
//!
//! That is the whole supervision signal. What comes out is a policy
//! `pi(action | state)` rolled out GREEDILY: one forward pass per action, no
//! tree, no priority queue, no search of any kind at inference. The contrast
//! worth naming is the value-function-plus-A* arrangement, which spends
//! millions of node expansions per instance at run time and is the reason
//! that approach is measured in seconds. A policy that can be rolled out is
//! measured in forward passes.
//!
//! ## What this costs you, stated plainly
//!
//! Retraced labels are not optimal labels. A walk of length `L` that revisits
//! ground gives a label that leads home the long way, so the policy learns to
//! SOLVE rather than to solve SHORTEST. Past the space's diameter the labels
//! get actively noisy - beyond it every extra step of walk is redundant, and
//! the action it names is arbitrary among many. [`Walk::depth`] is therefore
//! a real hyperparameter and not a budget knob: set it near the diameter, not
//! above it.
//!
//! ## What a caller supplies
//!
//! One [`StateSpace`]: what a state is, what the moves are, how to apply and
//! invert one, and how a state is written down as features. Nothing here
//! knows what a cube is.
//!
//! Swedish Embedded AB builds learned planners for discrete control problems -
//! the architecture, the label generator, and the measurement that says
//! whether the policy or the search is doing the work. If your team needs
//! that, you can procure our services by sending an email to
//! info@swedishembedded.com.

pub mod config;
pub mod data;
pub mod kern;
pub mod net;
pub mod rollout;

pub use config::Config;
pub use data::{Batch, Walk};
pub use net::Net;
pub use rollout::{Outcome, Rollout};

/// A discrete space with invertible moves and one goal state.
///
/// Implementors describe a problem; nothing in this crate is specialised to
/// any of them. The contract that matters is [`StateSpace::inverse`]: the
/// label generator is only correct because applying a move and then its
/// inverse returns the state it started from, and an implementation that
/// breaks that trains a policy on labels that do not lead home.
pub trait StateSpace {
    /// A position in the space. Cheap to copy and compare - the generator
    /// makes millions of them.
    type State: Clone + PartialEq;

    /// How many distinct moves there are. The policy's output width.
    fn moves(&self) -> usize;

    /// The state every walk starts from and every rollout aims at.
    fn goal(&self) -> Self::State;

    /// Apply move `m` (an index into `0..self.moves()`).
    fn apply(&self, s: &Self::State, m: usize) -> Self::State;

    /// The move that undoes `m`. `apply(apply(s, m), inverse(m)) == s` for
    /// every state, which [`data::inverse_is_an_inverse`] checks for any
    /// implementation that wants the guarantee tested.
    fn inverse(&self, m: usize) -> usize;

    /// Width of the feature vector [`StateSpace::write_features`] fills.
    fn feature_len(&self) -> usize;

    /// Write `s` into `out` (already `feature_len()` long and zeroed).
    ///
    /// A one-hot indicator is the usual answer and is what the first layer
    /// is shaped for: a matmul against a one-hot row is a table lookup, so
    /// width here costs almost nothing at run time.
    fn write_features(&self, s: &Self::State, out: &mut [f32]);

    /// Whether `s` is the goal. Defaults to comparing against
    /// [`StateSpace::goal`]; override when a cheaper test exists.
    fn is_goal(&self, s: &Self::State) -> bool {
        *s == self.goal()
    }

    /// Whether playing `b` straight after `a` is a move the generator should
    /// not make, because the pair wastes walk length it has already spent.
    ///
    /// Undoing the last move is always redundant and is the default. Spaces
    /// with more structure should say so: on a cube, any second turn of the
    /// SAME face is redundant (the pair is one turn of that face), and
    /// leaving that in makes a walk of length `L` reach a state well short of
    /// distance `L`, which is a label that teaches the long way home.
    fn redundant(&self, a: usize, b: usize) -> bool {
        b == self.inverse(a)
    }
}
