// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Run a model's served actions from a command line, on the machine that
//! holds its weights.
//!
//! A model's capabilities live in one place: the [`capability::Action`]s its
//! provider declares, which `brain serve` publishes over HTTP and D-Bus. This
//! crate is the local front end to exactly those actions, so a capability has
//! one implementation whichever way it is reached:
//!
//! - [`args::command`] turns an action's [`capability::ActionSpec`] into its
//!   command line: one `--<param>` per param with the manifest's own type,
//!   enum values and help, `--<input> PATH` (or `--in name=PATH`) per input
//!   blob and `--out name=PATH` per output. A flag left out is a param left
//!   out, so the action applies its own default - the manifest is the only
//!   place a default is written.
//! - [`media`] turns files into input blobs (images, masks, WAV audio, video
//!   containers, directories of numbered frames) and output blobs back into
//!   files (PNG/PPM, WAV, a video container carrying the clip's own sound).
//! - [`resolve`] finds the weights in the model store, taking a `--<role>
//!   PATH` override per checkpoint role.
//! - [`run_model`] does all of it for one model: help from the static
//!   manifest, weights from the store, then the action; [`run_with`] runs one
//!   action on a provider already built and hands its outcome back, for a
//!   program that chains actions.
//!
//! `brain <model> <action>` is [`run_model`] over the served catalog; a
//! per-model sample is [`run_model`] over one model's provider, plus whatever
//! workflow it builds on top.
//!
//! Swedish Embedded AB implements front ends that expose a model's whole
//! parameter surface without a second copy of it for its clients. If your team
//! needs expertise in serving and driving neural models from its own tools,
//! you can procure our services by sending an email to info@swedishembedded.com.

pub mod args;
pub mod media;
mod model;
pub mod resolve;
mod run;

pub use model::{main, run_model, Model, GLOBAL_HELP};
pub use run::{run_with, Program};
