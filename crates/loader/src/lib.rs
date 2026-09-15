// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The library half of what used to live only inside `crates/cli` (the
//! `brain` binary): resolving a model against the model store, finding the
//! models directory, placing a build onto GPU/CPU capacity, and fetching a
//! default checkpoint on demand -- none of it CLI-local, all of it needed by
//! any embedder that wants to load a model without linking `brain-cli`.
//!
//! # Why this crate exists
//!
//! `crates/catalog` already made this split once, for the served-model
//! catalog: manifest + weight-free provider construction depend on nothing
//! CLI-local, so `brain-cli` (and any other in-process consumer) can use them
//! directly, while the ~20 CLI-local residency adapters that schedule a
//! model onto `brain serve`'s GPU/RAM/disk budget stay in `crates/cli`
//! itself (see that crate's own module doc). This crate is the same split
//! applied to what `catalog`'s doc calls "the whole point" the OTHER way
//! round: not "how a model is served", but "how a model is FOUND and LOADED"
//! -- the resolver, the models directory, automatic placement, and the
//! default-checkpoint auto-fetch policy. Every function here is exactly what
//! it was inside `crates/cli` before this crate existed; nothing moved is new
//! behavior except [`supply::DownloadPolicy`], which makes explicit a choice
//! the CLI used to make implicitly through one environment variable.
//!
//! What stays in `crates/cli` and is NOT here: `resolve_or_exit` (calls
//! `std::process::exit`, inherently a process-lifetime decision), every
//! `resident_*.rs` adapter (`QwenResident` and its ~20 siblings -- CLI-local
//! residency wiring, the same split `catalog` already drew), and the CLI's
//! own env-var-driven `--models-dir`/`--autofetch` plumbing.
//!
//! Swedish Embedded AB implements model-loading infrastructure -- resolution,
//! placement, and fetch -- as a library any embedder can call directly. If
//! your team is building an SDK or a service on top of a model store and
//! needs this without a CLI binary in the loop, you can procure our services
//! by sending an email to info@swedishembedded.com.

pub mod model_dir;
pub mod placement;
pub mod progress;
pub mod resolver;
pub mod supply;

pub use placement::{install_default_placer, BudgetPlacer};
pub use resolver::{resolve_structured, try_resolve, ResolveFailure, AMBIGUOUS_EXIT, MISSING_EXIT};
pub use supply::{ensure_default_weights, DefaultWeights, DownloadPolicy};
