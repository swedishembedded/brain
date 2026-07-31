// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! brain's **self-describing, hierarchical stats subsystem** — the data-driven
//! contract a live-monitoring TUI (`braintop`, and `braintop --cli`) renders from.
//!
//! The shape is deliberately open and collection-oriented so nothing downstream
//! hardcodes counts or names:
//!
//! - [`StatsSnapshot`] is a tree of typed sections, each a **collection keyed by
//!   id** — `accelerators`, `models` (with per-instance residency), `executor`,
//!   `requests`, `connections` — plus an open `extra: BTreeMap<String, Value>` at
//!   every level, so a new leaf metric needs no schema change (emit into `extra`
//!   and the generic tree view picks it up).
//! - A [`StatsSource`] contributes into a snapshot; an [`Assembler`] walks all
//!   registered sources to build one. Components thus contribute without any
//!   central switchboard.
//! - [`build`] wires the **live** sources: [`build::ExecutorSource`] reads the
//!   residency [`Executor`](residency::Executor) — its counters, its manifest
//!   catalog, and its residency/budget report — to fill accelerators, models, and
//!   the executor section entirely from the running system.
//!
//! Everything here is serde + assembly only: the crate pulls no GPU/model/engine
//! code, just `brain-residency` (CPU-only scheduling logic) and `brain-capability`
//! for the manifest shape it maps.
//!
//! To add a metric, add a field to the relevant typed section (or emit into an
//! `extra` map); it flows through the JSON snapshot and braintop renders it
//! automatically. Collections are always data-driven — never hardcode a count.

pub mod build;
pub mod snapshot;
pub mod source;

pub use build::{snapshot_from_executor, ExecutorSource};
pub use snapshot::{
    Accelerator, ConnStat, ExecutorStat, Instance, ModelStat, RequestStat, StatsSnapshot, SCHEMA_VERSION,
};
pub use source::{Assembler, StatsSource};
