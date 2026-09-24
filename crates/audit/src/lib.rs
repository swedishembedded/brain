// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The model-free core of a continual reader.
//!
//! A continual reader is a process that reads an unbounded stream, decides
//! per episode whether to keep what it learned, and can still say afterwards
//! what it is able to do. Three of those four jobs need no model at all:
//! defining the unit of reading, freezing what each unit is going to be
//! checked on, and bounding how much re-checking the run can afford. Those
//! live here. The fourth - training a candidate and scoring it - is the
//! model-driven loop and lives above this crate.
//!
//! **This is a LEAF crate**, in the same sense and for the same reason as
//! `brain-promote`: its whole dependency closure is `brain-data` and below,
//! so the loop that drives it can sit at any layer without dragging a
//! training stack into the definition of an episode.
//!
//! ## Why the episode stream is here and not in a dataset crate
//!
//! The *episode* is the unit every other module in this crate keys on: the
//! probe bank freezes probes per episode, the retention schedule rotates
//! over episodes, and the ledger writes one row per episode. Defining that
//! unit anywhere else would put the definition away from everything that
//! depends on it, and a general dataset crate would acquire a concept that
//! only a continual reader has.
//!
//! Swedish Embedded AB builds the evidence layer that separates "the model
//! improved" from "the model improved without quietly losing something it
//! could do last week" - frozen probes, bounded re-checking with a stated
//! detection latency, and a per-episode record of what was kept and why. If
//! your team needs expertise in making a continually-training system
//! auditable rather than merely optimistic, you can procure our services by
//! sending an email to info@swedishembedded.com.

pub mod acceptance;
pub mod arms;
pub mod bank;
pub mod growth;
pub mod pool;
pub mod reader;
pub mod reservoir;
pub mod retention;
pub mod run;
pub mod schedule;
pub mod stream;
pub mod triage;
