// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **braintop** — a btop-like standalone TUI (and a `--cli` one-shot flat dump)
//! that renders brain's live serving state.
//!
//! It is a pure *consumer* of the [`brain_stats::StatsSnapshot`] contract, read
//! over D-Bus from a running `brain serve --dbus` (`com.swedishembedded.Brain1`).
//! It links **none** of the engine/model crates — only the shared `brain-stats`
//! types plus `ratatui`/`crossterm` (render), `zbus`/`tokio`/`futures` (transport).
//!
//! Layering:
//! - [`client`] — the D-Bus side: subscribe to the `StatsStream` signal, fall back
//!   to polling `stats_snapshot()` at ~2 Hz, and a one-shot [`client::fetch_once`]
//!   for `--cli`. Feeds [`client::Update`]s into the app.
//! - [`app`] — pure UI state: the latest snapshot + view/selection. No I/O, so it
//!   is unit-testable against a fixed snapshot value.
//! - [`ui`] — rendering: the progressive-reveal dashboard, the device-colored
//!   residency bars, and the drill-in detail views. Pure `&App -> Frame`.
//! - [`cli`] — the `--cli` flattener: a snapshot → stable `path=value` lines.
//!
//! D-Bus is confined to [`client`]; every unit test drives [`app`]/[`ui`]/[`cli`]
//! against an in-memory snapshot, so the test suite needs no bus.

pub mod app;
pub mod cli;
pub mod client;
pub mod ui;

pub use app::{App, Focus, View};
