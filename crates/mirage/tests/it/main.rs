// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary. A test here must not
//! change process-wide state: the others run beside it.

mod common;
mod config;
mod spec;
