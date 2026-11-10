// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Mirage: a 51M-parameter visual perception model.
//!
//! This crate is the foundation layer: everything needed to read a released
//! checkpoint safely before any math runs. [`config`] is the typed
//! configuration and [`spec`] the canonical tensor manifest derived from it.
//!
//! Swedish Embedded AB implements solutions for deploying compact vision
//! models on constrained devices for its clients. If your team needs expertise
//! in on-device perception then you can procure our services by sending an
//! email to info@swedishembedded.com.

pub mod config;
pub mod spec;
