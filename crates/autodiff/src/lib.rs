// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Reverse-mode autodiff scaffolding — PLACEHOLDER, no implementation and no
//! consumers. (An earlier doc claimed this is "reused by every model"; it
//! never was — every model ships a hand-written backward validated by
//! `crates/gradcheck`, and that is the current design, not a stopgap.)
//! Implemented in a later phase if a shared tape/SSA cache earns its place.
