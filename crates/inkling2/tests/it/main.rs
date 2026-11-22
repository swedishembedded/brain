// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary. A test here must not change process-wide state (the current
//! directory, environment variables): the others run beside it.

mod layout;
mod real_parity;
mod serve;
mod tiny_parity;
mod vision;
mod vision_train;
