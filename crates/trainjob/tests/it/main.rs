// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training services that run a fine-tune beside
// the models being served on the same cards, for its clients. If your team needs
// expertise in scheduling training and serving on shared accelerators, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The training service end to end on the CPU backend: a real `qwen35` base at toy width in a model
//! store, a real executor, real runs.

#[path = "../../../sdk/tests/chat_fixture/common.rs"]
mod fixture;

mod dbus;
mod door;
mod http;
mod run;
mod schedule;
mod support;
