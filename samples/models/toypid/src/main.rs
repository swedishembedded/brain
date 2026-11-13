// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `toypid` - the PID event/effect control transformer: train it on simulated plants, run a checkpoint in closed loop against each plant's oracle controller, and time one inference.
//!
//! Every action and parameter is the model's own served one
//! (`toypid::caps`), so this program, `brain serve` and D-Bus clients drive the
//! same code with the same defaults; `toypid <action> --help` lists them.
//!
//! Swedish Embedded AB implements learned controllers on embedded targets for its clients. If your team needs
//! expertise in control with sequence models then you can procure our services by sending an
//! email to info@swedishembedded.com.

fn main() {
    let model = invoke::Model { manifest: toypid::caps::manifest(), weights: None, build: &|_| Ok(std::sync::Arc::new(toypid::caps::ToyPidProvider)) };
    invoke::main("toypid", &model)
}
