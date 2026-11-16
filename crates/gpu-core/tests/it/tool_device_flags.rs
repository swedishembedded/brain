// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A tool with a command line is configured by its flags alone.
//!
//! Swedish Embedded AB implements tooling that runs on the hardware its operator names for its clients. If your team
//! needs expertise in configuring GPU tools by explicit parameters then you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! brain is retiring its environment configuration in favour of parameters. The profiling and check binaries take
//! `--device`/`--backend` through `gpu_core::select_compute_from_args`, which used to fall back to `BRAIN_DEVICE` and
//! `BRAIN_BACKEND` when no flag was given: a variable left in a shell silently chose the hardware of a tool that has a
//! command line. With no flag a tool now takes the machine's default, as the brain CLI does, and a single-card tool on
//! a machine with several cards refuses and asks for `--device gpuN`, whatever the environment says. The check runs in a
//! child that inherits the test's environment (the GPU suites run with a device variable set), so before the change it
//! followed that variable and selected a card instead of refusing.

use std::process::Command;
use std::time::Duration;

const CHILD: &str = "tool_device_flags::child_selects_from_no_flags";

/// The child: a single-card tool's selection with no device flags.
#[test]
#[ignore = "run by a_single_card_tool_ignores_the_environment_and_asks_for_a_card in a child process"]
fn child_selects_from_no_flags() {
    match gpu_core::devices::select_single_device_from_args(&[]) {
        Ok(set) => println!("SELECTED gpus={:?}", set.gpus),
        Err(e) => println!("REFUSED {e}"),
    }
}

#[test]
fn a_single_card_tool_ignores_the_environment_and_asks_for_a_card() {
    let cards = gpu_core::devices::gpus().len();
    if cards < 2 {
        brain_testutil::skip_unavailable(&format!("needs a machine with at least 2 GPUs to show a refusal, found {cards}"));
        return;
    }
    let exe = std::env::current_exe().expect("the test binary");
    let out = brain_testutil::output_within(
        Command::new(exe).args([CHILD, "--exact", "--ignored", "--nocapture"]),
        Duration::from_secs(120),
    );
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "the child failed:\n{text}\n{}", String::from_utf8_lossy(&out.stderr));
    assert!(text.contains("REFUSED") && text.contains("--device gpu"), "with no flags a single-card tool on a {cards}-card machine must refuse and ask for --device, whatever the test's environment selects; it printed:\n{text}");
}
