// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A CUDA handle opens only a card the selection names.
//!
//! Swedish Embedded AB implements GPU software that runs on exactly the hardware it was given for its clients. If your
//! team needs expertise in device selection on shared GPU machines then you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! `Gpu::try_new_cuda` resolved "no card selected" to CUDA ordinal 0, so a process whose selection named only the CPU
//! (or whose placement policy chose the host) still opened the first card of the machine - on a shared node, a card
//! another job owned. With cards present and none selected it now refuses, naming the selection.

use std::process::Command;
use std::time::Duration;

const CHILD: &str = "cuda_follows_selection::child_selects_the_cpu_then_asks_for_cuda";

/// The child: the process selects the CPU, then asks for a CUDA handle.
#[test]
#[ignore = "run by a_cpu_selection_never_opens_a_card_through_cuda in a child process"]
fn child_selects_the_cpu_then_asks_for_cuda() {
    gpu_core::devices::select_compute("cpu", None).expect("the CPU is always selectable");
    match gpu_core::Gpu::try_new_cuda(&[]) {
        Ok(_) => println!("OPENED a CUDA card"),
        Err(e) => println!("REFUSED {e}"),
    }
}

#[test]
fn a_cpu_selection_never_opens_a_card_through_cuda() {
    if gpu_core::devices::gpus().is_empty() {
        brain_testutil::skip_unavailable("needs a machine with a GPU to show that none is opened");
        return;
    }
    let exe = std::env::current_exe().expect("the test binary");
    let out = brain_testutil::output_within(Command::new(exe).args([CHILD, "--exact", "--ignored", "--nocapture"]), Duration::from_secs(120));
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "the child failed:\n{text}\n{}", String::from_utf8_lossy(&out.stderr));
    assert!(text.contains("REFUSED") && text.contains("selection"), "a process that selected the CPU must not open a card through try_new_cuda; it printed:\n{text}");
}
