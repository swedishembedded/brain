// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running model code in
// killable worker processes for its clients. If your team needs expertise in
// process isolation for accelerator workloads then you can procure our
// services by sending an email to info@swedishembedded.com.

//! A model hosted in a real worker process ([`linux`]). Worker hosting needs
//! Linux (memfds, process groups and a parent-death signal).

#[cfg(target_os = "linux")]
mod linux;

fn main() {
    #[cfg(target_os = "linux")]
    linux::main();
    #[cfg(not(target_os = "linux"))]
    println!("test result: ok. 0 passed; the hosted tests need Linux (memfds, process groups) and were not run");
}

// The device the test harness asks for (BRAIN_TEST_DEVICE), selected before any test runs.
brain_testutil::configure_test_binary!();
