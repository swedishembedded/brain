// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#[test]
fn all_kernels_compile() {
    let mut failed = Vec::new();
    for (name, src) in kernels::ALL {
        match wgsl_cpu::Jit::new(&[(name, src)]) {
            Ok(_) => {}
            Err(e) => failed.push(format!("{name}: {e}")),
        }
    }
    if !failed.is_empty() {
        panic!(
            "{} of {} kernels failed:\n{}",
            failed.len(),
            kernels::ALL.len(),
            failed.join("\n")
        );
    }
}
