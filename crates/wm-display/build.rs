// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

fn main() {
    // Link the system SDL2 only when the window is compiled in. Headless
    // builds (CI) never touch it.
    if std::env::var_os("CARGO_FEATURE_SDL").is_some() {
        println!("cargo:rustc-link-lib=SDL2");
    }
}
