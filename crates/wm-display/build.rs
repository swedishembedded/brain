// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

fn main() {
    // The SDL2 window is always compiled in; link the system libSDL2. The
    // window is only OPENED when a run needs it (SdlWindow::new), so headless
    // runs never touch SDL at runtime — but the symbols must resolve at link.
    println!("cargo:rustc-link-lib=SDL2");
}
