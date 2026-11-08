// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Link the SDL2 window's FFI against the system libSDL2.
//!
//! The window is always compiled in, so its symbols must resolve at link time
//! even though a headless run never opens it (see `src/sys.rs`).
//!
//! `rustc-link-lib=SDL2` names the library but says nothing about WHERE it
//! lives. That is fine where SDL2 sits in the linker's default search path (a
//! distro's `/usr/lib`, or a Windows SDK), but a macOS Homebrew install lands
//! in a prefix the linker does not search on its own - the first in the list
//! below, which only `brew --prefix sdl2` knows - so `-lSDL2` alone fails with
//! a bare `ld: library 'SDL2' not found` even after a correct `brew install sdl2`.
//! Emit the matching `rustc-link-search` so the two travel together.
//!
//! The lib dir is discovered by ASKING the tools that own the fact rather than
//! hard-coding a prefix: `pkg-config` (the formula ships an `sdl2.pc`) and,
//! on macOS, `brew --prefix sdl2`. Resolving it from a PATH tool is the same
//! discipline `crates/vulkan/build.rs` uses for `glslc`, and it keeps a
//! machine-specific absolute path out of tracked source (`make check/paths`).
//!
//! Discovery is best-effort and NEVER fails the build: when nothing turns up
//! we still emit `-lSDL2` and let the linker report the same error, plus a
//! macOS note naming what to install.

use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    if let Some(dir) = sdl2_libdir() {
        println!("cargo:rustc-link-search=native={dir}");
    } else if std::env::consts::OS == "macos" {
        println!(
            "cargo:warning=brain-wm-display: could not locate an SDL2 install. If the link \
             fails with `library 'SDL2' not found`, run `brew install sdl2`; if SDL2 lives \
             somewhere unusual, add its lib dir with `RUSTFLAGS=\"-L<dir>\"`."
        );
    }

    println!("cargo:rustc-link-lib=SDL2");
}

/// The directory holding `libSDL2`, or `None` when neither `pkg-config` nor
/// Homebrew can name one (the caller then falls back to the bare `-lSDL2`).
fn sdl2_libdir() -> Option<String> {
    // `pkg-config` knows the prefix from the shipped `sdl2.pc`. Absent on many
    // macOS boxes, so it is tried first but never the only path.
    if let Some(dir) = command_stdout("pkg-config", &["--variable=libdir", "sdl2"]) {
        if has_lib_sdl2(&dir) {
            return Some(dir);
        }
    }

    // Homebrew's own answer for "where did I install sdl2", which is exactly
    // the prefix the linker needs and needs no `HOMEBREW_PREFIX` guess.
    if std::env::consts::OS == "macos" {
        if let Some(prefix) = command_stdout("brew", &["--prefix", "sdl2"]) {
            let dir = format!("{prefix}/lib");
            if has_lib_sdl2(&dir) {
                return Some(dir);
            }
        }
    }

    None
}

/// Run `prog args…` and return its trimmed stdout, or `None` if the program is
/// missing, exits non-zero, or prints nothing.
fn command_stdout(prog: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(prog).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?.trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

/// Does `dir` contain a linkable `libSDL2` (dylib or static archive)?
fn has_lib_sdl2(dir: &str) -> bool {
    let dir = Path::new(dir);
    ["libSDL2.dylib", "libSDL2.a", "libSDL2.so"]
        .iter()
        .any(|name| dir.join(name).exists())
}
