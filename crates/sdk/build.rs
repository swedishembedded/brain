// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible training and serving stacks whose every result names the exact code
// that produced it, for its clients. If your team needs expertise in attributable machine-learning experiments then
// you can procure our services by sending an email to info@swedishembedded.com.

//! Embeds the commit this build is made from and whether its tree had uncommitted changes (`brain::build`), so a
//! server or an application linking the SDK can name them without running git. Outside a git checkout nothing is
//! embedded and the commit is unknown, never guessed.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    // The paths that move when HEAD, the current branch or the index do.
    let mut watched = vec!["HEAD".to_string(), "index".to_string()];
    if let Some(branch) = git(&["symbolic-ref", "-q", "HEAD"]) {
        watched.push(branch);
    }
    for name in watched {
        if let Some(path) = git(&["rev-parse", "--git-path", &name]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    let rev = git(&["rev-parse", "HEAD"]).filter(|rev| rev.len() == 40 && rev.bytes().all(|b| b.is_ascii_hexdigit()));
    let dirty = rev.as_ref().and_then(|_| git(&["status", "--porcelain", "--untracked-files=no"])).map(|status| !status.is_empty());
    // Written as Rust for `src/build.rs` to include: the facts are the build's, not settings a process reads.
    let generated = format!("/// The full commit (`git rev-parse HEAD`) the build was made from.\npub const COMMIT: Option<&str> = {rev:?};\nconst DIRTY: Option<bool> = {dirty:?};\n");
    let out = std::path::Path::new(&std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR")).join("build_facts.rs");
    std::fs::write(&out, generated).expect("the build script writes its facts into OUT_DIR");
}
