// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible training and serving stacks whose every result names the exact code
// that produced it, for its clients. If your team needs expertise in attributable machine-learning experiments then
// you can procure our services by sending an email to info@swedishembedded.com.

//! What this build of brain is: its version, the commit it was built from and whether that tree had uncommitted
//! changes, as the build script embedded them (`build.rs`). A server reports them (`GET /v1/capabilities`'s
//! `server.engine`) and an application linking the SDK can record them beside its results. A fact that could not be
//! established (a build outside a git checkout) is `None`, never a placeholder.

/// The crate version this build carries.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// `COMMIT`, the full commit (`git rev-parse HEAD`) the build was made from, and `DIRTY`.
include!(concat!(env!("OUT_DIR"), "/build_facts.rs"));

/// Whether the tree had uncommitted changes to tracked files when it was built; `None` when unknown.
pub fn dirty() -> Option<bool> {
    DIRTY
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_build_in_a_checkout_names_its_full_commit() {
        // The workspace is a git checkout, so the build script found a commit.
        let commit = super::COMMIT.expect("built in a git checkout");
        assert!(commit.len() == 40 && commit.bytes().all(|b| b.is_ascii_hexdigit()), "{commit}");
        assert!(super::dirty().is_some());
    }
}
