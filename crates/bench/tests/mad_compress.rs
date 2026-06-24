// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MAD compression placeholder guard.
//!
//! The compression (autoencoder) task cannot be expressed via the current
//! next-token `gpt::train` API, so the benchmark is a documented placeholder:
//! `evaluate` returns an `Unsupported` error and the task is not registered.
//! This test pins that contract (and needs no accelerator, so it is not gated).

use bench::mad_compress::MadCompress;
use bench::{known_names, Benchmark};

#[test]
fn mad_compress_is_unsupported_and_unregistered() {
    let b = MadCompress;
    // Not in the live registry (it would always fail its threshold).
    assert!(!known_names().contains(&"mad_compress".to_string()));

    // evaluate surfaces a clear Unsupported error, not a misleading score.
    let err = b.evaluate(&std::env::temp_dir(), 0).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
    assert!(err.to_string().contains(MadCompress::UNSUPPORTED));
}
