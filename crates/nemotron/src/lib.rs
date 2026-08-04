// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! NVIDIA Nemotron 3.5 ASR Streaming (0.6B): a FastConformer encoder + RNN-T
//! transducer. Built on the shared WGSL engine; parity-gated against the
//! HuggingFace reference. Kernels → encoder → RNN-T → greedy decode.

pub mod caps;
pub mod config;
pub mod encoder;
pub mod import;
pub mod kernels;
pub mod model;
pub mod reference;
pub mod stream;
pub mod tokenizer;
pub mod train;

pub use config::NemotronConfig;

/// Resolve a test-fixture path under the gitignored `testdata/` tree — never a
/// hardcoded absolute path (that invariant is enforced in `AGENTS.md`). The root is
/// `$BRAIN_TESTDATA`, defaulting to `<repo>/testdata` (populated by
/// `make fetch/testdata`). Tests skip themselves when the file is absent.
#[cfg(test)]
use brain_testutil::testdata;
/// Resolve the on-disk model-store directory for the real Nemotron checkpoint
/// tests import/parity-test against; see `brain_testutil::model_dir`. `None`
/// (no models dir resolvable) collapses to an empty path so the existing
/// `Path::new(&format!("{ckpt}/…")).exists()` skip checks stay correct.
#[cfg(test)]
use brain_testutil::model_dir;
