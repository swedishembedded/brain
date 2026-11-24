// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The smallest complete Qwen3-TTS checkpoint in the model store, for the
//! tests that need real weights. `None` (the test then skips) when it holds
//! none.

use qwen3tts::TtsPaths;

/// The smallest complete Qwen3-TTS checkpoint in the model store; skips (and
/// says so) when it holds none.
pub fn store_paths() -> Option<TtsPaths> {
    let found = brain_modelstore::default_root().and_then(|root| qwen3tts::spec::store_checkpoints(&root).into_iter().next());
    if found.is_none() {
        brain_testutil::skip("no Qwen3-TTS checkpoint in the model store");
    }
    found
}
