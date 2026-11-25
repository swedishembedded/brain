// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The smallest Qwen3-TTS checkpoint in the model store that clones a voice,
//! for the tests that need real weights. `None` (the test then skips) when it
//! holds none.

use qwen3tts::TtsPaths;

/// The smallest Qwen3-TTS checkpoint in the model store that clones a voice (its
/// speaker encoder included): what the real-weight tests speak, clone and train
/// with. Skips (and says so) when it holds none. A store also holds checkpoints
/// without one (a CustomVoice model, a local smoke export), which are smaller.
pub fn store_paths() -> Option<TtsPaths> {
    let found = brain_modelstore::default_root().and_then(|root| qwen3tts::spec::store_checkpoints(&root).into_iter().find(|p| p.require(true).is_ok()));
    if found.is_none() {
        brain_testutil::skip("no Qwen3-TTS checkpoint in the model store");
    }
    found
}
