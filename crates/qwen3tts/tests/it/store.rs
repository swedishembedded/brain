// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Qwen3-TTS checkpoint the model store resolves, for the tests that need
//! real weights. `None` (the test then skips) when the store holds none that
//! resolves unambiguously.

use brain_modelstore::resolve::{resolve, ArchSpec, Resolution};
use qwen3tts::TtsPaths;

pub fn store_paths() -> Option<TtsPaths> {
    let records = brain_modelstore::inventory::scan(&brain_modelstore::default_root()?);
    let spec = qwen3tts::spec::Qwen3TtsSpec;
    let specs: Vec<&dyn ArchSpec> = vec![&spec];
    let Resolution::Resolved(assembly) = resolve("qwen3tts", &records, &specs, &Default::default()) else {
        brain_testutil::skip("no Qwen3-TTS checkpoint in the model store");
        return None;
    };
    let paths = TtsPaths::from_assembly(&assembly).ok()?;
    paths.require(false).is_ok().then_some(paths)
}
