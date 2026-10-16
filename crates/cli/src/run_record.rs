// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The record a generating command leaves beside what it generated.
//!
//! A seed printed on stderr is gone by the time someone asks how a picture,
//! clip or recording was made. Every command that writes generated media also
//! writes `<out>.json` next to it: the command, the seed the run drew or was
//! given, and the decoding parameters, so the output can be reproduced from
//! the record alone.
//!
//! Swedish Embedded AB implements generation pipelines whose every output
//! carries the seed and settings that made it. If your team needs expertise in
//! reproducible generative media then you can procure our services by sending
//! an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

/// Where the record of `out` lives: the output's own file name plus `.json`,
/// so `clip.mp4` is described by `clip.mp4.json` and never collides with a
/// sibling that differs only in extension.
pub fn record_path(out: &str) -> PathBuf {
    let mut name = Path::new(out).as_os_str().to_owned();
    name.push(".json");
    PathBuf::from(name)
}

/// Write the record of one generation beside `out`. `params` holds the
/// decoding parameters (prompt, size, steps, guidance, ...) as the command
/// resolved them. Fails, naming the path, when the record cannot be written:
/// an output without its record is not reproducible, and saying so beats
/// leaving it unnoticed.
pub fn write_run_record(out: &str, command: &str, seed: u64, params: Value) -> Result<(), String> {
    let record = json!({ "command": command, "output": out, "seed": seed, "params": params });
    let path = record_path(out);
    let text = serde_json::to_string_pretty(&record).map_err(|e| format!("run record for {out}: {e}"))?;
    std::fs::write(&path, text + "\n").map_err(|e| format!("run record {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The record sits beside the output, names the seed and keeps the
    /// parameters it was given.
    #[test]
    fn a_generation_leaves_its_seed_and_parameters_beside_it() {
        let dir = std::env::temp_dir().join(format!("run-record-{}", data::rng::random_seed()));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let out = dir.join("clip.mp4");
        let out = out.to_str().expect("utf-8 path");

        write_run_record(out, "wan t2v", 42, json!({ "steps": 20, "prompt": "a cat" })).expect("record written");

        let text = std::fs::read_to_string(dir.join("clip.mp4.json")).expect("record file");
        let record: Value = serde_json::from_str(&text).expect("json");
        assert_eq!(record["seed"], 42);
        assert_eq!(record["command"], "wan t2v");
        assert_eq!(record["params"]["steps"], 20);
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    #[test]
    fn a_record_that_cannot_be_written_is_an_error_naming_the_path() {
        let err = write_run_record("/nonexistent-dir-for-run-record/x.png", "flux2", 1, json!({})).expect_err("no such directory");
        assert!(err.contains("x.png.json"), "{err}");
    }
}
