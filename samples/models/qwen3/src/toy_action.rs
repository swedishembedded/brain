// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving training methods on a model small enough to train in seconds,
// for its clients. If your team needs expertise in fast, falsifiable tests of fine-tuning pipelines then you can
// procure our services by sending an email to info@swedishembedded.com.

//! `toy`: writes a tiny randomly initialised chat checkpoint (2 layers, 32 wide, a byte-level tokenizer, a bare
//! chat template) that trains and samples on the CPU in seconds, so a consumer's tests need no real weights. The
//! same seed writes the same weights; without one a random seed is drawn and reported. `store_layout` names the
//! weights file as the model store reads it (`model.brain.safetensors`), so `out` may be `<models-dir>/local/<name>`
//! and the checkpoint is then served and trained on by that name.

use std::path::Path;

use capability::{Action, ActionResult, ActionSpec, Invocation, Outcome, ParamSpec, ParamType, Progress};
use serde_json::json;

pub fn spec() -> ActionSpec {
    ActionSpec::new("toy", "write a tiny randomly initialised chat checkpoint that trains and samples on the CPU in seconds, for tests that need no real weights")
        .param(ParamSpec::new("out", ParamType::Str, "the directory to write model.safetensors, tokenizer.json and tokenizer_config.json into").host_resolved().required())
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed: the same seed writes the same weights (omit for random; the outcome echoes the one used)"))
        .param(ParamSpec::new("store_layout", ParamType::Bool, "name the weights as the model store reads them (model.brain.safetensors), so `out` can be <models-dir>/local/<name>").default(json!(false)))
}

pub struct Toy;

impl Action for Toy {
    fn spec(&self) -> ActionSpec {
        spec()
    }

    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let out = inv.get_str("out").filter(|o| !o.is_empty()).ok_or("qwen3 toy: missing required param 'out'")?;
        let seed = inv.get_i64("seed").map_or_else(data::rng::random_seed, |s| s as u64);
        qwen3::toy::write_toy_chat_checkpoint(Path::new(&out), seed).map_err(|e| format!("qwen3 toy: {e}"))?;
        if inv.get_bool("store_layout").unwrap_or(false) {
            std::fs::rename(Path::new(&out).join("model.safetensors"), Path::new(&out).join("model.brain.safetensors")).map_err(|e| format!("qwen3 toy: {out}: {e}"))?;
        }
        Ok(Outcome::new().set("saved", json!(out)).set("seed", json!(seed)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(inv: Invocation) -> ActionResult {
        let inv = spec().validate(inv)?;
        Toy.run(&inv, &mut |_| {})
    }

    /// The toy checkpoint is written with the files a consumer's tests load, the same seed writes the same
    /// weights, and in the model store's layout the store finds it by name.
    #[test]
    fn toy_writes_a_loadable_checkpoint_the_store_finds_by_name() {
        let dir = std::env::temp_dir().join(format!("qwen3-toy-{}", std::process::id()));
        let (a, b) = (dir.join("a"), dir.join("b"));
        let at = |out: &Path| Invocation::new().set("out", json!(out.display().to_string())).set("seed", json!(5));
        assert_eq!(run(at(&a)).unwrap().outputs["seed"], 5);
        run(at(&b)).unwrap();
        for file in ["model.safetensors", "tokenizer.json", "tokenizer_config.json"] {
            assert!(a.join(file).is_file(), "{file}");
        }
        assert_eq!(std::fs::read(a.join("model.safetensors")).unwrap(), std::fs::read(b.join("model.safetensors")).unwrap());
        let models = dir.join("models");
        run(at(&models.join("local").join("toy")).set("store_layout", json!(true))).unwrap();
        let found = brain_modelstore::Store::new(&models).local(&brain_modelref::ModelRef::parse("local/toy").unwrap());
        assert!(found.is_some_and(|m| m.weights.ends_with("model.brain.safetensors")), "the store finds the toy checkpoint by name");
        assert!(run(Invocation::new()).unwrap_err().contains("out"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
