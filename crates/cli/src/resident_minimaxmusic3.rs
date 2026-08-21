// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Resident-model adapter putting MiniMax Music 3 behind the residency
//! [`Executor`], mirroring [`crate::resident_tts::TtsResident`] - the
//! "load per call, nothing kept warm" shape, not `wan`'s hot-DiT cache.
//!
//! Config is env-only, matching `crates/arch`'s own `weights_env`
//! registration for this architecture and `minimaxmusic3::generate::
//! Paths::from_env`'s own six roles:
//!   * `BRAIN_MINIMAXMUSIC3_LM` - Global LLM (Qwen3-8B architecture)
//!   * `BRAIN_MINIMAXMUSIC3_DEPTH` - RVQ depth decoder
//!   * `BRAIN_MINIMAXMUSIC3_CONDITION` - condition encoder
//!   * `BRAIN_MINIMAXMUSIC3_DIT` - flow-matching DiT
//!   * `BRAIN_MINIMAXMUSIC3_VOCODER` - vocoder
//!   * `BRAIN_MINIMAXMUSIC3_TOKENIZER` - tokenizer
//!
//! One action, `generate`, dispatched straight through
//! `minimaxmusic3::caps::generate_action` - the SAME param-decode +
//! generation + outcome-shaping implementation
//! `minimaxmusic3::caps::MinimaxMusic3Provider` (the direct/`brain do`
//! path) uses, so this file adds no second copy of that logic.

use capability::{ActionResult, Invocation, Manifest, Progress};
use minimaxmusic3::generate::Paths;
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

/// MiniMax Music 3 behind the scheduler. Holds only resolved PATHS - every
/// real `generate` call loads (and drops) all five components fresh; see
/// `minimaxmusic3::generate`'s own module doc for why a warm cache would
/// be actively wrong here (the whole checkpoint does not fit in RAM even
/// once on the machine this port was built on).
pub struct MinimaxMusic3Resident {
    paths: Paths,
}

impl MinimaxMusic3Resident {
    /// Configure from the environment. Returns `None` (not served) when
    /// any of the six roles is unset, like
    /// [`crate::resident::YoloResident::from_env`].
    pub fn from_env() -> Option<MinimaxMusic3Resident> {
        Paths::from_env().ok().map(|paths| MinimaxMusic3Resident { paths })
    }

    fn dir_bytes(dir: &str) -> u64 {
        let Ok(rd) = std::fs::read_dir(dir) else { return 0 };
        rd.filter_map(|e| e.ok()).filter_map(|e| e.metadata().ok()).map(|m| m.len()).sum()
    }
}

impl ResidentModel for MinimaxMusic3Resident {
    fn manifest(&self) -> Manifest {
        // The spec lives in minimaxmusic3::caps, next to the catalog's own
        // `generate` spec, so the two surfaces cannot silently diverge.
        minimaxmusic3::caps::resident_manifest()
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(minimaxmusic3::caps::MODEL, "default")
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // Two sequential stages, never resident together (see
        // `minimaxmusic3::generate`'s own doc) - budget the LARGER one,
        // not their sum. The AR stage loads the Global LLM checkpoint
        // TWICE (one instance per CFG branch) at fp32 (`Qwen::
        // new_shard_i8`'s int8 request silently promotes to fp32 on a
        // backend without real int8 dispatch - measured directly, not
        // assumed; a bf16-on-disk checkpoint then costs 2x its own file
        // size per instance), plus the depth decoder once.
        let lm_bytes = Self::dir_bytes(&self.paths.lm);
        let depth_bytes = Self::dir_bytes(&self.paths.depth);
        let ar_stage = 2 * (2 * lm_bytes) + depth_bytes;

        let dit_bytes = Self::dir_bytes(&self.paths.dit);
        let vocoder_bytes = Self::dir_bytes(&self.paths.vocoder);
        let condition_bytes = Self::dir_bytes(&self.paths.condition);
        let denoise_stage = dit_bytes + vocoder_bytes + condition_bytes;

        let ram = ar_stage.max(denoise_stage);
        let ram = if ram > 0 { ram } else { 32u64 << 30 };
        MemCost::new(0, ram)
    }

    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        // `minimaxmusic3::generate::generate` loads every component
        // internally per call, so the resident's only job here is to fail
        // fast when the configured directories are missing.
        for (dir, role) in [
            (&self.paths.lm, "Global LLM"),
            (&self.paths.depth, "depth decoder"),
            (&self.paths.condition, "condition encoder"),
            (&self.paths.dit, "DiT"),
            (&self.paths.vocoder, "vocoder"),
            (&self.paths.tokenizer, "tokenizer"),
        ] {
            if !std::path::Path::new(dir).exists() {
                return Err(format!("minimaxmusic3: {role} weights not found at {dir} (set the matching BRAIN_MINIMAXMUSIC3_* var)"));
            }
        }
        Ok(Box::new(MinimaxMusic3Instance { paths: self.paths.clone() }))
    }
}

struct MinimaxMusic3Instance {
    paths: Paths,
}

impl Instance for MinimaxMusic3Instance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        if action != "generate" {
            return Err(format!("minimaxmusic3: unsupported action '{action}' (this resident declares: generate)"));
        }
        minimaxmusic3::caps::generate_action(&self.paths, inv, progress)
    }
}
