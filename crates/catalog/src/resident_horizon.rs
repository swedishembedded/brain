// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The timeline model behind the residency scheduler.
//!
//! `activate` loads the saved model from `BRAIN_HORIZON_DIR` once, on the
//! assigned device; the [`Instance`] owns it, so dropping the instance frees
//! it. One action, `predict` - schema and work both come from
//! `horizon::caps`, so this file holds no second copy of either. Requests run
//! serially: a request already carries a whole file of subjects, which the
//! model predicts in device batches of its own.

use std::path::{Path, PathBuf};

use capability::{ActionResult, Invocation, Manifest, Progress};
use horizon::saved::{Saved, VOCAB_FILE, WEIGHTS_FILE};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

/// A saved timeline model behind the scheduler.
pub struct HorizonResident {
    dir: PathBuf,
}

impl HorizonResident {
    /// `BRAIN_HORIZON_DIR` if the operator set it; `None` (not served, never
    /// a startup failure) otherwise, or when the directory holds no saved
    /// model.
    pub fn from_env() -> Option<HorizonResident> {
        let dir = std::env::var(horizon::caps::DIR_VAR)
            .ok()
            .filter(|d| !d.is_empty())?;
        Self::new(Path::new(&dir))
    }

    /// From a directory `TimelineModel::save` wrote; `None` if it is not one.
    pub fn new(dir: &Path) -> Option<HorizonResident> {
        let missing: Vec<&str> = [WEIGHTS_FILE, VOCAB_FILE]
            .into_iter()
            .filter(|f| !dir.join(f).is_file())
            .collect();
        if !missing.is_empty() {
            eprintln!(
                "brain: horizon not served ({} is missing {missing:?})",
                dir.display()
            );
            return None;
        }
        Some(HorizonResident {
            dir: dir.to_path_buf(),
        })
    }
}

impl ResidentModel for HorizonResident {
    fn manifest(&self) -> Manifest {
        horizon::caps::manifest_resident()
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(horizon::caps::MODEL, "default")
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // The weights plus a prediction batch's activations: the model is
        // small, so a flat allowance over the weights bounds the batch.
        let weights = std::fs::metadata(self.dir.join(WEIGHTS_FILE))
            .map(|m| m.len())
            .unwrap_or(0);
        MemCost::new(weights.saturating_mul(2) + (256 << 20), 0)
    }

    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        let saved = crate::resident_llm::on_device(device, || Saved::load(&self.dir))??;
        Ok(Box::new(HorizonInstance { saved }))
    }
}

struct HorizonInstance {
    saved: Saved,
}

impl Instance for HorizonInstance {
    fn run(
        &mut self,
        _action: &str,
        inv: &Invocation,
        progress: &mut dyn FnMut(Progress),
    ) -> ActionResult {
        progress(Progress::step(1, 1, "predict"));
        horizon::caps::predict(&self.saved, inv)
    }
}
