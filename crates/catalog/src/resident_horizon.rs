// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The timeline model behind the residency scheduler.
//!
//! `activate` loads the saved model (or ensemble) from `BRAIN_HORIZON_DIR`
//! once, on the assigned device; the [`Instance`] owns it, so dropping the
//! instance frees it. Four actions - `predict`, `eval`, `calibrate` and
//! `train` - whose schema and work come from `horizon::caps` and
//! `horizon::lifecycle`, so this file holds no second copy of either.
//! Concurrent `predict` requests are one batch: `run_batch` puts every
//! request's subjects through one forward pass in device batches of the
//! model's size and splits the answers back per request
//! (`horizon::caps::predict_batch`).
//!
//! `eval` judges the held model on the request's subjects; `calibrate`
//! returns the calibration it would write and writes nothing, since the served
//! directory is the host's. `train` needs no loaded weights: it has its own
//! instance (`config: "train"`, nothing loaded) and writes the finished model to
//! the host's `BRAIN_HORIZON_TRAIN_DIR`, polling the job's cancel token every
//! step.

use std::path::{Path, PathBuf};

use capability::{ActionResult, Invocation, Manifest, Progress};
use horizon::ensemble::{Loaded, MEMBERS_DIR};
use horizon::saved::{VOCAB_FILE, WEIGHTS_FILE};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

/// The instance config of the `train` action: no weights are loaded for it.
const TRAIN_CONFIG: &str = "train";
/// What a training run is budgeted for on the device: the models are small,
/// so a flat allowance bounds the batches and the optimiser state.
const TRAIN_BUDGET: u64 = 1 << 30;

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

    /// From a directory `TimelineModel::save` or `TimelineEnsemble::save`
    /// wrote; `None` if it is neither.
    pub fn new(dir: &Path) -> Option<HorizonResident> {
        if !Loaded::is_dir(dir) {
            eprintln!(
                "brain: horizon not served ({} holds neither {WEIGHTS_FILE} + {VOCAB_FILE} nor an ensemble)",
                dir.display()
            );
            return None;
        }
        Some(HorizonResident {
            dir: dir.to_path_buf(),
        })
    }

    /// The size of every weights file below the directory: the model's, or
    /// each member's.
    fn weights_bytes(&self) -> u64 {
        let size = |dir: &Path| {
            std::fs::metadata(dir.join(WEIGHTS_FILE))
                .map(|m| m.len())
                .unwrap_or(0)
        };
        let members = std::fs::read_dir(self.dir.join(MEMBERS_DIR))
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| size(&e.path()))
            .sum::<u64>();
        size(&self.dir) + members
    }
}

impl ResidentModel for HorizonResident {
    fn manifest(&self) -> Manifest {
        horizon::caps::manifest_resident()
    }

    fn instance_key(&self, action: &str, _inv: &Invocation) -> InstanceKey {
        let config = if action == "train" { TRAIN_CONFIG } else { "default" };
        InstanceKey::new(horizon::caps::MODEL, config)
    }

    fn estimate(&self, key: &InstanceKey) -> MemCost {
        if key.config == TRAIN_CONFIG {
            return MemCost::new(TRAIN_BUDGET, 0);
        }
        // The weights plus a prediction batch's activations: the model is
        // small, so a flat allowance over the weights bounds the batch.
        MemCost::new(self.weights_bytes().saturating_mul(2) + (256 << 20), 0)
    }

    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        if key.config == TRAIN_CONFIG {
            return Ok(Box::new(HorizonInstance { loaded: None }));
        }
        let loaded = crate::resident_llm::on_device(device, || Loaded::load(&self.dir))??;
        Ok(Box::new(HorizonInstance { loaded: Some(loaded) }))
    }
}

struct HorizonInstance {
    /// `None` for the training instance.
    loaded: Option<Loaded>,
}

impl HorizonInstance {
    fn model(&self) -> Result<&Loaded, String> {
        self.loaded.as_ref().ok_or_else(|| "horizon: this instance holds no model".to_string())
    }
}

impl Instance for HorizonInstance {
    fn run(
        &mut self,
        action: &str,
        inv: &Invocation,
        progress: &mut dyn FnMut(Progress),
    ) -> ActionResult {
        match action {
            "predict" => {
                progress(Progress::step(1, 1, "predict"));
                horizon::caps::predict(self.model()?, inv)
            }
            "eval" => {
                progress(Progress::step(1, 1, "eval"));
                horizon::lifecycle::eval(self.model()?, inv)
            }
            "calibrate" => {
                progress(Progress::step(1, 1, "calibrate"));
                horizon::lifecycle::calibrate_loaded(self.model()?, inv)
            }
            "train" => horizon::lifecycle::train(inv, progress),
            other => Err(format!("horizon: no action '{other}'")),
        }
    }

    /// One forward pass over every `predict` request's subjects; a request
    /// that cannot be answered fails alone. The other actions answer one
    /// request at a time (training and evaluation are not batchable).
    fn run_batch(
        &mut self,
        action: &str,
        invs: &[Invocation],
        progress: &mut dyn FnMut(usize, Progress),
    ) -> Vec<ActionResult> {
        if action != "predict" {
            return invs
                .iter()
                .enumerate()
                .map(|(i, inv)| self.run(action, inv, &mut |p| progress(i, p)))
                .collect();
        }
        for i in 0..invs.len() {
            progress(i, Progress::step(1, 1, "predict"));
        }
        match self.model() {
            Ok(loaded) => horizon::caps::predict_batch(loaded, invs),
            Err(e) => invs.iter().map(|_| Err(e.clone())).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::{Blob, Media};
    use horizon::saved::Saved;
    use horizon::synthetic::{population, CODES};
    use horizon::vocab::{FitOptions, Vocab};
    use horizon::{Horizon, HorizonConfig};
    use serde_json::json;

    fn request(subjects: &[horizon::timeline::Subject]) -> Invocation {
        let jsonl: String = subjects
            .iter()
            .map(|s| serde_json::to_string(s).unwrap() + "\n")
            .collect();
        Invocation::new()
            .set("times", json!("2,5"))
            .blob("subjects", Blob::new(Media::Text, jsonl.into_bytes()))
    }

    /// The served `run_batch` answers every request exactly as `run` does and
    /// a request that cannot be answered fails alone.
    #[test]
    fn run_batch_answers_each_request_as_run_does() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let (subjects, _) = population(30, 9);
        let codes: Vec<String> = CODES.iter().map(|c| c.to_string()).collect();
        let vocab = Vocab::fit(&subjects, &codes, &codes[..2], &FitOptions::default()).unwrap();
        let mut cfg = HorizonConfig::default_for(vocab.len(), CODES.len() as u32);
        cfg.max_tokens = 8;
        cfg.d_model = 16;
        cfg.n_heads = 2;
        cfg.d_ff = 32;
        cfg.rank = 8;
        cfg.knots = vec![0.0, 2.0, 5.0, 10.0];
        let model = Horizon::new(cfg.clone(), 8, &horizon::init_weights(&cfg, 3));
        let dir = std::env::temp_dir().join(format!("catalog-horizon-{}", std::process::id()));
        Saved::new(model, vocab).save(&dir).unwrap();

        let resident = HorizonResident::new(&dir).expect("a saved model directory");
        let key = InstanceKey::new(horizon::caps::MODEL, "default");
        let mut instance = resident.activate(&key, Device::Gpu(0)).unwrap();
        let invs = [
            request(&subjects[..10]),
            Invocation::new().set("times", json!("2")), // no subjects blob
            request(&subjects[10..13]),
        ];
        let mut seen = Vec::new();
        let batched = instance.run_batch("predict", &invs, &mut |i, _| seen.push(i));
        assert_eq!(seen, vec![0, 1, 2], "progress is reported per request");
        assert!(batched[1].is_err(), "the bad request fails alone");
        for i in [0, 2] {
            let alone = instance.run("predict", &invs[i], &mut |_| {}).unwrap();
            let got = batched[i].as_ref().unwrap();
            assert_eq!(got.outputs["subjects"], alone.outputs["subjects"]);
            let lines = |o: &capability::Outcome| -> Vec<serde_json::Value> {
                std::str::from_utf8(&o.blobs["predictions"].bytes)
                    .unwrap()
                    .lines()
                    .map(|l| serde_json::from_str(l).unwrap())
                    .collect()
            };
            assert_eq!(lines(got), lines(&alone), "request {i}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
