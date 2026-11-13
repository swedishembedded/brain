// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A saved timeline model behind the residency scheduler.
//!
//! A directory `TimelineModel::save` (or an ensemble's) wrote, installed in the model store with a manifest of family
//! `horizon` (`brain.manifest.json`, which the training service's `timeline_train` writes beside the model it
//! installs), is one served model under its own id (`local/<name>`). `activate` loads it once, on the assigned
//! device; the [`Instance`] owns it, so dropping the instance frees it. Three actions - `predict`, `eval` and
//! `calibrate` - whose schema and work come from `horizon::caps` and `horizon::lifecycle`, so this file holds no
//! second copy of either. Concurrent `predict` requests are one batch: `run_batch` puts every request's subjects
//! through one forward pass in device batches of the model's size and splits the answers back per request
//! (`horizon::caps::predict_batch`).
//!
//! `eval` judges the held model on the request's subjects; `calibrate` returns the calibration it would write and
//! writes nothing, since the served directory is the host's. Training is not an action of a loaded model: it is the
//! training service's `timeline_train`, which publishes the model it trains.

use std::path::{Path, PathBuf};

use capability::{ActionResult, Invocation, Manifest, Progress};
use horizon::ensemble::{Loaded, MEMBERS_DIR};
use horizon::saved::{VOCAB_FILE, WEIGHTS_FILE};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

/// A saved timeline model behind the scheduler, served under `id`.
pub struct HorizonResident {
    id: String,
    dir: PathBuf,
}

impl HorizonResident {
    /// From a directory `TimelineModel::save` or `TimelineEnsemble::save` wrote, served as `id`; `None` if it is
    /// neither.
    pub fn new(id: &str, dir: &Path) -> Option<HorizonResident> {
        if !Loaded::is_dir(dir) {
            eprintln!("brain: horizon {id} not served ({} holds neither {WEIGHTS_FILE} + {VOCAB_FILE} nor an ensemble)", dir.display());
            return None;
        }
        Some(HorizonResident { id: id.to_string(), dir: dir.to_path_buf() })
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
        horizon::caps::manifest_resident(&self.id)
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(self.id.as_str(), "default")
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // The weights plus a prediction batch's activations: the model is
        // small, so a flat allowance over the weights bounds the batch.
        MemCost::new(self.weights_bytes().saturating_mul(2) + (256 << 20), 0)
    }

    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        let loaded = crate::resident_llm::on_device(device, || Loaded::load(&self.dir))??;
        Ok(Box::new(HorizonInstance { loaded }))
    }
}

struct HorizonInstance {
    loaded: Loaded,
}

impl HorizonInstance {
    fn model(&self) -> Result<&Loaded, String> {
        Ok(&self.loaded)
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

        let resident = HorizonResident::new("local/timeline", &dir).expect("a saved model directory");
        assert_eq!(resident.manifest().model, "local/timeline");
        assert_eq!(resident.manifest().actions.iter().map(|a| a.name.as_str()).collect::<Vec<_>>(), ["predict", "eval", "calibrate"], "a loaded model does not train");
        let key = InstanceKey::new("local/timeline", "default");
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
