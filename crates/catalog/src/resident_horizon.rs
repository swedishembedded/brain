// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The timeline model behind the residency scheduler.
//!
//! `activate` loads the saved model from `BRAIN_HORIZON_DIR` once, on the
//! assigned device; the [`Instance`] owns it, so dropping the instance frees
//! it. One action, `predict` - schema and work both come from
//! `horizon::caps`, so this file holds no second copy of either. Concurrent
//! requests are one batch: `run_batch` puts every request's subjects through
//! one forward pass in device batches of the model's size and splits the
//! answers back per request (`horizon::caps::predict_batch`).

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

    /// One forward pass over every request's subjects; a request that cannot
    /// be answered fails alone.
    fn run_batch(
        &mut self,
        _action: &str,
        invs: &[Invocation],
        progress: &mut dyn FnMut(usize, Progress),
    ) -> Vec<ActionResult> {
        for i in 0..invs.len() {
            progress(i, Progress::step(1, 1, "predict"));
        }
        horizon::caps::predict_batch(&self.saved, invs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::{Blob, Media};
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
