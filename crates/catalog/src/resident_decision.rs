// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Decision models behind the residency scheduler: one resident per
//! checkpoint, each addressed by its store id.
//!
//! * **Stable ids.** The id is the checkpoint's position in the model store,
//!   `<vendor>/<repo>`, so two checkpoints are two models a caller selects by
//!   name, and the name survives a restart. A checkpoint outside the store is
//!   `local/<directory name>`.
//! * **Lazy load.** Registration reads no weights. The scheduler calls
//!   `activate` on the first `decide` and drops the instance when it evicts
//!   the model. Whether a model is installed and fit to serve is answered
//!   without loading by the family model's `models` action
//!   ([`decisions::provider`]).
//! * **Bounded concurrency.** One instance per model, and the scheduler runs
//!   an instance's jobs one at a time on its device lane, so a model has at
//!   most one forward in flight; the rest wait in the scheduler's queue under
//!   its admission deadline, which sheds rather than grows without bound.
//! * **Eligibility.** The estimate declares GPU memory only, so the scheduler
//!   places a decision model on a GPU and never on the host tier.
//! * **Errors.** Load and request failures carry a
//!   [`capability::FailureClass`] tag; see [`decisions::load`] and
//!   [`decisions::caps`].
//! * **Batching.** `run_batch` is the serial default, deliberately: each
//!   architecture packs one request's state and questions into one bounded
//!   sequence, so concurrent requests share no pass. Within a request the
//!   `decide` architecture does encode the state once for all its questions.
//!
//! Swedish Embedded AB implements model residency for decision services:
//! models loaded when first asked, evicted when the card is needed, and
//! addressable by a name that does not change. If your team needs expertise
//! in serving many small models on shared accelerators, you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};

use std::sync::Arc;

use capability::{ActionResult, Invocation, Manifest, Progress};
use decisions::provider::{served_manifest, DecisionProvider};
use decisions::store::{Catalog, Readiness};
use residency::bridge::ProviderResident;
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

/// Scratch the activations of a decision forward need beyond the weights:
/// packed-sequence buffers and the head's workspace.
const SCRATCH_BYTES: u64 = 512 << 20;

/// The decision models an operator asked a serving process for, as parsed
/// flags: nothing is read from the environment.
#[derive(Clone, Debug, Default)]
pub struct DecisionServeConfig {
    /// `--decision-model`: model-store references or checkpoint directories,
    /// each registered whatever state its files are in.
    pub models: Vec<String>,
    /// `--decision-default`: the model id `"auto"` resolves to.
    pub default: Option<String>,
}

/// Every decision model a serving process schedules - each servable checkpoint
/// in the store plus each one named in `config` - and the family model that
/// answers "what is installed" and "what would `auto` pick".
///
/// A named model that cannot be served says so on stderr once, here, instead
/// of waiting for the first caller to find out.
pub fn serve_all(config: &DecisionServeConfig, models_dir: Option<&Path>) -> Vec<Arc<dyn ResidentModel>> {
    let catalog = Catalog::new(models_dir.map(Path::to_path_buf), config.models.clone(), config.default.clone());
    for record in catalog.records().iter().filter(|r| r.configured && !r.installed()) {
        eprintln!("brain: decision model '{}' is named but not ready ({})", record.id, record.readiness.slug());
    }
    let mut out: Vec<Arc<dyn ResidentModel>> = DecisionResident::all(&catalog).into_iter().map(|r| Arc::new(r) as Arc<dyn ResidentModel>).collect();
    out.push(Arc::new(ProviderResident::stateless_with_manifest(Arc::new(DecisionProvider::new(catalog)), served_manifest())));
    out
}

/// One decision checkpoint behind the scheduler.
pub struct DecisionResident {
    id: String,
    dir: PathBuf,
    weights_bytes: u64,
}

impl DecisionResident {
    /// A resident for the checkpoint `id` at `dir`. Registers whether or not
    /// the weights are present: a model that is named but missing answers
    /// `not-installed` instead of vanishing from the catalog.
    pub fn new(id: impl Into<String>, dir: impl Into<PathBuf>) -> DecisionResident {
        let dir = dir.into();
        let weights_bytes = std::fs::metadata(dir.join("model.safetensors")).map(|m| m.len()).unwrap_or(0);
        DecisionResident { id: id.into(), dir, weights_bytes }
    }

    /// Every decision model the operator named, plus every servable one in
    /// the store at `models_dir`.
    pub fn all(catalog: &Catalog) -> Vec<DecisionResident> {
        catalog
            .records()
            .into_iter()
            // A scan never offers a model that cannot be served; a NAMED
            // model is registered whatever its state, so its failure is
            // reported to the caller that asks for it.
            .filter(|r| r.configured || r.readiness == Readiness::Ready)
            .map(|r| DecisionResident::new(r.id, r.dir))
            .collect()
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

impl ResidentModel for DecisionResident {
    fn manifest(&self) -> Manifest {
        decisions::caps::manifest(&self.id)
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(&self.id, "default")
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // A bound, not a measurement: the weights may be held at twice their
        // on-disk width on the device (a bf16 file widened to f32), plus the
        // scratch above.
        MemCost::new(self.weights_bytes.saturating_mul(2) + SCRATCH_BYTES, 0)
    }

    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        // Built on the card the scheduler assigned, by scoped selection and
        // never by mutating the environment.
        let loaded = crate::resident_llm::on_device(device, || {
            decisions::load::load(&self.id, &self.dir, |backend| Ok(gpu_core::Gpu::new(decisions::load::pipelines(backend))))
        })??;
        Ok(Box::new(DecisionInstance { session: loaded.session }))
    }
}

struct DecisionInstance {
    session: decisions::caps::Session,
}

impl Instance for DecisionInstance {
    fn run(&mut self, _action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.session.run(inv)
    }
}
