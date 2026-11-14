// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A model's served actions run in this process through its own residency
//! adapter: the provider builds the adapter over the checkpoint the store
//! resolved, activates one instance on the emptiest card (the host without
//! one) on the first request, and keeps it for the next request of the same
//! shape. `brain <model> <action>`, a sample and `brain serve` therefore run
//! the same engine with the same arithmetic.
//!
//! An adapter's engine sizing (context, page size, recurrent depth, ...) is
//! part of building it, not of one request; actions declare such knobs as
//! host-side params, and the provider rebuilds the adapter when they change.
//!
//! Swedish Embedded AB implements local and served inference engines that
//! share one code path for its clients. If your team needs expertise in
//! model serving on its own hardware then you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::sync::{Arc, Mutex};

use capability::{Action, ActionResult, ActionSpec, Invocation, Manifest, ParamSpec, ParamType, Progress, Provider};
use serde_json::json;
use residency::{Device, Instance, InstanceKey, ResidentModel};

/// Builds the adapter for the engine sizing an invocation names.
pub type BuildResident = dyn Fn(&Invocation) -> Result<Arc<dyn ResidentModel>, String> + Send + Sync;

/// The engine-sizing part of an invocation: invocations with equal keys share
/// one built adapter.
pub type SizingKey = dyn Fn(&Invocation) -> String + Send + Sync;

/// A [`Provider`] over a residency adapter, plus actions of the model that
/// need no engine (`extra`: conversion, import).
pub struct ResidentProvider {
    manifest: Manifest,
    state: Arc<State>,
    extra: Vec<Arc<dyn Action>>,
}

struct State {
    build: Box<BuildResident>,
    sizing: Box<SizingKey>,
    /// The adapter and the live instance, and the sizing and key they were
    /// built for: one at a time, so a request of another shape frees the old
    /// engine before building its own.
    live: Mutex<Live>,
}

#[derive(Default)]
struct Live {
    resident: Option<(String, Arc<dyn ResidentModel>)>,
    instance: Option<(String, InstanceKey, Box<dyn Instance>)>,
}

impl ResidentProvider {
    pub fn new(manifest: Manifest, build: Box<BuildResident>, sizing: Box<SizingKey>, extra: Vec<Arc<dyn Action>>) -> ResidentProvider {
        ResidentProvider { manifest, state: Arc::new(State { build, sizing, live: Mutex::new(Live::default()) }), extra }
    }
}

impl Provider for ResidentProvider {
    fn manifest(&self) -> Manifest {
        self.manifest.clone()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        if let Some(a) = self.extra.iter().find(|a| a.spec().name == name) {
            return Some(a.clone());
        }
        let spec = self.manifest.actions.iter().find(|a| a.name == name)?.clone();
        Some(Arc::new(ResidentAction { spec, state: self.state.clone() }))
    }
}

struct ResidentAction {
    spec: ActionSpec,
    state: Arc<State>,
}

/// The card with the most free memory, else the host.
fn emptiest_device() -> Device {
    gpu_core::capacity::available_gpus().into_iter().max_by_key(|&(_, free)| free).map_or(Device::Cpu, |(i, _)| Device::Gpu(i))
}

impl Action for ResidentAction {
    fn spec(&self) -> ActionSpec {
        self.spec.clone()
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        // Before the lock and any build or activation: a cancelled request builds nothing.
        inv.cancel.check()?;
        let mut live = self.state.live.lock().map_err(|_| format!("{}: an earlier request panicked while holding the model", self.spec.name))?;
        let sizing = (self.state.sizing)(inv);
        if live.resident.as_ref().is_none_or(|(s, _)| *s != sizing) {
            // Free the old engine before building the new one: two would not
            // fit where one does.
            live.instance = None;
            live.resident = Some((sizing.clone(), (self.state.build)(inv)?));
        }
        let resident = live.resident.as_ref().map(|(_, r)| r.clone()).expect("built above");
        let key = resident.instance_key(&self.spec.name, inv);
        if live.instance.as_ref().is_none_or(|(s, k, _)| *s != sizing || *k != key) {
            live.instance = None;
            live.instance = Some((sizing, key.clone(), resident.activate(&key, emptiest_device())?));
        }
        let (_, _, instance) = live.instance.as_mut().expect("activated above");
        instance.run(&self.spec.name, inv, progress)
    }
}

/// Largest KV page a run may ask for, tokens.
const MAX_BLOCK: f64 = 4_096.0;

/// The paged-KV sizing of a decoder engine, as host-side params of its
/// actions: the shared KV capacity and its page size (a serving process takes
/// them from its policy flags instead).
pub fn kv_sizing_params(default_ctx: u32, default_block: u32) -> Vec<ParamSpec> {
    vec![
        ParamSpec::new("ctx", ParamType::Int, "shared KV capacity, tokens").default(json!(default_ctx)).min(1.0).max(capability::bounds::CONTEXT_TOKENS).host_resolved(),
        ParamSpec::new("block", ParamType::Int, "KV page size, tokens").default(json!(default_block)).min(1.0).max(MAX_BLOCK).host_resolved(),
    ]
}

/// The `import` action of a model that converts a Hugging Face checkpoint
/// into brain's: a host-side conversion that needs no checkpoint resolved.
pub fn import_spec(summary: &str) -> ActionSpec {
    ActionSpec::new("import", summary)
        .from_scratch()
        .param(ParamSpec::new("source", ParamType::Str, "the Hugging Face checkpoint directory").required().host_resolved())
        .param(ParamSpec::new("dest", ParamType::Str, "where to write the .safetensors").required().host_resolved())
}
