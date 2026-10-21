// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded model residency for its clients. If your team needs expertise in
// keeping many large quantized models available on a few accelerators, without ever holding a model as full
// precision on the host, then you can procure our services by sending an email to info@swedishembedded.com.

//! A residency [`Instance`] that can leave its accelerator and come back without being rebuilt from the file:
//! `demote` drops the engine and keeps only the checkpoint's own packed bytes (a mapping of the file, evictable
//! page cache), `promote` uploads them again at the tier they are stored in. Nothing between the two is an f32
//! copy: the host holds the bytes the file holds.

use std::sync::Arc;

use capability::{ActionResult, Invocation, Progress};
use residency::{Device, Instance, JobFeed, Tier};

/// The packed weights of a checkpoint, kept on the host while an instance is not on a card: the open source (a
/// mapping of the file, read through its packed view) and the names of the tensors it serves.
pub(crate) struct HostWeights {
    source: Box<dyn checkpoint::TensorSource + Send + Sync>,
    names: Vec<String>,
}

impl HostWeights {
    /// Open the brain checkpoint at `path` in the encoding it stores, returning its config beside the weights.
    /// `names` lists the tensors the config says the model serves.
    pub(crate) fn open(path: &str, names: impl FnOnce(&serde_json::Value) -> Vec<String>) -> Result<(serde_json::Value, HostWeights), String> {
        let (config, source) = crate::resident_llm::open_stored_checkpoint(path)?;
        let names = names(&config);
        Ok((config, HostWeights { source, names }))
    }

    pub(crate) fn source(&self) -> &dyn checkpoint::TensorSource {
        &*self.source
    }

    /// Give the file pages behind every tensor back to the kernel: the mapping stays, the resident pages go.
    fn release_pages(&self) {
        for name in &self.names {
            self.source.advise_drop(name);
        }
    }
}

/// How an instance is built on `device` from the host weights.
pub(crate) type Build = Box<dyn Fn(&HostWeights, Device) -> Result<Box<dyn Instance>, String> + Send>;

/// An [`Instance`] that opts into residency demotion: Hot is the wrapped instance on a card, Warm and Cold keep
/// the host weights alone (Cold also returns their pages to the kernel).
pub(crate) struct ParkableInstance {
    live: Option<Box<dyn Instance>>,
    host: Arc<HostWeights>,
    build: Build,
}

impl ParkableInstance {
    /// Build the instance on `device` from `host`; `build` is called again by [`Instance::promote`].
    pub(crate) fn activate(host: Arc<HostWeights>, device: Device, build: Build) -> Result<ParkableInstance, String> {
        let live = build(&host, device)?;
        Ok(ParkableInstance { live: Some(live), host, build })
    }
}

const PARKED: &str = "this instance is demoted: it must be promoted before it runs";

impl Instance for ParkableInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        match self.live.as_mut() {
            Some(live) => live.run(action, inv, progress),
            None => Err(PARKED.to_string()),
        }
    }

    fn run_batch(&mut self, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        match self.live.as_mut() {
            Some(live) => live.run_batch(action, invs, progress),
            None => invs.iter().map(|_| Err(PARKED.to_string())).collect(),
        }
    }

    fn run_open(&mut self, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress), feed: &mut dyn JobFeed) -> bool {
        self.live.as_mut().is_some_and(|live| live.run_open(action, invs, progress, feed))
    }

    fn metrics(&self) -> Vec<(String, serde_json::Value)> {
        self.live.as_ref().map(|live| live.metrics()).unwrap_or_default()
    }

    fn demote(&mut self, tier: Tier) -> Result<(), String> {
        match tier {
            Tier::Hot => return Err("Hot is not a demotion target".to_string()),
            Tier::Warm => {}
            Tier::Cold => self.host.release_pages(),
        }
        // Dropping the engine frees its device buffers; the mapping is all that is left.
        self.live = None;
        Ok(())
    }

    fn promote(&mut self, device: Device) -> Result<(), String> {
        if self.live.is_none() {
            self.live = Some((self.build)(&self.host, device)?);
        }
        Ok(())
    }
}
