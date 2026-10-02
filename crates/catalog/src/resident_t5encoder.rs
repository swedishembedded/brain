// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! T5-XXL / umT5-XXL behind the residency scheduler.
//!
//! `activate` builds a [`t5encoder::caps::Session`] on the assigned device;
//! the encoders themselves import lazily inside it on first use, keyed by
//! `(variant, batch, max_len)`. Both actions' schemas and all of their work
//! come from `t5encoder::caps`, so this file holds no second copy of the
//! tokenisation or the padding-mask handling.
//!
//! # Batching: grouped by (variant, max_len), the same shape as CLIP's
//!
//! [`t5encoder::model::T5Encoder`] takes `(b, t)` at build time and every row
//! is right-padded to the SAME `max_len`, so `run_batch` groups invocations by
//! `(variant, max_len)` and issues one forward per group - grouping by variant
//! alone would still build two different-width graphs for a mixed batch, and
//! grouping by `max_len` alone would build a graph per row if callers used
//! different context lengths. Order is preserved by carrying each invocation's
//! original index through the grouping, the same discipline
//! `resident_clip.rs` uses.

use capability::{ActionResult, Blob, Invocation, Manifest, Media, Outcome, Progress};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use serde_json::json;
use t5encoder::caps::Session;

/// The T5/umT5 encoders behind the scheduler (`BRAIN_T5ENCODER_DIR` - see
/// `t5encoder::caps`'s module docs for the layout).
pub struct T5encoderResident {
    dir: String,
}

impl T5encoderResident {
    /// `None` when the directory is unset or holds neither released layout -
    /// registering a model whose every call would fail is worse than not
    /// serving it.
    pub fn from_env() -> Option<T5encoderResident> {
        Self::new(std::env::var("BRAIN_T5ENCODER_DIR").ok().filter(|p| !p.is_empty())?)
    }

    /// Direct constructor (no env round-trip) - see
    /// `crate::resident_scrfd::ScrfdResident::new`'s rationale.
    pub fn new(dir: impl Into<String>) -> Option<T5encoderResident> {
        let dir = dir.into();
        let d = std::path::Path::new(&dir);
        let has_flux = d.join("text_encoder_2").exists() && d.join("tokenizer_2").join("tokenizer.json").exists();
        let has_wan = d.join("wan").join("models_t5_umt5-xxl-enc-bf16.pth").exists();
        if !has_flux && !has_wan {
            eprintln!("brain: t5encoder not served ({dir} holds neither the flux_xxl nor the wan_umt5 layout)");
            return None;
        }
        Some(T5encoderResident { dir })
    }
}

impl ResidentModel for T5encoderResident {
    fn manifest(&self) -> Manifest {
        t5encoder::caps::manifest()
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        // One session serves both variants: encoders are built lazily inside
        // it, so splitting the key would duplicate the device handle without
        // saving any weights.
        InstanceKey::new(t5encoder::caps::MODEL, "default")
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // T5-XXL v1.1 is 4.762 B params (~19.05 GB fp32) and umT5-XXL is
        // 5.681 B (~22.72 GB); a session may hold both plus per-batch graphs.
        // A flat bound, because which variant gets built depends on the calls
        // - see `t5encoder`'s crate docs for the exact per-variant sizes.
        MemCost::new(42u64 << 30, 0)
    }

    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        let gpu = crate::resident_llm::on_device(device, || gpu_core::Gpu::new(t5encoder::model::PIPELINES))?;
        Ok(Box::new(T5encoderInstance { session: Session::load(&self.dir, gpu)? }))
    }
}

/// A resident T5/umT5 stack on one shared device handle.
struct T5encoderInstance {
    session: Session,
}

fn hidden_outcome(v: &[f32], variant: &str, max_len: u32) -> Outcome {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    Outcome::new()
        .set("variant", json!(variant))
        .set("max_len", json!(max_len))
        .blob("hidden_states", Blob::new(Media::Bytes, bytes))
}

impl Instance for T5encoderInstance {
    fn run(&mut self, action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.session.run(action, inv)
    }

    /// One forward per `(variant, max_len)` group over the whole batch - see
    /// the module docs.
    fn run_batch(
        &mut self,
        action: &str,
        invs: &[Invocation],
        progress: &mut dyn FnMut(usize, Progress),
    ) -> Vec<ActionResult> {
        if action != "encode" {
            return invs
                .iter()
                .enumerate()
                .map(|(n, i)| self.run(action, i, &mut |p| progress(n, p)))
                .collect();
        }

        // Group by (variant, max_len), carrying the original index so results
        // come back in the caller's order regardless of how they grouped.
        let mut groups: std::collections::BTreeMap<(String, u32), (Vec<usize>, Vec<String>)> = Default::default();
        let mut out: Vec<Option<ActionResult>> = (0..invs.len()).map(|_| None).collect();
        for (i, inv) in invs.iter().enumerate() {
            let variant = inv.get_str("variant").unwrap_or_else(|| "flux_xxl".into());
            let max_len = inv.get_i64("max_len").map(|v| v as u32).unwrap_or(t5encoder::caps::DEFAULT_MAX_LEN);
            match inv.get_str("text") {
                Some(t) => {
                    let e = groups.entry((variant, max_len)).or_default();
                    e.0.push(i);
                    e.1.push(t);
                }
                None => out[i] = Some(Err("t5encoder: 'text' is required".to_string())),
            }
        }

        for ((variant, max_len), (idx, texts)) in groups {
            match self.session.encode_batch(&variant, &texts, max_len) {
                Ok(vs) => {
                    for (slot, v) in idx.iter().zip(vs) {
                        out[*slot] = Some(Ok(hidden_outcome(&v, &variant, max_len)));
                    }
                }
                // One bad group must not fail the rest of the batch.
                Err(e) => {
                    for slot in &idx {
                        out[*slot] = Some(Err(e.clone()));
                    }
                }
            }
        }

        out.into_iter().map(|r| r.unwrap_or_else(|| Err("t5encoder: batch slot unfilled".to_string()))).collect()
    }
}
