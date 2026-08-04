// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! CLIP text/image encoders behind the residency scheduler.
//!
//! `activate` builds a [`clip::caps::Session`] on the assigned device; the
//! towers themselves import lazily inside it on first use, keyed by
//! `(tower, batch)`. The [`Instance`] owns the session, so dropping it frees
//! every built tower. Both actions' schemas and all of their work come from
//! `clip::caps`, so this file holds no second copy of the tokenisation, the
//! pooling choice or the resize.
//!
//! # Batching: genuinely batched, unlike facenet
//!
//! `crates/cli/src/resident_facenet.rs` documents why the antelopev2 graphs
//! cannot batch (built for a single image, no N axis). CLIP is the opposite
//! case and gets no such excuse: [`clip::model::ClipText`] takes `b` at build
//! time, every row is the same fixed 77-token context, and
//! [`clip::caps::Session::embed_text_batch`] runs ONE forward over the whole
//! batch. `run_batch` therefore groups the invocations by tower and issues one
//! forward per group, rather than the serial default.
//!
//! Grouping by tower is required, not an optimisation: `clip_l` and
//! `openclip_bigg` are different graphs with different widths, so a mixed batch
//! is two forwards no matter what. Order is preserved by carrying each
//! invocation's original index through the grouping.
//!
//! `embed_image` stays serial — the EVA tower is built at `b = 1` and the
//! decode/resize per image dominates anyway.

use capability::{ActionResult, Blob, Invocation, Manifest, Media, Outcome, Progress};
use clip::caps::Session;
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use serde_json::json;

/// The CLIP encoders behind the scheduler (`BRAIN_CLIP_DIR` = the released
/// checkpoint root, in the SDXL layout: `text_encoder/`, `text_encoder_2/`,
/// `tokenizer/`, `tokenizer_2/`).
pub struct ClipResident {
    dir: String,
}

impl ClipResident {
    /// `None` when the directory is unset or holds neither released tokenizer —
    /// registering a model whose every call would fail is worse than not
    /// serving it.
    pub fn from_env() -> Option<ClipResident> {
        let dir = std::env::var("BRAIN_CLIP_DIR").ok().filter(|p| !p.is_empty())?;
        let d = std::path::Path::new(&dir);
        if !d.join("tokenizer").exists() && !d.join("tokenizer_2").exists() {
            eprintln!("brain: clip not served (BRAIN_CLIP_DIR={dir} holds neither tokenizer/ nor tokenizer_2/)");
            return None;
        }
        Some(ClipResident { dir })
    }
}

impl ResidentModel for ClipResident {
    fn manifest(&self) -> Manifest {
        clip::caps::manifest()
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        // One session serves both actions and every tower: the towers are built
        // lazily inside it, so splitting the key would duplicate the device
        // handle without saving any weights.
        InstanceKey::new(clip::caps::MODEL, "default")
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // CLIP-L is ~123 M params and OpenCLIP-bigG ~695 M; at fp32 that is
        // ~0.5 GB and ~2.8 GB, and a session may hold both plus one EVA-L/336
        // image tower (~0.4 GB) and a per-batch graph. A flat bound rather than
        // a file-size sum, because which towers get built depends on the calls.
        MemCost::new(5u64 << 30, 0)
    }

    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        // Build on the card the manager assigned (scoped registry selection —
        // never env mutation).
        let gpu = crate::resident_llm::on_device(device, || gpu_core::Gpu::new(clip::model::TEXT_PIPELINES))?;
        Ok(Box::new(ClipInstance { session: Session::load(&self.dir, gpu)? }))
    }
}

/// A resident CLIP stack: text towers (+ optionally the EVA image tower) on one
/// shared device handle.
struct ClipInstance {
    session: Session,
}

fn embedding_outcome(v: &[f32], tower: &str) -> Outcome {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    Outcome::new()
        .set("dim", json!(v.len()))
        .set("tower", json!(tower))
        .blob("embedding", Blob::new(Media::Bytes, bytes))
}

impl Instance for ClipInstance {
    fn run(&mut self, action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.session.run(action, inv)
    }

    /// One forward per tower over the whole batch — see the module docs.
    fn run_batch(
        &mut self,
        action: &str,
        invs: &[Invocation],
        progress: &mut dyn FnMut(Progress),
    ) -> Vec<ActionResult> {
        if action != "embed_text" {
            // `embed_image` has no batched build; fall back to the serial path
            // rather than pretending.
            return invs.iter().map(|i| self.run(action, i, progress)).collect();
        }

        // Group by tower, carrying the original index so the results come back
        // in the caller's order regardless of how they grouped.
        let mut groups: std::collections::BTreeMap<String, (Vec<usize>, Vec<String>)> = Default::default();
        let mut out: Vec<Option<ActionResult>> = (0..invs.len()).map(|_| None).collect();
        for (i, inv) in invs.iter().enumerate() {
            let tower = inv.get_str("tower").unwrap_or_else(|| "clip_l".into());
            match inv.get_str("text") {
                Some(t) => {
                    let e = groups.entry(tower).or_default();
                    e.0.push(i);
                    e.1.push(t);
                }
                None => out[i] = Some(Err("clip: 'text' is required".to_string())),
            }
        }

        for (tower, (idx, texts)) in groups {
            match self.session.embed_text_batch(&tower, &texts) {
                Ok(vs) => {
                    for (slot, v) in idx.iter().zip(vs) {
                        out[*slot] = Some(Ok(embedding_outcome(&v, &tower)));
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

        out.into_iter().map(|r| r.unwrap_or_else(|| Err("clip: batch slot unfilled".to_string()))).collect()
    }
}
