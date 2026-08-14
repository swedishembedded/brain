// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SAM 2.1 promptable segmentation behind the residency scheduler.
//!
//! `activate` imports the release checkpoint ONCE and uploads it to the assigned
//! device; the [`Instance`] owns the built [`sam2::caps::Session`], so dropping
//! it frees the weights and the cached image encoding. One action, `segment` —
//! the schema and the work both come from `sam2::caps`, so this file contains no
//! second copy of the preprocessing, the prompt parsing or the mask emission.
//!
//! # Batching: group by IMAGE, not by request
//!
//! SAM 2 is encode-once / prompt-many. The Hiera trunk is ~99 % of the cost and
//! depends only on the pixels; the two-way mask decoder is small and depends
//! only on the prompt. The trunk graph is also built for **one** image
//! (`Sam2::encode` asserts a single `[1, 3, S, S]` map and every window/`q_pool`
//! extent is derived from that), so there is no N-image axis to batch along.
//!
//! The architecture's real batching axis is therefore *prompts per image*:
//! [`Sam2Instance::run_batch`] groups the batch by image so N concurrent prompts
//! on one frame cost ONE trunk pass and N decoder passes, instead of N trunk
//! passes. Two requests on genuinely different images still encode twice —
//! correctly, and the comment above says why.

use capability::{ActionResult, Invocation, Manifest, Progress};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use sam2::caps::Session;

/// SAM 2.1 behind the scheduler. `BRAIN_SAM2_WEIGHTS` names the release
/// checkpoint (`sam2.1_hiera_{tiny,large}.pt`), `BRAIN_SAM2_VARIANT` its variant
/// (default `tiny`); a request may override the variant per call, which keys a
/// separate instance.
pub struct Sam2Resident {
    path: String,
    variant: String,
}

impl Sam2Resident {
    /// `None` when the checkpoint is unset or absent — registering a model whose
    /// every call would fail is worse than not serving it.
    pub fn from_env() -> Option<Sam2Resident> {
        let path = std::env::var("BRAIN_SAM2_WEIGHTS").ok().filter(|p| !p.is_empty())?;
        let variant = std::env::var("BRAIN_SAM2_VARIANT").ok().filter(|v| !v.is_empty()).unwrap_or_else(|| "tiny".into());
        Self::new(path, variant)
    }

    /// Direct constructor (no env round-trip) — see
    /// `crate::resident_scrfd::ScrfdResident::new`'s rationale.
    pub fn new(path: impl Into<String>, variant: impl Into<String>) -> Option<Sam2Resident> {
        let (path, variant) = (path.into(), variant.into());
        if !std::path::Path::new(&path).exists() {
            eprintln!("brain: sam2 not served ({path} does not exist)");
            return None;
        }
        Some(Sam2Resident { path, variant })
    }
}

impl ResidentModel for Sam2Resident {
    fn manifest(&self) -> Manifest {
        sam2::caps::manifest()
    }

    fn instance_key(&self, _action: &str, inv: &Invocation) -> InstanceKey {
        // The variant fixes the whole graph, so it is the config fingerprint:
        // two jobs with the same variant share one hot instance (and batch).
        InstanceKey::new(sam2::caps::MODEL, inv.get_str("variant").unwrap_or_else(|| self.variant.clone()))
    }

    fn estimate(&self, key: &InstanceKey) -> MemCost {
        // The imported tensors go into a `ParamStore` on the device, so the Hot
        // footprint is VRAM ~= the fp32 checkpoint plus the encoder activations.
        // hiera_tiny is ~156 MB of weights at 1024²; hiera_large ~900 MB. The
        // activation slack is the dominant term at this resolution (48 blocks of
        // SSA taps), hence the generous constant rather than a bare file size.
        let file = std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0);
        let activations: u64 = if key.config == "large" { 6u64 << 30 } else { 3u64 << 30 };
        MemCost::new(file.saturating_mul(12) / 10 + activations, 0)
    }

    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        let cfg = sam2::caps::variant_config(&key.config)?;
        // Build the engine on the card the manager assigned (scoped registry
        // selection — never env mutation), then import once.
        let gpu = crate::resident_llm::on_device(device, || gpu_core::Gpu::new(sam2::PIPELINES))?;
        let model = sam2::caps::load(&self.path, cfg, gpu)?;
        Ok(Box::new(Sam2Instance { session: Session::new(model) }))
    }
}

/// A resident SAM 2: the built model plus its one-entry image-encoder cache.
struct Sam2Instance {
    session: Session,
}

impl Instance for Sam2Instance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        // The batch callback carries the batch index; a single invocation is
        // index 0, so the one-shot path forwards its progress unchanged.
        self.run_batch(action, std::slice::from_ref(inv), &mut |_, p| progress(p))
            .pop()
            .expect("one result per invocation")
    }

    /// Group the batch by image and run ONE trunk pass per distinct image.
    ///
    /// Results are returned in the caller's order (the executor zips them back
    /// to jobs positionally), so the grouping is over indices, not over a
    /// reordered slice.
    fn run_batch(
        &mut self,
        action: &str,
        invs: &[Invocation],
        _progress: &mut dyn FnMut(usize, Progress),
    ) -> Vec<ActionResult> {
        if action != "segment" {
            return invs.iter().map(|_| Err(format!("sam2: unknown action '{action}'"))).collect();
        }
        let mut out: Vec<Option<ActionResult>> = (0..invs.len()).map(|_| None).collect();
        for group in group_by_image(invs) {
            // Consecutive prompts on the same image: the first misses the
            // session's encoder cache and pays the trunk, the rest hit it and
            // run the mask decoder alone.
            for i in group {
                out[i] = Some(self.session.segment(&invs[i]));
            }
        }
        out.into_iter().map(|r| r.expect("every index filled")).collect()
    }
}

/// Partition invocation indices into runs that share an image, in first-seen
/// order. Every index appears exactly once, so the caller can scatter results
/// back positionally — the executor zips them to jobs by position.
fn group_by_image(invs: &[Invocation]) -> Vec<Vec<usize>> {
    let mut groups: Vec<(u64, Vec<usize>)> = Vec::new();
    for (i, inv) in invs.iter().enumerate() {
        let key = Session::image_key(inv);
        match groups.iter_mut().find(|(k, _)| *k == key) {
            Some((_, idx)) => idx.push(i),
            None => groups.push((key, vec![i])),
        }
    }
    groups.into_iter().map(|(_, idx)| idx).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::{Blob, Media};
    use serde_json::json;

    fn inv(pixels: &[u8], points: &str) -> Invocation {
        let bytes: Vec<u8> = pixels.iter().flat_map(|&p| (p as f32 / 255.0).to_le_bytes()).collect();
        let n = pixels.len() as u64;
        Invocation::new()
            .set("points", json!(points))
            .blob("image", Blob::new(Media::Image, bytes).with_meta(json!({"w": n / 3, "h": 1, "c": 3})))
    }

    /// Interleaved prompts on two images must collapse to TWO groups (two trunk
    /// passes for four prompts), and every caller index must survive exactly
    /// once — a dropped or duplicated index would silently mis-answer a job.
    #[test]
    fn prompts_on_one_image_share_a_group_and_every_index_survives() {
        let a = [1u8, 2, 3];
        let b = [9u8, 8, 7];
        let invs = [inv(&a, "1,1"), inv(&b, "2,2"), inv(&a, "3,3"), inv(&b, "4,4"), inv(&a, "5,5")];
        let groups = group_by_image(&invs);
        assert_eq!(groups.len(), 2, "two distinct images -> two encoder passes, got {groups:?}");
        assert_eq!(groups[0], vec![0, 2, 4], "first-seen image keeps its caller order");
        assert_eq!(groups[1], vec![1, 3]);
        let mut all: Vec<usize> = groups.into_iter().flatten().collect();
        all.sort_unstable();
        assert_eq!(all, (0..invs.len()).collect::<Vec<_>>());
    }

    /// Different prompts on the same pixels must NOT split the group: the key is
    /// the image, not the invocation.
    #[test]
    fn the_key_is_the_image_not_the_prompt() {
        let a = [4u8, 5, 6];
        assert_eq!(group_by_image(&[inv(&a, "1,1"), inv(&a, "9,9")]), vec![vec![0, 1]]);
        // ...and one differing byte is a different image.
        assert_eq!(group_by_image(&[inv(&[4, 5, 6], "1,1"), inv(&[4, 5, 7], "1,1")]).len(), 2);
    }

    /// An empty batch is legal (the executor may drain to nothing) and must not
    /// panic or invent a group.
    #[test]
    fn an_empty_batch_produces_no_groups() {
        assert!(group_by_image(&[]).is_empty());
    }
}
