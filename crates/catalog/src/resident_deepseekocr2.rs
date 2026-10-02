// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DeepSeek-OCR-2 behind the residency scheduler.
//!
//! `activate` builds the whole composite ONCE - the mmproj import, the
//! decoder's streamed fp32 expansion, and the resident SAM tower - and the
//! [`Instance`] owns the resulting [`deepseekocr2::caps::Session`], so
//! dropping it frees every buffer. One action, `generate`; its schema and
//! all of its work come from `deepseekocr2::caps`, so this file holds no
//! second copy of the preprocessing, the prompt assembly, or the token
//! accounting.
//!
//! # Single device, unlike v1
//!
//! [`deepseek2ocr::caps::Session::load`] splits its two towers across wgpu
//! (vision) and the CPU backend (decoder) - a split that only exists there
//! because `crates/sam1`'s wgpu corruption at 1024x1024/3-or-more-blocks was
//! independently confirmed fixed at real-weight scale
//! (`crates/sam1/tests/wgpu_real_weight_parity.rs`) before that file moved
//! the vision tower off the CPU backend.
//!
//! [`deepseekocr2::caps::Session::load`] builds EVERY device (SAM, the new
//! 24-layer Qwen2 resampler, and the decoder) on the CPU backend, on
//! purpose: this crate's own real-weight test suite
//! (`crates/deepseekocr2/tests/common/real_vision.rs`'s header) pins CPU
//! specifically because stacking a SECOND 24-block-deep tower behind SAM on
//! one wgpu device has not been independently verified the way v1's single
//! SAM tower was - a real, disclosed gap, not an oversight. Moving the
//! vision half to wgpu here is a follow-up once that verification lands,
//! the same staged path v1 itself took.
//!
//! # Batching: the serial default, and why
//!
//! Same reasoning as `crate::resident_deepseekocr`'s header: each request's
//! image needs its own SAM pass (not batch-strided), and the decoder's own
//! batch axis is not wired for concurrent sequences.

use capability::{ActionResult, Invocation, Manifest, Progress};
use deepseekocr2::caps::{Session, DIR_VAR, MODEL};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

/// Peak process RSS of the whole composite (SAM + the 24-layer resampler +
/// the 2.9B-parameter decoder), all on the CPU backend, global view, a
/// several-token greedy decode. **Measured**, not derived from file sizes:
/// `cargo test --release -p brain-deepseekocr2 --test real_weight_generate
/// -- --ignored --nocapture` reports VmHWM 15.73 GiB for exactly this shape,
/// read off `/proc/self/status`. Rounded up to 16 GiB for the served
/// context.
pub const COMPOSITE_PEAK_BYTES: u64 = 16u64 << 30;

/// DeepSeek-OCR-2 behind the scheduler. `BRAIN_DEEPSEEKOCR2_DIR` names the
/// directory holding both shipped GGUFs.
pub struct DeepseekOcr2Resident {
    dir: String,
}

impl DeepseekOcr2Resident {
    /// `None` when the variable is unset or the directory does not hold both
    /// files - registering a model whose every call would fail is worse than
    /// not serving it.
    pub fn from_env() -> Option<DeepseekOcr2Resident> {
        Self::new(std::env::var(DIR_VAR).ok().filter(|p| !p.is_empty())?)
    }

    /// Direct constructor (no env round-trip) - see
    /// `crate::resident_scrfd::ScrfdResident::new`'s rationale.
    pub fn new(dir: impl Into<String>) -> Option<DeepseekOcr2Resident> {
        let dir = dir.into();
        match deepseekocr2::import::Files::locate(&dir) {
            Ok(_) => Some(DeepseekOcr2Resident { dir }),
            Err(e) => {
                eprintln!("brain: deepseek-ocr-2 not served ({e})");
                None
            }
        }
    }
}

impl ResidentModel for DeepseekOcr2Resident {
    fn manifest(&self) -> Manifest {
        deepseekocr2::caps::manifest_resident()
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        // One composite serves every request: the splice is sized at the
        // instruction-independent global-view run, so nothing in an
        // invocation can fork the graph.
        InstanceKey::new(MODEL, self.dir.clone())
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // RAM only (`vram = 0`): every device this build touches is the CPU
        // backend today - see this module's header. `MemCost::new` takes
        // `(vram, ram)`, in that order.
        MemCost::new(0, COMPOSITE_PEAK_BYTES)
    }

    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        // `Session::load` names its own devices (`Gpu::new_cpu` throughout),
        // so the assigned `Device` is not threaded through - this model is
        // single-shape, not placeable elsewhere yet.
        Ok(Box::new(DeepseekOcr2Instance { session: Session::load(&self.dir)? }))
    }
}

/// A resident DeepSeek-OCR-2: the built composite, its tokenizer, its
/// resident SAM tower, and the preprocessor's device handle.
struct DeepseekOcr2Instance {
    session: Session,
}

impl Instance for DeepseekOcr2Instance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        if action != "generate" {
            return Err(format!("deepseek-ocr-2: unknown action '{action}'"));
        }
        self.session.generate(inv, progress)
    }

    // `run_batch` is the serial default - see this module's header.
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An unconfigured checkpoint yields no resident at all, rather than one
    /// that fails every call.
    #[test]
    fn a_missing_checkpoint_is_not_registered() {
        assert!(DeepseekOcr2Resident::new("/definitely/not/a/deepseek-ocr2/dir").is_none());
    }

    /// The estimate is RAM-only and matches the measured peak, not a guess.
    #[test]
    fn the_estimate_is_ram_only_and_matches_the_measured_peak() {
        let r = DeepseekOcr2Resident { dir: "/tmp".into() };
        let cost = r.estimate(&r.instance_key("generate", &Invocation::new()));
        assert_eq!(cost.vram, 0, "every device this build touches is the CPU backend today");
        assert_eq!(cost.ram, COMPOSITE_PEAK_BYTES);
    }
}
