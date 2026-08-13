// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DeepSeek-OCR behind the residency scheduler.
//!
//! `activate` builds the whole composite ONCE - the mmproj import, the decoder's
//! streamed fp32 expansion, and the 273-row splice sized from the prompt - and
//! the [`Instance`] owns the resulting [`deepseekocr::caps::Session`], so
//! dropping it frees every buffer. One action, `generate`; its schema and all of
//! its work come from `deepseekocr::caps`, so this file holds no second copy of
//! the preprocessing, the prompt assembly or the token accounting.
//!
//! # This model is CPU-resident, and that is declared, not enforced by hand
//!
//! `crates/sam1`'s ViT tower USED TO corrupt its per-block buffers on wgpu at
//! 1024x1024 once the graph held three or more blocks - plausible garbage
//! rather than an error, hence the CPU pin. As of this repo's fourth pass on
//! that bug, the correctness question is CLOSED, with evidence, not "still
//! open": the root cause (Intel ANV not honoring an in-command-buffer
//! compute-compute barrier across a non-zero-offset/"sliced" storage buffer
//! binding - the exact bug class `crates/backend-vulkan` already root-caused
//! and fixed for this same box's Arc iGPU in commit `b6295e36`) is fixed in
//! `backend-wgpu::WgpuBackend::flush_serialized` (commit `04675800`), and that
//! fix was confirmed two independent ways: the real-weight, full-12-layer
//! parity test (`crates/sam1/tests/wgpu_real_weight_parity.rs`) ran clean
//! (worst cosine `1.0000000000`) five times in a row, and a checkpoint-free
//! reproduction of the ORIGINAL bisection's highest-failure-rate conditions
//! (`crates/sam1/tests/wgpu_block_count_corruption.rs`, run under induced
//! CPU contention pushing this 22-core box's load average to ~20-21, plus
//! four-way concurrent GPU contention, matching the original bisection's
//! reported conditions almost exactly) found zero failures across 32 trials.
//! See that test file's doc comment for the full record.
//!
//! **The CPU pin below is UNCHANGED regardless** - not because doubt remains
//! about `sam1`'s wgpu correctness, but because the code that actually
//! selects the device is NOT in this file: `crates/deepseekocr::caps::
//! Session::load` (`crates/deepseekocr/src/caps.rs`) hardcodes
//! `gpu_core::Gpu::new_cpu` for every stage - vision (SAM/CLIP/glue) AND the
//! decoder - via one `dev` closure, and calls it once per component
//! (`DeepEncoder::new` already calls `dev(sam1::model::PIPELINES)`,
//! `dev(CLIP_VISION_PIPELINES)`, and `dev(GLUE_PIPELINES)` separately from
//! `DeepseekOcr::build`'s `dev(deepseekv2::PIPELINES)` for the decoder), so
//! giving the vision tower wgpu while leaving the decoder on the CPU is a
//! small, well-scoped change to that one closure - but it requires editing
//! `crates/deepseekocr`, which is outside this pass's assigned scope
//! (reserved to avoid colliding with a concurrent sibling pass on Phase 8
//! performance in that same crate family). This file's own [`MemCost`] /
//! [`ResidentModel::activate`] gate below still declares CPU-only / `vram ==
//! 0` because that is what is actually true today - a scheduler edit here
//! alone, with `Session::load` still ignoring it, would be cosmetic, not a
//! real pin lift. Lifting the pin for real is a follow-up pass's job: change
//! `Session::load`'s `dev` closure in `crates/deepseekocr/src/caps.rs` to
//! build the vision stages on `gpu_core::Gpu::new_wgpu` and the decoder on
//! `gpu_core::Gpu::new_cpu`, then re-verify end to end.
//!
//! That is expressed as a **RAM-only [`MemCost`]** - `vram == 0` - which is the
//! scheduler's own vocabulary for "not a GPU model": `residency::place::
//! pick_device` skips the whole GPU class when `cost.vram == 0` and falls
//! through to the CPU/RAM pool. [`deepseekocr::caps::Session::load`] then builds
//! every stage with `gpu_core::Gpu::new_cpu`. Neither half touches
//! `BRAIN_DEVICE`: a resident lives for the life of the server process, and a
//! process-global env write from inside one model's activation would change the
//! backend every *other* resident builds on afterwards. (The one-shot test glue
//! in `crates/deepseekocr/tests/common/real_vision.rs` does mutate `BRAIN_DEVICE`
//! - correctly, for a single-threaded test binary that owns the process.)
//!
//! # Batching: the serial default, and why
//!
//! Each request's image needs its own DeepEncoder pass (the SAM tower is a
//! single-image graph - `sam1`'s windowed spans are not batch-strided), and the
//! decoder is `O(T²)` recompute with no KV cache, so there is no shared work
//! between two concurrent requests to hoist. A real batched forward here is a
//! performance phase of its own, not a wrapper this file could write.

use capability::{ActionResult, Invocation, Manifest, Progress};
use deepseekocr::caps::{Session, DIR_VAR, MODEL};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

/// DeepSeek-OCR behind the scheduler. `BRAIN_DEEPSEEK_OCR_DIR` names the
/// directory holding BOTH shipped GGUFs (`mmproj-DeepSeek-OCR-Q8_0.gguf` and
/// `DeepSeek-OCR-Q8_0.gguf`) - one variable for a multi-file checkpoint, the
/// same convention as `BRAIN_FACENET_DIR` and `BRAIN_CLIP_DIR`.
pub struct DeepseekOcrResident {
    dir: String,
}

impl DeepseekOcrResident {
    /// `None` when the variable is unset or the directory does not hold both
    /// files - registering a model whose every call would fail is worse than
    /// not serving it.
    pub fn from_env() -> Option<DeepseekOcrResident> {
        let dir = std::env::var(DIR_VAR).ok().filter(|p| !p.is_empty())?;
        Self::new(dir)
    }

    /// Direct constructor (no env round-trip) - see
    /// `crate::resident_facenet::FacenetResident::new`'s rationale.
    pub fn new(dir: impl Into<String>) -> Option<DeepseekOcrResident> {
        let dir = dir.into();
        match deepseekocr::import::Files::locate(&dir) {
            Ok(_) => Some(DeepseekOcrResident { dir }),
            Err(e) => {
                eprintln!("brain: deepseek-ocr not served ({e})");
                None
            }
        }
    }
}

impl ResidentModel for DeepseekOcrResident {
    fn manifest(&self) -> Manifest {
        deepseekocr::caps::manifest()
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        // One composite serves every request: the splice is sized at the
        // instruction-independent (1, 273) image run, so nothing in an
        // invocation can fork the graph. Keying on anything else would
        // duplicate a ~22 GiB build.
        InstanceKey::new(MODEL, self.dir.clone())
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // RAM, not VRAM -- see this module's header: `vram == 0` is how a model
        // tells `place::pick_device` it is not GPU-placeable.
        //
        // The figure is MEASURED, not derived from the file sizes: the
        // real-weight composite gate (`crates/deepseekocr/tests/
        // real_weight_generate.rs`) reports VmHWM 21.32 GiB for exactly this
        // build at exactly this shape, read off /proc/self/status. Rounded up
        // to 22 GiB for the served context (512 rows rather than the test's
        // ~260, i.e. one larger `[seq, 129280]` logit slab). A file-size sum
        // would say ~15 GB and be wrong by the whole activation working set.
        MemCost::new(0, 22u64 << 30)
    }

    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        if device != Device::Cpu {
            // Unreachable while `estimate` reports vram == 0, but a silent
            // wrong-backend build is precisely the failure this model already
            // paid for once. Note this is now a SCHEDULING contract, not a
            // live correctness doubt - see this module's header comment for
            // why `Session::load` still hardcodes the CPU backend regardless
            // of what device this function is handed.
            return Err(format!(
                "deepseek-ocr: assigned {device:?}, but this model is CPU-only \
                 (Session::load hardcodes gpu_core::Gpu::new_cpu; see this file's header comment) -- its MemCost declares vram == 0"
            ));
        }
        Ok(Box::new(DeepseekOcrInstance { session: Session::load(&key.config)? }))
    }
}

/// A resident DeepSeek-OCR: the built composite, its tokenizer, and the
/// preprocessor's device handle.
struct DeepseekOcrInstance {
    session: Session,
}

impl Instance for DeepseekOcrInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        if action != "generate" {
            return Err(format!("deepseek-ocr: unknown action '{action}'"));
        }
        self.session.generate(inv, progress)
    }

    // `run_batch` is the serial default: one encoder pass per image and an
    // O(T²)-recompute decoder share no work between requests -- see this
    // module's header.
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An unconfigured checkpoint yields no resident at all, rather than one
    /// that fails every call.
    #[test]
    fn a_missing_checkpoint_is_not_registered() {
        assert!(DeepseekOcrResident::new("/definitely/not/a/deepseek/dir").is_none());
    }

    /// The cost must be RAM-only: a non-zero `vram` would make
    /// `place::pick_device` consider a GPU, which this model is not correct on.
    #[test]
    fn the_cost_is_ram_only_so_the_placer_never_picks_a_gpu() {
        let r = DeepseekOcrResident { dir: "/tmp".into() };
        let c = r.estimate(&r.instance_key("generate", &Invocation::new()));
        assert_eq!(c.vram, 0, "a GPU placement would silently corrupt the SAM tower");
        assert_eq!(c.npu, 0);
        assert!(c.ram > 16u64 << 30, "the measured build peak is ~21.3 GiB");
    }

    /// A GPU assignment is refused loudly rather than built wrong.
    #[test]
    fn a_gpu_assignment_is_refused() {
        let r = DeepseekOcrResident { dir: "/tmp".into() };
        let e = r.activate(&r.instance_key("generate", &Invocation::new()), Device::Gpu(0)).err().unwrap_or_default();
        assert!(e.contains("CPU-only"), "{e}");
    }
}
