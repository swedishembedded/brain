// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DeepSeek-OCR behind the residency scheduler.
//!
//! `activate_multi` builds the whole composite ONCE - the mmproj import, the
//! decoder's streamed fp32 expansion, and the 273-row splice sized from the
//! prompt - and the [`Instance`] owns the resulting
//! [`deepseek2ocr::caps::Session`], so dropping it frees every buffer. One
//! action, `generate`; its schema and all of its work come from
//! `deepseek2ocr::caps`, so this file holds no second copy of the
//! preprocessing, the prompt assembly or the token accounting.
//!
//! # This model spans TWO devices, and now says so
//!
//! [`deepseek2ocr::caps::Session::load`] builds the vision encoder
//! (SAM + CLIP + glue) with `gpu_core::Gpu::new_wgpu` and the decoder with
//! `gpu_core::Gpu::new_cpu`. That split landed in `crates/deepseek2ocr` once
//! `crates/sam1`'s wgpu corruption at 1024x1024/3-or-more-blocks was fixed and
//! confirmed at real-weight scale (`crates/sam1/tests/
//! wgpu_real_weight_parity.rs`) - but this file was not updated with it, and
//! kept declaring a **RAM-only** `MemCost` (`vram == 0`) plus an `activate`
//! that refused any non-CPU assignment as "CPU-only". Both statements were
//! false from the moment the split landed, and the consequence was the exact
//! defect `residency::multi`'s own module doc names: the vision tower's real
//! device bytes were **invisible to the budget**, so on a host with a discrete
//! card the scheduler would happily place another model on top of memory this
//! one had already spoken for.
//!
//! [`MultiDeviceCost`] is the honest expression, and the reason this model is
//! registered through `Executor::register_multi` rather than the ordinary
//! single-device list: every device the instance touches is NAMED, with its own
//! real byte count, checked against its own real budget.
//!
//! * `(Device::Gpu(i), `[`VISION_DEVICE_BYTES`]`)` - the vision tower.
//! * `(Device::Cpu, `[`HOST_BYTES_SPLIT`]`)` - the decoder plus everything
//!   host-side.
//!
//! On a host whose GPU shares physical RAM with the CPU (the Intel Arc iGPU
//! every real-weight number quoted below was measured on), `build_executor`
//! has already declared `Device::Cpu` and that card into
//! ONE `memauth` pool, so naming both devices charges the shared pool once
//! rather than twice - which is precisely what that pool exists for. On a
//! discrete card the two figures are genuinely two different pools.
//!
//! Neither half touches `BRAIN_DEVICE`: a resident lives for the life of the
//! server process, and a process-global env write from inside one model's
//! activation would change the backend every *other* resident builds on
//! afterwards. Placement is a scoped registry selection
//! (`gpu_core::devices::with_gpu`, via [`crate::resident_llm::on_device`]) -
//! `Session::load`'s `Gpu::new_wgpu` resolves the ambient selection, so running
//! it inside that scope lands the vision tower on exactly the card
//! `estimate_multi` reserved. (The one-shot test glue in
//! `crates/deepseek2ocr/tests/common/real_vision.rs` does mutate `BRAIN_DEVICE`
//! - correctly, for a single-threaded test binary that owns the process.)
//!
//! # Batching: the serial default, and why
//!
//! Each request's image needs its own DeepEncoder pass (the SAM tower is a
//! single-image graph - `sam1`'s windowed spans are not batch-strided), and the
//! decoder's batch axis is not wired for concurrent sequences, so there is no
//! shared work between two concurrent requests to hoist. A real batched forward
//! here is a performance phase of its own, not a wrapper this file could write.

use capability::{ActionResult, Invocation, Manifest, Progress};
use deepseek2ocr::caps::{Session, DIR_VAR, MODEL};
use residency::multi::{MultiDeviceCost, MultiDeviceResidentModel};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

/// Device bytes the vision tower (SAM ViT-B at 1024² + the 16x compressor +
/// CLIP-L + the projector) holds while hot.
///
/// **Derived from one real measurement, not guessed and not measured twice.**
/// The encoder-only real-weight run (`crates/deepseek2ocr/tests/
/// real_weight_image.rs`) reports process VmHWM of 1.59 GiB after the mmproj
/// import, 3.44 GiB once the tower is built, and 7.17-7.22 GiB after a forward:
/// i.e. ~1.85 GiB of weights and ~3.7 GiB of activation buffers above the
/// import baseline. Those were host buffers (the run was all-CPU); on wgpu
/// they are the same buffers on the device. 6 GiB rounds that ~5.6 GiB up for
/// the served shape.
///
/// For scale, `crate::resident_sam2` budgets 3 GiB of activation slack for
/// `hiera_tiny` at the same 1024² input, on top of its weights - so this is the
/// same order of magnitude arrived at independently.
pub const VISION_DEVICE_BYTES: u64 = 6u64 << 30;

/// Peak footprint of the WHOLE composite with every stage on the CPU - the
/// figure this file used to report as a RAM-only cost.
///
/// MEASURED, not derived from the file sizes: the real-weight composite gate
/// (`crates/deepseek2ocr/tests/real_weight_generate.rs`) reports VmHWM
/// 21.32 GiB for exactly this build at exactly this shape, read off
/// `/proc/self/status`. Rounded up to 22 GiB for the served context (512 rows
/// rather than the test's ~260, i.e. one larger `[seq, 129280]` logit slab). A
/// file-size sum would say ~15 GB and be wrong by the whole activation working
/// set.
pub const COMPOSITE_PEAK_BYTES: u64 = 22u64 << 30;

/// Host bytes held once the vision half is on a card: the measured all-CPU peak
/// minus the part that moved.
///
/// The two constants therefore SUM to the one measurement rather than each
/// claiming it - reporting [`COMPOSITE_PEAK_BYTES`] of RAM *and*
/// [`VISION_DEVICE_BYTES`] of VRAM would over-reserve by the vision tower on
/// every host, and reporting only one of them is what this file did wrong
/// before. This is a decomposition of a single measurement, not two
/// independent ones; a direct measurement of the split build on a discrete card
/// would be better, and replacing both constants with one is open work.
pub const HOST_BYTES_SPLIT: u64 = COMPOSITE_PEAK_BYTES - VISION_DEVICE_BYTES;

/// DeepSeek-OCR behind the scheduler. `BRAIN_DEEPSEEK_OCR_DIR` names the
/// directory holding BOTH shipped GGUFs (`mmproj-DeepSeek-OCR-Q8_0.gguf` and
/// `DeepSeek-OCR-Q8_0.gguf`) - one variable for a multi-file checkpoint, the
/// same convention as `BRAIN_ARCFACE_DIR` and `BRAIN_CLIP_DIR`.
pub struct DeepseekOcrResident {
    dir: String,
    /// The canonical card the vision tower is placed on, or `None` when the
    /// caller budgeted no GPU that could hold it (see [`Self::pick_vision_gpu`]).
    vision_gpu: Option<u32>,
}

impl DeepseekOcrResident {
    /// `None` when the variable is unset or the directory does not hold both
    /// files - registering a model whose every call would fail is worse than
    /// not serving it.
    ///
    /// `gpus` is `build_executor`'s own budgeted `(index, TOTAL bytes)` list and
    /// `reserved` its per-card headroom, so the card this picks is chosen
    /// against genuinely usable capacity - the same figure the scheduler
    /// budgets against.
    pub fn from_env(gpus: &[(u32, u64)], reserved: u64) -> Option<DeepseekOcrResident> {
        let dir = std::env::var(DIR_VAR).ok().filter(|p| !p.is_empty())?;
        Self::new(dir, gpus, reserved)
    }

    /// Direct constructor (no env round-trip) - see
    /// `crate::resident_scrfd::ScrfdResident::new`'s rationale.
    pub fn new(dir: impl Into<String>, gpus: &[(u32, u64)], reserved: u64) -> Option<DeepseekOcrResident> {
        let dir = dir.into();
        match deepseek2ocr::import::Files::locate(&dir) {
            Ok(_) => Some(DeepseekOcrResident { dir, vision_gpu: Self::pick_vision_gpu(gpus, reserved) }),
            Err(e) => {
                eprintln!("brain: deepseek-ocr not served ({e})");
                None
            }
        }
    }

    /// The card the vision tower goes on: the budgeted GPU with the most usable
    /// capacity that can hold [`VISION_DEVICE_BYTES`] at all, or `None`.
    ///
    /// `residency::multi::pick_devices` deliberately does NOT renegotiate the
    /// device set a `MultiDeviceCost` names (see its doc: the set is the
    /// caller's placement decision), so this model has to choose, once, up
    /// front. Choosing the largest card rather than card 0 unconditionally is
    /// what keeps a machine with a small carve-out iGPU at index 0 and a big
    /// discrete card at index 1 from being permanently unplaceable. The tower is
    /// only ~6 GiB, so this is a fits-at-all decision, not load balancing;
    /// spreading it is not a thing a single ViT graph can do.
    fn pick_vision_gpu(gpus: &[(u32, u64)], reserved: u64) -> Option<u32> {
        gpus.iter()
            .map(|&(i, total)| (i, total.saturating_sub(reserved)))
            .filter(|&(_, usable)| usable >= VISION_DEVICE_BYTES)
            .max_by_key(|&(_, usable)| usable)
            .map(|(i, _)| i)
    }
}

impl ResidentModel for DeepseekOcrResident {
    fn manifest(&self) -> Manifest {
        // The stripped, weights-free spec: this resident's checkpoint
        // directory is already resolved at construction (`self.dir`), so a
        // served caller must never be told a `weights` param exists to set -
        // see `deepseek2ocr::caps::manifest_resident`'s doc. `Session::generate`
        // (what `DeepseekOcrInstance::run` actually calls) never reads
        // `weights` from `inv` either.
        deepseek2ocr::caps::manifest_resident()
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        // One composite serves every request: the splice is sized at the
        // instruction-independent (1, 273) image run, so nothing in an
        // invocation can fork the graph. Keying on anything else would
        // duplicate a ~22 GiB build.
        InstanceKey::new(MODEL, self.dir.clone())
    }

    /// Deliberately unusable: this model is registered via `register_multi` and
    /// claimed via `claim_multi`, so the single-device estimate is never
    /// consulted. Reporting a real figure here would invite the plain
    /// `register` path, whose single `budgets.alloc(device, cost.on(device))`
    /// can only ever charge ONE of the two devices this instance occupies -
    /// which is the accounting hole this file was fixed to close. Same
    /// convention as `crate::resident_omni::OmniResident`.
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        MemCost::new(0, 0)
    }

    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        Err(format!(
            "{MODEL}: single-device activate is not supported -- this model spans a GPU (vision) \
             and the CPU (decoder), so it must be claimed via ResidencyManager::claim_multi"
        ))
    }
}

impl MultiDeviceResidentModel for DeepseekOcrResident {
    fn estimate_multi(&self, _key: &InstanceKey) -> MultiDeviceCost {
        // Cheap and panic-free by construction (a little arithmetic over two
        // consts and an Option<u32>) -- this runs on the dispatcher thread on
        // every scheduling round, and a panic there kills serving for every
        // OTHER model too, not just this one.
        match self.vision_gpu {
            // With no card big enough for the tower, `Session::load`'s
            // `Gpu::new_wgpu` resolves to whatever wgpu offers -- a software
            // rasteriser on a GPU-less box -- whose buffers ARE host RAM. The
            // all-CPU peak is exactly the right figure for that case, and it is
            // the one that was actually measured.
            None => MultiDeviceCost::new(vec![(Device::Cpu, COMPOSITE_PEAK_BYTES)], 0),
            Some(i) => MultiDeviceCost::new(vec![(Device::Gpu(i), VISION_DEVICE_BYTES), (Device::Cpu, HOST_BYTES_SPLIT)], 0),
        }
    }

    fn activate_multi(&self, key: &InstanceKey, devices: &[Device]) -> Result<Box<dyn Instance>, String> {
        // `claim_multi` reserves against exactly the devices `estimate_multi`
        // named, so it hands back that same set. Insisting on it here (rather
        // than silently building on whatever arrives) is what makes the
        // reservation and the allocation describe the same bytes.
        let planned: Vec<Device> = self.estimate_multi(key).devices().collect();
        if devices.len() != planned.len() || !devices.iter().all(|d| planned.contains(d)) {
            return Err(format!("{MODEL}: activate_multi got devices {devices:?} but the plan placed {planned:?}"));
        }
        // Scoped registry selection, never env mutation: `Session::load` builds
        // the vision half with `Gpu::new_wgpu`, which resolves the ambient
        // selection, so this scope is what puts it on the reserved card. The
        // decoder's `Gpu::new_cpu` and the preprocessor's are unaffected by the
        // scope -- they name the CPU backend explicitly.
        let session = match self.vision_gpu {
            Some(i) => crate::resident_llm::on_device(Device::Gpu(i), || Session::load(&key.config))?,
            None => Session::load(&key.config),
        }?;
        Ok(Box::new(DeepseekOcrInstance { session }))
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

    // `run_batch` is the serial default: one encoder pass per image and a
    // decoder with no wired batch axis share no work between requests -- see
    // this module's header.
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1 << 30;

    /// Two 24 GB cards with a 2 GB reserve, the shape `build_executor` passes.
    fn two_cards() -> Vec<(u32, u64)> {
        vec![(0, 24 * GB), (1, 24 * GB)]
    }

    fn resident(gpus: &[(u32, u64)]) -> DeepseekOcrResident {
        DeepseekOcrResident { dir: "/tmp".into(), vision_gpu: DeepseekOcrResident::pick_vision_gpu(gpus, 2 * GB) }
    }

    fn key(r: &DeepseekOcrResident) -> InstanceKey {
        r.instance_key("generate", &Invocation::new())
    }

    /// An unconfigured checkpoint yields no resident at all, rather than one
    /// that fails every call.
    #[test]
    fn a_missing_checkpoint_is_not_registered() {
        assert!(DeepseekOcrResident::new("/definitely/not/a/deepseek/dir", &two_cards(), 2 * GB).is_none());
    }

    /// THE BUG THIS FILE WAS FIXED FOR: the vision tower runs on wgpu, so its
    /// device bytes must be named and budgeted. A cost that mentions no GPU is
    /// how another model gets placed on top of memory this one already holds.
    #[test]
    fn the_vision_tower_is_charged_to_a_real_card() {
        let r = resident(&two_cards());
        let cost = r.estimate_multi(&key(&r));
        let named: Vec<Device> = cost.devices().collect();
        assert!(named.contains(&Device::Gpu(0)) || named.contains(&Device::Gpu(1)), "no card named: {named:?}");
        assert!(named.contains(&Device::Cpu), "the CPU-side decoder must stay budgeted too: {named:?}");
        assert_eq!(cost.on(Device::Gpu(0)) + cost.on(Device::Gpu(1)), VISION_DEVICE_BYTES);
        assert_eq!(cost.on(Device::Cpu), HOST_BYTES_SPLIT);
    }

    /// The decomposition must not inflate the model: the two halves sum to the
    /// one figure that was actually measured, rather than each claiming it.
    #[test]
    fn the_two_halves_sum_to_the_measured_composite_peak() {
        let r = resident(&two_cards());
        let cost = r.estimate_multi(&key(&r));
        // `total_accelerator_bytes` sums every NAMED device, and `Device::Cpu`
        // is one of the two this instance names.
        assert_eq!(cost.total_accelerator_bytes(), COMPOSITE_PEAK_BYTES);
        assert_eq!(cost.ram(), 0, "the host figure is a named device, not the descriptive `ram` field claim_multi never budgets");
    }

    /// A card too small for the tower is not named at all - naming it would
    /// make the model permanently unplaceable, since `pick_devices` never
    /// substitutes a different device than the cost named.
    #[test]
    fn a_card_that_cannot_hold_the_tower_is_never_named() {
        // One 4 GB card, 2 GB reserved -> 2 GB usable, well under the tower.
        let r = resident(&[(0, 4 * GB)]);
        let cost = r.estimate_multi(&key(&r));
        assert_eq!(cost.devices().collect::<Vec<_>>(), vec![Device::Cpu]);
        assert_eq!(cost.on(Device::Cpu), COMPOSITE_PEAK_BYTES, "all-CPU falls back to the all-CPU measurement");
    }

    /// The biggest usable card wins, so a small iGPU at index 0 does not shadow
    /// a big discrete card at index 1.
    #[test]
    fn the_largest_usable_card_is_chosen_not_index_zero() {
        let r = resident(&[(0, 8 * GB), (1, 24 * GB)]);
        assert_eq!(r.vision_gpu, Some(1));
    }

    /// The single-device path is structurally refused, so this model cannot be
    /// registered through `Executor::register` by mistake and end up with one
    /// of its two devices unbudgeted.
    #[test]
    fn the_single_device_path_is_refused() {
        let r = resident(&two_cards());
        assert_eq!(r.estimate(&key(&r)), MemCost::new(0, 0));
        let e = r.activate(&key(&r), Device::Gpu(0)).err().unwrap_or_default();
        assert!(e.contains("claim_multi"), "{e}");
    }

    /// `activate_multi` refuses a device set that is not the one reserved -
    /// otherwise the reservation and the allocation describe different bytes.
    #[test]
    fn activate_multi_refuses_a_device_set_it_did_not_plan() {
        let r = resident(&two_cards());
        let e = r.activate_multi(&key(&r), &[Device::Cpu]).err().unwrap_or_default();
        assert!(e.contains("but the plan placed"), "{e}");
    }
}
