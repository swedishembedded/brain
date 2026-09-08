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
//! # This model spans TWO devices, and names both
//!
//! [`deepseek2ocr::caps::Session::load_with`] builds the vision encoder
//! (SAM + CLIP + glue) with `gpu_core::Gpu::new_wgpu`, and the decoder on
//! whichever device the [`deepseek2ocr::caps::DecoderDevice`] it is handed
//! names - a second card, the vision card, or the CPU Cranelift JIT.
//!
//! [`MultiDeviceCost`] is the honest expression, and the reason this model is
//! registered through `Executor::register_multi` rather than the ordinary
//! single-device list: every device the instance touches is NAMED, with its own
//! real byte count, checked against its own real budget. Which devices those
//! are depends on the placement, and all three shapes are budgeted:
//!
//! * decoder on a second card - `(Gpu(v), `[`VISION_DEVICE_BYTES`]`)`,
//!   `(Gpu(d), `[`DECODER_DEVICE_BYTES`]`)`, `(Cpu, `[`HOST_BYTES_GPU_DECODER`]`)`.
//! * decoder on the vision card - one GPU entry holding both, plus the same
//!   host figure.
//! * decoder on the CPU - `(Gpu(v), `[`VISION_DEVICE_BYTES`]`)` and
//!   `(Cpu, `[`HOST_BYTES_CPU_DECODER`]`)`.
//!
//! Every one of those constants is a direct measurement of the served build
//! (`nvidia-smi memory.used` per card and `/proc/self/status` VmHWM, sampled
//! for the life of a real page through this exact loader), not a decomposition
//! of one number across devices.
//!
//! On a host whose GPU shares physical RAM with the CPU, `build_executor` has
//! already declared `Device::Cpu` and that card into ONE `memauth` pool, so
//! naming both devices charges the shared pool once rather than twice - which
//! is precisely what that pool exists for. On discrete cards the figures are
//! genuinely separate pools.
//!
//! Nothing here touches `BRAIN_DEVICE`: a resident lives for the life of the
//! server process, and a process-global env write from inside one model's
//! activation would change the backend every *other* resident builds on
//! afterwards. Placement is a scoped registry selection
//! (`gpu_core::devices::with_gpu`, via [`crate::resident_llm::on_device`]) -
//! `Session::load_with`'s `Gpu::new_wgpu` resolves the ambient selection, so
//! running it inside that scope lands the vision tower on exactly the card
//! `estimate_multi` reserved, and the decoder's own card is entered by a
//! nested `with_gpu` inside that call. The placement is computed ONCE, here,
//! and passed down, so the reservation and the allocation cannot name
//! different cards. (The one-shot test glue in
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
use deepseek2ocr::caps::{DecoderDevice, Session, MODEL};
use residency::multi::{MultiDeviceCost, MultiDeviceResidentModel};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

pub use deepseek2ocr::caps::{DECODER_DEVICE_BYTES, HOST_BYTES_CPU_DECODER, HOST_BYTES_GPU_DECODER, VISION_DEVICE_BYTES};

/// Peak footprint of the WHOLE composite with the decoder on the CPU backend:
/// [`HOST_BYTES_CPU_DECODER`] of host RAM plus [`VISION_DEVICE_BYTES`] on the
/// card. Kept as one name because that is what a GPU-less host pays entirely
/// out of RAM (there `Gpu::new_wgpu` resolves to a software rasteriser whose
/// buffers ARE host RAM, so the tower's bytes are host bytes too).
pub const COMPOSITE_PEAK_BYTES: u64 = HOST_BYTES_CPU_DECODER + VISION_DEVICE_BYTES;

/// DeepSeek-OCR behind the scheduler. `dir` names the directory holding BOTH
/// shipped GGUFs (`mmproj-DeepSeek-OCR-Q8_0.gguf` and `DeepSeek-OCR-Q8_0.gguf`),
/// resolved through `deepseek2ocr::spec::Deepseek2ocrSpec`'s `dir` role (see
/// [`Self::from_assembly`]).
pub struct DeepseekOcrResident {
    dir: String,
    /// The canonical card the vision tower is placed on, or `None` when the
    /// caller budgeted no GPU that could hold it (see [`Self::pick_vision_gpu`]).
    vision_gpu: Option<u32>,
    /// Where the decoder half goes - decided ONCE, at construction, and both
    /// budgeted by [`Self::estimate_multi`] and handed to
    /// `Session::load_with` by [`Self::activate_multi`], so the reservation
    /// and the allocation cannot name different devices.
    decoder: DecoderDevice,
}

impl DeepseekOcrResident {
    /// The [`catalog::MultiCtor`] shape: `dir` comes from an already-resolved
    /// [`capability::Assembly`] (`deepseek2ocr::spec::Deepseek2ocrSpec`'s
    /// `dir` role - see that module's doc) instead of `BRAIN_DEEPSEEK_OCR_DIR`.
    pub fn from_assembly(assembly: &capability::Assembly, gpus: &[(u32, u64)], reserved: u64) -> Option<DeepseekOcrResident> {
        let dir = match deepseek2ocr::spec::dir_from_assembly(assembly) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("brain: deepseek-ocr not served ({e})");
                return None;
            }
        };
        Self::new(dir, gpus, reserved)
    }

    /// Direct constructor (no env round-trip) - see
    /// `crate::resident_scrfd::ScrfdResident::new`'s rationale.
    pub fn new(dir: impl Into<String>, gpus: &[(u32, u64)], reserved: u64) -> Option<DeepseekOcrResident> {
        let dir = dir.into();
        match deepseek2ocr::import::Files::locate(&dir) {
            Ok(_) => {
                let vision_gpu = Self::pick_vision_gpu(gpus, reserved);
                let decoder = deepseek2ocr::caps::decoder_device(vision_gpu);
                Some(DeepseekOcrResident { dir, vision_gpu, decoder })
            }
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
        // duplicate a ~21 GiB build.
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
        // Cheap and panic-free by construction (a little arithmetic over four
        // consts, an Option<u32> and an enum decided at construction) -- this
        // runs on the dispatcher thread on every scheduling round, and a panic
        // there kills serving for every OTHER model too, not just this one.
        let Some(v) = self.vision_gpu else {
            // With no card big enough for the tower, `Gpu::new_wgpu` resolves
            // to whatever wgpu offers -- a software rasteriser on a GPU-less
            // box -- whose buffers ARE host RAM. Everything this model holds
            // is then host bytes, which is exactly [`COMPOSITE_PEAK_BYTES`].
            return MultiDeviceCost::new(vec![(Device::Cpu, COMPOSITE_PEAK_BYTES)], 0);
        };
        let cost = match self.decoder {
            DecoderDevice::Cpu => vec![(Device::Gpu(v), VISION_DEVICE_BYTES), (Device::Cpu, HOST_BYTES_CPU_DECODER)],
            DecoderDevice::SameCard => {
                vec![(Device::Gpu(v), VISION_DEVICE_BYTES + DECODER_DEVICE_BYTES), (Device::Cpu, HOST_BYTES_GPU_DECODER)]
            }
            // A `Card(i)` that happens to name the vision card is the same
            // pool as `SameCard`, and must be charged once rather than as two
            // entries the budget would see as two devices.
            DecoderDevice::Card(i) if i == v => {
                vec![(Device::Gpu(v), VISION_DEVICE_BYTES + DECODER_DEVICE_BYTES), (Device::Cpu, HOST_BYTES_GPU_DECODER)]
            }
            DecoderDevice::Card(i) => vec![
                (Device::Gpu(v), VISION_DEVICE_BYTES),
                (Device::Gpu(i), DECODER_DEVICE_BYTES),
                (Device::Cpu, HOST_BYTES_GPU_DECODER),
            ],
        };
        MultiDeviceCost::new(cost, 0)
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
        // Scoped registry selection, never env mutation: `Session::load_with`
        // builds the vision half with `Gpu::new_wgpu`, which resolves the
        // ambient selection, so this scope is what puts it on the reserved
        // card. A decoder on its OWN card enters a nested `with_gpu` inside
        // that call, and a decoder on the CPU names that backend explicitly;
        // neither is affected by this scope.
        let placement = self.decoder;
        let session = match self.vision_gpu {
            Some(i) => crate::resident_llm::on_device(Device::Gpu(i), || Session::load_with(&key.config, placement))?,
            None => Session::load_with(&key.config, placement),
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

    /// A resident at an EXPLICIT decoder placement. The real
    /// `deepseek2ocr::caps::decoder_device` reads this box's own device
    /// registry, so a test that let it choose would assert about whatever
    /// hardware it last ran on; naming the placement is what makes each case
    /// below a statement about the accounting rather than about a machine.
    fn resident_with(gpus: &[(u32, u64)], decoder: DecoderDevice) -> DeepseekOcrResident {
        DeepseekOcrResident { dir: "/tmp".into(), vision_gpu: DeepseekOcrResident::pick_vision_gpu(gpus, 2 * GB), decoder }
    }

    fn resident(gpus: &[(u32, u64)]) -> DeepseekOcrResident {
        resident_with(gpus, DecoderDevice::Cpu)
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

    /// THE BUG THIS FILE WAS FIXED FOR: every device this instance really
    /// holds bytes on must be named and budgeted. A cost that omits one is how
    /// another model gets placed on top of memory this one already holds -
    /// and now that the decoder can land on a SECOND card, there are three
    /// placements to get right, not one.
    #[test]
    fn every_placement_names_every_device_it_occupies() {
        // Decoder on the CPU: the tower's card and the host.
        let r = resident_with(&two_cards(), DecoderDevice::Cpu);
        let cost = r.estimate_multi(&key(&r));
        assert_eq!(cost.devices().collect::<Vec<_>>(), vec![Device::Gpu(1), Device::Cpu]);
        assert_eq!(cost.on(Device::Gpu(1)), VISION_DEVICE_BYTES);
        assert_eq!(cost.on(Device::Cpu), HOST_BYTES_CPU_DECODER);

        // Decoder on its own card: THREE devices, each with its own figure.
        let r = resident_with(&two_cards(), DecoderDevice::Card(0));
        let cost = r.estimate_multi(&key(&r));
        assert_eq!(cost.on(Device::Gpu(1)), VISION_DEVICE_BYTES, "the tower stays on the card pick_vision_gpu chose");
        assert_eq!(cost.on(Device::Gpu(0)), DECODER_DEVICE_BYTES);
        assert_eq!(cost.on(Device::Cpu), HOST_BYTES_GPU_DECODER, "the host keeps only the working set once the weights are on a card");

        // Decoder on the tower's own card: ONE card entry holding both, not
        // two entries the budget would read as two separate pools.
        for placement in [DecoderDevice::SameCard, DecoderDevice::Card(1)] {
            let r = resident_with(&two_cards(), placement);
            let cost = r.estimate_multi(&key(&r));
            assert_eq!(cost.devices().collect::<Vec<_>>(), vec![Device::Gpu(1), Device::Cpu], "{placement:?}");
            assert_eq!(cost.on(Device::Gpu(1)), VISION_DEVICE_BYTES + DECODER_DEVICE_BYTES, "{placement:?}");
            assert_eq!(cost.on(Device::Cpu), HOST_BYTES_GPU_DECODER, "{placement:?}");
        }
    }

    /// Moving the decoder onto a card must MOVE its ~11.4 GiB of fp32
    /// weights off the host, not merely add a card entry beside an unchanged
    /// host claim - which is exactly the accounting hole this file was fixed
    /// for once already, in the other direction.
    ///
    /// The two placements do not claim the same TOTAL, and that is a real
    /// difference rather than a rounding artefact: the decoder's KV cache,
    /// per-expert MoE scratch and attention slabs (~1.8 GiB at the served
    /// shape) are fully reserved when they live on a card, while on the host
    /// VmHWM only ever counts the pages a run actually touches. Asserting
    /// equality here would be asserting something the measurements say is
    /// false; what IS asserted is the direction and the magnitude.
    #[test]
    fn putting_the_decoder_on_a_card_takes_its_weights_off_the_host() {
        let cpu = resident_with(&two_cards(), DecoderDevice::Cpu);
        let gpu = resident_with(&two_cards(), DecoderDevice::Card(0));
        let (a, b) = (cpu.estimate_multi(&key(&cpu)), gpu.estimate_multi(&key(&gpu)));
        assert_eq!(a.total_accelerator_bytes(), COMPOSITE_PEAK_BYTES);
        let freed = a.on(Device::Cpu) - b.on(Device::Cpu);
        assert!(
            freed >= 10 * GB,
            "moving the decoder to a card freed only {freed} host bytes - the fp32 weights alone are ~11.4 GiB, so they did not move"
        );
        assert!(b.on(Device::Gpu(0)) >= 10 * GB, "and those bytes must be charged to the card that now holds them");
        for c in [&a, &b] {
            assert_eq!(c.ram(), 0, "the host figure is a named device, not the descriptive `ram` field claim_multi never budgets");
        }
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
