// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Resident-model adapter putting Qwen3-Omni's Thinker text generation behind
//! the residency [`Executor`], mirroring [`crate::resident_tts::TtsResident`].
//!
//! This is the path with the full chat/multimodal surface (`generate` with
//! `messages` + audio/image/video, plus `speak`), served straight from the
//! published checkpoint directory the model store resolves, with no import
//! step.
//!
//! **It is multi-device, and it budgets honestly.** It used to report
//! `MemCost::new(0, checkpoint_bytes)` - zero VRAM - and then quietly build a
//! GPU of its own inside `activate`, so the scheduler placed it on the CPU
//! lane while it filled a card. Nothing budgeted the bytes it actually spent,
//! and at the real 30B shape it walked a 24 GB card to `wgpu error: Out of
//! Memory` inside one request. Now [`OmniResident`] computes a real placement
//! from the checkpoint's own per-tensor byte costs
//! (`qwen3omnimoe::thinker_plan`/`model::shard`), reports it through
//! [`MultiDeviceResidentModel::estimate_multi`], and activates on exactly the
//! devices that reservation named - the same discipline
//! [`qwen3omnimoe::int8_thinker_resident::Int8ThinkerResident`] follows, now shared
//! rather than being a property of the int8 path specifically.
//!
//! Weight PRECISION is a separate axis from placement, and stays as the
//! checkpoint stores it: this path serves whatever dtype is on disk (bf16
//! here), decoded to f32 on the card. That is why a 30B checkpoint still does
//! not fit two 24 GB cards and part of it streams per token - slow, but
//! bounded and correct. The int8 Thinker ([`int8_thinker_multi_from_store`])
//! is what changes the precision, and it uses the identical placement
//! mechanism.

use capability::{ActionResult, Invocation, Manifest, Progress};
use checkpoint::weightio::WeightReader;
use qwen3omnimoe::caps::OmniProvider;
use residency::multi::{MultiDeviceCost, MultiDeviceResidentModel};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use std::sync::{Arc, OnceLock};

/// The placement this resident committed to: which cards, with what usable
/// capacity, and how many bytes each will hold. Computed once from the
/// checkpoint header (no GPU) and reused by both `estimate_multi` and
/// `activate_multi`, so the reservation and the load can never describe
/// different cards.
#[derive(Clone, Debug, Default)]
struct Plan {
    /// `(canonical GPU index, usable bytes)` for the devices the plan uses -
    /// exactly what is handed to `OmniProvider::load_on`.
    devices: Vec<(u32, u64)>,
    /// Resident device bytes per entry of `devices`.
    bytes: Vec<u64>,
    /// Host bytes the instance holds regardless of placement.
    host_ram: u64,
}

/// Qwen3-Omni behind the scheduler, placed across as many GPUs as its real
/// per-layer bytes need. Dispatches both declared actions (`generate`,
/// `speak`) through `qwen3omnimoe::caps::run_action`.
///
/// Reachable ONLY via `Executor::register_multi` - see [`Self::activate`],
/// which refuses the single-device path rather than silently building a GPU
/// the scheduler did not reserve.
pub struct OmniResident {
    hf_dir: String,
    /// Candidate devices as `(index, USABLE bytes)`: the caller's budgeted
    /// cards minus its per-card reserve. Capacity travels with identity
    /// because the split must respect it (a 24 GB and an 8 GB card cannot
    /// take the same number of layers).
    devices: Vec<(u32, u64)>,
    plan: OnceLock<Plan>,
}

impl OmniResident {
    /// Over the published directory the model store resolved
    /// (`qwen3omnimoe::spec::SPEC`'s `weights`); `None` (not served) when it
    /// resolved none. `gpus` is the caller's budgeted `(index, TOTAL bytes)`
    /// list and `reserved` its per-card headroom, so what is carried on is
    /// genuinely usable capacity - the same figure the scheduler budgets
    /// against.
    pub fn from_assembly(assembly: &capability::Assembly, gpus: &[(u32, u64)], reserved: u64) -> Option<OmniResident> {
        let hf_dir = assembly.roles.get("weights")?.to_string_lossy().into_owned();
        let devices = if gpus.is_empty() {
            eprintln!("brain: omni has no budgeted GPU; it will fall back to the ambient device");
            Vec::new()
        } else {
            gpus.iter().map(|&(i, total)| (i, total.saturating_sub(reserved))).collect()
        };
        Some(OmniResident { hf_dir, devices, plan: OnceLock::new() })
    }

    /// Rough on-disk size of the checkpoint directory (sum of `*.safetensors`
    /// shards) - the mmap footprint the loader's reads touch, used as the host
    /// RAM figure.
    fn checkpoint_bytes(&self) -> u64 {
        std::fs::read_dir(&self.hf_dir)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
            .filter_map(|e| e.metadata().ok().map(|m| m.len()))
            .sum()
    }

    /// The placement, computed once. Returns a plan naming ZERO devices (never
    /// a panic) when the checkpoint cannot be opened or does not fit - the
    /// documented "this model is unavailable" signal, which `claim_multi`
    /// turns into a clean per-job error instead of a dispatcher crash.
    /// `estimate_multi` runs on the dispatcher thread, so neither a panic nor
    /// a re-parse of a 28k-tensor header per scheduling round is acceptable.
    fn plan(&self) -> Plan {
        if let Some(p) = self.plan.get() {
            return p.clone();
        }
        let computed = self.plan_uncached();
        // A losing racer's value is simply dropped: `plan_uncached` is a pure
        // function of `self`, so which racer wins cannot matter.
        let _ = self.plan.set(computed.clone());
        computed
    }

    fn plan_uncached(&self) -> Plan {
        let host_ram = {
            let bytes = self.checkpoint_bytes();
            if bytes > 0 { bytes } else { 70u64 << 30 } // 70 GiB fallback: the real checkpoint's own bf16 size
        };
        if self.devices.is_empty() {
            return Plan { devices: Vec::new(), bytes: Vec::new(), host_ram };
        }
        let reader = match WeightReader::open_hf_dir(std::path::Path::new(&self.hf_dir)) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("brain/qwen3omnimoe: cannot open '{}': {e} -- reporting zero devices so the claim fails placement instead of panicking", self.hf_dir);
                return Plan { devices: Vec::new(), bytes: Vec::new(), host_ram };
            }
        };
        let cfg = match std::fs::read_to_string(std::path::Path::new(&self.hf_dir).join("config.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        {
            Some(root) => qwen3omnimoe::config::OmniConfig::from_json(&root).thinker.text,
            None => {
                eprintln!("brain/qwen3omnimoe: cannot read '{}/config.json' -- reporting zero devices", self.hf_dir);
                return Plan { devices: Vec::new(), bytes: Vec::new(), host_ram };
            }
        };
        let Some(cost) = qwen3omnimoe::thinker_plan::layer_cost(&reader, &cfg) else {
            eprintln!("brain/qwen3omnimoe: '{}' is missing Thinker tensors this model loads -- reporting zero devices", self.hf_dir);
            return Plan { devices: Vec::new(), bytes: Vec::new(), host_ram };
        };
        let caps: Vec<(usize, u64)> = self.devices.iter().map(|&(_, c)| c).enumerate().collect();
        let Some(placement) = qwen3omnimoe::thinker_plan::place_fewest_devices(&cost, &caps) else {
            eprintln!(
                "brain/qwen3omnimoe: does not fit the {} budgeted device(s) even streamed ({} bytes available) -- reporting zero devices",
                caps.len(),
                caps.iter().map(|&(_, c)| c).sum::<u64>()
            );
            return Plan { devices: Vec::new(), bytes: Vec::new(), host_ram };
        };
        let devices = placement.stages.iter().map(|s| self.devices[s.device]).collect();
        let bytes = placement.stages.iter().map(|s| s.bytes).collect();
        Plan { devices, bytes, host_ram }
    }
}

impl ResidentModel for OmniResident {
    fn manifest(&self) -> Manifest {
        qwen3omnimoe::caps::manifest()
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(qwen3omnimoe::caps::MODEL, "default")
    }

    /// Deliberately unusable: this model is registered via `register_multi`
    /// and claimed via `claim_multi`, so the single-device estimate is never
    /// consulted. Reporting a real figure here would invite the plain
    /// `register` path, which is exactly how it previously ended up on the CPU
    /// lane with an unbudgeted GPU behind it.
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        MemCost::new(0, 0)
    }

    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        Err("brain/qwen3omnimoe: single-device activate is not supported -- this model places itself across devices, claim it via ResidencyManager::claim_multi".to_string())
    }
}

impl MultiDeviceResidentModel for OmniResident {
    fn estimate_multi(&self, _key: &InstanceKey) -> MultiDeviceCost {
        let plan = self.plan();
        let per_device = plan.devices.iter().zip(&plan.bytes).map(|(&(i, _), &b)| (Device::Gpu(i), b)).collect();
        MultiDeviceCost::new(per_device, plan.host_ram)
    }

    fn activate_multi(&self, _key: &InstanceKey, devices: &[Device]) -> Result<Box<dyn Instance>, String> {
        let plan = self.plan();
        if plan.devices.is_empty() {
            return Err("brain/qwen3omnimoe: no placement (checkpoint unreadable, or it does not fit the budgeted devices)".to_string());
        }
        // `claim_multi` reserves against exactly the devices `estimate_multi`
        // named, so it hands back that same set. Insisting on it here (rather
        // than silently re-planning for whatever arrives) is what makes the
        // reservation and the allocation describe the same bytes.
        let planned: Vec<Device> = plan.devices.iter().map(|&(i, _)| Device::Gpu(i)).collect();
        if devices.len() != planned.len() || !devices.iter().all(|d| planned.contains(d)) {
            return Err(format!("brain/qwen3omnimoe: activate_multi got devices {devices:?} but the plan placed {planned:?}"));
        }
        let provider = OmniProvider::load_on(&self.hf_dir, &plan.devices)?;
        Ok(Box::new(OmniInstance { inner: provider.inner() }))
    }
}

struct OmniInstance {
    inner: Arc<qwen3omnimoe::caps::OmniInner>,
}

impl Instance for OmniInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        // Single dispatch path shared with `qwen3omnimoe::caps::OmniProvider` -- this
        // resident (not that Provider) is what `brain serve` actually routes
        // D-Bus/HTTP requests through, and it previously IGNORED the action
        // name: an advertised `speak` silently ran `generate` (text out, no
        // audio, no error). `run_action` matches on the action and errors on
        // anything the manifest doesn't declare.
        qwen3omnimoe::caps::run_action(&self.inner, action, inv, progress)
    }
}

/// The int8 GPU-sharded Thinker (`qwen3omnimoe::int8_thinker_resident::
/// Int8ThinkerResident`) - layer-sharded across as many budgeted GPUs as its
/// real per-layer bytes need, streamed straight from a pre-quantized
/// brain-native checkpoint. Reachable ONLY via `Executor::register_multi`
/// (never `register` - see that type's own doc on why a multi-device-only
/// model must stay out of the plain single-device registry). A SEPARATE
/// resident from [`OmniResident`] above: that one is the validation-tier,
/// single-device, HF-checkpoint path that re-streams every layer per token;
/// this one is genuinely GPU-resident, and the two are not interchangeable.
///
/// Its checkpoint comes from the model store: the brain-native int8 file
/// `brain qwen3omnimoe import` wrote (`qwen3omnimoe::spec::INT8_SPEC`), else the
/// published directory (`qwen3omnimoe::spec::SPEC`), quantized to int8 as it
/// is loaded (`qwen3omnimoe::import::Int8View`). Its tokenizer, for the CHAT
/// request shape, is the int8 checkpoint's `tokenizer` role, else the files
/// beside the checkpoint, else the published directory; resolving to nothing is
/// not an error - the model then serves only raw token ids. The published
/// directory also supplies the vision/audio towers for multimodal input.
///
/// `gpus` is `build_executor`'s own budgeted GPU list as `(index, TOTAL
/// bytes)`, and `reserved` the per-card headroom it keeps free - so what is
/// handed on is each card's genuinely USABLE capacity, the same figure the
/// scheduler will budget against. Passing capacity (not just identity) is
/// what lets `model::shard::plan_fewest_devices` size the split to the
/// hardware: a 24 GB and an 8 GB card get layers in that proportion, a model
/// that fits one card is not spread over three, and a model that fits none of
/// them is reported as unplaceable instead of OOMing mid-load. Deriving the
/// device list from `gpus` (rather than hardcoding `[Gpu(0), Gpu(1)]`) is
/// also what keeps it correct on a 1-GPU box and on a 3+-GPU one.
pub fn int8_thinker_multi_from_store(store: &loader::served::ServedStore, gpus: &[(u32, u64)], reserved: u64) -> Option<qwen3omnimoe::int8_thinker_resident::Int8ThinkerResident> {
    let role = |assembly: Option<capability::Assembly>, name: &str| assembly.and_then(|a| a.roles.get(name).map(|p| p.to_string_lossy().into_owned()));
    let int8 = loader::served::served_assembly(store, qwen3omnimoe::spec::INT8_ARCH, &qwen3omnimoe::spec::INT8_SPEC);
    let published = role(loader::served::served_assembly(store, qwen3omnimoe::spec::ARCH, &qwen3omnimoe::spec::SPEC), "weights");
    let tokenizer_role = role(int8.clone(), "tokenizer");
    let checkpoint_path = role(int8, "weights").or_else(|| published.clone())?;
    if gpus.is_empty() {
        residency::log::not_served(&format!("brain: {} not served (no GPU budgeted -- it is GPU-sharded only, no CPU path)", qwen3omnimoe::int8_thinker_resident::MODEL));
        return None;
    }
    let devices: Vec<(Device, u64)> = gpus.iter().map(|&(i, total)| (Device::Gpu(i), total.saturating_sub(reserved))).collect();
    // Real numbers confirmed against the real checkpoint's config (not
    // assumed) -- validated end to end on two physically separate GPUs
    // during the int8 dual-GPU residency work. `ThinkerConfig::defaults()`
    // carries the same text-decoder numbers `MoeTextConfig::thinker_defaults()`
    // did, plus the special media token ids `qwen3omnimoe::mm::build_multimodal_prompt`
    // needs -- see that function's own doc for why this is one config, not
    // two independently-maintained copies of the same numbers.
    let cfg = qwen3omnimoe::config::ThinkerConfig::defaults();
    let tokenizer_dir = int8_tokenizer_dir(&checkpoint_path, tokenizer_role.as_deref(), published.as_deref());
    if tokenizer_dir.is_none() {
        eprintln!(
            "brain: {} has no tokenizer -- it will serve raw token ids only, NOT /v1/chat/completions. \
             Put tokenizer.json beside the checkpoint, or --pin {}.tokenizer=FILE.",
            qwen3omnimoe::int8_thinker_resident::MODEL,
            qwen3omnimoe::spec::INT8_ARCH
        );
    }
    // Multimodal input needs the published directory for the vision/audio
    // tower weights (the int8 checkpoint's own audio.*/vision.* tensors are
    // quantized; see qwen3omnimoe::int8_thinker_resident's module doc for why
    // this model does not read them); without it this model serves text-only.
    let hf_dir = published;
    if hf_dir.is_none() {
        eprintln!("brain: {} has no published checkpoint directory in the model store -- it will serve text-only generate (no audio/image/video input).", qwen3omnimoe::int8_thinker_resident::MODEL);
    }
    Some(qwen3omnimoe::int8_thinker_resident::Int8ThinkerResident::new(checkpoint_path, cfg, devices).with_tokenizer_dir(tokenizer_dir).with_hf_dir(hf_dir))
}

/// Where the int8 resident reads its tokenizer from, in preference order:
/// the directory of its resolved `tokenizer`, then the checkpoint's own
/// directory, then the published directory. `None` when none of them holds
/// tokenizer files.
///
/// The search exists because a brain-native int8 checkpoint is a single
/// `.safetensors` - it carries weights and a model card, never vocab files -
/// so the tokenizer has to come from somewhere the operator already has. Each
/// candidate is CHECKED for the files rather than assumed, so a stale pin
/// falls through to a directory that works instead of disabling chat.
fn int8_tokenizer_dir(checkpoint_path: &str, tokenizer: Option<&str>, published: Option<&str>) -> Option<String> {
    let has_tokenizer = |dir: &std::path::Path| {
        dir.join("tokenizer.json").exists() || (dir.join("vocab.json").exists() && dir.join("merges.txt").exists())
    };
    let explicit = tokenizer.and_then(|t| std::path::Path::new(t).parent()).map(|d| d.to_string_lossy().into_owned());
    // A downloaded checkpoint directory holds its own tokenizer; a converted
    // file has it beside it.
    let path = std::path::Path::new(checkpoint_path);
    let beside = if path.is_dir() { Some(checkpoint_path.to_string()) } else { path.parent().map(|p| p.to_string_lossy().into_owned()) };
    let hf = published.map(str::to_string);
    [explicit, beside, hf].into_iter().flatten().find(|d| has_tokenizer(std::path::Path::new(d)))
}
