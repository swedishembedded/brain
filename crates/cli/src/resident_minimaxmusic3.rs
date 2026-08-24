// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Resident-model adapter putting MiniMax Music 3 behind the residency
//! [`Executor`](residency::Executor), on `crate::resident_ltxv`'s pattern -
//! the closest sibling this model has, and for structural reasons rather
//! than stylistic ones: both are multi-minute generative pipelines that
//! split one request's classifier-free-guidance pair across two cards, and
//! both hold their expensive weights in a process-wide, checkpoint-keyed
//! host store rather than in a field here.
//!
//! Config is env-only, matching `crates/arch`'s own `weights_env`
//! registration for this architecture and `minimaxmusic3::generate::
//! Paths::from_env`'s own six roles:
//!   * `BRAIN_MINIMAXMUSIC3_LM` - Global LLM (Qwen3-8B architecture)
//!   * `BRAIN_MINIMAXMUSIC3_DEPTH` - RVQ depth decoder
//!   * `BRAIN_MINIMAXMUSIC3_CONDITION` - condition encoder
//!   * `BRAIN_MINIMAXMUSIC3_DIT` - flow-matching DiT
//!   * `BRAIN_MINIMAXMUSIC3_VOCODER` - vocoder
//!   * `BRAIN_MINIMAXMUSIC3_TOKENIZER` - tokenizer
//!
//! One action, `generate`, dispatched straight through
//! `minimaxmusic3::caps::generate_action` - the SAME param-decode +
//! generation + outcome-shaping implementation
//! `minimaxmusic3::caps::MinimaxMusic3Provider` (the direct/`brain do`
//! path) uses, so this file adds no second copy of that logic.
//!
//! # What this instance holds, and where it lives
//!
//! Nothing in a field except the six paths and the assigned card. The thing
//! worth holding warm is the four components whose imported form is a plain
//! tree of host `Vec<f32>` (the DiT's 9.7 GB, the depth decoder's 2.6 GB,
//! the vocoder and the condition encoder), and they live in
//! `minimaxmusic3::weightcache` - a process-wide store keyed on the
//! checkpoint directory, for the same reason `ltxv::weightcache` is: the
//! pipeline reaches them from deep inside `generate()`, and their correct
//! identity is the checkpoint, not this instance. What this file holds is
//! the RESIDENCY side of that store - the `estimate`/`estimate_at`/
//! `demote`/`promote`/`metrics` surface that lets the manager budget it and
//! release it, instead of it being a cache nothing can see or bound.
//!
//! The Global LLM is deliberately not warm; `minimaxmusic3::weightcache`'s
//! own module doc gives the three reasons (it owns a `Gpu`, it is `!Sync`,
//! and its KV capacity is a function of the request, not the checkpoint).

use capability::{ActionResult, Invocation, Manifest, Progress};
use minimaxmusic3::generate::Paths;
use minimaxmusic3::{memory, weightcache};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel, Tier};

/// Host bytes a generation holds that are NOT the warm weight cache: the
/// Global LLM's `lm_head` read back to the host for the sampling head
/// (3.28 GB - `generate::generate`'s `lm_cond.read_weight`), the whole
/// song's per-frame hidden states (`num_frames * 8 * 4096 * 4` bytes, ~786
/// MB for a four-minute track), every chunk's latents held across the
/// denoise/vocoder boundary, and the output waveform.
///
/// A single number rather than a function of `duration_seconds` because the
/// instance key deliberately does not carry the duration (see
/// [`MinimaxMusic3Resident::instance_key`]), and because at 5 GB it already
/// covers a track far longer than the AR stage's own throughput makes
/// practical. Charging a term that grows without bound with a client-
/// supplied parameter would make the model unplaceable on a request nobody
/// would wait for.
const TRANSIENT_HOST_BYTES: u64 = 5u64 << 30;

/// MiniMax Music 3 behind the scheduler.
pub struct MinimaxMusic3Resident {
    paths: Paths,
}

impl MinimaxMusic3Resident {
    /// Configure from the environment. Returns `None` (not served) when
    /// any of the six roles is unset, like
    /// [`crate::resident::YoloResident::from_env`].
    pub fn from_env() -> Option<MinimaxMusic3Resident> {
        Paths::from_env().ok().map(|paths| MinimaxMusic3Resident { paths })
    }

    /// Host bytes `minimaxmusic3::weightcache` holds once every component
    /// has been read once.
    ///
    /// Derived where a closed form exists (the DiT's block stack and the
    /// depth decoder's whole weight set, both functions of this model's own
    /// `::real()` configs - see `minimaxmusic3::memory`), and read from the
    /// checkpoint directory where one does not. The `* 2` on the two
    /// file-sized roles is this repo's safetensors reader materialising
    /// every tensor as f32: a bf16 checkpoint doubles on the way into host
    /// memory. It over-charges an already-fp32 file, which is the safe
    /// direction, and both roles are under 250 MB on disk.
    fn warm_host_bytes(&self) -> u64 {
        memory::derivable_warm_host_bytes() + 2 * (weightcache::checkpoint_bytes(&self.paths.vocoder) + weightcache::checkpoint_bytes(&self.paths.condition))
    }
}

impl ResidentModel for MinimaxMusic3Resident {
    fn manifest(&self) -> Manifest {
        // The spec lives in minimaxmusic3::caps, next to the catalog's own
        // `generate` spec, so the two surfaces cannot silently diverge.
        minimaxmusic3::caps::resident_manifest()
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        // A constant, deliberately, unlike `resident_wan`/`resident_ltxv`
        // which key on the latent extent. Those key on what fixes a
        // resident DiT's graph sizes; this instance builds no graph at all
        // (every `Gpu` and every device-resident object is created and
        // dropped inside one `generate` call), and the warm weights are
        // keyed on the CHECKPOINT inside `minimaxmusic3::weightcache`, not
        // on anything a request carries. Splitting on `duration_seconds`
        // would therefore multiply budget charges across instances that
        // share every byte they hold.
        InstanceKey::new(minimaxmusic3::caps::MODEL, "default")
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // # VRAM
        //
        // The three stages are sequential and never co-resident
        // (`minimaxmusic3::generate`'s sequential-stage device discipline),
        // so the budget is the MAX over them, not their sum -
        // `minimaxmusic3::memory::StagePeaks::peak` is that max, derived
        // from this model's own `::real()` configs and pinned by that
        // module's tests. At the released dims:
        //
        //   AR       16.09 GB   one int8 Global LLM instance (6.95 GB of
        //                       linears, MEASURED, plus `tok.weight` and
        //                       `lm_head` at 3.28 GB each - they stay fp32)
        //                       plus the depth decoder's 2.58 GB, which
        //                       `generate::depth_decoder_device` puts on
        //                       the SAME card as the conditional branch
        //   denoise  10.05 GB   one `dit::Resident`: a 9.66 GB block stack
        //                       plus a measured 384 MiB of non-block
        //                       tensors, driver context and scratch
        //   vocode   12.86 GB   MEASURED peak decoding one 689-latent
        //                       chunk, which `denoise::CHUNK_FRAMES` caps
        //                       for every chunk however long the song is
        //
        // This replaces a hardcoded `vram: 0`, which was not a conservative
        // choice but a disabling one: `residency::place::pick_device` skips
        // every GPU class when `cost.vram == 0`, so on a 2x24 GB box this
        // model could never be GPU-placed at all, and the CPU fallback it
        // fell through to cannot run it.
        //
        // # The second card, and what is NOT charged here
        //
        // A generation on a two-card box occupies BOTH cards: the AR
        // stage's unconditional Global LLM branch goes on the other card
        // (`generate::ar_branch_devices`) and the denoise stage's
        // zero-condition CFG forward goes there too
        // (`minimaxmusic3::devplan`). This figure is the footprint on ONE
        // card - the card residency assigned - and the second card's
        // identical footprint is charged to no budget.
        //
        // That gap is deliberate and is the same one `resident_ltxv.rs`
        // has, not an oversight. The honest seam for it is
        // `residency::MultiDeviceResidentModel`, and it is not taken here
        // for three reasons, each of which would be a regression:
        // `multi::pick_devices` is all-or-nothing over a FIXED device set,
        // so naming two cards would make this model unplaceable on a
        // one-card box (where it degrades correctly today);
        // multi-device residents are not auto-evicted
        // (`residency::multi`'s own scope note); and
        // `crates/stats`' `ModelStat` schema is single-device, so a
        // multi-device resident disappears from `braintop`. Closing it
        // properly means changing those three things first. Until then
        // what bounds the borrowed card is `memauth`'s
        // `--limit-vram-total` and `BRAIN_MINIMAXMUSIC3_CFG_PARALLEL=0`,
        // as `minimaxmusic3::devplan`'s module doc records.
        //
        // # RAM
        //
        // What the warm weight cache holds plus what one generation holds
        // beside it. The old figure here was the on-disk checkpoint size
        // times four, a multiplier justified by an int8-promotes-to-fp32
        // claim this crate's own ledger later disproved for GPU backends.
        MemCost::new(memory::stage_peaks().peak(), self.warm_host_bytes() + TRANSIENT_HOST_BYTES)
    }

    /// Below `Hot` the warm weight cache is gone, so the host figure drops
    /// by exactly what [`Self::estimate`] added for it, and the VRAM figure
    /// drops to zero - nothing device-side survives a `generate` call here
    /// in the first place, so a demoted instance holds no card at all.
    ///
    /// `Cold` additionally reports the checkpoints' on-disk footprint as
    /// `mapped`: those pages are reclaimable by the kernel in a way a live
    /// allocation is not, which is exactly the distinction
    /// [`MemCost::mapped`] exists to carry.
    fn estimate_at(&self, key: &InstanceKey, tier: Tier) -> MemCost {
        let hot = self.estimate(key);
        match tier {
            Tier::Hot => hot,
            Tier::Warm => MemCost::new(0, TRANSIENT_HOST_BYTES),
            Tier::Cold => {
                let on_disk: u64 = [&self.paths.dit, &self.paths.depth, &self.paths.vocoder, &self.paths.condition].iter().map(|d| weightcache::checkpoint_bytes(d)).sum();
                MemCost::new(0, TRANSIENT_HOST_BYTES).with_mapped(on_disk)
            }
        }
    }

    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        // `minimaxmusic3::generate::generate` builds every device object
        // internally per call, so the resident's only job here is to fail
        // fast when the configured directories are missing - a placement
        // error, not a request error.
        for (dir, role) in [
            (&self.paths.lm, "Global LLM"),
            (&self.paths.depth, "depth decoder"),
            (&self.paths.condition, "condition encoder"),
            (&self.paths.dit, "DiT"),
            (&self.paths.vocoder, "vocoder"),
            (&self.paths.tokenizer, "tokenizer"),
        ] {
            if !std::path::Path::new(dir).exists() {
                return Err(format!("minimaxmusic3: {role} weights not found at {dir} (set the matching BRAIN_MINIMAXMUSIC3_* var)"));
            }
        }
        Ok(Box::new(MinimaxMusic3Instance { paths: self.paths.clone(), device }))
    }
}

struct MinimaxMusic3Instance {
    paths: Paths,
    /// The card the scheduler assigned. `activate` used to take this and
    /// drop it, so every generation ran on whatever the ambient selection
    /// happened to be and the scheduler's placement decision was silently
    /// discarded.
    device: Device,
}

impl Instance for MinimaxMusic3Instance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        if action != "generate" {
            return Err(format!("minimaxmusic3: unsupported action '{action}' (this resident declares: generate)"));
        }
        // Scope the WHOLE generation to the assigned card. Every stage
        // builds its own `Gpu` at call time, so a scoped selection - not a
        // constructor argument - is what makes all of them land on the
        // assigned device, and it is also what `minimaxmusic3::devplan`
        // reads (`devices::current_gpu()`) to decide which card is the
        // conditional branch's. `on_device` is the shared helper for this;
        // never an env mutation, which AGENTS.md forbids from a
        // server-lifetime resident.
        crate::resident_llm::on_device(self.device, || minimaxmusic3::caps::generate_action(&self.paths, inv, progress))?
    }

    /// **Serial, and here is the reason** - which AGENTS.md requires
    /// whenever a model does not override this with a genuine batched
    /// forward.
    ///
    /// Three separate facts, each independently sufficient:
    ///
    /// * **One request already occupies every card on this box.** A
    ///   generation's AR stage puts one ~13.5 GB Global LLM instance on
    ///   each of two cards (`generate::ar_branch_devices`) and its denoise
    ///   stage puts one ~10 GB `dit::Resident` on each
    ///   (`denoise::CfgDevices`, `minimaxmusic3::devplan`). Two of those
    ///   AR instances do not fit one 24 GB card - `minimaxmusic3::memory`'s
    ///   own test asserts exactly that, and it is why the upstream
    ///   implementation documents two GPUs as a hard requirement. So the
    ///   `residency::DevicePool` trick `resident_ltxv::run_batch` uses -
    ///   one whole generation per card, concurrently - has no spare card to
    ///   offer here. On a machine with four or more schedulable cards it
    ///   would, and the shape of that change is a card-PAIR pool rather
    ///   than a card pool; `generate` takes no pair argument today, so that
    ///   is real follow-up work, not a line that was forgotten.
    ///
    /// * **The DiT has no batch axis to fill.** `dit.rs`'s module doc says
    ///   `batch=1` by construction, and it is not a missing loop: at
    ///   `b > 1` the RoPE table's `tmod` (`model::block`'s
    ///   `rope2d_partial_fwd` passes `tmod = rows`, so rows past the first
    ///   sequence index off the end of a table sized for one), the
    ///   row-0 timestep slice into `proj_out_postprocess`, the
    ///   `preprocess_hidden_lc` transpose, the timestep row assembly, and
    ///   the `Bidir { b: 1, .. }` scores/probs allocation each produce
    ///   WRONG NUMBERS rather than an error - and because this attention is
    ///   fully bidirectional and unmasked, a batched slab would let every
    ///   request attend to every other one silently. Batching there is a
    ///   correctness project with a parity gate, not a `run_batch`.
    ///
    /// * **The AR stage is single-sequence at the API level.**
    ///   `global_llm::import` asserts `b == 1` in as many words, because
    ///   `crates/qwen3`'s KV-cache decode path sizes `kcache`/`vcache` as
    ///   `t * kv_dim` with no batch axis at all. Cross-request batching
    ///   there means adopting the `model::serve::PagedDecoder` seam - the
    ///   same seam `qwen3::serve::Engine` implements for continuous
    ///   batching - which is a substantially larger change than this
    ///   adapter.
    ///
    /// What a batch DOES get, and it is not nothing: the requests run
    /// through ONE instance against `minimaxmusic3::weightcache`, so the
    /// second and later generations skip the ~10.7 GB checkpoint read the
    /// first paid. [`Self::metrics`] reports whether that actually
    /// happened rather than leaving it to be assumed.
    ///
    /// The default `Instance::run_batch` is this same serial loop; it is
    /// spelled out here so the reason lives in the file rather than in an
    /// absence.
    fn run_batch(&mut self, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        invs.iter().enumerate().map(|(i, inv)| self.run(action, inv, &mut |p| progress(i, p))).collect()
    }

    /// The warm weight cache's own hit/miss/eviction counters and current
    /// footprint, so an operator can see whether residency is doing
    /// anything without re-deriving it from a trace.
    fn metrics(&self) -> Vec<(String, serde_json::Value)> {
        let s = weightcache::stats();
        vec![
            ("minimaxmusic3_weight_cache_hits".into(), s.hits.into()),
            ("minimaxmusic3_weight_cache_misses".into(), s.misses.into()),
            ("minimaxmusic3_weight_cache_evictions".into(), s.evictions.into()),
            ("minimaxmusic3_weight_cache_entries".into(), s.entries.into()),
            ("minimaxmusic3_weight_cache_bytes".into(), s.bytes.into()),
        ]
    }

    /// Release the warm weight cache.
    ///
    /// `Instance::demote`'s contract is "`Warm`: release device buffers,
    /// keep host bytes". For this model there are no device buffers to
    /// release BETWEEN calls - every `Gpu` and every device-resident object
    /// is built and dropped inside one `generate` call - so an instance's
    /// entire reclaimable resident footprint is host RAM, exactly as it is
    /// for `resident_ltxv`. Releasing it is safe at any moment for the
    /// reason that module's doc gives: the entries are a pure function of
    /// immutable checkpoint bytes, so dropping one can only cost time. A
    /// generation already holding an `Arc` keeps its copy until it
    /// finishes.
    fn demote(&mut self, tier: Tier) -> Result<(), String> {
        debug_assert_ne!(tier, Tier::Hot, "demote is never a promotion");
        if tier == Tier::Hot {
            return Err("minimaxmusic3: demote(Hot) is not a demotion".into());
        }
        // Reporting `Ok` with nothing held would let the manager charge a
        // Warm cost against progress it did not make.
        if weightcache::bytes() == 0 {
            return Err("minimaxmusic3: nothing resident to demote (no component has been loaded yet)".into());
        }
        weightcache::clear();
        Ok(())
    }

    /// Return to `Hot`. Deliberately lazy: the next generation re-reads
    /// exactly the components it needs, in the code that already knows how,
    /// and re-reading ~13 GB eagerly here would block the manager's worker
    /// thread for minutes to do work the request itself does anyway.
    fn promote(&mut self, _device: Device) -> Result<(), String> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: f64 = 1e9;

    fn resident() -> MinimaxMusic3Resident {
        MinimaxMusic3Resident { paths: Paths { lm: "/lm".into(), depth: "/depth".into(), condition: "/condition".into(), dit: "/dit".into(), vocoder: "/vocoder".into(), tokenizer: "/tok".into() } }
    }

    fn key() -> InstanceKey {
        InstanceKey::new(minimaxmusic3::caps::MODEL, "default")
    }

    /// The regression this file exists to fix: `vram == 0` made
    /// `place::pick_device` skip every GPU (its GPU loop `continue`s on
    /// `need == 0`), so a model that needs 16 GB of a 24 GB card could
    /// never be placed on one. The estimate must now name a real,
    /// GPU-placeable figure - and must still fit a single card with the
    /// serving default reserve, or `could_ever_fit` reports `TooLarge` and
    /// the model becomes unservable rather than merely unplaced.
    #[test]
    fn the_estimate_is_gpu_placeable_on_a_24_gb_card() {
        let cost = resident().estimate(&key());
        assert!(cost.vram > 0, "a vram of 0 is not conservative, it is unplaceable on any GPU");
        assert_eq!(cost.vram, minimaxmusic3::memory::stage_peaks().peak());

        let gb = 1u64 << 30;
        let mut budgets = residency::budget::Budgets::new();
        budgets.set(Device::Gpu(0), 24 * gb, 2 * gb).set(Device::Gpu(1), 24 * gb, 2 * gb).set(Device::Cpu, 184 * gb, 0);
        assert!(residency::place::could_ever_fit(&cost, &budgets), "{:.2} GB must fit a 24 GiB card minus the 2 GiB reserve", cost.vram as f64 / GB);
        assert!(matches!(residency::place::pick_device(&cost, &budgets, &residency::place::no_exclude()), Some(Device::Gpu(_))), "must place on a GPU, not fall through to the CPU");
    }

    /// The stage figures are a MAX over three stages that are never
    /// co-resident, never a sum - the shape `estimate` has always had, now
    /// with numbers behind it. A sum would be 39 GB and unplaceable on any
    /// card this repo targets.
    #[test]
    fn the_vram_figure_is_the_largest_stage_and_not_their_sum() {
        let p = minimaxmusic3::memory::stage_peaks();
        let cost = resident().estimate(&key());
        assert_eq!(cost.vram, p.ar.max(p.denoise).max(p.vocode));
        assert!(cost.vram < p.ar + p.denoise + p.vocode);
        assert!((cost.vram as f64 / GB - 16.09).abs() < 0.01, "{:.2} GB", cost.vram as f64 / GB);
    }

    /// Below `Hot` the warm weight cache is gone, so the host figure must
    /// drop by exactly what it contributed - and nothing device-side
    /// survives a call here, so a demoted instance must hold no card.
    /// `Cold` reports the checkpoints as reclaimable `mapped` pages rather
    /// than as live bytes.
    #[test]
    fn a_demoted_estimate_drops_exactly_the_weight_cache() {
        let r = resident();
        let hot = r.estimate(&key());
        let warm = r.estimate_at(&key(), Tier::Warm);
        assert_eq!(r.estimate_at(&key(), Tier::Hot), hot, "estimate_at(Hot) must be estimate");
        assert_eq!(warm.ram, hot.ram - r.warm_host_bytes(), "a Warm estimate must be the Hot one minus exactly the cache");
        assert_eq!(warm.vram, 0, "nothing device-side survives a generate call here");
        // These paths do not exist, so their on-disk figure is 0; the
        // structural claim is that Cold reports it as `mapped`, not `ram`.
        let cold = r.estimate_at(&key(), Tier::Cold);
        assert_eq!(cold.ram, warm.ram);
        assert_eq!(cold.vram, 0);
    }

    /// The key must NOT split on anything a request carries: every
    /// invocation shares one instance, because the instance holds no
    /// per-request graph and the warm weights are keyed on the checkpoint.
    #[test]
    fn every_request_shares_one_instance() {
        use serde_json::json;
        let r = resident();
        let a = r.instance_key("generate", &Invocation::new().set("lyrics", json!("a")).set("duration_seconds", json!(10.0)));
        let b = r.instance_key("generate", &Invocation::new().set("lyrics", json!("b")).set("duration_seconds", json!(240.0)).set("seed", json!(7)));
        assert_eq!(a, b);
        assert_eq!(a.model, minimaxmusic3::caps::MODEL);
    }

    /// `demote` must refuse rather than claim progress it did not make when
    /// nothing is held - the manager would otherwise re-budget against a
    /// release that never happened. (The cache is process-wide, so this
    /// asserts the empty case only, which is the one that can lie.)
    #[test]
    fn demote_refuses_when_nothing_is_held() {
        let mut inst = MinimaxMusic3Instance { paths: resident().paths, device: Device::Gpu(0) };
        weightcache::clear();
        if weightcache::bytes() == 0 {
            assert!(inst.demote(Tier::Warm).is_err(), "an empty cache must not report a successful demotion");
        }
        // `demote(Hot)` is not exercised here: it is a caller bug the
        // implementation debug-asserts on, exactly like
        // `resident_ltxv`'s, so a test that called it would fail the debug
        // build rather than assert anything about the release one.
        assert!(inst.promote(Device::Gpu(0)).is_ok(), "promote is lazy and always succeeds");
    }

    /// The adapter advertises the shared manifest, and rejects an action it
    /// does not declare rather than silently running `generate`.
    #[test]
    fn the_adapter_advertises_the_shared_manifest_and_rejects_other_actions() {
        let m = resident().manifest();
        assert_eq!(m.model, minimaxmusic3::caps::MODEL);
        assert_eq!(m.actions.len(), minimaxmusic3::caps::manifest().actions.len());
        let mut inst = MinimaxMusic3Instance { paths: resident().paths, device: Device::Cpu };
        let err = inst.run("t2v", &Invocation::new(), &mut |_| {}).unwrap_err();
        assert!(err.contains("unsupported action"), "{err}");
    }

    /// `metrics` must be present and data-driven - every key is reported
    /// whether or not anything is cached, so `braintop`'s generic tree view
    /// renders a zero rather than a missing row.
    #[test]
    fn metrics_report_the_weight_cache() {
        let inst = MinimaxMusic3Instance { paths: resident().paths, device: Device::Cpu };
        let names: Vec<String> = inst.metrics().into_iter().map(|(k, _)| k).collect();
        assert_eq!(names, ["minimaxmusic3_weight_cache_hits", "minimaxmusic3_weight_cache_misses", "minimaxmusic3_weight_cache_evictions", "minimaxmusic3_weight_cache_entries", "minimaxmusic3_weight_cache_bytes"]);
    }
}
