// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device residency for the streamed 22B DiT: what stays on the card between
//! two denoise-step forwards, for how long, and against whose budget.
//!
//! Swedish Embedded AB implements device-memory residency and scheduling for
//! production inference pipelines - keeping the weights that never change on
//! the accelerator instead of re-uploading them every step, without giving up
//! bit-exact reproducibility. If your team needs expertise in GPU memory
//! lifecycle design, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # The overhead this removes
//!
//! [`crate::dit::forward_q_streamed`] used to open a FRESH `gpu_core::Gpu` on
//! every call and re-upload all 48 already-quantized blocks - ~13 GB of int8
//! bytes that a generation never changes - from the host-RAM
//! [`crate::weightcache`] store to that fresh device, from scratch, on every
//! one of a generation's 8-16 forwards. Measured on this box (two Tesla P40s,
//! PCIe, no NVLink), real `ltx-2.5-22b-distilled-transformer-Q8_0.gguf`, all
//! 48 layers, T=3520 (720p), cache-warm: the `block GPU upload+forward+wait`
//! bucket was **the majority of the call**, and real GPU kernel time in the
//! same call was a small fraction of it. Nearly all of the difference is buffer
//! creation + `queue.write_buffer` staging for weights that were already
//! correct on the card one step earlier.
//!
//! # The lifecycle
//!
//! * A [`DitSession`] is one card, for one generation. It owns ONE `Gpu`
//!   handle - opened once, not per forward - and, optionally, a
//!   [`BlockWindow`].
//! * A [`BlockWindow`] is a fixed number of device SLOTS over the model's 48
//!   blocks. Each occupied slot owns its own `Gpu` handle (a `Gpu::share` of
//!   the session's) plus one uploaded [`crate::block::LtxBlockQ`], so dropping
//!   a slot releases exactly that slot's `memauth` grants and exactly that
//!   slot's VRAM.
//! * Which slot a block lands in is [`weightset`]'s decision, not this
//!   module's. A denoise loop visits its blocks in an order known exactly in
//!   advance - `Schedule::cyclic(48, passes)` - which is the problem
//!   `weightset` was built for and the reason it is used here rather than a
//!   second bespoke window. With enough slots for all 48 the plan pins
//!   everything and never evicts; with fewer it pins the longest prefix and
//!   rotates the tail through the remaining slots by furthest-next-use
//!   (Bélády, exact rather than heuristic because the future IS the schedule).
//! * Every forward's ACTIVATIONS run on a fresh scratch handle
//!   (`Gpu::share`), dropped when the call returns - see
//!   [`crate::block::LtxBlockQ::forward_on`] for why that separation is what
//!   keeps the `memauth` accounting honest.
//! * The whole session dies with the `RealDit` that owns it, which
//!   `crate::pipeline::generate` already drops before the VAE decode opens its
//!   own device.
//!
//! # The budget, and the graceful path when it does not fit
//!
//! The slot count is not a guess and not a constant: it is
//! `(usable_vram - activation_reserve) / cached_block_bytes`, clamped to the
//! layer count, where `usable_vram` is the `memauth` authority's real headroom
//! when a `--limit-vram-total` ceiling is published and the card's own
//! DEVICE_LOCAL heap otherwise. A tight ceiling, a very large token count, or
//! CFG-parallel needing a resident set on BOTH cards therefore produces FEWER
//! slots rather than an out-of-memory abort - and zero slots degrades to
//! exactly the pre-residency behaviour, one upload per block per call, which
//! is still the reference definition of the math.
//!
//! `BRAIN_LTXV_RESIDENT_BLOCKS` overrides the computed count (`0` disables
//! residency entirely) - the bisect handle a measurement pass needs, and the
//! way the bit-identity gate runs both arms of the same generation.

use crate::block::{cached_av_block_bytes, cached_block_bytes, BlockTimings, CachedQAvBlockWeights, CachedQBlockWeights, LtxAvBlockQ, LtxBlockQ, QTier, KERNELS};
use crate::config::{LtxAvDitConfig, LtxDitConfig};
use gpu_core::Gpu;
use std::sync::Mutex;
use weightset::{CyclicScan, GroupId, Schedule, WeightSet};

/// How many passes over the block list the resident window plans against.
///
/// The cursor is taken modulo the layer count, so a lookahead window of one
/// full extra pass is all Bélády ever needs to see: from any position inside
/// pass 0 the schedule's remaining `[cursor, 2*n)` range contains every group
/// exactly once more, which is precisely the "furthest next use" ordering a
/// cyclic scan has. Planning against the REAL step count instead would build a
/// schedule up to 16x longer to compute the same answer, and would need a step
/// count `forward_q_streamed` does not have.
const PLAN_PASSES: u32 = 2;

/// Bytes of device memory a streamed forward needs for everything that is NOT
/// resident block weights, at `t` video tokens.
///
/// Measured on a 24 GiB Tesla P40 (wgpu/Vulkan) with the real
/// `ltx-2.5-22b-distilled-transformer-Q8_0.gguf`, all 48 layers, int8. Peak
/// VRAM, sampled outside any timed region, at the token count a real
/// generation actually runs at (T=12320, ctx 1024) - reproduce with
/// `ltxv_bench streamed 48 12320 1024 1 1 1` under
/// `BRAIN_LTXV_RESIDENT_BLOCKS=<n>`:
///
/// | resident blocks | resident weights | peak VRAM |
/// |---:|---:|---:|
/// | 0 | 0 | 16833 MiB |
/// | 16 | 4096 MiB | 22318 MiB |
/// | 24 | 6144 MiB | 17766 MiB |
///
/// Read that table twice. It is **not monotone**, and that is the single most
/// important fact about this budget: wgpu's allocator pool is elastic and
/// greedy. Left alone it grows to fill whatever is free; under pressure from
/// long-lived allocations it works in far less. Holding MORE weights resident
/// made the peak go DOWN. So the plateau is never the requirement - the
/// requirement is what the pool shrinks to when the weights are already there,
/// and that number at the real width is a little over 11 GiB of non-weight
/// working set.
///
/// Three consequences, all paid for:
///
/// * The resident window must be filled BEFORE the pool has grown (see
///   `DitSession::prefill`). Filling it lazily, block by block, loses a race
///   against the pool and aborts - measured, at 24009 MiB of a 24576 MiB card.
/// * **A slope fitted at one token count is not a model.** The slope this
///   constant used to carry was fitted at T=3520 and extrapolated to tens of
///   gigabytes at the real width, so `card - reserve` underflowed and the
///   policy declined residency ENTIRELY (`slots=0`) at exactly the shape the
///   whole mechanism exists for - every block's weights crossing the bus on
///   every forward, on the video path as well as the audio+video one. The fit
///   below is taken across a sweep instead, and the sweep is what the table
///   above is.
/// * The largest single transient the old fit was sized around - `attn2`'s
///   materialized `[heads, t, context_len]` score+probability pair - **no
///   longer exists**: text cross-attention is a fused online-softmax kernel
///   now and allocates no slab at all. What remains is `[t, dim]` and
///   `[t, 4*dim]` activation buffers, plus the generation-lifetime RoPE
///   tables ([`RopeCache`]).
///
/// Still deliberately generous: under-reserving costs a driver-level abort,
/// over-reserving costs a few resident blocks and the graceful partial-window
/// path picks up the difference.
pub fn activation_reserve_bytes(t: usize, backend: &str) -> u64 {
    /// Everything that does not follow the token count: the head tensors, the
    /// connector's working set, the allocator's own floor.
    const BASE: u64 = 3 << 29; // 1.5 GiB
    /// Measured slope for the DEFAULT wgpu backend, bytes per video token.
    ///
    /// Fitted to the UNDER-PRESSURE working set in the table above (about
    /// 11.6 GiB of non-weight peak at T=12320 with the window full), plus
    /// roughly half again as margin, spread over the token count: at T=12320
    /// it reserves ~17.0 GiB of a 24576 MiB card, which leaves more than the
    /// [`MAX_CARD_FRACTION_DENOM`] cap allows and so lets that cap - the
    /// VAE-decode headroom rule, which is the constraint that actually
    /// matters on this card - be the binding one at every width a real
    /// generation uses. Past roughly 13000 tokens the reserve takes over again
    /// and the window shrinks smoothly rather than falling off a cliff to
    /// zero, which is what the old fit did.
    const PER_TOKEN_WGPU: u64 = 1321 * 1024;
    /// brain's own native Vulkan backend recycles transient buffers, uniforms
    /// and descriptor sets explicitly at every flush (`crates/vulkan`'s
    /// `VkContext` reclaim path) instead of growing an opportunistic pool, and
    /// it does not carry the doubled per-uploaded-buffer resident cost wgpu
    /// measured (`crates/gpu-core/tests/vram_overhead.rs`). Its churn is
    /// therefore close to the working set rather than to the card.
    ///
    /// DERIVED, not measured - there is no Vulkan plateau to fit, and this
    /// crate has no real-weight Vulkan run on this box. The derivation is the
    /// analytic per-block working set (about twenty `[t, dim]` buffers plus
    /// the FFN's `[t, 4*dim]` pair, double-buffered across a flush, with no
    /// wgpu doubling and no `attn2` slab any more), rounded generously up.
    /// Kept strictly below the wgpu slope, which is the one thing about it
    /// that IS certain: a backend that reclaims explicitly cannot need more
    /// headroom than one that does not.
    const PER_TOKEN_VULKAN: u64 = 800 * 1024;
    let per_token = match backend {
        "vulkan" => PER_TOKEN_VULKAN,
        _ => PER_TOKEN_WGPU,
    };
    BASE + per_token * t as u64
}

/// Bytes this process may still put on `device`.
///
/// With no `--limit-vram-total` ceiling published, the card's own largest
/// DEVICE_LOCAL heap is the honest bound and each card is independent.
///
/// With a ceiling published it is the authority's real headroom, **divided by
/// the number of schedulable cards**, and that division is not conservatism for
/// its own sake. `--limit-vram-total` is a process-wide TOTAL across every
/// card, and the two concurrent CFG branches build their sessions at the SAME
/// moment on two threads: without the division both would read the full
/// headroom, both would plan a full resident set, and the second one to
/// actually allocate would be refused mid-upload - a panic, which is exactly
/// the unhandled failure a ceiling exists to prevent. Dividing makes the two
/// plans fit together by construction, before either allocates anything.
fn usable_vram(device: memauth::Device) -> u64 {
    if let Some(auth) = memauth::authority() {
        auth.refresh_now();
        let cards = gpu_core::devices::ambient_compute_set().gpus.len().max(1) as u64;
        return auth.headroom(device) / cards;
    }
    match device {
        memauth::Device::Gpu(i) => gpu_core::devices::device(i).map(|d| d.identity.vram_bytes).unwrap_or(0),
        // The CPU backend's "device memory" is host RAM, and a resident window
        // there buys nothing (there is no upload to skip). Reported as zero so
        // the policy declines residency rather than reasoning about host RAM
        // with a VRAM formula.
        _ => 0,
    }
}

/// Whether one more resident block can be charged against a published ceiling
/// right now.
///
/// The backstop under [`usable_vram`]'s static division: a run can hold things
/// this module never planned for (another model resident on the same card, a
/// VAE decode that has not released yet), so the last check before a ~270 MB
/// weight upload is against the ceiling's LIVE headroom, not against the plan.
/// A refusal degrades that one block to per-call streaming - traced, and still
/// bit-identical - instead of letting `Gpu::storage`'s infallible facade panic
/// mid-upload. `true` whenever no ceiling is published, which is the default
/// and costs one atomic load.
fn can_charge_a_block(device: memauth::Device, per_block: u64) -> bool {
    let Some(auth) = memauth::authority() else { return true };
    auth.refresh_now();
    // One block of margin: two threads may check before either allocates, and
    // the loser must still find room for the block it was promised.
    auth.headroom(device) >= per_block.saturating_mul(2)
}

/// `BRAIN_LTXV_RESIDENT_BLOCKS`, when set to something parseable.
fn slots_override() -> Option<u32> {
    let raw = std::env::var("BRAIN_LTXV_RESIDENT_BLOCKS").ok()?;
    match raw.trim().parse::<u32>() {
        Ok(v) => Some(v),
        Err(_) => {
            tracing::warn!(value = %raw, "BRAIN_LTXV_RESIDENT_BLOCKS is not a number; ignoring");
            None
        }
    }
}

/// How many of `cfg.num_layers` blocks fit on `device` alongside one forward's
/// activations - the whole residency policy, in one closed form over numbers
/// that are each either measured (`cached_block_bytes`,
/// [`activation_reserve_bytes`]) or read from the machine (the heap size, the
/// published ceiling's headroom).
pub fn planned_slots(cfg: &LtxDitConfig, tier: QTier, t: usize, device: memauth::Device, backend: &str) -> u32 {
    slots_for(cached_block_bytes(cfg, tier), cfg.num_layers, t, device, backend)
}

/// [`planned_slots`] over an already-known per-block footprint and layer
/// count - the whole policy, shared verbatim by the video-only and the
/// audio+video sessions. The AV block is a different SIZE, never a different
/// rule, so it gets this function rather than a second copy of the
/// arithmetic.
pub fn slots_for(per_block: u64, num_layers: u32, t: usize, device: memauth::Device, backend: &str) -> u32 {
    if let Some(n) = slots_override() {
        return n.min(num_layers);
    }
    if per_block == 0 {
        return 0;
    }
    let card = usable_vram(device);
    let by_reserve = card.saturating_sub(activation_reserve_bytes(t, backend)) / per_block;
    // A generation is not just its denoise loop, and the denoise loop's own
    // activation reserve does not see the rest of it. `pipeline::generate`
    // runs the Gemma-4 text encode BEFORE and the VAE decode AFTER, each on
    // its own `Gpu`, and a fresh wgpu device cannot reuse the pool a dropped
    // one left behind - so weights this loop held are not usefully free to the
    // next stage even though they have been released.
    //
    // Found the hard way rather than reasoned: at a small token count the
    // reserve above is tiny, the policy granted all 48 blocks (~13 GB), the
    // denoise loop finished, and the VAE decode's own device then aborted with
    // `wgpu error: Out of Memory` at 24211 MiB of a 24576 MiB card
    // (`crates/ltxv/tests/cfg_parallel.rs`, 9 frames at 64x64 - a shape with
    // no memory problem of its own at all). The VAE decode alone needs up to
    // ~16.5 GiB at the shapes this pipeline supports.
    //
    // So the window never takes more than this fraction of the card, whatever
    // the token count says. It costs resident blocks at small shapes, where
    // the forward is cheap anyway; it buys a generation that finishes.
    const MAX_CARD_FRACTION_DENOM: u64 = 4;
    let by_cap = card / MAX_CARD_FRACTION_DENOM / per_block;
    (by_reserve.min(by_cap) as u32).min(num_layers)
}

/// Device-resident RoPE tables, kept for as long as the session that built
/// them.
///
/// A generation's RoPE tables are a pure function of `(config, positions, t)`,
/// and a denoise loop holds all three fixed across every one of its steps -
/// yet both streamed forwards rebuilt them on the HOST in f64 and re-uploaded
/// them on EVERY call. At a real generation's token count that build is a
/// visible share of a warm forward's wall clock (it is its own
/// `BRAIN_PROFILE` stage, `RoPE table build (host, f64)`), and the upload is
/// `heads * t * head_dim/2` floats twice over, per table, per forward - the
/// audio+video path has FOUR such tables.
///
/// Keyed on a hash of everything the tables are a function of, never on "the
/// caller says it is the same": a long-form window, a refinement pass or a
/// second scene changes `positions` while leaving every dimension alone, and
/// silently reusing another shape's rotation is the kind of defect that
/// produces plausible video. `f32::to_bits` rather than `==`, for the reason
/// `dit::adaln`'s own dedup uses bits: `0.0 == -0.0` is true for two different
/// inputs.
///
/// Only a RESIDENT session caches. A transient one opens a fresh device per
/// forward, so buffers from the previous call belong to a device that no
/// longer exists.
/// One cached RoPE table set: the shape key it was built for, then the cos
/// and sin tables. Named rather than inline because the tuple is the same
/// three things every call site destructures.
type RopeSlot = (u64, Vec<gpu_core::DeviceBuffer>, Vec<gpu_core::DeviceBuffer>);

#[derive(Default)]
struct RopeCache {
    slots: Mutex<Vec<RopeSlot>>,
}

/// How many distinct RoPE table sets one session remembers. The audio+video
/// forward uses four (each stream's self-attention table plus each stream's
/// cross-modal one); the video-only forward uses one. Eight leaves room for a
/// second shape - the boundary between two long-form windows - without letting
/// a long-running process accumulate a table per window forever.
const MAX_ROPE_SLOTS: usize = 8;

impl RopeCache {
    /// This key's tables, built and uploaded on first use. `resident` is the
    /// session's own `gpu.is_some()`: `false` builds every time and stores
    /// nothing.
    fn get(&self, resident: bool, key: u64, build: &mut dyn FnMut() -> (Vec<gpu_core::DeviceBuffer>, Vec<gpu_core::DeviceBuffer>)) -> (Vec<gpu_core::DeviceBuffer>, Vec<gpu_core::DeviceBuffer>) {
        if !resident {
            return build();
        }
        let mut slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, c, s)) = slots.iter().find(|(k, _, _)| *k == key) {
            // A `DeviceBuffer` clone is an `Arc` bump onto the SAME
            // allocation, so this hands back the identical device bytes, not
            // a copy of them.
            return (c.clone(), s.clone());
        }
        let (c, s) = build();
        if slots.len() >= MAX_ROPE_SLOTS {
            slots.remove(0);
        }
        slots.push((key, c.clone(), s.clone()));
        (c, s)
    }
}

/// A [`RopeCache`] key over everything a table is a function of - the geometry
/// AND the positions, by BITS.
pub fn rope_key(tag: &str, inner_dim: u32, heads: u32, theta: f64, max_pos: &[u32], positions: &[f32], t: usize) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    tag.hash(&mut h);
    (inner_dim, heads, t).hash(&mut h);
    theta.to_bits().hash(&mut h);
    max_pos.hash(&mut h);
    for p in positions {
        p.to_bits().hash(&mut h);
    }
    h.finish()
}

/// The shape a [`BlockWindow`]'s uploaded blocks are valid for. A generation
/// holds `t`/`context_len` fixed across every one of its steps; a caller that
/// changes either (`crate::dfr`'s multi-stage pipeline is the shape that
/// could) gets the window rebuilt rather than a block asserting on a
/// mismatched context width.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct WindowShape {
    t: u32,
    context_len: u32,
    tier: QTier,
    num_layers: u32,
}

/// What a caller wants to know about a window's behaviour, without reaching
/// into it. `uploads` is the number this design exists to drive down: with
/// every block resident it is `num_layers` for the whole generation, not
/// `num_layers` per forward.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidencyStats {
    /// Slots this window was built with.
    pub slots: u32,
    /// Block visits served straight from a resident slot.
    pub hits: u64,
    /// Block visits that had to (re-)upload weights to the device.
    pub uploads: u64,
    /// Block visits a published `--limit-vram-total` ceiling refused a resident
    /// slot for, and which streamed instead. Nonzero means the run degraded
    /// gracefully rather than aborting; always 0 with no ceiling published.
    pub refusals: u64,
}

/// A fixed window of device slots over one model's blocks, scheduled by
/// [`weightset`].
///
/// Generic in the BLOCK type: the video-only stack keeps
/// [`crate::block::LtxBlockQ`]s and the audio+video stack keeps
/// [`crate::block::LtxAvBlockQ`]s, but the slot table, the Bélády plan, the
/// upload/hit bookkeeping and the budget backstop are the same mechanism and
/// are written once. Everything below this line that is NOT generic is the
/// per-stack forward loop, which genuinely differs (one stream vs two).
struct BlockWindow<B> {
    shape: WindowShape,
    /// Slot `i`'s current occupant: which layer it holds and the uploaded
    /// block. Each entry owns its own `Gpu` handle, so `None`-ing an entry
    /// releases that slot's VRAM and its `memauth` grants together.
    slots: Vec<Option<(u32, B)>>,
    ws: WeightSet,
    hits: u64,
    uploads: u64,
    /// Times the live `memauth` headroom refused one more resident block and
    /// it was streamed instead - see [`can_charge_a_block`]. Always 0 with no
    /// ceiling published.
    refusals: u64,
}

impl<B> BlockWindow<B> {
    fn build(shape: WindowShape, slots: u32) -> Option<BlockWindow<B>> {
        if slots == 0 {
            return None;
        }
        let sched = Schedule::cyclic(shape.num_layers, PLAN_PASSES);
        let ws = match WeightSet::build(shape.num_layers, slots, sched, Box::new(CyclicScan { lookahead: 1 })) {
            Ok(ws) => ws,
            Err(e) => {
                tracing::warn!(error = %e, "device residency window declined; falling back to per-call block upload");
                return None;
            }
        };
        Some(BlockWindow { shape, slots: (0..slots).map(|_| None).collect(), ws, hits: 0, uploads: 0, refusals: 0 })
    }

    fn stats(&self) -> ResidencyStats {
        ResidencyStats { slots: self.slots.len() as u32, hits: self.hits, uploads: self.uploads, refusals: self.refusals }
    }

    /// The slots this window has planned but not yet filled - what a pre-fill
    /// loads, in plan order. See [`DitSession::prefill`] for why the ORDER
    /// (before anything else in the forward touches the device) is the whole
    /// finding.
    fn pending_pins(&self) -> Vec<(usize, u32)> {
        self.ws.slot_contents().iter().enumerate().filter_map(|(i, g)| g.map(|g| (i, g.0))).filter(|(i, _)| self.slots[*i].is_none()).collect()
    }

    /// Make layer `l`'s weights resident and say which slot (if any) holds
    /// them, WITHOUT keeping a borrow alive across the forward that follows.
    ///
    /// Returns `(slot, uploaded, upload_time)`; `None` means "run this block
    /// streamed" - either there is no window at all or a published ceiling
    /// refused one more resident block right now. One implementation, shared
    /// by both stacks, because every subtlety in it was paid for once:
    /// `plan_miss` alone is not enough (a pinned prefix reports resident
    /// before anything has been uploaded into it), and the outgoing occupant
    /// must drop BEFORE the incoming one allocates or a full window's peak is
    /// `slots + 1` blocks.
    fn acquire(&mut self, l: usize, per_block: u64, device: memauth::Device, upload: impl FnOnce() -> B) -> (Option<usize>, bool, std::time::Duration) {
        let (slot, plan_miss) = self.ws.advance(l);
        debug_assert_eq!(self.ws.schedule().order[l], GroupId(l as u32), "the window's schedule must visit blocks in the forward's own order");
        let idx = slot.0 as usize;
        let occupied_by_this_layer = self.slots[idx].as_ref().map(|(g, _)| *g == l as u32).unwrap_or(false);
        if occupied_by_this_layer && !plan_miss {
            self.hits += 1;
            return (Some(idx), false, std::time::Duration::ZERO);
        }
        if !can_charge_a_block(device, per_block) {
            self.slots[idx] = None;
            self.refusals += 1;
            tracing::warn!(layer = l, per_block, "no VRAM budget headroom for a resident block; streaming this one instead");
            return (None, true, std::time::Duration::ZERO);
        }
        let s = std::time::Instant::now();
        self.slots[idx] = None;
        self.slots[idx] = Some((l as u32, upload()));
        self.uploads += 1;
        (Some(idx), true, s.elapsed())
    }
}

/// One card, for one generation: an open device plus whatever weights are
/// resident on it. See this module's doc for the lifecycle.
pub struct DitSession {
    /// Opened once. `None` when this session is the transient shape (a fresh
    /// device per forward, the pre-residency behaviour).
    gpu: Option<Gpu>,
    device: Option<String>,
    /// `Mutex` rather than `&mut self`, so `crate::pipeline::RealDit` can keep
    /// handing `&self` to `Denoiser::forward` and stay `Sync` for the
    /// concurrent two-card CFG dispatch. One card's session is used by exactly
    /// one thread at a time (the other branch has its OWN session on the other
    /// card), so this is never contended.
    window: Mutex<Option<BlockWindow<LtxBlockQ>>>,
    /// Slots this session was asked to plan for - `0` means residency is off
    /// and every forward opens its own device, exactly as before.
    slots: u32,
    /// This generation's RoPE tables, built and uploaded once - see
    /// [`RopeCache`].
    rope: RopeCache,
}

impl DitSession {
    /// A session that keeps nothing: every forward opens its own `Gpu` and
    /// uploads every block. The behaviour this crate shipped before device
    /// residency existed, kept as the fallback and as the arm a bit-identity
    /// gate compares against.
    pub fn transient(device: Option<&str>) -> DitSession {
        DitSession { gpu: None, device: device.map(str::to_string), window: Mutex::new(None), slots: 0, rope: RopeCache::default() }
    }

    /// A session that holds its device open and keeps up to
    /// [`planned_slots`]-many blocks resident on it.
    ///
    /// `t` sizes the activation reserve, so the same card gives a 720p run
    /// more resident blocks than a 1080p one - which is the correct direction,
    /// since the 1080p run's activations are what would otherwise collide with
    /// them.
    pub fn resident(cfg: &LtxDitConfig, tier: QTier, device: Option<&str>, t: usize) -> DitSession {
        let gpu = Gpu::open(device, &KERNELS);
        let slots = planned_slots(cfg, tier, t, gpu.memory_device(), gpu.kind());
        tracing::info!(
            slots,
            layers = cfg.num_layers,
            backend = gpu.kind(),
            block_mb = cached_block_bytes(cfg, tier) / (1 << 20),
            reserve_mb = activation_reserve_bytes(t, gpu.kind()) / (1 << 20),
            device = ?gpu.memory_device(),
            "device residency planned"
        );
        DitSession { gpu: Some(gpu), device: device.map(str::to_string), window: Mutex::new(None), slots, rope: RopeCache::default() }
    }

    /// [`Self::resident`] at an EXPLICIT slot count, bypassing the VRAM
    /// policy.
    ///
    /// What a gate uses to drive the narrow-window (partial residency) path
    /// deliberately, at a config small enough to run in milliseconds, rather
    /// than by starving a real card until the policy happens to pick the shape
    /// under test. `slots = 0` is the no-window fallback, which is a legal
    /// answer here rather than an error.
    pub fn resident_with_slots(device: Option<&str>, slots: u32) -> DitSession {
        DitSession { gpu: Some(Gpu::open(device, &KERNELS)), device: device.map(str::to_string), window: Mutex::new(None), slots, rope: RopeCache::default() }
    }

    /// This session's device handle for one forward call, plus whether it is
    /// long-lived. A transient session opens (and, on return, drops) a fresh
    /// one; a resident session hands back a `Gpu::share` of the one it holds,
    /// which is the same adapter, queue and compiled pipelines.
    pub fn device_for_call(&self) -> Gpu {
        match &self.gpu {
            Some(g) => g.share(),
            None => Gpu::open(self.device.as_deref(), &KERNELS),
        }
    }

    /// True when this session holds its device open across calls.
    pub fn is_resident(&self) -> bool {
        self.gpu.is_some()
    }

    /// This session's device-resident RoPE tables for `key`, built and
    /// uploaded on first use and reused by every later forward of the same
    /// generation. See [`RopeCache`] for the key's contract.
    pub fn rope_tables(&self, key: u64, build: &mut dyn FnMut() -> (Vec<gpu_core::DeviceBuffer>, Vec<gpu_core::DeviceBuffer>)) -> (Vec<gpu_core::DeviceBuffer>, Vec<gpu_core::DeviceBuffer>) {
        self.rope.get(self.gpu.is_some(), key, build)
    }

    pub fn stats(&self) -> ResidencyStats {
        self.window.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map(|w| w.stats()).unwrap_or_default()
    }

    /// Build (or rebuild) this session's resident window and fill its pinned
    /// slots - **before anything else in the forward touches the device**.
    ///
    /// The ORDER is the whole finding here, not a tidiness preference.
    /// `weightset`'s own `slot_contents` doc says a caller must load the
    /// initial pins; what a measurement added is WHEN. wgpu's allocator pool
    /// is elastic but grows GREEDILY and hands nothing back: at the real
    /// 22B/720p shape, filling the window after the embeddings connector had
    /// already run reached **24368 MiB of a 24576 MiB card** and aborted with
    /// `wgpu error: Out of Memory`, and filling it lazily block by block
    /// reached 24009 MiB and aborted at block 28 - while the same forward with
    /// zero resident blocks plateaus at 16522 MiB. Long-lived weights taken
    /// FIRST leave the pool to size itself against what remains; taken later
    /// they lose a race against it.
    ///
    /// Idempotent: a second call on a warm session with an unchanged shape
    /// finds every pinned slot occupied and does nothing.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill(&self, scratch: &Gpu, cfg: &LtxDitConfig, tier: QTier, t: usize, context_len: usize, weights: &mut dyn FnMut(usize) -> std::sync::Arc<CachedQBlockWeights>) {
        if self.slots == 0 {
            return;
        }
        let shape = WindowShape { t: t as u32, context_len: context_len as u32, tier, num_layers: cfg.num_layers };
        let per_block = cached_block_bytes(cfg, tier);
        let mut guard = self.window.lock().unwrap_or_else(|e| e.into_inner());
        // A window is only valid for the shape its blocks were built at.
        // Rebuild (releasing the old slots first) rather than assert.
        if guard.as_ref().map(|w| w.shape != shape).unwrap_or(true) {
            if guard.is_some() {
                tracing::debug!(?shape, "device residency window rebuilt for a new forward shape");
            }
            *guard = None;
            *guard = BlockWindow::build(shape, self.slots);
        }
        let Some(w) = guard.as_mut() else { return };
        let pins = w.pending_pins();
        if pins.is_empty() {
            return;
        }
        let s = std::time::Instant::now();
        // A one-word probe to drain wgpu's `write_buffer` staging after every
        // block - see `LtxBlockQ::forward_prod_dev` for the measured doubling
        // this exists to stop accruing. Without it, uploading 48 blocks of
        // 270 MB reaches 24392 MiB of a 24576 MiB card and aborts.
        let probe = scratch.storage(1);
        for (idx, g) in pins {
            if !can_charge_a_block(scratch.memory_device(), per_block) {
                tracing::warn!(slot = idx, "no VRAM budget headroom while pre-filling the resident window; the rest will stream");
                w.refusals += 1;
                break;
            }
            let cached = weights(g as usize);
            w.slots[idx] = Some((g, LtxBlockQ::on_cached(scratch.share(), cfg, &cached, t as u32, context_len as u32, tier)));
            w.uploads += 1;
            let _ = scratch.read(&probe, 1);
        }
        tracing::info!(resident = w.slots.iter().filter(|s| s.is_some()).count(), slots = w.slots.len(), ms = s.elapsed().as_secs_f32() * 1e3, "resident weight window pre-filled");
    }

    /// Run one forward's whole block stack, entirely on the device.
    ///
    /// `x` in as host floats, `x` out as host floats, and NOTHING in between
    /// crosses PCIe that does not have to:
    ///
    /// * `x` is uploaded ONCE, chained block to block as a device buffer, and
    ///   read back once - not uploaded and read back 48 times;
    /// * the text context is uploaded once, not once per block;
    /// * the model-level adaLN table is uploaded once, and each block's own
    ///   nine `[t, dim]` modulation vectors are derived FROM it on the card
    ///   (`crate::block::LtxBlockQ::forward_prod_dev`) instead of being combined
    ///   and sliced on the host and uploaded per block;
    /// * the three parity taps every block used to read back, and which a
    ///   production forward discards, are never read at all.
    ///
    /// Returns the stack's output plus the FORWARD-level costs (the three
    /// per-forward uploads and the single readback); the per-BLOCK costs go to
    /// `after_block`.
    ///
    /// `weights(layer)` is called ONLY for a layer whose bytes are not already
    /// on the card, which is what turns the host-RAM weight cache's `Arc` clone
    /// plus a ~270 MB device upload into work paid once per generation instead
    /// of once per step. `after_block` gets each layer's index, whether it had
    /// to upload, and the split of where its time went.
    ///
    /// A session with no window runs the IDENTICAL loop with a freshly
    /// uploaded block per layer, dropped immediately - one block-stack
    /// implementation, not one per residency mode.
    #[allow(clippy::too_many_arguments)]
    pub fn run_blocks(
        &self,
        scratch: &Gpu,
        cfg: &LtxDitConfig,
        tier: QTier,
        t: usize,
        context_len: usize,
        x: Vec<f32>,
        adaln_table: &dit::adaln::RowTable,
        connector_context: &[f32],
        cos_bufs: &[gpu_core::DeviceBuffer],
        sin_bufs: &[gpu_core::DeviceBuffer],
        weights: &mut dyn FnMut(usize) -> std::sync::Arc<CachedQBlockWeights>,
        after_block: &mut dyn FnMut(usize, bool, std::time::Duration, &BlockTimings),
    ) -> (Vec<f32>, BlockTimings) {
        let per_block = cached_block_bytes(cfg, tier);
        let mut guard = self.window.lock().unwrap_or_else(|e| e.into_inner());

        // The three per-forward uploads, in place of ~26 GB of per-block ones:
        // the model-level adaLN table and the row map that says which of its
        // rows each token uses (each block derives its own nine modulation
        // vectors FROM the pair on the card), and the text context.
        //
        // The table is one row per DISTINCT token timestep, not one per token
        // (`dit::adaln::RowTable`), so at a real step it is 1-2 rows - 147 KB
        // where the dense `[t, 9*dim]` form was 519 MB at T=3520 and 1.2 GB at
        // T=8160. The map is `t` u32s (14 KB at T=3520).
        //
        // Checked, not assumed: `adaln_row` indexes `map[r]` for every one of
        // the `t` token rows it writes, so a map shorter than `t` is an
        // out-of-bounds device read - which `backend-cpu` does not bounds-check
        // at dispatch at all, so it would return plausible numbers rather than
        // fail.
        assert_eq!(adaln_table.len(), t, "DitSession::run_blocks: the adaLN row map covers {} tokens, the forward has {t}", adaln_table.len());
        assert_eq!(adaln_table.width(), cfg.adaln_rows() as usize * cfg.inner_dim as usize, "DitSession::run_blocks: the adaLN table is the wrong width for this config");
        let s_up = std::time::Instant::now();
        // `x` onto the card ONCE, for the whole stack. Every block hands the
        // next one a device buffer (`LtxBlockQ::forward_prod_dev`), so the
        // activation crosses PCIe twice per FORWARD instead of twice per
        // BLOCK - see that method's doc for why the per-block round trip was
        // there and what replaced the one thing it bought.
        let x_len = x.len();
        let mut x_dev = scratch.storage(x_len as u64);
        scratch.write_f32_chunked(&x_dev, &x, 1 << 20);
        let adaln_buf = scratch.storage(adaln_table.distinct().len() as u64);
        // CHUNKED, not one `write_f32`: a table with as many distinct rows as
        // tokens is back to the dense size, and on a non-ReBAR card one giant
        // `write_buffer` measured more than an order of magnitude slower than
        // the chunked form at 519 MB, because the staging allocation it needs
        // is the same size as the payload. Same chunk size `paramstore`'s own weight-upload loop uses,
        // for the same reason.
        scratch.write_f32_chunked(&adaln_buf, adaln_table.distinct(), 1 << 20);
        let adaln_map_buf = scratch.storage(adaln_table.row_of().len() as u64);
        scratch.write_at(&adaln_map_buf, 0, adaln_table.row_of());
        let ctx_buf = scratch.storage(connector_context.len() as u64);
        scratch.write_f32(&ctx_buf, connector_context);
        let forward_upload = s_up.elapsed();

        for l in 0..cfg.num_layers as usize {
            // Ensure this layer's weights are on the card, and learn which slot
            // (if any) holds them, WITHOUT keeping a borrow of the window alive
            // across the forward below.
            // The schedule is cyclic with period `num_layers`, so the layer
            // index IS the plan cursor: from position `l` the remaining
            // `[l, 2n)` range holds every group exactly once more, which is
            // the whole lookahead Bélády needs. Deriving it from `l` rather
            // than from a running counter also means an aborted forward
            // cannot leave the window's cursor out of step with the stack.
            let (slot_idx, uploaded, mut up_ms) = match guard.as_mut() {
                Some(w) => w.acquire(l, per_block, scratch.memory_device(), || LtxBlockQ::on_cached(scratch.share(), cfg, &weights(l), t as u32, context_len as u32, tier)),
                None => (None, true, std::time::Duration::ZERO),
            };

            let mut bt = BlockTimings::default();
            x_dev = match slot_idx {
                Some(idx) => {
                    let w = guard.as_ref().expect("a slot index implies a window");
                    let blk = &w.slots[idx].as_ref().expect("the slot was just filled").1;
                    blk.forward_prod_dev(scratch, &x_dev, &adaln_buf, &adaln_map_buf, &ctx_buf, cos_bufs, sin_bufs, t as u32, &mut bt)
                }
                None => {
                    // No window (or a slot the budget refused): upload, run,
                    // drop - the pre-residency shape, on the same production
                    // block forward, so there is one implementation and not
                    // one per residency mode.
                    //
                    // TIMED, and that is not cosmetic: this upload used to be
                    // the one cost the profile could not see. `up_ms` was
                    // hardcoded to zero on this arm, so a run that had been
                    // DENIED residency - which, at the token count a real
                    // generation uses, was every run - reported its
                    // `block weight upload` stage as exactly zero while doing
                    // nothing but block weight uploads. A stage that reads
                    // zero exactly when it is the bottleneck is worse than no
                    // stage at all.
                    let s = std::time::Instant::now();
                    let cached = weights(l);
                    let blk = LtxBlockQ::on_cached(scratch.share(), cfg, &cached, t as u32, context_len as u32, tier);
                    up_ms += s.elapsed();
                    blk.forward_prod_dev(scratch, &x_dev, &adaln_buf, &adaln_map_buf, &ctx_buf, cos_bufs, sin_bufs, t as u32, &mut bt)
                }
            };
            after_block(l, uploaded, up_ms, &bt);
        }
        let s_back = std::time::Instant::now();
        let x = scratch.read(&x_dev, x_len);
        (x, BlockTimings { record_upload: forward_upload, readback: s_back.elapsed(), ..Default::default() })
    }
}

// ---------------------------------------------------------------------------
// The audio+video session - the same lifecycle, the same [`BlockWindow`], the
// same Bélády plan and the same budget, over [`LtxAvBlockQ`] instead of
// [`LtxBlockQ`].
//
// Only two things genuinely differ, and both are consequences of there being
// two streams rather than one: an AV block is bigger (28 quantized linears
// instead of 10 - `crate::block::cached_av_block_bytes`), so the same card
// affords fewer slots; and one forward uploads four model-level modulation
// tables and two text contexts instead of two and one. Everything else is
// literally the code above.
// ---------------------------------------------------------------------------

/// Everything one AV forward's block stack needs that is not a block weight -
/// the per-FORWARD inputs, uploaded once by [`AvDitSession::run_blocks`] and
/// then read by every block off the card.
///
/// The four tables are [`dit::adaln::RowTable`]s, i.e. one row per DISTINCT
/// per-token timestep rather than one per token, which is the whole reason
/// they are cheap to upload: a plain step has one distinct timestep and an
/// anchored or long-form one has two, however many thousand tokens there are.
/// The two gate rows are single rows by construction (each is driven by the
/// OTHER modality's SCALAR sigma).
pub struct AvForwardInputs<'a> {
    pub v_adaln: &'a dit::adaln::RowTable,
    pub a_adaln: &'a dit::adaln::RowTable,
    pub av_v_ss: &'a dit::adaln::RowTable,
    pub av_a_ss: &'a dit::adaln::RowTable,
    pub av_a2v_gate: &'a [f32],
    pub av_v2a_gate: &'a [f32],
    pub v_context: &'a [f32],
    pub a_context: &'a [f32],
    pub rope: crate::block::AvRope<'a>,
    pub tv: usize,
    pub ta: usize,
    pub v_context_len: usize,
    pub a_context_len: usize,
}

/// One card, for one audio+video generation - [`DitSession`]'s AV twin. See
/// this module's doc for the lifecycle, which is identical.
pub struct AvDitSession {
    gpu: Option<Gpu>,
    device: Option<String>,
    window: Mutex<Option<BlockWindow<LtxAvBlockQ>>>,
    slots: u32,
    rope: RopeCache,
}

impl AvDitSession {
    /// A session that keeps nothing: a fresh `Gpu` per forward and every
    /// block re-uploaded. The fallback, and the arm a bit-identity gate
    /// compares a resident run against.
    pub fn transient(device: Option<&str>) -> AvDitSession {
        AvDitSession { gpu: None, device: device.map(str::to_string), window: Mutex::new(None), slots: 0, rope: RopeCache::default() }
    }

    /// A session that holds its device open and keeps up to
    /// [`slots_for`]-many AV blocks resident on it.
    pub fn resident(cfg: &LtxAvDitConfig, tier: QTier, device: Option<&str>, t: usize) -> AvDitSession {
        let gpu = Gpu::open(device, &KERNELS);
        let per_block = cached_av_block_bytes(&cfg.video, &cfg.audio, tier);
        let slots = slots_for(per_block, cfg.video.num_layers, t, gpu.memory_device(), gpu.kind());
        tracing::info!(
            slots,
            layers = cfg.video.num_layers,
            backend = gpu.kind(),
            block_mb = per_block / (1 << 20),
            reserve_mb = activation_reserve_bytes(t, gpu.kind()) / (1 << 20),
            device = ?gpu.memory_device(),
            "audio+video device residency planned"
        );
        AvDitSession { gpu: Some(gpu), device: device.map(str::to_string), window: Mutex::new(None), slots, rope: RopeCache::default() }
    }

    /// [`Self::resident`] at an EXPLICIT slot count, bypassing the VRAM
    /// policy - what a gate uses to drive the partial-residency path
    /// deliberately at a config small enough to run in milliseconds.
    pub fn resident_with_slots(device: Option<&str>, slots: u32) -> AvDitSession {
        AvDitSession { gpu: Some(Gpu::open(device, &KERNELS)), device: device.map(str::to_string), window: Mutex::new(None), slots, rope: RopeCache::default() }
    }

    pub fn device_for_call(&self) -> Gpu {
        match &self.gpu {
            Some(g) => g.share(),
            None => Gpu::open(self.device.as_deref(), &KERNELS),
        }
    }

    pub fn is_resident(&self) -> bool {
        self.gpu.is_some()
    }

    /// This session's device-resident RoPE tables for `key`, built and
    /// uploaded on first use and reused by every later forward of the same
    /// generation. See [`RopeCache`] for the key's contract.
    pub fn rope_tables(&self, key: u64, build: &mut dyn FnMut() -> (Vec<gpu_core::DeviceBuffer>, Vec<gpu_core::DeviceBuffer>)) -> (Vec<gpu_core::DeviceBuffer>, Vec<gpu_core::DeviceBuffer>) {
        self.rope.get(self.gpu.is_some(), key, build)
    }

    pub fn stats(&self) -> ResidencyStats {
        self.window.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map(|w| w.stats()).unwrap_or_default()
    }

    /// Build (or rebuild) this session's resident window and fill its pinned
    /// slots BEFORE anything else in the forward touches the device - see
    /// [`DitSession::prefill`] for the measurement that says the order
    /// matters, which is a property of wgpu's allocator and not of the stream
    /// count.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill(&self, scratch: &Gpu, cfg: &LtxAvDitConfig, tier: QTier, tv: usize, v_context_len: usize, a_context_len: usize, weights: &mut dyn FnMut(usize) -> std::sync::Arc<CachedQAvBlockWeights>) {
        if self.slots == 0 {
            return;
        }
        // `context_len` in the shape key is the VIDEO stream's, plus audio's
        // folded in: a window is only valid for the widths its blocks were
        // built at, and an AV block carries two.
        let shape = WindowShape { t: tv as u32, context_len: (v_context_len as u32) << 16 | a_context_len as u32, tier, num_layers: cfg.video.num_layers };
        let per_block = cached_av_block_bytes(&cfg.video, &cfg.audio, tier);
        let mut guard = self.window.lock().unwrap_or_else(|e| e.into_inner());
        if guard.as_ref().map(|w| w.shape != shape).unwrap_or(true) {
            if guard.is_some() {
                tracing::debug!(?shape, "audio+video residency window rebuilt for a new forward shape");
            }
            *guard = None;
            *guard = BlockWindow::build(shape, self.slots);
        }
        let Some(w) = guard.as_mut() else { return };
        let pins = w.pending_pins();
        if pins.is_empty() {
            return;
        }
        let s = std::time::Instant::now();
        // A one-word probe read after every block, to drain wgpu's
        // `write_buffer` staging - see `DitSession::prefill`.
        let probe = scratch.storage(1);
        for (idx, g) in pins {
            if !can_charge_a_block(scratch.memory_device(), per_block) {
                tracing::warn!(slot = idx, "no VRAM budget headroom while pre-filling the AV resident window; the rest will stream");
                w.refusals += 1;
                break;
            }
            let cached = weights(g as usize);
            w.slots[idx] = Some((g, LtxAvBlockQ::on_cached(scratch.share(), &cfg.video, &cfg.audio, &cached, v_context_len as u32, a_context_len as u32, tier)));
            w.uploads += 1;
            let _ = scratch.read(&probe, 1);
        }
        tracing::info!(resident = w.slots.iter().filter(|s| s.is_some()).count(), slots = w.slots.len(), ms = s.elapsed().as_secs_f32() * 1e3, "AV resident weight window pre-filled");
    }

    /// Run one AV forward's whole block stack. Both streams in as host floats,
    /// both out as host floats, and nothing in between crosses PCIe that does
    /// not have to - see [`DitSession::run_blocks`], which this mirrors
    /// exactly at two streams instead of one.
    #[allow(clippy::too_many_arguments)]
    pub fn run_blocks(
        &self,
        scratch: &Gpu,
        cfg: &LtxAvDitConfig,
        tier: QTier,
        vx: Vec<f32>,
        ax: Vec<f32>,
        inp: &AvForwardInputs,
        weights: &mut dyn FnMut(usize) -> std::sync::Arc<CachedQAvBlockWeights>,
        after_block: &mut dyn FnMut(usize, bool, std::time::Duration, &BlockTimings),
    ) -> (Vec<f32>, Vec<f32>, BlockTimings) {
        let per_block = cached_av_block_bytes(&cfg.video, &cfg.audio, tier);
        let mut guard = self.window.lock().unwrap_or_else(|e| e.into_inner());

        // Checked, not assumed - `adaln_row` indexes `map[r]` for every one of
        // the token rows it writes, and `backend-cpu` does not bounds-check a
        // dispatch, so a short map returns plausible numbers rather than
        // failing.
        assert_eq!(inp.v_adaln.len(), inp.tv, "AvDitSession::run_blocks: the video adaLN row map covers {} tokens, the forward has {}", inp.v_adaln.len(), inp.tv);
        assert_eq!(inp.a_adaln.len(), inp.ta, "AvDitSession::run_blocks: the audio adaLN row map covers {} tokens, the forward has {}", inp.a_adaln.len(), inp.ta);
        assert_eq!(inp.av_v_ss.len(), inp.tv, "AvDitSession::run_blocks: the A<->V video scale/shift map covers {} tokens, the forward has {}", inp.av_v_ss.len(), inp.tv);
        assert_eq!(inp.av_a_ss.len(), inp.ta, "AvDitSession::run_blocks: the A<->V audio scale/shift map covers {} tokens, the forward has {}", inp.av_a_ss.len(), inp.ta);
        assert_eq!(inp.av_a2v_gate.len(), cfg.video.inner_dim as usize, "AvDitSession::run_blocks: the A2V gate row must be [video.dim]");
        assert_eq!(inp.av_v2a_gate.len(), cfg.audio.inner_dim as usize, "AvDitSession::run_blocks: the V2A gate row must be [audio.dim]");

        let s_up = std::time::Instant::now();
        // Both streams onto the card ONCE, for the whole stack - see
        // `DitSession::run_blocks` and `LtxAvBlockQ::forward_prod_dev`.
        let (vx_len, ax_len) = (vx.len(), ax.len());
        let mut vx_dev = scratch.storage(vx_len as u64);
        scratch.write_f32_chunked(&vx_dev, &vx, 1 << 20);
        let mut ax_dev = scratch.storage(ax_len as u64);
        scratch.write_f32_chunked(&ax_dev, &ax, 1 << 20);
        // CHUNKED, for the same reason `DitSession::run_blocks` chunks: a
        // table with as many distinct rows as tokens is back to the dense
        // size, and one giant `write_buffer` on a non-ReBAR card needs a
        // staging allocation the size of the payload.
        let table = |t: &dit::adaln::RowTable| -> (gpu_core::DeviceBuffer, gpu_core::DeviceBuffer) {
            let b = scratch.storage(t.distinct().len() as u64);
            scratch.write_f32_chunked(&b, t.distinct(), 1 << 20);
            let m = scratch.storage(t.row_of().len() as u64);
            scratch.write_at(&m, 0, t.row_of());
            (b, m)
        };
        let (v_adaln_buf, v_adaln_map) = table(inp.v_adaln);
        let (a_adaln_buf, a_adaln_map) = table(inp.a_adaln);
        let (v_ss_buf, v_ss_map) = table(inp.av_v_ss);
        let (a_ss_buf, a_ss_map) = table(inp.av_a_ss);
        let row = |v: &[f32]| -> gpu_core::DeviceBuffer {
            let b = scratch.storage(v.len() as u64);
            scratch.write_f32(&b, v);
            b
        };
        let a2v_gate = row(inp.av_a2v_gate);
        let v2a_gate = row(inp.av_v2a_gate);
        let v_ctx = row(inp.v_context);
        let a_ctx = row(inp.a_context);
        let forward_upload = s_up.elapsed();
        let av = crate::block::AvModelTables { v_ss: &v_ss_buf, v_ss_map: &v_ss_map, a_ss: &a_ss_buf, a_ss_map: &a_ss_map, a2v_gate: &a2v_gate, v2a_gate: &v2a_gate };

        let (tv, ta) = (inp.tv as u32, inp.ta as u32);
        let build = |scratch: &Gpu, c: &CachedQAvBlockWeights| LtxAvBlockQ::on_cached(scratch.share(), &cfg.video, &cfg.audio, c, inp.v_context_len as u32, inp.a_context_len as u32, tier);
        for l in 0..cfg.video.num_layers as usize {
            let (slot_idx, uploaded, mut up_ms) = match guard.as_mut() {
                Some(w) => w.acquire(l, per_block, scratch.memory_device(), || build(scratch, &weights(l))),
                None => (None, true, std::time::Duration::ZERO),
            };
            let mut bt = BlockTimings::default();
            let (v, a) = match slot_idx {
                Some(idx) => {
                    let w = guard.as_ref().expect("a slot index implies a window");
                    let blk = &w.slots[idx].as_ref().expect("the slot was just filled").1;
                    blk.forward_prod_dev(scratch, &vx_dev, &ax_dev, &v_adaln_buf, &v_adaln_map, &a_adaln_buf, &a_adaln_map, &av, &v_ctx, &a_ctx, inp.rope, tv, ta, &mut bt)
                }
                None => {
                    // No window (or a slot the budget refused): upload, run,
                    // drop - on the SAME production block forward, so there is
                    // one implementation and not one per residency mode.
                    // Timed for the reason `DitSession::run_blocks`'s own arm
                    // states: an untimed upload on the arm that does nothing
                    // BUT upload reads as zero exactly when it dominates.
                    let s = std::time::Instant::now();
                    let blk = build(scratch, &weights(l));
                    up_ms += s.elapsed();
                    blk.forward_prod_dev(scratch, &vx_dev, &ax_dev, &v_adaln_buf, &v_adaln_map, &a_adaln_buf, &a_adaln_map, &av, &v_ctx, &a_ctx, inp.rope, tv, ta, &mut bt)
                }
            };
            vx_dev = v;
            ax_dev = a;
            after_block(l, uploaded, up_ms, &bt);
        }
        let s_back = std::time::Instant::now();
        let vx = scratch.read(&vx_dev, vx_len);
        let ax = scratch.read(&ax_dev, ax_len);
        (vx, ax, BlockTimings { record_upload: forward_upload, readback: s_back.elapsed(), ..Default::default() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The policy must be monotone in every input that matters and must never
    /// promise more slots than the model has layers - the two ways a budget
    /// formula silently becomes an out-of-memory abort.
    #[test]
    fn the_slot_policy_is_bounded_by_the_layer_count_and_shrinks_as_tokens_grow() {
        let cfg = LtxDitConfig::ltx25_22b();
        std::env::remove_var("BRAIN_LTXV_RESIDENT_BLOCKS");
        // A synthetic 24 GiB card, via the reserve/per-block arithmetic the
        // policy itself uses - no GPU needed to check the shape of the answer.
        let per_block = cached_block_bytes(&cfg, QTier::Int8);
        assert!(per_block > 0, "a real config must have a nonzero per-block footprint");
        let slots_at = |cap: u64, t: usize| (((cap.saturating_sub(activation_reserve_bytes(t, "wgpu")) / per_block).min(cap / 4 / per_block)) as u32).min(cfg.num_layers);
        let card = 24u64 << 30;
        // Never the whole model, whatever the token count says: the VAE decode
        // that follows the denoise loop needs the rest of the card, and it does
        // not get to reuse this window's freed pool.
        assert!(slots_at(card, 1000) < cfg.num_layers, "the window must never claim the whole card, even at a token count whose own reserve is tiny");
        assert!(slots_at(card, 1000) >= 20, "a small token count must still get a large window");
        assert!(slots_at(card, 3520) < cfg.num_layers, "720p must NOT plan a full window on a 24 GiB card - the measured plateau does not leave room");
        assert!(slots_at(card, 8160) <= slots_at(card, 3520), "a larger token count must never buy MORE resident blocks");
        assert_eq!(slots_at(1 << 30, 3520), 0, "a card smaller than the reserve must ask for zero slots, not a negative count");

        // THE REGRESSION THIS FILE EXISTS TO PREVENT. The previous fit was
        // taken at one token count and extrapolated; at the width a real
        // generation runs at it reserved more than the whole card, `card -
        // reserve` underflowed, and the policy declined residency ENTIRELY -
        // so every block crossed the bus on every forward at exactly the shape
        // the mechanism was built for. A window that is merely "smaller at a
        // bigger shape" is correct; a window that is ZERO at the production
        // shape is the bug.
        for t in [8800usize, 12320, crate::longform::LONGFORM_MAX_TOKENS] {
            assert!(slots_at(card, t) > 0, "the residency policy must plan a nonempty window at {t} tokens, the width a real generation runs at");
        }

        // The native Vulkan backend reclaims transients explicitly and does not
        // double resident bytes, so the SAME card affords strictly more resident
        // blocks there - the budget must not hardcode wgpu's pathology.
        //
        // Compared where the RESERVE binds, not where the card-fraction cap
        // does: on a 24 GiB card at a production width both backends are held
        // by the cap, so a comparison there would read "equal" and prove
        // nothing about the slopes.
        let vk = |cap: u64, t: usize| (((cap.saturating_sub(activation_reserve_bytes(t, "vulkan")) / per_block).min(cap / 4 / per_block)) as u32).min(cfg.num_layers);
        let wide = 20000usize;
        assert!(slots_at(card, wide) < cfg.num_layers / 4, "pick a width where the reserve, not the card-fraction cap, is what binds");
        assert!(vk(card, wide) > slots_at(card, wide), "the Vulkan backend must be budgeted more resident blocks than wgpu at the same shape");
    }

    /// Zero slots is not an error, it is the fallback: no window is built and
    /// the forward runs the pre-residency upload-per-block path.
    #[test]
    fn a_zero_slot_window_is_declined_rather_than_built() {
        let shape = WindowShape { t: 16, context_len: 128, tier: QTier::Int8, num_layers: 4 };
        assert!(BlockWindow::<LtxBlockQ>::build(shape, 0).is_none());
        assert!(BlockWindow::<LtxBlockQ>::build(shape, 2).is_some());
    }

    /// The window's own bookkeeping, driven the way `run_blocks` drives it:
    /// with a slot for every block, the FIRST pass uploads each block once and
    /// every later pass is a pure hit - which is the entire claim this module
    /// makes about a warm generation.
    #[test]
    fn a_full_window_uploads_each_block_once_and_never_again() {
        let n = 6u32;
        let shape = WindowShape { t: 16, context_len: 128, tier: QTier::Int8, num_layers: n };
        let mut w = BlockWindow::<LtxBlockQ>::build(shape, n).expect("a full window must build");
        let mut occupied = vec![None::<u32>; n as usize];
        for pass in 0..4 {
            for l in 0..n {
                let (slot, plan_miss) = w.ws.advance(l as usize);
                let idx = slot.0 as usize;
                let hit = occupied[idx] == Some(l) && !plan_miss;
                if hit {
                    w.hits += 1;
                } else {
                    occupied[idx] = Some(l);
                    w.uploads += 1;
                }
                assert_eq!(hit, pass > 0, "pass {pass}, block {l}: only the first pass may upload");
            }
        }
        assert_eq!(w.uploads, n as u64, "a full window uploads exactly once per block, ever");
        assert_eq!(w.hits, n as u64 * 3);
    }

    /// A window NARROWER than the model must still be correct and must still
    /// beat the no-window arm: the pinned prefix stops re-uploading after the
    /// first pass and only the rotating tail reloads. The number is exact, not
    /// a bound - `CyclicScan` pins `slots - 1` blocks and rotates the rest
    /// through one slot.
    #[test]
    fn a_narrow_window_pins_a_prefix_and_reloads_only_the_tail() {
        let n = 8u32;
        let slots = 4u32;
        let passes = 3u32;
        let shape = WindowShape { t: 16, context_len: 128, tier: QTier::Int8, num_layers: n };
        let mut w = BlockWindow::<LtxBlockQ>::build(shape, slots).expect("a narrow window must still build");
        let mut occupied = vec![None::<u32>; slots as usize];
        for _ in 0..passes {
            for l in 0..n {
                let (slot, plan_miss) = w.ws.advance(l as usize);
                let idx = slot.0 as usize;
                if occupied[idx] == Some(l) && !plan_miss {
                    w.hits += 1;
                } else {
                    occupied[idx] = Some(l);
                    w.uploads += 1;
                }
            }
        }
        let pinned = slots - 1;
        let tail = (n - pinned) as u64;
        assert_eq!(w.uploads, pinned as u64 + tail * passes as u64, "the pinned prefix uploads once; only the tail reloads per pass");
        assert!(w.uploads < n as u64 * passes as u64, "a narrow window must still upload strictly less than the no-window arm");
    }
}
