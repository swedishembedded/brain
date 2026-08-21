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
//! one of a generation's 8-16 forwards. Measured on this box (2x Tesla P40,
//! PCIe, no NVLink), real `ltx-2.5-22b-distilled-transformer-Q8_0.gguf`, all
//! 48 layers, T=3520 (720p), cache-warm: the `block GPU upload+forward+wait`
//! bucket was **104.0 s** of a 183 s call, against **~15.7 s** of real GPU
//! kernel time in the same call. Nearly all of the difference is buffer
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

use crate::block::{cached_block_bytes, open_device, BlockTimings, CachedQBlockWeights, LtxBlockQ, QTier};
use crate::config::LtxDitConfig;
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
/// Measured, not derived, on a 24 GiB Tesla P40 (wgpu/Vulkan) with the real
/// `ltx-2.5-22b-distilled-transformer-Q8_0.gguf`, all 48 layers, int8:
///
/// | arm | resident weights | peak VRAM | churn |
/// |---|---:|---:|---:|
/// | chained, 0 resident blocks, T=3520 | 0 | 16522 MiB | 16522 MiB |
/// | 48 resident blocks, T=3520 | 12360 MiB | 18062 MiB | 5702 MiB |
///
/// Those two numbers are not in conflict, and the difference between them is
/// the single most important fact about this budget: wgpu's allocator pool is
/// **elastic and greedy**. Left alone it grows to fill the card (16.5 GiB);
/// under pressure from long-lived allocations it works in a third of that
/// (5.7 GiB). So the plateau is NOT the requirement - the requirement is what
/// the pool shrinks to, and the constant below is that number plus margin.
///
/// Two consequences, both real:
///
/// * The resident window must be filled BEFORE the pool has grown (see
///   `DitSession::run_blocks`'s pre-fill). Filling it lazily, block by block,
///   loses a race against the pool and aborts - measured, at 24009 MiB of a
///   24576 MiB card.
/// * The slope is per-token because the churn is: the largest single transient
///   is `attn2`'s materialized `[heads, t, context_len]` score slab (461 MB at
///   T=3520/ctx 1024, 1.07 GB at T=8160), and every activation buffer in the
///   block is `[t, dim]` or `[t, 4*dim]`.
///
/// Deliberately generous: under-reserving costs a driver-level abort,
/// over-reserving costs a few resident blocks and the graceful partial-window
/// path picks up the difference.
pub fn activation_reserve_bytes(t: usize, backend: &str) -> u64 {
    /// Everything that does not follow the token count: the head tensors, the
    /// connector's working set, the RoPE tables, the allocator's own floor.
    const BASE: u64 = 3 << 29; // 1.5 GiB
    /// Measured slope for the DEFAULT wgpu backend, bytes per video token,
    /// fitted to the GREEDY plateau plus real headroom: at T=3520 it reserves
    /// 19.0 GiB against a plateau measured at 16.2-16.5 GiB, leaving 20
    /// resident blocks and a measured 2.4 GiB of the 24576 MiB card still
    /// free. Fitting it to the plateau alone allows 25 blocks and lands the
    /// peak at 23623 MiB, which works on an otherwise-idle card and is too
    /// thin to ship as a default.
    const PER_TOKEN_WGPU: u64 = 5200 * 1024;
    /// brain's own native Vulkan backend recycles transient buffers, uniforms
    /// and descriptor sets explicitly at every flush (`crates/vulkan`'s
    /// `VkContext` reclaim path) instead of growing an opportunistic pool, and
    /// it does not carry wgpu's measured 2.00x per-uploaded-buffer resident
    /// cost (`crates/gpu-core/tests/vram_overhead.rs`: 1.00x). Its churn is
    /// therefore close to the working set rather than to the card, so the same
    /// card affords far more resident blocks. Sized from the analytic working
    /// set (`attn2`'s `[heads, t, context_len]` score+probability pair plus
    /// ~20 `[t, dim]`/`[t, 4*dim]` activation buffers, double-buffered across
    /// the flush) rather than from a plateau, because there is no plateau to
    /// fit - and left deliberately generous, since the partial-window path
    /// costs a few uploads where an under-reserve costs an abort.
    const PER_TOKEN_VULKAN: u64 = 2000 * 1024;
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
    if let Some(n) = slots_override() {
        return n.min(cfg.num_layers);
    }
    let per_block = cached_block_bytes(cfg, tier);
    if per_block == 0 {
        return 0;
    }
    let usable = usable_vram(device).saturating_sub(activation_reserve_bytes(t, backend));
    ((usable / per_block) as u32).min(cfg.num_layers)
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
struct BlockWindow {
    shape: WindowShape,
    /// Slot `i`'s current occupant: which layer it holds and the uploaded
    /// block. Each entry owns its own `Gpu` handle, so `None`-ing an entry
    /// releases that slot's VRAM and its `memauth` grants together.
    slots: Vec<Option<(u32, LtxBlockQ)>>,
    ws: WeightSet,
    hits: u64,
    uploads: u64,
    /// Times the live `memauth` headroom refused one more resident block and
    /// it was streamed instead - see [`can_charge_a_block`]. Always 0 with no
    /// ceiling published.
    refusals: u64,
}

impl BlockWindow {
    fn build(shape: WindowShape, slots: u32) -> Option<BlockWindow> {
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
    window: Mutex<Option<BlockWindow>>,
    /// Slots this session was asked to plan for - `0` means residency is off
    /// and every forward opens its own device, exactly as before.
    slots: u32,
}

impl DitSession {
    /// A session that keeps nothing: every forward opens its own `Gpu` and
    /// uploads every block. The behaviour this crate shipped before device
    /// residency existed, kept as the fallback and as the arm a bit-identity
    /// gate compares against.
    pub fn transient(device: Option<&str>) -> DitSession {
        DitSession { gpu: None, device: device.map(str::to_string), window: Mutex::new(None), slots: 0 }
    }

    /// A session that holds its device open and keeps up to
    /// [`planned_slots`]-many blocks resident on it.
    ///
    /// `t` sizes the activation reserve, so the same card gives a 720p run
    /// more resident blocks than a 1080p one - which is the correct direction,
    /// since the 1080p run's activations are what would otherwise collide with
    /// them.
    pub fn resident(cfg: &LtxDitConfig, tier: QTier, device: Option<&str>, t: usize) -> DitSession {
        let gpu = open_device(device);
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
        DitSession { gpu: Some(gpu), device: device.map(str::to_string), window: Mutex::new(None), slots }
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
        DitSession { gpu: Some(open_device(device)), device: device.map(str::to_string), window: Mutex::new(None), slots }
    }

    /// This session's device handle for one forward call, plus whether it is
    /// long-lived. A transient session opens (and, on return, drops) a fresh
    /// one; a resident session hands back a `Gpu::share` of the one it holds,
    /// which is the same adapter, queue and compiled pipelines.
    pub fn device_for_call(&self) -> Gpu {
        match &self.gpu {
            Some(g) => g.share(),
            None => open_device(self.device.as_deref()),
        }
    }

    /// True when this session holds its device open across calls.
    pub fn is_resident(&self) -> bool {
        self.gpu.is_some()
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
        let pins: Vec<(usize, u32)> = w.ws.slot_contents().iter().enumerate().filter_map(|(i, g)| g.map(|g| (i, g.0))).filter(|(i, _)| w.slots[*i].is_none()).collect();
        if pins.is_empty() {
            return;
        }
        let s = std::time::Instant::now();
        // A one-word probe to drain wgpu's `write_buffer` staging after every
        // block - see `LtxBlockQ::forward_chained` for the measured 2.00x this
        // exists to stop accruing. Without it, uploading 48 blocks of 270 MB
        // reaches 24392 MiB of a 24576 MiB card and aborts.
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
    ///   (`crate::block::LtxBlockQ::forward_chained`) instead of being combined
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
        adaln_table: &[f32],
        connector_context: &[f32],
        cos_bufs: &[gpu_core::DeviceBuffer],
        sin_bufs: &[gpu_core::DeviceBuffer],
        weights: &mut dyn FnMut(usize) -> std::sync::Arc<CachedQBlockWeights>,
        after_block: &mut dyn FnMut(usize, bool, std::time::Duration, &BlockTimings),
    ) -> (Vec<f32>, BlockTimings) {
        let per_block = cached_block_bytes(cfg, tier);
        let mut guard = self.window.lock().unwrap_or_else(|e| e.into_inner());

        // The two per-forward uploads, in place of ~26 GB of per-block ones:
        // the model-level adaLN table (each block derives its own nine
        // modulation vectors FROM it on the card) and the text context.
        let s_up = std::time::Instant::now();
        let mut x = x;
        let adaln_buf = scratch.storage(adaln_table.len() as u64);
        // CHUNKED, not one 519 MB `write_f32`: this is the largest single
        // upload in a forward (`[t, 9*dim]` - 519 MB at T=3520, 1.2 GB at
        // T=8160) and on a non-ReBAR card one giant `write_buffer` measured
        // 7.6 s against 0.5 s chunked, because the staging allocation it needs
        // is the same size as the payload. Same chunk size `paramstore`'s own
        // weight-upload loop uses, for the same reason.
        scratch.write_f32_chunked(&adaln_buf, adaln_table, 1 << 20);
        let ctx_buf = scratch.storage(connector_context.len() as u64);
        scratch.write_f32(&ctx_buf, connector_context);
        let forward_upload = s_up.elapsed();

        for l in 0..cfg.num_layers as usize {
            // Ensure this layer's weights are on the card, and learn which slot
            // (if any) holds them, WITHOUT keeping a borrow of the window alive
            // across the forward below.
            let (slot_idx, uploaded, up_ms) = match guard.as_mut() {
                Some(w) => {
                    // The schedule is cyclic with period `num_layers`, so the
                    // layer index IS the plan cursor: from position `l` the
                    // remaining `[l, 2n)` range holds every group exactly once
                    // more, which is the whole lookahead Bélády needs. Deriving
                    // it from `l` rather than from a running counter also means
                    // an aborted forward cannot leave the window's cursor out
                    // of step with the block stack.
                    let (slot, plan_miss) = w.ws.advance(l);
                    debug_assert_eq!(w.ws.schedule().order[l], GroupId(l as u32), "the window's schedule must visit blocks in the forward's own order");
                    let idx = slot.0 as usize;
                    let occupied_by_this_layer = w.slots[idx].as_ref().map(|(g, _)| *g == l as u32).unwrap_or(false);
                    // `plan_miss` alone is not enough: `CyclicScan` PINS a
                    // prefix at build time, which the plan reports as already
                    // resident even though nothing has been uploaded into
                    // those slots yet (weightset owns slot assignment, never
                    // device bytes - its own `slot_contents` doc says the
                    // caller must load on that transition too).
                    if occupied_by_this_layer && !plan_miss {
                        w.hits += 1;
                        (Some(idx), false, std::time::Duration::ZERO)
                    } else if !can_charge_a_block(scratch.memory_device(), per_block) {
                        // A published ceiling says there is no room for one
                        // more resident block RIGHT NOW - degrade this block to
                        // per-call streaming rather than let the allocation
                        // facade panic. Loud, because a run that silently lost
                        // its residency would just look mysteriously slow.
                        w.slots[idx] = None;
                        w.refusals += 1;
                        tracing::warn!(layer = l, per_block, "no VRAM budget headroom for a resident block; streaming this one instead");
                        (None, true, std::time::Duration::ZERO)
                    } else {
                        let s = std::time::Instant::now();
                        let cached = weights(l);
                        // Drop the outgoing occupant BEFORE allocating the
                        // incoming one, so a full window's peak is `slots`
                        // blocks and not `slots + 1`.
                        w.slots[idx] = None;
                        let blk = LtxBlockQ::on_cached(scratch.share(), cfg, &cached, t as u32, context_len as u32, tier);
                        w.slots[idx] = Some((l as u32, blk));
                        w.uploads += 1;
                        (Some(idx), true, s.elapsed())
                    }
                }
                None => (None, true, std::time::Duration::ZERO),
            };

            let mut bt = BlockTimings::default();
            x = match slot_idx {
                Some(idx) => {
                    let w = guard.as_ref().expect("a slot index implies a window");
                    let blk = &w.slots[idx].as_ref().expect("the slot was just filled").1;
                    blk.forward_prod(scratch, &x, &adaln_buf, &ctx_buf, cos_bufs, sin_bufs, t as u32, &mut bt)
                }
                None => {
                    // No window (or a slot the budget refused): upload, run,
                    // drop - the pre-residency shape, on the same production
                    // block forward, so there is one implementation and not
                    // one per residency mode.
                    let cached = weights(l);
                    let blk = LtxBlockQ::on_cached(scratch.share(), cfg, &cached, t as u32, context_len as u32, tier);
                    blk.forward_prod(scratch, &x, &adaln_buf, &ctx_buf, cos_bufs, sin_bufs, t as u32, &mut bt)
                }
            };
            after_block(l, uploaded, up_ms, &bt);
        }
        (x, BlockTimings { record_upload: forward_upload, ..Default::default() })
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
        let slots_at = |cap: u64, t: usize| ((cap.saturating_sub(activation_reserve_bytes(t, "wgpu")) / per_block) as u32).min(cfg.num_layers);
        let card = 24u64 << 30;
        assert_eq!(slots_at(card, 1000), cfg.num_layers, "a 512x512-scale token count must fit every block on a 24 GiB card");
        assert!(slots_at(card, 3520) < cfg.num_layers, "720p must NOT plan a full window on a 24 GiB card - the measured plateau does not leave room");
        assert!(slots_at(card, 8160) <= slots_at(card, 3520), "a larger token count must never buy MORE resident blocks");
        assert_eq!(slots_at(1 << 30, 3520), 0, "a card smaller than the reserve must ask for zero slots, not a negative count");
        // The native Vulkan backend reclaims transients explicitly and has no
        // 2.00x resident cost, so the SAME card affords strictly more resident
        // blocks there - the budget must not hardcode wgpu's pathology.
        let vk = |cap: u64, t: usize| ((cap.saturating_sub(activation_reserve_bytes(t, "vulkan")) / per_block) as u32).min(cfg.num_layers);
        assert!(vk(card, 3520) > slots_at(card, 3520), "the Vulkan backend must be budgeted more resident blocks than wgpu at the same shape");
        assert_eq!(vk(card, 3520), cfg.num_layers, "720p must fit the whole model on the Vulkan backend");
    }

    /// Zero slots is not an error, it is the fallback: no window is built and
    /// the forward runs the pre-residency upload-per-block path.
    #[test]
    fn a_zero_slot_window_is_declined_rather_than_built() {
        let shape = WindowShape { t: 16, context_len: 128, tier: QTier::Int8, num_layers: 4 };
        assert!(BlockWindow::build(shape, 0).is_none());
        assert!(BlockWindow::build(shape, 2).is_some());
    }

    /// The window's own bookkeeping, driven the way `run_blocks` drives it:
    /// with a slot for every block, the FIRST pass uploads each block once and
    /// every later pass is a pure hit - which is the entire claim this module
    /// makes about a warm generation.
    #[test]
    fn a_full_window_uploads_each_block_once_and_never_again() {
        let n = 6u32;
        let shape = WindowShape { t: 16, context_len: 128, tier: QTier::Int8, num_layers: n };
        let mut w = BlockWindow::build(shape, n).expect("a full window must build");
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
        let mut w = BlockWindow::build(shape, slots).expect("a narrow window must still build");
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
