// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Chunked, CFG-guided flow-matching denoise: turns the per-frame hidden
//! states `pipeline::generate_frames` produced into Flow-VAE latents, one
//! 200-frame chunk (100-frame hop) at a time, splicing consecutive chunks
//! over a 172-latent overlap so the DiT never sees a hard seam.
//!
//! Ported directly from the reference `diffusers` PR's `ChunkConditionStep`
//! / `ChunkPrepareLatentsStep` / `ChunkSetTimestepsStep` / `ChunkDenoiseInner`
//! / `ChunkUpdateStep` (`before_denoise.py`/`denoise.py`), not reimagined:
//! every constant and blend formula below has a named counterpart there.
//!
//! Layout convention (matching `dit::forward`'s own parameters): latents are
//! `[in_channels, length]` NCL (channel-major); condition is `[length,
//! condition_dim]` row-major (frame-major, straight out of
//! `condition_encoder::forward`). The overlap carried between chunks slices
//! `length` (the last axis of latents, the first axis of condition) - a
//! strided extraction for latents, a contiguous one for condition.
//!
//! CFG here is over the DiT's own *conditioning*, not its logits: the
//! unconditional branch is a zeroed condition tensor (`denoise.py`'s
//! `zeros_like(condition)`), not a second Global-LLM/depth-decoder pass -
//! unrelated to `pipeline::generate_frames`'s own AR-stage CFG, which
//! blends two full model branches. Two independent CFG axes, ported
//! independently, matching the reference's own two independent `Guider`
//! components.
//!
//! Those two forwards share `latents`, the timestep and `length` and differ
//! only in the condition, so on a two-card box they run **concurrently, one
//! per card** ([`CfgDevices`], placed by [`crate::devplan`]) - which is the
//! whole point, because this stage is ~all of a generation's wall clock and
//! the second card was otherwise idle for every minute of it. They exchange
//! only host-side `Vec<f32>`, so there is no cross-device transfer to
//! arrange, and the fold below still runs on the orchestrating thread in
//! the same order either way: the concurrent path is **bit-identical**, and
//! [`tests::the_concurrent_cfg_pair_is_bit_identical_to_the_sequential_one`]
//! gates that rather than assuming it.

use crate::condition_encoder::{self, ConditionEncoderWeights};
use crate::config::{ConditionEncoderConfig, DitConfig};
use crate::devplan::{self, DevicePlan, Placement};
use crate::dit::{self, DitWeights};
use data::rng::Rng;
use diffusion::scheduler::{default_z_image_sigmas, FlowMatchConfig, FlowMatchEulerScheduler};
use gpu_core::Gpu;

/// Frames per chunk (`_CHUNK_FRAMES`).
pub const CHUNK_FRAMES: usize = 200;
/// Frame stride between consecutive chunk starts (`_CHUNK_HOP`).
pub const CHUNK_HOP: usize = 100;
/// Latent-axis overlap carried from one chunk into the next (`_OVERLAP_LATENT_LENGTH`).
pub const OVERLAP_LATENT_LENGTH: usize = 172;
/// Euler steps per chunk when the caller doesn't override it.
pub const DEFAULT_NUM_INFERENCE_STEPS: usize = 30;
/// The DiT's own classifier-free guidance scale (distinct from the AR
/// stage's `pipeline::AR_CFG_SCALE`).
pub const GUIDANCE_SCALE: f32 = 1.7;

/// `[0]` for a song that fits in one chunk, else every 100-frame-hop start
/// up to (not including) the tail that would run past `num_frames` -
/// `range(0, num_frames-100, 100)` in the reference.
pub fn chunk_starts(num_frames: usize) -> Vec<usize> {
    if num_frames <= CHUNK_FRAMES {
        vec![0]
    } else {
        (0..num_frames - CHUNK_HOP).step_by(CHUNK_HOP).collect()
    }
}

/// The state one chunk hands to the next: the trailing `in_channels x span`
/// slice of the just-denoised latents and the matching `span x
/// condition_dim` slice of that chunk's own condition (`span <=
/// OVERLAP_LATENT_LENGTH`, and less on the first couple of chunks of a
/// short song). `None` before the first chunk.
#[derive(Clone, Debug, Default)]
pub struct ChunkState {
    pub previous_latent: Option<Vec<f32>>,
    pub previous_condition: Option<Vec<f32>>,
}

/// `n` samples of standard-normal noise via [`Rng::next_gaussian`] - the
/// canonical Gaussian source (`data::rng::Lcg` has no Gaussian sampler and
/// this crate's own convention, per `data::rng`'s doc, is to reach for
/// `Rng` rather than hand-roll a fresh Box-Muller copy on top of `Lcg`).
fn gaussian_vec(seed: u64, n: usize) -> Vec<f32> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| r.next_gaussian() as f32).collect()
}

/// The device handle(s) one generation's CFG branches run on, opened once
/// for the whole denoise stage.
///
/// `uncond` is `Some` only for a genuinely split [`Placement`] - one card
/// per branch. On every other machine (one schedulable card, the CPU
/// backend, an explicitly pinned device, `BRAIN_MINIMAXMUSIC3_CFG_PARALLEL=0`)
/// it is `None` and both forwards run on `cond`, one after the other, which
/// is byte-for-byte what this stage did before.
///
/// A `Gpu` is expensive to build (device init plus one shader compile per
/// kernel) and hostile to the driver when several exist per card, so this
/// is built ONCE per generation and reused across every chunk - never per
/// chunk and never per step.
pub struct CfgDevices {
    cond: Gpu,
    cond_card: Option<u32>,
    /// `(card, handle)` for the unconditional branch's own card.
    uncond: Option<(u32, Gpu)>,
}

impl CfgDevices {
    /// Both branches on `gpu`, sequentially.
    pub fn single(gpu: Gpu) -> CfgDevices {
        CfgDevices { cond: gpu, cond_card: None, uncond: None }
    }

    /// Resolve [`DevicePlan::Auto`] against this machine and open one handle
    /// per card it names.
    ///
    /// `stage_device` is the `--device`-shaped token
    /// `crate::generate::stage_devices` picked for the denoise stage, used
    /// only when the plan declines to split; `explicit` is the caller's own
    /// `GenOpts::device`, which forces that decline (see
    /// [`DevicePlan::resolve`]). When the plan DOES split it names both
    /// cards itself, from the same schedulable set `stage_devices` reads.
    pub fn open(stage_device: Option<&str>, explicit: Option<&str>) -> CfgDevices {
        let place = DevicePlan::Auto.resolve(explicit);
        Self::open_placed(place, stage_device)
    }

    /// [`CfgDevices::open`] against an already-resolved placement - the seam
    /// the bit-identity gate drives so it can build both shapes on one box.
    pub fn open_placed(place: Placement, stage_device: Option<&str>) -> CfgDevices {
        if !place.cfg_is_parallel() {
            return CfgDevices::single(Gpu::open(stage_device, dit::PIPELINES));
        }
        let (cond, uncond) = (place.cond.expect("a parallel placement names both cards"), place.uncond.expect("a parallel placement names both cards"));
        // By canonical index, not through a formatted `gpu<i>` token: the
        // index came from `ambient_compute_set()` already, and an
        // out-of-range one is an error here rather than a silent fall-through
        // to the ambient card (AGENTS.md: never a silent clamp).
        let on = |i: u32| Gpu::new_on_index(i, dit::PIPELINES).unwrap_or_else(|e| panic!("minimaxmusic3: the CFG placement named gpu{i}, which this machine refused: {e}"));
        CfgDevices { cond: on(cond), cond_card: Some(cond), uncond: Some((uncond, on(uncond))) }
    }

    /// True when the two branches really do have a card each.
    pub fn is_parallel(&self) -> bool {
        self.uncond.is_some()
    }
}

/// The generation's uploaded DiT weights and current RoPE tables, **one set
/// per card**.
///
/// Built ONCE for the whole denoise stage and reused across every chunk and
/// all `2 * num_inference_steps` evaluations within each - see
/// [`dit::Resident`] for what a rebuild costs (~9.7 GB of host->device
/// traffic at `DitConfig::real()` dims, ~22 s per card on a P40). A split
/// placement costs ~9.7 GB on EACH card, not 19.4 GB on one.
///
/// The only per-chunk input is the chunk's `length`, and the only thing it
/// feeds is the RoPE tables, so [`ChunkResidents::bind`] rebuilds those and
/// nothing else. This follows [`CfgDevices`]'s own precedent exactly: the
/// expensive, invariant thing is built once per generation at the stage's
/// scope in `crate::generate::generate`, and the chunk loop borrows it.
///
/// Lifetime: it is the denoise stage's steady-state VRAM, so it must live
/// and die inside that stage's block scope - `generate::generate`'s
/// sequential-stage RAM discipline requires the DiT to be gone before the
/// vocoder loads. Borrowing `devices` is what makes the compiler enforce
/// half of that: these residents cannot outlive the handles they were
/// uploaded through.
pub struct ChunkResidents<'a> {
    devices: &'a CfgDevices,
    cond: dit::Resident,
    uncond: Option<dit::Resident>,
}

impl<'a> ChunkResidents<'a> {
    /// Upload the DiT's weights to every card the placement named, with the
    /// RoPE tables for a first chunk of `length` latent frames.
    ///
    /// The two uploads are the same ~9.7 GB of host bytes going to two
    /// different cards over two different PCIe paths, so they run
    /// concurrently for the same reason the forwards do. Serialising them
    /// would put a second full weight upload on the generation's critical
    /// path and hand back a slice of what the concurrent forwards just won.
    pub fn new(devices: &'a CfgDevices, cfg: &DitConfig, w: &DitWeights, length: usize) -> ChunkResidents<'a> {
        let Some((card_u, gpu_u)) = devices.uncond.as_ref() else {
            return ChunkResidents { devices, cond: dit::Resident::new(&devices.cond, cfg, w, length), uncond: None };
        };
        std::thread::scope(|s| {
            let u = s.spawn(move || devplan::on_gpu(Some(*card_u), || dit::Resident::new(gpu_u, cfg, w, length)));
            let c = devplan::on_gpu(devices.cond_card, || dit::Resident::new(&devices.cond, cfg, w, length));
            let u = u.join().unwrap_or_else(|payload| std::panic::resume_unwind(payload));
            ChunkResidents {
                devices,
                cond: c.expect("the conditional card is in the schedulable set"),
                uncond: Some(u.expect("the unconditional card is in the schedulable set")),
            }
        })
    }

    /// Point every card's resident at a chunk of `length` latent frames.
    ///
    /// Only [`dit::Resident::rebind`]'s ~90 kB of RoPE tables move; the
    /// blocks are untouched. `chunk_starts` hands out mostly full
    /// `CHUNK_FRAMES` chunks with a possibly shorter tail, so in a real
    /// generation this is a no-op for every chunk but the last.
    ///
    /// Sequential even on a split placement, unlike the upload in
    /// [`ChunkResidents::new`]: two table builds are microseconds, and a
    /// thread scope to overlap them would cost more than it saves. That is
    /// also why there is no `devplan::on_gpu` scope here where `new` and
    /// `cfg_pair` both have one - those exist because they dispatch from
    /// SPAWNED threads, which do not inherit the scoped selection. This runs
    /// on the orchestrating thread and only allocates, and a `Gpu` resolves
    /// the card its allocations are charged to once at construction
    /// (`gpu_core::Gpu`'s `mem_device`), not per call.
    fn bind(&mut self, cfg: &DitConfig, length: usize) {
        self.cond.rebind(&self.devices.cond, cfg, length);
        if let (Some(res), Some((_, gpu))) = (self.uncond.as_mut(), self.devices.uncond.as_ref()) {
            res.rebind(gpu, cfg, length);
        }
    }

    /// One Euler step's two forwards: sequentially when both branches share
    /// a card, and concurrently - one thread per card, each scoped with
    /// `devplan::on_gpu` - when they do not.
    ///
    /// Sharing `&self` across the two threads is safe for reasons that are
    /// properties of the fields rather than assumptions about them, and the
    /// argument is checked by the compiler at this call site rather than
    /// asserted in prose (`std::thread::scope` requires every captured
    /// reference's target to be `Sync`):
    ///
    /// * each branch touches only ITS OWN `Gpu` and its own `Resident`, so
    ///   no device handle is used from two threads at once;
    /// * `cfg` and `w` are read-only host data shared by both;
    /// * `latents` is the same immutable slice for both - they read it, they
    ///   do not step it; the step happens on the orchestrating thread after
    ///   both have returned.
    ///
    /// The progress callback is deliberately NOT reachable from here: it is
    /// a `&mut dyn FnMut` (`crate::ProgressSink`), it is not `Sync`, and
    /// calling it from inside either worker would reorder the per-step
    /// reports. It stays on the orchestrating thread, called after the join.
    fn cfg_pair(&self, cfg: &DitConfig, w: &DitWeights, latents: &[f32], condition: &[f32], zero_condition: &[f32], t: f32, length: usize) -> (Vec<f32>, Vec<f32>) {
        let (Some(res_u), Some((card_u, gpu_u))) = (self.uncond.as_ref(), self.devices.uncond.as_ref()) else {
            let cond = dit::forward_resident(&self.devices.cond, cfg, w, &self.cond, latents, condition, t, length);
            let uncond = dit::forward_resident(&self.devices.cond, cfg, w, &self.cond, latents, zero_condition, t, length);
            return (cond, uncond);
        };
        std::thread::scope(|s| {
            let u = s.spawn(move || devplan::on_gpu(Some(*card_u), || dit::forward_resident(gpu_u, cfg, w, res_u, latents, zero_condition, t, length)));
            let c = devplan::on_gpu(self.devices.cond_card, || dit::forward_resident(&self.devices.cond, cfg, w, &self.cond, latents, condition, t, length));
            // Join before touching either result: an early return would drop
            // the scope's guard and block on the same join anyway, and
            // reporting the conditional branch's failure while the
            // unconditional one is still running reads as a hang. A worker
            // panic is re-raised with its ORIGINAL payload - the kernel-level
            // message is what a debugger needs, and wrapping it in a
            // placement-flavoured string would hide it.
            let u = u.join().unwrap_or_else(|payload| std::panic::resume_unwind(payload));
            (c.expect("the conditional card is in the schedulable set"), u.expect("the unconditional card is in the schedulable set"))
        })
    }
}

/// Denoise one chunk: `frame_hiddens` is the WHOLE song's per-frame hidden
/// states (`[num_frames_total, num_condition_layers*condition_hidden_dim]`
/// row-major, `pipeline::generate_frames`'s own output layout), sliced here
/// to `[chunk_start, chunk_start+CHUNK_FRAMES)` (clipped to
/// `num_frames_total`). Returns this chunk's denoised latents, `[in_channels,
/// length]` NCL, and advances `state` for the next call.
///
/// `residents` is the generation's already-uploaded weights, taken by
/// `&mut` for one reason: this call re-points their RoPE tables at THIS
/// chunk's length ([`ChunkResidents::bind`]). Nothing else about them
/// changes, and in particular nothing is re-uploaded.
#[allow(clippy::too_many_arguments)]
pub fn denoise_chunk(
    residents: &mut ChunkResidents<'_>,
    dit_cfg: &DitConfig,
    dit_w: &DitWeights,
    cond_cfg: &ConditionEncoderConfig,
    cond_w: &ConditionEncoderWeights,
    frame_hiddens: &[f32],
    num_frames_total: usize,
    chunk_start: usize,
    state: &mut ChunkState,
    num_inference_steps: usize,
    seed: u64,
    progress: crate::ProgressSink<'_>,
) -> Vec<f32> {
    let chunk_end = (chunk_start + CHUNK_FRAMES).min(num_frames_total);
    let chunk_frames = chunk_end - chunk_start;
    let per_frame = (cond_cfg.num_condition_layers * cond_cfg.condition_hidden_dim) as usize;
    let chunk_hidden = &frame_hiddens[chunk_start * per_frame..chunk_end * per_frame];
    let (mut condition, length) = condition_encoder::forward(cond_cfg, cond_w, chunk_hidden, 1, chunk_frames);
    let condition_dim = cond_cfg.out_dim as usize;
    let cin = dit_cfg.in_channels as usize;

    // `overlap = min(previous_latent's own span, this chunk's length)` -
    // `previous_condition` was sliced in lockstep with `previous_latent` by
    // the prior call, so it always covers at least `overlap` frames too.
    let overlap = match (&state.previous_latent, &state.previous_condition) {
        (Some(prev_latent), Some(prev_condition)) => {
            let span = prev_latent.len() / cin;
            debug_assert_eq!(prev_condition.len(), span * condition_dim, "denoise_chunk: previous_latent/previous_condition span mismatch");
            span.min(length)
        }
        _ => 0,
    };
    if overlap > 0 {
        let prev_condition = state.previous_condition.as_ref().unwrap();
        condition[..overlap * condition_dim].copy_from_slice(&prev_condition[..overlap * condition_dim]);
    }

    let mut latents = gaussian_vec(seed, cin * length);
    // `noise_prompt`: the freshly-drawn noise in the overlap region, before
    // any denoise step touches it - the blend below interpolates between
    // this and `previous_latent` every step, not just once.
    let noise_prompt: Vec<f32> = if overlap > 0 {
        (0..cin).flat_map(|c| latents[c * length..c * length + overlap].to_vec()).collect()
    } else {
        Vec::new()
    };

    let mut scheduler = FlowMatchEulerScheduler::new(FlowMatchConfig { num_train_timesteps: 1, shift: 1.0, invert_sigmas: true });
    scheduler.set_timesteps(&default_z_image_sigmas(num_inference_steps));
    let timesteps: Vec<f32> = scheduler.timesteps().to_vec();

    let zero_condition = vec![0.0f32; condition.len()];
    let prev_span = state.previous_latent.as_ref().map(|p| span_of(p, cin));
    // The DiT's weights are identical for every one of the `2 *
    // num_inference_steps` evaluations below AND for every other chunk of
    // this generation, so they were uploaded ONCE - per card - by the
    // caller. All that is per-chunk is the RoPE tables, which depend only
    // on `length`; see `dit::Resident::rebind` and `ChunkResidents`.
    residents.bind(dit_cfg, length);
    let total_steps = timesteps.len() as u32;
    for (step_index, &t) in timesteps.iter().enumerate() {
        if overlap > 0 {
            let prev_latent = state.previous_latent.as_ref().unwrap();
            let span = prev_span.unwrap();
            for c in 0..cin {
                for j in 0..overlap {
                    latents[c * length + j] = (1.0 - (1.0 - 1e-6) * t) * noise_prompt[c * overlap + j] + t * prev_latent[c * span + j];
                }
            }
        }
        let (v_cond, v_uncond) = residents.cfg_pair(dit_cfg, dit_w, &latents, &condition, &zero_condition, t, length);
        let velocity: Vec<f32> = v_cond.iter().zip(&v_uncond).map(|(c, u)| u + (c - u) * GUIDANCE_SCALE).collect();
        latents = scheduler.step(&velocity, &latents);
        // On the orchestrating thread, after both branches have joined - see
        // `ChunkResidents::cfg_pair` for why this may never move inside one.
        progress(step_index as u32 + 1, total_steps, "denoise");
    }

    if overlap > 0 {
        let prev_latent = state.previous_latent.as_ref().unwrap();
        let span = span_of(prev_latent, cin);
        for c in 0..cin {
            for j in 0..overlap {
                latents[c * length + j] = prev_latent[c * span + j];
            }
        }
    }

    let overlap_start = length.saturating_sub(2 * OVERLAP_LATENT_LENGTH);
    let overlap_end = overlap_start.max(length.saturating_sub(OVERLAP_LATENT_LENGTH));
    state.previous_latent = Some((0..cin).flat_map(|c| latents[c * length + overlap_start..c * length + overlap_end].to_vec()).collect());
    state.previous_condition = Some(condition[overlap_start * condition_dim..overlap_end * condition_dim].to_vec());

    latents
}

fn span_of(prev_latent: &[f32], cin: usize) -> usize {
    prev_latent.len() / cin
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dit_train;
    use data::rng::Lcg;

    fn random_condition_weights(cfg: &ConditionEncoderConfig, seed: u64) -> ConditionEncoderWeights {
        let mut r = Lcg::new(seed);
        let (layers, hidden, out_dim) = (cfg.num_condition_layers as usize, cfg.condition_hidden_dim as usize, cfg.out_dim as usize);
        ConditionEncoderWeights {
            layer_weight_logits: r.vec_scaled(layers, 0.5),
            layer_scale: 1.0,
            proj_weight: r.vec_scaled(out_dim * hidden * 3, 0.2),
            proj_bias: r.vec_scaled(out_dim, 0.1),
        }
    }

    /// **The two cards must agree, bit for bit.** The gate on concurrent CFG
    /// dispatch: running the conditional and zero-condition forwards at the
    /// same time on two cards must produce results identical to running them
    /// one after the other on one.
    ///
    /// Not a tolerance and not an epsilon: a bit-pattern comparison, because
    /// the claim is exactness. It should hold trivially - the two forwards
    /// are independent computations over the same latents, not two halves of
    /// one reduction, so moving one of them reassociates no sum - and if it
    /// ever does not, that is a real defect (a nondeterministic kernel, an
    /// uninitialised read, a `Resident` shared across cards) worth failing on
    /// rather than papering over with a wider bound.
    ///
    /// Runs on whatever this box has: with two schedulable cards it really
    /// does dispatch across both, at `DitConfig::tiny()` dims so it costs
    /// milliseconds and needs no checkpoint; with one (or none)
    /// `DevicePlan::Auto` resolves `Single` and it degenerates to "the same
    /// thing twice", which still gates the fold and the plumbing.
    #[test]
    fn the_concurrent_cfg_pair_is_bit_identical_to_the_sequential_one() {
        let cfg = DitConfig::tiny();
        let w = dit_train::random_weights(&cfg, 0xC0FFEE);
        let (length, cin) = (6usize, cfg.in_channels as usize);
        let cond_dim = cfg.condition_dim as usize;
        let mut r = Lcg::new(0xBEEF);
        let latents = r.vec_scaled(cin * length, 0.7);
        let condition = r.vec_scaled(length * cond_dim, 0.4);
        let zero = vec![0.0f32; condition.len()];
        let t = 0.63f32;

        // The pooled test device for the sequential arm (AGENTS.md: test
        // binaries share one device rather than building a fixture each);
        // the concurrent arm must open the two cards its placement names.
        let seq = CfgDevices::single(gpu_core::testgpu::dev(dit::PIPELINES));
        let (c0, u0) = ChunkResidents::new(&seq, &cfg, &w, length).cfg_pair(&cfg, &w, &latents, &condition, &zero, t, length);

        let place = DevicePlan::Auto.resolve(None);
        // Printed, not asserted: this gate is meaningful on a one-card box
        // too, and the reader of a passing run needs to know WHICH claim it
        // just checked - two cards, or the same card twice.
        eprintln!("cfg placement: cond={:?} uncond={:?} genuinely concurrent={}", place.cond, place.uncond, place.cfg_is_parallel());
        let par = CfgDevices::open_placed(place, None);
        assert_eq!(par.is_parallel(), place.cfg_is_parallel(), "the opened handles must match the placement they came from");
        let (c1, u1) = ChunkResidents::new(&par, &cfg, &w, length).cfg_pair(&cfg, &w, &latents, &condition, &zero, t, length);

        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        assert_eq!(bits(&c0), bits(&c1), "the conditional branch changed value by being placed on {:?} (parallel={})", place.cond, place.cfg_is_parallel());
        assert_eq!(bits(&u0), bits(&u1), "the zero-condition branch changed value by being placed on {:?}", place.uncond);
        // The two branches must genuinely differ, or a dispatch that ran the
        // conditional forward twice would pass the assertions above.
        assert_ne!(bits(&c0), bits(&u0), "the real and zeroed conditions must produce different velocities, or this gate proves nothing");
    }

    #[test]
    fn chunk_starts_matches_the_reference_windowing() {
        assert_eq!(chunk_starts(50), vec![0]);
        assert_eq!(chunk_starts(200), vec![0]);
        assert_eq!(chunk_starts(250), vec![0, 100]);
        assert_eq!(chunk_starts(400), vec![0, 100, 200]);
    }

    #[test]
    fn single_chunk_denoise_produces_the_expected_shape_with_no_overlap() {
        let dit_cfg = DitConfig::tiny();
        let cond_cfg = ConditionEncoderConfig::tiny();
        let dit_w = dit_train::random_weights(&dit_cfg, 1);
        let cond_w = random_condition_weights(&cond_cfg, 2);
        let devices = CfgDevices::single(Gpu::new_cpu(dit::PIPELINES));

        let num_frames = 5usize;
        let per_frame = (cond_cfg.num_condition_layers * cond_cfg.condition_hidden_dim) as usize;
        let mut r = Lcg::new(3);
        let frame_hiddens = r.vec_scaled(num_frames * per_frame, 0.3);

        let expected_length = condition_encoder::latent_length(&cond_cfg, num_frames);
        let mut residents = ChunkResidents::new(&devices, &dit_cfg, &dit_w, expected_length);
        let mut state = ChunkState::default();
        let latents = denoise_chunk(&mut residents, &dit_cfg, &dit_w, &cond_cfg, &cond_w, &frame_hiddens, num_frames, 0, &mut state, 4, 7, &mut crate::ignore_progress());

        assert_eq!(latents.len(), dit_cfg.in_channels as usize * expected_length);
        assert!(state.previous_latent.is_some());
        assert!(state.previous_condition.is_some());
    }

    /// A chunk must report one `denoise` step per Euler step.
    ///
    /// The `generate` action has always declared `.streaming()`, but every
    /// layer bound its progress callback as `_progress` and dropped it, so
    /// a multi-minute call reported nothing to the CLI, to D-Bus, or to
    /// `brain perf` - which derives its whole timeline from exactly these
    /// callbacks. An advertised capability that nothing exercises is how
    /// that regressed unnoticed, so it is exercised here.
    #[test]
    fn every_euler_step_reports_progress() {
        let dit_cfg = DitConfig::tiny();
        let cond_cfg = ConditionEncoderConfig::tiny();
        let dit_w = dit_train::random_weights(&dit_cfg, 11);
        let cond_w = random_condition_weights(&cond_cfg, 12);
        let devices = CfgDevices::single(Gpu::new_cpu(dit::PIPELINES));

        let num_frames = 5usize;
        let per_frame = (cond_cfg.num_condition_layers * cond_cfg.condition_hidden_dim) as usize;
        let mut r = Lcg::new(13);
        let frame_hiddens = r.vec_scaled(num_frames * per_frame, 0.3);

        let steps = 4usize;
        let mut seen: Vec<(u32, u32, String)> = Vec::new();
        let mut residents = ChunkResidents::new(&devices, &dit_cfg, &dit_w, condition_encoder::latent_length(&cond_cfg, num_frames));
        let mut state = ChunkState::default();
        let _ = denoise_chunk(&mut residents, &dit_cfg, &dit_w, &cond_cfg, &cond_w, &frame_hiddens, num_frames, 0, &mut state, steps, 7, &mut |done, total, stage| {
            seen.push((done, total, stage.to_string()));
        });

        assert_eq!(seen.len(), steps, "expected one progress report per Euler step, got {seen:?}");
        assert!(seen.iter().all(|(_, _, stage)| stage == "denoise"), "every report must name the denoise stage: {seen:?}");
        // Monotonic and terminating at the total - a progress stream that
        // never reaches its own total reads as a stall to any client.
        let done: Vec<u32> = seen.iter().map(|(d, _, _)| *d).collect();
        assert_eq!(done, (1..=steps as u32).collect::<Vec<_>>());
        assert!(seen.iter().all(|(_, total, _)| *total == steps as u32));
    }

    /// **Hoisting the weight upload out of the chunk loop must change
    /// nothing.** The gate on [`ChunkResidents`] outliving one chunk: a
    /// multi-chunk denoise driven by ONE set of residents, rebound per
    /// chunk, must produce byte-identical latents to one that rebuilds the
    /// residents from scratch for every chunk.
    ///
    /// Bit-for-bit, not a tolerance: uploading the same bytes once instead
    /// of N times reorders no arithmetic, so equality is the prediction. The
    /// two chunks here deliberately have DIFFERENT lengths, so the arm under
    /// test really does exercise the RoPE rebind rather than passing because
    /// nothing ever changed.
    #[test]
    fn residents_hoisted_across_chunks_are_bit_identical_to_rebuilding_them_per_chunk() {
        let dit_cfg = DitConfig::tiny();
        let cond_cfg = ConditionEncoderConfig::tiny();
        let dit_w = dit_train::random_weights(&dit_cfg, 31);
        let cond_w = random_condition_weights(&cond_cfg, 32);
        let devices = CfgDevices::single(Gpu::new_cpu(dit::PIPELINES));

        let num_frames = 6usize;
        let per_frame = (cond_cfg.num_condition_layers * cond_cfg.condition_hidden_dim) as usize;
        let mut r = Lcg::new(33);
        let frame_hiddens = r.vec_scaled(num_frames * per_frame, 0.3);
        let starts = [0usize, 2];
        let length_at = |start: usize| condition_encoder::latent_length(&cond_cfg, num_frames - start);
        assert_ne!(length_at(starts[0]), length_at(starts[1]), "the two chunks must differ in length or the rebind is never exercised");

        let run = |per_chunk: bool| -> Vec<Vec<f32>> {
            let mut state = ChunkState::default();
            let mut residents = ChunkResidents::new(&devices, &dit_cfg, &dit_w, length_at(starts[0]));
            starts
                .iter()
                .map(|&start| {
                    if per_chunk {
                        residents = ChunkResidents::new(&devices, &dit_cfg, &dit_w, length_at(start));
                    }
                    denoise_chunk(&mut residents, &dit_cfg, &dit_w, &cond_cfg, &cond_w, &frame_hiddens, num_frames, start, &mut state, 4, 41, &mut crate::ignore_progress())
                })
                .collect()
        };

        let fresh = run(true);
        let hoisted = run(false);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        for (i, (a, b)) in fresh.iter().zip(&hoisted).enumerate() {
            assert_eq!(bits(a), bits(b), "chunk {i} changed value by reusing the residents from the previous chunk");
        }
    }

    #[test]
    fn two_chunks_carry_forward_a_consistent_overlap() {
        let dit_cfg = DitConfig::tiny();
        let cond_cfg = ConditionEncoderConfig::tiny();
        let dit_w = dit_train::random_weights(&dit_cfg, 11);
        let cond_w = random_condition_weights(&cond_cfg, 12);
        let devices = CfgDevices::single(Gpu::new_cpu(dit::PIPELINES));

        // Small enough that `chunk_starts` still only emits [0], but we
        // drive `denoise_chunk` twice by hand (at real scale `chunk_starts`
        // would supply the second start) to exercise the overlap path
        // without needing a 250-frame fixture.
        let num_frames = 6usize;
        let per_frame = (cond_cfg.num_condition_layers * cond_cfg.condition_hidden_dim) as usize;
        let mut r = Lcg::new(13);
        let frame_hiddens = r.vec_scaled(num_frames * per_frame, 0.3);

        let first_length = condition_encoder::latent_length(&cond_cfg, num_frames);
        let mut residents = ChunkResidents::new(&devices, &dit_cfg, &dit_w, first_length);
        let mut state = ChunkState::default();
        let first = denoise_chunk(&mut residents, &dit_cfg, &dit_w, &cond_cfg, &cond_w, &frame_hiddens, num_frames, 0, &mut state, 4, 21, &mut crate::ignore_progress());
        assert_eq!(first.len(), dit_cfg.in_channels as usize * first_length);

        let prev_latent = state.previous_latent.clone().unwrap();
        let prev_condition = state.previous_condition.clone().unwrap();
        let span = prev_latent.len() / dit_cfg.in_channels as usize;
        assert_eq!(prev_condition.len(), span * cond_cfg.out_dim as usize);
        assert!(span <= first_length.min(OVERLAP_LATENT_LENGTH));

        let second = denoise_chunk(&mut residents, &dit_cfg, &dit_w, &cond_cfg, &cond_w, &frame_hiddens, num_frames, 2, &mut state, 4, 22, &mut crate::ignore_progress());
        let second_length = condition_encoder::latent_length(&cond_cfg, num_frames - 2);
        assert_eq!(second.len(), dit_cfg.in_channels as usize * second_length);
    }
}
