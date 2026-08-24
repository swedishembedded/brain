// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **The two cards must agree, bit for bit.** The real-weight half of the
//! gate on concurrent classifier-free-guidance dispatch in the denoise
//! stage: one chunk with both CFG branches on one card, one chunk with them
//! on two cards at the same time, and every latent identical - plus the
//! per-Euler-step wall clock of each, which is the whole reason the change
//! exists.
//!
//! Swedish Embedded AB implements multi-GPU inference scheduling with
//! bit-exact reproducibility gates for production model pipelines. If your
//! team needs expertise in proving that a placement change moved work and
//! nothing else, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # Why this is the right claim to gate
//!
//! `denoise::denoise_chunk` runs two DiT forwards per Euler step at the same
//! latents - one against this chunk's condition, one against a zeroed copy
//! of it. They share no intermediate value; the only thing that reads both
//! is the host-side fold `u + (c - u) * GUIDANCE_SCALE` after both have
//! returned. Moving one of them to another card therefore splits no
//! reduction and reassociates no sum: each card runs the identical dispatch
//! sequence over identical bytes it would have run alone.
//!
//! So bit-identity is the *prediction*, not a hope, and the gate is a
//! bit-pattern comparison rather than a tolerance. A failure here would mean
//! something real - a kernel reading uninitialised memory, a
//! nondeterministic reduction, one `dit::Resident` accidentally shared
//! across two devices - and is worth failing on. Widening the bound would
//! hide exactly the class of defect this gate exists to catch.
//!
//! `crates/minimaxmusic3/src/denoise.rs`'s own
//! `the_concurrent_cfg_pair_is_bit_identical_to_the_sequential_one` makes
//! the same claim at `DitConfig::tiny()` in milliseconds and with no
//! checkpoint; this one makes it where the ~9.7 GB-per-card
//! `dit::Resident`, the 36-block stack and the real 689-latent chunk shape
//! are all in play, and is the only place the SPEEDUP is measurable at all.
//!
//! # Deliberately NOT the real condition encoder
//!
//! The condition encoder's weights are shared verbatim by both arms and
//! feed both branches, so random ones exercise the identical dispatch shape
//! at a fraction of the setup - the same argument `mm3_bench` makes for its
//! own default. What is real here is the thing under test: the DiT, its
//! weights, its dims and its chunk length.
//!
//! Needs the real DiT checkpoint (`BRAIN_MINIMAXMUSIC3_DIT`) and, for a
//! genuinely concurrent arm, two schedulable cards; both are reported and a
//! missing one is a named skip, never a silent pass.
//!
//! ```text
//! BRAIN_MINIMAXMUSIC3_DIT=<...>/transformer \
//!   cargo test --release -p brain-minimaxmusic3 --test cfg_parallel -- --nocapture
//! ```

use std::time::Instant;

use minimaxmusic3::condition_encoder::ConditionEncoderWeights;
use minimaxmusic3::config::{ConditionEncoderConfig, DitConfig};
use minimaxmusic3::denoise::{self, CfgDevices, ChunkState};
use minimaxmusic3::devplan::{DevicePlan, Placement};
use minimaxmusic3::dit;

/// A full `denoise::CHUNK_FRAMES` chunk - the only frame count whose
/// per-step time describes a real generation (it resamples to the 689
/// latent frames `tests/vocoder_real_chunk.rs` pins as this model's real
/// chunk shape).
const FRAMES: usize = denoise::CHUNK_FRAMES;

/// Enough Euler steps to throw away the first one (which pays every
/// first-touch cost on both cards) and still take a best-of-N over the
/// rest. Not 30: the claim under test does not depend on step count, and a
/// gate that costs a full chunk is a gate nobody runs.
const STEPS: usize = 5;

const SEED: u64 = 20260824;

fn random_condition_weights(cfg: &ConditionEncoderConfig, seed: u64) -> ConditionEncoderWeights {
    let mut r = data::rng::Lcg::new(seed);
    let (layers, hidden, out_dim) = (cfg.num_condition_layers as usize, cfg.condition_hidden_dim as usize, cfg.out_dim as usize);
    ConditionEncoderWeights {
        layer_weight_logits: r.vec_scaled(layers, 0.5),
        layer_scale: 1.0,
        proj_weight: r.vec_scaled(out_dim * hidden * 3, 0.02),
        proj_bias: r.vec_scaled(out_dim, 0.01),
    }
}

/// One arm: denoise one chunk under `place`, returning its latents and the
/// per-Euler-step wall clock the progress callback observed.
///
/// The timing comes from the progress sink because that is the ONE point
/// per step that runs on the orchestrating thread after both branches have
/// joined - exactly the boundary a "seconds per Euler step" number should
/// be measured across, and it needs no instrumentation inside the loop.
fn run(place: Placement, label: &str, dit_cfg: &DitConfig, dit_w: &dit::DitWeights, cond_cfg: &ConditionEncoderConfig, cond_w: &ConditionEncoderWeights, hiddens: &[f32]) -> (Vec<f32>, Vec<f64>) {
    let devices = CfgDevices::open_placed(place, None);
    assert_eq!(devices.is_parallel(), place.cfg_is_parallel(), "{label}: the opened handles must match the placement");
    let length = minimaxmusic3::condition_encoder::latent_length(cond_cfg, FRAMES);
    let mut residents = denoise::ChunkResidents::new(&devices, dit_cfg, dit_w, length);
    let mut state = ChunkState::default();
    let mut marks: Vec<Instant> = Vec::new();
    let t0 = Instant::now();
    let latents = denoise::denoise_chunk(&mut residents, dit_cfg, dit_w, cond_cfg, cond_w, hiddens, FRAMES, 0, &mut state, STEPS, SEED, &mut |_, _, _| marks.push(Instant::now()));
    let setup = marks.first().map(|m| m.duration_since(t0).as_secs_f64()).unwrap_or_default();
    let per_step: Vec<f64> = marks.windows(2).map(|w| w[1].duration_since(w[0]).as_secs_f64()).collect();
    eprintln!("[{label}] per-card Resident upload + first step {setup:.2} s, then per step: {}", per_step.iter().map(|s| format!("{s:.2}")).collect::<Vec<_>>().join(" "));
    (latents, per_step)
}

/// The gate. Same seed, same weights, same chunk - only the placement
/// differs, and every latent must match bit for bit.
///
/// The wall times are reported, never asserted on: a correctness gate that
/// fails when the box is loaded is a flaky gate. The measured figure belongs
/// in this crate's roadmap ledger, not in an assertion here.
///
/// `#[ignore]`d, unlike this crate's other real-weight tests, because it is
/// the only one that needs BOTH cards at once: ~9.7 GB of resident weights
/// on each for ~2.5 minutes. Run inside a general `make test` it would
/// contend with every other GPU test in the suite for a claim that does not
/// change between runs. Run it explicitly (see this module's doc), on a box
/// whose cards are otherwise idle - a neighbour holding a few GB lands
/// entirely on the concurrent arm and under-reports the speedup.
#[test]
#[ignore = "needs the real DiT checkpoint (BRAIN_MINIMAXMUSIC3_DIT) and ~9.7 GB on each of two idle cards"]
fn concurrent_two_card_cfg_is_bit_identical_to_sequential_one_card() {
    let Some(dir) = std::env::var("BRAIN_MINIMAXMUSIC3_DIT").ok().filter(|s| !s.is_empty()) else {
        brain_testutil::skip("set BRAIN_MINIMAXMUSIC3_DIT to the real transformer directory to run this gate");
        return;
    };
    let dit_cfg = DitConfig::real();
    let dit_w = match dit::import(&dir, &dit_cfg) {
        Ok(w) => w,
        Err(e) => {
            brain_testutil::skip(&format!("cannot import the DiT at {dir}: {e}"));
            return;
        }
    };

    let place = DevicePlan::Auto.resolve(None);
    eprintln!("cfg placement: cond={:?} uncond={:?} genuinely concurrent={}", place.cond, place.uncond, place.cfg_is_parallel());
    if !place.cfg_is_parallel() {
        brain_testutil::skip_unavailable("fewer than two schedulable GPUs (or BRAIN_MINIMAXMUSIC3_CFG_PARALLEL=0): there is no concurrent arm to compare");
        return;
    }

    let cond_cfg = ConditionEncoderConfig::real();
    let cond_w = random_condition_weights(&cond_cfg, 0xC0FFEE);
    let per_frame = (cond_cfg.num_condition_layers * cond_cfg.condition_hidden_dim) as usize;
    let hiddens = data::rng::Lcg::new(0xBEEF).vec_scaled(FRAMES * per_frame, 0.3);

    let (seq, seq_steps) = run(Placement::single(), "sequential, one card", &dit_cfg, &dit_w, &cond_cfg, &cond_w, &hiddens);
    let (par, par_steps) = run(place, "concurrent, two cards", &dit_cfg, &dit_w, &cond_cfg, &cond_w, &hiddens);

    assert_eq!(seq.len(), par.len(), "same latent count");
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    if bits(&seq) != bits(&par) {
        let diff = seq.iter().zip(&par).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let worst = seq.iter().zip(&par).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        panic!("concurrent CFG dispatch changed the chunk - {diff} of {} latents differ, worst |delta| {worst:e}. This is a real defect, not a tolerance to widen (see this module's doc).", seq.len());
    }

    // Best-of-N with the first step (every first-touch cost on both cards)
    // already excluded by construction: the minimum is the least
    // contaminated sample, and a mean over a shared box measures the box.
    let best = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, f64::min);
    let (s, p) = (best(&seq_steps), best(&par_steps));
    eprintln!("bit-identical over {} latents. Best Euler step: {s:.2} s sequential vs {p:.2} s concurrent = {:.2}x", seq.len(), s / p);

    // A chunk of constant latents would make the comparison vacuous: any two
    // runs of a frozen loop agree. Assert the output really varies first.
    let flat = seq.iter().all(|&x| x == seq[0]);
    assert!(!flat, "every latent is identical - the bit-identity assertion above is vacuous");
}
