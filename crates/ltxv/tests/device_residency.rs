// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `ltxv::devres`: keeping the DiT's already-quantized blocks resident on the
//! CARD across a generation's denoise steps must change WHEN bytes move and
//! nothing else.
//!
//! Gated on BIT PATTERNS, not a cosine threshold and not `==` on `f32` (which
//! calls two NaNs unequal) - a resident forward is supposed to compute the
//! exact same function as a streaming one, just skip re-uploading ~270 MB per
//! block per step to get there. This extends to the DEVICE tier the same proof
//! style `block_weight_cache.rs` established for the HOST tier.
//!
//! Three things each gate has to do, because each is a way this could pass
//! vacuously:
//!
//! 1. compare the two arms bit for bit;
//! 2. assert residency ACTUALLY engaged (a window that silently declined to
//!    build would agree perfectly with the arm it is supposed to differ from
//!    in behaviour);
//! 3. cover the NARROW window too - partial residency is the graceful-degrade
//!    path, so it is production code, not a theoretical branch.

use ltxv::block::{GenerationCache, QTier};
use ltxv::devres::DitSession;
use ltxv::dit::{forward_q_streamed_in, load_head_tensors_from_source, random_tiny_weights};
use ltxv::modelgrad::Cfg;
use ltxv::LtxDitConfig;

struct Inputs {
    latent: Vec<f32>,
    timesteps: Vec<f32>,
    positions: Vec<f32>,
    keyframes_mask: Vec<f32>,
    context: Vec<f32>,
    context_valid: Vec<f32>,
    context_len: usize,
    t: usize,
}

fn synthetic_inputs(cfg: &LtxDitConfig, t: usize, context_len: usize, salt: f32) -> Inputs {
    let mcfg = Cfg::from_ltx(cfg, t, context_len);
    let positions = mcfg.simple_positions();
    let latent: Vec<f32> = (0..t * cfg.in_channels as usize).map(|i| ((i % 23) as f32 / 23.0 - 0.5) * 1.1 + salt).collect();
    let context: Vec<f32> = (0..context_len * cfg.cross_attention_dim as usize).map(|i| ((i % 7) as f32 / 7.0 - 0.5) * 1.4 - salt).collect();
    let timesteps: Vec<f32> = (0..t).map(|i| 0.2 + 0.05 * (i % 5) as f32).collect();
    let mut keyframes_mask = vec![0f32; t];
    keyframes_mask[0] = 1.0;
    let context_valid = vec![1.0f32; context_len];
    Inputs { latent, timesteps, positions, keyframes_mask, context, context_valid, context_len, t }
}

/// Bit patterns, not values: `assert_eq!` on `f32` would call two NaNs
/// unequal and two differently-signed zeros equal, and neither is what
/// "identical" means for a device-placement change.
fn bits_differ(a: &[f32], b: &[f32]) -> Option<(usize, f32, f32)> {
    assert_eq!(a.len(), b.len(), "length mismatch ({} vs {})", a.len(), b.len());
    a.iter().zip(b).enumerate().find(|(_, (x, y))| x.to_bits() != y.to_bits()).map(|(i, (x, y))| (i, *x, *y))
}

#[allow(clippy::too_many_arguments)]
fn run(session: &DitSession, cfg: &LtxDitConfig, src: &dyn checkpoint::TensorSource, head: &std::collections::HashMap<String, (Vec<usize>, Vec<f32>)>, i: &Inputs, cache: &GenerationCache) -> Vec<f32> {
    forward_q_streamed_in(session, cfg, src, head, QTier::Int8, &i.latent, &i.timesteps, &i.positions, &i.keyframes_mask, &i.context, i.context_len, i.t, &i.context_valid, cache)
}

/// The core claim, at a full window: N forwards through ONE resident session
/// must be bit-identical, forward by forward, to the same N forwards through
/// the transient (open-a-device-and-re-upload-everything-every-call) path -
/// including forwards whose INPUTS differ, which is what a real denoise loop
/// looks like.
///
/// The residency assertions are what stop this passing vacuously: with a slot
/// per block the session must upload each block exactly ONCE across all three
/// forwards, and serve every other visit from the card.
#[test]
fn a_device_resident_forward_is_bit_identical_to_the_streaming_one() {
    let cfg = LtxDitConfig { num_layers: 4, ..LtxDitConfig::tiny() };
    let w = random_tiny_weights(&cfg, 0x00DE_1CE1);
    let head = load_head_tensors_from_source(&w, &cfg);
    let steps: Vec<Inputs> = (0..3).map(|s| synthetic_inputs(&cfg, 7, 5, s as f32 * 0.07)).collect();

    // Arm A: the pre-residency path - a session that keeps nothing.
    let stream = DitSession::transient(None);
    let cache_a = GenerationCache::default();
    let a: Vec<Vec<f32>> = steps.iter().map(|i| run(&stream, &cfg, &w, &head, i, &cache_a)).collect();
    assert_eq!(stream.stats().slots, 0, "the transient arm must hold no device slots at all");

    // Arm B: ONE session, held across all three forwards, with a slot per
    // block - what `RealDit` does for a whole generation.
    let resident = DitSession::resident_with_slots(None, cfg.num_layers);
    let cache_b = GenerationCache::default();
    let b: Vec<Vec<f32>> = steps.iter().map(|i| run(&resident, &cfg, &w, &head, i, &cache_b)).collect();

    for (s, (x, y)) in a.iter().zip(&b).enumerate() {
        if let Some((i, x, y)) = bits_differ(x, y) {
            panic!("forward {s}, element {i}: device residency changed the answer ({x:e} vs {y:e}, bits {:#x} vs {:#x})", x.to_bits(), y.to_bits());
        }
    }

    let rs = resident.stats();
    assert_eq!(rs.slots, cfg.num_layers, "a full window must claim one slot per block");
    assert_eq!(rs.uploads, cfg.num_layers as u64, "every block must reach the card exactly once for the WHOLE session, not once per forward");
    assert_eq!(rs.hits, cfg.num_layers as u64 * steps.len() as u64, "with the window pre-filled, EVERY block visit of every forward is served from resident weights");

    // The inputs really did differ, so the bit-identity above is not the
    // trivial "same input, same output" statement.
    assert!(bits_differ(&a[0], &a[1]).is_some(), "test setup: three forwards at different latents must not all produce the same answer");
}

/// The graceful-degrade path: a window NARROWER than the model must still be
/// bit-identical, and must still upload strictly less than the streaming arm.
///
/// This is where a slot-assignment bug lives - a window that handed back the
/// WRONG resident block would run the model with layer `j`'s weights at layer
/// `i`, which the bit comparison catches and a "did it complete" check would
/// not.
#[test]
fn a_narrow_window_is_bit_identical_and_still_uploads_less_than_streaming() {
    let cfg = LtxDitConfig { num_layers: 6, ..LtxDitConfig::tiny() };
    let w = random_tiny_weights(&cfg, 0x00DE_1CE2);
    let head = load_head_tensors_from_source(&w, &cfg);
    let steps: Vec<Inputs> = (0..3).map(|s| synthetic_inputs(&cfg, 6, 5, s as f32 * 0.11)).collect();

    let stream = DitSession::transient(None);
    let cache_a = GenerationCache::default();
    let a: Vec<Vec<f32>> = steps.iter().map(|i| run(&stream, &cfg, &w, &head, i, &cache_a)).collect();

    // Three slots for six blocks: `CyclicScan` pins two and rotates four
    // through the third.
    let slots = 3u32;
    let narrow = DitSession::resident_with_slots(None, slots);
    let cache_b = GenerationCache::default();
    let b: Vec<Vec<f32>> = steps.iter().map(|i| run(&narrow, &cfg, &w, &head, i, &cache_b)).collect();

    for (s, (x, y)) in a.iter().zip(&b).enumerate() {
        if let Some((i, x, y)) = bits_differ(x, y) {
            panic!("forward {s}, element {i}: a partially-resident window changed the answer ({x:e} vs {y:e})");
        }
    }

    let rs = narrow.stats();
    assert_eq!(rs.slots, slots);
    let streaming_uploads = cfg.num_layers as u64 * steps.len() as u64;
    assert!(rs.uploads < streaming_uploads, "a narrow window must still upload less than streaming ({} vs {streaming_uploads})", rs.uploads);
    assert!(rs.hits > 0, "a narrow window must serve SOMETHING from the card, or it is not a window");
    // Exact, not a bound: `CyclicScan` pins `slots - 1` blocks (they upload
    // once, ever) and reloads the remaining tail once per pass.
    let pinned = (slots - 1) as u64;
    let tail = cfg.num_layers as u64 - pinned;
    assert_eq!(rs.uploads, pinned + tail * steps.len() as u64, "the pinned prefix must upload once and only the tail reload per pass");
}

/// Zero slots is the terminal fallback a tight `--limit-vram-total`, a huge
/// token count or an unschedulable device produces. It must be a working
/// forward, not an error - and still bit-identical, since it IS the streaming
/// path with the device merely held open.
#[test]
fn a_zero_slot_session_falls_back_to_streaming_and_is_still_exact() {
    let cfg = LtxDitConfig { num_layers: 3, ..LtxDitConfig::tiny() };
    let w = random_tiny_weights(&cfg, 0x00DE_1CE3);
    let head = load_head_tensors_from_source(&w, &cfg);
    let i = synthetic_inputs(&cfg, 5, 5, 0.0);

    let stream = DitSession::transient(None);
    let cache_a = GenerationCache::default();
    let a = run(&stream, &cfg, &w, &head, &i, &cache_a);

    let starved = DitSession::resident_with_slots(None, 0);
    let cache_b = GenerationCache::default();
    let b = run(&starved, &cfg, &w, &head, &i, &cache_b);
    assert!(bits_differ(&a, &b).is_none(), "a zero-slot session must be bit-identical to the transient one");
    assert_eq!(starved.stats(), Default::default(), "a zero-slot session builds no window and reports no residency");
}

/// The other half of the claim, and the one the two arms above CANNOT make:
/// both of them run the chained, on-device modulation path, so neither can see
/// a bug in `adaln_row` itself.
///
/// This one compares the chained block stack against the EAGER
/// `LtxDit::forward_q`, which goes through `LtxBlockQ::forward` and therefore
/// through `dit::adaln::add_table` + `slice_mod` on the HOST followed by nine
/// per-block uploads - the reference definition of the modulation arithmetic.
/// Bit patterns, because `adaln_row` reproduces that arithmetic exactly
/// (`tbl[i] + tab[..]`, then `1.0 + x` for the three `1 + scale` rows) rather
/// than approximately.
///
/// Mutation-verified: swapping any two of `MOD_ROWS`' nine row indices, or
/// dropping the `plus_one` add, moves this to a gross disagreement.
#[test]
fn on_device_modulation_is_bit_identical_to_the_host_combine_and_slice() {
    let cfg = LtxDitConfig { num_layers: 3, ..LtxDitConfig::tiny() };
    let w = random_tiny_weights(&cfg, 0x00DE_1CE4);
    let head = load_head_tensors_from_source(&w, &cfg);
    let i = synthetic_inputs(&cfg, 9, 5, 0.0);

    // Eager: host `add_table` + `slice_mod` + nine uploads, per block.
    let eager = ltxv::dit::LtxDit::new(cfg, w.clone(), None);
    let want = eager.forward_q(&i.latent, &i.timesteps, &i.positions, &i.keyframes_mask, &i.context, i.context_len, i.t, &i.context_valid, QTier::Int8).out;

    // Chained: `adaln_row` on the card, from one per-forward table upload.
    let session = DitSession::resident_with_slots(None, cfg.num_layers);
    let cache = GenerationCache::default();
    let got = run(&session, &cfg, &w, &head, &i, &cache);

    if let Some((idx, x, y)) = bits_differ(&want, &got) {
        panic!("element {idx}: on-device adaLN modulation changed the answer ({x:e} host vs {y:e} device, bits {:#x} vs {:#x})", x.to_bits(), y.to_bits());
    }
    assert!(want.iter().any(|v| *v != 0.0), "test setup: an all-zero reference would make any comparison vacuous");
}

/// The VRAM policy must never promise more blocks than the model has, must
/// shrink as the token count grows, and must reach zero rather than a negative
/// count on a device too small for one forward's activations - the three ways
/// a budget formula turns into a driver-level abort.
#[test]
fn the_slot_policy_never_over_promises() {
    let cfg = LtxDitConfig::ltx25_22b();
    let per_block = ltxv::block::cached_block_bytes(&cfg, QTier::Int8);
    let fit = |cap: u64, t: usize| ((cap.saturating_sub(ltxv::devres::activation_reserve_bytes(t, "wgpu")) / per_block) as u32).min(cfg.num_layers);
    // 512x512-scale shapes fit the whole model; the real 720p/1080p token
    // counts do NOT on a 24 GiB card under the wgpu backend, and the policy
    // must say so rather than plan a window that aborts - see this crate's
    // roadmap ledger, Phase 18, for the measured plateau this reserve encodes.
    assert_eq!(fit(24 << 30, 1000), cfg.num_layers, "a 512x512-scale token count must fit every block on a 24 GiB card");
    assert!(fit(24 << 30, 3520) > 0 && fit(24 << 30, 3520) < cfg.num_layers, "720p must get a PARTIAL window on a 24 GiB card, neither zero nor all 48");
    assert_eq!(fit(24 << 30, 8160), 0, "1080p leaves no room for a resident block on a 24 GiB card, and the policy must ask for none");
    assert!(fit(24 << 30, 8160) <= cfg.num_layers);
    assert!(fit(24 << 30, 8160) <= fit(24 << 30, 3520), "more tokens must never buy more resident blocks");
    assert_eq!(fit(2 << 30, 3520), 0, "a card that cannot hold one forward's activations must ask for zero slots");
}

// --------------------------------------------------------------- real weights

/// The same bit-identity claim against the REAL production checkpoint's own
/// bytes at reduced depth - because `random_tiny_weights` cannot exercise the
/// real per-block footprint, the real `apply_gated_attention` path, or the
/// real embeddings connector, and this phase changes exactly how those bytes
/// reach the card.
///
/// `#[ignore]`d: needs `BRAIN_LTXV_DIT` and a real card.
mod real_weight {
    use super::*;

    #[test]
    #[ignore = "needs BRAIN_LTXV_DIT (the real 22B Q8_0 GGUF) and a GPU"]
    fn a_resident_real_checkpoint_forward_is_bit_identical_to_the_streaming_one() {
        let path = std::env::var("BRAIN_LTXV_DIT").expect("set BRAIN_LTXV_DIT to the real 22B distilled Q8_0 GGUF");
        let src = ltxv::gguf_src::LtxvGgufSource::open(&path).expect("opening the real DiT GGUF");
        let cfg = LtxDitConfig { num_layers: 2, ..src.config().video };
        let head = load_head_tensors_from_source(&src, &cfg);
        // 128 tokens, 128 context (the connector's own register count, its
        // minimum legal context width) - small enough to run in seconds, wide
        // enough to be the real 4096-dim block.
        let steps: Vec<Inputs> = (0..3).map(|s| synthetic_inputs(&cfg, 128, 128, s as f32 * 0.05)).collect();

        // ONE host cache for both arms: this gate is about the DEVICE tier, so
        // the host tier must be held constant rather than re-measured.
        let cache = GenerationCache::default();
        let stream = DitSession::transient(Some("gpu"));
        let a: Vec<Vec<f32>> = steps.iter().map(|i| run(&stream, &cfg, &src, &head, i, &cache)).collect();

        let resident = DitSession::resident_with_slots(Some("gpu"), cfg.num_layers);
        let b: Vec<Vec<f32>> = steps.iter().map(|i| run(&resident, &cfg, &src, &head, i, &cache)).collect();

        for (s, (x, y)) in a.iter().zip(&b).enumerate() {
            if let Some((i, x, y)) = bits_differ(x, y) {
                panic!("real-weight forward {s}, element {i}: device residency changed the answer ({x:e} vs {y:e})");
            }
        }
        let rs = resident.stats();
        assert_eq!(rs.uploads, cfg.num_layers as u64, "the real blocks must reach the card once for the whole session");
        assert_eq!(rs.hits, cfg.num_layers as u64 * steps.len() as u64);
        assert!(a[0].iter().all(|v| v.is_finite()), "the real-weight forward must produce finite output");
    }
}
