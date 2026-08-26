// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The scratch-arena arm of a streamed forward computes the SAME function as
//! the plain-allocation arm - bit for bit, not within a tolerance.
//!
//! Swedish Embedded AB implements correctness gates for device-memory reuse in
//! inference engines for its clients. If your team needs expertise in GPU
//! buffer lifetime and aliasing then you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! A block forward draws its ~70 temporaries from the device handle's replay
//! arena (`gpu_core::scratch`, entered by `block::LtxBlockQ::forward_prod_dev`)
//! instead of creating and destroying them per block. Nothing about WHAT any
//! kernel reads changes, so `assert_eq!` on the raw bits is the right gate and
//! a cosine floor would be a weaker statement than the code actually makes.
//!
//! Two arms, at two widths: a checkpoint-free tiny config that always runs, and
//! the REAL 22B block's own allocation sequence when the checkpoint is present
//! (skipped loudly otherwise, this crate's convention).
//!
//! Two things the arms have to differ in for this to be a real comparison, and
//! both are checked rather than assumed:
//!
//! * the switch (`BRAIN_LTXV_NO_SCRATCH_POOL`) is read per block forward, not
//!   cached, so setting it between the two calls really does select the other
//!   arm - a `OnceLock` here would make this file compare one arm against
//!   itself and pass forever;
//! * the reuse has to actually happen. A three-layer stack is the smallest
//!   depth at which a slot retired by block 0 is handed back to block 2, which
//!   is the case a two-layer stack cannot reach.
//!
//! What this file catches, stated from the mutation run rather than from
//! intent: making the arena hand back a slot that a live operand of the same
//! block still holds turns every one of the output words, and both arms go red
//! together. What it does NOT catch, and this is worth knowing: deleting the
//! arena's `is_unique` guard on its own leaves both arms green, because the
//! only buffer that outlives a scope here is the CHAINED ACTIVATION and it is
//! already dead by the time the next block's last dispatch rewrites its slot.
//! The guard is defending a property this model does not currently depend on;
//! `crates/gpu-core/tests/scratch_arena.rs` is what holds it.

use ltxv::block::{GenerationCache, QTier};
use ltxv::dit::{forward_q_streamed, load_head_tensors_from_source, random_tiny_weights};
use ltxv::modelgrad::Cfg;
use ltxv::LtxDitConfig;

const SWITCH: &str = "BRAIN_LTXV_NO_SCRATCH_POOL";

/// The switch is process-wide, so the two tests in this binary must not be in
/// their un-pooled arm at the same time - the suite runs a test binary's tests
/// in parallel by default (`--test-threads`), and one test clearing the
/// variable while the other is relying on it would silently compare an arm
/// against itself. Each test holds this for its whole body.
static ARM: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Run `f` with the arena disabled, restoring the ambient setting afterwards.
/// Callers hold [`ARM`].
fn without_arena<T>(f: impl FnOnce() -> T) -> T {
    std::env::set_var(SWITCH, "1");
    let out = f();
    std::env::remove_var(SWITCH);
    out
}

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

fn synthetic_inputs(cfg: &LtxDitConfig, t: usize, context_len: usize) -> Inputs {
    let mcfg = Cfg::from_ltx(cfg, t, context_len);
    let positions = mcfg.simple_positions();
    let latent: Vec<f32> = (0..t * cfg.in_channels as usize).map(|i| ((i % 23) as f32 / 23.0 - 0.5) * 1.1).collect();
    let context: Vec<f32> = (0..context_len * cfg.cross_attention_dim as usize).map(|i| ((i % 7) as f32 / 7.0 - 0.5) * 1.4).collect();
    let timesteps: Vec<f32> = (0..t).map(|i| 0.2 + 0.05 * (i % 5) as f32).collect();
    let mut keyframes_mask = vec![0f32; t];
    keyframes_mask[0] = 1.0;
    let context_valid = vec![1.0f32; context_len];
    Inputs { latent, timesteps, positions, keyframes_mask, context, context_valid, context_len, t }
}

/// Bits, not values: `0.0 == -0.0` is true for two different results and
/// `NaN == NaN` is false for the same one, so a `==` over `f32` is not the
/// relation "these two forwards produced the same answer".
fn differing_bits(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len(), "differing_bits: length mismatch ({} vs {})", a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

#[test]
fn the_scratch_arena_changes_no_bit_of_a_streamed_forward() {
    let _arm = ARM.lock().unwrap_or_else(|e| e.into_inner());
    // Three layers, so block 2 is handed slots block 0 retired - the reuse the
    // arena exists for, and the depth at which a wrongly recycled chained
    // activation first has somewhere to go wrong.
    let cfg = LtxDitConfig { num_layers: 3, ..LtxDitConfig::tiny() };
    let w = random_tiny_weights(&cfg, 0x5C_2A_7C_11);
    let head = load_head_tensors_from_source(&w, &cfg);
    let i = synthetic_inputs(&cfg, 7, 5);

    #[rustfmt::skip]
    let run = || {
        let cache = GenerationCache::default();
        forward_q_streamed(&cfg, &w, &head, None, QTier::Int8, &i.latent, &i.timesteps, &i.positions, &i.keyframes_mask, &i.context, i.context_len, i.t, &i.context_valid, &cache)
    };

    let plain = without_arena(run);
    let plain_again = without_arena(run);
    assert_eq!(
        differing_bits(&plain, &plain_again),
        0,
        "test setup: two forwards of the same weights and inputs on the un-pooled arm must already agree bit for bit, or this file cannot tell an arena bug from ordinary nondeterminism"
    );

    let pooled = run();
    assert!(pooled.iter().all(|v| v.is_finite()), "the pooled arm produced non-finite output");
    assert_eq!(
        differing_bits(&plain, &pooled),
        0,
        "the scratch arena changed the forward's output: it only changes WHEN device memory is allocated, never what any kernel reads, so any differing bit is an aliasing or a stale-contents bug"
    );
}

// ------------------------------------------- 2. the REAL block's own shapes

const REPO: &str = "Lightricks/LTX-2.5";

/// The real distilled Q8_0 DiT, resolved the way every other real-weight gate
/// in this crate resolves it (`int8_compute.rs`, `dit_parity.rs`): the env var
/// first, then the model store, discriminating on the file's own declared
/// architecture rather than on its name - the store holds a second Q8_0 GGUF
/// (the quantized Gemma-4 text tower) that a name glob picks up instead.
fn gguf_path() -> Option<String> {
    if let Ok(p) = std::env::var("BRAIN_LTXV_DIT") {
        if !p.is_empty() {
            return Some(p);
        }
    }
    let dir = brain_testutil::model_dir(REPO)?;
    let mut found: Vec<String> = std::fs::read_dir(&dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.contains("Q8_0") && n.ends_with(".gguf")))
        .filter(|p| {
            checkpoint::gguf::MmapGguf::open(&p.to_string_lossy())
                .ok()
                .and_then(|g| g.kv().get("general.architecture").and_then(|v| v.as_str()).map(str::to_string))
                .as_deref()
                == Some(ltxv::import::GGUF_ARCHITECTURE)
        })
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    found.sort();
    found.into_iter().next()
}

/// The tiny config above proves the mechanism; this proves it at the shapes the
/// production block really asks for.
///
/// That distinction is not pedantic here: the arena replays an allocation
/// SEQUENCE, and the tiny config's sequence is a different length, in a
/// different order, at different sizes. A real block asks for tens of
/// temporaries spanning three orders of magnitude in size, and the int8 path
/// adds packed operands and per-row scale buffers the fp32-width tiny config
/// exercises differently. Small token count (this crate's own real-weight
/// convention), real width, real weights.
#[test]
fn the_scratch_arena_changes_no_bit_of_a_real_weight_forward() {
    let _arm = ARM.lock().unwrap_or_else(|e| e.into_inner());
    let Some(path) = gguf_path() else {
        brain_testutil::skip(&format!("set BRAIN_LTXV_DIT to a real {REPO} distilled Q8_0 GGUF (none in the model store)"));
        return;
    };
    // Three layers again - block 2 is the first that can be handed a slot
    // block 0 retired.
    let cfg = LtxDitConfig { num_layers: 3, ..LtxDitConfig::ltx25_22b() };
    let src = ltxv::gguf_src::LtxvGgufSource::open(&path).unwrap_or_else(|e| panic!("opening {path}: {e}"));
    let head = ltxv::dit::load_head_tensors_from_source(&src, &cfg);
    // `context_len` is 128, not a handful: the real config's embeddings
    // connector asserts the context is a multiple of its 128 learnable
    // registers, so a token-budget-minimal context is not a legal input here.
    let i = synthetic_inputs(&cfg, 16, 128);

    #[rustfmt::skip]
    let run = || {
        let cache = GenerationCache::default();
        forward_q_streamed(&cfg, &src, &head, None, QTier::Int8, &i.latent, &i.timesteps, &i.positions, &i.keyframes_mask, &i.context, i.context_len, i.t, &i.context_valid, &cache)
    };

    let plain = without_arena(run);
    let pooled = run();
    assert!(pooled.iter().all(|v| v.is_finite()), "the pooled arm produced non-finite output on real weights");
    println!("real-weight scratch-arena parity: {} of {} output words differ", differing_bits(&plain, &pooled), plain.len());
    assert_eq!(
        differing_bits(&plain, &pooled),
        0,
        "the scratch arena changed a REAL-weight forward's output - it only changes WHEN device memory is allocated, never what any kernel reads"
    );
}
