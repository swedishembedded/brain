// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Overlapping a block stack's HOST work with the previous block's DEVICE work
//! computes the SAME function - bit for bit, not within a tolerance.
//!
//! Swedish Embedded AB implements host/device pipelining for inference engines
//! for its clients. If your team needs expertise in GPU queue scheduling and
//! the buffer-lifetime rules that make overlap safe then you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! `ltxv::block::block_pipeline` moves the per-block blocking wait from AFTER
//! the block's own submit to BEFORE it. Nothing arithmetic changes and no
//! dispatch is reordered, so `assert_eq!` on the raw output bits is the
//! statement the code actually makes and a cosine floor would be a weaker one.
//!
//! ## Why this is a FOUR-arm test and not a two-arm one
//!
//! The pipelining and the scratch arena (`BRAIN_LTXV_NO_SCRATCH_POOL`,
//! `crates/ltxv/tests/scratch_pool.rs`) interact, and the interaction is the
//! whole risk. The arena hands block `l+1` the buffers block `l` just used; the
//! pipelining lets block `l` still be running when block `l+1` asks for them.
//! Either alone is safe for a reason that does not mention the other, so a gate
//! that only ever ran them together could not say which arm a failure came
//! from. All four combinations must produce the identical bits, and the arm
//! with both switches OFF is the reference every other arm is compared against.
//!
//! ## What has to be true for this file to mean anything
//!
//! * both switches are read per block forward, never cached, so setting one
//!   between two calls really does select the other arm - a `OnceLock` in
//!   either would make this file compare one arm against itself and pass
//!   forever;
//! * the stack has to be deep enough for the overlap to have somewhere to go
//!   wrong. Three layers is the minimum: block 2 is the first that is handed a
//!   slot block 0 retired, and it is also the first recorded while a block
//!   other than the immediately preceding one could still be in flight.

use ltxv::block::{GenerationCache, QTier};
use ltxv::dit::{forward_q_streamed, load_head_tensors_from_source, random_tiny_weights};
use ltxv::modelgrad::Cfg;
use ltxv::LtxDitConfig;

const NO_PIPELINE: &str = "BRAIN_LTXV_NO_PIPELINE";
const NO_ARENA: &str = "BRAIN_LTXV_NO_SCRATCH_POOL";

/// Both switches are process-wide, so the tests in this binary must not be in
/// different arms at the same time - the suite runs a binary's tests in
/// parallel by default, and one test clearing a variable while another relies
/// on it would silently compare an arm against itself.
static ARM: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Run `f` with the two switches forced to the given arm, restoring the
/// ambient setting afterwards. Callers hold [`ARM`].
fn arm<T>(pipeline: bool, arena: bool, f: impl FnOnce() -> T) -> T {
    let set = |k: &str, on: bool| match on {
        true => std::env::remove_var(k),
        false => std::env::set_var(k, "1"),
    };
    set(NO_PIPELINE, pipeline);
    set(NO_ARENA, arena);
    let out = f();
    std::env::remove_var(NO_PIPELINE);
    std::env::remove_var(NO_ARENA);
    out
}

/// Bits, not values: `0.0 == -0.0` is true for two different results and
/// `NaN == NaN` is false for the same one, so a `==` over `f32` is not the
/// relation "these two forwards produced the same answer".
fn differing_bits(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len(), "differing_bits: length mismatch ({} vs {})", a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
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

#[test]
fn pipelining_changes_no_bit_of_a_streamed_forward_with_or_without_the_arena() {
    let _arm = ARM.lock().unwrap_or_else(|e| e.into_inner());
    let cfg = LtxDitConfig { num_layers: 3, ..LtxDitConfig::tiny() };
    let w = random_tiny_weights(&cfg, 0x9B_31_04_7F);
    let head = load_head_tensors_from_source(&w, &cfg);
    let i = synthetic_inputs(&cfg, 7, 5);

    #[rustfmt::skip]
    let run = || {
        let cache = GenerationCache::default();
        forward_q_streamed(&cfg, &w, &head, None, QTier::Int8, &i.latent, &i.timesteps, &i.positions, &i.keyframes_mask, &i.context, i.context_len, i.t, &i.context_valid, &cache)
    };

    // The reference arm: no overlap, no recycling. Run twice first, because a
    // file that cannot tell a pipelining bug from ordinary nondeterminism is
    // not a gate.
    let reference = arm(false, false, run);
    let reference_again = arm(false, false, run);
    assert_eq!(
        differing_bits(&reference, &reference_again),
        0,
        "test setup: two forwards of the same weights and inputs on the fully serial arm must already agree bit for bit"
    );

    for (pipeline, arena) in [(false, true), (true, false), (true, true)] {
        let got = arm(pipeline, arena, run);
        assert!(got.iter().all(|v| v.is_finite()), "arm (pipeline={pipeline}, arena={arena}) produced non-finite output");
        assert_eq!(
            differing_bits(&reference, &got),
            0,
            "arm (pipeline={pipeline}, arena={arena}) changed the forward's output. Pipelining moves WHEN the host waits and the arena moves WHEN device memory is allocated; neither changes what any kernel reads, so any differing bit is an aliasing or a stale-contents bug"
        );
    }
}

// ------------------------------------------- the REAL block's own shapes

const REPO: &str = "Lightricks/LTX-2.5";

/// The real distilled Q8_0 DiT, resolved the way every other real-weight gate
/// in this crate resolves it: the env var first, then the model store,
/// discriminating on the file's own declared architecture rather than on its
/// name - the store holds a second Q8_0 GGUF (the quantized Gemma-4 text
/// tower) that a name glob picks up instead.
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
/// The distinction is not pedantic for pipelining specifically. What overlaps
/// the card here is the REAL per-block host work - a hundred-odd bind groups
/// over int8 operands and per-row scale buffers, at widths the tiny config does
/// not reach - and the arena slots being handed forward while a block is still
/// running are that block's real temporaries, spanning three orders of
/// magnitude in size.
#[test]
fn pipelining_changes_no_bit_of_a_real_weight_forward() {
    let _arm = ARM.lock().unwrap_or_else(|e| e.into_inner());
    let Some(path) = gguf_path() else {
        brain_testutil::skip(&format!("set BRAIN_LTXV_DIT to a real {REPO} distilled Q8_0 GGUF (none in the model store)"));
        return;
    };
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

    let reference = arm(false, false, run);
    let shipped = arm(true, true, run);
    assert!(shipped.iter().all(|v| v.is_finite()), "the pipelined arm produced non-finite output on real weights");
    println!("real-weight pipelining parity: {} of {} output words differ", differing_bits(&reference, &shipped), reference.len());
    assert_eq!(
        differing_bits(&reference, &shipped),
        0,
        "pipelining changed a REAL-weight forward's output - it only changes WHEN the host waits for the card, never what any kernel reads"
    );
}
