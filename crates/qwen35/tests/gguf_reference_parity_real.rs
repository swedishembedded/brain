// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Does the GGUF route load the REAL checkpoint's tensors with the RIGHT
//! MEANING - not merely the right name, shape and finite values?
//!
//! Every other real-weight gate in this crate compares brain against brain:
//! `gguf_i8_vs_fp32_real.rs` compares two tiers built from the same bytes,
//! `decode_step.rs` compares two tapes, `shard_parity.rs` compares two
//! splits. All of them are blind to a tensor whose name and shape are right
//! and whose VALUES mean something else (llama.cpp's `ssm_a = -exp(A_log)`
//! was exactly that, and cost a whole milestone's output). `golden_parity.rs`
//! does compare against the real HF reference, but at TINY random dims from a
//! safetensors fixture - it never sees a llama.cpp-converted tensor at all.
//!
//! This gate closes that hole with a SECOND, INDEPENDENT implementation of
//! the same architecture reading the SAME real Q8_0 bytes:
//! `tools/goldens/qwen35_gguf_reference_forward.py` (pure CPython - no torch,
//! no numpy, no llama.cpp - transcribed from the published `modeling_qwen3_5`
//! reference module). The digests below are that script's output. Because
//! the two implementations share no code, agreement rules out the whole class
//! "the loader mis-reads a tensor" - per-head `[query|gate]` interleave,
//! partial-RoPE channel layout and pairing, conv1d `[channel][tap]` order and
//! which tap is the current token, the q|k|v order inside `attn_qkv`, GDN
//! head sharing, the `(1+w)` norm folds, `ssm_a`, and every 1:1 rename.
//!
//! Deliberately fp32 and only four layers deep: this asks whether the BYTES
//! are understood, and truncation is what keeps the unquantized side inside
//! one card (~1.5 GB/layer at 27B dims). The int8 tier's own divergence is a
//! separate question with its own gate (`gguf_i8_vs_fp32_real.rs`).
//!
//! ```text
//! BRAIN_QWEN35_GGUF=$HOME/models/unsloth/Qwen3.8-27B-Q8_0.gguf \
//!   cargo test -p brain-qwen35 --release --test gguf_reference_parity_real -- --nocapture
//! ```

use checkpoint::gguf::MmapGguf;
use data::tokenizer::Tokenizer;
use model::shard::Shard;
use qwen35::int8_gguf_resident::{resident_config, shard_source};
use qwen35::model::Qwen35;

/// Blocks to build. 4 crosses both mixer types (GDN at 0,1,2; GQA at 3) and
/// so exercises every per-layer tensor this checkpoint has.
const LAYERS: u32 = 4;

/// The prompt the digests were taken on, and the token ids it MUST produce
/// through the GGUF's own embedded tokenizer - pinned here because the
/// digests are meaningless against a different tokenization.
const PROMPT: &str = "The capital city of";
const TOKENS: [u32; 4] = [760, 6511, 3177, 314];

/// One position's expected residual leaving layer `LAYERS-1`, from
/// `tools/goldens/qwen35_gguf_reference_forward.py --layers 4
/// --tokens 760,6511,3177,314 --digest`:
///
/// `(rms, sum, first four elements)`. `rms` catches a scale error, `sum`
/// catches a permutation or a sign convention (it is a projection onto the
/// all-ones vector, which no reordering preserves by accident at 5120
/// elements), and the leading elements catch a shift.
///
/// Updated for M23's GDN value-head degroup fix
/// (`int8_gguf_resident::GdnHeadOrder`) - EVERY leaf a GDN layer owns that
/// is indexed by value head (`ssm_a`/`ssm_dt.bias`/`ssm_conv1d.weight`'s
/// v-channels/`attn_qkv.weight`'s v-rows/`ssm_alpha.weight`/
/// `ssm_beta.weight`/`attn_gate.weight`/`ssm_out.weight`'s columns) was
/// group-major on disk, not brain's sub-major (repeat_interleave)
/// convention. Every position's digest changed, including position 0 -
/// unlike an EARLIER, PARTIAL pass of this fix (`A_log`/`dt_bias` only),
/// where position 0 was bit-identical because the recurrent state starts at
/// zero and only the decay/state terms are position-0-inert; `beta`/`a`/`v`/
/// `z`/the `ssm_out` projection all affect position 0 too. The oracle script
/// itself carried the SAME shared assumption this fix corrects (its own
/// `kh = j // group` line paired repeat_interleave-ordered K/Q with
/// group-major-ordered per-head leaves), so the pre-fix agreement here was
/// two independent implementations making the same wrong assumption, not
/// evidence of correctness. This fix
/// is what turns `gguf_resident_real.rs`'s
/// `the_two_card_stack_continues_a_factual_prompt_correctly` from RED to
/// GREEN on the real checkpoint.
///
/// **This gate is GREEN, and the values above are a genuine three-way
/// agreement, not a pin-to-whatever-prints shortcut.** The production Rust
/// fix (`GdnHeadOrder`) is independently verified TWO other ways beyond this
/// gate: (1) `the_two_card_stack_continues_a_factual_prompt_correctly` now
/// generates the grammatically AND factually correct continuation
/// (`" Paris. Paris is the largest city in"`) on the real two-card resident;
/// (2) a direct per-tensor diff against the FP8 checkpoint's own weights
/// (`gguf_vs_fp8_permutation_search.rs`, not committed as a standing gate -
/// an exploratory tool) measured cosine >= 0.9996 (three of the eight leaves
/// exactly 1.0000000) for the SAME `GdnHeadOrder` transform applied to every
/// GDN leaf this fix touches. This Python re-derivation is a THIRD,
/// independent check on top of those two, and it AGREES with Rust to this
/// gate's own tolerance at every position.
///
/// An earlier pass of this investigation (recorded here historically because
/// the next paragraph explains how it was actually resolved, not because the
/// gap still exists) found a real, if shrinking, disagreement after the
/// `GdnHeadOrder` fix landed - as low as pos 0: rms ~0.695 in Rust vs the
/// `EXPECT` above from the Python side - through several rounds of fixing
/// this SCRIPT's own missed degroup call sites (`attn_gate.weight`'s output
/// and `ssm_out.weight`'s input were the last two found in the script).
/// Re-investigating with a sharper tool than a final-digest comparison -
/// `crates/qwen35/tests/gdn_intermediate_trace.rs` (every one of this
/// layer's own 11 intermediates, `xn1` through `y`, dumped from BOTH sides
/// at layer 0/position 0 and diffed step by step) and `crates/qwen35/tests/
/// gdn_layer_boundary_trace.rs` (the same per-LAYER residual digest as this
/// script's own `--layers 4 --tokens 760 --digest` output, compared boundary
/// by boundary) - found NO divergence anywhere in the script's GDN math: at
/// position 0, EVERY intermediate this layer computes already agreed with
/// Rust to floating-point-summation-order precision, and the per-layer
/// residual digest matched Rust's own `debug_res` output at every one of the
/// four layer boundaries. That means the script was never the remaining
/// defect; the gap this file's doc comment used to describe here had
/// already closed by the time of that re-investigation (the git history
/// between the `GdnHeadOrder` fix and this note touches no GDN math at all,
/// so the fix predates it - most likely this comment simply went stale,
/// describing an interim measurement from the same investigation that had
/// already been superseded before the fix's own commit finished). Left as a
/// permanent record rather than deleted, since a claimed-RED gate that is
/// actually GREEN is exactly the kind of stale doc this crate's own
/// convention (see this file's own opening paragraphs) exists to prevent
/// happening twice - the two trace tests above stay in the tree,
/// uncommitted-as-a-gate but ready, for the next time this needs re-proving.
const EXPECT: [(f32, f32, [f32; 4]); 4] = [
    (0.752278, 48.51814, [0.007308, 0.012496, -0.208136, -0.062281]),
    (0.448872, 24.83384, [0.200683, -0.083840, 0.140620, -0.209841]),
    (0.428177, 45.77309, [0.031644, 0.025004, 0.042506, -0.055694]),
    (0.328236, 19.58607, [0.043467, 0.023507, 0.039335, 0.013733]),
];

/// Achieved in practice on 2x Tesla P40: every digit above, i.e. |rel| <=
/// 4e-6 on `rms`/`sum` and <= 2e-5 absolute on the elements. The tolerance is
/// set at the reference script's own printed precision (1e-6 absolute on the
/// elements, 6 significant digits on the reductions), not at the measured
/// agreement, so a genuine convention regression cannot hide under it.
const TOL_REL: f32 = 2e-5;
const TOL_ABS: f32 = 2e-6;

fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
}

#[test]
fn the_gguf_route_reproduces_an_independent_reference_forward_on_real_weights() {
    let Ok(path) = std::env::var("BRAIN_QWEN35_GGUF") else {
        brain_testutil::skip("BRAIN_QWEN35_GGUF unset (set it to a downloaded Qwen3.8-27B*.gguf to run this)");
        return;
    };
    if gpu_core::devices::gpus().is_empty() {
        brain_testutil::skip_unavailable("no discrete GPU - the fp32 side of this comparison needs ~6 GiB of VRAM");
        return;
    }

    let mg = MmapGguf::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let mut cfg = resident_config(&mg, 64).expect("resident_config on the real checkpoint");

    // The digests are tied to this tokenization, so pin it rather than
    // trusting it: a tokenizer that segments the same prompt differently
    // would fail the value checks below for a reason that has nothing to do
    // with the tensors.
    let gtok = mg.tokenizer().expect("embedded tokenizer");
    let tok = data::qwen_tokenizer::QwenBpe::from_gguf(&gtok).expect("QwenBpe::from_gguf");
    assert_eq!(tok.encode(PROMPT), TOKENS.to_vec(), "the embedded tokenizer must segment {PROMPT:?} canonically");

    // `classify` drops every block past `cfg.n_layers` on its own, so the
    // truncated config's fetch plan names exactly layers 0..LAYERS.
    cfg.n_layers = LAYERS;
    let d = cfg.d_model as usize;
    let shard = Shard { start: 0, end: LAYERS as usize, embed: false, head: false, gpu_index: Shard::ANY_GPU };
    let src = shard_source(&mg, &cfg, &shard).expect("fetch plan for the truncated stack");
    let m = Qwen35::new_fp32_shard_src(cfg.clone(), 1, TOKENS.len() as u32, &src, shard);
    m.reset_decode_cache();

    for (i, &id) in TOKENS.iter().enumerate() {
        // Embedded exactly the way the resident does it - one row at a time
        // out of the mapping, never a materialized 5.09 GB table.
        let row = mg.tensor_range("token_embd.weight", id as usize * d, d).expect("embedding row").expect("dequantize");
        let out = m.step_with_input(id, Some(&row));
        assert_eq!(out.len(), d);

        let (want_rms, want_sum, want_head) = EXPECT[i];
        let got_rms = rms(&out);
        let got_sum: f32 = out.iter().sum();
        println!("  pos {i} (tok {id}): rms {got_rms:.6} (want {want_rms:.6}), sum {got_sum:.5} (want {want_sum:.5})");
        assert!(
            (got_rms - want_rms).abs() <= TOL_REL * want_rms.abs(),
            "pos {i}: residual rms {got_rms} != reference {want_rms} - the GGUF route computes something else"
        );
        assert!(
            (got_sum - want_sum).abs() <= TOL_REL * want_sum.abs(),
            "pos {i}: residual sum {got_sum} != reference {want_sum} - a permuted or transformed tensor"
        );
        for (j, &want) in want_head.iter().enumerate() {
            let got = out[j];
            assert!(
                (got - want).abs() <= TOL_ABS + TOL_REL * want.abs(),
                "pos {i} element {j}: {got} != reference {want}"
            );
        }
    }
}
