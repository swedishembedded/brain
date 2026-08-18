// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **Real-weight stage parity for the DeepSeek-OCR MoE decoder**, against llama.cpp.
//!
//! `tests/gradcheck.rs` proves the backward agrees with this crate's own
//! forward at toy dims under four router policies. This test is the other axis:
//! the real `DeepSeek-OCR-Q8_0.gguf` language-model weights, the real 12-layer
//! /1280-wide/64-expert shape, run on a real token sequence and compared layer
//! by layer against a dump taken from **llama.cpp**, the upstream consumer this
//! GGUF targets. Both sides dequantize the same Q8_0 blocks, so the expected
//! agreement is fp32-tight - see `crates/sam1/tests/parity.rs` for why the
//! floor is where it is.
//!
//! What it does NOT cover is the third axis: this is ONE forward over a FIXED
//! sequence, so nothing here can see a decode loop that advances RoPE's position
//! wrongly or lets the causal mask slip as the sequence grows. `tests/generate.rs`
//! is that gate, and it anchors its multi-step reference on the single-step
//! argmax [`real_lm_decoder_matches_llamacpp`] proves below.
//!
//! ## Input
//!
//! Token ids `[0, 19923]` - llama.cpp's own tokenizer output for the prompt
//! `"Hello"` under this checkpoint (0 is BOS), recorded in the fixture as data.
//! They are replayed verbatim, so tokenizer correctness is deliberately out of
//! scope here and cannot contribute a mismatch.
//!
//! ## The shrinking sequence axis
//!
//! ggml only needs the FINAL position's activations once the graph reaches the
//! output head, so the reference's own tensors drop from 2 rows to 1 partway
//! down the stack (`l_out-11`, layer 11's router, `result_norm`,
//! `result_output`). brain computes every row for every layer, so each
//! comparison takes the **last** `golden_rows` rows of brain's tap - see
//! [`tail`].
//!
//! ## What this pins that no fixture could
//!
//! The router semantics. This architecture's GGUF carries **no**
//! `scoring_func`/`topk_method`/`norm_topk_prob`/`routed_scaling_factor` key at
//! all, so what runs is llama.cpp's compiled-in default for the arch. The
//! roadmap recorded that as plain softmax + top-6 + **no renormalization** +
//! scale 1.0, argued from the merged upstream PR. Here it is measured on the
//! shipped weights: `moe_probs` sums to 1.0 over all 64 experts while the gate
//! values the layer actually multiplies by sum to between 0.22 and 0.82 over
//! the 6 selected ones, and equal the corresponding softmax entries exactly.
//! [`real_weight_router_gates_are_raw_not_renormalized`] asserts both halves on
//! the reference AND on brain's own gate.
//!
//! ## Cost, and the cached fp32 expansion
//!
//! ~2.9 B parameters. Materializing the import into a host `HashMap` and then
//! uploading it would need the fp32 expansion twice over (~23 GB) - more than
//! the machines this runs on have. So the test uses the streaming path this
//! crate documents for exactly this case: `import::import_file` converts the
//! GGUF to a brain-native `.safetensors` **once** (peak host ≈ one dequantized
//! tensor), and `DeepseekV2::new_on` then streams that file straight to the
//! device. The expansion is ~12 GB and is cached beside the checkpoint in the
//! model store; set `$BRAIN_DEEPSEEK_OCR_LM_ST` to put it elsewhere. Nothing is
//! written under `testdata/`, which stays fixtures-only.
//!
//! ## Backend
//!
//! Pinned to `backend-cpu`, for two reasons and both recorded rather than
//! assumed. (1) `crates/sam1/tests/parity.rs` documents a device-level buffer
//! corruption on the wgpu path that appears once a graph holds more than a
//! couple of blocks' worth of large buffers at production shape - this decoder
//! is ~12 GB of fp32 parameters, far past that. (2) The card here is an
//! integrated GPU whose reported `max_buffer_size` is 2047 MiB against a
//! ~12 GB working set. A GPU number would not be trustworthy, and a parity test
//! that cannot be trusted is worse than one that is honest about where it ran.
//!
//! Fixture: `<testdata>/deepseek-ocr/real/decoder.safetensors`, produced by
//! `tools/goldens/deepseek_ocr_convert_llamacpp_dump.py` (its header documents
//! the capture command). The test SKIPS ITSELF when the fixture or the real
//! checkpoint is absent - it never panics for a missing input.

use brain_testutil::parity::{compare, load, Report};
use brain_testutil::testdata_path as testdata;
use checkpoint::safetensors::StTensor;
use deepseek2::DeepseekV2;

/// The model-store lookup, the one-off fp32 expansion, the CPU-backend pin and
/// the inference build - shared with `tests/generate.rs`.
#[path = "common/real_lm.rs"]
mod real_lm;

/// Same reasoning as `sam1`'s: identical Q8_0 blocks on both sides.
const FLOOR: f64 = 0.999;

/// The last `want.len() / width` rows of `got` - the alignment rule for a
/// reference tensor that carries only the final position(s). See the module
/// header.
fn tail<'a>(got: &'a [f32], want: &StTensor, width: usize) -> &'a [f32] {
    assert_eq!(want.data.len() % width, 0, "golden tensor is not a whole number of rows");
    assert!(want.data.len() <= got.len(), "golden has more rows than brain computed");
    &got[got.len() - want.data.len()..]
}

fn build() -> Option<(DeepseekV2, std::collections::HashMap<String, StTensor>)> {
    let fixture = testdata("deepseek-ocr/real/decoder.safetensors");
    if !fixture.exists() {
        brain_testutil::skip(&format!("deepseekv2 parity: fixture missing at {}", fixture.display()));
        return None;
    }
    let golden = load(&fixture);
    let tokens: Vec<u32> = golden["tokens"].data.iter().map(|v| *v as u32).collect();
    let m = real_lm::open(tokens.len() as u32)?;
    m.logits_all(&tokens);
    Some((m, golden))
}

#[ignore = "real 2.9 B-parameter checkpoint: ~12 GB resident and a one-off ~12 GB fp32 expansion on disk. Two of these in parallel would exhaust any machine that can run one, so it stays out of the fast lane. `make test/slow`, or `cargo test --release -p brain-deepseekv2 --test parity -- --nocapture --test-threads=1`."]
#[test]
fn real_lm_decoder_matches_llamacpp() {
    let Some((m, golden)) = build() else { return };
    let d = m.cfg.d_model() as usize;
    let vocab = m.cfg.vocab() as usize;
    println!("== deepseekv2 real-weight parity");

    let mut r = Report::wide(FLOOR, 18);
    r.check("embd", tail(&m.read_res(0), &golden["embd"], d), &golden["embd"].data);
    for l in 0..m.cfg.n_layers() as usize {
        let a = format!("attn_out_L{l:02}");
        let o = format!("l_out_L{l:02}");
        r.check(&a, tail(&m.read_attn_out(l), &golden[&a], d), &golden[&a].data);
        r.check(&o, tail(&m.read_res(l + 1), &golden[&o], d), &golden[&o].data);
    }
    r.check("result_norm", tail(&m.read_final_norm(), &golden["result_norm"], d), &golden["result_norm"].data);

    // The model's actual output distribution. If anything above is subtly
    // wrong this is where it shows up amplified, so it is reported separately
    // as well as gated: an argmax that disagrees is a different model, whatever
    // the cosine says.
    let logits = m.read_logits();
    let ours = tail(&logits, &golden["result_output"], vocab);
    r.check("result_output", ours, &golden["result_output"].data);
    let am = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).expect("nonempty").0;
    println!("  greedy next token: brain {} vs reference {}", am(ours), am(&golden["result_output"].data));
    assert_eq!(am(ours), am(&golden["result_output"].data), "greedy next token disagrees");
    r.finish("deepseekv2 real-weight");
}

/// The MoE router internals of every MoE layer, and the raw-gate fact.
///
/// `ffn_moe_gate-N` (the per-selected-expert gathered activations) is captured
/// in the fixture but NOT compared: brain's `MoeActs` keeps those per expert
/// rather than gathered per selected slot, and reshaping one into the other in
/// a test would be a second, untested implementation of the gather. The three
/// tensors that actually decide routing - the full softmax, the selection, and
/// the applied gate - are all compared.
#[ignore = "real 2.9 B-parameter checkpoint: ~12 GB resident and a one-off ~12 GB fp32 expansion on disk. Two of these in parallel would exhaust any machine that can run one, so it stays out of the fast lane. `make test/slow`, or `cargo test --release -p brain-deepseekv2 --test parity -- --nocapture --test-threads=1`."]
#[test]
fn real_weight_router_gates_are_raw_not_renormalized() {
    let Some((m, golden)) = build() else { return };
    let e = m.cfg.n_experts() as usize;
    let k = m.cfg.top_k() as usize;
    println!("== deepseekv2 real-weight router");

    let mut r = Report::wide(FLOOR, 18);
    let mut checked = 0usize;
    for l in 0..m.cfg.n_layers() as usize {
        let (Some(logits), Some(gate)) = (m.read_router_logits(l), m.read_router_gate(l)) else {
            assert_eq!(l, 0, "only the leading dense layer may lack a router");
            continue;
        };
        let pname = format!("moe_probs_L{l:02}");
        let want_probs = &golden[&pname];
        let want_topk = &golden[&format!("moe_topk_L{l:02}")];
        let want_w = &golden[&format!("moe_weights_L{l:02}")];
        let g_rows = want_probs.data.len() / e;

        // Our router logits, softmaxed on the host, against the reference's own
        // post-softmax tensor. (brain keeps the pre-softmax logits as the tap;
        // the softmax itself lives inside `router_gate.wgsl`.)
        let ours_probs: Vec<f32> = logits
            .chunks_exact(e)
            .flat_map(|row| {
                let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let ex: Vec<f32> = row.iter().map(|v| (v - mx).exp()).collect();
                let s: f32 = ex.iter().sum();
                ex.into_iter().map(move |v| v / s)
            })
            .collect();
        r.check(&pname, tail(&ours_probs, want_probs, e), &want_probs.data);

        // Selection and applied gate, per row, at the reference's own indices.
        let ours_tail = &gate[gate.len() - g_rows * e..];
        for row in 0..g_rows {
            let ours_row = &ours_tail[row * e..(row + 1) * e];
            let want_idx: Vec<usize> = want_topk.data[row * k..(row + 1) * k].iter().map(|v| *v as usize).collect();
            let mut ours_idx: Vec<usize> = (0..e).filter(|&i| ours_row[i] != 0.0).collect();
            ours_idx.sort_by(|&a, &b| ours_row[b].total_cmp(&ours_row[a]));
            assert_eq!(ours_idx.len(), k, "layer {l} row {row}: {} experts selected, want {k}", ours_idx.len());
            let (mut a, mut b) = (ours_idx.clone(), want_idx.clone());
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b, "layer {l} row {row}: expert selection disagrees");

            let ours_w: Vec<f32> = want_idx.iter().map(|&i| ours_row[i]).collect();
            let want_row = &want_w.data[row * k..(row + 1) * k];
            let (cos, max_abs) = compare(&ours_w, want_row);
            let (sum_ours, sum_ref) = (ours_w.iter().sum::<f32>(), want_row.iter().sum::<f32>());
            println!(
                "  moe_weights_L{l:02}[{row}] cos {cos:.10}  max_abs {max_abs:.3e}  sum brain {sum_ours:.6} ref {sum_ref:.6}"
            );
            r.rows.push((format!("moe_weights_L{l:02}[{row}]"), cos, max_abs));

            // THE fact: the gate the layer applies is the RAW softmax
            // probability of the selected expert. Renormalized gates would sum
            // to exactly 1; a `routed_scaling_factor != 1` would break the
            // equality with `moe_probs`.
            assert!(sum_ref < 0.999, "reference gate sums to {sum_ref} -- it IS renormalized after all");
            assert!(sum_ours < 0.999, "brain's gate sums to {sum_ours} -- norm_topk_prob leaked in");
            for (j, &i) in want_idx.iter().enumerate() {
                let p = want_probs.data[row * e + i];
                assert!(
                    (p - want_row[j]).abs() <= 0.0,
                    "layer {l} row {row}: applied gate {} != raw softmax prob {p} for expert {i}",
                    want_row[j]
                );
            }
            checked += 1;
        }
    }
    assert!(checked >= 21, "only {checked} router rows checked -- the fixture shrank");
    r.finish("deepseekv2 real-weight router");
}
