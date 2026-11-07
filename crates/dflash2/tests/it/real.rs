// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements speculative decoding for clients serving large models on modest hardware.
// If your team needs expertise in gating a ported drafter against its reference then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The published `Ripple-65B-Preview-DFlash2` (4.04B parameters, bf16) against the host reference, on the CPU backend.
//!
//! The 80-layer target cannot run on a CPU, so its contribution is an input: synthetic hidden states of the right
//! shape for the five tapped layers, a real anchor token (the vocabulary piece ` the`), and the shared embedding and
//! head from the real partial target checkpoint. Compared: the draft's output after each of its five layers (four
//! sliding, one full NoPE), its post-norm rows, the head's logits over them, and the selector's path and
//! candidates.
//!
//! Inputs, all skipped by name when absent:
//!
//! ```text
//! testdata/dflash2/ripple-65b-preview-dflash2/   config.json, model.safetensors (hf download ricursiveAI/Ripple-65B-Preview-DFlash2)
//! testdata/ricursive/ripple-65b-partial/         layers-0-4.safetensors (embed_tokens, lm_head)
//! testdata/golden/dflash2/real/golden.safetensors
//!   python3 -I tools/goldens/dflash2_reference_forward.py real --safetensors <draft>/model.safetensors \
//!     --config <draft>/config.json --target <partial>/layers-0-4.safetensors --tokenizer <partial>/tokenizer.json \
//!     --out testdata/golden/dflash2/real --ctx-len 24 --seed 7
//! ```
//!
//! About 8 GB of draft weights are resident (bf16 as stored) and the reference holds the same decoded to f32; run
//! it with `BRAIN_DEVICE=cpu`.

use checkpoint::packed::{CodecRegistry, PackedReader, PackedSource};
use checkpoint::TensorSource;
use dflash2::{BlockJob, Dflash2, Dflash2Config, HfDraft};

use crate::common::{scaled_error, Golden};

/// Scaled error bound against the float32 reference. The draft amplifies its input by orders of magnitude (the
/// embedding it starts from has RMS 0.01, the hidden state it ends with has RMS in the hundreds), so the bound is
/// relative to each tensor's own scale.
const TOLERANCE: f32 = 2e-3;

fn paths() -> Option<(String, String, String)> {
    let draft = brain_testutil::testdata("dflash2/ripple-65b-preview-dflash2");
    let partial = brain_testutil::testdata("ricursive/ripple-65b-partial/layers-0-4.safetensors");
    let golden = brain_testutil::testdata("golden/dflash2/real/golden.safetensors");
    for (what, p) in [("the Ripple draft", format!("{draft}/model.safetensors")), ("the partial target", partial.clone()), ("the reference golden", golden.clone())] {
        if !std::path::Path::new(&p).exists() {
            brain_testutil::skip(&format!("{what} is absent ({p}); see the header of this test"));
            return None;
        }
    }
    Some((draft, partial, golden))
}

#[test]
fn the_published_ripple_draft_matches_the_reference_layer_by_layer() {
    let Some((draft, partial, golden)) = paths() else { return };
    let gold = Golden::read(&golden);
    let cfg = Dflash2Config::from_json(&std::fs::read_to_string(format!("{draft}/config.json")).unwrap()).unwrap();
    assert_eq!((cfg.n_layers, cfg.block_size, cfg.window, cfg.target_layers.clone()), (5, 16, 4096, vec![1, 20, 39, 58, 77]));
    let d = cfg.d_model as usize;

    let t = std::time::Instant::now();
    let gpu = gpu_core::Gpu::open(None, &dflash2::pipelines(&cfg));
    let model = Dflash2::load(gpu, cfg.clone(), Box::new(HfDraft::open(std::path::Path::new(&format!("{draft}/model.safetensors"))).unwrap()), 1, 64).unwrap();
    eprintln!("loaded the draft: {:.2} GiB resident in {:.1}s", model.weight_bytes() as f64 / (1u64 << 30) as f64, t.elapsed().as_secs_f64());

    // The target's shared tables.
    let target = PackedSource::identity_lenient(Box::new(PackedReader::open(std::path::Path::new(&partial)).unwrap()), CodecRegistry::reference());
    let ids = gold.ids("ids");
    let embed: Vec<f32> = ids
        .iter()
        .flat_map(|&id| {
            let mut row = Vec::new();
            assert!(target.with_tensor_range("model.embed_tokens.weight", id as usize * d, d, &mut |r| row = r.to_vec()));
            row
        })
        .collect();
    err_close("noise embedding", &embed, gold.get("noise"), 1e-6);

    let taps = gold.get("taps");
    let ctx_len = taps.len() / cfg.tap_width();
    let t = std::time::Instant::now();
    model.append_context(0, taps, 0).unwrap();
    eprintln!("fed {ctx_len} context rows in {:.1}s", t.elapsed().as_secs_f64());
    let t = std::time::Instant::now();
    let mut trace = Vec::new();
    let out = model.denoise_traced(&[BlockJob { slot: 0, pos: gold.ids("start")[0], noise: &embed }], Some(&mut trace)).unwrap();
    eprintln!("denoised one block of {} rows in {:.1}s", cfg.block_size, t.elapsed().as_secs_f64());

    let mut worst = 0.0f32;
    for (l, layer) in trace.iter().enumerate() {
        let err = scaled_error(&format!("layer {l}"), layer, gold.get(&format!("layer.{l}")));
        assert!(err <= TOLERANCE, "layer {l}: scaled error {err:.3e} exceeds {TOLERANCE:.0e}");
        worst = worst.max(err);
    }
    let mask_rows = &out[0][d..];
    let normed = model.normed(mask_rows);
    let err = scaled_error("post-norm hidden", &normed, gold.get("hidden"));
    assert!(err <= TOLERANCE, "post-norm hidden: scaled error {err:.3e}");
    worst = worst.max(err);

    // The head's logits over the draft's rows, from the real lm_head.
    let vocab = cfg.vocab as usize;
    let mut logits = Vec::with_capacity(normed.len() / d * vocab);
    target.with_tensor("lm_head.weight", &mut |head| {
        for row in normed.chunks(d) {
            logits.extend(model::hostmath::matvec_par(head, row, vocab, d));
        }
    });
    let err = scaled_error("logits", &logits, gold.get("logits"));
    assert!(err <= TOLERANCE, "logits: scaled error {err:.3e}");
    worst = worst.max(err);

    let path = model.select(mask_rows, &logits, ids[0]).unwrap();
    let want = gold.ids("path");
    eprintln!("selector path {path:?}");
    assert_eq!(path, want, "the selector's path");
    eprintln!("worst scaled error {worst:.3e} (bound {TOLERANCE:.0e})");
}

fn err_close(label: &str, got: &[f32], want: &[f32], bound: f32) {
    let err = scaled_error(label, got, want);
    assert!(err <= bound, "{label}: scaled error {err:.3e}");
}
