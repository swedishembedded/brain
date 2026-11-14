// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of quantized
// language models on a single GPU for its clients. If your team needs
// expertise in converting a checkpoint once so that training never converts
// it again then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Converting a base once (`fp8_source::write_converted_base`) and converting it on every load
//! (`PackedBase::converting_to`) give the same device weights, bit for bit; the converted directory is a base
//! the trainer opens like any other, holds fewer bytes than the FP8 it came from, and its matrices reach the
//! device as NF4 with no host requantization.

use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::{training_kernel_list, Ops, TierPolicy, Weight};
use qwen35::config::Qwen35Config;
use qwen35::fp8_source::{write_converted_base, PackedBase};

use crate::common;
use common::hf_fixture::{write_checkpoint, BLOCK};

/// The codes (eight per word) and block scales of an `[n, k]` NF4 weight, read back from the device.
fn nf4_words(ops: &Ops, weight: &Weight, n: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
    let Weight::NF4 { w, s, .. } = weight else { panic!("expected an NF4 weight, got {:?}", weight.dtype()) };
    let g = ops.gpu();
    (g.read(w, n * k / 8).into_iter().map(f32::to_bits).collect(), g.read(s, n * k / 32))
}

#[test]
fn a_base_converted_once_equals_the_base_converted_on_every_load() {
    let cfg = Qwen35Config { vocab: 64, n_layers: 4, ..Qwen35Config::tiny_i8() };
    let dir = write_checkpoint(&cfg, "convert_src", None);
    let out = std::env::temp_dir().join(format!("convert_base_out_{}", std::process::id()));
    let report = write_converted_base(&dir, &cfg, BLOCK, &out, &TierPolicy::uniform(Dtype::NF4), &mut || true).expect("convert once");
    assert!(report.converted > 0 && report.converted < report.tensors, "the matrices convert, the norms and tables stay as stored: {report:?}");

    let ops = Ops::new(Gpu::new_cpu(training_kernel_list())).expect("Ops::new");
    let on_the_fly = PackedBase::open(&dir, &cfg, BLOCK).unwrap().converting_to(Dtype::NF4);
    let from_file = PackedBase::open(&out, &cfg, BLOCK).expect("the converted directory opens as a base").converting_to(Dtype::NF4);
    let mut matrices = 0;
    for (l, ty) in cfg.layer_types().iter().enumerate() {
        for (suffix, n, k) in cfg.layer_leaves(*ty) {
            let name = format!("blocks.{l}.{suffix}");
            let a = on_the_fly.weight(&ops, &name, n as usize, k as usize).unwrap_or_else(|e| panic!("{name}: {e}"));
            let b = from_file.weight(&ops, &name, n as usize, k as usize).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(nf4_words(&ops, &a, n as usize, k as usize), nf4_words(&ops, &b, n as usize, k as usize), "{name}: converted once vs converted on load");
            matrices += 1;
        }
        for small in ["ln1.weight", "ln2.weight"] {
            let name = format!("blocks.{l}.{small}");
            assert_eq!(on_the_fly.small_f32(&name).unwrap(), from_file.small_f32(&name).unwrap(), "{name}");
        }
    }
    assert!(matrices > 0);

    let bytes = |d: &std::path::Path| -> u64 { std::fs::read_dir(d).unwrap().filter_map(|e| e.ok()).filter(|e| e.path().extension().is_some_and(|x| x == "safetensors")).map(|e| e.metadata().unwrap().len()).sum() };
    assert!(bytes(&out) < bytes(&dir), "the converted base is smaller on disk: {} vs {} bytes", bytes(&out), bytes(&dir));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&out).ok();
}

/// A token that fires mid-conversion stops it at the next tensor and leaves nothing in the destination: no weights,
/// no copied config, no directory it made.
#[test]
fn a_conversion_stops_at_the_next_tensor_and_leaves_no_directory() {
    use capability::{Invocation, Outcome, Progress};
    use capability_mock::cancellation::{certify, Bound};
    let cfg = Qwen35Config { vocab: 64, n_layers: 4, ..Qwen35Config::tiny_i8() };
    let dir = write_checkpoint(&cfg, "convert_cancel_src", None);
    let out = std::env::temp_dir().join(format!("convert_base_cancel_out_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    let run = |inv: &Invocation, progress: &mut dyn FnMut(Progress)| {
        inv.cancel.check()?;
        let converted = write_converted_base(&dir, &cfg, BLOCK, &out, &TierPolicy::uniform(Dtype::NF4), &mut inv.keep_going(progress, 0, "tensor"));
        // A finished reference run leaves its base behind; the next run starts clean.
        if converted.is_ok() {
            std::fs::remove_dir_all(&out).ok();
        }
        converted?;
        Ok(Outcome::new())
    };
    certify("qwen35 convert", &Invocation::new, &run, Bound::ONE_STEP);
    assert!(!out.exists(), "a stopped conversion leaves no directory it made");
    std::fs::remove_dir_all(&dir).ok();
}
