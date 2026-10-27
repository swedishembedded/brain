// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context inference at a
// fixed memory budget for its clients. If your team needs expertise in
// bounded-memory model architectures then you can procure our services by
// sending an email to info@swedishembedded.com.

//! EdgeRock at the RELEASED weights against the independent reference
//! forward (`tools/goldens/edgerock_dump_real_reference.py`, fp32 on the CPU):
//! a 1400-token prompt that slides the 1024-token window and commits five
//! memory blocks, then asks about a fact stated at its start.
//!
//! The checkpoint is read from the model store (`brain pull
//! rockmatrix-ai/EdgeRock`) through the lazy source at the bf16 weight tier -
//! the released precision, so the weights themselves are exactly the
//! reference's; only the arithmetic differs (bf16 operands with fp32
//! accumulation here, fp32 there). Gates: per-layer cosine at the sampled
//! rows, the tail logits, the greedy answer, and the memory banks. A bank is
//! held to nearly every slot, not all: admission scores of near-tied records
//! move with the arithmetic and can swap rank.

use std::collections::HashMap;
use std::path::Path;

use brain_testutil::parity::Table;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::TierPolicy;
use qwen35::model::{pipelines, Qwen35};

#[test]
#[ignore = "slow: the released 9 GB checkpoint and a 1400-token forward"]
fn released_weights_match_the_reference_forward() {
    let Some(dir) = brain_testutil::model_dir("rockmatrix-ai/EdgeRock").filter(|d| Path::new(d).join("config.json").exists()) else {
        brain_testutil::skip("rockmatrix-ai/EdgeRock is not in the model store - brain pull rockmatrix-ai/EdgeRock");
        return;
    };
    let golden_path = brain_testutil::testdata("golden/edgerock/real/edgerock_real.safetensors");
    if !Path::new(&golden_path).exists() {
        brain_testutil::skip(&format!("fixture {golden_path} absent - run tools/goldens/edgerock_dump_real_reference.py"));
        return;
    }
    let golden: HashMap<String, Vec<f32>> = checkpoint::safetensors::read(&golden_path).expect("read golden").into_iter().map(|t| (t.name, t.data)).collect();
    let tokens: Vec<u32> = golden["tokens"].iter().map(|&v| v.round() as u32).collect();
    let rows: Vec<usize> = golden["rows"].iter().map(|&v| v.round() as usize).collect();
    let t = tokens.len() as u32;

    let dir = Path::new(&dir);
    let mut cfg = qwen35::fp8_source::hf_config(dir).expect("config.json");
    cfg.block_size = t;
    let src = qwen35::fp8_source::open_dir(dir, &cfg, 128).expect("every tensor is covered");
    let m = Qwen35::new_on_dt(Gpu::new(pipelines()), cfg.clone(), 1, t, &src, &TierPolicy::uniform(Dtype::BF16));

    m.enable_memory_taps();
    let logits = m.logits_all(&tokens);
    let taps: HashMap<String, Vec<f32>> = m.take_memory_taps().into_iter().collect();

    let d = cfg.d_model as usize;
    let vocab = cfg.vocab as usize;
    let pick = |all: &[f32]| rows.iter().flat_map(|&r| all[r * d..(r + 1) * d].iter().copied()).collect::<Vec<f32>>();
    let mut table = Table::new(0.999, 5e-2);
    for l in 0..cfg.n_layers as usize {
        table.check(&format!("layer{l}.out"), &pick(&m.debug_res(l + 1)), &golden[&format!("layer{l}.out")]);
    }
    let tail = &logits[(t as usize - 8) * vocab..];
    table.check("logits_tail", tail, &golden["logits_tail"]);
    table.print();

    let mut banks = Vec::new();
    for l in (0..cfg.n_layers as usize).filter(|&l| cfg.layer_has_memory(l)) {
        let name = format!("layer{l}.mem.refs");
        let (ours, theirs) = (&taps[&name], &golden[&name]);
        assert_eq!(ours.len(), theirs.len(), "{name}: bank shape");
        let equal = ours.iter().zip(theirs).filter(|(a, b)| a == b).count() as f64 / theirs.len() as f64;
        println!("{name}: {:.2}% of slots equal the reference", equal * 100.0);
        banks.push((name, equal));
    }

    let last = &tail[7 * vocab..];
    let greedy = (0..vocab).max_by(|&a, &b| last[a].total_cmp(&last[b])).unwrap();
    let want = golden["logits_tail"][7 * vocab..].iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    assert_eq!(greedy, want, "the greedy answer after the question differs from the reference's");
    assert!(table.failures.is_empty(), "parity failures: {:#?}", table.failures);
    for (name, equal) in banks {
        assert!(equal >= 0.95, "{name}: only {:.1}% of the bank equals the reference", equal * 100.0);
    }
}
