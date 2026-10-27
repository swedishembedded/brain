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
use gpu_core::provider::library::GemmPath;
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
    // The tier the model's matrix products run at decides the tolerance, as in the golden tests.
    let exact = matches!(m.gemm_path(), GemmPath::Fp32Kernels { .. });
    let mut table = Table::new(0.999, 5e-2);
    for l in 0..cfg.n_layers as usize {
        table.check(&format!("layer{l}.out"), &pick(&m.debug_res(l + 1)), &golden[&format!("layer{l}.out")]);
    }
    let tail = &logits[(t as usize - 8) * vocab..];
    table.check("logits_tail", tail, &golden["logits_tail"]);
    table.print();

    // The bank a block commits is a top-`capacity` selection by score. A near tie swaps one record, and every slot
    // after it in rank order then differs, so slot-wise equality understates agreement; what the memory reads is the
    // SET of records, so the bank is compared per (block, group) as a set. Exact slot equality is held only on the
    // fp32 reference tier; the tensor-core tier is held to the overlap its arithmetic supports.
    let (capacity, groups) = (cfg.anchor_memory.expect("EdgeRock has an anchor memory").capacity as usize, cfg.n_kv_heads as usize);
    let mut banks = Vec::new();
    for l in (0..cfg.n_layers as usize).filter(|&l| cfg.layer_has_memory(l)) {
        let name = format!("layer{l}.mem.refs");
        let (ours, theirs) = (&taps[&name], &golden[&name]);
        assert_eq!(ours.len(), theirs.len(), "{name}: bank shape");
        let equal = ours.iter().zip(theirs).filter(|(a, b)| a == b).count() as f64 / theirs.len() as f64;
        let (mut overlap, mut cells) = (0.0, 0usize);
        for (a, b) in ours.chunks(capacity).zip(theirs.chunks(capacity)) {
            let set = |row: &[f32]| row.iter().filter(|&&r| r != 0.0).map(|&r| r as u32).collect::<std::collections::HashSet<u32>>();
            let (sa, sb) = (set(a), set(b));
            let denom = sa.len().max(sb.len());
            if denom > 0 {
                overlap += sa.intersection(&sb).count() as f64 / denom as f64;
                cells += 1;
            }
        }
        assert_eq!(ours.len() % (groups * capacity), 0, "{name}: bank layout");
        let overlap = if cells == 0 { 1.0 } else { overlap / cells as f64 };
        println!("{name}: {:.2}% of slots equal the reference, {:.2}% set overlap per (block, group)", equal * 100.0, overlap * 100.0);
        banks.push((name, equal, overlap));
    }

    let last = &tail[7 * vocab..];
    let greedy = (0..vocab).max_by(|&a, &b| last[a].total_cmp(&last[b])).unwrap();
    let want = golden["logits_tail"][7 * vocab..].iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    assert_eq!(greedy, want, "the greedy answer after the question differs from the reference's");
    assert!(table.failures.is_empty(), "parity failures: {:#?}", table.failures);
    for (name, equal, overlap) in banks {
        if exact {
            assert!(equal >= 0.95, "{name}: only {:.1}% of the bank equals the reference", equal * 100.0);
        } else {
            assert!(overlap >= 0.99, "{name}: only {:.1}% set overlap with the reference bank", overlap * 100.0);
        }
    }
}
