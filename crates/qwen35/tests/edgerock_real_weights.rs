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

/// The precision matrix: the released weights quantized on load to each weight tier the generic machinery offers
/// (the checkpoint is bf16; the admission scorer, the norms and the read gate stay exact fp32, the memory's
/// read-side projections and the embedding follow the tier at 16 bits or the tier itself), against the same
/// reference forward. Prints, per tier, the worst layer cosine, the tail-logit cosine and relative L2, whether the
/// greedy answer after the question is the reference's, and the worst per-(block, group) bank set overlap. A
/// measurement, not a gate: what a tier costs in fidelity is the finding. A tier the kernels cannot run is
/// reported with the panic message, not skipped.
#[test]
#[ignore = "slow: loads the released checkpoint once per tier"]
fn precision_matrix_against_the_reference_forward() {
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
    let (d, vocab) = (cfg.d_model as usize, cfg.vocab as usize);
    let (capacity, _groups) = (cfg.anchor_memory.expect("EdgeRock has an anchor memory").capacity as usize, cfg.n_kv_heads as usize);
    println!("{:<8} {:>14} {:>14} {:>10} {:>8} {:>14}", "tier", "worst layer cos", "logits cos", "logits rel_l2", "greedy", "bank overlap");
    for name in ["f32", "bf16", "f16", "i8", "q4", "nf4", "f8e4m3", "f4e2m1"] {
        let tier = TierPolicy::parse(name).expect("a tier name");
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let src = qwen35::fp8_source::open_dir(dir, &cfg, 128).expect("every tensor is covered");
            let m = Qwen35::new_on_dt(Gpu::new(pipelines()), cfg.clone(), 1, t, &src, &tier);
            m.enable_memory_taps();
            let logits = m.logits_all(&tokens);
            let taps: HashMap<String, Vec<f32>> = m.take_memory_taps().into_iter().collect();
            let pick = |all: &[f32]| rows.iter().flat_map(|&r| all[r * d..(r + 1) * d].iter().copied()).collect::<Vec<f32>>();
            let mut table = Table::new(0.0, f64::MAX);
            for l in 0..cfg.n_layers as usize {
                table.check(&format!("layer{l}.out"), &pick(&m.debug_res(l + 1)), &golden[&format!("layer{l}.out")]);
            }
            let tail = &logits[(t as usize - 8) * vocab..];
            table.check("logits_tail", tail, &golden["logits_tail"]);
            let worst_layer = table.rows.iter().filter(|r| r.0.starts_with("layer")).map(|r| r.1).fold(1.0, f64::min);
            let (cos, rel) = table.rows.iter().find(|r| r.0 == "logits_tail").map(|r| (r.1, r.3)).expect("compared");
            let last = &tail[7 * vocab..];
            let greedy = (0..vocab).max_by(|&a, &b| last[a].total_cmp(&last[b])).unwrap();
            let want = golden["logits_tail"][7 * vocab..].iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
            let mut overlap = 1.0f64;
            for l in (0..cfg.n_layers as usize).filter(|&l| cfg.layer_has_memory(l)) {
                let key = format!("layer{l}.mem.refs");
                let (mut sum, mut cells) = (0.0, 0usize);
                for (a, b) in taps[&key].chunks(capacity).zip(golden[&key].chunks(capacity)) {
                    let set = |row: &[f32]| row.iter().filter(|&&r| r != 0.0).map(|&r| r as u32).collect::<std::collections::HashSet<u32>>();
                    let (sa, sb) = (set(a), set(b));
                    if sa.len().max(sb.len()) > 0 {
                        sum += sa.intersection(&sb).count() as f64 / sa.len().max(sb.len()) as f64;
                        cells += 1;
                    }
                }
                overlap = overlap.min(if cells == 0 { 1.0 } else { sum / cells as f64 });
            }
            (worst_layer, cos, rel, greedy == want, overlap)
        }));
        match ran {
            Ok((layer, cos, rel, greedy, overlap)) => println!("{name:<8} {layer:>14.8} {cos:>14.8} {rel:>10.3e} {:>8} {:>13.2}%", if greedy { "same" } else { "DIFFERS" }, overlap * 100.0),
            Err(p) => {
                let why = p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
                println!("{name:<8} cannot run: {}", why.lines().next().unwrap_or(""));
            }
        }
    }
}

/// The serving engine's state formats against the reference: the golden prompt (1400 tokens, past the window and
/// five memory blocks) prefilled in 256-row chunks through the fixed-state slot (the fused windowed attention, the
/// pooled recurrence, the bank) with the ring held in each format, the last position's logits against the
/// reference's. The greedy answer must be the reference's and the logits cosine at least 0.9999.
#[test]
#[ignore = "slow: the released checkpoint, once per ring format"]
fn the_serving_slot_matches_the_reference_with_a_bf16_ring() {
    use model::paged::BlockTable;
    use model::serve::PagedDecoder;
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
    let vocab = golden["logits_tail"].len() / 8;
    let want = &golden["logits_tail"][7 * vocab..];
    let want_token = want.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
    let dir = Path::new(&dir);
    let cfg = qwen35::fp8_source::hf_config(dir).expect("config.json");
    for ring in [model::kv_tier::KvTier::F32, model::kv_tier::KvTier::BF16] {
        let src = qwen35::fp8_source::open_dir(dir, &cfg, 128).expect("every tensor is covered");
        let mut engine = qwen35::serve_ring::RingEngine::from_source_tier(cfg.clone(), &src, 4096, 1, 0, &TierPolicy::uniform(Dtype::BF16), ring);
        let mut table = BlockTable::new();
        let hidden = engine.prefill(&mut table, &tokens);
        let got = engine.logits(&hidden);
        let cos = {
            let dot: f64 = got.iter().zip(want).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
            let (na, nb): (f64, f64) = (got.iter().map(|a| f64::from(*a).powi(2)).sum::<f64>().sqrt(), want.iter().map(|b| f64::from(*b).powi(2)).sum::<f64>().sqrt());
            dot / (na * nb)
        };
        let token = engine.admit_greedy(&hidden);
        println!("ring {ring}: logits cosine {cos:.7}, greedy {token} (reference {want_token})");
        assert_eq!(token, want_token, "ring {ring}: the greedy answer differs from the reference's");
        assert!(cos >= 0.9999, "ring {ring}: logits cosine {cos}");
    }
}
