// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements device-independent cost accounting for LLM serving for its clients. If your team
// needs expertise in pricing a served model's work the same way on every accelerator then you can procure our
// services by sending an email to info@swedishembedded.com.

// hf-fixture: qwen35
//! A served qwen35 step costs the same FLOPs whichever kernels run it. The resident's prefill and batched decode on
//! a CUDA device reach the native kernels (the paged decode, the fused GDN pools, the fused norms, the provider
//! products); each states the portable work it replaces, so the step's report covers every dispatch and its total
//! equals the same step's with every native kernel and the CUDA provider withheld.
//!
//! Run with `-- --device gpu<i> --backend cuda`.

use gpu_core::cost::{CostReport, Recording};
use model::kv_tier::KvTier;
use model::paged::BlockTable;
use model::serve::PagedDecoder;
use qwen35::config::Qwen35Config;
use qwen35::int8_gguf_resident::{budgeted_gpus, held_tiers_at, Qwen35GgufEngine, Qwen35GgufResident};

mod common;
use common::hf_fixture::{fp8_resident_cfg, write_checkpoint_conditioned, write_tokenizer};

/// The fixture's dimensions at the head widths the native kernels serve: 128-wide attention heads (the paged decode
/// and prefill kernels) and 128-wide GDN key and value heads (the chunked and decode GDN kernels).
fn served_cfg() -> Qwen35Config {
    Qwen35Config {
        d_model: 256,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 128,
        linear_num_key_heads: 1,
        linear_num_value_heads: 2,
        linear_key_head_dim: 128,
        linear_value_head_dim: 128,
        ..fp8_resident_cfg()
    }
}

fn engine(tag: &str) -> Qwen35GgufEngine {
    let dir = write_checkpoint_conditioned(&served_cfg(), tag, 128);
    write_tokenizer(&dir);
    let path = dir.to_string_lossy().into_owned();
    let tier = held_tiers_at(&path).expect("the directory is a servable checkpoint");
    let devices = budgeted_gpus(&gpu_core::devices::ambient_compute_set().gpus, 2 << 30);
    let device = devices.first().expect("the test needs a GPU").0;
    let resident = Qwen35GgufResident::new(path, devices, 192, tier).with_max_batch(4).with_kv_block_rows(16).with_kv_tier(KvTier::F32);
    let (engine, _, _) = resident.activate_engine(&[device]).expect("the fixture loads");
    std::fs::remove_dir_all(dir).ok();
    engine
}

fn prompt_of(seed: u32, n: u32) -> Vec<u32> {
    (0..n).map(|i| 3 + (seed * 5 + i * 7 + i * i) % 50).collect()
}

/// The cost of one prefill per sequence and one batched decode step over all of them.
fn serve_steps(tag: &str) -> (CostReport, CostReport) {
    let mut e = engine(tag);
    e.instance().set_decode_plans(false);
    let lens = [30u32, 41];
    let mut tables: Vec<BlockTable> = lens.iter().map(|_| BlockTable::new()).collect();
    let rec = Recording::live();
    let mut next = Vec::new();
    for (i, t) in tables.iter_mut().enumerate() {
        let h = e.prefill(t, &prompt_of(i as u32 + 1, lens[i]));
        next.push(e.admit_greedy(&h));
    }
    let prefill = rec.take();
    let rec = Recording::live();
    let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
    e.forward_batched_greedy(&mut refs, &next);
    let decode = rec.take();
    (prefill, decode)
}

/// The native kernels a report accounted, by the names they run under.
fn natives(report: &CostReport) -> Vec<&str> {
    report.by_kernel.keys().map(String::as_str).filter(|k| k.starts_with("native:")).collect()
}

fn a_served_step_is_fully_priced_and_costs_what_the_portable_step_costs() {
    let native = serve_steps("cost-native");
    // The same steps with every native kernel and the CUDA provider withheld: the portable dispatches they replace. Both
    // switches are process-wide and read when a handle is built, so they are set between the two engines.
    gpu_core::set_native_kernels(false);
    gpu_core::provider::set_disabled_providers(&["cuda"]);
    let portable = serve_steps("cost-portable");
    gpu_core::set_native_kernels(true);
    gpu_core::provider::set_disabled_providers(&[]);

    for ((what, got), want) in [("prefill", &native.0), ("decode", &native.1)].into_iter().zip([&portable.0, &portable.1]) {
        eprintln!("serve_step_cost: {what}: native kernels {:?}, {:?} against the portable {:?}", natives(got), got.total, want.total);
        assert!(!natives(got).is_empty(), "{what}: the served step reached no native kernel, so nothing here is exercised");
        assert!(natives(want).is_empty(), "{what}: the portable step ran native kernels {:?}", natives(want));
        assert_eq!(want.coverage(), 1.0, "{what}: portable dispatches without a cost formula: {:?}", want.uncovered);
        assert_eq!(got.coverage(), 1.0, "{what}: steps that state no portable work: {:?}", got.uncovered);
        assert_eq!(got.total.flops, want.total.flops, "{what}: the native step's FLOPs differ from the portable step's");
    }
}

gpu_core::card_tests!(a_served_step_is_fully_priced_and_costs_what_the_portable_step_costs);
