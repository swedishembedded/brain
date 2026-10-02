// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The compact KV tiers on the REAL Qwen3.8-27B Q8_0 checkpoint: scored along
//! the same token stream as the `f32` cache, do `bf16` and per-row `int8` give
//! the logits - and the same top-1 - a trained model gives with full precision?
//!
//! Swedish Embedded AB implements long-context inference serving for clients
//! whose GPU memory decides how many users one card carries. If your team needs
//! expertise in fitting more concurrent sequences into the same device memory
//! without giving up accuracy, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! Self-skips unless `BRAIN_QWEN35_GGUF` names the file. Each tier is a cold
//! load of ~27 GiB of weights, one after the other (the previous instance is
//! dropped first, so the card holds one model at a time), then a ~1.8k-token
//! prompt through the chunked prefill tape and 24 forced decode steps through
//! the decode tape:
//!
//! ```text
//! BRAIN_QWEN35_GGUF=$HOME/.local/share/brain/models/unsloth/Qwen3.8-27B-GGUF/Q8_0.gguf //!   cargo test -p brain-qwen35 --test gguf_kv_tier_real -- --nocapture --test-threads=1
//! ```
//!
//! # What a real model can and cannot show here
//!
//! The reference is the `f32` cache's own free-running greedy continuation, so
//! "top-1 agrees" means the compact tier would have chosen the token the full
//! precision cache chose at every one of those positions.
//!
//! The logits themselves are a different matter. This model runs W8A8 (INT8
//! weights AND dynamically quantised activations), and that pipeline is chaotic
//! at the 10% level: `gguf_i8_vs_fp32_real.rs` records the divergence compounding
//! along the sequence through the recurrent state, and the same `f32` cache
//! scored by its two dispatch shapes (the decode tape and the one-row chunk tape,
//! [`Qwen35GgufInstance::tape_comparison_trace`]) differs by a relative L2 of
//! the same order - because a perturbation far below any KV rounding re-draws
//! which activation values round up. So the bound on a compact tier's logits is
//! stated against THAT floor, measured in the same run, and not as an absolute:
//! a tier is acceptable when it moves the logits no more than the numerics
//! already move them between two correct renderings of the same cache. How
//! faithfully each tier stores K and V is gated independently, against the
//! formats' own rounding bounds, in `crates/model/tests/kv_tier.rs` and at
//! random weights in `kv_tier.rs` beside this file.

use model::kv_tier::KvTier;
use model::ops::TierPolicy;
use gpu_core::select::Dtype;
use qwen35::int8_gguf_resident::{Qwen35GgufInstance, Qwen35GgufResident, GGUF_ENV};
use residency::multi::MultiDeviceResidentModel;
use residency::{Device, ResidentModel};

/// Per-sequence `prompt + forced` ceiling the instances are built for.
const CAP: u32 = 2048;
/// Forced decode positions scored after the prompt.
const STEPS: u32 = 24;
/// Bytes kept free per card, `brain serve`'s default reserve.
const RESERVE: u64 = 2 << 30;

fn load(path: &str, kv: KvTier) -> Option<Qwen35GgufInstance> {
    let devices: Vec<(Device, u64)> = gpu_core::devices::gpus()
        .iter()
        .map(|d| (Device::Gpu(d.index), d.identity.vram_bytes.saturating_sub(RESERVE)))
        .filter(|&(_, usable)| usable > 0)
        .collect();
    if devices.is_empty() {
        brain_testutil::skip_unavailable("no GPU with queryable VRAM");
        return None;
    }
    let r = Qwen35GgufResident::new(path.to_string(), devices, CAP, TierPolicy::uniform(Dtype::I8)).with_kv_tier(kv);
    let placed: Vec<Device> = r.estimate_multi(&r.instance_key("generate", &capability::Invocation::new())).devices().collect();
    if placed.is_empty() {
        brain_testutil::skip_unavailable("the checkpoint does not fit the GPUs this run may use");
        return None;
    }
    Some(r.activate_owned(&placed).expect("activate the real checkpoint"))
}

fn l2(v: &[f32]) -> f64 {
    v.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt()
}

fn rel_l2(want: &[f32], got: &[f32]) -> f64 {
    l2(&want.iter().zip(got).map(|(w, g)| w - g).collect::<Vec<_>>()) / l2(want)
}

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap()
}

/// A real natural-language prompt of about 1k tokens: a fixed page of prose
/// kept beside the test so the prompt cannot change underneath it.
fn prompt_text() -> String {
    let body = include_str!("fixtures/real_prompt.txt");
    format!("{body}\n\nIn one sentence, what does the page above describe?")
}

#[test]
fn compact_kv_tiers_agree_with_the_f32_cache_on_a_real_prompt() {
    let Ok(path) = std::env::var(GGUF_ENV) else {
        brain_testutil::skip(&format!("{GGUF_ENV} unset (set it to a downloaded Qwen3.8-27B*.gguf to run this)"));
        return;
    };
    let text = prompt_text();

    let Some(reference) = load(&path, KvTier::F32) else { return };
    assert_eq!(reference.kv_tier(), KvTier::F32);
    let prompt = reference.tokenize(&text);
    assert!(prompt.len() > 700 && prompt.len() + (STEPS as usize) < CAP as usize, "prompt of {} tokens is out of range for this test", prompt.len());
    let trace = reference.tape_comparison_trace(&prompt, STEPS).expect("f32 trace");
    drop(reference);
    println!("prompt {} tokens; f32 greedy continuation: {:?}", prompt.len(), trace.ids);

    // The floor: the f32 cache through its two dispatch shapes.
    let floor = Stats::of(&trace.decode, &trace.chunk, &trace.ids);
    println!("f32 decode tape vs f32 chunk tape (the numerics' own floor): {floor}");

    for (tier, max_floor_multiple, min_top1) in [(KvTier::Bf16, 2.0, 0.9), (KvTier::Int8, 3.0, 0.9)] {
        if std::env::var("KV_TIER_ONLY").is_ok_and(|only| only != tier.as_str()) {
            continue;
        }
        let Some(inst) = load(&path, tier) else { return };
        assert_eq!(inst.kv_tier(), tier);
        let rows = inst.teacher_forced_logits(&prompt, &trace.ids).expect("forced logits");
        drop(inst);
        for (i, got) in rows.iter().enumerate() {
            assert!(got.iter().all(|x| x.is_finite()), "{tier}: step {i} has non-finite logits");
        }
        let st = Stats::of(&trace.decode, &rows, &trace.ids);
        println!("{tier} vs f32 cache: {st}");
        // A flipped argmax is only informative next to how close the reference
        // already was to flipping it.
        for (i, (got, want)) in rows.iter().zip(&trace.decode).enumerate() {
            if argmax(got) != trace.ids[i] {
                let mut sorted: Vec<f32> = want.clone();
                sorted.sort_by(|a, b| b.total_cmp(a));
                println!("  {tier} step {i}: picks {} where f32 picked {}; the f32 margin over its runner-up was {:.3}", argmax(got), trace.ids[i], sorted[0] - sorted[1]);
            }
        }
        assert!(
            st.mean_rel_l2 <= max_floor_multiple * floor.mean_rel_l2,
            "{tier}: mean logits rel-L2 {:.3e} exceeds {max_floor_multiple}x the numerics' own floor {:.3e}",
            st.mean_rel_l2,
            floor.mean_rel_l2
        );
        assert!(st.top1 >= min_top1, "{tier}: top-1 agrees with the f32 cache at only {:.1}% of positions (< {:.0}%)", st.top1 * 100.0, min_top1 * 100.0);
    }
}

/// How one set of logits rows differs from a reference set.
struct Stats {
    mean_rel_l2: f64,
    worst_rel_l2: f64,
    /// Fraction of positions whose argmax is the reference's greedy token.
    top1: f64,
    /// Mean overlap of the two top-5 token sets.
    top5_overlap: f64,
}

impl Stats {
    fn of(want: &[Vec<f32>], got: &[Vec<f32>], greedy: &[u32]) -> Stats {
        let n = want.len() as f64;
        let rel: Vec<f64> = want.iter().zip(got).map(|(w, g)| rel_l2(w, g)).collect();
        let top5 = |v: &[f32]| -> Vec<u32> {
            let mut idx: Vec<u32> = (0..v.len() as u32).collect();
            idx.sort_by(|&a, &b| v[b as usize].total_cmp(&v[a as usize]));
            idx.truncate(5);
            idx
        };
        let overlap: f64 = want.iter().zip(got).map(|(w, g)| top5(w).iter().filter(|t| top5(g).contains(t)).count() as f64 / 5.0).sum::<f64>() / n;
        Stats {
            mean_rel_l2: rel.iter().sum::<f64>() / n,
            worst_rel_l2: rel.iter().copied().fold(0.0, f64::max),
            top1: got.iter().zip(greedy).filter(|(g, &id)| argmax(g) == id).count() as f64 / n,
            top5_overlap: overlap,
        }
    }
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "logits rel-L2 mean {:.3e} worst {:.3e}; top-1 {:.1}%; top-5 overlap {:.1}%", self.mean_rel_l2, self.worst_rel_l2, self.top1 * 100.0, self.top5_overlap * 100.0)
    }
}
