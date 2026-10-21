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
//! dropped first, so the card holds one model at a time), then each of 20 fixed
//! real prompts (~1k tokens) through the chunked prefill tape and 64 forced
//! decode steps through the decode tape:
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
//! scored by its two dispatch shapes (the decode tape and the one-row chunk tape
//! with [`Qwen35GgufInstance::set_exact_rounds`] off,
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
use qwen35::int8_gguf_resident::{Qwen35GgufInstance, Qwen35GgufResident};
use residency::multi::MultiDeviceResidentModel;
use residency::{Device, ResidentModel};

/// Per-sequence `prompt + forced` ceiling the instances are built for.
const CAP: u32 = 2048;
/// Forced decode positions scored after the prompt.
const STEPS: u32 = 64;
/// The bars below are derived from at least this many prompts; fewer cannot separate the tiers.
const MIN_PROMPTS: usize = 20;
/// Per tier: the most the pooled mean logits rel-L2 may be times the f32 cache's own floor.
///
/// Measured on the real Qwen3.8-27B Q8_0, 20 prompts x 64 forced steps = 1280
/// positions (log kept under runs/qevid, 2026-10-21):
/// perf-number: measured accuracy evidence, floor and tier rel-L2
/// the floor is mean rel-L2 0.1168; bf16 / int8 / fp8 sit at 0.1389 / 0.1425 / 0.1389,
/// perf-number: the same measurement as a multiple of the floor
/// i.e. 1.19x / 1.22x / 1.19x of it. 1.5x leaves room for a rerun and fails a
/// tier that really loses information.
const MAX_FLOOR_MULTIPLE: f64 = 1.5;
/// The least pooled top-1 agreement with the f32 cache any compact tier may have.
///
/// Measured over the same 1280 positions:
/// perf-number: measured accuracy evidence, top-1 agreement
/// the numerics' own floor agrees at 92.4%, bf16 / int8 / fp8 at 92.0% / 91.9% / 91.6%.
/// One position in 1280 moves a pooled figure by under a tenth of a point and its
/// binomial standard error is about 0.8 points, so the bf16 baseline minus a
/// 3-point margin (about 4 standard errors) is the bar for every tier: fp8 must
/// hold what bf16 holds, within noise. (The earlier bar was lowered after one
/// failing prompt of 24 positions, where a single flip is 4 points.)
const MIN_TOP1: f64 = 0.89;
/// Bytes kept free per card, `brain serve`'s default reserve.
const RESERVE: u64 = 2 << 30;
const TIERS: [KvTier; 3] = [KvTier::BF16, KvTier::INT8, KvTier::FP8];

fn load(path: &str, kv: KvTier) -> Option<Qwen35GgufInstance> {
    let devices: Vec<(Device, u64)> = qwen35::int8_gguf_resident::budgeted_gpus(&gpu_core::devices::ambient_compute_set().gpus, RESERVE);
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

/// The fixed real prompts: the page the sweeps use plus 19 passages of this
/// repository's own documentation, kept beside the test so the set cannot
/// change underneath it. Each ends with the same one-sentence question.
fn prompt_texts() -> Vec<String> {
    let first = include_str!("fixtures/real_prompt.txt");
    let rest = include_str!("fixtures/real_prompts.txt");
    std::iter::once(first).chain(rest.split("\n=====\n")).map(|body| format!("{}\n\nIn one sentence, what does the page above describe?", body.trim())).collect()
}

/// One prompt's f32 reference: its greedy continuation, the decode-tape rows
/// the tiers are scored against, and the numerics floor of that prompt.
struct Reference {
    ids: Vec<u32>,
    decode: Vec<Vec<f32>>,
    floor: Stats,
}

#[test]
fn compact_kv_tiers_agree_with_the_f32_cache_on_real_prompts() {
    // Test-gating convention only - selects the real-checkpoint gate, it is
    // not a serving knob. The serving path finds its GGUFs in the model
    // store.
    const REAL_GGUF_ENV: &str = "BRAIN_QWEN35_GGUF";
    let Ok(path) = std::env::var(REAL_GGUF_ENV) else {
        brain_testutil::skip(&format!("{REAL_GGUF_ENV} unset (set it to a downloaded Qwen3.8-27B*.gguf to run this)"));
        return;
    };
    let texts = prompt_texts();
    assert!(texts.len() >= MIN_PROMPTS, "{} prompts is fewer than the {MIN_PROMPTS} the bars are derived from", texts.len());

    let Some(reference) = load(&path, KvTier::F32) else { return };
    assert_eq!(reference.kv_tier(), KvTier::F32);
    // The floor needs two DIFFERENT renderings of the same cache, and a short
    // chunk round now IS the decode tape unless that is withdrawn; the decode
    // side of the trace is unaffected either way.
    reference.set_exact_rounds(false);
    let mut prompts: Vec<Vec<u32>> = Vec::new();
    let mut refs: Vec<Reference> = Vec::new();
    for (n, text) in texts.iter().enumerate() {
        let prompt = reference.tokenize(text);
        assert!(prompt.len() > 400 && prompt.len() + (STEPS as usize) < CAP as usize, "prompt {n} of {} tokens is out of range for this test", prompt.len());
        let trace = reference.tape_comparison_trace(&prompt, STEPS).expect("f32 trace");
        // The floor: the f32 cache through its two dispatch shapes, the decode
        // tape and the chunk tape proper.
        let floor = Stats::of(&trace.decode, &trace.chunk, &trace.ids);
        println!("prompt {n}: {} tokens; f32 decode tape vs f32 chunk tape (the numerics' own floor): {floor}", prompt.len());
        refs.push(Reference { ids: trace.ids, decode: trace.decode, floor });
        prompts.push(prompt);
    }
    drop(reference);
    let floor = Stats::pooled(refs.iter().map(|r| &r.floor));
    println!("ALL {} prompts, f32 decode tape vs f32 chunk tape (the numerics' own floor): {floor}", refs.len());

    for tier in TIERS {
        if std::env::var("KV_TIER_ONLY").is_ok_and(|only| only != tier.as_str()) {
            continue;
        }
        let Some(inst) = load(&path, tier) else { return };
        assert_eq!(inst.kv_tier(), tier);
        let mut per_prompt = Vec::new();
        for (n, (prompt, want)) in prompts.iter().zip(&refs).enumerate() {
            let rows = inst.teacher_forced_logits(prompt, &want.ids).expect("forced logits");
            for (i, got) in rows.iter().enumerate() {
                assert!(got.iter().all(|x| x.is_finite()), "{tier}: prompt {n} step {i} has non-finite logits");
            }
            let st = Stats::of(&want.decode, &rows, &want.ids);
            println!("{tier} prompt {n} vs f32 cache: {st}");
            per_prompt.push(st);
        }
        drop(inst);
        let st = Stats::pooled(per_prompt.iter());
        println!("{tier} ALL {} prompts vs f32 cache: {st}", per_prompt.len());
        assert!(
            st.mean_rel_l2 <= MAX_FLOOR_MULTIPLE * floor.mean_rel_l2,
            "{tier}: mean logits rel-L2 {:.3e} exceeds {MAX_FLOOR_MULTIPLE}x the numerics' own floor {:.3e}",
            st.mean_rel_l2,
            floor.mean_rel_l2
        );
        assert!(st.top1 >= MIN_TOP1, "{tier}: top-1 agrees with the f32 cache at only {:.1}% of positions (< {:.1}%)", st.top1 * 100.0, MIN_TOP1 * 100.0);
    }
}

/// How one set of logits rows differs from a reference set.
struct Stats {
    /// Positions scored.
    n: usize,
    mean_rel_l2: f64,
    worst_rel_l2: f64,
    /// Fraction of positions whose argmax is the reference's greedy token.
    top1: f64,
    /// Mean overlap of the two top-5 token sets.
    top5_overlap: f64,
}

impl Stats {
    /// The statistics of several runs taken together, each position counting once.
    fn pooled<'a>(runs: impl IntoIterator<Item = &'a Stats>) -> Stats {
        let runs: Vec<&Stats> = runs.into_iter().collect();
        let n: usize = runs.iter().map(|s| s.n).sum();
        let mean = |f: fn(&Stats) -> f64| runs.iter().map(|s| f(s) * s.n as f64).sum::<f64>() / n as f64;
        Stats {
            n,
            mean_rel_l2: mean(|s| s.mean_rel_l2),
            worst_rel_l2: runs.iter().map(|s| s.worst_rel_l2).fold(0.0, f64::max),
            top1: mean(|s| s.top1),
            top5_overlap: mean(|s| s.top5_overlap),
        }
    }

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
            n: want.len(),
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
