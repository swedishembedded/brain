// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real-checkpoint gates for the **DFlash2 block-diffusion drafter**
//! (`qwen35::dflash2`) driving `Qwen35GgufInstance::generate_speculative`.
//!
//! Two questions, and as in `tests/gguf_resident_spec_real.rs` they are not
//! the same question.
//!
//! **Does it compute the right thing?** A drafter cannot be gated on output
//! text: every proposal it makes is verified by the target, so a drafter with
//! transposed conv taps produces exactly the same TOKENS as a correct one and
//! differs only in how often they are accepted. A silently-wrong port shows up
//! as a slightly disappointing speedup, which is indistinguishable from "this
//! architecture is not that good", which is the very question this work exists
//! to answer. So the device forward is gated against an independent host
//! reference over the same bytes - `tools/goldens/dflash2_reference_forward.py`,
//! a dependency-light re-implementation of the published `dflash/model.py` -
//! at the level of the proposed token ids AND the hidden state they come from.
//!
//! **Is it worth serving?** Measured the same way and on the same prompts the
//! synthetic drafters were, so the numbers are directly comparable to that
//! file's table. Two Tesla P40s, Q8_0 target served INT8, Q8_0 draft served
//! INT8, greedy, one load per test:
//!
//! The `was` columns are the same ladder before the qwen35 ledger's M29
//! removed a per-round cost from the verify tape; nothing about this drafter
//! differs between them.
//!
//! | drafter             | free-form: was | now | vs plain | acc/round | repetition: was | now | vs plain | acc/round |
//! |---------------------|------:|------:|------:|-----:|------:|------:|------:|-----:|
//! | plain decode        |  6.60 |  6.76 | 1.00x |  -   |  5.73 |  5.68 | 1.00x |  -   |
//! | chunk tape, no spec |  3.65 |  6.14 | 0.91x |  -   |  3.16 |  4.76 | 0.84x |  -   |
//! | n-gram, k=7         |  4.45 |  6.72 | 1.00x | 0.57 |  9.04 | 10.35 | 1.82x | 5.12 |
//! | **DFlash2, k=3**    |  7.10 | **10.13** | **1.50x** | 2.30 |  7.44 |  8.94 | 1.57x | 3.00 |
//! | DFlash2, k=5        |  6.73 |  9.12 | 1.35x | 3.12 |  7.37 |  8.60 | 1.51x | 4.44 |
//! | DFlash2, k=7        |  5.70 |  7.47 | 1.10x | 3.12 |  8.22 | **9.26** | **1.63x** | 6.00 |
//!
//! The free-form column is the result that matters: **DFlash2 is the first
//! drafter on this stack that is a net WIN on free-form text**, where the
//! model-free one is exactly break-even. It out-drafts n-gram by 4x there
//! (2.30 against 0.57 accepted per round) and still beats it on n-gram's own
//! best workload (6.00 against 5.12) - though not on wall clock there, because
//! n-gram costs nothing to run and this costs a 1.9B forward. Every `k` is now
//! a win on both workloads; `k = 7` on free-form used to be a 0.86x LOSS, for
//! a reason that was in the target's tape and not in the drafter.
//!
//! **Why `k` is swept rather than fixed at 7.** A wider window proposes more
//! per round but widens both the verify chunk and the re-commit a rejecting
//! round pays, so the best `k` is a property of the workload's acceptance rate
//! and not a constant: `k = 3` wins on free-form text, `k = 7` on repetition.
//!
//! The table above carries two columns because the per-round cost these
//! numbers are measured against CHANGED under them. When this port was first
//! measured, every speculative round paid a 256-token prefill round's
//! per-layer device drain (the qwen35 ledger's M29 profiles and removes it), so
//! a chunk forward cost ~2.4 plain decode steps almost regardless of its row
//! count, the `chunk tape, no spec` floor was 0.55x, and the real break-even
//! sat near 5 accepted tokens per round - which is why `k = 7` on free-form
//! text was a measured LOSS. The floor is now 0.91x and break-even is near 1,
//! so the same drafter, unchanged, is a win at every `k` swept here.
//!
//! Both checkpoints are needed and everything self-skips loudly without them:
//!
//! ```text
//! BRAIN_QWEN35_GGUF=$HOME/.local/share/brain/models/unsloth/Qwen3.8-27B-Q8_0.gguf \
//! BRAIN_DFLASH2_GGUF=$HOME/.local/share/brain/models/incoai/Qwen3.8-27B-DFlash2-GGUF/Q8_0.gguf \
//!   cargo test --release --offline -p brain-qwen35 --test dflash2_real \
//!   -- --nocapture --test-threads=1
//! ```
//!
//! `BRAIN_DFLASH2_DUMP=<dir>` additionally writes the host reference's inputs
//! (the tapped target hidden states, and the anchor/position they pair with)
//! so the Python oracle can be re-run against exactly what the device saw -
//! which is how the pinned numbers below were obtained and how they are
//! re-derived if the loader ever changes.

use std::time::Instant;

use gpu_core::select::Dtype;
use model::ops::TierPolicy;
use qwen35::dflash2::{Dflash2, Dflash2Drafter};
use qwen35::int8_gguf_resident::{Qwen35GgufInstance, Qwen35GgufResident};
use residency::multi::MultiDeviceResidentModel;
use residency::{Device, ResidentModel};

const CAP: u32 = 1024;
/// Per card, and deliberately larger than `gguf_resident_spec_real.rs`'s 2 GiB:
/// the draft model is ~2 GB of INT8 weights plus its own K/V cache, and it has
/// to fit BESIDE the target on a card the target's planner has already been
/// told it may fill. Reserving here is the whole mechanism - the planner
/// respects the usable-bytes figure it is given and nothing else.
const RESERVE: u64 = 6 << 30;

fn gguf_path() -> Option<String> {
    match std::env::var("BRAIN_QWEN35_GGUF") {
        Ok(p) if !p.is_empty() => Some(p),
        _ => {
            brain_testutil::skip("BRAIN_QWEN35_GGUF unset (set it to a downloaded Qwen3.8-27B*.gguf to run this)");
            None
        }
    }
}

fn draft_path() -> Option<String> {
    match std::env::var("BRAIN_DFLASH2_GGUF") {
        Ok(p) if !p.is_empty() => Some(p),
        _ => {
            brain_testutil::skip("BRAIN_DFLASH2_GGUF unset (set it to a downloaded Qwen3.8-27B-DFlash2 GGUF to run this)");
            None
        }
    }
}

fn real_devices() -> Vec<(Device, u64)> {
    gpu_core::devices::gpus()
        .iter()
        .map(|d| (Device::Gpu(d.index), d.identity.vram_bytes.saturating_sub(RESERVE)))
        .filter(|&(_, usable)| usable > 0)
        .collect()
}

fn load() -> Option<Qwen35GgufInstance> {
    let path = gguf_path()?;
    let devices = real_devices();
    if devices.is_empty() {
        brain_testutil::skip_unavailable("no GPU with enough free memory for the real checkpoint");
        return None;
    }
    let r = Qwen35GgufResident::new(path, devices, CAP, TierPolicy::uniform(Dtype::I8));
    let key = r.instance_key("generate", &capability::Invocation::new());
    let placed: Vec<Device> = r.estimate_multi(&key).devices().collect();
    Some(r.activate_owned(&placed).expect("activate the real checkpoint across the real cards"))
}

/// The draft model lands on the LAST stage's card - the one that also carries
/// the head epilogue, so the draft's hidden states reach the projection that
/// turns them into logits without crossing a third device.
fn load_draft(inst: &Qwen35GgufInstance) -> Option<Dflash2> {
    load_draft_dt(inst, Dtype::I8)
}

fn load_draft_dt(inst: &Qwen35GgufInstance, dt: Dtype) -> Option<Dflash2> {
    let path = draft_path()?;
    let gpu = inst.stage_gpu(inst.stages() - 1).new_like(qwen35::dflash2::pipelines());
    match Dflash2::load_dt(gpu, &path, CAP, dt) {
        Ok(m) => Some(m),
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("DFlash2 draft model did not load: {e}"));
            None
        }
    }
}

struct Run {
    text: String,
    ids: Vec<u32>,
    decode_s: f64,
    tok_s: f64,
}

fn plain(inst: &Qwen35GgufInstance, prompt: &[u32], max_new: u32) -> Run {
    let t = Instant::now();
    let ids = inst.generate(prompt, max_new, 0.0, 0, 1.0, 0, &mut |_| false).expect("plain greedy decode");
    inst.poll_wait();
    let decode_s = t.elapsed().as_secs_f64();
    Run { text: inst.detokenize(&ids), tok_s: ids.len() as f64 / decode_s, ids, decode_s }
}

fn speculative(
    inst: &Qwen35GgufInstance,
    prompt: &[u32],
    max_new: u32,
    k: u32,
    draft: &mut dyn FnMut(&[u32], u32) -> Vec<u32>,
) -> (Run, qwen35::model::SpecDecodeStats) {
    let t = Instant::now();
    let (ids, stats) = inst.generate_speculative(prompt, max_new, k, draft, &mut |_| false).expect("speculative decode");
    inst.poll_wait();
    let decode_s = t.elapsed().as_secs_f64();
    (Run { text: inst.detokenize(&ids), tok_s: ids.len() as f64 / decode_s, ids, decode_s }, stats)
}

fn report(label: &str, r: &Run, stats: Option<&qwen35::model::SpecDecodeStats>, base: f64) {
    match stats {
        Some(s) => println!(
            "  {label:<24} {:>6.2} tok/s  ({:.2}x)  |  {:>2} tokens in {:>4.1}s, {} target forwards, {}/{} draft accepted, {:.2} accepted/round",
            r.tok_s,
            r.tok_s / base,
            r.ids.len(),
            r.decode_s,
            s.target_forwards,
            s.accepted,
            s.proposed,
            s.accepted_per_round()
        ),
        None => println!("  {label:<24} {:>6.2} tok/s  (1.00x)  |  {:>2} tokens in {:>4.1}s", r.tok_s, r.ids.len(), r.decode_s),
    }
}

/// The n-gram drafter from `tests/gguf_resident_spec_real.rs`, copied verbatim
/// so the two files' numbers are the same measurement - this is the baseline
/// DFlash2 has to beat to be worth a second checkpoint at all.
fn ngram_draft(ctx: &[u32], want: u32) -> Vec<u32> {
    for n in (2usize..=4).rev() {
        if ctx.len() <= n {
            continue;
        }
        let suffix = &ctx[ctx.len() - n..];
        for start in (0..ctx.len() - n).rev() {
            if &ctx[start..start + n] == suffix {
                let from = start + n;
                if from < ctx.len() {
                    let end = (from + want as usize).min(ctx.len());
                    let out = ctx[from..end].to_vec();
                    if !out.is_empty() {
                        return out;
                    }
                }
            }
        }
    }
    Vec::new()
}

fn digest(label: &str, v: &[f32]) -> String {
    let rms = (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
    format!("{label} rms={rms:.6} sum={:+.6} first={:?}", v.iter().sum::<f32>(), &v[..4.min(v.len())])
}

/// **The correctness gate**: one draft block, on a fixed prompt, against an
/// independent host reference over the same bytes.
///
/// A drafter cannot be gated on its text. Every proposal it makes is verified
/// by the target, so a wrong port still produces correct output and shows up
/// only as an acceptance rate that is mysteriously low - which is
/// indistinguishable from "this architecture is not that good", the very
/// question this work exists to answer. So the gate is exact equality of the
/// proposed ids against `tools/goldens/dflash2_reference_forward.py`, plus the
/// pre-final-norm hidden state they were read off (the ids are two nested step
/// functions over it, and a conv tap read from the wrong axis can move the
/// hidden state a long way without moving an easy prompt's argmax at all).
///
/// **It runs at the fp32 weight tier**, which is the only tier the claim can
/// be made at. The exactly-equal comparison is against a reference that does
/// not quantize activations; at INT8 the two would differ by ~10% on a hidden
/// state this model has amplified by five orders of magnitude from an
/// embedding of RMS 0.014, and no token-level equality survives that. What the
/// INT8 tier costs is measured instead, below, and it is not nothing.
///
/// How the expected values were obtained, and how to re-derive them:
///
/// ```text
/// BRAIN_DFLASH2_DUMP=[dump-dir] cargo test ... one_draft_block_matches
/// tools/goldens/dflash2_reference_forward.py --gguf [dflash.gguf] \
///     --target-gguf [qwen.gguf] --hidden [dump-dir]/target_hidden.f32 \
///     --anchor 369 --start 5 --digest
/// ```
#[test]
fn one_draft_block_matches_the_host_reference() {
    let Some(inst) = load() else { return };
    if draft_path().is_none() {
        return;
    }
    let prompt = inst.tokenize("The capital city of France is");
    let anchor = *prompt.last().expect("non-empty prompt");
    let pos = prompt.len() as u32 - 1;

    // Fill the taps exactly as the drafter's first round would see them: the
    // speculative loop replays `prompt[..len-1]` and then verifies one chunk,
    // so after one no-op round the target's hidden states for positions
    // `0..pos` exist and the anchor is `prompt`'s last token.
    inst.enable_hidden_taps(&[5, 19, 33, 47, 61]);
    let _ = speculative(&inst, &prompt, 1, 7, &mut |_c: &[u32], _w: u32| Vec::new());
    let rows = inst.target_hidden(0, pos).expect("tapped target hidden states");
    assert_eq!(rows.len(), pos as usize * inst.hidden_tap_width(), "the tap handed back the wrong shape");

    if let Ok(dir) = std::env::var("BRAIN_DFLASH2_DUMP") {
        if !dir.is_empty() {
            let bytes: Vec<u8> = rows.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(format!("{dir}/target_hidden.f32"), &bytes).expect("write the hidden dump");
            std::fs::write(format!("{dir}/block.txt"), format!("anchor={anchor}\nstart={pos}\nprompt={prompt:?}\n")).expect("write the block description");
            println!("  dumped {pos} context rows to {dir}/target_hidden.f32 (anchor={anchor}, start={pos})");
        }
    }

    // One draft at each tier, loaded one at a time: fp32 is 7.4 GB and INT8 is
    // 2 GB, and they share a card with 9 GB of the target.
    let mut drafts: Vec<(Dtype, Vec<u32>, Vec<f32>)> = Vec::new();
    for dt in [Dtype::F32, Dtype::I8] {
        // `reclaiming`: this loop drops a whole model per iteration, 7.4 GB of
        // device buffers at the fp32 tier - exactly the allocate-then-drop
        // pattern `gpu_core`'s own reclamation guard refuses to let a caller
        // get wrong (it must drop first and poll after, not the reverse).
        let done = gpu_core::reclaiming(inst.stage_gpu(inst.stages() - 1), || {
            let model = load_draft_dt(&inst, dt)?;
            if dt == Dtype::F32 {
                println!("\nDFlash2 config: {:?}", model.cfg);
            }
            model.append_context(&rows, 0).expect("project the tapped context");
            let mut ids = vec![anchor];
            ids.extend(std::iter::repeat_n(model.cfg.mask_token_id, (model.cfg.block_size - 1) as usize));
            let noise = inst.embed_rows_of(&ids).expect("noise embedding from the target's table");
            let hidden = model.denoise_block(&noise, pos).expect("denoise the block");
            let d = model.cfg.d_model as usize;
            let logits = inst.logits_with_norm(&hidden[d..], model.cfg.block_size - 1, model.output_norm()).expect("head projection");
            let path = model.select(&hidden[d..], &logits, anchor).expect("selector walk");
            println!("\n{dt:?} tier: proposals {path:?} = {:?}", inst.detokenize(&path));
            for i in 0..(model.cfg.block_size - 1) as usize {
                println!("  {}", digest(&format!("pre-norm row {i}"), &hidden[(i + 1) * d..(i + 2) * d]));
            }
            Some((dt, path, hidden[d..2 * d].to_vec()))
        });
        let Some(got) = done else { return };
        drafts.push(got);
    }

    let (_, f32_path, f32_row0) = &drafts[0];
    let (_, i8_path, i8_row0) = &drafts[1];

    // The hidden state first: this is what a transposed tap or a mis-strided
    // RoPE moves, and it moves it by far more than 1%.
    let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
    let rel = (rms(f32_row0) - HOST_REFERENCE_ROW0_RMS).abs() / HOST_REFERENCE_ROW0_RMS;
    println!("\n  fp32 pre-norm row 0 rms {:.4} against the host reference's {HOST_REFERENCE_ROW0_RMS:.4} ({:.3}% apart)", rms(f32_row0), 100.0 * rel);
    assert!(rel < 1e-3, "the fp32 device forward and the host reference disagree on the block's hidden state by {:.2}%", 100.0 * rel);

    assert_eq!(
        f32_path,
        &HOST_REFERENCE_PROPOSALS.to_vec(),
        "the fp32 device drafter and the host reference disagree on what to propose"
    );

    // **What the INT8 tier costs**, measured rather than assumed. The hidden
    // states barely move; what moves is the SELECTOR's greedy walk, which is a
    // left-to-right chain - one flipped near-tie changes the predecessor every
    // later position is scored against, so the two paths share a prefix and
    // then diverge completely. On this prompt the reference's own margin
    // between the top two candidates at position 1 is 0.0086 on scores of
    // order 23, i.e. a coin flip that INT8 loses.
    let shared = f32_path.iter().zip(i8_path).take_while(|(a, b)| a == b).count();
    println!(
        "  INT8 tier agrees with fp32 for {shared} of {} positions, then diverges: {:?} against {:?}",
        f32_path.len(),
        inst.detokenize(i8_path),
        inst.detokenize(f32_path)
    );
    println!("  INT8 pre-norm row 0 rms {:.4} ({:.2}% off fp32)", rms(i8_row0), 100.0 * (rms(i8_row0) - rms(f32_row0)).abs() / rms(f32_row0));
    assert_eq!(i8_path[0], f32_path[0], "even at INT8 the first proposal must be the one the head is most confident about");
}

/// The host reference's answer for `"The capital city of France is"` at
/// `block_size = 8`, `temperature = 0`, reading the same Q8_0 GGUF the device
/// reads - `tools/goldens/dflash2_reference_forward.py`.
const HOST_REFERENCE_PROPOSALS: [u32; 7] = [11751, 13, 198, 248069, 271, 332, 314];

/// The same reference's pre-final-norm RMS at the first mask row.
const HOST_REFERENCE_ROW0_RMS: f32 = 1089.6241;

/// The ORIGINAL bf16 checkpoint's answer to the identical question, through
/// the authors' own `dflash/model.py` under `transformers` - reproduced
/// exactly by the same host reference run with `--safetensors`, which is what
/// establishes that the reference (and therefore the port gated against it)
/// implements the published architecture and not merely a self-consistent one.
///
/// It is NOT what the device is asserted against, because the device reads the
/// Q8_0 file. The gap between the two lists is the entire measured cost of
/// that quantization on this prompt, and it is larger than the near-identical
/// hidden states suggest: `[" Paris", ",", " and", " the", " capital", " of",
/// " Paris"]` against `[" Paris", ".", "\n", "</think>", "\n\n", "**", " of"]`.
/// One flipped near-tie at position 1, amplified by the selector's greedy
/// left-to-right walk.
#[allow(dead_code)]
const ORIGINAL_BF16_PROPOSALS: [u32; 7] = [11751, 11, 321, 279, 6511, 314, 11751];

/// **The throughput ladder**, laid out exactly like
/// `tests/gguf_resident_spec_real.rs::speculative_decode_speedup_ladder` so the
/// two are one table: plain decode, the same loop with nothing to speculate on,
/// the model-free n-gram drafter, and DFlash2.
///
/// All of them must produce the SAME tokens as the no-speculation floor. A
/// drafter that changed the output would not be a fast drafter, it would be a
/// different model, and the losslessness of this loop is not DFlash2's to
/// break - it is verified by the target either way, which is what makes this
/// assertion a real check on the WIRING (positions, the anchor split, the
/// context watermark) rather than on the draft model's quality.
#[test]
fn dflash2_speedup_ladder_on_a_free_form_prompt() {
    let Some(inst) = load() else { return };
    let Some(model) = load_draft(&inst) else { return };
    let prompt = inst.tokenize("The capital city of France is");
    let max_new = 32u32;

    let base = plain(&inst, &prompt, max_new);
    println!("\nDFlash2 ladder, free-form prompt, {max_new} tokens, greedy, one load:");
    report("plain decode", &base, None, base.tok_s);

    let (floor, floor_stats) = speculative(&inst, &prompt, max_new, 7, &mut |_c: &[u32], _w: u32| Vec::new());
    report("chunk tape, no spec", &floor, Some(&floor_stats), base.tok_s);

    let (ng, ng_stats) = speculative(&inst, &prompt, max_new, 7, &mut ngram_draft);
    report("n-gram draft (k=7)", &ng, Some(&ng_stats), base.tok_s);

    // **Swept over `k`, not measured at one value.** A wider window proposes
    // more per round but makes the verify chunk wider AND makes the re-commit
    // a rejecting round pays wider, and on this stack neither of those is
    // free: the chunk tape's own floor row above is already 0.57x, and an
    // 8-row chunk is not an 8-times-cheaper 1-row chunk. Quoting `k = 7`
    // alone would report the model's best acceptance at the machine's worst
    // per-round cost. The drafter answers any `k` up to `block_size - 1`; the
    // reference's own loop already shortens the block at the tail of a
    // generation, so this is its supported range, not an extrapolation.
    let drafter = Dflash2Drafter::new(&inst, model);
    let mut best: Option<(u32, Run, qwen35::model::SpecDecodeStats)> = None;
    for k in [3u32, 5, 7] {
        drafter.reset();
        let (df, df_stats) = speculative(&inst, &prompt, max_new, k, &mut |c: &[u32], w: u32| drafter.propose(c, w));
        report(&format!("DFlash2 (k={k})"), &df, Some(&df_stats), base.tok_s);
        assert_eq!(df.ids, floor.ids, "DFlash2 speculative decode changed the output at k={k} - the wiring, not the drafter, is wrong");
        assert!(df_stats.proposed > 0, "the DFlash2 drafter proposed nothing at k={k}, so nothing here was measured");
        if best.as_ref().is_none_or(|(_, b, _)| df.tok_s > b.tok_s) {
            best = Some((k, df, df_stats));
        }
    }
    let (bk, df, df_stats) = best.expect("the sweep ran at least one k");
    println!("  text: {:?}", df.text);
    let cost = drafter.stats();
    println!(
        "  drafter cost over the whole sweep: {} calls, {:.2}s drafting ({:.0} ms/call), {:.2}s selecting ({:.0} ms/call)",
        cost.calls,
        cost.draft_s,
        1000.0 * cost.draft_s / cost.calls.max(1) as f64,
        cost.select_s,
        1000.0 * cost.select_s / cost.calls.max(1) as f64
    );
    println!(
        "  -> best at k={bk}: {:.2} accepted draft tokens per round and {:.2}x plain decode \
         (n-gram on this same prompt: {:.2} accepted/round, {:.2}x)",
        df_stats.accepted_per_round(),
        df.tok_s / base.tok_s,
        ng_stats.accepted_per_round(),
        ng.tok_s / base.tok_s
    );
    assert!(
        df_stats.accepted_per_round() > ng_stats.accepted_per_round(),
        "DFlash2 accepted {:.2} draft tokens per round against the model-free n-gram drafter's {:.2} - a second checkpoint that does not out-draft a suffix match is not worth serving",
        df_stats.accepted_per_round(),
        ng_stats.accepted_per_round()
    );
}

/// The same ladder on the workload where the model-free drafter is at its
/// BEST - quoting a passage back, where an n-gram match is right for long
/// stretches. Measured separately because "DFlash2 beats n-gram" is only worth
/// claiming where n-gram is actually good.
#[test]
fn dflash2_speedup_ladder_on_a_repetition_workload() {
    let Some(inst) = load() else { return };
    let Some(model) = load_draft(&inst) else { return };
    let passage = "The Gated DeltaNet recurrence maintains a matrix-valued state that is \
updated by a delta rule at every token, and decayed by a per-head gate. Because the state \
is a single matrix rather than a growing cache, its memory cost does not depend on the \
sequence length, which is what makes the hybrid layer stack cheaper to serve at long \
context than a pure attention stack would be.";
    let prompt = inst.tokenize(&format!("{passage}\n\nRepeat the paragraph above exactly, word for word:\n\n"));
    let max_new = 48u32;
    println!("\nDFlash2 ladder, repetition workload: {}-token prompt, {max_new} tokens, greedy:", prompt.len());

    let base = plain(&inst, &prompt, max_new);
    report("plain decode", &base, None, base.tok_s);

    let (floor, floor_stats) = speculative(&inst, &prompt, max_new, 7, &mut |_c: &[u32], _w: u32| Vec::new());
    report("chunk tape, no spec", &floor, Some(&floor_stats), base.tok_s);

    let (ng, ng_stats) = speculative(&inst, &prompt, max_new, 7, &mut ngram_draft);
    report("n-gram draft (k=7)", &ng, Some(&ng_stats), base.tok_s);

    let drafter = Dflash2Drafter::new(&inst, model);
    let mut best: Option<(u32, Run, qwen35::model::SpecDecodeStats)> = None;
    for k in [3u32, 5, 7] {
        drafter.reset();
        let (df, df_stats) = speculative(&inst, &prompt, max_new, k, &mut |c: &[u32], w: u32| drafter.propose(c, w));
        report(&format!("DFlash2 (k={k})"), &df, Some(&df_stats), base.tok_s);
        assert_eq!(df.ids, floor.ids, "DFlash2 speculative decode changed the output on the repetition workload at k={k}");
        if best.as_ref().is_none_or(|(_, b, _)| df.tok_s > b.tok_s) {
            best = Some((k, df, df_stats));
        }
    }
    let (bk, df, df_stats) = best.expect("the sweep ran at least one k");
    println!("  text: {:?}", df.text);
    println!(
        "  -> best DFlash2 k={bk}: {:.2} accepted/round ({:.2}x); n-gram {:.2} accepted/round ({:.2}x)",
        df_stats.accepted_per_round(),
        df.tok_s / base.tok_s,
        ng_stats.accepted_per_round(),
        ng.tok_s / base.tok_s
    );
}

/// A diagnostic, not a gate: what the drafter's own hidden state and its
/// per-position top-1 look like, next to the token the selector actually
/// chose. Prints the same digests the host reference prints under `--digest`,
/// so a disagreement can be localized to a layer rather than to "the numbers
/// differ".
#[test]
fn what_the_drafter_sees() {
    let dt = match std::env::var("BRAIN_DFLASH2_TIER").as_deref() {
        Ok("f32") => Dtype::F32,
        _ => Dtype::I8,
    };
    let Some(inst) = load() else { return };
    let Some(model) = load_draft_dt(&inst, dt) else { return };
    println!("\ndraft weight tier: {dt:?}");
    let prompt = inst.tokenize("The capital city of France is");
    let layers = model.cfg.target_layers.clone();
    inst.enable_hidden_taps(&layers);

    let mut once = true;
    let _ = speculative(&inst, &prompt, 8, 7, &mut |ctx: &[u32], _want: u32| {
        if !once {
            return Vec::new();
        }
        once = false;
        let pos = ctx.len() as u32 - 1;
        let anchor = *ctx.last().expect("non-empty context");
        let rows = inst.target_hidden(0, pos).expect("tapped hidden states");
        let w = inst.hidden_tap_width();
        println!("\ntapped target hidden, {pos} rows x {w} ({} layers concatenated):", layers.len());
        for (b, l) in layers.iter().enumerate() {
            let band = &rows[(pos as usize - 1) * w + b * 5120..(pos as usize - 1) * w + (b + 1) * 5120];
            println!("  {}", digest(&format!("layer {l:>2} @ last ctx row"), band));
        }

        model.append_context(&rows, 0).expect("project the context");
        let mut ids = vec![anchor];
        ids.extend(std::iter::repeat_n(model.cfg.mask_token_id, 7));
        let noise = inst.embed_rows_of(&ids).expect("noise embedding");
        println!("  {}", digest("noise row 0 (anchor)", &noise[..5120]));
        println!("  {}", digest("noise row 1 (MASK) ", &noise[5120..10240]));
        let hidden = model.denoise_block(&noise, pos).expect("denoise");
        println!("draft hidden (pre-final-norm):");
        for i in 0..8 {
            println!("  {}", digest(&format!("row {i}"), &hidden[i * 5120..(i + 1) * 5120]));
        }
        let logits = inst.logits_with_norm(&hidden[5120..], 7, model.output_norm()).expect("head");
        let v = model.cfg.vocab as usize;
        let path = model.select(&hidden[5120..], &logits, anchor).expect("selector");
        println!("per-position argmax vs selector choice:");
        for i in 0..7 {
            let row = &logits[i * v..(i + 1) * v];
            let am = row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(j, _)| j).expect("non-empty");
            println!(
                "  row {i}: argmax={am:<7} ({:+.4}) {:?}   selector={:<7} ({:+.4}) {:?} {}",
                row[am],
                inst.detokenize(&[am as u32]),
                path[i],
                row[path[i] as usize],
                inst.detokenize(&[path[i]]),
                if am as u32 == path[i] { "same" } else { "MOVED" }
            );
        }
        Vec::new()
    });
}
