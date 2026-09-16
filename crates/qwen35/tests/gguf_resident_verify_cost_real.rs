// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **The per-round cost of the CHUNK tape**, measured on the real checkpoint -
//! the one number that bounds speculative decoding on this model from above.
//!
//! Swedish Embedded AB implements inference-throughput engineering for edge and
//! on-premise LLM deployments for its clients. If your team needs expertise in
//! GPU dispatch scheduling and speculative decoding then you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! `Qwen35GgufInstance::generate_speculative` verifies every round through
//! `stack_chunk_carry`, the same function a 256-token prefill round uses. A
//! verify round is 1-8 rows. So any cost that is FIXED per round - paid once
//! whatever the row count - is amortized 256-fold in prefill and 32-to-256-fold
//! less in verify, and it shows up as a speculative decoder that cannot beat
//! plain decode no matter how good its drafter is.
//!
//! These gates measure that directly rather than inferring it from a
//! disappointing end-to-end speedup:
//!
//! * `a_one_row_chunk_round_is_not_slower_than_a_one_row_decode_step` is the
//!   claim the speculative FLOOR rests on: the two tapes compute the same
//!   function at one row, so the chunk tape has no right to be slower by a
//!   factor, and every tok/s speculation gives up to it is given up before any
//!   drafting has happened.
//! * `an_all_accepted_verify_round_beats_plain_decode_by_a_real_factor` is the
//!   CEILING: an 8-row round commits 8 tokens, so its cost against 8 decode
//!   steps is the best any drafter can buy here. Both are stated as multiples
//!   of this instance's own measured decode rate, so neither encodes the card's
//!   clock.
//! * `the_verify_round_cost_ladder` prints the whole ladder (1 to 256 rows, in
//!   both scratch regimes, layer half against head half, against a decode step
//!   and against the GDN snapshot) for the roadmap. It asserts nothing beyond
//!   the two gates above.
//!
//! Everything self-skips loudly without the real `Qwen3.8-27B*.gguf` named by
//! `BRAIN_QWEN35_GGUF`. Run:
//!
//! ```text
//! BRAIN_QWEN35_GGUF=$HOME/.local/share/brain/models/unsloth/Qwen3.8-27B-Q8_0.gguf \
//!   cargo test --release --offline -p brain-qwen35 --test gguf_resident_verify_cost_real \
//!   -- --nocapture --test-threads=1
//! ```

use gpu_core::select::Dtype;
use model::ops::TierPolicy;
use qwen35::int8_gguf_resident::{Qwen35GgufInstance, Qwen35GgufResident};
use residency::multi::MultiDeviceResidentModel;
use residency::{Device, ResidentModel};

/// Wide enough for a 256-row round profiled twice past a real prompt.
const CAP: u32 = 2048;
const RESERVE: u64 = 2 << 30;

fn load() -> Option<Qwen35GgufInstance> {
    let path = match std::env::var("BRAIN_QWEN35_GGUF") {
        Ok(p) if !p.is_empty() => p,
        _ => {
            brain_testutil::skip("BRAIN_QWEN35_GGUF unset (set it to a downloaded Qwen3.8-27B*.gguf to run this)");
            return None;
        }
    };
    let devices: Vec<(Device, u64)> = gpu_core::devices::gpus()
        .iter()
        .map(|d| (Device::Gpu(d.index), d.identity.vram_bytes.saturating_sub(RESERVE)))
        .filter(|&(_, usable)| usable > 0)
        .collect();
    if devices.is_empty() {
        brain_testutil::skip_unavailable("no GPU with enough free memory for the real checkpoint");
        return None;
    }
    let r = Qwen35GgufResident::new(path, devices, CAP, TierPolicy::uniform(Dtype::I8));
    let key = r.instance_key("generate", &capability::Invocation::new());
    let placed: Vec<Device> = r.estimate_multi(&key).devices().collect();
    Some(r.activate_owned(&placed).expect("activate the real checkpoint across the real cards"))
}

/// A prompt long enough that the profiled rounds sit at a realistic KV depth
/// rather than at position zero, and short enough to leave room for a 256-row
/// round profiled twice.
fn warm_prompt(inst: &Qwen35GgufInstance) -> Vec<u32> {
    inst.tokenize(
        "The Gated DeltaNet recurrence maintains a matrix-valued state that is updated by a \
delta rule at every token and decayed by a per-head gate. Because the state is a single \
matrix rather than a growing cache, its memory cost does not depend on the sequence length.",
    )
}

/// **The ladder**, for the record: what one round of the chunk tape costs at
/// every row count speculation and prefill actually use, in BOTH scratch
/// regimes, against one decode step.
///
/// Both columns, because the fix this file exists to gate is a threshold
/// between them (`Qwen35::set_chunk_arena_min_rows`), and a threshold quoted
/// from one measurement of one side is a guess. Pooled is the arena open - a
/// device fence at every layer boundary, which is what a 256-row prefill round
/// wants; unpooled is plain allocation and a `flush` per layer, the decode
/// tape's own discipline.
#[test]
fn the_verify_round_cost_ladder() {
    let Some(inst) = load() else { return };
    let prompt = warm_prompt(&inst);

    let dec = inst.profile_decode(&prompt, 8);
    println!("\nchunk-tape per-round cost, real checkpoint, {}-token warm prompt:", prompt.len());
    println!("  decode tape, 1 row          {:>8.1} ms/row   ({:.2} tok/s)", dec.wall_s * 1e3 / dec.steps as f64, dec.tok_per_s());
    // The OTHER per-round cost, measured rather than inferred from its byte
    // count: every round that proposes anything snapshots the recurrent state
    // before verifying, because whether a restore is needed is only known
    // after.
    println!("  GDN snapshot                {:>8.2} ms/round", inst.profile_gdn_snapshot(16));
    println!("   rows | pooled layer ms |  unpooled layer ms | pooled/unpooled | head ms | unpooled ms/row");
    for rows in [1u32, 3, 7, 8, 16, 32, 64, 128, 256] {
        let rounds = if rows >= 128 { 2 } else { 6 };
        inst.set_chunk_arena_min_rows(1);
        let pooled = inst.profile_chunk_round(&prompt, rows, rounds);
        inst.set_chunk_arena_min_rows(u32::MAX);
        let plain = inst.profile_chunk_round(&prompt, rows, rounds);
        println!(
            "  {rows:>5} | {:>15.1} | {:>18.1} | {:>15.2} | {:>7.1} | {:>15.2}",
            pooled.carry_ms(),
            plain.carry_ms(),
            pooled.carry_ms() / plain.carry_ms(),
            plain.head_ms(),
            plain.carry_ms_per_row()
        );
    }
}

/// **The ceiling an oracle drafter could reach**, computed from measured round
/// cost rather than from an end-to-end run: an 8-row verify round is the
/// `k = 7`, everything-accepted case, and it commits 8 tokens for one round's
/// cost. Divided by what those 8 tokens cost as plain decode steps, that is the
/// best speedup ANY drafter can buy on this stack.
///
/// This is the number the whole technique lives on, and the reason it is gated
/// here rather than only reported: it is a property of the target's tape, so it
/// can regress without any drafter changing and without any speculative test
/// failing - `tests/gguf_resident_spec_real.rs`'s ladder asserts token equality
/// and prints throughput, it does not assert throughput.
///
/// Stated as a multiple of THIS instance's own measured decode rate, so it does
/// not encode these cards' clock. The bound is a separator, not a target: the
/// per-layer drain this milestone removed put the same figure at 2.8x, and it
/// is 3.8x without it.
#[test]
fn an_all_accepted_verify_round_beats_plain_decode_by_a_real_factor() {
    let Some(inst) = load() else { return };
    let prompt = warm_prompt(&inst);

    let dec = inst.profile_decode(&prompt, 8);
    let step_ms = dec.wall_s * 1e3 / dec.steps as f64;
    let eight = inst.profile_chunk_round(&prompt, 8, 6);
    // 8 rows in, 8 tokens committed: `k = 7` proposals plus the correction row.
    let speedup = 8.0 * step_ms / eight.round_ms();
    println!(
        "\n8-row verify round {:.1} ms ({:.1} layer + {:.1} head) commits 8 tokens; 8 decode steps cost {:.1} ms -> {speedup:.2}x ceiling",
        eight.round_ms(),
        eight.carry_ms(),
        eight.head_ms(),
        8.0 * step_ms
    );
    assert!(
        speedup > 3.2,
        "an all-accepted 8-row verify round is only {speedup:.2}x plain decode, so no drafter however good can beat that - \
         the round is paying a per-round cost it should not (see the ladder test in this file)"
    );
}

/// **The claim speculation's floor rests on**: at one row the chunk tape and
/// the decode tape compute the same function, so the chunk tape must not be
/// dramatically slower.
///
/// It is allowed to be somewhat slower - the two run different mixers at one
/// row (a chunked-parallel GDN against the sequential recurrence, a causal
/// chunk fill against a paged single-row decode), and only the decode tape's
/// were shaped for `m = 1`. What it must not be is slower by a FACTOR, because
/// `generate_speculative` pays this on every round including the ones where the
/// drafter proposed nothing, which is exactly the regime a real drafter with a
/// middling acceptance rate spends most of its time in.
///
/// The bound separates the two states this has been in rather than naming an
/// ideal: with the per-layer drain a 1-row round measured 1.93x a decode step
/// (3.7 tok/s against 6.6), and without it 1.13x. The residual is the chunked
/// GDN form doing real work at `n = 1` that the recurrence does not, which is a
/// different change and one that would cost this path its single-tape
/// losslessness property.
#[test]
fn a_one_row_chunk_round_is_not_slower_than_a_one_row_decode_step() {
    let Some(inst) = load() else { return };
    let prompt = warm_prompt(&inst);

    let dec = inst.profile_decode(&prompt, 8);
    let step_ms = dec.wall_s * 1e3 / dec.steps as f64;
    let one = inst.profile_chunk_round(&prompt, 1, 8);
    // The decode step's own cost includes its head projection; compare like
    // with like by adding the chunk round's.
    let round_ms = one.round_ms();
    println!("\none decode step {step_ms:.1} ms vs one 1-row chunk round {round_ms:.1} ms ({:.2}x)", round_ms / step_ms);
    assert!(
        round_ms < step_ms * 1.30,
        "a 1-row verify round costs {round_ms:.1} ms against a decode step's {step_ms:.1} ms ({:.2}x) - speculative decoding \
         gives that factor up on every round before it wins anything back",
        round_ms / step_ms
    );
}
