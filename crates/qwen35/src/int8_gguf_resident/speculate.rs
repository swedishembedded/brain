// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements speculative decoding - drafters, verifiers
// and the state rollback a recurrent stack needs - for clients serving large
// models on modest hardware. If your team needs expertise in serving a hybrid
// Gated DeltaNet model faster than one token per weight pass, then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Speculative decoding for the released-GGUF Qwen3.8-27B: the configuration an
//! operator sets, and the batched **verify** pass the generic scheduler
//! (`model::serve::speculate`) drives through [`super::Qwen35GgufEngine`].
//!
//! A verify pass scores `1 + k` consecutive tokens of every sequence in one
//! batched step, so the weights are read once for all of them. The
//! full-attention layers need nothing special; the Gated DeltaNet layers take a
//! sequence's rows in order from its pooled state and snapshot the state
//! between rows (`model::gdn_verify`), which is what lets a rejection put the
//! state back to the last accepted row without a second forward pass. The
//! snapshots live in per-layer scratch sized once for [`SpecServing::scratch_rows`]
//! packed rows (`state_len + hist_len` floats per row per Gated DeltaNet
//! layer, 150 MB per row at the 27B shape), charged into the placement beside
//! each recurrent layer ([`SpecServing::cost`], [`super::serving_cost`]).

use super::*;
use model::gdn_verify::GdnRowSnapshots;

use crate::model::verify::VerifyBatch;

/// What `--speculative` selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SpeculativeMode {
    /// No speculation: every decode step produces one token per sequence.
    #[default]
    Off,
    /// The best drafter the checkpoints on disk provide; off when there is none.
    Auto,
    /// Draft with the checkpoint's own multi-token-prediction head.
    Mtp,
    /// Draft with the DFlash2 block-diffusion draft model.
    Dflash,
    /// Draft by copying what the context repeats (prompt lookup): no second
    /// model, strong on code edits and tool-call arguments.
    Ngram,
}

impl SpeculativeMode {
    /// The flag spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            SpeculativeMode::Off => "off",
            SpeculativeMode::Auto => "auto",
            SpeculativeMode::Mtp => "mtp",
            SpeculativeMode::Dflash => "dflash",
            SpeculativeMode::Ngram => "ngram",
        }
    }

    /// Parse a `--speculative` value.
    pub fn parse(s: &str) -> Result<SpeculativeMode, String> {
        match s {
            "off" => Ok(SpeculativeMode::Off),
            "auto" => Ok(SpeculativeMode::Auto),
            "mtp" => Ok(SpeculativeMode::Mtp),
            "dflash" => Ok(SpeculativeMode::Dflash),
            "ngram" => Ok(SpeculativeMode::Ngram),
            other => Err(format!("unknown speculative mode '{other}': expected auto, off, mtp, dflash or ngram")),
        }
    }
}

/// How a resident is asked to speculate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpecServing {
    pub mode: SpeculativeMode,
    /// Hand the MTP head the target's final-norm output (`true`, the default)
    /// instead of its raw residual. The checkpoint was trained on the normed
    /// output: it accepts 0.72 of a code prompt's proposals against 0.62 for the
    /// raw residual.
    pub mtp_normed_hidden: bool,
    /// Packed verify rows the recurrent-state scratch holds: the most
    /// `sum(1 + k)` over the sequences of one verify step.
    pub scratch_rows: u32,
    /// Running sequences past which a step decodes plainly instead of
    /// speculating (the measured crossover). Unbounded by default.
    pub batch_limit: usize,
}

impl Default for SpecServing {
    fn default() -> SpecServing {
        SpecServing { mode: SpeculativeMode::Off, mtp_normed_hidden: true, scratch_rows: DEFAULT_SCRATCH_ROWS, batch_limit: usize::MAX }
    }
}

impl std::fmt::Display for SpeculativeMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Tokens of context the n-gram drafter matches on.
const NGRAM_ORDER: usize = 3;

/// Default scratch: 32 packed rows (4.7 GiB at the 27B shape) - 8 sequences at
/// a window of 3, or 4 at a window of 7.
pub const DEFAULT_SCRATCH_ROWS: u32 = 32;

/// What speculative decoding keeps on the cards, for the placement
/// ([`super::serving_cost`]): verify scratch rows beside every recurrent layer,
/// and whether the MTP head is resident beside the head.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpecCost {
    pub scratch_rows: u32,
    pub mtp_head: bool,
}

impl SpecCost {
    /// Nothing: speculation off, or nothing it would run with.
    pub const NONE: SpecCost = SpecCost { scratch_rows: 0, mtp_head: false };
}

impl SpecServing {
    /// What this configuration will really allocate: nothing when it decodes
    /// plainly - speculation off, the device without the verify kernels
    /// (`verify_kernels`), or `auto` on a checkpoint with no MTP block
    /// (`has_mtp`) - else the verify scratch, and the MTP head for a mode that
    /// loads it. `mtp` on a checkpoint without one is refused when it loads.
    pub fn cost(&self, has_mtp: bool, verify_kernels: bool) -> SpecCost {
        let scratch = SpecCost { scratch_rows: self.scratch_rows, mtp_head: false };
        match self.mode {
            SpeculativeMode::Off | SpeculativeMode::Dflash => SpecCost::NONE,
            _ if !verify_kernels || self.scratch_rows == 0 => SpecCost::NONE,
            SpeculativeMode::Ngram => scratch,
            SpeculativeMode::Auto | SpeculativeMode::Mtp if has_mtp => SpecCost { mtp_head: true, ..scratch },
            SpeculativeMode::Auto | SpeculativeMode::Mtp => SpecCost::NONE,
        }
    }
}

impl Qwen35GgufResident {
    /// Speculative decoding for this resident ([`SpecServing`]). What it will
    /// allocate is part of the placement ([`SpecServing::cost`]). Must be set
    /// before activation.
    pub fn with_speculation(mut self, spec: SpecServing) -> Qwen35GgufResident {
        self.spec = spec;
        self
    }

    /// See [`Self::with_speculation`].
    pub fn speculation(&self) -> SpecServing {
        self.spec
    }

    /// Install the scheduler-side half of the configured speculation: the
    /// external n-gram drafter for [`SpeculativeMode::Ngram`] (the MTP head
    /// proposes from inside the engine, so it needs nothing here).
    pub fn configure_scheduler(&self, sched: &mut model::serve::Scheduler<Qwen35GgufEngine>) {
        sched.decoder_mut().set_spec_batch_limit(self.spec.batch_limit);
        if self.spec.mode == SpeculativeMode::Ngram {
            sched.set_drafter(Box::new(model::serve::NgramDrafter::new(NGRAM_ORDER)));
        }
    }
}

impl Qwen35GgufInstance {
    /// Allocate the verify scratch if `spec` asks for speculation and every
    /// stage's device can run the verify kernels; otherwise leave this instance
    /// non-speculative (the engine then reports `spec_max_draft() == 0` and the
    /// scheduler decodes plainly).
    pub(super) fn enable_speculation(&mut self, spec: SpecServing, gguf_path: &str, tier: &TierPolicy, gpu_index: usize) -> Result<(), String> {
        if spec.mode == SpeculativeMode::Off || spec.scratch_rows == 0 {
            return Ok(());
        }
        if spec.mode == SpeculativeMode::Dflash {
            return Err(format!(
                "{MODEL}: --speculative dflash needs a DFlash2 draft checkpoint, and none is configured: the drafter (qwen35::dflash2) is built but runs only on the single-stream path, not behind the batched scheduler"
            ));
        }
        if !self.shards.iter().all(|s| s.qwen35.verify_supported()) {
            return match spec.mode {
                SpeculativeMode::Mtp => Err(format!("{MODEL}: --speculative mtp needs the native CUDA verify kernels, which this device does not offer")),
                _ => Ok(()), // auto: decode plainly
            };
        }
        for s in &mut self.shards {
            let snaps = (0..s.caches.recurrent.len())
                .map(|l| if s.caches.recurrent[l] { s.qwen35.new_row_snapshots(spec.scratch_rows) } else { GdnRowSnapshots { state: s.qwen35.gpu.storage(1), hist: s.qwen35.gpu.storage(1) } })
                .collect();
            s.caches.verify_snaps = snaps;
        }
        self.spec_rows = spec.scratch_rows;
        let has_mtp = matches!(spec.mode, SpeculativeMode::Auto | SpeculativeMode::Mtp) && self.enable_mtp(gguf_path, tier, gpu_index, spec.mtp_normed_hidden)?;
        if spec.mode == SpeculativeMode::Mtp && !has_mtp {
            return Err(format!("{MODEL}: --speculative mtp: '{gguf_path}' carries no multi-token-prediction block"));
        }
        if spec.mode == SpeculativeMode::Auto && !has_mtp {
            // Nothing to draft with: do not pay for verify rows that no one will propose.
            self.spec_rows = 0;
            for s in &mut self.shards {
                s.caches.verify_snaps.clear();
            }
        }
        Ok(())
    }

    /// Packed rows one verify step may carry; 0 when this instance does not speculate.
    pub fn speculation_rows(&self) -> u32 {
        self.spec_rows
    }

    /// Most rows one sequence may carry through a verify step.
    pub fn max_verify_rows_per_sequence(&self) -> u32 {
        model::gdn_verify::MAX_VERIFY_ROWS
    }

    /// **The verify pass**: `rows[i]` are consecutive tokens of the sequence in
    /// slot `slots[i]`, the first at absolute position `positions[i]`; all of
    /// them go through every stage in one batched step. Returns the packed
    /// `[sum(rows), d_model]` pre-norm hidden block (sequence after sequence).
    /// The recurrent state of each sequence is left after its LAST row, with the
    /// state after every earlier row kept for [`Self::commit_verify`].
    pub(super) fn stack_verify_hidden(&self, rows: &[&[u32]], slots: &[u32], positions: &[u32]) -> Result<Vec<f32>, String> {
        let counts: Vec<u32> = rows.iter().map(|r| r.len() as u32).collect();
        let total: u32 = counts.iter().sum();
        if total > self.spec_rows {
            return Err(format!("{MODEL}: a verify step of {total} rows exceeds the {} rows of scratch", self.spec_rows));
        }
        let tokens: Vec<u32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        let row_slots: Vec<u32> = slots.iter().zip(&counts).flat_map(|(&s, &c)| std::iter::repeat_n(s, c as usize)).collect();
        let row_positions: Vec<u32> = positions.iter().zip(&counts).flat_map(|(&p, &c)| p..p + c).collect();
        if let Some(&p) = row_positions.iter().find(|&&p| p >= self.cap) {
            return Err(format!("{MODEL}: verify position {p} is beyond this instance's context capacity {}", self.cap));
        }
        let mut carry = self.embed_rows(&tokens)?;
        for s in &self.shards {
            let seqs = s.caches.batch(&row_slots, &row_positions);
            let caches = BatchDecodeCaches { gqa_kv: &s.caches.gqa_kv, gqa_cap: s.caches.kv_block(), seqs: &seqs, one_sequence: false, gdn_pool: Some(s.caches.gdn_pool()), memory: &[] };
            let verify = VerifyBatch::new(slots, &counts, &s.caches.verify_snaps);
            carry = s.qwen35.verify_batch_stage(&tokens, &caches, &verify, Some(&carry));
        }
        Ok(carry)
    }

    /// Settle the verify pass just run: sequence `i` keeps its first `keep[i]`
    /// rows, so its recurrent state goes back to the state after row
    /// `keep[i] - 1` (a sequence that keeps every row is already there). The
    /// full-attention rows past the kept ones are never read again once the
    /// block table is truncated.
    pub(super) fn commit_verify(&self, slots: &[u32], counts: &[u32], keep: &[usize]) {
        for s in &self.shards {
            let verify = VerifyBatch::new(slots, counts, &s.caches.verify_snaps);
            s.qwen35.commit_verify_stage(&s.caches.gdn_pool(), &verify, keep);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_mode_round_trips_through_its_flag_spelling() {
        for mode in [SpeculativeMode::Off, SpeculativeMode::Auto, SpeculativeMode::Mtp, SpeculativeMode::Dflash, SpeculativeMode::Ngram] {
            assert_eq!(SpeculativeMode::parse(mode.as_str()), Ok(mode));
        }
        let err = SpeculativeMode::parse("fast").unwrap_err();
        assert!(err.contains("auto, off, mtp"), "the refusal names the accepted values: {err}");
    }

    #[test]
    fn speculation_costs_only_what_it_will_allocate() {
        let spec = |mode| SpecServing { mode, scratch_rows: 4, ..SpecServing::default() };
        let scratch = SpecCost { scratch_rows: 4, mtp_head: false };
        assert_eq!(spec(SpeculativeMode::Off).cost(true, true), SpecCost::NONE);
        assert_eq!(spec(SpeculativeMode::Ngram).cost(true, true), scratch);
        assert_eq!(spec(SpeculativeMode::Auto).cost(true, true), SpecCost { mtp_head: true, ..scratch }, "the MTP head on top of the scratch");
        assert_eq!(spec(SpeculativeMode::Auto).cost(false, true), SpecCost::NONE, "auto on a checkpoint with no MTP block decodes plainly");
        assert_eq!(spec(SpeculativeMode::Auto).cost(true, false), SpecCost::NONE, "nor on a device without the verify kernels");
    }

    #[test]
    fn the_scratch_sits_beside_each_recurrent_layer_and_the_head_beside_the_head() {
        let cfg = Qwen35Config::tiny();
        let shape = ServingShape::new(128, TierPolicy::uniform(gpu_core::select::Dtype::I8), KvTier::F32, 2);
        let plain = serving_cost(&cfg, &shape);
        let spec = serving_cost(&cfg, &ServingShape { spec: SpecCost { scratch_rows: 4, mtp_head: true }, ..shape });
        for ((ty, a), b) in cfg.layer_types().into_iter().zip(&plain.per_layer).zip(&spec.per_layer) {
            assert_eq!(b > a, ty == LayerType::Linear, "{ty:?}: only a recurrent layer carries verify scratch");
        }
        assert!(spec.head > plain.head, "the MTP head is charged on the head's card");
    }
}
