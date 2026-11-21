// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Routing replay for mixture-of-experts reinforcement learning: the expert selection a rollout engine made, recorded
//! per token and per MoE call, and forced in the trainer's forward.
//!
//! An MoE router picks the top-k experts of each token from a score. When the serving engine and the trainer compute
//! that score with different kernels, reduction orders or precisions, a near-tie can fall the other way, and the
//! trainer then runs different experts than the ones that produced the sampled token: its log-probability of that
//! token is not the behaviour policy's, whatever the weights. Replay removes that source of gap. The engine records
//! which experts each row selected ([`RoutingLog`], written by the engine through the sequence's
//! [`crate::paged::BlockTable`]); the record travels with the completion ([`RoutingRecord`], on
//! [`crate::rollout::Completion::routing`]); the trainer lays the records of a batch out ([`RoutingReplay`]) and hands
//! each MoE call's ids to the router ([`crate::moe::router_fwd_replay`]), which keeps the recorded SELECTION and
//! computes the WEIGHTS from the trainer's own logits by the router's own formula, so the router backward is
//! unchanged.
//!
//! Keys. One MoE call is one execution of a router in forward order: a looped model that visits a layer several times
//! ([`crate::looped::VisitPlan`]) makes one call per visit, so a record is indexed by call, never by layer. A position
//! is the token's index in the sequence (prompt then generated tokens). A position whose routing was not recorded (a
//! prompt prefix the engine served from its prefix cache, the last sampled token that was never fed back) is masked
//! ([`RoutingRecord::recorded`]) and the trainer routes it itself.
//!
//! Records come from outside the trainer (another process, a file), so [`RoutingRecord::validate`] checks every field
//! against the model's MoE calls and names the first one that does not fit.
//!
//! Swedish Embedded AB implements solutions for training mixture-of-experts models on the exact routing they were
//! served with for its clients. If your team needs expertise in reinforcement learning on sparse models then you can
//! procure our services by sending an email to info@swedishembedded.com.

use std::fmt;

/// The id written for a position whose routing was not recorded: no expert has it.
pub const UNRECORDED: u32 = u32::MAX;

/// The largest `top_k` the replay kernel (`router_gate_replay.wgsl`) handles.
pub const MAX_REPLAY_TOP_K: u32 = 32;

/// One MoE call as a model runs it: how many experts its router chooses from and how many it keeps per row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouterCallShape {
    pub n_experts: u32,
    pub top_k: u32,
}

/// The experts one MoE call selected for every position of a sequence: `experts[pos * top_k .. (pos + 1) * top_k]`,
/// in the order the engine reported them, [`UNRECORDED`] throughout a position that was not recorded.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoutedCall {
    pub top_k: u32,
    pub experts: Vec<u32>,
}

/// The routing of one sequence as the engine that sampled it ran it.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoutingRecord {
    /// The policy version that sampled the sequence ([`crate::rollout::Completion::policy_version`]).
    pub policy_version: u64,
    /// Positions covered: the prompt and the generated tokens.
    pub positions: u32,
    /// `recorded[pos]`: whether every call's routing of `pos` was recorded.
    pub recorded: Vec<bool>,
    /// One entry per MoE call, in forward order.
    pub calls: Vec<RoutedCall>,
}

/// A routing record that does not fit the model or the rollout, naming the field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutingError {
    pub field: String,
    pub why: String,
}

impl RoutingError {
    fn new(field: impl Into<String>, why: impl Into<String>) -> RoutingError {
        RoutingError { field: field.into(), why: why.into() }
    }
}

impl fmt::Display for RoutingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "routing record {}: {}", self.field, self.why)
    }
}

impl std::error::Error for RoutingError {}

impl RoutingRecord {
    /// Check the record against the model's MoE calls (`calls`, in forward order) and, when given, the policy version
    /// the trainer expects: shapes, expert ranges, distinct experts per recorded row, an all-[`UNRECORDED`] row where
    /// the mask says unrecorded.
    pub fn validate(&self, calls: &[RouterCallShape], policy_version: Option<u64>) -> Result<(), RoutingError> {
        if let Some(want) = policy_version.filter(|&v| v != self.policy_version) {
            return Err(RoutingError::new("policy_version", format!("is {}, the trainer expects {want}", self.policy_version)));
        }
        let n = self.positions as usize;
        if self.recorded.len() != n {
            return Err(RoutingError::new("recorded", format!("has {} entries for {n} positions", self.recorded.len())));
        }
        if self.calls.len() != calls.len() {
            return Err(RoutingError::new("calls", format!("has {} MoE calls, the model makes {}", self.calls.len(), calls.len())));
        }
        for (c, (call, shape)) in self.calls.iter().zip(calls).enumerate() {
            if call.top_k != shape.top_k {
                return Err(RoutingError::new(format!("calls[{c}].top_k"), format!("is {}, the model's call keeps {}", call.top_k, shape.top_k)));
            }
            let k = call.top_k as usize;
            if k == 0 || call.top_k > MAX_REPLAY_TOP_K {
                return Err(RoutingError::new(format!("calls[{c}].top_k"), format!("is {k}; replay takes 1..={MAX_REPLAY_TOP_K}")));
            }
            if call.experts.len() != n * k {
                return Err(RoutingError::new(format!("calls[{c}].experts"), format!("has {} ids, {n} positions x top_k {k} need {}", call.experts.len(), n * k)));
            }
            for (pos, row) in call.experts.chunks_exact(k).enumerate() {
                if !self.recorded[pos] {
                    if let Some(j) = row.iter().position(|&e| e != UNRECORDED) {
                        return Err(RoutingError::new(format!("calls[{c}].experts[{}]", pos * k + j), format!("position {pos} is not recorded, so its ids must be {UNRECORDED}")));
                    }
                    continue;
                }
                for (j, &e) in row.iter().enumerate() {
                    if e >= shape.n_experts {
                        return Err(RoutingError::new(format!("calls[{c}].experts[{}]", pos * k + j), format!("{e} is not an expert of a {}-expert call", shape.n_experts)));
                    }
                    if row[..j].contains(&e) {
                        return Err(RoutingError::new(format!("calls[{c}].experts[{}]", pos * k + j), format!("expert {e} is selected twice at position {pos}")));
                    }
                }
            }
        }
        Ok(())
    }

    /// Positions whose routing was recorded.
    pub fn recorded_positions(&self) -> usize {
        self.recorded.iter().filter(|&&r| r).count()
    }
}

/// What an engine writes while it serves a sequence whose routing is being recorded: per MoE call, the experts each
/// position selected. A position written again (a recomputed or re-verified token) keeps the last write.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RoutingLog {
    calls: Vec<RoutedCall>,
    n_experts: Vec<u32>,
}

impl RoutingLog {
    /// An empty log for an engine whose MoE calls are `calls`, in forward order.
    pub fn new(calls: &[RouterCallShape]) -> RoutingLog {
        assert!(calls.iter().all(|c| (1..=MAX_REPLAY_TOP_K).contains(&c.top_k)), "RoutingLog::new: every call keeps 1..={MAX_REPLAY_TOP_K} experts, got {calls:?}");
        RoutingLog { calls: calls.iter().map(|c| RoutedCall { top_k: c.top_k, experts: Vec::new() }).collect(), n_experts: calls.iter().map(|c| c.n_experts).collect() }
    }

    /// Record that MoE call `call` (forward order) selected `experts` (its `top_k` ids) at sequence position
    /// `position`. Panics on a call the engine did not declare, a wrong id count or an id out of range: an engine bug.
    pub fn record(&mut self, call: usize, position: u32, experts: &[u32]) {
        assert!(call < self.calls.len(), "RoutingLog::record: call {call} of an engine that declared {} MoE calls", self.calls.len());
        let entry = &mut self.calls[call];
        let k = entry.top_k as usize;
        assert_eq!(experts.len(), k, "RoutingLog::record: call {call} keeps {k} experts, position {position} reports {}", experts.len());
        assert!(experts.iter().all(|&e| e < self.n_experts[call]), "RoutingLog::record: call {call} position {position} reports {experts:?}, outside its {} experts", self.n_experts[call]);
        let at = position as usize * k;
        if entry.experts.len() < at + k {
            entry.experts.resize(at + k, UNRECORDED);
        }
        entry.experts[at..at + k].copy_from_slice(experts);
    }

    /// The record of a sequence of `positions` positions sampled by policy `policy_version`. A position is recorded
    /// when every call wrote it; writes past `positions` (rejected speculative tokens) are dropped.
    pub fn into_record(self, positions: u32, policy_version: u64) -> RoutingRecord {
        let n = positions as usize;
        let mut calls = self.calls;
        let mut recorded = vec![!calls.is_empty(); n];
        for c in &mut calls {
            let k = c.top_k as usize;
            c.experts.resize(n * k, UNRECORDED);
            for (pos, row) in c.experts.chunks_exact(k).enumerate() {
                recorded[pos] &= row[0] != UNRECORDED;
            }
        }
        for c in &mut calls {
            let k = c.top_k as usize;
            for (pos, row) in c.experts.chunks_exact_mut(k).enumerate() {
                if !recorded[pos] {
                    row.fill(UNRECORDED);
                }
            }
        }
        RoutingRecord { policy_version, positions, recorded, calls }
    }
}

/// The routing records of one trainer batch of `batch` rows of `seq` positions: row `b`'s position `t` replays
/// position `start + t` of its record, when that position exists and was recorded; every other position routes itself.
#[derive(Clone, Debug, Default)]
pub struct RoutingReplay {
    pub batch: u32,
    pub seq: u32,
    /// Per row: its record and the record position the row's position 0 is.
    pub rows: Vec<Option<(std::sync::Arc<RoutingRecord>, u32)>>,
}

impl RoutingReplay {
    /// Check every row's record against the model's calls; the error names the row.
    pub fn validate(&self, calls: &[RouterCallShape]) -> Result<(), RoutingError> {
        if self.rows.len() != self.batch as usize {
            return Err(RoutingError::new("rows", format!("has {} entries for a batch of {}", self.rows.len(), self.batch)));
        }
        for (b, (record, _)) in self.rows.iter().enumerate().filter_map(|(b, r)| r.as_ref().map(|r| (b, r))) {
            record.validate(calls, None).map_err(|e| RoutingError::new(format!("of batch row {b}: {}", e.field), e.why))?;
        }
        Ok(())
    }

    /// The ids MoE call `call` (keeping `top_k`) is forced to, `[batch * seq, top_k]` row-major, [`UNRECORDED`] where
    /// the trainer routes itself. The layout [`crate::moe::router_fwd_replay`] reads.
    pub fn call_experts(&self, call: usize, top_k: u32) -> Vec<u32> {
        let (k, seq) = (top_k as usize, self.seq as usize);
        let mut out = vec![UNRECORDED; self.batch as usize * seq * k];
        for (b, row) in self.rows.iter().enumerate() {
            let Some((record, start)) = row else { continue };
            let c = &record.calls[call];
            assert_eq!(c.top_k, top_k, "RoutingReplay::call_experts: call {call} keeps {} in the record, {top_k} in the model (validate first)", c.top_k);
            for t in 0..seq {
                let pos = *start as usize + t;
                if pos < record.positions as usize && record.recorded[pos] {
                    out[(b * seq + t) * k..(b * seq + t + 1) * k].copy_from_slice(&c.experts[pos * k..(pos + 1) * k]);
                }
            }
        }
        out
    }

    /// Positions of the batch that replay a recorded routing.
    pub fn replayed_positions(&self) -> usize {
        self.rows
            .iter()
            .flatten()
            .map(|(record, start)| (0..self.seq as usize).filter(|t| record.recorded.get(*start as usize + t).copied().unwrap_or(false)).count())
            .sum()
    }

    /// Upload [`Self::call_experts`] for `call` into a fresh device buffer.
    pub fn upload_call(&self, g: &gpu_core::Gpu, call: usize, top_k: u32) -> gpu_core::DeviceBuffer {
        upload_experts(g, &self.call_experts(call, top_k))
    }
}

/// A `[rows, top_k]` u32 expert-id buffer on the device, the bytes uploaded exactly (an [`UNRECORDED`] id is a NaN
/// pattern as f32, so it never travels through a float upload).
pub fn upload_experts(g: &gpu_core::Gpu, ids: &[u32]) -> gpu_core::DeviceBuffer {
    let buf = g.storage_uninit(ids.len().max(1) as u64);
    let bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    g.write_bytes_at(&buf, 0, &bytes);
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shapes() -> Vec<RouterCallShape> {
        vec![RouterCallShape { n_experts: 4, top_k: 2 }; 2]
    }

    fn logged() -> RoutingRecord {
        let mut log = RoutingLog::new(&shapes());
        for pos in 1..4u32 {
            log.record(0, pos, &[pos % 4, (pos + 1) % 4]);
            log.record(1, pos, &[3, 0]);
        }
        log.record(1, 0, &[1, 2]);
        log.record(0, 9, &[0, 1]); // a rejected speculative position past the end
        log.into_record(4, 7)
    }

    #[test]
    fn a_log_becomes_a_record_masking_positions_a_call_missed() {
        let r = logged();
        assert_eq!(r.recorded, vec![false, true, true, true], "position 0 was written by call 1 only");
        assert_eq!(r.calls[1].experts[..2], [UNRECORDED, UNRECORDED], "an unrecorded position carries no ids");
        assert_eq!(r.calls[0].experts[2..4], [1, 2]);
        assert_eq!(r.validate(&shapes(), Some(7)), Ok(()));
    }

    #[test]
    fn validation_names_the_field_that_does_not_fit() {
        let field = |r: &RoutingRecord, v: Option<u64>| r.validate(&shapes(), v).unwrap_err().field;
        assert_eq!(field(&logged(), Some(8)), "policy_version");
        let mut r = logged();
        r.calls[0].experts[2] = 4;
        assert_eq!(field(&r, None), "calls[0].experts[2]");
        let mut r = logged();
        r.calls[1].experts[3] = r.calls[1].experts[2];
        assert_eq!(field(&r, None), "calls[1].experts[3]");
        let mut r = logged();
        r.calls.pop();
        assert_eq!(field(&r, None), "calls");
        let mut r = logged();
        r.calls[0].experts.pop();
        assert_eq!(field(&r, None), "calls[0].experts");
        let mut r = logged();
        r.recorded[0] = true;
        r.calls[0].experts[0] = UNRECORDED;
        assert_eq!(field(&r, None), "calls[0].experts[0]");
        let mut r = logged();
        r.recorded.push(false);
        assert_eq!(field(&r, None), "recorded");
        let r = logged();
        assert_eq!(r.validate(&[RouterCallShape { n_experts: 4, top_k: 3 }; 2], None).unwrap_err().field, "calls[0].top_k");
    }

    #[test]
    fn a_record_from_outside_with_an_unknown_field_is_refused() {
        let mut v = serde_json::to_value(logged()).unwrap();
        assert_eq!(serde_json::from_value::<RoutingRecord>(v.clone()).unwrap(), logged());
        v["layer"] = serde_json::json!(3);
        assert!(serde_json::from_value::<RoutingRecord>(v).unwrap_err().to_string().contains("layer"));
    }

    #[test]
    fn a_batch_lays_each_row_out_from_its_start() {
        let r = std::sync::Arc::new(logged());
        let replay = RoutingReplay { batch: 2, seq: 3, rows: vec![Some((r.clone(), 1)), None] };
        assert_eq!(replay.validate(&shapes()), Ok(()));
        let ids = replay.call_experts(0, 2);
        // row 0 positions 0..3 are record positions 1..4 (all recorded); row 1 routes itself
        assert_eq!(ids[..6], r.calls[0].experts[2..8]);
        assert!(ids[6..].iter().all(|&e| e == UNRECORDED));
        assert_eq!(replay.replayed_positions(), 3);
        let shifted = RoutingReplay { batch: 1, seq: 3, rows: vec![Some((r, 2))] };
        assert_eq!(shifted.replayed_positions(), 2, "record position 4 does not exist");
    }
}
