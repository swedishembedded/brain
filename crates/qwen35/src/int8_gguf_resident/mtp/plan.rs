// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements speculative decoding - drafters, verifiers
// and the state rollback a recurrent stack needs - for clients serving large
// models on modest hardware. If your team needs expertise in drafting with a
// model's own multi-token-prediction head without a host round trip per
// proposal, then you can procure our services by sending an email to
// info@swedishembedded.com.

//! **The MTP drafter as one recorded graph on the card.**
//!
//! A draft round feeds the head the target's hidden states it has not seen (the *catch-up*: every position the target
//! consumed since the last draft, paired with the token that followed it) and then chains proposals: each proposal is
//! embedded and paired with the head's own output for the next one. Run from the host that is four device round trips
//! per proposal (embedding rows and norms on the host, the projection read back, the layer read back, the head's norm
//! and logits uploaded again). Here the round is a single tape per [`DraftKey`]:
//!
//! ```text
//! partner tokens --embed--> rms(enorm) --+
//!                                        +--copy--> [n*C, 2d] --eh_proj--> layer --gather last row--> head --argmax--> id_1
//! pending rows --gather--> rms(norm) --rms(hnorm)--+                                       |
//!                                                                  id_1, head output --pair--> eh_proj --> layer --> head --> id_2 ...
//! ```
//!
//! The pending rows are the slot's span of [`MtpDrafter::pend`], which a recorded target step writes itself, so the
//! round starts from the card. The host writes the small per-round inputs (partner tokens, row indices, the layer's
//! positions and block tables), launches the tape and reads the `K * n` proposals back: one readback per round.
//!
//! **Fixed shapes.** A sequence's catch-up is `C` rows for every sequence: its pending run, padded by repeating the last
//! row (same position, same pair). A padded row computes and writes exactly what that row does, so it changes nothing and
//! reaches no position the sequence's blocks do not hold.

use super::*;
use crate::decode_tape::DecodeInputs;
use crate::model::rms_step;
use gpu_core::tape::Tape;

/// Layer invocations a recording holds: the catch-up and the chain steps after the first proposal.
const MAX_INVOCATIONS: usize = model::gdn_verify::MAX_VERIFY_ROWS as usize;

/// Rows the catch-up is padded to a multiple of, so that the windows the pending runs vary over share a recording.
const CATCH_UP_QUANTUM: usize = 4;

/// Draft recordings kept: a serving batch moves between few shapes, slowly between context buckets.
const MAX_DRAFT_PLANS: usize = 8;

/// Everything a recorded round's dispatch list depends on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct DraftKey {
    /// Sequences drafting.
    n: u32,
    /// Catch-up rows each.
    catch_up: u32,
    /// Proposals each (the first from the catch-up, the rest chained).
    chain: u32,
    /// Attention capacity of each layer invocation ([`Qwen35::paged_step_shape`]).
    caps: [u32; MAX_INVOCATIONS],
}

/// One recorded round: its tape, the buffers the host rewrites before a replay, and the buffer it answers in.
struct DraftPlan {
    key: DraftKey,
    tape: Option<Tape>,
    /// `[n*C]` the token each catch-up row is paired with.
    partner: DeviceBuffer,
    /// `[n*C]` the pending-store row each catch-up row reads.
    pending: DeviceBuffer,
    /// `[n]` the catch-up row whose output proposes the first token (a sequence's last real row).
    last: DeviceBuffer,
    /// One per layer invocation.
    layer_inputs: Vec<DecodeInputs>,
    /// `[K, n]` proposals, step after step.
    ids: DeviceBuffer,
}

/// The recorded rounds, most recently used last.
#[derive(Default)]
pub(super) struct DraftPlans {
    plans: Vec<DraftPlan>,
}

/// The rows of one layer invocation, as the host knows them.
struct Rows {
    slots: Vec<u32>,
    positions: Vec<u32>,
}

/// What one round drafts for: the live sequences' slots, how far the head is fed and the target consumed, and the
/// partner token of each catch-up row.
struct Round {
    n: usize,
    catch_up: usize,
    chain: usize,
    slots: Vec<u32>,
    /// `[n*C]`
    partner: Vec<u32>,
    pending: Vec<u32>,
    /// `[n]`
    last: Vec<u32>,
    /// Layer invocation 0 (the catch-up) then one per chained proposal after the first.
    invocations: Vec<Rows>,
}

impl MtpDrafter {
    /// `copy_cols` of `cols` columns of `src` (row stride `src_stride`) into `dst` (row stride `dst_stride`) at `dst_off`.
    pub(super) fn copy_cols_step(g: &Gpu, src: &DeviceBuffer, dst: &DeviceBuffer, rows: u32, cols: u32, src_stride: u32, src_off: u32, dst_stride: u32, dst_off: u32) -> gpu_core::Step {
        let kernel = g.kernel_index("copy_cols").expect("copy_cols is registered with the qwen35 pipelines");
        g.step(kernel, &[src, dst], &[rows, cols, src_stride, src_off, dst_stride, dst_off], rows * cols)
    }

    /// `out[i] = table[ids[i]]` for `rows` rows of `d` words: a row gather by device-side index.
    pub(super) fn gather_rows_step(g: &Gpu, ids: &DeviceBuffer, table: &DeviceBuffer, out: &DeviceBuffer, rows: u32, d: u32) -> gpu_core::Step {
        let kernel = g.kernel_index("embed").expect("embed is registered with the qwen35 pipelines");
        g.step(kernel, &[ids, table, out], &[d, rows], rows * d)
    }
}

impl Qwen35GgufInstance {
    /// The head's input projection of `rows` pairs on `g` (this head's card handle): `eh_proj([rms(embed(token), enorm),
    /// rms(rms(hidden, hidden_norm), hnorm)])`, where `hidden_norm` is the norm the hidden state is handed through first
    /// (the target's final norm; none for the head's own output, which is already normed or kept raw).
    fn issue_pairs(&self, g: &Gpu, tokens: &DeviceBuffer, hidden: &DeviceBuffer, hidden_norm: Option<&DeviceBuffer>, rows: u32) -> DeviceBuffer {
        let m = self.mtp.as_ref().expect("a draft round has a head");
        let d = self.cfg.d_model;
        let table = self.shards[0].qwen35.embed_table().expect("device_drafter_supported");
        let embedded = g.storage(u64::from(rows * d));
        let mut s = Vec::new();
        m.stage.ops().embed(&mut s, table, tokens, rows, &embedded);
        g.submit(&[], &s);
        let pair = g.storage(u64::from(rows * 2 * d));
        let e_normed = g.storage(u64::from(rows * d));
        g.submit(&[], &[rms_step(g, &embedded, &m.enorm_dev, &e_normed, d, rows)]);
        g.submit(&[], &[MtpDrafter::copy_cols_step(g, &e_normed, &pair, rows, d, d, 0, 2 * d, 0)]);
        let h_in = match hidden_norm {
            Some(norm) => {
                let normed = g.storage(u64::from(rows * d));
                g.submit(&[], &[rms_step(g, hidden, norm, &normed, d, rows)]);
                normed
            }
            None => hidden.clone(),
        };
        let h_normed = g.storage(u64::from(rows * d));
        g.submit(&[], &[rms_step(g, &h_in, &m.hnorm_dev, &h_normed, d, rows)]);
        g.submit(&[], &[MtpDrafter::copy_cols_step(g, &h_normed, &pair, rows, d, d, 0, 2 * d, d)]);
        let proj = g.storage(u64::from(rows * d));
        let mut s = Vec::new();
        let act = m.stage.ops().act(&mut s, &pair, 0, rows, 2 * d);
        m.stage.ops().matmul(&mut s, &m.fc, &act, &proj, 0);
        g.submit(&[], &s);
        proj
    }

    /// Feed the head the pairs `(hidden[i], tokens[i])` (`hidden`: `[n, d]` pre-norm target rows on the host) at head
    /// positions `from..from + n` of `slot`, a chunk at a time: the prompt's rows, or a slot's pending run that grew too
    /// long. Each chunk is uploaded, normed, projected and run through the head's layer on the card.
    pub(super) fn mtp_feed_run(&self, slot: u32, from: u32, tokens: &[u32], hidden: &[f32]) -> Result<(), String> {
        let m = self.mtp.as_ref().expect("a feed has a head");
        let g = &m.stage.gpu;
        let d = self.cfg.d_model as usize;
        if let Some(&bad) = tokens.iter().find(|&&t| t >= self.cfg.vocab) {
            return Err(format!("{MODEL}: token id {bad} is outside the vocabulary"));
        }
        let view = m.caches.view(slot);
        for (i, chunk) in tokens.chunks(FEED_CHUNK_ROWS).enumerate() {
            let at = i * FEED_CHUNK_ROWS;
            let rows = chunk.len() as u32;
            let ids = g.storage(u64::from(rows));
            g.write(&ids, chunk);
            let h = g.storage((chunk.len() * d) as u64);
            g.write_f32(&h, &hidden[at * d..(at + chunk.len()) * d]);
            let proj = self.issue_pairs(g, &ids, &h, m.normed_hidden.then_some(&m.target_norm_dev), rows);
            m.stage.run_prefill_chunk_stage(chunk, from + at as u32, &view, Some(StageRows::device(g, &proj, chunk.len() * d)));
        }
        Ok(())
    }
}

impl Qwen35GgufInstance {
    /// Whether the device drafter can run here: a recording is possible and the embedding table is on the card the
    /// head's layer and the target share.
    pub(in super::super) fn device_drafter_supported(&self) -> bool {
        self.shards.len() == 1 && !self.exact_decode && self.embeds_on_device() && self.shards[0].qwen35.embed_table().is_some()
    }

    /// Everything a draft round needs from the host's view of the sequences, or `None` when no sequence can draft.
    fn plan_round(&mut self, slots: &[u32], consumed: &[u32], next_inputs: &[u32], ks: &[usize]) -> Option<Round> {
        let Self { mtp, .. } = self;
        let m = mtp.as_mut()?;
        let n = slots.len();
        // Sequences that can draft: their pending run covers exactly the positions the target consumed. One whose does not
        // (it was not tracked while speculation was off for a big batch) starts over from here: it drafts again once a
        // step has given the head a fresh row to continue from.
        for i in (0..n).filter(|&i| !m.feeding.contains(&slots[i])) {
            let in_step = m.slots.get(&slots[i]).is_some_and(|st| st.fed + st.rows() as u32 == consumed[i]);
            if !in_step {
                m.slots.insert(slots[i], SlotState { fed: consumed[i], ..SlotState::default() });
            }
        }
        let live: Vec<usize> = (0..n).filter(|&i| ks[i] > 0 && !m.feeding.contains(&slots[i]) && m.slots.get(&slots[i]).is_some_and(|st| st.rows() > 0)).collect();
        if live.is_empty() {
            return None;
        }
        let rows_max = live.iter().map(|&i| m.slots[&slots[i]].rows()).max().expect("a live sequence");
        let catch_up = rows_max.next_multiple_of(CATCH_UP_QUANTUM);
        let chain = live.iter().map(|&i| ks[i]).max().expect("a live sequence").min(MAX_INVOCATIONS);
        let mut round = Round { n: live.len(), catch_up, chain, slots: Vec::new(), partner: Vec::new(), pending: Vec::new(), last: Vec::new(), invocations: Vec::new() };
        let mut cu = Rows { slots: Vec::new(), positions: Vec::new() };
        for (q, &i) in live.iter().enumerate() {
            let st = &m.slots[&slots[i]];
            let r = st.rows();
            round.slots.push(slots[i]);
            for j in 0..catch_up {
                // A padded row repeats the last real row, position and pair included, so it writes the very values that row
                // does and reaches no position the sequence's blocks do not hold.
                let real = j.min(r - 1);
                cu.slots.push(slots[i]);
                cu.positions.push(st.fed + real as u32);
                round.partner.push(if real + 1 < r { st.toks[real + 1] } else { next_inputs[i] });
                round.pending.push(m.pending_row(slots[i], real));
            }
            round.last.push((q * catch_up + r - 1) as u32);
        }
        round.invocations.push(cu);
        // A chained step runs every sequence, one that wants fewer proposals than the round's widest included. Its extra steps
        // are scratch: they stay on the last position the sequence asked for and inside the blocks it holds, and the host
        // drops their proposals.
        for t in 0..chain.saturating_sub(1) {
            let positions = live
                .iter()
                .map(|&i| {
                    let room = m.caches.window(slots[i]).capacity(m.caches.kv_block()).saturating_sub(1);
                    (consumed[i] + t.min(ks[i].saturating_sub(2)) as u32).min(room)
                })
                .collect();
            round.invocations.push(Rows { slots: round.slots.clone(), positions });
        }
        Some(round)
    }

    /// The dispatches of one round, issued on the head layer's card in order: submitted straight away when no tape is
    /// open, recorded when one is. `inputs` are the buffers the round reads its per-round data from and `ids` where the
    /// proposals go.
    fn issue_round(&self, round: &Round, partner: &DeviceBuffer, pending: &DeviceBuffer, last: &DeviceBuffer, layer_inputs: &[DecodeInputs], ids: &DeviceBuffer) {
        let m = self.mtp.as_ref().expect("a draft round has a head");
        let g = &m.stage.gpu;
        let (n, c, d) = (round.n as u32, round.catch_up as u32, self.cfg.d_model);
        let mut chain_ids: Option<DeviceBuffer> = None;
        let mut chain_hidden: Option<DeviceBuffer> = None;
        for (step, inputs) in layer_inputs.iter().enumerate().take(round.chain) {
            let rows = if step == 0 { n * c } else { n };
            let tokens = match (&chain_ids, step) {
                (Some(prev), 1..) => prev,
                _ => partner,
            };
            let (hidden, hidden_norm) = if step == 0 {
                let gathered = g.storage(u64::from(rows * d));
                g.submit(&[], &[MtpDrafter::gather_rows_step(g, pending, &m.pend, &gathered, rows, d)]);
                (gathered, m.normed_hidden.then_some(&m.target_norm_dev))
            } else {
                (chain_hidden.clone().expect("a chained step follows a step"), None)
            };
            let proj = self.issue_pairs(g, tokens, &hidden, hidden_norm, rows);
            let seqs = m.caches.batch(&round.invocations[step].slots, &round.invocations[step].positions);
            let caches = BatchDecodeCaches { gqa_kv: &m.caches.gqa_kv, gqa_cap: m.caches.kv_block(), seqs: &seqs, one_sequence: false, gdn_pool: None, memory: &[] };
            let block = m.stage.record_layers_over(&caches, inputs, &proj);
            // The row that proposes: a sequence's last real catch-up row, or the only row of a chained step.
            let out = if step == 0 {
                let picked = g.storage(u64::from(n * d));
                g.submit(&[], &[MtpDrafter::gather_rows_step(g, last, &block, &picked, n, d)]);
                picked
            } else {
                block
            };
            // The shared head over the head's own norm, then the pick.
            let normed = g.storage(u64::from(n * d));
            g.submit(&[], &[rms_step(g, &out, &m.head_norm, &normed, d, n)]);
            let logits = crate::stream::head_logits_issue(g, m.stage.ops(), &self.cfg, &m.head_norm, &self.head.w, &out, n);
            let picked = crate::model::argmax_rows_issue_of(g, &logits, n, self.cfg.vocab);
            g.submit(&[], &[MtpDrafter::copy_cols_step(g, &picked, ids, 1, n, n, 0, n, step as u32 * n)]);
            chain_ids = Some(picked);
            chain_hidden = Some(if m.normed_hidden { normed } else { out });
        }
    }

    /// **Propose up to `ks[i]` tokens after `next_inputs[i]`** for the sequence in `slots[i]`, which has consumed
    /// `consumed[i]` positions: one replayed tape and one readback.
    pub(in super::super) fn mtp_draft(&mut self, slots: &[u32], consumed: &[u32], next_inputs: &[u32], ks: &[usize]) -> Vec<Vec<u32>> {
        let mut out = vec![Vec::new(); slots.len()];
        let Some(round) = self.plan_round(slots, consumed, next_inputs, ks) else { return out };
        let started = std::time::Instant::now();
        let ids = self.run_round(&round);
        // Proposal `step` of sequence `q` is `ids[step * n + q]`.
        let live: Vec<usize> = (0..slots.len()).filter(|i| round.slots.contains(&slots[*i]) && ks[*i] > 0).collect();
        let Self { mtp, .. } = self;
        let m = mtp.as_mut().expect("a draft round has a head");
        for (q, slot) in round.slots.iter().enumerate() {
            let i = *live.iter().find(|&&i| slots[i] == *slot).expect("a live slot");
            out[i] = (0..round.chain.min(ks[i])).map(|step| ids[step * round.n + q]).collect();
            let st = m.slots.get_mut(slot).expect("live slot");
            st.fed = consumed[i];
            st.toks.clear();
        }
        m.forwards += 1;
        m.times[0] += started.elapsed();
        out
    }

    /// Run `round` (replaying its recording, made on first use) and read its proposals.
    fn run_round(&self, round: &Round) -> Vec<u32> {
        let m = self.mtp.as_ref().expect("a draft round has a head");
        let g = &m.stage.gpu;
        let (n, c) = (round.n as u32, round.catch_up as u32);
        let mut caps = [0u32; MAX_INVOCATIONS];
        let mut shapes = Vec::with_capacity(round.invocations.len());
        for (i, inv) in round.invocations.iter().enumerate() {
            let seqs = m.caches.batch(&inv.slots, &inv.positions);
            let caches = BatchDecodeCaches { gqa_kv: &m.caches.gqa_kv, gqa_cap: m.caches.kv_block(), seqs: &seqs, one_sequence: false, gdn_pool: None, memory: &[] };
            let (cap, max_bt) = m.stage.paged_step_shape(&caches);
            caps[i] = cap;
            shapes.push(max_bt);
        }
        let key = DraftKey { n, catch_up: c, chain: round.chain as u32, caps };
        let write = |plan: &DraftPlan| {
            g.write(&plan.partner, &round.partner);
            g.write(&plan.pending, &round.pending);
            g.write(&plan.last, &round.last);
            for (inv, inputs) in round.invocations.iter().zip(&plan.layer_inputs) {
                let seqs = m.caches.batch(&inv.slots, &inv.positions);
                let caches = BatchDecodeCaches { gqa_kv: &m.caches.gqa_kv, gqa_cap: m.caches.kv_block(), seqs: &seqs, one_sequence: false, gdn_pool: None, memory: &[] };
                m.stage.write_decode_inputs(inputs, None, &caches);
            }
        };
        let fresh = || DraftPlan {
            key,
            tape: None,
            partner: g.storage(u64::from(n * c)),
            pending: g.storage(u64::from(n * c)),
            last: g.storage(u64::from(n)),
            layer_inputs: round.invocations.iter().zip(&shapes).map(|(inv, &bt)| m.stage.alloc_decode_inputs_at(inv.slots.len() as u32, bt)).collect(),
            ids: g.storage(u64::from(round.chain as u32 * n)),
        };
        let recorded = self.plans_on.get().then(|| m.draft_plans.borrow_mut().take(&key)).flatten();
        let plan = match recorded {
            Some(plan) => plan,
            None => {
                let mut plan = fresh();
                if self.plans_on.get() {
                    g.begin_tape();
                    self.issue_round(round, &plan.partner, &plan.pending, &plan.last, &plan.layer_inputs, &plan.ids);
                    plan.tape = Some(g.end_tape());
                }
                plan
            }
        };
        write(&plan);
        match &plan.tape {
            Some(tape) => g.replay_tape(tape),
            None => self.issue_round(round, &plan.partner, &plan.pending, &plan.last, &plan.layer_inputs, &plan.ids),
        }
        let ids = g.read(&plan.ids, (round.chain as u32 * n) as usize).into_iter().map(|x| x as u32).collect();
        if plan.tape.is_some() {
            m.draft_plans.borrow_mut().put(plan);
        }
        ids
    }
}

impl DraftPlans {
    fn take(&mut self, key: &DraftKey) -> Option<DraftPlan> {
        let at = self.plans.iter().position(|p| p.key == *key)?;
        Some(self.plans.remove(at))
    }

    fn put(&mut self, plan: DraftPlan) {
        if self.plans.len() >= MAX_DRAFT_PLANS {
            self.plans.remove(0);
        }
        self.plans.push(plan);
    }

    pub(super) fn clear(&mut self) {
        self.plans.clear();
    }
}
