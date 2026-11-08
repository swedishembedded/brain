// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The serving drafter: one slot of draft cache per sequence, fed from the target's retained hidden states, proposing
//! for many sequences in one batched forward.
//!
//! Swedish Embedded AB implements speculative decoding - drafters, verifiers and the state rollback a hybrid or
//! recurrent stack needs - for clients serving large models on modest hardware. If your team needs inference
//! throughput without a second model's worth of accuracy loss, you can procure our services by sending an email
//! to info@swedishembedded.com.
//!
//! # What a round does for a sequence
//!
//! The scheduler asks for up to `k` proposals after a sequence's context `c[0 .. L]`. The target has consumed
//! positions `0 .. L - 1` (so `table.len() == L - 1`); `c[L - 1]` is the *anchor*, the verified token its next step
//! consumes, whose own hidden state does not exist yet. The drafter
//!
//! 1. projects the taps of the positions it has not seen into its cache slot ([`Dflash2::append_context`]), which is
//!    all of them the first time and the few the last verify accepted afterwards;
//! 2. denoises one block `[anchor, MASK x (block - 1)]` at position `L - 1`, with the other sequences' blocks in the
//!    same forward;
//! 3. runs the shared head over the mask rows with the draft's own final norm and lets the selector walk its candidates.
//!
//! The block is always the draft's full `block_size`: its rows attend to each other, so a shorter block would
//! propose differently. Proposals beyond what the scheduler asked for are dropped.
//!
//! A sequence's cache slot is keyed by its request id. The cache is indexed by absolute position, so a verify that
//! rejects proposals needs no rollback: the next round's context rows overwrite exactly the rows the denoising block
//! wrote past the last accepted position. A sequence that is absent for a while keeps its slot until another needs it;
//! evicted, it is rebuilt from the taps when it returns.

use std::collections::HashMap;
use std::time::Instant;

use model::paged::BlockTable;

use crate::model::{BlockJob, Dflash2};

/// What the draft shares with its target: the token embedding in and the output head out.
pub trait DraftHost {
    /// The target's embedding of `tokens`, `[tokens, d_model]`, as the draft's noise input.
    fn embed(&self, tokens: &[u32]) -> Result<Vec<f32>, String>;

    /// The target head's logits (`[rows, vocab]`) for `rows` rows that already carry the final norm the caller chose
    /// (the draft's own).
    fn project(&self, normed: &[f32]) -> Result<Vec<f32>, String>;
}

/// The target's retained hidden states, per sequence ([`model::serve::PagedDecoder::read_taps`]).
pub trait TapRows {
    /// Rows of positions `from .. from + count` of `table`, `[count, tap_width]`.
    fn tap_rows(&self, table: &BlockTable, from: u32, count: u32) -> Result<Vec<f32>, String>;
}

/// One sequence to draft for.
pub struct DraftRequest<'a> {
    /// Stable for the life of the request; keys the cache slot.
    pub id: u64,
    /// The prompt followed by every token generated so far; the last is the anchor.
    pub context: &'a [u32],
    pub table: &'a BlockTable,
    /// How many proposals the scheduler wants.
    pub want: usize,
}

struct Seq {
    slot: u32,
    /// Positions `0 .. fed` are in the slot's cache.
    fed: u32,
    last_round: u64,
}

/// What the drafter cost, as opposed to what it bought.
#[derive(Clone, Copy, Debug, Default)]
pub struct DraftStats {
    pub rounds: u64,
    pub sequences: u64,
    /// Context rows projected into slots (the first round of a sequence pays its whole prompt).
    pub fed_rows: u64,
    pub proposed: u64,
    pub feed_s: f64,
    pub denoise_s: f64,
    pub head_s: f64,
    pub select_s: f64,
}

/// See the module doc.
pub struct Dflash2Drafter {
    model: Dflash2,
    seqs: HashMap<u64, Seq>,
    round: u64,
    stats: DraftStats,
}

impl Dflash2Drafter {
    pub fn new(model: Dflash2) -> Dflash2Drafter {
        Dflash2Drafter { model, seqs: HashMap::new(), round: 0, stats: DraftStats::default() }
    }

    pub fn model(&self) -> &Dflash2 {
        &self.model
    }

    pub fn stats(&self) -> DraftStats {
        self.stats
    }

    /// The most proposals one round can give: a block is `block_size` rows, of which the first is the anchor.
    pub fn max_draft(&self) -> usize {
        self.model.cfg.block_size as usize - 1
    }

    /// The target layers whose residual the draft reads, in row order.
    pub fn target_layers(&self) -> &[usize] {
        &self.model.cfg.target_layers
    }

    /// Forget a sequence (finished or cancelled), freeing its slot.
    pub fn forget(&mut self, id: u64) {
        if let Some(seq) = self.seqs.remove(&id) {
            self.model.release_slot(seq.slot);
        }
    }

    /// Sequences that hold a cache slot now.
    pub fn live(&self) -> usize {
        self.seqs.len()
    }

    /// The cache slot for `id`, evicting the least recently drafted sequence that is not in this round when none is
    /// free. `None` when every slot belongs to a sequence of this round.
    fn slot_for(&mut self, id: u64, in_round: &[u64]) -> Option<u32> {
        if let Some(s) = self.seqs.get(&id) {
            return Some(s.slot);
        }
        let used: Vec<u32> = self.seqs.values().map(|s| s.slot).collect();
        let slot = match (0..self.model.slots()).find(|s| !used.contains(s)) {
            Some(free) => free,
            None => {
                let victim = self.seqs.iter().filter(|(k, _)| !in_round.contains(k)).min_by_key(|(_, s)| s.last_round).map(|(k, _)| *k)?;
                let slot = self.seqs.remove(&victim)?.slot;
                self.model.release_slot(slot);
                slot
            }
        };
        self.seqs.insert(id, Seq { slot, fed: 0, last_round: self.round });
        Some(slot)
    }

    /// Proposals for each request, in order: up to `want` tokens (possibly none) continuing its context. A request
    /// the drafter cannot serve this round (no slot free, a context out of step with its table, taps that cannot be
    /// read) proposes nothing and says why in the log; the verify loop is correct for no proposals.
    pub fn propose(&mut self, host: &dyn DraftHost, taps: &dyn TapRows, requests: &[DraftRequest<'_>]) -> Vec<Vec<u32>> {
        match self.try_propose(host, taps, requests) {
            Ok(out) => out,
            Err(e) => {
                tracing::warn!("dflash2: drafting skipped this round: {e}");
                vec![Vec::new(); requests.len()]
            }
        }
    }

    fn try_propose(&mut self, host: &dyn DraftHost, taps: &dyn TapRows, requests: &[DraftRequest<'_>]) -> Result<Vec<Vec<u32>>, String> {
        let cfg = self.model.cfg.clone();
        let block = cfg.block_size as usize;
        let d = cfg.d_model as usize;
        self.round += 1;
        let in_round: Vec<u64> = requests.iter().map(|r| r.id).collect();
        let mut out = vec![Vec::new(); requests.len()];
        // Which requests ride the forward, with their slot and the noise rows `[anchor, MASK ...]`.
        let mut jobs: Vec<(usize, u32, u32, Vec<f32>)> = Vec::new();
        for (i, r) in requests.iter().enumerate() {
            if r.want == 0 || r.context.len() < 2 {
                continue;
            }
            let ctx_len = r.context.len() as u32 - 1;
            if r.table.len() != ctx_len {
                tracing::warn!(id = r.id, table = r.table.len(), context = r.context.len(), "dflash2: the sequence's cache and context disagree; not drafting for it");
                continue;
            }
            if ctx_len + cfg.block_size > self.model.cap() + cfg.block_size {
                tracing::warn!(id = r.id, ctx_len, cap = self.model.cap(), "dflash2: the sequence outgrew the draft's cache; not drafting for it");
                continue;
            }
            let Some(slot) = self.slot_for(r.id, &in_round) else {
                tracing::warn!(id = r.id, slots = self.model.slots(), "dflash2: more sequences than draft cache slots; not drafting for it");
                continue;
            };
            let t0 = Instant::now();
            let seq = self.seqs.get_mut(&r.id).expect("slot_for registered it");
            seq.last_round = self.round;
            if seq.fed > ctx_len {
                seq.fed = 0;
            }
            let from = seq.fed;
            let fed = (|| -> Result<(), String> {
                if ctx_len > from {
                    let rows = taps.tap_rows(r.table, from, ctx_len - from)?;
                    self.model.append_context(slot, &rows, from)?;
                }
                // The denoising block writes one block of rows past the context.
                self.model.reserve(slot, ctx_len + cfg.block_size)
            })();
            match fed {
                Ok(()) => {
                    self.stats.fed_rows += u64::from(ctx_len - from);
                    self.seqs.get_mut(&r.id).expect("registered above").fed = ctx_len;
                }
                Err(e) => {
                    // What it holds is incomplete: it starts over (and gives its blocks back) when it is next asked.
                    tracing::warn!(id = r.id, "dflash2: not drafting for a sequence this round: {e}");
                    self.forget(r.id);
                    continue;
                }
            }
            self.stats.feed_s += t0.elapsed().as_secs_f64();
            let mut ids = Vec::with_capacity(block);
            ids.push(*r.context.last().expect("a context of at least two tokens"));
            ids.extend(std::iter::repeat_n(cfg.mask_token_id, block - 1));
            jobs.push((i, slot, ctx_len, host.embed(&ids)?));
        }
        if jobs.is_empty() {
            return Ok(out);
        }
        self.stats.rounds += 1;
        self.stats.sequences += jobs.len() as u64;

        let t = Instant::now();
        let batch: Vec<BlockJob<'_>> = jobs.iter().map(|(_, slot, pos, noise)| BlockJob { slot: *slot, pos: *pos, noise }).collect();
        let hidden = self.model.denoise_blocks(&batch)?;
        self.stats.denoise_s += t.elapsed().as_secs_f64();

        // Row 0 of each block is the anchor's own denoised state and proposes nothing; the MASK rows are the proposals.
        let t = Instant::now();
        let mask_rows: Vec<&[f32]> = hidden.iter().map(|h| &h[d..]).collect();
        let normed: Vec<f32> = mask_rows.iter().flat_map(|rows| self.model.normed(rows)).collect();
        let logits = host.project(&normed)?;
        self.stats.head_s += t.elapsed().as_secs_f64();

        let t = Instant::now();
        let vocab = cfg.vocab as usize;
        for (j, (i, ..)) in jobs.iter().enumerate() {
            let rows = block - 1;
            let anchor = *requests[*i].context.last().expect("checked above");
            let mut path = self.model.select(mask_rows[j], &logits[j * rows * vocab..(j + 1) * rows * vocab], anchor)?;
            path.truncate(requests[*i].want);
            self.stats.proposed += path.len() as u64;
            out[*i] = path;
        }
        self.stats.select_s += t.elapsed().as_secs_f64();
        Ok(out)
    }
}
