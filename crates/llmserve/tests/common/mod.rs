// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for low-latency multi-tenant LLM
// serving for its clients. If your team needs expertise in serving that keeps
// every running conversation generating while new inputs are encoded, then you
// can procure our services by sending an email to info@swedishembedded.com.

//! A toy decoder and tokenizer the serving specs share.

#![allow(dead_code)] // each spec file uses a part

use data::qwen_tokenizer::QwenBpe;
use model::paged::{BlockAllocator, BlockTable};
use model::serve::{AdapterAvailability, AdapterId, PagedDecoder, PrefillProgress, PromptEmbeds};

pub const VOCAB: usize = 64;
pub const EMBED_WIDTH: usize = 4;

/// A decoder whose next token is the previous one plus one. Its prefix cache "holds" the first `cached` tokens of
/// every prompt.
pub struct Toy {
    alloc: BlockAllocator,
    cached: u32,
    scoring: bool,
    /// Scoring answers "not ready" this many times before it answers "ready" (a prompt being read in).
    busy_polls: std::cell::Cell<u32>,
    /// Offer every token as a candidate ([`Toy::sampling`]); greedy only otherwise.
    sampling: bool,
}

impl Toy {
    pub fn new(alloc: BlockAllocator, cached: u32) -> Toy {
        Toy { alloc, cached, scoring: false, busy_polls: std::cell::Cell::new(0), sampling: false }
    }

    /// The same decoder, able to sample: after token `p` its row ranks every token, `p + 1` first, then `p + 2`, ...,
    /// with logits falling by [`ROW_STEP`] a rank ([`row`]), and its candidate list holds the whole row, which it also reads back whole.
    pub fn sampling(mut self) -> Toy {
        self.sampling = true;
        self
    }

    /// The same decoder, able to score a continuation: the log-probability of token `t` is `-t / 100`, whatever
    /// precedes it, so a test reads which tokens were scored from the numbers.
    pub fn scoring(mut self) -> Toy {
        self.scoring = true;
        self
    }

    /// The same, refusing to be ready for the first `polls` times it is asked.
    pub fn scoring_after(self, polls: u32) -> Toy {
        self.busy_polls.set(polls);
        self.scoring()
    }
}

impl PagedDecoder for Toy {
    fn scores_continuations(&self) -> bool {
        self.scoring
    }
    fn scoring_ready(&self) -> bool {
        let left = self.busy_polls.get();
        self.busy_polls.set(left.saturating_sub(1));
        left == 0
    }
    fn score_continuation(&mut self, _namespace: model::paged::KvNamespace, tokens: &[u32], from: usize, _temperature: f32) -> Result<Vec<f32>, String> {
        if !self.scoring {
            return Err("the toy cannot score".to_string());
        }
        Ok(tokens[from..].iter().map(|&t| -(t as f32) / 100.0).collect())
    }
    fn alloc_mut(&mut self) -> &mut BlockAllocator {
        &mut self.alloc
    }
    fn max_prefill_tokens(&self) -> u32 {
        64
    }
    fn free_blocks(&self) -> u32 {
        self.alloc.free_blocks()
    }
    fn max_seq_len(&self) -> usize {
        1024
    }
    fn vocab(&self) -> usize {
        VOCAB
    }
    fn blocks_for(&self, tokens: u32) -> u32 {
        tokens.div_ceil(8)
    }
    fn reclaim_prefix(&mut self, _want: u32) -> u32 {
        0
    }
    fn release_table(&mut self, t: &mut BlockTable) {
        t.release(&mut self.alloc);
    }
    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        table.reserve(prompt.len() as u32, &mut self.alloc).unwrap();
        table.mark_cached(self.cached.min(prompt.len() as u32));
        vec![*prompt.last().unwrap() as f32]
    }
    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        if self.sampling {
            let mut l = vec![0.0; VOCAB];
            for (token, logit) in row(hidden[0] as u32) {
                l[token as usize] = logit;
            }
            return l;
        }
        let mut l = vec![0.0; VOCAB];
        l[(hidden[0] as usize + 1) % VOCAB] = 1.0;
        l
    }
    /// Rows the toy accepts and ignores; with the adapter answer below, it lets a spec drive the admission of
    /// adapter requests that carry rows without a model.
    fn embed_width(&self) -> Option<usize> {
        Some(EMBED_WIDTH)
    }
    fn prefill_chunk_embeds(&mut self, table: &mut BlockTable, prompt: &[u32], _embeds: &PromptEmbeds, _budget: usize) -> PrefillProgress {
        PrefillProgress::Done(self.prefill(table, prompt))
    }
    fn adapter_availability(&mut self, _adapter: AdapterId) -> AdapterAvailability {
        AdapterAvailability::Resident
    }
    fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        for t in tables.iter_mut() {
            t.append(&mut self.alloc).unwrap();
        }
        inputs.iter().map(|&i| (i + 1) % VOCAB as u32).collect()
    }
    fn forward_batched_greedy_window(&mut self, _: &mut [&mut BlockTable], _: &[u32], _: usize) -> Vec<Vec<u32>> {
        unreachable!("window capacity is 1")
    }
    fn prefix_stats(&self) -> (u64, u64, usize) {
        (0, 0, 0)
    }
    fn device_stats(&self) -> Option<gpu_core::DeviceStats> {
        None
    }
    fn kv_pool_bytes(&self) -> u64 {
        0
    }
    fn kv_pool_capacity_tokens(&self) -> u64 {
        4096
    }
    fn decode_window_capacity(&self) -> usize {
        1
    }
    fn forward_batched_topk(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<(u32, f32)>> {
        assert!(self.sampling, "greedy only");
        for t in tables.iter_mut() {
            t.append(&mut self.alloc).unwrap();
        }
        inputs.iter().map(|&i| row(i).into_iter().take(k).collect()).collect()
    }
    fn full_row_readback(&self) -> bool {
        self.sampling
    }
    fn forward_batched_logit_rows(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<Vec<f32>> {
        for t in tables.iter_mut() {
            t.append(&mut self.alloc).unwrap();
        }
        inputs.iter().map(|&i| self.logits(&[i as f32])).collect()
    }
    fn topk_capacity(&self) -> usize {
        if self.sampling {
            VOCAB
        } else {
            1
        }
    }
}

/// How far a [`Toy::sampling`] row's logits fall from one rank to the next.
pub const ROW_STEP: f32 = 0.25;

/// The sampling toy's row after token `previous`, best first: `previous + 1`, `previous + 2`, ... (mod the vocabulary).
pub fn row(previous: u32) -> Vec<(u32, f32)> {
    (0..VOCAB as u32).map(|rank| ((previous + 1 + rank) % VOCAB as u32, -(rank as f32) * ROW_STEP)).collect()
}

pub fn tokenizer() -> QwenBpe {
    let gt = checkpoint::gguf::GgufTokenizer {
        model: "gpt2".into(),
        pre: Some("deepseek-v3".into()),
        tokens: (0..VOCAB).map(|i| format!("t{i}")).collect(),
        merges: Vec::new(),
        token_types: vec![1; VOCAB],
        bos: Some(0),
        eos: Some(1),
        unk: None,
        pad: None,
        ..Default::default()
    };
    QwenBpe::from_gguf(&gt).expect("toy tokenizer")
}

