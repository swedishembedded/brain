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
use model::serve::PagedDecoder;

pub const VOCAB: usize = 64;

/// A decoder whose next token is the previous one plus one. Its prefix cache "holds" the first `cached` tokens of
/// every prompt.
pub struct Toy {
    alloc: BlockAllocator,
    cached: u32,
}

impl Toy {
    pub fn new(alloc: BlockAllocator, cached: u32) -> Toy {
        Toy { alloc, cached }
    }
}

impl PagedDecoder for Toy {
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
        let mut l = vec![0.0; VOCAB];
        l[(hidden[0] as usize + 1) % VOCAB] = 1.0;
        l
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
    fn forward_batched_topk(&mut self, _: &mut [&mut BlockTable], _: &[u32], _: usize) -> Vec<Vec<(u32, f32)>> {
        unreachable!("greedy only")
    }
    fn topk_capacity(&self) -> usize {
        1
    }
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

