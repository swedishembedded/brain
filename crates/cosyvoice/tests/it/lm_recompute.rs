// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of speech synthesis models for its clients. If your team
// needs expertise in trading activation memory for recomputation in a TTS language model, you can procure our services
// by sending an email to info@swedishembedded.com.

//! The CosyVoice speech-token LM's host trainer holds its layer stack as a `model::saved::BlockStack`: recomputing
//! keeps only every layer's input and gives the loss and gradients of keeping every layer's cache, bit for bit
//! (`model::saved::check_block_recompute`).

use cosyvoice::lmgrad::{backward, forward_saved, grad_views, init_weights, loss, Example, LmDims};
use data::rng::{random_seed, Lcg};
use model::saved::{check_block_recompute, BlockStep};

#[test]
fn the_speech_lm_recomputes_its_layers_exactly() {
    let dims = LmDims::tiny();
    let mut rng = Lcg::new(random_seed());
    let w = init_weights::<f64>(&dims, random_seed());
    let special_task = if dims.special_vocab > 0 { dims.special_vocab - 1 } else { dims.speech_vocab - 1 };
    let ex = Example {
        text_ids: (0..4).map(|_| (rng.next_u32() as usize) % dims.text_vocab).collect(),
        special_sos: 0,
        special_task,
        speech_tokens: (0..5).map(|_| (rng.next_u32() as usize) % dims.speech_vocab).collect(),
    };
    check_block_recompute("cosyvoice", |policy| {
        let cache = forward_saved(&dims, &w, &ex, policy)?;
        let (l, dlogits) = loss(&dims, &cache, &ex.speech_tokens);
        let g = backward(&dims, &w, &cache, &dlogits);
        let grads = grad_views(&g).into_iter().map(|(n, v)| (n, v.iter().map(|x| x.to_bits()).collect())).collect();
        Ok(BlockStep { loss: l.to_bits(), grads, held_caches: cache.held_layer_caches() })
    })
    .unwrap();
}
