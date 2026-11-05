// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! An encoder too large for one card runs as consecutive layer stages, each
//! handing its output stream to the next. Split anywhere, it must encode
//! exactly what the whole model encodes at the same depth.

use gpu_core::select::Dtype;
use model::Shard;
use qwen3::{Qwen, QwenConfig};

#[test]
fn a_split_encode_is_the_whole_models_encode() {
    let cfg = QwenConfig::tiny();
    let weights = qwen3::init::init_weights(&cfg, data::rng::random_seed());
    let n = cfg.n_layers as usize;
    let cut = n / 2;
    let ids: Vec<u32> = (0..11u32).map(|i| (i * 7 + 3) % cfg.vocab).collect();
    let shard = |start: usize, end: usize| Shard { start, end, embed: start == 0, head: false, gpu_index: Shard::ANY_GPU };

    let whole = Qwen::new_shard_dt(cfg.clone(), 1, 16, &weights, shard(0, n), Dtype::F32);
    let want = whole.encode_hidden(&ids, n);

    let first = Qwen::new_shard_dt(cfg.clone(), 1, 16, &weights, shard(0, cut), Dtype::F32);
    let second = Qwen::new_shard_dt(cfg.clone(), 1, 16, &weights, shard(cut, n), Dtype::F32);
    let boundary = first.encode_stage(&ids, None);
    let got = second.encode_stage(&ids, Some(&boundary));

    assert_eq!(boundary.len(), ids.len() * cfg.d_model as usize, "only the prompt's rows cross the cut");
    assert_eq!(got, want);
}
