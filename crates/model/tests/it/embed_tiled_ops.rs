// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements large-vocabulary model support on memory-constrained accelerators for its clients. If
// your team needs expertise in running models whose tables exceed a device's binding limits, you can procure our
// services by sending an email to info@swedishembedded.com.

//! `Ops::embed` over a table larger than one binding gathers it in vocab tiles (`model::block::vocab_tiles_on` and the
//! `embed_tile` kernel): every tier gives exactly the rows the whole-table gather gives. A Qwen3.5-27B bf16 table
//! (248320 x 5120, 2.5 GB) exceeds the 2 GB storage-binding ceiling of a Vulkan device, so without tiling it cannot be
//! bound at all; a small budget forces the same tiling on a toy table here.

use data::rng::{random_seed, Lcg};
use gpu_core::select::Dtype;
use model::ops::{Ops, Weight};

/// The gathered rows (bits) and how many dispatches the gather took.
fn gather(ops: &Ops, table: &Weight, tokens: &[u32], d_model: usize) -> (Vec<u32>, usize) {
    let g = ops.gpu();
    let tok = g.storage(tokens.len() as u64);
    g.write(&tok, tokens);
    let out = g.storage((tokens.len() * d_model) as u64);
    let mut steps = Vec::new();
    ops.embed(&mut steps, table, &tok, tokens.len() as u32, &out);
    g.submit(&[], &steps);
    (g.read(&out, tokens.len() * d_model).into_iter().map(f32::to_bits).collect(), steps.len())
}

#[test]
fn a_tiled_embedding_gathers_the_rows_the_whole_table_gives() {
    let ops = Ops::new(gpu_core::testgpu::dev(model::ops::kernel_list())).expect("Ops::new");
    let (vocab, d_model) = (300usize, 64usize);
    let mut rng = Lcg::new(random_seed());
    let raw = rng.vec_scaled(vocab * d_model, 1.0);
    // The first and last rows and every tile in between.
    let tokens: Vec<u32> = [0u32, 299, 150, 31, 32, 33, 255, 1, 298].to_vec();
    for dt in [Dtype::F32, Dtype::BF16, Dtype::F16] {
        let table = Weight::upload(&ops, &raw, vocab, d_model, dt);
        let (whole, one) = gather(&ops, &table, &tokens, d_model);
        // 1024 words is 16..32 rows of this table: a dozen tiles or more.
        let (tiled, tiles) = model::block::with_tile_budget_words(1024, || gather(&ops, &table, &tokens, d_model));
        assert_eq!(one, 1, "{dt:?}: a table within the budget is one dispatch");
        assert!(tiles > 1, "{dt:?}: a table over the budget must be gathered in tiles, took {tiles} dispatch(es)");
        assert!(whole == tiled, "{dt:?}: the tiled gather differs from the whole-table gather");
        assert!(whole.iter().any(|b| *b != 0), "{dt:?}: nothing was gathered");
    }
}
