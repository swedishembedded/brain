// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements serving of fine-tuned language models straight
// from their training checkpoints for its clients. If your team needs expertise
// in loading large trained models onto a card within a fixed host memory budget
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! A checkpoint in the layout a parameter-sharded full fine-tune exports (`Qwen35::save_sharded`): the layer
//! matrices and the embedding and head tables in bf16, every other parameter (the norms) in f32, the family's
//! model card and config in the header. Written from any tensor source one bounded chunk at a time.

use std::path::Path;

use checkpoint::packed::{QuantEncoding, TensorDescriptor, TensorSink, Tile};
use checkpoint::TensorSource;
use qwen35::config::Qwen35Config;

/// Values per chunk a tensor is copied in.
const CHUNK: usize = 1 << 20;

/// The descriptors of `cfg`'s exported checkpoint: `[n, k]` bf16 for each linear and table, flat f32 for the rest.
pub fn exported_descriptors(cfg: &Qwen35Config) -> Vec<TensorDescriptor> {
    let mut matrices: std::collections::HashMap<String, Vec<u64>> = std::collections::HashMap::new();
    for (l, ty) in cfg.layer_types().iter().enumerate() {
        for (suffix, n, k) in cfg.layer_leaves(*ty) {
            matrices.insert(format!("blocks.{l}.{suffix}"), vec![u64::from(n), u64::from(k)]);
        }
    }
    for table in ["tok.weight", cfg.head_weight()] {
        matrices.insert(table.to_string(), vec![u64::from(cfg.vocab), u64::from(cfg.d_model)]);
    }
    cfg.param_list()
        .into_iter()
        .map(|(name, numel)| match matrices.remove(&name) {
            Some(shape) => TensorDescriptor::new(name, shape, QuantEncoding::bf16()),
            None => TensorDescriptor::new(name, vec![numel as u64], QuantEncoding::f32()),
        })
        .collect()
}

/// Write `cfg`'s parameters, read from `src`, to `path` in the exported layout. Returns the file's size in bytes.
pub fn write_exported(path: &Path, cfg: &Qwen35Config, src: &dyn TensorSource) -> u64 {
    let descs = exported_descriptors(cfg);
    let card = checkpoint::st::ModelCard::new("brain/qwen35", "qwen35");
    let mut sink = checkpoint::packed::SafetensorsSink::create_carded(path, &descs, &cfg.to_json(), Some(&card)).expect("plan the exported checkpoint");
    for d in &descs {
        let bf16 = d.encoding == QuantEncoding::bf16();
        sink.begin(d).unwrap();
        let found = src.with_tensor_chunks(&d.name, CHUNK, &mut |off, vals| {
            let payload: Vec<u8> = if bf16 { vals.iter().flat_map(|v| model::half::f32_to_bf16(*v).to_le_bytes()).collect() } else { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
            sink.write_tile(d, &Tile { elems: off..off + vals.len() as u64, payload: &payload, scales: Vec::new() }).unwrap();
        });
        assert!(found, "the source has no {}", d.name);
        sink.finish(d).unwrap();
    }
    sink.commit().expect("commit the exported checkpoint");
    std::fs::metadata(path).unwrap().len()
}
