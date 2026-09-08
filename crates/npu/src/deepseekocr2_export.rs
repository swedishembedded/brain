// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Export DeepSeek-OCR-2's two ONNX-eligible pieces - the new resampler
//! (`crate::deepseekocr2_topology`, one graph per view size) and the decoder
//! (`crate::deepseek2_topology`) - to fp32 ONNX files for a best-effort
//! OpenVINO/NPU compile attempt. Pure Rust; reuses `deepseekocr2::import`'s
//! own real-checkpoint expansion so this export reads the SAME tensors every
//! other real-weight test in this campaign does, rather than a second
//! checkpoint-reading path.
//!
//! **SAM is not exported here** - see `crate::deepseekocr2_topology`'s module
//! doc for why (a real, unresolved gap: no windowed-attention/decomposed-
//! relative-position-bias ONNX precedent exists anywhere in this crate).

use std::path::Path;

use deepseek2::config::DeepseekV2Config;
use deepseekocr2::import::Files;

use crate::deepseek2_topology::build_deepseek2_graph;
use crate::deepseekocr2_topology::build_resampler_graph;
use onnx::builder::GraphBuilder;

/// Export both resampler views (local/768-tile and global/1024-view) plus
/// the decoder, into `out_dir`, named `resampler_local.onnx`,
/// `resampler_global.onnx`, `decoder.onnx`. `seq_len` is the decoder's fixed
/// export sequence length.
pub fn export_all(checkpoint_dir: &str, seq_len: usize, out_dir: &str) -> Result<(), String> {
    let files = Files::locate(checkpoint_dir)?;
    std::fs::create_dir_all(out_dir).map_err(|e| format!("{out_dir}: {e}"))?;

    let vcfg = deepseekocr2::import::vision_config(&files.mmproj, DeepseekV2Config::deepseek_ocr(seq_len as u32).d_model())?;
    let vreader = deepseekocr2::import::vision_reader(&files)?;

    for (local, n_query, tag) in [(true, vcfg.encoder.n_query_local as usize, "local"), (false, vcfg.encoder.n_query_global as usize, "global")] {
        let mut g = GraphBuilder::new(&format!("deepseekocr2_resampler_{tag}"));
        build_resampler_graph(&vcfg.encoder, &vreader, n_query, local, vcfg.decoder_hidden as usize, &mut g);
        let path = Path::new(out_dir).join(format!("resampler_{tag}.onnx"));
        std::fs::write(&path, g.finish()).map_err(|e| format!("{}: {e}", path.display()))?;
    }

    let dcfg = DeepseekV2Config::deepseek_ocr(seq_len as u32);
    let dreader = deepseekocr2::import::decoder_reader(&files)?;
    let mut g = GraphBuilder::new("deepseekocr2_decoder");
    build_deepseek2_graph(&dcfg, &dreader, seq_len, &mut g);
    let path = Path::new(out_dir).join("decoder.onnx");
    std::fs::write(&path, g.finish()).map_err(|e| format!("{}: {e}", path.display()))?;

    Ok(())
}
