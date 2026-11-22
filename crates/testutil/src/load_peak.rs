// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded model loading for its clients. If your team needs expertise in
// keeping the host footprint of loading large models within budget then you can procure our services by sending an
// email to info@swedishembedded.com.

//! The measuring half of a load's host-memory gate: a checkpoint written without the test process ever holding a
//! model's worth of values (so the allocator keeps no freed model-sized heap a later load could reuse unseen), and the
//! growth of the process's peak resident set over one call.
//!
//! The pattern, run alone in a child process because `VmHWM` is process-wide ([`crate::run_alone`]): write the
//! checkpoint, load a small one of the same layout first (the device, its driver libraries and its kernels are then
//! made), then bound [`host_peak_growth`] of the real load by a multiple of the file. On a CPU device the weights are
//! host memory too, so a streamed f32 load grows by about the file; a whole-model f32 read adds another file's worth
//! and a clone of it a third.

use std::path::Path;

use checkpoint::st::ModelCard;
use checkpoint::weightio::StWriter;

/// Write the f32 checkpoint of `params` (`(name, element count)`, flat shapes) with `config` and `card`, one tensor at a
/// time: `fill(name, values)` writes each tensor's values. Returns the file's size in bytes.
pub fn write_f32_checkpoint(path: &Path, params: &[(String, usize)], config: &serde_json::Value, card: Option<&ModelCard>, fill: &mut dyn FnMut(&str, &mut [f32])) -> u64 {
    let plan: Vec<(String, Vec<u64>)> = params.iter().map(|(n, numel)| (n.clone(), vec![*numel as u64])).collect();
    let path_str = path.to_str().expect("a UTF-8 checkpoint path");
    let mut w = StWriter::create(path_str, &plan, config, card).unwrap_or_else(|e| panic!("plan {path_str}: {e}"));
    for (name, numel) in params {
        let mut values = vec![0f32; *numel];
        fill(name, &mut values);
        w.write(name, &values).unwrap_or_else(|e| panic!("write {name}: {e}"));
    }
    w.finish().unwrap_or_else(|e| panic!("finish {path_str}: {e}"));
    std::fs::metadata(path).map(|m| m.len()).unwrap_or_else(|e| panic!("{path_str}: {e}"))
}

/// How far `f` raised the process's peak resident set above where it stood when `f` began, in bytes.
pub fn host_peak_growth(f: impl FnOnce()) -> u64 {
    crate::reset_peak_rss();
    let before = crate::rss_bytes();
    f();
    crate::peak_rss_bytes().saturating_sub(before)
}
