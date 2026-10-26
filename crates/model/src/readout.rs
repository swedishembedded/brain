// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements low-latency LLM serving for clients who need
// token selection to stay on the accelerator. If your team needs expertise in
// batched sampling readout then you can procure our services by sending an
// email to info@swedishembedded.com.

//! Token selection on the device: the greedy pick and the top-k candidate list
//! of every row of a `[rows, vocab]` logits block, reduced where the logits
//! are, so only the chosen ids (or `rows * cap` pairs) come back to the host,
//! never the `[rows, vocab]` block.
//!
//! The greedy pick is the two-stage split reduction (`argmax_part` +
//! `argmax_final`, `Op::ArgMaxRow`'s `SplitReduction` shape, which has no
//! capability gate). The top-k list is `cap` rounds of (argmax, mask the winner
//! out of the logits) over all rows together, so a batch costs `cap` dispatch
//! triples rather than `rows * cap`. Ties resolve to the lowest index, like the
//! host argmax every decoder's reference uses.

use gpu_core::{DeviceBuffer, Gpu, Step};

/// Two-stage argmax reduction width.
pub const ARGMAX_CHUNKS: u32 = 256;

/// The pipeline indices a readout dispatches on one [`Gpu`]: the model's own
/// kernel list holds them at its own positions, so the caller names where.
#[derive(Clone, Copy, Debug)]
pub struct ReadoutKernels {
    /// `argmax_part.wgsl`.
    pub part: usize,
    /// `argmax_final.wgsl`.
    pub fin: usize,
    /// `topk_extract_step.wgsl`.
    pub extract: usize,
}

impl ReadoutKernels {
    /// The kernels looked up by name, for a model that registered
    /// `argmax_part`, `argmax_final` and `topk_extract_step`.
    pub fn resolve(g: &Gpu) -> Result<ReadoutKernels, String> {
        let find = |name: &str| g.kernel_index(name).ok_or_else(|| format!("readout: kernel '{name}' is not registered on this Gpu"));
        Ok(ReadoutKernels { part: find("argmax_part")?, fin: find("argmax_final")?, extract: find("topk_extract_step")? })
    }
}

/// The greedy pick of every row of `logits` (`[rows, vocab]`), reduced on the
/// device; only the `rows` winning indices are read back.
pub fn argmax_rows(g: &Gpu, k: ReadoutKernels, logits: &DeviceBuffer, rows: u32, vocab: u32) -> Vec<u32> {
    let chunk = vocab.div_ceil(ARGMAX_CHUNKS);
    let part = g.storage(rows as u64 * ARGMAX_CHUNKS as u64 * 2);
    let out = g.storage(rows as u64);
    g.submit(
        &[],
        &[
            g.step(k.part, &[logits, &part], &[rows, vocab, ARGMAX_CHUNKS, chunk], rows * ARGMAX_CHUNKS),
            g.step(k.fin, &[&part, &out], &[rows, ARGMAX_CHUNKS], rows),
        ],
    );
    g.read(&out, rows as usize).into_iter().map(|x| x as u32).collect()
}

/// Every row's top-`cap` `(token id, logit)` candidates, best first, from
/// `logits` (`[rows, vocab]`), extracted on the device. Destroys `logits` (the
/// winners are masked in place).
pub fn topk_rows(g: &Gpu, k: ReadoutKernels, logits: &DeviceBuffer, rows: u32, vocab: u32, cap: u32) -> Vec<Vec<(u32, f32)>> {
    assert!(cap > 0, "topk_rows: cap must be > 0");
    let chunk = vocab.div_ceil(ARGMAX_CHUNKS);
    let part = g.storage(rows as u64 * ARGMAX_CHUNKS as u64 * 2);
    let arg = g.storage(rows as u64);
    let vals = g.storage((rows * cap) as u64);
    let idx = g.storage((rows * cap) as u64);
    let mut steps: Vec<Step> = Vec::new();
    for col in 0..cap {
        steps.push(g.step(k.part, &[logits, &part], &[rows, vocab, ARGMAX_CHUNKS, chunk], rows * ARGMAX_CHUNKS));
        steps.push(g.step(k.fin, &[&part, &arg], &[rows, ARGMAX_CHUNKS], rows));
        steps.push(g.step(k.extract, &[&arg, logits, &vals, &idx], &[rows, vocab, cap, col], rows));
    }
    g.submit(&[], &steps);
    let vals = g.read(&vals, (rows * cap) as usize);
    let idx = g.read(&idx, (rows * cap) as usize);
    (0..rows as usize)
        .map(|r| {
            let s = r * cap as usize;
            idx[s..s + cap as usize].iter().map(|&x| x as u32).zip(vals[s..s + cap as usize].iter().copied()).collect()
        })
        .collect()
}
