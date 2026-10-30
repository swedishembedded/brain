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
    let out = argmax_rows_issue(g, k, logits, rows, vocab);
    read_argmax(g, &out, rows)
}

/// The dispatches of [`argmax_rows`], submitted without a read: the `[rows]` winning indices are left in the returned
/// buffer. Split from the read so a recorded step ([`gpu_core::tape`]) can hold the reduction and read after a replay.
pub fn argmax_rows_issue(g: &Gpu, k: ReadoutKernels, logits: &DeviceBuffer, rows: u32, vocab: u32) -> DeviceBuffer {
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
    out
}

/// The indices [`argmax_rows_issue`] left in `out`.
pub fn read_argmax(g: &Gpu, out: &DeviceBuffer, rows: u32) -> Vec<u32> {
    g.read(out, rows as usize).into_iter().map(|x| x as u32).collect()
}

/// Where [`topk_rows_issue`] left a block's candidates: `[rows, cap]` logits and token ids, best first.
#[derive(Clone)]
pub struct TopkOut {
    pub vals: DeviceBuffer,
    pub idx: DeviceBuffer,
    pub rows: u32,
    pub cap: u32,
}

impl TopkOut {
    /// The candidates of every row as `(token id, logit)`, read from the card.
    pub fn read(&self, g: &Gpu) -> Vec<Vec<(u32, f32)>> {
        let n = (self.rows * self.cap) as usize;
        let vals = g.read(&self.vals, n);
        let idx = g.read(&self.idx, n);
        (0..self.rows as usize)
            .map(|r| {
                let s = r * self.cap as usize;
                idx[s..s + self.cap as usize].iter().map(|&x| x as u32).zip(vals[s..s + self.cap as usize].iter().copied()).collect()
            })
            .collect()
    }
}

/// Every row's top-`cap` `(token id, logit)` candidates, best first, from
/// `logits` (`[rows, vocab]`), extracted on the device. Destroys `logits` (the
/// winners are masked in place).
pub fn topk_rows(g: &Gpu, k: ReadoutKernels, logits: &DeviceBuffer, rows: u32, vocab: u32, cap: u32) -> Vec<Vec<(u32, f32)>> {
    topk_rows_issue(g, k, logits, rows, vocab, cap).read(g)
}

/// The dispatches of [`topk_rows`], submitted without a read.
pub fn topk_rows_issue(g: &Gpu, k: ReadoutKernels, logits: &DeviceBuffer, rows: u32, vocab: u32, cap: u32) -> TopkOut {
    assert!(cap > 0, "topk_rows: cap must be > 0");
    let vals = g.storage((rows * cap) as u64);
    let idx = g.storage((rows * cap) as u64);
    let mut steps: Vec<Step> = Vec::new();
    // One launch for all candidates where the device is offered the kernel; else a round of three launches a candidate.
    if let Some(step) = native_topk(g, logits, &vals, &idx, rows, vocab, cap) {
        steps.push(step);
    } else {
        let chunk = vocab.div_ceil(ARGMAX_CHUNKS);
        let part = g.storage(rows as u64 * ARGMAX_CHUNKS as u64 * 2);
        let arg = g.storage(rows as u64);
        for col in 0..cap {
            steps.push(g.step(k.part, &[logits, &part], &[rows, vocab, ARGMAX_CHUNKS, chunk], rows * ARGMAX_CHUNKS));
            steps.push(g.step(k.fin, &[&part, &arg], &[rows, ARGMAX_CHUNKS], rows));
            steps.push(g.step(k.extract, &[&arg, logits, &vals, &idx], &[rows, vocab, cap, col], rows));
        }
    }
    g.submit(&[], &steps);
    TopkOut { vals, idx, rows, cap }
}

#[cfg(not(target_arch = "wasm32"))]
fn native_topk(g: &Gpu, logits: &DeviceBuffer, vals: &DeviceBuffer, idx: &DeviceBuffer, rows: u32, vocab: u32, cap: u32) -> Option<Step> {
    gpu_core::provider::cuda::topk_rows_step(g, &[logits, vals, idx], rows, vocab, cap)
}

#[cfg(target_arch = "wasm32")]
fn native_topk(_g: &Gpu, _logits: &DeviceBuffer, _vals: &DeviceBuffer, _idx: &DeviceBuffer, _rows: u32, _vocab: u32, _cap: u32) -> Option<Step> {
    None
}
