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
    let steps = topk_rows_steps(g, k, logits, &vals, &idx, rows, vocab, cap, cap);
    g.submit(&[], &steps);
    TopkOut { vals, idx, rows, cap }
}

/// The steps that write each row's best `count` candidates, best first, into the first `count` columns of
/// `[rows, stride]` buffers `vals` and `idx` the caller owns (`count <= stride`), for a caller that records them beside
/// its own steps. One launch for all candidates where the device is offered the kernel and `count == stride`, else a
/// round of three launches a candidate. Destroys `logits` (the winners are masked in place).
#[allow(clippy::too_many_arguments)]
pub fn topk_rows_steps(g: &Gpu, k: ReadoutKernels, logits: &DeviceBuffer, vals: &DeviceBuffer, idx: &DeviceBuffer, rows: u32, vocab: u32, stride: u32, count: u32) -> Vec<Step> {
    assert!(count > 0 && count <= stride, "topk_rows: {count} candidates in {stride} columns");
    if count == stride {
        if let Some(step) = native_topk(g, logits, vals, idx, rows, vocab, stride) {
            return vec![rounds_work(g, k, rows, vocab, stride, count).and_then(|work| step.priced_as(work)).unwrap_or(step)];
        }
    }
    let chunk = vocab.div_ceil(ARGMAX_CHUNKS);
    let part = g.storage(rows as u64 * ARGMAX_CHUNKS as u64 * 2);
    let arg = g.storage(rows as u64);
    let mut steps: Vec<Step> = Vec::with_capacity(3 * count as usize);
    for col in 0..count {
        steps.push(g.step(k.part, &[logits, &part], &[rows, vocab, ARGMAX_CHUNKS, chunk], rows * ARGMAX_CHUNKS));
        steps.push(g.step(k.fin, &[&part, &arg], &[rows, ARGMAX_CHUNKS], rows));
        steps.push(g.step(k.extract, &[&arg, logits, vals, idx], &[rows, vocab, stride, col], rows));
    }
    steps
}

/// The portable work of [`topk_rows_steps`]'s rounds, what its single native launch is priced as: per candidate the two
/// argmax stages and the extraction, at the uniforms the rounds dispatch.
fn rounds_work(g: &Gpu, k: ReadoutKernels, rows: u32, vocab: u32, stride: u32, count: u32) -> Option<Vec<gpu_core::PortableDispatch>> {
    let name = |kind: usize| g.kernel_name(kind).map(|n| std::borrow::Cow::<'static, str>::Owned(n.to_string()));
    let (part, fin, extract) = (name(k.part)?, name(k.fin)?, name(k.extract)?);
    let chunk = vocab.div_ceil(ARGMAX_CHUNKS);
    let round = |col: u32| {
        [
            gpu_core::PortableDispatch { kernel: part.clone(), params: vec![rows, vocab, ARGMAX_CHUNKS, chunk], threads: rows * ARGMAX_CHUNKS },
            gpu_core::PortableDispatch { kernel: fin.clone(), params: vec![rows, ARGMAX_CHUNKS], threads: rows },
            gpu_core::PortableDispatch { kernel: extract.clone(), params: vec![rows, vocab, stride, col], threads: rows },
        ]
    };
    Some((0..count).flat_map(round).collect())
}

#[cfg(not(target_arch = "wasm32"))]
fn native_draw(g: &Gpu, logits: &DeviceBuffer, spec: &DeviceBuffer, out: &DeviceBuffer, rows: u32, vocab: u32) -> Option<Step> {
    gpu_core::provider::cuda::draw_rows_step(g, &[logits, spec, out], rows, vocab)
}

#[cfg(target_arch = "wasm32")]
fn native_draw(_g: &Gpu, _logits: &DeviceBuffer, _spec: &DeviceBuffer, _out: &DeviceBuffer, _rows: u32, _vocab: u32) -> Option<Step> {
    None
}

/// Whether [`RowDrawer::steps`] draws with the native one-launch kernel on `g` at this shape (diagnostics and tests).
pub fn draws_natively(g: &Gpu, rows: u32, vocab: u32) -> bool {
    let probe = |n: u64| g.storage(n.max(1));
    native_draw(g, &probe(rows as u64 * vocab as u64), &probe(rows as u64 * 4), &probe(rows as u64 * 2), rows, vocab).is_some()
}

#[cfg(not(target_arch = "wasm32"))]
fn native_topk(g: &Gpu, logits: &DeviceBuffer, vals: &DeviceBuffer, idx: &DeviceBuffer, rows: u32, vocab: u32, cap: u32) -> Option<Step> {
    gpu_core::provider::cuda::topk_rows_step(g, &[logits, vals, idx], rows, vocab, cap)
}

#[cfg(target_arch = "wasm32")]
fn native_topk(_g: &Gpu, _logits: &DeviceBuffer, _vals: &DeviceBuffer, _idx: &DeviceBuffer, _rows: u32, _vocab: u32, _cap: u32) -> Option<Step> {
    None
}

/// Slices each row is cut into by the keyed draw's partial stage (`draw_part`): the threads a row reduces over.
pub const DRAW_SLICES: u32 = 256;

/// The keyed draw's kernels, for a model's kernel list: a decoder that registers them draws its sampled rows on the
/// device ([`RowDrawer`]).
pub const DRAW_KERNELS: [(&str, &str); 2] = [("draw_part", kernels::DRAW_PART), ("draw_final", kernels::DRAW_FINAL)];

/// Draws every keyed row of a `[rows, vocab]` logits block where the logits are ([`crate::serve::draw_row`]'s
/// rule, the same tokens): two launches over all rows (one more pair when a row has a min-p floor), and only
/// `[rows]` (token, log-probability) pairs read back. Owns its scratch, sized for `max_rows`.
pub struct RowDrawer {
    part_kernel: usize,
    final_kernel: usize,
    max_rows: u32,
    /// `[max_rows, 4]` u32 per row: temperature bits, min-p bits, key, draw flag.
    spec: DeviceBuffer,
    /// `[max_rows, DRAW_SLICES, 4]` partial reductions.
    part: DeviceBuffer,
    /// `[max_rows]` best `logit / temp` of each row, for the min-p pass.
    best: DeviceBuffer,
    /// `[max_rows, 2]` drawn token (as f32) and log-probability.
    out: DeviceBuffer,
}

impl RowDrawer {
    /// A drawer for up to `max_rows` rows on `g`, or `None` when the model did not register [`DRAW_KERNELS`].
    pub fn new(g: &Gpu, max_rows: u32) -> Option<RowDrawer> {
        let (part_kernel, final_kernel) = (g.kernel_index("draw_part")?, g.kernel_index("draw_final")?);
        let rows = max_rows.max(1) as u64;
        Some(RowDrawer { part_kernel, final_kernel, max_rows: max_rows.max(1), spec: g.storage(rows * 4), part: g.storage(rows * DRAW_SLICES as u64 * 4), best: g.storage(rows), out: g.storage(rows * 2) })
    }

    /// The steps that draw `draws[i]` from row `i` of `logits` (`[draws.len(), vocab]`), after uploading the rows'
    /// keys ([`Self::upload`] then [`Self::dispatch`]). A row whose draw is `None` is left alone. The caller submits
    /// them after the steps that write `logits` and then reads the answer with [`Self::answer`].
    pub fn steps(&self, g: &Gpu, logits: &DeviceBuffer, draws: &[Option<crate::serve::RowDraw>], vocab: u32) -> Vec<Step> {
        self.upload(g, draws);
        self.dispatch(g, logits, draws.len() as u32, vocab, Self::has_floor(draws))
    }

    /// Whether any of `draws` has a min-p floor: the dispatch a step of them needs ([`Self::dispatch`]'s `floors`).
    pub fn has_floor(draws: &[Option<crate::serve::RowDraw>]) -> bool {
        draws.iter().flatten().any(|d| d.min_p > 0.0)
    }

    /// Write the rows' temperatures, floors and keys where the dispatch reads them: before every run of the steps,
    /// recorded ones included (a recording holds the dispatch, not the keys).
    pub fn upload(&self, g: &Gpu, draws: &[Option<crate::serve::RowDraw>]) {
        assert!(!draws.is_empty() && draws.len() as u32 <= self.max_rows, "RowDrawer: {} rows for a drawer of {}", draws.len(), self.max_rows);
        let mut spec = vec![0u32; draws.len() * 4];
        for (s, d) in spec.chunks_mut(4).zip(draws) {
            if let Some(d) = d {
                s.copy_from_slice(&[d.temp.to_bits(), d.min_p.to_bits(), d.key, 1]);
            }
        }
        g.write(&self.spec, &spec);
    }

    /// The dispatches that draw `rows` rows of `logits` under the keys last uploaded, with the min-p pass when
    /// `floors`: the native one-launch kernel where the device offers it (priced as the WGSL pair), else the pair.
    pub fn dispatch(&self, g: &Gpu, logits: &DeviceBuffer, rows: u32, vocab: u32, floors: bool) -> Vec<Step> {
        assert!(rows > 0 && rows <= self.max_rows, "RowDrawer: {rows} rows for a drawer of {}", self.max_rows);
        let passes = if floors { 2 } else { 1 };
        let reference: Vec<Step> = (0..passes)
            .flat_map(|pass| {
                [
                    g.step(self.part_kernel, &[logits, &self.spec, &self.best, &self.part], &[rows, vocab, DRAW_SLICES, pass], rows * DRAW_SLICES),
                    g.step(self.final_kernel, &[logits, &self.spec, &self.part, &self.best, &self.out], &[rows, vocab, DRAW_SLICES, pass], rows),
                ]
            })
            .collect();
        match native_draw(g, logits, &self.spec, &self.out, rows, vocab) {
            Some(step) => vec![g.priced_as_chain(step, &reference)],
            None => reference,
        }
    }

    /// Each row's answer once [`Self::steps`] ran: the drawn token where `draws[i]` is `Some`, and row `i` of
    /// `logits` read back where it is `None` (only those rows cross to the host).
    pub fn answer(&self, g: &Gpu, logits: &DeviceBuffer, draws: &[Option<crate::serve::RowDraw>], vocab: u32) -> Vec<crate::serve::Drawn> {
        use crate::serve::Drawn;
        let out = if draws.iter().any(Option::is_some) { g.read(&self.out, draws.len() * 2) } else { Vec::new() };
        draws
            .iter()
            .enumerate()
            .map(|(i, d)| match d {
                Some(_) => Drawn::Token(out[2 * i] as u32, out[2 * i + 1]),
                None => {
                    let mut row = vec![0.0f32; vocab as usize];
                    g.read_into(logits, i as u64 * vocab as u64, &mut row);
                    Drawn::Row(row)
                }
            })
            .collect()
    }
}
