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
fn native_draw(g: &Gpu, logits: &DeviceBuffer, spec: &DeviceBuffer, out: &DeviceBuffer, rows: u32, vocab: u32, step: u32) -> Option<Step> {
    gpu_core::provider::cuda::draw_rows_step(g, &[logits, spec, out], rows, vocab, step)
}

#[cfg(target_arch = "wasm32")]
fn native_draw(_g: &Gpu, _logits: &DeviceBuffer, _spec: &DeviceBuffer, _out: &DeviceBuffer, _rows: u32, _vocab: u32, _step: u32) -> Option<Step> {
    None
}

/// Whether [`RowDrawer::steps`] draws with the native one-launch kernel on `g` at this shape (diagnostics and tests).
pub fn draws_natively(g: &Gpu, rows: u32, vocab: u32) -> bool {
    let probe = |n: u64| g.storage(n.max(1));
    native_draw(g, &probe(rows as u64 * vocab as u64), &probe(rows as u64 * 9), &probe(rows as u64 * 4), rows, vocab, 0).is_some()
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

/// The greedy pick's and the candidate extraction's kernels, for a model's kernel list ([`ReadoutKernels::resolve`]).
pub const READOUT_KERNELS: [(&str, &str); 3] = [("argmax_part", kernels::ARGMAX_PART), ("argmax_final", kernels::ARGMAX_FINAL), ("topk_extract_step", kernels::TOPK_EXTRACT_STEP)];

/// The keyed draw's kernels, for a model's kernel list: a decoder that registers them draws its sampled rows on the
/// device ([`RowDrawer`]).
pub const DRAW_KERNELS: [(&str, &str); 6] = [
    ("draw_part", kernels::DRAW_PART),
    ("draw_final", kernels::DRAW_FINAL),
    ("nucleus_part", kernels::NUCLEUS_PART),
    ("nucleus_step", kernels::NUCLEUS_STEP),
    ("nucleus_cut", kernels::NUCLEUS_CUT),
    ("draw_feed", kernels::DRAW_FEED),
];

/// The most steps of a decode window a drawer holds the keys of ([`RowDrawer::upload_window_keys`]).
pub const DRAW_WINDOW: u32 = 8;

/// The widest best-first list a drawer takes on the device: a `top_k` above it is drawn whole on the host.
pub const MAX_DRAW_CANDIDATES: u32 = 1024;

/// Halvings of the nucleus threshold search: an ordered 32-bit key range ends on adjacent keys.
const NUCLEUS_STEPS: u32 = 32;

/// The ordered key of the bottom of the nucleus search (negative infinity's) and its top (past every float).
const NUCLEUS_LOW: u32 = 0x007f_ffff;
const NUCLEUS_HIGH: u32 = u32::MAX;

/// What one keyed step dispatches beyond the rows' keys: the min-p pass, the nucleus search, and the best-first
/// candidates a `top_k` row is answered from (`0` for none). Part of a recording's identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DrawShape {
    pub floors: bool,
    pub nucleus: bool,
    pub candidates: u32,
}

/// How the drawer treats a row: over the whole row (flag 1), over its nucleus (flag 2, the threshold found on the
/// device), or from its best-first candidates (flag 1 too: the row's statistics are what the list is cut with).
fn flag_of(d: &crate::serve::RowDraw, vocab: usize) -> u32 {
    if d.has_nucleus() && !d.keeps_top_k(vocab) {
        2
    } else {
        1
    }
}

/// Draws every keyed row of a `[rows, vocab]` logits block where the logits are ([`crate::serve::draw_row`]'s
/// rule): two launches over all rows (one more pair when a row has a min-p floor; one native launch on CUDA), and
/// only `[rows]` (token, log-probability, best, mass) quadruples read back. A nucleus row is drawn over its exact
/// nucleus, found on the device by a threshold search over the row (the largest logit whose tokens at or above it
/// carry `top_p` of the mass, equal logits taken lowest column first, as the host sampler's best-first prefix takes
/// them). A `top_k` row is answered from its best-first candidates, taken on the device after the draw
/// ([`topk_rows_steps`]) and finished on the host ([`crate::serve::draw_from_candidates`]); a `top_k` wider than
/// [`MAX_DRAW_CANDIDATES`] is read back and drawn whole ([`crate::serve::draw_row`]). Owns its scratch, sized for
/// `max_rows`.
pub struct RowDrawer {
    part_kernel: usize,
    final_kernel: usize,
    nucleus_part: usize,
    nucleus_step: usize,
    nucleus_cut: usize,
    feed_kernel: usize,
    /// The argmax and extraction kernels the candidate lists are taken with, when the model registered them.
    topk: Option<ReadoutKernels>,
    max_rows: u32,
    /// `[max_rows, 8]` u32 per row (temperature bits, min-p bits, unused, draw flag, top-p bits, three unused), then
    /// `[DRAW_WINDOW, rows]` the rows' keys of each step of a decode window (laid out for the rows of the step).
    spec: DeviceBuffer,
    /// `[max_rows, DRAW_SLICES, 4]` partial reductions (the nucleus search uses the first `DRAW_SLICES` a row).
    part: DeviceBuffer,
    /// `[max_rows, 4]` per row: best `logit / temp`, the nucleus search's low and high keys, the cut column.
    state: DeviceBuffer,
    /// `[max_rows, 4]` drawn token (as f32), its log-probability, the row's best `logit / temp` and its mass.
    out: DeviceBuffer,
    /// `[max_rows, candidates]` best-first candidates of a step that takes them (logits and token ids).
    cand_vals: DeviceBuffer,
    cand_idx: DeviceBuffer,
    /// `[DRAW_WINDOW, max_rows]` a decode window's drawn tokens (as f32) and log-probabilities ([`Self::feed`]).
    hist: DeviceBuffer,
    lphist: DeviceBuffer,
}

impl RowDrawer {
    /// A drawer for up to `max_rows` rows on `g`, or `None` when the model did not register [`DRAW_KERNELS`].
    pub fn new(g: &Gpu, max_rows: u32) -> Option<RowDrawer> {
        let k = |name: &str| g.kernel_index(name);
        let (part_kernel, final_kernel) = (k("draw_part")?, k("draw_final")?);
        let (nucleus_part, nucleus_step, nucleus_cut) = (k("nucleus_part")?, k("nucleus_step")?, k("nucleus_cut")?);
        let feed_kernel = k("draw_feed")?;
        let topk = ReadoutKernels::resolve(g).ok();
        let rows = max_rows.max(1) as u64;
        let cands = if topk.is_some() { rows * MAX_DRAW_CANDIDATES as u64 } else { 1 };
        Some(RowDrawer {
            part_kernel,
            final_kernel,
            nucleus_part,
            nucleus_step,
            nucleus_cut,
            feed_kernel,
            topk,
            max_rows: max_rows.max(1),
            spec: g.storage(rows * (8 + DRAW_WINDOW as u64)),
            part: g.storage(rows * DRAW_SLICES as u64 * 4),
            state: g.storage(rows * 4),
            out: g.storage(rows * 4),
            cand_vals: g.storage(cands),
            cand_idx: g.storage(cands),
            hist: g.storage(rows * DRAW_WINDOW as u64),
            lphist: g.storage(rows * DRAW_WINDOW as u64),
        })
    }

    /// The steps that draw `draws[i]` from row `i` of `logits` (`[draws.len(), vocab]`): [`Self::upload`], then
    /// [`Self::dispatch`] at [`Self::shape`]. A row whose draw is `None` is left alone. The caller submits them after
    /// the steps that write `logits` and then reads the answer with [`Self::answer`] at the same shape.
    pub fn steps(&self, g: &Gpu, logits: &DeviceBuffer, draws: &[Option<crate::serve::RowDraw>], vocab: u32) -> Vec<Step> {
        self.upload(g, draws, vocab);
        self.dispatch(g, logits, draws.len() as u32, vocab, self.shape(draws, vocab))
    }

    /// What a step of `draws` over a `vocab`-wide row dispatches: the min-p pass when a whole-row draw has a floor, the
    /// nucleus search when a row has a nucleus, and as many candidates as the widest `top_k` (up to
    /// [`MAX_DRAW_CANDIDATES`]) asks for; none on a model without the extraction kernels (those rows are then drawn
    /// whole on the host).
    pub fn shape(&self, draws: &[Option<crate::serve::RowDraw>], vocab: u32) -> DrawShape {
        Self::shape_for(draws, vocab, self.topk.is_some())
    }

    /// [`Self::shape`] for a drawer that has the candidate extraction (`extracts`) or not, before one is built (a
    /// recording is keyed on it).
    pub fn shape_for(draws: &[Option<crate::serve::RowDraw>], vocab: u32, extracts: bool) -> DrawShape {
        let v = vocab as usize;
        let keyed = || draws.iter().flatten();
        let floors = keyed().any(|d| flag_of(d, v) == 1 && d.min_p > 0.0);
        let nucleus = keyed().any(|d| flag_of(d, v) == 2);
        let want = |d: &crate::serve::RowDraw| if d.keeps_top_k(v) && d.top_k as u32 <= MAX_DRAW_CANDIDATES { d.top_k as u32 } else { 0 };
        let candidates = if extracts { keyed().map(want).max().unwrap_or(0).min(vocab) } else { 0 };
        DrawShape { floors, nucleus, candidates }
    }

    /// Write the rows' temperatures, floors, nuclei and keys where the dispatch reads them, and reset each row's
    /// nucleus search: before every run of the steps, recorded ones included (a recording holds the dispatch, not
    /// the keys).
    pub fn upload(&self, g: &Gpu, draws: &[Option<crate::serve::RowDraw>], vocab: u32) {
        assert!(!draws.is_empty() && draws.len() as u32 <= self.max_rows, "RowDrawer: {} rows for a drawer of {}", draws.len(), self.max_rows);
        let mut spec = vec![0u32; draws.len() * 9];
        let (rows, keys) = spec.split_at_mut(draws.len() * 8);
        for ((s, k), d) in rows.chunks_mut(8).zip(keys.iter_mut()).zip(draws) {
            if let Some(d) = d {
                s[..5].copy_from_slice(&[d.temp.to_bits(), d.min_p.to_bits(), 0, flag_of(d, vocab as usize), d.top_p.to_bits()]);
                *k = d.key;
            }
        }
        g.write(&self.spec, &spec);
        if draws.iter().flatten().any(|d| flag_of(d, vocab as usize) == 2) {
            let state: Vec<u32> = draws.iter().flat_map(|_| [0, NUCLEUS_LOW, NUCLEUS_HIGH, 0]).collect();
            g.write(&self.state, &state);
        }
    }

    /// The keys of steps `1..` of a decode window over the rows last uploaded: `keys[s - 1][i]` is row `i`'s at step `s`
    /// (step 0's are [`Self::upload`]'s). A window's steps differ only in their keys.
    pub fn upload_window_keys(&self, g: &Gpu, keys: &[Vec<u32>]) {
        let Some(rows) = keys.first().map(Vec::len) else { return };
        assert!(keys.len() < DRAW_WINDOW as usize, "RowDrawer: a window of {} steps for a drawer of {DRAW_WINDOW}", keys.len() + 1);
        let flat: Vec<u32> = keys.iter().flat_map(|k| k.iter().copied()).collect();
        g.write_at(&self.spec, (rows * 9) as u64, &flat);
    }

    /// [`Self::dispatch_at`] step 0: a single step's draw.
    pub fn dispatch(&self, g: &Gpu, logits: &DeviceBuffer, rows: u32, vocab: u32, shape: DrawShape) -> Vec<Step> {
        self.dispatch_at(g, logits, rows, vocab, shape, 0)
    }

    /// The step that writes step `s` of a decode window's drawn tokens into the window history and into `tok` (the
    /// `[rows]` u32 inputs the next step reads): what a decoder chains between a window's steps on the device.
    pub fn feed(&self, g: &Gpu, tok: &DeviceBuffer, rows: u32, s: u32) -> Step {
        assert!(s < DRAW_WINDOW, "RowDrawer: window step {s} past {DRAW_WINDOW}");
        g.step(self.feed_kernel, &[&self.out, tok, &self.hist, &self.lphist], &[rows, s], rows)
    }

    /// The `(token, log-probability)` of every row at each of a window's `k` steps ([`Self::feed`]), row by row.
    pub fn read_window(&self, g: &Gpu, rows: usize, k: usize) -> Vec<Vec<(u32, f32)>> {
        let (toks, lps) = (g.read(&self.hist, k * rows), g.read(&self.lphist, k * rows));
        (0..rows).map(|i| (0..k).map(|s| (toks[s * rows + i] as u32, lps[s * rows + i])).collect()).collect()
    }

    /// The dispatches that draw `rows` rows of `logits` under the keys last uploaded for window step `step`, at `shape`: the native
    /// one-launch kernel where the device offers it (priced as the WGSL passes), else the passes (the draw, the
    /// min-p pass, the nucleus search, cut and draw), then the candidate extraction, which masks the taken
    /// candidates in `logits` ([`Self::answer`] restores a row it reads back).
    pub fn dispatch_at(&self, g: &Gpu, logits: &DeviceBuffer, rows: u32, vocab: u32, shape: DrawShape, step: u32) -> Vec<Step> {
        assert!(step < DRAW_WINDOW, "RowDrawer: window step {step} past {DRAW_WINDOW}");
        assert!(rows > 0 && rows <= self.max_rows, "RowDrawer: {rows} rows for a drawer of {}", self.max_rows);
        let draw = |stage: u32| {
            [
                g.step(self.part_kernel, &[logits, &self.spec, &self.state, &self.part], &[rows, vocab, DRAW_SLICES, stage, step], rows * DRAW_SLICES),
                g.step(self.final_kernel, &[logits, &self.spec, &self.part, &self.state, &self.out], &[rows, vocab, DRAW_SLICES, stage, step], rows),
            ]
        };
        let mut reference: Vec<Step> = draw(0).to_vec();
        if shape.floors {
            reference.extend(draw(1));
        }
        if shape.nucleus {
            for _ in 0..NUCLEUS_STEPS {
                reference.push(g.step(self.nucleus_part, &[logits, &self.spec, &self.state, &self.part], &[rows, vocab, DRAW_SLICES, 0], rows * DRAW_SLICES));
                reference.push(g.step(self.nucleus_step, &[&self.spec, &self.out, &self.part, &self.state], &[rows, DRAW_SLICES], rows));
            }
            reference.push(g.step(self.nucleus_cut, &[logits, &self.spec, &self.out, &self.state], &[rows, vocab], rows));
            reference.extend(draw(2));
        }
        let mut steps = match native_draw(g, logits, &self.spec, &self.out, rows, vocab, step) {
            Some(step) => vec![g.priced_as_chain(step, &reference)],
            None => reference,
        };
        if let (Some(k), true) = (self.topk, shape.candidates > 0) {
            let n = shape.candidates;
            steps.extend(topk_rows_steps(g, k, logits, &self.cand_vals, &self.cand_idx, rows, vocab, n, n));
        }
        steps
    }

    /// Each row's answer once [`Self::dispatch`] ran at `shape`: the drawn token where `draws[i]` is `Some` (from the
    /// device's draw over the row or its nucleus, from the row's candidates, or from the row itself read back when
    /// its `top_k` is wider than the candidates), and row `i` of `logits` read back where it is `None`. Only those
    /// rows cross to the host.
    pub fn answer(&self, g: &Gpu, logits: &DeviceBuffer, draws: &[Option<crate::serve::RowDraw>], vocab: u32, shape: DrawShape) -> Vec<crate::serve::Drawn> {
        use crate::serve::{draw_from_candidates, draw_row_whole, Drawn};
        let rows = draws.len();
        let v = vocab as usize;
        let out = if draws.iter().any(Option::is_some) { g.read(&self.out, rows * 4) } else { Vec::new() };
        let n = shape.candidates as usize;
        let (vals, idx) = if n > 0 { (g.read(&self.cand_vals, rows * n), g.read(&self.cand_idx, rows * n)) } else { (Vec::new(), Vec::new()) };
        let list = |i: usize| -> Vec<(u32, f32)> { (0..n).map(|c| (idx[i * n + c] as u32, vals[i * n + c])).collect() };
        // A row read back has its candidates masked by the extraction: they are written back from the list.
        let read_row = |i: usize| {
            let mut row = vec![0.0f32; v];
            g.read_into(logits, i as u64 * vocab as u64, &mut row);
            for (t, l) in list(i) {
                row[t as usize] = l;
            }
            row
        };
        let mut answers: Vec<Option<Drawn>> = draws
            .iter()
            .enumerate()
            .map(|(i, d)| match d {
                None => Some(Drawn::Row(read_row(i))),
                Some(d) if !d.keeps_top_k(v) => {
                    let (token, lp) = (out[4 * i] as u32, out[4 * i + 1]);
                    // A row drawn over every token has its own figure; a nucleus or min-p floor drew over fewer, and the
                    // whole row's figure needs the token's logit with the row's best and mass.
                    let whole = if d.has_nucleus() || d.min_p > 0.0 {
                        let logit = list(i).iter().find(|&&(t, _)| t == token).map(|&(_, l)| l).unwrap_or_else(|| {
                            let mut one = [0.0f32];
                            g.read_into(logits, i as u64 * vocab as u64 + token as u64, &mut one);
                            one[0]
                        });
                        let whole = logit / d.temp.max(1e-6) - out[4 * i + 2] - out[4 * i + 3].ln();
                        if whole.is_nan() { lp } else { whole }
                    } else {
                        lp
                    };
                    Some(Drawn::Token(token, lp, whole))
                }
                Some(d) if n > 0 => draw_from_candidates(&list(i), out[4 * i + 2], out[4 * i + 3], *d, v).map(|(t, lp, whole)| Drawn::Token(t, lp, whole)),
                Some(_) => None,
            })
            .collect();
        // The rows their candidates could not answer, drawn whole on the CPU pool.
        let whole: Vec<(usize, Vec<f32>)> = answers.iter().enumerate().filter(|(_, a)| a.is_none()).map(|(i, _)| (i, read_row(i))).collect();
        let drawn = backend_cpu::par::map(whole.len(), |j| draw_row_whole(&whole[j].1, draws[whole[j].0].expect("a keyed row")));
        for ((i, _), (t, lp, whole)) in whole.iter().zip(drawn) {
            answers[*i] = Some(Drawn::Token(t, lp, whole));
        }
        answers.into_iter().map(|a| a.expect("every row answered")).collect()
    }
}

/// Each row's answer to a keyed step from `logits` (`[draws.len(), vocab]`, on `g`): drawn on the device by a drawer
/// built for these rows ([`RowDrawer`]), or, on a model that did not register [`DRAW_KERNELS`], from the rows read
/// back and drawn on the CPU pool. The one entry point an engine without a drawer of its own answers
/// `PagedDecoder::forward_batched_drawn` with.
pub fn draw_rows(g: &Gpu, logits: &DeviceBuffer, draws: &[Option<crate::serve::RowDraw>], vocab: u32) -> Vec<crate::serve::Drawn> {
    if draws.is_empty() {
        return Vec::new();
    }
    let Some(drawer) = RowDrawer::new(g, draws.len() as u32) else {
        let rows = g.read(logits, draws.len() * vocab as usize).chunks(vocab as usize).map(<[f32]>::to_vec).collect();
        return crate::serve::draw_rows_host(rows, draws);
    };
    let shape = drawer.shape(draws, vocab);
    let steps = drawer.steps(g, logits, draws, vocab);
    g.submit(&[], &steps);
    drawer.answer(g, logits, draws, vocab, shape)
}
