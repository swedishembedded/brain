// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Frame-synchronous streaming for the Nemotron FastConformer encoder + RNN-T.
//!
//! The model is *built* for this: attention is `chunked_limited` (a query in chunk
//! `qc = i/4` attends only chunks `[qc-14, qc]` — 56 frames of left context plus
//! its own 4-frame chunk), and every conv in the model is causal. So streaming is
//! not an approximation here — pushing samples incrementally with per-layer caches
//! computes **the same** pooler frames and tokens as the offline `transcribe` over
//! the whole utterance: bit-for-bit when the backend's kernel selection is
//! shape-invariant (`BRAIN_NO_FASTCONV=1`), and within the CPU conv fast path's
//! documented ≤1-ulp reassociation tolerance otherwise — the token sequence is
//! asserted identical either way (tests below):
//!
//!   * mel front end — [`audio::asr_frontend::NemotronMelStream`] shares the
//!     per-frame implementation with the offline extractor;
//!   * subsampling — each stride-2 causal stage caches its last `k-1 = 2` input
//!     rows; a subsampled row is emitted the moment its (past-only) window is real;
//!   * Conformer blocks — per layer: the last 56 projected key/value rows (the
//!     attention band) and the last `conv_kernel-1 = 8` GLU rows (the causal
//!     depthwise conv window). The macaron FFs / LayerNorms are per-frame and need
//!     no state. The relative-position term only ever sees offsets `i-j ∈ [-3,59]`
//!     inside the band, so a fixed 63-row projected table (per layer, built once —
//!     [`Encoder::rel_tables`]) replaces the offline `[2T-1]` ladder exactly;
//!   * RNN-T — [`DecodeState`] carries the LSTM prediction-net state across pushes
//!     (the same loop the offline `rnnt_greedy` runs).
//!
//! Rows pass through the blocks in complete 4-row chunks (a chunk's queries may
//! attend *within* the whole chunk, so it is the emission granularity): ~0.32 s of
//! audio per chunk of algorithmic latency. `stream_finish` flushes the tails —
//! feeding each subsampling stage the zero rows the offline mask supplies — and
//! processes the final partial chunk.
//!
//! [`Encoder::stream_push_batch`] steps N concurrent streams at once: the
//! per-frame ops (macaron FFs, LayerNorms) run over the row-concatenation of every
//! stream's ready rows — the same genuine device batching as `encode_batch` — while
//! attention and the depthwise conv stay per-stream (they mix positions).

use audio::asr_frontend::NemotronMelStream;

use crate::encoder::{DecodeState, Encoder};
use crate::reference::{layernorm, rel_pos_rows, sigmoid, silu};

/// One subsampling stage's carry: the last `k-1` input rows plus absolute row
/// counters (row-major rows of `[cin, f_in]` each).
#[derive(Default)]
struct SubStage {
    tail: Vec<f32>,
    tail_rows: usize,
    rows_in: usize,
    rows_out: usize,
}

/// One Conformer layer's carry: the attention band's projected K/V rows and the
/// causal conv's GLU rows (all row-major `[rows, hidden]`, absolute base indices).
#[derive(Default)]
struct LayerCache {
    k: Vec<f32>,
    v: Vec<f32>,
    kv_base: usize,
    kv_rows: usize,
    glu: Vec<f32>,
    glu_base: usize,
    glu_rows: usize,
}

/// The full per-stream state for frame-synchronous transcription. Create with
/// [`Encoder::stream_new`], feed with [`Encoder::stream_push`], and flush with
/// [`Encoder::stream_finish`]. Holds ~12 MB for the 0.6B model (K/V band + GLU
/// tails across 24 layers); everything heavy (weights, device) stays on the
/// shared [`Encoder`].
pub struct StreamState {
    prompt_id: usize,
    mel: NemotronMelStream,
    stages: Vec<SubStage>,
    /// Subsampled (post-linear) rows awaiting a chunk boundary, `[n, hidden]`.
    pend: Vec<f32>,
    /// Absolute subsampled row index of `pend[0]` (= rows already processed).
    pos: usize,
    layers: Vec<LayerCache>,
    dec: DecodeState,
    finished: bool,
    /// Test-only capture of every pooler row produced, for exactness gates.
    #[cfg(test)]
    pub(crate) pooler_log: Vec<f32>,
}

impl StreamState {
    /// Tokens emitted so far (across all pushes).
    pub fn tokens(&self) -> &[u32] {
        &self.dec.emitted
    }

    /// True once `stream_finish` ran; further pushes are rejected.
    pub fn finished(&self) -> bool {
        self.finished
    }
}

impl Encoder {
    /// Start a streaming transcription session with the given language prompt.
    pub fn stream_new(&self, prompt_id: usize) -> StreamState {
        let cfg = self.config();
        StreamState {
            prompt_id,
            mel: NemotronMelStream::new(),
            stages: (0..cfg.subsampling_stages()).map(|_| SubStage::default()).collect(),
            pend: Vec::new(),
            pos: 0,
            layers: (0..cfg.n_layers).map(|_| LayerCache::default()).collect(),
            dec: DecodeState::new(self, cfg.blank_token_id),
            finished: false,
            #[cfg(test)]
            pooler_log: Vec::new(),
        }
    }

    /// Push 16 kHz mono samples into one stream; returns the tokens newly emitted
    /// by this push (complete 4-row chunks only — the rest waits in the state).
    pub fn stream_push(&self, st: &mut StreamState, samples: &[f32]) -> Vec<u32> {
        let mut items = [(st, samples)];
        self.stream_push_batch(&mut items).pop().unwrap()
    }

    /// Step N concurrent streams at once. Per stream: front end + subsampling, then
    /// every complete 4-row chunk of every stream runs through the Conformer stack
    /// **together** — per-frame ops batched over the row-concatenation, attention /
    /// conv per stream against its own caches. Returns each stream's newly emitted
    /// tokens, in order. Bit-identical, per stream, to pushing alone.
    pub fn stream_push_batch(&self, items: &mut [(&mut StreamState, &[f32])]) -> Vec<Vec<u32>> {
        for (st, samples) in items.iter_mut() {
            assert!(!st.finished, "stream_push after stream_finish");
            let mel_rows = st.mel.push(samples);
            self.ingest_mel(st, &mel_rows);
        }
        let mut sts: Vec<&mut StreamState> = items.iter_mut().map(|(st, _)| &mut **st).collect();
        self.process_ready(&mut sts, false)
    }

    /// Flush one stream: emit the mel tail, feed each subsampling stage the zero
    /// rows the offline mask supplies, process the final partial chunk, and decode.
    /// Returns the tokens newly emitted by the flush. The state is consumed for
    /// pushing (`finished`), but keeps the full token history readable.
    pub fn stream_finish(&self, st: &mut StreamState) -> Vec<u32> {
        assert!(!st.finished, "stream_finish called twice");
        let (tail, valid) = st.mel.finish();
        self.ingest_mel(st, &tail);
        self.stream_finish_mel(st, valid)
    }

    /// Flush below the front end (`valid` = real mel rows) — the test seam that
    /// lets synthetic-mel streams bypass the fixed-128-bin front end.
    pub(crate) fn stream_finish_mel(&self, st: &mut StreamState, valid: usize) -> Vec<u32> {
        st.finished = true;
        if valid == 0 {
            return Vec::new(); // an empty stream has nothing to flush
        }
        // Feed each stage zeros until it has emitted exactly the offline stage
        // length for its valid input count — the rows the offline mask zeroes.
        let cfg = *self.config();
        let (k, s) = (cfg.subsampling_kernel as usize, cfg.subsampling_stride as usize);
        let mut vin = valid;
        for stage in 0..st.stages.len() {
            let target = (vin + (k - 1) + (s - 1) - k) / s + 1; // offline stage_len
            let need_in = s * (target - 1) + 1;
            let have = st.stages[stage].rows_in;
            if need_in > have {
                let f_in = self.stage_dims(stage).1;
                let cin = self.stage_dims(stage).0;
                let zeros = vec![0.0f32; (need_in - have) * cin * f_in];
                self.sub_feed_from(st, stage, &zeros, need_in - have);
            }
            debug_assert_eq!(st.stages[stage].rows_out, target, "stage {stage} flush");
            vin = target;
        }
        let mut sts = [st];
        self.process_ready(&mut sts, true).pop().unwrap()
    }

    /// `(cin, f_in)` of subsampling stage `stage`'s input rows.
    fn stage_dims(&self, stage: usize) -> (usize, usize) {
        let cfg = self.config();
        let (k, s) = (cfg.subsampling_kernel as usize, cfg.subsampling_stride as usize);
        let mut f = cfg.num_mel_bins as usize;
        for _ in 0..stage {
            f = (f + (k - 1) + (s - 1) - k) / s + 1;
        }
        if stage == 0 {
            (1, f)
        } else {
            (cfg.subsampling_channels as usize, f)
        }
    }

    /// Push complete mel rows (`[n, num_mel]`) through the subsampling cascade into
    /// `pend`. Test seam for synthetic-mel streams.
    pub(crate) fn ingest_mel(&self, st: &mut StreamState, rows: &[f32]) {
        let n = rows.len() / self.config().num_mel_bins.max(1) as usize;
        if n > 0 {
            self.sub_feed_from(st, 0, rows, n);
        }
    }

    /// Mel-level push (test seam): ingest complete mel rows and process every
    /// complete chunk, returning the newly emitted tokens.
    #[cfg(test)]
    pub(crate) fn stream_push_mel(&self, st: &mut StreamState, rows: &[f32]) -> Vec<u32> {
        assert!(!st.finished, "stream_push after stream_finish");
        self.ingest_mel(st, rows);
        let mut sts = [st];
        self.process_ready(&mut sts, false).pop().unwrap()
    }

    /// Feed `n` input rows into stage `from`, cascading every newly available row
    /// through the remaining stages and the final linear into `pend`.
    fn sub_feed_from(&self, st: &mut StreamState, from: usize, rows: &[f32], n: usize) {
        let cfg = *self.config();
        let mut cur = rows.to_vec();
        let mut cn = n;
        for stage in from..st.stages.len() {
            if cn == 0 {
                return;
            }
            (cur, cn) = self.sub_stage_feed(&mut st.stages[stage], stage, &cur, cn);
        }
        if cn == 0 {
            return;
        }
        // final linear: rows are [cn, ch*f_out] already in [c][f] order per row
        let flat = cfg.subsampling_out_hidden() as usize;
        let hidden = cfg.hidden as usize;
        let mut lin = self.mm(&cur, "encoder.subsampling.linear.weight", cn, flat, hidden);
        let lb = self.rw("encoder.subsampling.linear.bias");
        for r in 0..cn {
            for j in 0..hidden {
                lin[r * hidden + j] += lb[j];
            }
        }
        st.pend.extend_from_slice(&lin);
    }

    /// One causal stride-2 stage over `n` new input rows (row-major `[n, cin*f_in]`).
    /// Returns the newly computable output rows (row-major `[m, ch*f_out]`) — a row
    /// is emitted the moment input row `s*o` exists, exactly the offline window (the
    /// conv kernel sees the identical `k×k` patches, so values are bit-identical).
    fn sub_stage_feed(&self, ss: &mut SubStage, stage: usize, new_rows: &[f32], n: usize) -> (Vec<f32>, usize) {
        let cfg = *self.config();
        let (k, s) = (cfg.subsampling_kernel as usize, cfg.subsampling_stride as usize);
        let (cin, f_in) = self.stage_dims(stage);
        let f_out = (f_in + (k - 1) + (s - 1) - k) / s + 1;
        let ch = cfg.subsampling_channels as usize;
        let row_len = cin * f_in;
        let rows_in_new = ss.rows_in + n;
        // outputs available once input row s*o exists: o <= (rows_in_new-1)/s
        let avail_out = if rows_in_new == 0 { 0 } else { (rows_in_new - 1) / s + 1 };
        let m = avail_out - ss.rows_out;
        if m == 0 {
            Self::keep_tail(ss, new_rows, n, row_len, k - 1);
            ss.rows_in = rows_in_new;
            return (Vec::new(), 0);
        }
        // slab of input rows [s*rows_out - (k-1) .. s*(avail_out-1)] (zeros before row 0),
        // uploaded NCHW [1, cin, h_slab, f_in+ (k-1)+(s-1) freq pad], conv with pad=0.
        let first = (s * ss.rows_out) as i64 - (k - 1) as i64;
        let last = (s * (avail_out - 1)) as i64;
        let h_slab = (last - first + 1) as usize;
        let fp = f_in + (k - 1) + (s - 1);
        let mut slab = vec![0.0f32; cin * h_slab * fp];
        let tail_start = ss.rows_in - ss.tail_rows;
        for h in 0..h_slab {
            let abs = first + h as i64;
            if abs < 0 {
                continue; // causal top pad
            }
            let abs = abs as usize;
            let row = if abs < ss.rows_in {
                debug_assert!(abs >= tail_start, "stage {stage}: row {abs} dropped (tail from {tail_start})");
                &ss.tail[(abs - tail_start) * row_len..(abs - tail_start + 1) * row_len]
            } else {
                &new_rows[(abs - ss.rows_in) * row_len..(abs - ss.rows_in + 1) * row_len]
            };
            // row-major [cin, f_in] -> NCHW slab [cin][h][fp] with (k-1) left freq pad
            for c in 0..cin {
                let dst = (c * h_slab + h) * fp + (k - 1);
                slab[dst..dst + f_in].copy_from_slice(&row[c * f_in..(c + 1) * f_in]);
            }
        }
        // device conv (+ bias); stem is dense 1->ch, stages are depthwise ch groups
        let (wname, bname, groups) = if stage == 0 {
            ("encoder.subsampling.conv_in.weight".to_string(), "encoder.subsampling.conv_in.bias".to_string(), 1)
        } else {
            (
                format!("encoder.subsampling.layers.{}.depthwise_conv.weight", stage - 1),
                format!("encoder.subsampling.layers.{}.depthwise_conv.bias", stage - 1),
                ch as u32,
            )
        };
        let mut y = self.conv_slab(&slab, cin, h_slab, fp, ch, &wname, &bname, groups, m, f_out);
        if stage > 0 {
            // pointwise 1x1 (+ bias) on the NCHW slab
            y = self.pointwise_slab(
                &y,
                ch,
                m,
                f_out,
                &format!("encoder.subsampling.layers.{}.pointwise_conv.weight", stage - 1),
                &format!("encoder.subsampling.layers.{}.pointwise_conv.bias", stage - 1),
            );
        }
        for v in &mut y {
            *v = v.max(0.0); // relu
        }
        // NCHW [ch, m, f_out] -> row-major [m, ch*f_out]
        let mut out = vec![0.0f32; m * ch * f_out];
        for c in 0..ch {
            for r in 0..m {
                let src = (c * m + r) * f_out;
                let dst = r * ch * f_out + c * f_out;
                out[dst..dst + f_out].copy_from_slice(&y[src..src + f_out]);
            }
        }
        Self::keep_tail(ss, new_rows, n, row_len, k - 1);
        ss.rows_in = rows_in_new;
        ss.rows_out = avail_out;
        (out, m)
    }

    /// Retain the last `keep` input rows across `tail ++ new_rows` as the new tail.
    fn keep_tail(ss: &mut SubStage, new_rows: &[f32], n: usize, row_len: usize, keep: usize) {
        let mut rows: Vec<f32> = Vec::with_capacity((ss.tail_rows + n) * row_len);
        rows.extend_from_slice(&ss.tail[..ss.tail_rows * row_len]);
        rows.extend_from_slice(&new_rows[..n * row_len]);
        let total = ss.tail_rows + n;
        let keep = keep.min(total);
        ss.tail = rows[(total - keep) * row_len..].to_vec();
        ss.tail_rows = keep;
    }

    /// Process every stream's ready rows (complete 4-row chunks; everything when
    /// `flush`) through the Conformer stack, projectors and decoder.
    fn process_ready(&self, sts: &mut [&mut StreamState], flush: bool) -> Vec<Vec<u32>> {
        let cfg = *self.config();
        let c = cfg.hidden as usize;
        let chunk = cfg.default_lookahead as usize + 1;
        // spans: (row offset, rows, absolute start, stream index)
        let mut spans: Vec<(usize, usize, usize, usize)> = Vec::new();
        let mut h: Vec<f32> = Vec::new();
        let mut offset = 0usize;
        for (idx, st) in sts.iter().enumerate() {
            let pend_rows = st.pend.len() / c;
            let r = if flush { pend_rows } else { pend_rows / chunk * chunk };
            if r > 0 {
                spans.push((offset, r, st.pos, idx));
                h.extend_from_slice(&st.pend[..r * c]);
                offset += r;
            }
        }
        let tt = offset;
        if tt == 0 {
            return sts.iter().map(|_| Vec::new()).collect();
        }
        for b in 0..cfg.n_layers {
            h = self.block_stream_batch(&h, b, &spans, sts, tt);
        }
        // projectors + decode per stream (per-frame math; prompt id is per stream)
        let mut out: Vec<Vec<u32>> = sts.iter().map(|_| Vec::new()).collect();
        for &(off, r, _abs, idx) in &spans {
            let st = &mut *sts[idx];
            let pooler = self.project_rows(&h[off * c..(off + r) * c], r, st.prompt_id);
            #[cfg(test)]
            st.pooler_log.extend_from_slice(&pooler);
            let before = st.dec.emitted.len();
            st.dec.step_frames(self, &pooler, r);
            out[idx] = st.dec.emitted[before..].to_vec();
            st.pend.drain(..r * c);
            st.pos += r;
        }
        out
    }

    /// One Conformer block over the batched ready rows — the streaming sibling of
    /// `block_dev_batch`: per-frame ops (macaron FFs, LayerNorms) run once over all
    /// `tt` rows; attention and the causal conv run per span against that stream's
    /// caches. Row-wise identical to the offline block, so single-utterance streams
    /// reproduce the offline pooler bit-for-bit.
    fn block_stream_batch(&self, h: &[f32], b: u32, spans: &[(usize, usize, usize, usize)], sts: &mut [&mut StreamState], tt: usize) -> Vec<f32> {
        let cfg = *self.config();
        let c = cfg.hidden as usize;
        let pre = format!("encoder.layers.{b}");
        let ln = |x: &[f32], n: &str| layernorm(x, self.rw(&format!("{pre}.{n}.weight")), self.rw(&format!("{pre}.{n}.bias")), tt, c, cfg.ln_eps);
        let mut h = h.to_vec();
        // macaron FF1 — batched over every row
        let ff1 = self.ff_dev(&ln(&h, "norm_feed_forward1"), &format!("{pre}.feed_forward1"), tt);
        for i in 0..tt * c {
            h[i] += 0.5 * ff1[i];
        }
        // banded self-attention — per span, against the K/V cache
        let hn = ln(&h, "norm_self_att");
        for &(off, n, abs, idx) in spans {
            let att = self.attn_stream(&hn[off * c..(off + n) * c], b, n, abs, &mut sts[idx].layers[b as usize]);
            for i in 0..n * c {
                h[off * c + i] += att[i];
            }
        }
        // causal conv module — per span, against the GLU cache
        let hn = ln(&h, "norm_conv");
        for &(off, n, abs, idx) in spans {
            let cv = self.conv_stream(&hn[off * c..(off + n) * c], b, n, abs, &mut sts[idx].layers[b as usize]);
            for i in 0..n * c {
                h[off * c + i] += cv[i];
            }
        }
        // macaron FF2 — batched over every row
        let ff2 = self.ff_dev(&ln(&h, "norm_feed_forward2"), &format!("{pre}.feed_forward2"), tt);
        for i in 0..tt * c {
            h[i] += 0.5 * ff2[i];
        }
        ln(&h, "norm_out")
    }

    /// Per-layer relative-position tables for the `chunked_limited` band, projected
    /// through each layer's `relative_k_proj` — built once, lazily. A query at the
    /// *end* of its chunk reaches `left_chunks·chunk + (chunk-1)` rows back and one
    /// at the *start* sees `chunk-1` rows ahead, so offsets span
    /// `i-j ∈ [-(chunk-1), left_chunks·chunk + chunk-1]` (`[-3, 59]` for this
    /// model). Row `r` holds position value `max_off - r`, matching the offline
    /// ladder's rows for the same values bit-for-bit (`rel_pos_rows` is shared).
    pub(crate) fn rel_tables(&self) -> &Vec<Vec<f32>> {
        self.rel_band.get_or_init(|| {
            let cfg = self.config();
            let c = cfg.hidden as usize;
            let (max_off, min_off) = self.band_offsets();
            let bw = (max_off - min_off + 1) as usize;
            let pos: Vec<f32> = (0..bw).map(|r| (max_off - r as i64) as f32).collect();
            let pe = rel_pos_rows(&pos, c);
            (0..cfg.n_layers)
                .map(|b| self.mm(&pe, &format!("encoder.layers.{b}.self_attn.relative_k_proj.weight"), bw, c, c))
                .collect()
        })
    }

    /// `(max, min)` relative offsets `i-j` reachable inside the attention band.
    fn band_offsets(&self) -> (i64, i64) {
        let cfg = self.config();
        let left = (cfg.sliding_window - 1) as i64;
        let chunk = cfg.default_lookahead as i64 + 1;
        ((left / chunk) * chunk + chunk - 1, -(chunk - 1))
    }

    /// Streaming banded rel-pos attention for `n` new rows starting at absolute row
    /// `abs`: Q from the new rows, K/V from `cache ++ new`, scores over the
    /// `chunked_limited` band only. Score-for-score the offline computation (the
    /// out-of-band entries it sums are exact zeros), so the context is bit-identical.
    fn attn_stream(&self, hn: &[f32], b: u32, n: usize, abs: usize, cache: &mut LayerCache) -> Vec<f32> {
        let cfg = *self.config();
        let pre = format!("encoder.layers.{b}.self_attn");
        let (c, heads, hd) = (cfg.hidden as usize, cfg.n_heads as usize, cfg.head_dim() as usize);
        let scale = 1.0 / (hd as f32).sqrt();
        let (left, right) = ((cfg.sliding_window - 1) as usize, cfg.default_lookahead as usize);
        let chunk = right + 1;
        let left_chunks = left / chunk;
        let q = self.mm(hn, &format!("{pre}.q_proj.weight"), n, c, c);
        let k_new = self.mm(hn, &format!("{pre}.k_proj.weight"), n, c, c);
        let v_new = self.mm(hn, &format!("{pre}.v_proj.weight"), n, c, c);
        debug_assert_eq!(cache.kv_base + cache.kv_rows, abs, "K/V cache is contiguous with the new rows");
        // combined K/V rows [kv_base .. abs+n)
        let mut kall = Vec::with_capacity((cache.kv_rows + n) * c);
        kall.extend_from_slice(&cache.k[..cache.kv_rows * c]);
        kall.extend_from_slice(&k_new);
        let mut vall = Vec::with_capacity((cache.kv_rows + n) * c);
        vall.extend_from_slice(&cache.v[..cache.kv_rows * c]);
        vall.extend_from_slice(&v_new);
        let kv_base = cache.kv_base;
        let kv_end = abs + n;
        let (max_off, _) = self.band_offsets();
        let rel = &self.rel_tables()[b as usize];
        let bu = self.rw(&format!("{pre}.bias_u"));
        let bv = self.rw(&format!("{pre}.bias_v"));
        let (q, kall, vall) = (&q, &kall, &vall);
        let head_ctx: Vec<Vec<f32>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..heads)
                .map(|hh| {
                    s.spawn(move || {
                        let (bus, bvs) = (&bu[hh * hd..hh * hd + hd], &bv[hh * hd..hh * hd + hd]);
                        let mut out = vec![0.0f32; n * hd];
                        let mut sc = Vec::new();
                        for qi in 0..n {
                            let i_abs = abs + qi;
                            let qc = i_abs / chunk;
                            let j_lo = qc.saturating_sub(left_chunks) * chunk;
                            debug_assert!(j_lo >= kv_base, "attention band exceeds the K/V cache");
                            let j_hi = ((qc + 1) * chunk).min(kv_end);
                            sc.clear();
                            sc.resize(j_hi - j_lo, 0.0f32);
                            for j in j_lo..j_hi {
                                let kr = &kall[(j - kv_base) * c + hh * hd..(j - kv_base) * c + hh * hd + hd];
                                let rr = &rel[(max_off - (i_abs as i64 - j as i64)) as usize * c + hh * hd..];
                                let (mut ac, mut bd) = (0.0f32, 0.0f32);
                                for d in 0..hd {
                                    ac += (q[qi * c + hh * hd + d] + bus[d]) * kr[d];
                                    bd += (q[qi * c + hh * hd + d] + bvs[d]) * rr[d];
                                }
                                sc[j - j_lo] = ac * scale + bd * scale;
                            }
                            let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let mut den = 0.0f32;
                            for sv in sc.iter_mut() {
                                *sv = (*sv - mx).exp();
                                den += *sv;
                            }
                            let inv = if den > 0.0 { 1.0 / den } else { 0.0 };
                            for d in 0..hd {
                                let mut acc = 0.0f32;
                                for j in j_lo..j_hi {
                                    acc += sc[j - j_lo] * vall[(j - kv_base) * c + hh * hd + d];
                                }
                                out[qi * hd + d] = acc * inv;
                            }
                        }
                        out
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut ctx = vec![0.0f32; n * c];
        for (hh, hc) in head_ctx.iter().enumerate() {
            for i in 0..n {
                ctx[i * c + hh * hd..i * c + hh * hd + hd].copy_from_slice(&hc[i * hd..i * hd + hd]);
            }
        }
        // retain the band still reachable from future queries (14 chunks + current)
        let keep_from = (kv_end / chunk).saturating_sub(left_chunks) * chunk;
        let keep_from = keep_from.max(kv_base);
        let mut knext = kall[(keep_from - kv_base) * c..].to_vec();
        let mut vnext = vall[(keep_from - kv_base) * c..].to_vec();
        std::mem::swap(&mut cache.k, &mut knext);
        std::mem::swap(&mut cache.v, &mut vnext);
        cache.kv_base = keep_from;
        cache.kv_rows = kv_end - keep_from;
        self.mm(&ctx, &format!("{pre}.o_proj.weight"), n, c, c)
    }

    /// Streaming Conformer conv module for `n` new rows at absolute row `abs`: the
    /// causal depthwise conv reads its `k-1` past GLU rows from the cache. Same
    /// accumulation order as the offline module (`kk` ascending, `src >= 0` only).
    fn conv_stream(&self, hn: &[f32], b: u32, n: usize, abs: usize, cache: &mut LayerCache) -> Vec<f32> {
        let cfg = *self.config();
        let pre = format!("encoder.layers.{b}.conv");
        let (c, k) = (cfg.hidden as usize, cfg.conv_kernel as usize);
        let pc1 = self.mm(hn, &format!("{pre}.pointwise_conv1.weight"), n, c, 2 * c);
        let mut glu_new = vec![0.0f32; n * c];
        for i in 0..n {
            for j in 0..c {
                glu_new[i * c + j] = pc1[i * 2 * c + j] * sigmoid(pc1[i * 2 * c + c + j]);
            }
        }
        let dw = self.rw(&format!("{pre}.depthwise_conv.weight"));
        let mut conv = vec![0.0f32; n * c];
        for ch in 0..c {
            for i in 0..n {
                let i_abs = (abs + i) as i64;
                let mut acc = 0.0f32;
                for kk in 0..k {
                    let src = i_abs - (k as i64 - 1) + kk as i64;
                    if src >= 0 {
                        let src = src as usize;
                        let val = if src < abs { cache.glu[(src - cache.glu_base) * c + ch] } else { glu_new[(src - abs) * c + ch] };
                        acc += val * dw[ch * k + kk];
                    }
                }
                conv[i * c + ch] = acc;
            }
        }
        // retain the last k-1 GLU rows across cache ++ new
        let total_from = cache.glu_base;
        let total_rows = cache.glu_rows + n;
        let keep = (k - 1).min(total_rows);
        let mut all = Vec::with_capacity(total_rows * c);
        all.extend_from_slice(&cache.glu[..cache.glu_rows * c]);
        all.extend_from_slice(&glu_new);
        cache.glu = all[(total_rows - keep) * c..].to_vec();
        cache.glu_base = total_from + total_rows - keep;
        cache.glu_rows = keep;

        let mut act = layernorm(&conv, self.rw(&format!("{pre}.norm.weight")), self.rw(&format!("{pre}.norm.bias")), n, c, cfg.ln_eps);
        for v in &mut act {
            *v = silu(*v);
        }
        self.mm(&act, &format!("{pre}.pointwise_conv2.weight"), n, c, c)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use gpu_core::Gpu;

    use crate::config::NemotronConfig;
    use crate::encoder::{encoder_pipelines, Encoder};

    /// A tiny random-weight config exercising every architectural feature (3
    /// subsampling stages, 2 Conformer layers, banded attention, RNN-T) at a size
    /// that runs in milliseconds — streaming vs offline is a *structural* identity,
    /// so it holds for any weights.
    fn tiny_cfg() -> NemotronConfig {
        NemotronConfig {
            num_mel_bins: 16,
            hidden: 32,
            n_layers: 2,
            n_heads: 2,
            intermediate: 48,
            conv_kernel: 9,
            subsampling_factor: 8,
            subsampling_channels: 8,
            subsampling_kernel: 3,
            subsampling_stride: 2,
            sliding_window: 57,
            default_lookahead: 3,
            ln_eps: 1e-5,
            decoder_hidden: 24,
            num_decoder_layers: 2,
            vocab: 40,
            blank_token_id: 39,
            max_symbols_per_step: 10,
            num_prompts: 8,
            prompt_intermediate: 16,
            default_prompt_id: 0,
        }
    }

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        }
        fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n).map(|_| self.f() * scale).collect()
        }
    }

    fn tiny_weights(cfg: &NemotronConfig) -> HashMap<String, Vec<f32>> {
        let mut r = Lcg(0xB5AD4ECEDA1CE2A9);
        let (c, ch, ffn) = (cfg.hidden as usize, cfg.subsampling_channels as usize, cfg.intermediate as usize);
        let (dh, np, pi, v) = (cfg.decoder_hidden as usize, cfg.num_prompts as usize, cfg.prompt_intermediate as usize, cfg.vocab as usize);
        let k = cfg.conv_kernel as usize;
        let flat = cfg.subsampling_out_hidden() as usize;
        let mut w = HashMap::new();
        let mut put = |name: String, data: Vec<f32>| {
            w.insert(name, data);
        };
        put("encoder.subsampling.conv_in.weight".into(), r.vec(ch * 1 * 9, 0.3));
        put("encoder.subsampling.conv_in.bias".into(), r.vec(ch, 0.1));
        for i in 0..2 {
            put(format!("encoder.subsampling.layers.{i}.depthwise_conv.weight"), r.vec(ch * 9, 0.3));
            put(format!("encoder.subsampling.layers.{i}.depthwise_conv.bias"), r.vec(ch, 0.1));
            put(format!("encoder.subsampling.layers.{i}.pointwise_conv.weight"), r.vec(ch * ch, 0.3));
            put(format!("encoder.subsampling.layers.{i}.pointwise_conv.bias"), r.vec(ch, 0.1));
        }
        put("encoder.subsampling.linear.weight".into(), r.vec(c * flat, 0.2));
        put("encoder.subsampling.linear.bias".into(), r.vec(c, 0.1));
        for b in 0..cfg.n_layers {
            let pre = format!("encoder.layers.{b}");
            for n in ["norm_feed_forward1", "norm_feed_forward2", "norm_self_att", "norm_conv", "norm_out"] {
                put(format!("{pre}.{n}.weight"), vec![1.0; c]);
                put(format!("{pre}.{n}.bias"), r.vec(c, 0.05));
            }
            for f in ["feed_forward1", "feed_forward2"] {
                put(format!("{pre}.{f}.linear1.weight"), r.vec(ffn * c, 0.2));
                put(format!("{pre}.{f}.linear2.weight"), r.vec(c * ffn, 0.2));
            }
            for p in ["q_proj", "k_proj", "v_proj", "o_proj", "relative_k_proj"] {
                put(format!("{pre}.self_attn.{p}.weight"), r.vec(c * c, 0.2));
            }
            put(format!("{pre}.self_attn.bias_u"), r.vec(c, 0.1));
            put(format!("{pre}.self_attn.bias_v"), r.vec(c, 0.1));
            put(format!("{pre}.conv.pointwise_conv1.weight"), r.vec(2 * c * c, 0.2));
            put(format!("{pre}.conv.depthwise_conv.weight"), r.vec(c * k, 0.2));
            put(format!("{pre}.conv.norm.weight"), vec![1.0; c]);
            put(format!("{pre}.conv.norm.bias"), r.vec(c, 0.05));
            put(format!("{pre}.conv.pointwise_conv2.weight"), r.vec(c * c, 0.2));
        }
        put("prompt_projector.linear_1.weight".into(), r.vec(pi * (c + np), 0.2));
        put("prompt_projector.linear_1.bias".into(), r.vec(pi, 0.1));
        put("prompt_projector.linear_2.weight".into(), r.vec(c * pi, 0.2));
        put("prompt_projector.linear_2.bias".into(), r.vec(c, 0.1));
        put("encoder_projector.weight".into(), r.vec(dh * c, 0.2));
        put("encoder_projector.bias".into(), r.vec(dh, 0.1));
        put("joint.head.weight".into(), r.vec(v * dh, 0.3));
        put("joint.head.bias".into(), r.vec(v, 0.1));
        put("decoder.embedding.weight".into(), r.vec(v * dh, 0.3));
        for l in 0..2 {
            put(format!("decoder.lstm.weight_ih_l{l}"), r.vec(4 * dh * dh, 0.2));
            put(format!("decoder.lstm.weight_hh_l{l}"), r.vec(4 * dh * dh, 0.2));
            put(format!("decoder.lstm.bias_ih_l{l}"), r.vec(4 * dh, 0.1));
            put(format!("decoder.lstm.bias_hh_l{l}"), r.vec(4 * dh, 0.1));
        }
        put("decoder.decoder_projector.weight".into(), r.vec(dh * dh, 0.2));
        put("decoder.decoder_projector.bias".into(), r.vec(dh, 0.1));
        w
    }

    /// Streaming (ragged mel pushes + flush) must equal the offline whole-utterance
    /// forward: the token sequence exactly, and every pooler value bit-for-bit under
    /// BRAIN_NO_FASTCONV=1 (shape-invariant kernels) / within ulp-tolerance otherwise.
    #[test]
    fn streaming_matches_offline() {
        let cfg = tiny_cfg();
        let w = tiny_weights(&cfg);
        let g = Gpu::new_cpu(encoder_pipelines());
        let enc = Encoder::new(g, cfg, &w);
        let nm = cfg.num_mel_bins as usize;
        let dh = cfg.decoder_hidden as usize;

        // 521 mel frames -> 66 subsampled rows: an odd length that exercises the
        // partial-chunk flush, the full [-3,59] band-offset range AND K/V trimming
        let t = 521usize;
        let mut r = Lcg(0x5EED);
        let mel = r.vec(t * nm, 1.0);

        let (pooler, valid) = enc.encode(&mel, t as u32, t as u32, 0);
        let tokens = enc.rnnt_greedy(&pooler, valid as usize);

        let mut st = enc.stream_new(0);
        let mut got = Vec::new();
        // ragged pushes: 1, 7, 32, 3, 64, ... rows at a time
        let sizes = [1usize, 7, 32, 3, 64, 11, 2, 40];
        let (mut i, mut si) = (0usize, 0usize);
        while i < t {
            let n = sizes[si % sizes.len()].min(t - i);
            got.extend(enc.stream_push_mel(&mut st, &mel[i * nm..(i + n) * nm]));
            i += n;
            si += 1;
        }
        got.extend(enc.stream_finish_mel(&mut st, t));

        assert_eq!(st.pooler_log.len(), valid as usize * dh, "streamed pooler row count");
        // With BRAIN_NO_FASTCONV=1 the backend's kernel choice is shape-invariant and
        // the poolers are bit-identical; with the AVX2 conv fast path on, its
        // documented "<=1 ulp reassociation" applies per slab shape, so streamed
        // slabs may differ from the offline whole-utterance slab by a few ulp.
        let exact = std::env::var("BRAIN_NO_FASTCONV").map(|v| v != "0").unwrap_or(false);
        let tol = if exact { 0.0 } else { 1e-5 };
        let mut maxd = 0.0f32;
        for (idx, (a, b)) in st.pooler_log.iter().zip(&pooler[..valid as usize * dh]).enumerate() {
            let d = (a - b).abs();
            maxd = maxd.max(d);
            assert!(d <= tol, "pooler[{}][{}]: stream {a} vs offline {b} (tol {tol})", idx / dh, idx % dh);
        }
        eprintln!("pooler maxdiff {maxd} (exact mode: {exact})");
        assert!(!tokens.is_empty(), "tiny model should emit something");
        assert_eq!(got, tokens, "streamed tokens == offline tokens");
        assert_eq!(st.tokens(), &tokens[..], "state history matches");
    }

    /// Batched streaming (two concurrent streams stepped together) must equal each
    /// stream pushed alone — the invariant that makes concurrent serving safe.
    #[test]
    fn batched_streaming_matches_single() {
        let cfg = tiny_cfg();
        let w = tiny_weights(&cfg);
        let g = Gpu::new_cpu(encoder_pipelines());
        let enc = Encoder::new(g, cfg, &w);
        let nm = cfg.num_mel_bins as usize;

        let mut r = Lcg(0xABCD);
        let mel_a = r.vec(96 * nm, 1.0);
        let mel_b = r.vec(57 * nm, 1.0);

        // singles
        let mut sa = enc.stream_new(0);
        let mut ta = enc.stream_push_mel(&mut sa, &mel_a);
        ta.extend(enc.stream_finish_mel(&mut sa, 96));
        let mut sb = enc.stream_new(1);
        let mut tb = enc.stream_push_mel(&mut sb, &mel_b);
        tb.extend(enc.stream_finish_mel(&mut sb, 57));

        // batched: ingest both, process together, then flush each
        let mut ba = enc.stream_new(0);
        let mut bb = enc.stream_new(1);
        enc.ingest_mel(&mut ba, &mel_a);
        enc.ingest_mel(&mut bb, &mel_b);
        let mut sts = [&mut ba, &mut bb];
        let out = enc.process_ready(&mut sts, false);
        let mut ga = out[0].clone();
        let mut gb = out[1].clone();
        ga.extend(enc.stream_finish_mel(&mut ba, 96));
        gb.extend(enc.stream_finish_mel(&mut bb, 57));

        assert_eq!(ga, ta, "stream A batched == single");
        assert_eq!(gb, tb, "stream B batched == single");
        assert_eq!(ba.pooler_log, sa.pooler_log, "stream A pooler bit-identical");
        assert_eq!(bb.pooler_log, sb.pooler_log, "stream B pooler bit-identical");
    }

    /// End-to-end with the real 0.6B checkpoint: pushing the LibriSpeech clip in
    /// irregular sample-sized chunks must reproduce the offline `transcribe` token
    /// stream exactly. Heavy (loads the checkpoint); run explicitly:
    /// `cargo test -p brain-nemotron --release streaming_e2e -- --ignored`.
    #[test]
    #[ignore = "loads the 0.6B checkpoint (run explicitly)"]
    fn streaming_e2e_matches_offline_transcribe() {
        use std::path::Path;
        let ckpt = crate::model_dir("nvidia/nemotron-3.5-asr-streaming-0.6b").unwrap_or_default();
        let wav_path = crate::testdata("asr/audio/librispeech_mr_quilter.wav");
        if !Path::new(&wav_path).exists() || !Path::new(&format!("{ckpt}/model.safetensors")).exists() {
            eprintln!("skipping: assets absent (run `make fetch/testdata`)");
            return;
        }
        let cfg = NemotronConfig::nemotron_3_5_asr_0_6b();
        let wav = audio::wav::read(&wav_path).expect("wav");
        let w = crate::import::load_tensors(Path::new(&ckpt)).expect("load");
        let g = Gpu::new_cpu(encoder_pipelines());
        let enc = Encoder::new(g, cfg, &w);

        let offline = enc.transcribe(&wav.samples, 0);

        let mut st = enc.stream_new(0);
        let mut got = Vec::new();
        // ~120 ms pushes with ragged sizes, like a mic delivers
        let sizes = [1600usize, 2048, 640, 3200, 1919];
        let (mut i, mut si) = (0usize, 0usize);
        let t0 = std::time::Instant::now();
        let mut first_tok_at = None;
        while i < wav.samples.len() {
            let n = sizes[si % sizes.len()].min(wav.samples.len() - i);
            let new = enc.stream_push(&mut st, &wav.samples[i..i + n]);
            if first_tok_at.is_none() && !new.is_empty() {
                first_tok_at = Some((i + n, t0.elapsed()));
            }
            got.extend(new);
            i += n;
            si += 1;
        }
        got.extend(enc.stream_finish(&mut st));
        let dt = t0.elapsed();
        let audio_s = wav.samples.len() as f32 / 16000.0;
        eprintln!("streaming: {} tokens in {:?} (audio {audio_s:.2}s, RTF {:.3}), first token after {:?}", got.len(), dt, dt.as_secs_f32() / audio_s, first_tok_at);
        assert_eq!(got, offline, "streamed tokens == offline transcribe");
    }
}
