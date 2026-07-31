// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LFM2.5-Encoder behind the residency [`Executor`] — what makes the generic
//! D-Bus interface (`Run`/`Subscribe`, fd blobs) and the scheduler serve the
//! encoder with true concurrent batching.
//!
//! Concurrency shape (an encoder is one-shot — no KV continuation):
//! - **Tokenization runs on the dispatcher** (inside `instance_key`, off the
//!   device lane), so lane time is pure forward — the cheap pipelining that
//!   actually matters here; the heavy stage is one GPU forward.
//! - The instance key is the **exact token length**: equal-length requests
//!   share one built graph and batch into a TRUE batched forward (`b` slots,
//!   the tail chunk padded by REPEATING the last sequence, YOLO-style —
//!   sound under bidirectional attention, unlike token padding, which is NOT
//!   and is deliberately absent until zeroed-pad-states + kmask land).
//! - Different lengths build different instances; the LRU/budget machinery
//!   swaps them like any other resident.

use std::sync::Arc;

use capability::{ActionResult, Invocation, Manifest, Progress};
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

use lfm::model::Lfm;

/// Attention-slab budget per forward (chunk 2048 at T=8192, H=16).
const SLAB_BUDGET: u64 = 512 << 20;
/// Probe capacity for fill_mask (masks per request).
const PROBE_CAP: u32 = 64;

pub struct LfmResident {
    weights: String,
    tokenizer_path: String,
    tok: Arc<QwenBpe>,
    /// Batched-forward slots per instance (`BRAIN_LFM_BATCH`, default 2).
    batch: u32,
    /// Bytes of the checkpoint (device weight footprint ≈ file size, fp32).
    weight_bytes: u64,
}

impl LfmResident {
    pub fn from_env() -> Option<LfmResident> {
        let weights = std::env::var("BRAIN_LFM").ok().filter(|p| !p.is_empty())?;
        let tokenizer = std::env::var("BRAIN_LFM_TOKENIZER").ok().filter(|p| !p.is_empty())?;
        match Self::new(&weights, &tokenizer) {
            Ok(r) => Some(r),
            Err(e) => {
                eprintln!("brain: lfm not served over the scheduler ({e})");
                None
            }
        }
    }

    /// Explicit-path constructor (the perf harness builds it directly).
    pub fn new(weights: &str, tokenizer_path: &str) -> Result<LfmResident, String> {
        let tok = Arc::new(QwenBpe::from_file(tokenizer_path)?);
        let batch = std::env::var("BRAIN_LFM_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(2u32).max(1);
        let weight_bytes = std::fs::metadata(weights).map(|m| m.len()).unwrap_or(1 << 30);
        Ok(LfmResident {
            weights: weights.to_string(),
            tokenizer_path: tokenizer_path.to_string(),
            tok,
            batch,
            weight_bytes,
        })
    }

    /// Tokenize an invocation exactly as the instance will (template prefix +
    /// text param or text blob). Errors surface at run time, not key time.
    fn ids_of(&self, inv: &Invocation) -> Option<Vec<u32>> {
        let text = match inv.get_str("text").filter(|t| !t.is_empty()) {
            Some(t) => t,
            None => String::from_utf8(inv.get_blob("text")?.bytes.clone()).ok()?,
        };
        let mut ids: Vec<u32> = self.tok.template_prefix().to_vec();
        ids.extend(self.tok.encode(&text));
        let max_tokens = inv.get_i64("max_tokens").unwrap_or(0);
        if max_tokens > 0 {
            ids.truncate(max_tokens as usize);
        }
        (!ids.is_empty()).then_some(ids)
    }
}

impl ResidentModel for LfmResident {
    fn manifest(&self) -> Manifest {
        lfm::caps::manifest_resident()
    }

    fn instance_key(&self, _action: &str, inv: &Invocation) -> InstanceKey {
        // Tokenize on the dispatcher; carry the length in the key so only
        // equal-length jobs group. Un-tokenizable requests share a key and
        // fail cleanly inside run().
        let len = self.ids_of(inv).map(|v| v.len()).unwrap_or(0);
        InstanceKey::new(lfm::caps::MODEL, format!("t{len}"))
    }

    fn estimate(&self, key: &InstanceKey) -> MemCost {
        let t = key.config.strip_prefix('t').and_then(|s| s.parse::<u64>().ok()).unwrap_or(1024);
        let n = t * self.batch as u64;
        // Weights + the chunked regime's working set (fused qkv ~3nd, FFN ~3nf,
        // scratch ~12nd, slabs ≤ 2×SLAB_BUDGET) — a deliberate over-estimate so
        // the budget arithmetic errs toward eviction, never OOM.
        let act = n * 1024 * 4 * 30 + 2 * SLAB_BUDGET.min(1 << 30);
        MemCost::new(self.weight_bytes + act, 0)
    }

    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        let t: u32 = key
            .config
            .strip_prefix('t')
            .and_then(|s| s.parse().ok())
            .filter(|&t| t > 0)
            .ok_or_else(|| format!("lfm: bad instance key '{}' (empty/untokenizable input)", key.config))?;
        // Stream weights from the mmap: peak host allocation is ~one tensor, not
        // a whole-model f32 copy on top of the device weights.
        let reader = checkpoint::weightio::WeightReader::open(&self.weights).map_err(|e| format!("lfm: {e}"))?;
        let model = crate::resident_llm::on_device(device, || {
            Lfm::from_reader_chunked(&reader, self.batch, t, SLAB_BUDGET, PROBE_CAP)
        })?;
        Ok(Box::new(LfmInstance { model, tok: self.tok.clone(), tokenizer_path: self.tokenizer_path.clone(), t, batch: self.batch as usize }))
    }
}

struct LfmInstance {
    model: Lfm,
    tok: Arc<QwenBpe>,
    tokenizer_path: String,
    t: u32,
    batch: usize,
}

impl LfmInstance {
    fn ids_of(&self, inv: &Invocation) -> Result<Vec<u32>, String> {
        let text = match inv.get_str("text").filter(|t| !t.is_empty()) {
            Some(t) => t,
            None => {
                let b = inv.get_blob("text").ok_or("lfm: provide the 'text' param or a 'text' input blob")?;
                String::from_utf8(b.bytes.clone()).map_err(|e| format!("lfm: text blob not UTF-8: {e}"))?
            }
        };
        let mut ids: Vec<u32> = self.tok.template_prefix().to_vec();
        ids.extend(self.tok.encode(&text));
        let max_tokens = inv.get_i64("max_tokens").unwrap_or(0);
        if max_tokens > 0 {
            ids.truncate(max_tokens as usize);
        }
        if ids.len() as u32 != self.t {
            return Err(format!("lfm: length {} does not match instance t{} (scheduler key drift)", ids.len(), self.t));
        }
        Ok(ids)
    }

    /// One TRUE batched forward over `jobs` (≤ batch, equal length): sequences
    /// laid out `[b_use, t]` and run at the GROUP's size (the model prebuilds a
    /// forward per group size, so a partial group never pays padding compute).
    fn forward_group(&mut self, action: &str, jobs: &[(usize, Vec<u32>, &Invocation)], out: &mut [ActionResult]) {
        let m = &self.model;
        let t = self.t as usize;
        let b_use = jobs.len() as u32;
        let mut flat: Vec<u32> = Vec::with_capacity(jobs.len() * t);
        for (_, ids, _) in jobs {
            flat.extend(ids);
        }
        m.set_tokens(&flat);

        match action {
            "fill_mask" => {
                // Probe rows across the whole [b*t] space, per job.
                let mask_id = self.tok.special_id("<|mask|>").unwrap_or(16);
                let mut rows: Vec<u32> = Vec::new();
                let mut spans: Vec<(usize, usize)> = Vec::new(); // per-job (start, count) in `rows`
                for (j, (_, ids, _)) in jobs.iter().enumerate() {
                    let start = rows.len();
                    rows.extend(ids.iter().enumerate().filter_map(|(i, &tk)| (tk == mask_id).then_some((j * t + i) as u32)));
                    spans.push((start, rows.len() - start));
                }
                if rows.len() as u32 > PROBE_CAP {
                    for (slot, _, _) in jobs {
                        out[*slot] = Err(format!("lfm fill_mask: {} masks in group > capacity {PROBE_CAP}", rows.len()));
                    }
                    return;
                }
                if rows.is_empty() {
                    for (slot, _, _) in jobs {
                        out[*slot] = Err("lfm fill_mask: no <|mask|> in the input".into());
                    }
                    return;
                }
                m.set_probe_rows(&rows);
                m.forward_group(b_use);
                let logits = m.read_probe_logits();
                let v = m.cfg.vocab as usize;
                for ((slot, _, inv), &(r0, rn)) in jobs.iter().zip(&spans) {
                    if rn == 0 {
                        out[*slot] = Err("lfm fill_mask: no <|mask|> in the input".into());
                        continue;
                    }
                    let topk = inv.get_i64("topk").unwrap_or(5).clamp(1, 64) as usize;
                    let results: Vec<serde_json::Value> = (r0..r0 + rn)
                        .map(|ri| {
                            let lrow = &logits[ri * v..(ri + 1) * v];
                            let mut idx: Vec<u32> = (0..v as u32).collect();
                            idx.sort_unstable_by(|&x, &y| lrow[y as usize].total_cmp(&lrow[x as usize]));
                            let toks: Vec<serde_json::Value> = idx[..topk]
                                .iter()
                                .map(|&id| serde_json::json!({"id": id, "token": self.tok.decode(&[id]), "logit": lrow[id as usize]}))
                                .collect();
                            serde_json::json!({"row": rows[ri] as usize % t, "tokens": toks})
                        })
                        .collect();
                    let payload = serde_json::to_vec(&results).unwrap_or_default();
                    out[*slot] = Ok(capability::Outcome::new()
                        .set("masks", serde_json::json!(rn))
                        .set("predictions", serde_json::json!(results))
                        .blob("predictions", capability::Blob::new(capability::Media::Text, payload)));
                }
            }
            "embed" => {
                m.forward_group(b_use);
                let d = m.cfg.d_model as usize;
                let hidden = m.read_hidden_rows(jobs.len() * t);
                for (j, (slot, _, _)) in jobs.iter().enumerate() {
                    let h = &hidden[j * t * d..(j + 1) * t * d];
                    let mut mean = vec![0.0f32; d];
                    for row in h.chunks_exact(d) {
                        for (mv, &x) in mean.iter_mut().zip(row) {
                            *mv += x;
                        }
                    }
                    for x in &mut mean {
                        *x /= t as f32;
                    }
                    let bytes: Vec<u8> = h.iter().flat_map(|f| f.to_le_bytes()).collect();
                    let blob = capability::Blob::new(capability::Media::Bytes, bytes)
                        .with_meta(serde_json::json!({"shape": [t, d], "dtype": "f32le"}));
                    out[*slot] = Ok(capability::Outcome::new()
                        .set("tokens", serde_json::json!(t))
                        .set("dim", serde_json::json!(d))
                        .set("mean", serde_json::json!(mean))
                        .blob("embeddings", blob));
                }
            }
            other => {
                for (slot, _, _) in jobs {
                    out[*slot] = Err(format!("lfm: unknown action {other}"));
                }
            }
        }
    }
}

impl Instance for LfmInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.run_batch(action, std::slice::from_ref(inv), &mut |_i, p| progress(p)).pop().unwrap()
    }

    fn run_batch(&mut self, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        let mut out: Vec<ActionResult> = (0..invs.len()).map(|_| Err(String::new())).collect();
        // Tokenize all jobs (per-job errors stay per-job).
        let mut ready: Vec<(usize, Vec<u32>, &Invocation)> = Vec::new();
        for (slot, inv) in invs.iter().enumerate() {
            match self.ids_of(inv) {
                Ok(ids) => ready.push((slot, ids, inv)),
                Err(e) => out[slot] = Err(e),
            }
        }
        let total = ready.len().div_ceil(self.batch.max(1)) as u32;
        let groups: Vec<Vec<(usize, Vec<u32>, &Invocation)>> = {
            let mut gs = Vec::new();
            let mut it = ready.into_iter().peekable();
            while it.peek().is_some() {
                gs.push(it.by_ref().take(self.batch).collect());
            }
            gs
        };
        for (gi, group) in groups.iter().enumerate() {
            self.forward_group(action, group, &mut out);
            let msg = format!("batch {}/{total}", gi + 1);
            for i in 0..invs.len() {
                progress(i, Progress::step(gi as u32 + 1, total, msg.clone()));
            }
        }
        let _ = &self.tokenizer_path; // key identity; kept for diagnostics
        out
    }
}
