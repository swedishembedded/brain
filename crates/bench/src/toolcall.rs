// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! TOOL-CALLING — map a natural-language-shaped *user intent* to exactly one
//! structured **tool call** (a tool id plus its ordered argument values).
//!
//! This is the smallest faithful probe of the tool-calling training pattern:
//! given a request, emit the right function name **and** fill its arguments from
//! the request. Each example is one masked line
//!
//! ```text
//!   VERB_k  F0 v0  F1 v1  ...  Fm vm   =   TOOL_k  a0 a1 ... a(p-1)   NL
//!   └────────────── user intent (prompt) ──────────┘   └─ tool call (assistant) ─┘
//! ```
//!
//! - `VERB_k` is the request's *verb*, which determines the tool `TOOL_k`.
//! - Each tool has a **fixed named signature**: `arg_j` of `TOOL_k` always comes
//!   from the same field-name token (its canonical field). The intent lists those
//!   `p` labelled fields `Fj vj` (each `Fj` is the field-name token, `vj` its
//!   value) **plus `d` distractor fields** drawn from a disjoint name pool (extra
//!   `name value` pairs the call must ignore). All fields are shuffled, so
//!   position carries no information — the model must route by field *name*, not
//!   slot. The signature being fixed across the corpus is what makes that routing
//!   learnable (a per-example-random field→slot map is not).
//! - The assistant span after `=` is the tool call: `TOOL_k` followed by the `p`
//!   argument values **in the tool's canonical argument order** (not intent
//!   order), then `NL`. To produce it the model must (1) map the verb to the tool
//!   id and (2) for each canonical arg, find that tool's field by name in the
//!   intent and copy its value — a routing + induction-head-copy composition,
//!   exactly the shape of real function-call argument filling.
//!
//! ## How it reuses the engine
//! - **Tokens.** A small synthetic vocab (see [`Toolcall::itos`]): `NL`
//!   (sequence end), `SEP` (`=`, the mask char), then disjoint id ranges for
//!   verbs, tool ids, field names (real + distractor), and argument *values*.
//!   Written in brain's standard char-token layout
//!   (`train.bin`/`val.bin`/`meta.json`) so `gpt2::train` loads it unchanged - each
//!   id maps to a distinct Private-Use-Area char.
//! - **Masking.** Loss masked up to & including `SEP`, per line, line-aligned
//!   (`mask_before='='`, `mask_per_line`, `align_to_lines`). This is the **key
//!   tool-calling pattern**: train only on the assistant/tool-call span (tool id +
//!   args), never on the prompt — the gradient learns to *produce* calls, not to
//!   model the (random) request fields.
//! - **Scoring.** Exact-match over the whole predicted tool call. From the prompt
//!   (everything up to & including `=`) the model greedily decodes `p+1` tokens
//!   (tool id + `p` args); the call is correct **only if the tool id and every
//!   argument value match** the reference. See [`exact_match`].
//!
//! ## Calibration (CPU / Cranelift JIT backend)
//! Default config: `n_tools=4`, `args_per_tool=2`, `n_distractors=2`,
//! `arg_values=12`, 8000 sequences, 800 steps, 2-layer / d_model-64 / 4-head GPT.
//! **Chance** for a full correct call is `(1/n_tools) * (1/arg_values)^p` (guess
//! the tool *and* both arg values) `= (1/4)*(1/12)^2 ≈ 0.0017`. **Measured
//! exact-match = 1.00 across seeds {1337, 7, 42}** (train_ce ≈ 0.29), saturating
//! the metric far above chance and clear of the **0.85** threshold, in ~1.5-2 min
//! on CPU (see `tests/toolcall.rs`). The small GPT learns the routing cleanly:
//! the verb→tool map is trivial, and with each tool's argument fields *named by a
//! fixed signature*, filling the slots is two reliable induction-head copies past
//! the distractors. (Scaling `args_per_tool`/`n_tools`/`n_distractors` up keeps it
//! at 1.00 but costs more steps; this config is the fast, clearly-passing point.
//! An earlier variant that re-randomized which field maps to which slot *per
//! example* was unlearnable — plateauing near chance — confirming the metric
//! tracks genuine name-routing, not memorization.)
//!
//! ## Generalizing to real tool-calling (future work)
//! This is deliberately a *single*-call, fixed-arity, token-valued schema so it
//! trains in minutes on the CPU backend. Real tool-calling extends it along
//! orthogonal axes that need no change to the masking/scoring recipe, only richer
//! data generation:
//! - **Multi-call traces.** Emit a sequence of calls (and tool *results* fed back
//!   in) in one assistant span; mask each prompt/result region, supervise each
//!   call span — the same `mask_before`/`align_to_lines` recipe, just more
//!   separators.
//! - **JSON function-call schema.** Replace the synthetic id tokens with a real
//!   tokenizer over `{"name": "...", "arguments": {...}}`; exact-match becomes
//!   parse-then-compare (tool name + named-argument map) instead of a token
//!   tuple, but the "mask the prompt, train+score only the call" pattern is
//!   identical.
//! - **Typed / free-form arguments.** Numbers, strings, enums — values become
//!   multi-token spans; scoring stays exact-match per argument span.

use std::path::Path;

use data::binio::{self, Meta};
use data::rng::Rng;

use crate::metrics::{exact_match, Metrics};
use crate::model::{argmax, DecoderLm, TrainConfig};
use crate::Benchmark;

/// Token id of the newline / end-of-sequence marker (maps to `'\n'`).
const NL: u16 = 0;
/// Token id of the intent→call separator (maps to `'='`, the mask char).
const SEP: u16 = 1;
/// First non-reserved content id. The content id space is laid out in disjoint
/// blocks (verbs, tool ids, field names, arg values) whose sizes derive from the
/// config; see [`Toolcall::layout`].
const CONTENT0: u16 = 2;

/// Disjoint id ranges for the synthetic schema vocab, all relative to
/// [`CONTENT0`]. Kept in one struct so id math has a single source of truth.
struct Layout {
    /// `[verb0, verb0 + n_tools)` — one verb per tool (selects the tool).
    verb0: u16,
    /// `[tool0, tool0 + n_tools)` — the emitted tool-id tokens.
    tool0: u16,
    /// `[argfield0, argfield0 + n_tools*args_per_tool)` — the **argument field
    /// names**. Tool `k`'s canonical argument fields are the contiguous slice
    /// `[argfield0 + k*p, argfield0 + (k+1)*p)`; the j-th entry is tool `k`'s
    /// `arg_j` field name. Fixed across the corpus — this is the tool's *named
    /// signature*, which is what makes name-based routing learnable.
    argfield0: u16,
    /// `[distractor0, distractor0 + n_distractor_names)` — a **disjoint** pool of
    /// field names used only for distractor fields, so a distractor name can never
    /// collide with any tool's argument field name.
    distractor0: u16,
    /// `[val0, val0 + arg_values)` — argument value tokens.
    val0: u16,
    /// Total vocab size (including `NL`/`SEP`).
    vocab: usize,
}

/// Tool-calling configuration. Defaults are calibrated to be clearly learnable by
/// a small GPT in ~1-2 min on the CPU backend (see [`Toolcall::default`]).
#[derive(Clone, Debug)]
pub struct Toolcall {
    /// Number of distinct tools (and verbs). The verb→tool map is the identity
    /// `VERB_k → TOOL_k`, fixed across the corpus.
    pub n_tools: usize,
    /// Argument fields each tool takes (its arity `p`).
    pub args_per_tool: usize,
    /// Extra distractor `name value` fields in the intent the call must ignore.
    pub n_distractors: usize,
    /// Size of the distractor field-name pool (must be `>= n_distractors`). The
    /// argument field names are *not* drawn from here — each tool has its own
    /// fixed argument fields (see [`Layout::argfield0`]).
    pub n_distractor_names: usize,
    /// Distinct argument value tokens. Chance per-arg is `1 / arg_values`.
    pub arg_values: usize,
    /// Number of sequences in the generated corpus.
    pub n_sequences: usize,
    /// Training steps.
    pub steps: u32,
    /// GPT depth / width / heads for the scoring model.
    pub n_layers: u32,
    pub d_model: u32,
    pub n_heads: u32,
    /// Sequences scored for the exact-match metric (drawn from the val split).
    pub eval_sequences: usize,
}

impl Default for Toolcall {
    /// Calibrated config: see the module doc comment. 4 tools, 2 args each, 2
    /// distractor fields, 12 value tokens. Chance for a full correct call ≈
    /// 0.0017; measured exact-match = 1.00 across seeds; threshold 0.85.
    fn default() -> Self {
        Toolcall {
            n_tools: 4,
            args_per_tool: 2,
            n_distractors: 2,
            n_distractor_names: 4,
            arg_values: 12,
            n_sequences: 8000,
            steps: 800,
            n_layers: 2,
            d_model: 64,
            n_heads: 4,
            eval_sequences: 200,
        }
    }
}

impl Toolcall {
    /// The disjoint id blocks derived from the config.
    fn layout(&self) -> Layout {
        let verb0 = CONTENT0;
        let tool0 = verb0 + self.n_tools as u16;
        let argfield0 = tool0 + self.n_tools as u16;
        let distractor0 = argfield0 + (self.n_tools * self.args_per_tool) as u16;
        let val0 = distractor0 + self.n_distractor_names as u16;
        let vocab = val0 as usize + self.arg_values;
        Layout { verb0, tool0, argfield0, distractor0, val0, vocab }
    }

    /// Number of `name value` field pairs in the intent (real args + distractors).
    fn n_fields(&self) -> usize {
        self.args_per_tool + self.n_distractors
    }

    /// Sequence length in tokens:
    /// intent `1 (verb) + 2*n_fields` + `1 (SEP)` + call `1 (tool) + p (args)` + `1 (NL)`.
    fn seq_len(&self) -> usize {
        (1 + 2 * self.n_fields()) + 1 + (1 + self.args_per_tool) + 1
    }

    /// Block size used for both training and scoring: one whole sequence fits.
    fn block_size(&self) -> u32 {
        self.seq_len() as u32
    }

    fn vocab(&self) -> usize {
        self.layout().vocab
    }

    /// Generate one example. Returns the token ids and the **reference tool call**
    /// `[TOOL_k, a0, a1, ..]` (tool id followed by its `p` argument values in
    /// canonical order) — exactly the span scoring must reproduce.
    fn gen_sequence(&self, rng: &mut Rng) -> (Vec<u16>, Vec<u16>) {
        let l = self.layout();
        let p = self.args_per_tool;

        // Pick the tool (and thus the verb). The verb→tool map is the identity.
        let tool = rng.gen_range_inclusive(0, self.n_tools as i64 - 1) as u16;

        // This tool's **fixed** argument field names: arg_j of TOOL_k is always the
        // field `argfield0 + k*p + j`. Because the signature is constant across the
        // corpus, the model can learn "to fill arg_j of this tool, copy the value
        // of *this* field name" — name-based routing rather than impossible
        // per-example guessing.
        let real_fields: Vec<u16> =
            (0..p).map(|j| l.argfield0 + (tool as usize * p + j) as u16).collect();

        // Distractor field names: distinct draws from the disjoint distractor pool,
        // so a distractor name never equals any argument field name.
        let distractor_fields: Vec<u16> =
            sample_distinct_indices(self.n_distractors, self.n_distractor_names, rng)
                .into_iter()
                .map(|i| l.distractor0 + i as u16)
                .collect();

        // Draw a value for each real argument (canonical order) and each
        // distractor field. Values may repeat freely across fields.
        let arg_values: Vec<u16> = (0..p)
            .map(|_| l.val0 + rng.gen_range_inclusive(0, self.arg_values as i64 - 1) as u16)
            .collect();
        let distractor_values: Vec<u16> = (0..self.n_distractors)
            .map(|_| l.val0 + rng.gen_range_inclusive(0, self.arg_values as i64 - 1) as u16)
            .collect();

        // Build the shuffled list of `(field_name, value)` pairs for the intent:
        // real arg fields + distractor fields, in random order (so the assistant
        // cannot rely on position — it must route by field name).
        let mut fields: Vec<(u16, u16)> = Vec::with_capacity(self.n_fields());
        for j in 0..p {
            fields.push((real_fields[j], arg_values[j]));
        }
        for j in 0..self.n_distractors {
            fields.push((distractor_fields[j], distractor_values[j]));
        }
        shuffle(&mut fields, rng);

        // ---- Intent (prompt) ----
        let mut seq = Vec::with_capacity(self.seq_len());
        seq.push(l.verb0 + tool); // VERB_k
        for &(name, val) in &fields {
            seq.push(name);
            seq.push(val);
        }
        // ---- Separator ----
        seq.push(SEP);
        // ---- Tool call (assistant span) ----
        let mut call = Vec::with_capacity(1 + p);
        let tool_tok = l.tool0 + tool;
        seq.push(tool_tok);
        call.push(tool_tok);
        for &v in &arg_values {
            seq.push(v);
            call.push(v);
        }
        seq.push(NL);
        (seq, call)
    }

    fn build_corpus(&self, seed: u64) -> Vec<u16> {
        let mut rng = Rng::new(seed);
        let mut out = Vec::with_capacity(self.n_sequences * self.seq_len());
        for _ in 0..self.n_sequences {
            let (seq, _) = self.gen_sequence(&mut rng);
            out.extend_from_slice(&seq);
        }
        out
    }

    /// Synthetic `itos`: SEP→`'='` (mask char), NL→`'\n'`, every content id → a
    /// distinct Private-Use-Area char, so the standard char-dataset loader +
    /// masking path works without a real text corpus.
    fn itos(&self) -> Vec<char> {
        let vocab = self.vocab();
        let mut itos = vec!['\0'; vocab];
        itos[NL as usize] = '\n';
        itos[SEP as usize] = '=';
        // U+E000.. is the BMP Private Use Area; the schema stays well within.
        for (k, c) in itos.iter_mut().enumerate().skip(CONTENT0 as usize) {
            *c = char::from_u32(0xE000 + (k - CONTENT0 as usize) as u32).unwrap();
        }
        itos
    }
}

impl Benchmark for Toolcall {
    fn name(&self) -> &str {
        "toolcall"
    }

    fn description(&self) -> &str {
        "tool-calling: map a user intent to one structured tool call (id + args)"
    }

    fn prepare(&self, dir: &Path, seed: u64) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        let corpus = self.build_corpus(seed);
        // 90/10 train/val split on whole sequences.
        let split_seqs = (self.n_sequences * 9) / 10;
        let split = split_seqs * self.seq_len();
        binio::write_u16_bin(&dir.join("train.bin"), &corpus[..split])?;
        binio::write_u16_bin(&dir.join("val.bin"), &corpus[split..])?;
        let meta = Meta { vocab_size: self.vocab(), itos: self.itos() };
        std::fs::write(dir.join("meta.json"), meta.to_json())?;
        Ok(())
    }


    fn threshold(&self) -> f32 {
        // Far above chance (~0.0017). The measured exact-match is 1.00 across
        // seeds; 0.85 leaves generous margin for fp32 / single-run noise on the
        // software CPU backend while still flagging any real routing regression.
        0.85
    }

    fn report_fields(&self) -> Vec<&str> {
        vec!["chance", "train_ce"]
    }
    /// Train + score this benchmark with a specific architecture (any
    /// [`DecoderLm`]). [`Benchmark::evaluate`] calls this with the GPT baseline;
    /// scoring an alternative architecture is just passing a different
    /// `DecoderLm` — no other change. This is the architecture-agnostic core.
    fn evaluate_with(&self, lm: &dyn DecoderLm, dir: &Path, seed: u64) -> std::io::Result<Metrics> {
        // ---- TRAIN (architecture-agnostic via DecoderLm) ---------------------
        let block = self.block_size();
        let train_cfg = TrainConfig {
            steps: self.steps,
            batch_size: 32,
            lr: 3e-3,
            n_layers: self.n_layers,
            d_model: self.d_model,
            n_heads: self.n_heads,
            mask_before: Some('='), // SEP — train only on the tool-call span
            mask_per_line: true,
            align_to_lines: true, // each window is one full sequence (NL-aligned)
            seed,
        };
        let out = dir.join("toolcall.safetensors");
        let (init_loss, final_loss) = lm.train_decoder(dir, block, &train_cfg, &out)?;

        // ---- SCORE: exact-match of the full tool call on held-out sequences ---
        let scorer = lm.load_scorer(&out, block);
        let val = binio::read_u16_bin(&dir.join("val.bin"))?;
        let seq_len = self.seq_len();
        let n_val = val.len() / seq_len;
        let to_score = self.eval_sequences.min(n_val);

        // Replay the rng to recover the val tail (gen_sequence is pure in its
        // rng), then fast-forward past the train sequences.
        let mut rng = Rng::new(seed);
        let train_seqs = (self.n_sequences * 9) / 10;
        for _ in 0..train_seqs {
            self.gen_sequence(&mut rng);
        }

        // The prompt is everything up to & including SEP; the call is the next
        // `1 + args_per_tool` tokens (tool id + args). Greedily decode that many
        // tokens and compare the whole tuple.
        let call_len = 1 + self.args_per_tool;
        let prompt_len = seq_len - call_len - 1; // minus trailing NL
        let v = scorer.vocab();

        let mut pairs: Vec<(Vec<u16>, Vec<u16>)> = Vec::with_capacity(to_score);
        for s in 0..to_score {
            let (seq, reference) = self.gen_sequence(&mut rng);
            debug_assert_eq!(&seq[..], &val[s * seq_len..(s + 1) * seq_len]);
            debug_assert_eq!(seq[prompt_len - 1], SEP);

            // Autoregressively decode the call span from the (teacher-free) prompt.
            let mut ctx: Vec<u32> = seq[..prompt_len].iter().map(|&t| t as u32).collect();
            let mut predicted: Vec<u16> = Vec::with_capacity(call_len);
            for _ in 0..call_len {
                let logits = scorer.logits_all(&ctx);
                let last = &logits[logits.len() - v..];
                let tok = argmax(last) as u16;
                predicted.push(tok);
                ctx.push(tok as u32);
            }
            pairs.push((predicted, reference));
        }

        let acc = exact_match(&pairs);
        // Chance: guess the tool id (1/n_tools) AND every argument value
        // (1/arg_values each), independently.
        let chance = (1.0 / self.n_tools as f32)
            * (1.0 / self.arg_values as f32).powi(self.args_per_tool as i32);
        Ok(Metrics::new(acc)
            .with("exact_match", acc)
            .with("chance", chance)
            .with("train_ce", final_loss)
            .with("init_ce", init_loss))
    }
}

/// `k` distinct indices in `[0, n)` via partial Fisher–Yates (requires `k <= n`).
fn sample_distinct_indices(k: usize, n: usize, rng: &mut Rng) -> Vec<usize> {
    assert!(k <= n, "cannot draw {k} distinct of {n}");
    let mut pool: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = rng.gen_range_inclusive(i as i64, n as i64 - 1) as usize;
        pool.swap(i, j);
    }
    pool[..k].to_vec()
}

/// In-place Fisher–Yates shuffle driven by the shared deterministic [`Rng`].
fn shuffle<T>(items: &mut [T], rng: &mut Rng) {
    let n = items.len();
    for i in (1..n).rev() {
        let j = rng.gen_range_inclusive(0, i as i64) as usize;
        items.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_shape_and_call() {
        let m = Toolcall {
            n_tools: 3,
            args_per_tool: 2,
            n_distractors: 2,
            n_distractor_names: 4,
            arg_values: 8,
            ..Toolcall::default()
        };
        let mut rng = Rng::new(1);
        let (seq, call) = m.gen_sequence(&mut rng);
        assert_eq!(seq.len(), m.seq_len());
        assert_eq!(*seq.last().unwrap(), NL);

        let l = m.layout();
        let call_len = 1 + m.args_per_tool;
        let prompt_len = m.seq_len() - call_len - 1;
        // SEP sits right before the call span.
        assert_eq!(seq[prompt_len - 1], SEP);
        // The reference call is exactly the span after SEP (excl. the trailing NL).
        assert_eq!(&seq[prompt_len..prompt_len + call_len], &call[..]);

        // The verb selects the tool: VERB_k in the intent, TOOL_k in the call.
        let verb = seq[0];
        assert!((l.verb0..l.verb0 + m.n_tools as u16).contains(&verb));
        let tool = call[0];
        assert!((l.tool0..l.tool0 + m.n_tools as u16).contains(&tool));
        assert_eq!(verb - l.verb0, tool - l.tool0, "verb->tool map is the identity");

        // Every argument value in the call is the value of the tool's canonical
        // arg field in the intent — the assistant copies by name, never invents.
        // Build the intent's field-name → value map (pairs after the leading verb).
        let p = m.args_per_tool;
        let intent = &seq[1..prompt_len - 1]; // skip verb, drop SEP
        let mut by_name = std::collections::HashMap::new();
        for pair in intent.chunks(2) {
            by_name.insert(pair[0], pair[1]);
        }
        let tool_k = (tool - l.tool0) as usize;
        for j in 0..p {
            let field = l.argfield0 + (tool_k * p + j) as u16; // fixed signature
            let arg = call[1 + j];
            assert!((l.val0..l.val0 + m.arg_values as u16).contains(&arg));
            assert_eq!(
                by_name.get(&field).copied(),
                Some(arg),
                "arg {j} must equal the value of the tool's canonical field"
            );
        }
    }

    #[test]
    fn layout_blocks_are_disjoint_and_packed() {
        let m = Toolcall::default();
        let l = m.layout();
        assert_eq!(l.verb0, CONTENT0);
        assert_eq!(l.tool0, l.verb0 + m.n_tools as u16);
        assert_eq!(l.argfield0, l.tool0 + m.n_tools as u16);
        assert_eq!(l.distractor0, l.argfield0 + (m.n_tools * m.args_per_tool) as u16);
        assert_eq!(l.val0, l.distractor0 + m.n_distractor_names as u16);
        assert_eq!(l.vocab, l.val0 as usize + m.arg_values);
        assert_eq!(l.vocab, m.vocab());
        // itos covers exactly the vocab with distinct chars for content ids.
        let itos = m.itos();
        assert_eq!(itos.len(), m.vocab());
        assert_eq!(itos[SEP as usize], '=');
        assert_eq!(itos[NL as usize], '\n');
    }

    #[test]
    fn corpus_split_is_sequence_aligned() {
        let m = Toolcall { n_sequences: 100, ..Toolcall::default() };
        let corpus = m.build_corpus(7);
        assert_eq!(corpus.len(), m.n_sequences * m.seq_len());
    }

    #[test]
    fn distractor_names_disjoint_from_arg_fields() {
        // Distractor names come from a pool disjoint from every tool's argument
        // field names, so a distractor can never be mistaken for a real arg field.
        let m = Toolcall {
            n_tools: 3,
            args_per_tool: 2,
            n_distractors: 3,
            n_distractor_names: 5,
            ..Toolcall::default()
        };
        let mut rng = Rng::new(9);
        let l = m.layout();
        let arg_field_lo = l.argfield0;
        let arg_field_hi = l.argfield0 + (m.n_tools * m.args_per_tool) as u16;
        let dist_lo = l.distractor0;
        let dist_hi = l.distractor0 + m.n_distractor_names as u16;
        for _ in 0..50 {
            let (seq, _) = m.gen_sequence(&mut rng);
            let call_len = 1 + m.args_per_tool;
            let prompt_len = m.seq_len() - call_len - 1;
            let intent = &seq[1..prompt_len - 1]; // skip verb, drop SEP
            // Field-name tokens are the even positions of the field list.
            let names: Vec<u16> = intent.iter().step_by(2).copied().collect();
            // All names are distinct (no two fields share a name in one intent).
            let mut sorted = names.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), names.len(), "field names must be distinct");
            // Every name is either an arg-field id or a (disjoint) distractor id.
            for n in &names {
                let is_arg = (arg_field_lo..arg_field_hi).contains(n);
                let is_dist = (dist_lo..dist_hi).contains(n);
                assert!(is_arg ^ is_dist, "name {n} must be exactly one of arg/distractor");
            }
        }
    }
}
