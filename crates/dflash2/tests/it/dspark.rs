// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements speculative decoding for clients serving large models on modest hardware.
// If your team needs expertise in gating a ported drafter against its reference then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The DSpark draft (the Inkling 2 MTP drafter) against its reference, `tools/goldens/inkling2_mtp_reference.py`:
//!
//! - a tiny random draft (three layers, the first and last local with a window of 6, grouped queries, a block of five,
//!   a Markov head of rank 8, four taps): every layer's rows, the post-norm mask rows, the Markov head's per-step
//!   logits and its path, for one round and for a second round after the context grew (fed in pieces, the rows a
//!   denoising block wrote overwritten by the real context), two sequences in one forward, and the serving drafter
//!   end to end over a block table;
//! - the released `mtp.safetensors` (bf16 as stored) over real target features of the three reference prompts: the
//!   same comparisons, on the CPU backend.
//!
//! Fixtures, each skipped by name when absent:
//!
//! ```text
//! python3 -I tools/goldens/inkling2_mtp_reference.py tiny --out testdata/golden/inkling2/mtp-tiny
//! python3 -I tools/goldens/inkling2_mtp_reference.py real --checkpoint <store>/thinkingmachines/Inkling-2-Mini \
//!     --features testdata/golden/inkling2/real-f32/golden.safetensors --out testdata/golden/inkling2/mtp-real
//! ```

use std::cell::RefCell;

use dflash2::markov::MarkovJob;
use dflash2::{BlockJob, Dflash2, Dflash2Config, DraftCache, DraftHost, DraftRequest, Dflash2Drafter, DsparkShape, HfDraft, TapRows};
use model::paged::{BlockAllocator, BlockTable};

use crate::common::{gather, logits, scaled_error, Golden};

const TOLERANCE: f32 = 1e-4;
const TINY: &str = "golden/inkling2/mtp-tiny";
const REAL: &str = "golden/inkling2/mtp-real";
const MODEL: &str = "thinkingmachines/Inkling-2-Mini";

/// The shape a target's `mtp_config` declares (the target's crate parses it for real; here the fixture's keys).
fn shape(mtp: &serde_json::Value, vocab: u32) -> DsparkShape {
    let u = |k: &str| mtp[k].as_u64().unwrap_or_else(|| panic!("mtp_config.{k}")) as u32;
    let list = |k: &str| mtp[k].as_array().unwrap_or_else(|| panic!("mtp_config.{k}")).iter().map(|v| v.as_u64().unwrap() as usize).collect();
    DsparkShape {
        d_model: u("hidden_size"),
        n_layers: u("num_hidden_layers") as usize,
        n_heads: u("num_attention_heads"),
        n_kv_heads: u("num_key_value_heads"),
        head_dim: u("head_dim"),
        d_ff: u("intermediate_size"),
        eps: mtp["rms_norm_eps"].as_f64().unwrap() as f32,
        rope_theta: mtp["rope_theta"].as_f64().unwrap() as f32,
        local_layers: list("local_layer_ids"),
        window: u("sliding_window_size"),
        block_size: u("block_size"),
        markov_rank: u("markov_rank"),
        target_depths: list("target_layer_ids"),
        vocab,
    }
}

struct Fixture {
    cfg: Dflash2Config,
    dir: String,
    gold: Golden,
    cap: f32,
}

fn tiny() -> Option<Fixture> {
    let dir = brain_testutil::testdata(TINY);
    if !std::path::Path::new(&dir).join("golden.safetensors").exists() {
        brain_testutil::skip(&format!("{dir} absent: run tools/goldens/inkling2_mtp_reference.py tiny --out {dir}"));
        return None;
    }
    let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/draft/config.json")).unwrap()).unwrap();
    let cfg = Dflash2Config::dspark(&shape(&json["mtp_config"], json["vocab_size"].as_u64().unwrap() as u32)).unwrap();
    Some(Fixture { cfg, gold: Golden::read(&format!("{dir}/golden.safetensors")), cap: json["final_logit_softcapping"].as_f64().unwrap() as f32, dir })
}

fn load(fx: &Fixture, slots: u32, cap: u32) -> Dflash2 {
    let gpu = gpu_core::Gpu::open(None, &dflash2::pipelines(&fx.cfg));
    let src = HfDraft::open(std::path::Path::new(&format!("{}/draft/mtp.safetensors", fx.dir))).unwrap();
    Dflash2::load_with(gpu, fx.cfg.clone(), Box::new(src), slots, cap, DraftCache::dense(&fx.cfg, slots, cap)).unwrap()
}

/// The target's head of the fixture: `softcap(x @ head^T)`.
fn project(head: &[f32], d: usize, cap: f32, rows: &[f32]) -> Vec<f32> {
    logits(rows, head, d).into_iter().map(|v| cap * (v / cap).tanh()).collect()
}

/// One round against the golden under `prefix`: every layer, the post-norm mask rows, each Markov step's logits and
/// the path. `job` of `jobs` blocks in the trace.
fn check(fx: &Fixture, m: &Dflash2, prefix: &str, hidden: &[f32], trace: &[Vec<f32>], job: usize, tolerance: f32) {
    let d = fx.cfg.d_model as usize;
    let per_job = fx.cfg.block_size as usize * d;
    for (l, layer) in trace.iter().enumerate() {
        let err = scaled_error(&format!("{prefix}layer {l}"), &layer[job * per_job..(job + 1) * per_job], fx.gold.get(&format!("{prefix}layer.{l}")));
        assert!(err <= tolerance, "{prefix}layer {l}: scaled error {err:.3e}");
    }
    let normed = m.normed(&hidden[d..]);
    let err = scaled_error(&format!("{prefix}post-norm"), &normed, fx.gold.get(&format!("{prefix}hidden")));
    assert!(err <= tolerance, "{prefix}post-norm hidden: scaled error {err:.3e}");
    let taps = fx.gold.get(&format!("{prefix}taps"));
    let wide = fx.cfg.tap_width();
    let last = &taps[taps.len() - wide..];
    let anchor = fx.gold.ids(&format!("{prefix}anchor"))[0];
    let steps = RefCell::new(Vec::new());
    let head = fx.gold.get("head");
    let mut proj = |rows: &[f32]| {
        let l = project(head, d, fx.cap, rows);
        steps.borrow_mut().extend_from_slice(&l);
        Ok(l)
    };
    let want = fx.cfg.block_size as usize - 1;
    let path = m.markov_walk(&[MarkovJob { hidden: &normed, last_taps: last, anchor, want }], &mut proj).unwrap().remove(0);
    let err = scaled_error(&format!("{prefix}markov logits"), &steps.borrow(), fx.gold.get(&format!("{prefix}logits")));
    assert!(err <= tolerance, "{prefix}Markov logits: scaled error {err:.3e}");
    assert_eq!(path, fx.gold.ids(&format!("{prefix}path")), "{prefix}Markov path");
}

fn noise(fx: &Fixture, m: &Dflash2, prefix: &str) -> Vec<f32> {
    let anchor = fx.gold.ids(&format!("{prefix}anchor"));
    m.learned_noise(&gather(fx.gold.get("embed"), fx.cfg.d_model as usize, &anchor)).expect("a DSpark draft learned its mask rows")
}

#[test]
fn one_round_and_a_grown_context_match_the_reference() {
    let Some(fx) = tiny() else { return };
    let m = load(&fx, 3, 32);
    let wide = fx.cfg.tap_width();
    let taps_a = fx.gold.get("a.taps");
    // Round a, the context arriving in pieces.
    for (from, to) in [(0usize, 5usize), (5, 6), (6, 14)] {
        m.append_context(1, &taps_a[from * wide..to * wide], from as u32).unwrap();
    }
    let mut trace = Vec::new();
    let out = m.denoise_traced(&[BlockJob { slot: 1, pos: 14, noise: &noise(&fx, &m, "a.") }], Some(&mut trace)).unwrap();
    check(&fx, &m, "a.", &out[0], &trace, 0, TOLERANCE);
    // The target commits three more positions: the block rows round a wrote past 14 are overwritten by the context.
    m.append_context(1, &fx.gold.get("b.taps")[14 * wide..], 14).unwrap();
    let mut trace = Vec::new();
    let out = m.denoise_traced(&[BlockJob { slot: 1, pos: 17, noise: &noise(&fx, &m, "b.") }], Some(&mut trace)).unwrap();
    check(&fx, &m, "b.", &out[0], &trace, 0, TOLERANCE);
}

#[test]
fn two_sequences_in_one_forward_are_each_what_they_are_alone() {
    let Some(fx) = tiny() else { return };
    let m = load(&fx, 3, 32);
    m.append_context(0, fx.gold.get("a.taps"), 0).unwrap();
    m.append_context(2, fx.gold.get("b.taps"), 0).unwrap();
    let (na, nb) = (noise(&fx, &m, "a."), noise(&fx, &m, "b."));
    let mut trace = Vec::new();
    let out = m.denoise_traced(&[BlockJob { slot: 0, pos: 14, noise: &na }, BlockJob { slot: 2, pos: 17, noise: &nb }], Some(&mut trace)).unwrap();
    check(&fx, &m, "a.", &out[0], &trace, 0, TOLERANCE);
    check(&fx, &m, "b.", &out[1], &trace, 1, TOLERANCE);
}

/// The target side of the drafter for the fixture: its embedding, its head and the taps per position.
struct Target<'a> {
    fx: &'a Fixture,
    taps: &'a [f32],
}

impl DraftHost for Target<'_> {
    fn embed(&self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        Ok(gather(self.fx.gold.get("embed"), self.fx.cfg.d_model as usize, tokens))
    }
    fn project(&self, normed: &[f32]) -> Result<Vec<f32>, String> {
        Ok(project(self.fx.gold.get("head"), self.fx.cfg.d_model as usize, self.fx.cap, normed))
    }
}

impl TapRows for Target<'_> {
    fn tap_rows(&self, table: &BlockTable, from: u32, count: u32) -> Result<Vec<f32>, String> {
        let wide = self.fx.cfg.tap_width();
        assert!(from + count <= table.len());
        Ok(self.taps[from as usize * wide..(from + count) as usize * wide].to_vec())
    }
}

/// The serving drafter: a sequence of 17 committed positions whose anchor is the golden's proposes the golden's path,
/// cut to what the scheduler asks for.
#[test]
fn the_drafter_proposes_the_reference_path() {
    let Some(fx) = tiny() else { return };
    let m = load(&fx, 2, 32);
    let mut drafter = Dflash2Drafter::new(m);
    assert_eq!(drafter.max_draft(), 4);
    let mut alloc = BlockAllocator::new(8, 8);
    let mut table = BlockTable::new();
    table.reserve(17, &mut alloc).unwrap();
    let anchor = fx.gold.ids("b.anchor")[0];
    let mut context: Vec<u32> = (0..17).map(|i| i % 50 + 1).collect();
    context.push(anchor);
    let target = Target { fx: &fx, taps: fx.gold.get("b.taps") };
    for want in [4usize, 2] {
        let got = drafter.propose(&target, &target, &[DraftRequest { id: 7, context: &context, table: &table, want }]);
        assert_eq!(got[0], fx.gold.ids("b.path")[..want], "want {want}");
    }
}

#[test]
fn the_released_drafter_matches_the_reference_over_real_target_features() {
    let golden = brain_testutil::testdata(&format!("{REAL}/golden.safetensors"));
    let Some(dir) = brain_testutil::model_dir(MODEL).filter(|d| std::path::Path::new(d).join("mtp.safetensors").is_file()) else {
        brain_testutil::skip(&format!("{MODEL} (mtp.safetensors) is not in the model store"));
        return;
    };
    if !std::path::Path::new(&golden).is_file() {
        brain_testutil::skip(&format!("{golden} absent: run tools/goldens/inkling2_mtp_reference.py real (see this file's header)"));
        return;
    }
    let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/config.json")).unwrap()).unwrap();
    let text = &json["text_config"];
    let cfg = Dflash2Config::dspark(&shape(&json["mtp_config"], text["vocab_size"].as_u64().unwrap() as u32)).unwrap();
    assert_eq!((cfg.n_layers, cfg.block_size, cfg.window, cfg.target_layers.clone(), cfg.markov_rank()), (3, 8, 1024, vec![5, 23, 41, 48], Some(256)));
    let (cap, valid) = (text["final_logit_softcapping"].as_f64().unwrap() as f32, text["unpadded_vocab_size"].as_u64().unwrap() as usize);
    let d = cfg.d_model as usize;
    let gold = Golden::read(&golden);
    // The target's head, as the served engine applies it: the soft cap over the valid vocabulary.
    let reader = checkpoint::packed::PackedReader::open(std::path::Path::new(&dir)).unwrap();
    let src = checkpoint::packed::PackedSource::identity(Box::new(reader), checkpoint::packed::CodecRegistry::reference()).unwrap();
    let head = src.read_range("model.llm.unembed.weight", 0, (valid * d) as u64).unwrap().expect("the target's head");
    let gpu = gpu_core::Gpu::open(None, &dflash2::pipelines(&cfg));
    let draft = HfDraft::open(std::path::Path::new(&format!("{dir}/mtp.safetensors"))).unwrap();
    let planned = draft.device_weight_bytes(&cfg);
    let m = Dflash2::load(gpu, cfg.clone(), Box::new(draft), 1, 64).unwrap();
    assert_eq!(planned, m.weight_bytes(), "the memory plan charges what the load holds");
    let fx = Fixture { cfg, dir: dir.clone(), gold, cap };
    for p in 0..3 {
        let prefix = format!("p{p}.");
        let taps = fx.gold.get(&format!("{prefix}taps"));
        let n = (taps.len() / fx.cfg.tap_width()) as u32;
        m.release_slot(0);
        m.append_context(0, taps, 0).unwrap();
        let noise = m.learned_noise(fx.gold.get(&format!("{prefix}anchor_embed"))).unwrap();
        let mut trace = Vec::new();
        let out = m.denoise_traced(&[BlockJob { slot: 0, pos: n, noise: &noise }], Some(&mut trace)).unwrap();
        let normed = m.normed(&out[0][d..]);
        let err = scaled_error(&format!("{prefix}post-norm"), &normed, fx.gold.get(&format!("{prefix}hidden")));
        assert!(err <= TOLERANCE, "{prefix}post-norm hidden: scaled error {err:.3e}");
        for (l, layer) in trace.iter().enumerate() {
            let err = scaled_error(&format!("{prefix}layer {l}"), layer, fx.gold.get(&format!("{prefix}layer.{l}")));
            assert!(err <= TOLERANCE, "{prefix}layer {l}: scaled error {err:.3e}");
        }
        let steps = RefCell::new(Vec::new());
        let mut proj = |rows: &[f32]| {
            let l: Vec<f32> = rows.chunks(d).flat_map(|row| model::hostmath::matvec_par(&head, row, valid, d)).map(|v| cap * (v / cap).tanh()).collect();
            steps.borrow_mut().extend_from_slice(&l);
            Ok(l)
        };
        let anchor = fx.gold.ids(&format!("{prefix}anchor"))[0];
        let wide = fx.cfg.tap_width();
        let path = m.markov_walk(&[MarkovJob { hidden: &normed, last_taps: &taps[taps.len() - wide..], anchor, want: 7 }], &mut proj).unwrap().remove(0);
        let err = scaled_error(&format!("{prefix}markov logits"), &steps.borrow(), fx.gold.get(&format!("{prefix}logits")));
        assert!(err <= TOLERANCE, "{prefix}Markov logits: scaled error {err:.3e}");
        assert_eq!(path, fx.gold.ids(&format!("{prefix}path")), "{prefix}Markov path");
    }
}
