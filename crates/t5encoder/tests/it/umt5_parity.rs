// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Stage-by-stage forward parity for the **umT5-XXL** encoder Wan2.1 conditions
//! on, against the goldens dumped by
//! `tools/goldens/wan_t5_dump_reference.py`.
//!
//! ```text
//! BRAIN_WAN_T5=/path/to/models_t5_umt5-xxl-enc-bf16.pth
//! ```
//!
//! Fixtures live under `$BRAIN_TESTDATA` (default `<repo>/testdata`) in
//! `golden/wan/t5/`; the test SKIPS itself (never fails) without either.
//!
//! ## What this gates that `tests/parity.rs` cannot
//!
//! Same block topology, same widths, three different things - and two of the
//! three fail SILENTLY rather than loudly:
//!
//! 1. **Per-block relative position bias.** `umt5_xxl` passes
//!    `shared_pos=False`, so there are 24 independent `[32, 64]` tables where
//!    T5 v1.1 has one. `b0_position_bias` and `b23_position_bias` are checked
//!    separately for exactly this reason - the reference's own two tables
//!    differ by max_abs 53, so a shared-bias port cannot pass both.
//! 2. **Key padding.** Wan tokenizes to a fixed 512 and drives the encoder with
//!    the mask. The reference dump measures 1.5 max_abs between the masked and
//!    unmasked runs on the CONTENT rows, so this is not a rounding-scale
//!    choice.
//! 3. **The 512 pad is applied AFTER the encoder, as hard zeros.**
//!    `T5EncoderModel.__call__` trims to `seq_len`, and `WanModel.forward`
//!    re-pads with `new_zeros`, so the DiT's cross-attention keys at the pad
//!    positions are zero - NOT the encoder's output there, which peaks at 0.87.
//!
//! Every metric is reported three ways (cosine, rel_l2, max_abs) and the
//! `[B, 512, 4096]` tensors additionally split into **content rows** and **pad
//! rows**. That split is the point: 487 of the 512 rows in the second prompt
//! are padding, so a whole-tensor number is dominated by a population with
//! completely different semantics.
//!
//! ## Where it runs
//!
//! 5.681 B parameters is **22.72 GB in fp32**, plus ~4 GB of activations at
//! B=2, T=512. That does not fit the 24 GB card this was written on, so this
//! test is a `BRAIN_DEVICE=cpu` gate today. INT8 (`model::int8`) is what would
//! make it single-card, exactly as the crate docs say for T5-XXL.

use std::path::{Path, PathBuf};

use brain_testutil::parity::rel_l2;
use t5encoder::config::T5Config;
use t5encoder::model::{T5Encoder, Tap};

const GATE: f64 = 0.9999;
const TEXT_LEN: usize = 512;

fn testdata(rel: &str) -> PathBuf {
    PathBuf::from(brain_testutil::testdata(rel))
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    if na == 0.0 && nb == 0.0 {
        return 1.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn max_abs(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).abs()).fold(0.0, f64::max)
}

/// Collected results so a run prints one table and fails once, at the end - a
/// single failing stage must not hide the twenty behind it.
#[derive(Default)]
struct Report {
    rows: Vec<(String, f64)>,
    failures: Vec<String>,
}

impl Report {
    fn check(&mut self, stage: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{stage}: len {} != golden {}", got.len(), want.len());
        let (c, m, r) = (cosine(got, want), max_abs(got, want), rel_l2(got, want));
        eprintln!("  {stage:<34} cosine={c:.10}  rel_l2={r:.3e}  max_abs={m:.3e}  n={}", got.len());
        self.rows.push((stage.to_string(), c));
        // NaN-safe: a NaN cosine (an all-zero or poisoned stage) must FAIL, so
        // this is an explicit NaN check plus `<`, not `!(>=)`.
        if c.is_nan() || c < GATE {
            self.failures.push(format!("{stage}: cosine {c:.10} < {GATE}"));
        }
    }

    /// Same stage, restricted to the rows `keep` selects.
    fn check_rows(&mut self, stage: &str, got: &[f32], want: &[f32], width: usize, keep: &dyn Fn(usize) -> bool) {
        let (mut g, mut w) = (Vec::new(), Vec::new());
        for r in 0..got.len() / width {
            if keep(r) {
                g.extend_from_slice(&got[r * width..(r + 1) * width]);
                w.extend_from_slice(&want[r * width..(r + 1) * width]);
            }
        }
        if g.is_empty() {
            return;
        }
        self.check(stage, &g, &w);
    }

    fn finish(self, what: &str) {
        let worst = self.rows.iter().map(|r| r.1).fold(f64::INFINITY, f64::min);
        eprintln!(
            "{what}: {} stages checked, {} failed, worst cosine {worst:.10}",
            self.rows.len(),
            self.failures.len()
        );
        assert!(self.failures.is_empty(), "{what} parity failures:\n  {}", self.failures.join("\n  "));
    }
}

struct Golden {
    t: Vec<checkpoint::safetensors::StTensor>,
}

impl Golden {
    fn open(rel: &str) -> Option<Golden> {
        let p = testdata(rel);
        if !p.exists() {
            brain_testutil::skip(&format!("golden {} absent - run tools/goldens/wan_t5_dump_reference.py", p.display()));
            return None;
        }
        Some(Golden { t: checkpoint::safetensors::read(p.to_str().unwrap()).expect("read golden") })
    }
    fn find(&self, name: &str) -> &checkpoint::safetensors::StTensor {
        self.t.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("golden tensor {name}"))
    }
    fn get(&self, name: &str) -> &Vec<f32> {
        &self.find(name).data
    }
    fn shape(&self, name: &str) -> &Vec<usize> {
        &self.find(name).shape
    }
    fn ids(&self, name: &str) -> Vec<u32> {
        self.get(name).iter().map(|&x| x as u32).collect()
    }
}

fn env_path(var: &str) -> Option<PathBuf> {
    let v = std::env::var(var).ok().filter(|s| !s.is_empty())?;
    let p = PathBuf::from(v);
    if !p.exists() {
        brain_testutil::skip(&format!("{var}={} not found", p.display()));
        return None;
    }
    Some(p)
}

#[test]
fn umt5_xxl_encoder_stage_parity() {
    let Some(g) = Golden::open("golden/wan/t5/encoder.safetensors") else { return };
    let Some(weights) = env_path("BRAIN_WAN_T5") else {
        brain_testutil::skip("set BRAIN_WAN_T5 to models_t5_umt5-xxl-enc-bf16.pth");
        return;
    };

    let cfg = T5Config::umt5_xxl();
    // Build the device FIRST: at 22.72 GB of fp32 weights this does not fit any
    // 24 GB card, and finding that out after a 60 s checkpoint read - as an
    // opaque `wgpu error: Out of Memory` from inside `ParamStore` - is a worse
    // failure than a skip that says so. `BRAIN_WAN_T5_FORCE_GPU=1` runs it
    // anyway on a card that can hold it.
    let gpu = gpu_core::testgpu::dev(t5encoder::model::PIPELINES);
    let forced = std::env::var("BRAIN_WAN_T5_FORCE_GPU").is_ok_and(|v| v == "1");
    if gpu.caps().class != gpu_core::DeviceClass::Cpu && !forced {
        brain_testutil::skip_unavailable(&format!("umT5-XXL is {:.2} GB in fp32 plus ~4 GB of activations at B=2, T=512, \
             which exceeds this device; re-run with BRAIN_DEVICE=cpu (or \
             BRAIN_WAN_T5_FORCE_GPU=1 on a card that fits it)",
            cfg.param_count() as f64 * 4.0 / 1e9));
        return;
    }
    let ids_shape = g.shape("input_ids").clone();
    assert_eq!(ids_shape.len(), 2, "input_ids must be [B, T]");
    let (b, t) = (ids_shape[0] as u32, ids_shape[1] as u32);
    assert_eq!(t as usize, TEXT_LEN, "Wan tokenizes to text_len = 512");
    let ids = g.ids("input_ids");
    let mask = g.ids("attention_mask");
    let n = (b * t) as usize;
    let (d, ff, inner) = (cfg.d_model as usize, cfg.d_ff as usize, cfg.inner() as usize);

    // ---- rung 1: mapping units ------------------------------------------
    // The bucket table is host integer math; check it EXACTLY (not by cosine)
    // against the reference's own table before anything reads it. It is shared
    // with T5 v1.1 - `T5RelativeEmbedding._relative_position_bucket` and
    // transformers' `_relative_position_bucket` are the same formula - so this
    // also proves umT5 did not change it.
    let buckets = t5encoder::hostbias::buckets(t, cfg.rel_buckets, cfg.rel_max_distance);
    assert_eq!(buckets, g.ids("relative_position_bucket"), "bucket table differs from the reference");
    eprintln!("relative_position_bucket: {} entries, exact match", buckets.len());

    let t0 = std::time::Instant::now();
    let src = t5encoder::import::read_encoder_pth(&weights).expect("read umT5 .pth");
    let src_count = src.len();
    let imported = t5encoder::import::import_wan(src, &cfg).expect("import_wan");
    eprintln!(
        "umt5: {src_count} source tensors -> {} parameters ({:.3} B) in {:.0}s",
        imported.len(),
        cfg.param_count() as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );
    assert_eq!(imported.len(), cfg.tensor_manifest().len());
    assert_eq!(src_count, 242, "the umT5 encoder ships 242 tensors");

    // The 24 relative-position tables must arrive INTACT and DISTINCT: the
    // silent failure mode is every block reading block 0's table, and at this
    // rung it costs one exact compare against the reference's own stack.
    let tables = g.get("pos_emb_tables");
    let per = (cfg.rel_buckets * cfg.heads) as usize;
    for l in 0..cfg.layers as usize {
        let got = &imported[&cfg.rel_bias_name(l)].1;
        assert_eq!(got.as_slice(), &tables[l * per..(l + 1) * per], "block {l} rel_bias table");
    }
    eprintln!("pos_emb_tables: {} tables of {per} values, exact match", cfg.layers);

    // ---- rungs 2-3: stage + single-forward parity ------------------------
    let t0 = std::time::Instant::now();
    let m = T5Encoder::new_on(
        gpu,
        cfg.clone(),
        b,
        t,
        &t5encoder::import::to_init(imported),
    );
    m.set_tokens(&ids);
    m.set_mask(&mask);
    eprintln!("built in {:.0}s; forward B={b}, T={t} ...", t0.elapsed().as_secs_f64());
    let t0 = std::time::Instant::now();
    m.forward();
    m.poll_wait();
    eprintln!("forward in {:.0}s", t0.elapsed().as_secs_f64());

    let mut rep = Report::default();
    let last = cfg.layers as usize - 1;
    rep.check("b0_position_bias", &m.read_block_bias(0), g.get("b0_position_bias"));
    rep.check("b23_position_bias", &m.read_block_bias(last), g.get("b23_position_bias"));
    // Not a metric but a mutation gate: if the two block biases were equal the
    // pair of checks above would prove nothing about `shared_pos`.
    let spread = max_abs(g.get("b0_position_bias"), g.get("b23_position_bias"));
    assert!(spread > 1.0, "the reference's block 0 and 23 biases differ by only {spread:.3e}");
    eprintln!("  (blocks 0 and 23 differ by max_abs {spread:.3e} in the reference)");

    rep.check("embed", &m.read_x(0), g.get("embed"));

    // block-0 internals
    let qkv = m.read_block_tap(0, Tap::Qkv);
    let region = |off: usize| -> Vec<f32> {
        (0..n).flat_map(|r| qkv[r * 3 * inner + off..r * 3 * inner + off + inner].to_vec()).collect()
    };
    rep.check("b0_attn_norm", &m.read_block_tap(0, Tap::AttnNorm), g.get("b0_attn_norm"));
    rep.check("b0_q", &region(0), g.get("b0_q"));
    rep.check("b0_k", &region(inner), g.get("b0_k"));
    rep.check("b0_v", &region(2 * inner), g.get("b0_v"));
    rep.check("b0_attn_ctx", &m.read_block_tap(0, Tap::Ctx), g.get("b0_attn_ctx"));
    rep.check("b0_attn_out", &m.read_block_tap(0, Tap::AttnOut), g.get("b0_attn_out"));
    rep.check("b0_attn_res", &m.read_block_tap(0, Tap::AttnRes), g.get("b0_attn_res"));
    rep.check("b0_ff_norm", &m.read_block_tap(0, Tap::FfNorm), g.get("b0_ff_norm"));
    // wi_0 is the GELU'd half (`ffn.gate.0`) and wi_1 the linear one
    // (`ffn.fc1`) - the reference numbers them the other way round, so a
    // swapped import shows up HERE, one tap before `b0_gated` hides it.
    rep.check("b0_wi0", &m.read_block_tap(0, Tap::Wi0), g.get("b0_wi0"));
    rep.check("b0_wi1", &m.read_block_tap(0, Tap::Wi1), g.get("b0_wi1"));
    rep.check("b0_gated", &m.read_block_tap(0, Tap::Gated), g.get("b0_gated"));
    rep.check("b0_ff_out", &m.read_block_tap(0, Tap::FfOut), g.get("b0_ff_out"));
    rep.check_rows("b0_wi0[content]", &m.read_block_tap(0, Tap::Wi0), g.get("b0_wi0"), ff, &|r| mask[r] == 1);

    // every block output - this is where a per-layer drift shows up
    for l in 0..cfg.layers as usize {
        rep.check(&format!("block{l}_out"), &m.read_x(l + 1), g.get(&format!("block{l}_out")));
    }

    let hidden = m.read_hidden();
    let want = g.get("last_hidden_state");
    rep.check("last_hidden_state", &hidden, want);
    rep.check_rows("last_hidden_state[content]", &hidden, want, d, &|r| mask[r] == 1);
    rep.check_rows("last_hidden_state[pad]", &hidden, want, d, &|r| mask[r] == 0);

    // The DiT-facing context: content rows from the encoder, pad rows EXACTLY
    // zero on both sides. Cosine on the pad population is vacuous once both are
    // zero, so that half is asserted as an identity instead of scored.
    let ctx = m.read_context();
    let want_ctx = g.get("context_padded");
    rep.check_rows("context_padded[content]", &ctx, want_ctx, d, &|r| mask[r] == 1);
    let pad_max = (0..n)
        .filter(|&r| mask[r] == 0)
        .flat_map(|r| ctx[r * d..(r + 1) * d].iter().chain(want_ctx[r * d..(r + 1) * d].iter()))
        .fold(0.0f32, |a, &v| a.max(v.abs()));
    eprintln!("  context_padded[pad]: every value exactly zero on both sides (max {pad_max:e})");
    assert_eq!(pad_max, 0.0, "the DiT context is not zero on the pad rows");

    // porting.md section 6, kept visible in the log rather than in a comment:
    // the unmasked run is a DIFFERENT answer on the content rows, so the mask
    // is a requirement and not a preference.
    let unmasked = g.get("last_hidden_state_unmasked");
    let content: Vec<usize> = (0..n).filter(|&r| mask[r] == 1).collect();
    let (mut a, mut c) = (Vec::new(), Vec::new());
    for &r in &content {
        a.extend_from_slice(&want[r * d..(r + 1) * d]);
        c.extend_from_slice(&unmasked[r * d..(r + 1) * d]);
    }
    eprintln!(
        "  (an UNMASKED reference run differs on the {} content rows by max_abs={:.3e}, \
         cosine={:.6} - brain implements the masked Wan contract)",
        content.len(),
        max_abs(&a, &c),
        cosine(&a, &c)
    );

    rep.finish("umt5-xxl");
}

/// The tokenizer and the encoder meet here: the ids brain's own unigram
/// tokenizer produces for Wan's prompts must be the ids the golden encoder ran
/// on. Cheap (no weights, no forward), and it is the seam a prompt-to-video
/// pipeline would otherwise only discover as bad video.
#[test]
fn brain_tokenizes_the_golden_prompts_to_the_golden_ids() {
    let Some(g) = Golden::open("golden/wan/t5/encoder.safetensors") else { return };
    let dir = std::env::var("BRAIN_WAN_TOKENIZER").ok().filter(|s| !s.is_empty());
    let Some(dir) = dir else {
        brain_testutil::skip("set BRAIN_WAN_TOKENIZER to a google/umt5-xxl tokenizer directory");
        return;
    };
    if !Path::new(&format!("{dir}/tokenizer.json")).exists() {
        brain_testutil::skip(&format!("{dir}/tokenizer.json not found"));
        return;
    }
    let tok = data::unigram::UnigramTokenizer::from_dir(&dir).expect("load tokenizer");

    // Mirrors `ENCODER_PROMPTS` in tools/goldens/wan_t5_dump_reference.py.
    let prompts = [
        "A belgian malinois running on a paved highway, cinematic lighting",
        "两只可爱的橘猫戴着墨镜,在阳光下的沙滩上散步。",
    ];
    let ids = g.ids("input_ids");
    let mask = g.ids("attention_mask");
    assert_eq!(g.shape("input_ids"), &vec![prompts.len(), TEXT_LEN]);
    for (p, text) in prompts.iter().enumerate() {
        let cleaned = data::unigram::UnigramTokenizer::clean_whitespace(text);
        let (got_ids, got_mask) = tok.encode_padded(&cleaned, TEXT_LEN);
        assert_eq!(got_ids, ids[p * TEXT_LEN..(p + 1) * TEXT_LEN], "prompt {p} ids");
        assert_eq!(got_mask, mask[p * TEXT_LEN..(p + 1) * TEXT_LEN], "prompt {p} mask");
        eprintln!("  prompt {p}: {} tokens, ids match the golden", got_mask.iter().sum::<u32>());
    }
}
