// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Forward parity for the state encoder, climbed one rung at a time against a
//! live `transformers.BertModel` run on the released all-MiniLM-L6-v2 weights.
//!
//! Regenerate the goldens with `tools/goldens/minilm_dump_reference.py`; the
//! checkpoint comes from the model store
//! (`brain pull sentence-transformers/all-MiniLM-L6-v2`).
//!
//! The ladder is the point: `emb` -> `l0.attn_ctx` -> `l0.attn_out` ->
//! `l0.ffn_act` -> every layer -> `pooled_mean`. A single end-of-stack
//! comparison would report one wrong number; these say WHICH stage produced
//! it. `l0.attn_ctx` in particular is where a q/k/v fusion in the wrong order
//! shows up, while the layer output can still look plausible.
//!
//! **Only the reference's UNPADDED positions are compared.** The golden's
//! second row carries six `[PAD]` positions because `transformers` takes a
//! rectangular batch; `decide::Encoder` packs spans instead and never computes
//! those rows at all (see `decide::model`'s docs). Comparing them would be
//! comparing against positions this architecture deliberately does not have.

use std::collections::HashMap;
use std::path::Path;

use decide::config::EncoderConfig;
use decide::kern::PIPELINES;
use decide::model::Encoder;

/// Per-element tolerance, set from what this actually measures rather than
/// from a round number: the worst stage over both backends is `l0.ffn_act` at
/// **1.34e-5** on the CPU backend and **6.20e-6** on the GPU one, with every
/// other rung between 8e-7 and 6e-6. 5e-5 leaves under 4x of headroom, so a
/// real divergence trips this instead of being absorbed by a loose bound.
///
/// The gap is reduction ORDER, not precision: the reference sums an fp32 GEMM
/// one way and these kernels another, and a 6-layer post-LayerNorm stack
/// carries that difference forward. Every failure message prints the observed
/// worst case and where it was, so widening is always visible.
const ATOL: f32 = 5e-5;

/// One dumped tensor: its shape and its values.
type Tensor = (Vec<usize>, Vec<f32>);

struct Golden {
    t: HashMap<String, Tensor>,
}

impl Golden {
    fn get(&self, name: &str) -> &Tensor {
        self.t.get(name).unwrap_or_else(|| panic!("golden has no tensor {name:?}"))
    }
}

fn load() -> Option<(Golden, EncoderConfig, HashMap<String, Vec<f32>>)> {
    let dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden"));
    let dump = dir.join("minilm.safetensors");
    let dumper = "tools/goldens/minilm_dump_reference.py";
    if !dump.exists() {
        brain_testutil::skip(&format!("{} missing - run {dumper}", dump.display()));
        return None;
    }
    let src = brain_testutil::golden::Source::open(dir, dumper)?;

    let hf = brain_testutil::model_dir("sentence-transformers/all-MiniLM-L6-v2")?;
    let cfg_path = Path::new(&hf).join("config.json");
    if !cfg_path.exists() {
        brain_testutil::skip(&format!(
            "{} missing - run `brain pull sentence-transformers/all-MiniLM-L6-v2`",
            cfg_path.display()
        ));
        return None;
    }
    let cfg = decide::import::config_from_hf(&std::fs::read_to_string(&cfg_path).unwrap())
        .expect("parse the checkpoint's config.json");

    // Refuse a golden dumped from a DIFFERENT tier rather than comparing
    // against it: every field here fixes a shape in the dump.
    if !src.require(&[
        ("hidden_size", cfg.d_model as i64),
        ("num_hidden_layers", cfg.n_layers as i64),
        ("num_attention_heads", cfg.n_heads as i64),
        ("intermediate_size", cfg.d_ff as i64),
        ("vocab_size", cfg.vocab as i64),
        ("max_position_embeddings", cfg.max_positions as i64),
        ("type_vocab_size", cfg.type_vocab as i64),
    ]) {
        return None;
    }

    let raw = checkpoint::safetensors::read(dump.to_str().unwrap()).expect("read golden");
    let mut t = HashMap::new();
    for x in raw {
        t.insert(x.name.clone(), (x.shape.clone(), x.data));
    }

    let weights = checkpoint::safetensors::read(Path::new(&hf).join("model.safetensors").to_str().unwrap())
        .expect("read checkpoint");
    let init = decide::import::brain_init_from_hf(weights, &cfg).expect("import the checkpoint");
    Some((Golden { t }, cfg, init))
}

/// The golden's rectangular batch, re-laid as the packed form this encoder
/// takes: the flat token and segment streams, the `(row0, len)` span of each
/// original row within them, and those rows' real lengths (which is what says
/// where the reference's padding starts).
struct Packed {
    ids: Vec<u32>,
    types: Vec<u32>,
    spans: Vec<(u32, u32)>,
    lens: Vec<usize>,
}

fn packed(g: &Golden) -> Packed {
    let (shape, ids) = g.get("input_ids");
    let (_, mask) = g.get("attention_mask");
    let (_, types) = g.get("token_type_ids");
    let (b, t) = (shape[0], shape[1]);
    let (mut pid, mut ptype, mut spans, mut lens) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for r in 0..b {
        let len = (0..t).filter(|&c| mask[r * t + c] != 0.0).count();
        // Padding is a suffix in this dump; a hole would silently change what
        // is being compared, so it is asserted rather than assumed.
        assert!(
            (0..len).all(|c| mask[r * t + c] != 0.0),
            "row {r}'s attention mask has a hole - the packing below assumes a suffix"
        );
        spans.push((pid.len() as u32, len as u32));
        lens.push(len);
        for c in 0..len {
            pid.push(ids[r * t + c] as u32);
            ptype.push(types[r * t + c] as u32);
        }
    }
    Packed { ids: pid, types: ptype, spans, lens }
}

/// Compare a packed `[rows, width]` result against the golden's rectangular
/// `[b, t, width]`, valid positions only.
fn cmp(name: &str, got: &[f32], g: &Golden, key: &str, lens: &[usize], width: usize) {
    let (shape, want) = g.get(key);
    let t = shape[1];
    let mut worst = 0.0f32;
    let mut at = (0, 0, 0);
    let mut row = 0usize;
    for (r, &len) in lens.iter().enumerate() {
        for c in 0..len {
            for k in 0..width {
                let a = got[row * width + k];
                let b = want[(r * t + c) * width + k];
                let d = (a - b).abs();
                if d > worst {
                    worst = d;
                    at = (r, c, k);
                }
            }
            row += 1;
        }
    }
    assert!(
        worst <= ATOL,
        "{name}: max |diff| {worst:.3e} > {ATOL:.1e} at row {} pos {} channel {}",
        at.0,
        at.1,
        at.2
    );
    eprintln!("  {name:14} max |diff| {worst:.3e}");
}

#[test]
fn minilm_forward_matches_the_reference_stage_by_stage() {
    let Some((g, cfg, init)) = load() else { return };
    let p = packed(&g);
    let (lens, max_span) = (&p.lens, p.spans.iter().map(|&(_, l)| l).max().unwrap());

    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let mut enc = Encoder::new_on(gpu, cfg.clone(), p.ids.len() as u32, max_span, &init);
    enc.set_batch(&p.ids, &p.types, &p.spans);
    enc.forward();

    let h = cfg.d_model as usize;
    cmp("emb", &enc.embeddings(), &g, "emb", lens, h);
    cmp("l0.attn_ctx", &enc.attn_ctx(0), &g, "l0.attn_ctx", lens, h);
    cmp("l0.attn_out", &enc.attn_out(0), &g, "l0.attn_out", lens, h);
    cmp("l0.ffn_act", &enc.ffn_act(0), &g, "l0.ffn_act", lens, cfg.d_ff as usize);
    for l in 0..cfg.n_layers as usize {
        cmp(&format!("layer.{l}"), &enc.layer_out(l), &g, &format!("layer.{l}"), lens, h);
    }

    // The sentence-transformer head. Packing makes this exact: there are no
    // pad rows in the mean, so nothing has to be excluded from it - which is
    // also why it is compared against the reference's OWN mask-aware mean.
    let (_, want) = g.get("pooled_mean");
    let got = enc.pooled_mean();
    let worst = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(worst <= ATOL, "pooled_mean: max |diff| {worst:.3e} > {ATOL:.1e}");
    eprintln!("  pooled_mean    max |diff| {worst:.3e}");
}

/// The importer must account for every tensor in the released checkpoint:
/// mapped, or dropped for a stated reason. A silent drop is how a model loads
/// cleanly and runs with an untrained tensor.
#[test]
fn every_checkpoint_tensor_is_accounted_for() {
    let Some(hf) = brain_testutil::model_dir("sentence-transformers/all-MiniLM-L6-v2") else { return };
    let w = Path::new(&hf).join("model.safetensors");
    if !w.exists() {
        brain_testutil::skip(&format!(
            "{} missing - run `brain pull sentence-transformers/all-MiniLM-L6-v2`",
            w.display()
        ));
        return;
    }
    let cfg = decide::import::config_from_hf(
        &std::fs::read_to_string(Path::new(&hf).join("config.json")).unwrap(),
    )
    .unwrap();
    let tensors = checkpoint::safetensors::read(w.to_str().unwrap()).unwrap();
    let init = decide::import::brain_init_from_hf(tensors, &cfg).expect("two-way coverage");
    for (name, shape) in cfg.tensor_manifest() {
        let want: usize = shape.iter().product();
        assert_eq!(init[&name].len(), want, "{name}");
    }
}
