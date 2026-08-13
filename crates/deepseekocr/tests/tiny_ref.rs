// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **The composite's gate**: stage-by-stage forward parity against the
//! checkpoint-free DeepSeek-OCR reference dump, at dims chosen to break every
//! coincidence the real model hides.
//!
//! ```text
//! python3 tools/goldens/deepseek_ocr_dump_reference.py --out testdata/deepseek-ocr
//! ```
//!
//! writes `testdata/deepseek-ocr/tiny/{ckpt/model.safetensors, golden.safetensors}`
//! and `manifest-tiny.json`. No released checkpoint is involved: the reference
//! weights are seeded random and the whole reference forward is written out
//! longhand in that script, so this gate exists long before any real weights do.
//! Fixtures resolve from `$BRAIN_TESTDATA`; the test SKIPS itself when absent.
//!
//! ## What this proves that the three sub-crates' own gates cannot
//!
//! `sam1`, `clip::ClipVision` and `deepseekv2` are each gradient-checked and
//! (for CLIP) bit-identity-checked in isolation. None of that says the three
//! COMPOSE: the flatten between SAM and CLIP, the class-token drop, the concat
//! ORDER, the projector and the decoder splice are all new here, and every one
//! of them is a layout question whose wrong answer produces a plausible tensor.
//! The fixture's dims are what make each one visible:
//!
//! | confusion | at real scale | here |
//! |---|---|---|
//! | concat order `[clip, comp]` vs `[comp, clip]` | both halves 1024 wide -- invisible | 14 vs 11, and each half is asserted against its own source tap |
//! | compressor grid transposed | 16x16 -- invisible | 4x2 |
//! | SAM grid transposed | 64x64 -- invisible | 13x7 |
//! | window pad dropped | 14 divides 64 -- never pads | 4x3 over 13x7 pads 53 positions |
//! | CLIP position resample | native == run grid, so it is the identity | 3x3 -> 4x2 (height up, width down) |
//! | `norm_topk_prob` true vs false | -- | the two gates differ by 0.084; the RAW one is asserted |
//!
//! ## The fixture's one non-model tensor
//!
//! `clip.patch_bypass` is a widening `Linear(c_out, clip_width)` that exists
//! ONLY because this fixture forces `c_out != clip_width`. At real scale both
//! are 1024 and the compressor output is CLIP's patch token verbatim, so the
//! production config path asserts that equality and REFUSES the bridge
//! (`DeepseekOcrConfig::check_real_scale_shaped`). It is carried here as
//! `patch_bypass: true`, which only a fixture may set.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use checkpoint::safetensors::StTensor;
use deepseekocr::config::{BYPASS_B, BYPASS_W, IMAGE_NEWLINE, PROJECTOR_B, PROJECTOR_W, VIEW_SEPARATOR};
use deepseekocr::prompt::Prompt;
use deepseekocr::rows::{RowPlan, Src, ViewGrid};
use deepseekocr::{DeepseekOcr, DeepseekOcrConfig};

/// fp32 end to end with no quantization anywhere: anything below this is a bug,
/// not noise. (The worst observed tap is ~1e-8 away from 1.0.)
const GATE: f64 = 0.999_99;

fn testdata(rel: &str) -> PathBuf {
    brain_testutil::testdata_path(rel)
}

fn load(path: &Path) -> HashMap<String, StTensor> {
    checkpoint::safetensors::read(path.to_str().expect("utf-8 path"))
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
        .into_iter()
        .map(|t| (t.name.clone(), t))
        .collect()
}

// ---------------------------------------------------------------------------
// the parity accumulator (this repo's `Report { rows, floor }` convention)
// ---------------------------------------------------------------------------

fn compare(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len(), "length mismatch {} vs {}", a.len(), b.len());
    let (mut dot, mut na, mut nb, mut mx) = (0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        mx = mx.max((x - y).abs());
    }
    if na == 0.0 && nb == 0.0 {
        return (1.0, 0.0);
    }
    (dot / (na.sqrt() * nb.sqrt()).max(1e-30), mx)
}

struct Report {
    rows: Vec<(String, f64, f64)>,
    floor: f64,
}

impl Report {
    fn new(floor: f64) -> Report {
        Report { rows: Vec::new(), floor }
    }
    /// Cosine AND max_abs, always both: cosine alone cannot see a scale error,
    /// max_abs alone cannot see a shape error.
    fn check(&mut self, name: &str, got: &[f32], want: &[f32]) {
        let (c, m) = compare(got, want);
        println!("  {name:<22} cos {c:.10}  max_abs {m:.3e}  n={}", want.len());
        // A tap that is identically zero on both sides compares "perfectly" and
        // proves nothing; the fixture's random weights make that a real defect.
        assert!(want.iter().any(|v| v.abs() > 1e-6), "{name}: the REFERENCE tap is all ~zero -- degenerate comparison");
        self.rows.push((name.to_string(), c, m));
    }
    fn against(&mut self, name: &str, got: &[f32], golden: &HashMap<String, StTensor>) {
        let want = golden.get(name).unwrap_or_else(|| panic!("golden tap {name} missing"));
        self.check(name, got, &want.data);
    }
    fn finish(self, label: &str) {
        let worst = self.rows.iter().min_by(|a, b| a.1.total_cmp(&b.1)).expect("no stages compared");
        println!("  [{label}] {} taps, worst cosine {:.10} at {}", self.rows.len(), worst.1, worst.0);
        let bad: Vec<&(String, f64, f64)> = self.rows.iter().filter(|r| r.1.is_nan() || r.1 < self.floor).collect();
        assert!(bad.is_empty(), "{label}: {} tap(s) below cosine {}: {bad:?}", bad.len(), self.floor);
    }
}

// ---------------------------------------------------------------------------
// import: the reference's flat attribute-path names -> brain's manifests
// ---------------------------------------------------------------------------

/// Translate the dump's checkpoint into one flat init map covering all four
/// manifests (SAM, CLIP, glue, decoder), with **two-way coverage**: every source
/// tensor must be consumed and every declared tensor must be produced. Without
/// both halves a renamed leaf is a silently zero-filled parameter.
///
/// Three real shape translations happen here, and they are the same ones a real
/// `deepseekocr::import` will do:
///
/// * CLIP's separate `q_proj`/`k_proj`/`v_proj` become one fused `attn.qkv`
///   (`q | k | v` row-major), which is what the mmproj itself carries.
/// * `patch_embed.weight` is declared by CLIP's manifest but never dispatched on
///   the `PatchSource::Tokens` path, and the reference therefore has no such
///   tensor; it is zero-filled, which is exactly what the bypass means.
/// * The decoder's dense leading layer and its MoE layers name their MLPs
///   differently on both sides, so the split is driven by
///   `DeepseekV2Config::is_moe_layer`, not by string sniffing.
fn build_init(cfg: &DeepseekOcrConfig, ck: &HashMap<String, StTensor>) -> HashMap<String, Vec<f32>> {
    let mut used: HashSet<String> = HashSet::new();
    let mut out: HashMap<String, Vec<f32>> = HashMap::new();
    let take = |name: &str, used: &mut HashSet<String>| -> Vec<f32> {
        used.insert(name.to_string());
        ck.get(name).unwrap_or_else(|| panic!("checkpoint tensor {name} missing")).data.clone()
    };

    // ---- SAM tower (+ neck + compressor) ----
    let put = |k: &str, v: Vec<f32>, out: &mut HashMap<String, Vec<f32>>| {
        assert!(out.insert(k.to_string(), v).is_none(), "{k} produced twice");
    };
    put("vision.sam.patch_embed.weight", take("sam.patch_embed.proj.weight", &mut used), &mut out);
    put("vision.sam.patch_embed.bias", take("sam.patch_embed.proj.bias", &mut used), &mut out);
    put("vision.sam.pos_embed", take("sam.pos_embed", &mut used), &mut out);
    for l in 0..cfg.sam.n_layers {
        for leaf in ["norm1.weight", "norm1.bias", "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight", "attn.proj.bias", "attn.rel_pos_h", "attn.rel_pos_w", "norm2.weight", "norm2.bias"] {
            put(&format!("vision.sam.blocks.{l}.{leaf}"), take(&format!("sam.blocks.{l}.{leaf}"), &mut used), &mut out);
        }
        for (dst, src) in [("mlp.fc1.weight", "mlp.lin1.weight"), ("mlp.fc1.bias", "mlp.lin1.bias"), ("mlp.fc2.weight", "mlp.lin2.weight"), ("mlp.fc2.bias", "mlp.lin2.bias")] {
            put(&format!("vision.sam.blocks.{l}.{dst}"), take(&format!("sam.blocks.{l}.{src}"), &mut used), &mut out);
        }
    }
    for (dst, src) in [
        ("vision.sam.neck.conv1.weight", "compressor.neck.0.weight"),
        ("vision.sam.neck.norm1.weight", "compressor.neck.1.weight"),
        ("vision.sam.neck.norm1.bias", "compressor.neck.1.bias"),
        ("vision.sam.neck.conv2.weight", "compressor.neck.2.weight"),
        ("vision.sam.neck.norm2.weight", "compressor.neck.3.weight"),
        ("vision.sam.neck.norm2.bias", "compressor.neck.3.bias"),
        ("vision.sam.compress.conv1.weight", "compressor.net_2.weight"),
        ("vision.sam.compress.conv2.weight", "compressor.net_3.weight"),
    ] {
        put(dst, take(src, &mut used), &mut out);
    }

    // ---- CLIP tower ----
    let w = cfg.clip_width() as usize;
    put("class_embed", take("clip.class_embedding", &mut used), &mut out);
    put("pos_embed", take("clip.position_embedding", &mut used), &mut out);
    put("pre_norm.weight", take("clip.pre_norm.weight", &mut used), &mut out);
    put("pre_norm.bias", take("clip.pre_norm.bias", &mut used), &mut out);
    // Declared by the manifest, never dispatched on the bypass path.
    let p = cfg.clip.patch() as usize;
    put("patch_embed.weight", vec![0.0; w * 3 * p * p], &mut out);
    for l in 0..cfg.clip.layers() {
        for leaf in ["norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias"] {
            put(&format!("blocks.{l}.{leaf}"), take(&format!("clip.blocks.{l}.{leaf}"), &mut used), &mut out);
        }
        for (dst, src) in [("attn.proj.weight", "attn.out_proj.weight"), ("attn.proj.bias", "attn.out_proj.bias")] {
            put(&format!("blocks.{l}.{dst}"), take(&format!("clip.blocks.{l}.{src}"), &mut used), &mut out);
        }
        // q | k | v, row-major -- the fused layout `model::vit` binds at offsets
        // 0, D and 2D of the qkv buffer.
        for (dst, tail) in [("attn.qkv.weight", "weight"), ("attn.qkv.bias", "bias")] {
            let mut fused = Vec::new();
            for proj in ["q_proj", "k_proj", "v_proj"] {
                fused.extend(take(&format!("clip.blocks.{l}.attn.{proj}.{tail}"), &mut used));
            }
            put(&format!("blocks.{l}.{dst}"), fused, &mut out);
        }
    }

    // ---- glue: the projector, and the fixture's widening bridge ----
    put(PROJECTOR_W, take("projector.weight", &mut used), &mut out);
    put(PROJECTOR_B, take("projector.bias", &mut used), &mut out);
    // Declared by the glue manifest, never dispatched on THIS path -- the same
    // situation as `patch_embed.weight` above. The reference dump is
    // single-view and contiguous, so it has no image-block layout and therefore
    // no `image_newline` / `view_separator` tensors to compare against; the
    // interleaved layout that reads them is gated separately
    // (`deepseekocr::layout`'s unit tests and `tests/real_weight_layout.rs`).
    // Zero-filled, which is what "this fixture places no such row" means.
    for name in [IMAGE_NEWLINE, VIEW_SEPARATOR] {
        put(name, vec![0.0; cfg.projector_out() as usize], &mut out);
    }
    if cfg.patch_bypass {
        put(BYPASS_W, take("clip.patch_bypass.weight", &mut used), &mut out);
        put(BYPASS_B, take("clip.patch_bypass.bias", &mut used), &mut out);
    }

    // ---- decoder ----
    put("tok.weight", take("decoder.embed_tokens.weight", &mut used), &mut out);
    for l in 0..cfg.decoder.n_layers() {
        let s = |leaf: &str| format!("decoder.layers.{l}.{leaf}");
        put(&format!("blocks.{l}.ln1.weight"), take(&s("input_layernorm.weight"), &mut used), &mut out);
        put(&format!("blocks.{l}.ln2.weight"), take(&s("post_attention_layernorm.weight"), &mut used), &mut out);
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            let leaf = format!("self_attn.{proj}.weight");
            put(&format!("blocks.{l}.{leaf}"), take(&s(&leaf), &mut used), &mut out);
        }
        if cfg.decoder.is_moe_layer(l) {
            put(&format!("blocks.{l}.mlp.router.weight"), take(&s("mlp.gate.weight"), &mut used), &mut out);
            for e in 0..cfg.decoder.n_experts() {
                for nm in ["gate", "up", "down"] {
                    put(&format!("blocks.{l}.mlp.experts.{e}.{nm}.weight"), take(&s(&format!("mlp.experts.{e}.{nm}_proj.weight")), &mut used), &mut out);
                }
            }
            for nm in ["gate", "up", "down"] {
                put(&format!("blocks.{l}.mlp.shared.{nm}.weight"), take(&s(&format!("mlp.shared_experts.{nm}_proj.weight")), &mut used), &mut out);
            }
        } else {
            for nm in ["gate", "up", "down"] {
                put(&format!("blocks.{l}.mlp.{nm}.weight"), take(&s(&format!("mlp.{nm}_proj.weight")), &mut used), &mut out);
            }
        }
    }
    put("norm.weight", take("decoder.norm.weight", &mut used), &mut out);
    put("lm_head.weight", take("decoder.lm_head.weight", &mut used), &mut out);

    // ---- two-way coverage ----
    let unused: Vec<&String> = ck.keys().filter(|k| !used.contains(*k)).collect();
    assert!(unused.is_empty(), "checkpoint tensors nothing consumed: {unused:?}");
    let mut declared: Vec<(String, usize)> = cfg.sam.param_list();
    declared.extend(cfg.clip.tensor_manifest().into_iter().map(|(n, s)| (n, s.iter().product::<usize>())));
    declared.extend(cfg.glue_param_list());
    declared.extend(cfg.decoder.param_list());
    for (name, numel) in &declared {
        let v = out.get(name).unwrap_or_else(|| panic!("declared tensor {name} was not produced"));
        assert_eq!(v.len(), *numel, "{name}: {} elements, manifest says {numel}", v.len());
    }
    assert_eq!(out.len(), declared.len(), "produced {} tensors for {} declared", out.len(), declared.len());
    out
}

// ---------------------------------------------------------------------------
// the gate
// ---------------------------------------------------------------------------

#[test]
fn tiny_reference_stage_parity() {
    let ckpt = testdata("deepseek-ocr/tiny/ckpt/model.safetensors");
    let golden_path = testdata("deepseek-ocr/tiny/golden.safetensors");
    if !ckpt.exists() || !golden_path.exists() {
        eprintln!("SKIP: {} / {} absent -- run tools/goldens/deepseek_ocr_dump_reference.py", ckpt.display(), golden_path.display());
        return;
    }
    let ck = load(&ckpt);
    let g = load(&golden_path);
    let shape = |n: &str| g.get(n).unwrap_or_else(|| panic!("golden tap {n} missing")).shape.clone();

    let cfg = DeepseekOcrConfig::tiny();
    cfg.check();

    // ---- the config IS the fixture's, asserted against the dump's own shapes ----
    // A fixture regenerated at other dims must fail HERE, loudly, rather than
    // compare the wrong tensors further down (the `t5/tests/tiny_ref.rs` rule).
    assert_eq!(shape("image"), vec![1, 3, cfg.sam.image_h() as usize, cfg.sam.image_w() as usize], "image");
    assert_eq!(shape("sam_embed"), vec![1, cfg.sam.grid_h as usize, cfg.sam.grid_w as usize, cfg.sam.d_model as usize], "SAM grid/width");
    let (gh, gw) = cfg.token_grid();
    assert_eq!(shape("compressor_out"), vec![1, cfg.compressor_out() as usize, gh as usize, gw as usize], "compressor grid");
    assert_eq!(shape("clip_out"), vec![1, (cfg.image_tokens() + 1) as usize, cfg.clip_width() as usize], "CLIP seq/width");
    assert_eq!(shape("clip_pos_full"), vec![(cfg.image_tokens() + 1) as usize, cfg.clip_width() as usize], "CLIP position rows");
    assert_eq!(ck["clip.position_embedding"].shape, vec![1, cfg.clip.n_positions() as usize, cfg.clip_width() as usize], "CLIP native position table");
    assert_eq!(shape("vision_concat"), vec![1, cfg.image_tokens() as usize, cfg.projector_in() as usize], "projector input width");
    assert_eq!(shape("projector_out"), vec![1, cfg.image_tokens() as usize, cfg.projector_out() as usize], "projector output width");
    let logits_shape = shape("logits");
    assert_eq!(logits_shape[0], 1, "the composite is batch-1 (sam1 is a single-image tower)");
    let seq = logits_shape[1] as u32;
    assert_eq!(logits_shape[2], cfg.decoder.vocab() as usize, "vocab");
    assert_eq!(shape("moe_router_logits"), vec![1, seq as usize, cfg.decoder.n_experts() as usize], "router width");
    assert_eq!(shape("moe_gate_raw")[2], cfg.decoder.top_k() as usize, "top_k");
    // The inequalities the fixture exists for, re-asserted on THIS side.
    assert_ne!(cfg.compressor_out(), cfg.clip_width(), "the concat-order gate needs distinct widths");
    assert_ne!(gh, gw, "the compressor grid must be non-square");
    assert_ne!(cfg.sam.grid_h, cfg.sam.grid_w, "the SAM grid must be non-square");

    let init = build_init(&cfg, &ck);

    // ---- the image-token run, read off the reference's own input_ids ----
    let ids: Vec<u32> = g["input_ids"].data.iter().map(|v| *v as u32).collect();
    assert_eq!(ids.len(), seq as usize);
    let image_token_id = 0u32;
    let row0 = ids.iter().position(|v| *v == image_token_id).expect("no image placeholder token") as u32;
    assert_eq!(ids.iter().filter(|v| **v == image_token_id).count() as u32, cfg.image_tokens(), "placeholder count");
    assert!(
        ids[row0 as usize..(row0 + cfg.image_tokens()) as usize].iter().all(|v| *v == image_token_id),
        "the placeholders must be one contiguous run"
    );

    // Forward parity only: build the INFERENCE composite, so this gate also
    // proves the frozen build records the same forward graph as the trainable
    // one (the backward gate below is the other half).
    let m = DeepseekOcr::new(&|p| gpu_core::testgpu::dev(p), cfg.clone(), &init, 7, seq, row0, false);
    assert_eq!(m.image_run(), (row0, cfg.image_tokens()));
    m.set_tokens_unsupervised(&ids);
    let loss = m.forward(&g["image"].data);
    assert_eq!(loss, 0.0, "every target is IGNORE, so the parity run reports no loss");

    let enc = m.encoder();
    let sam = enc.sam();
    let clip = enc.clip();
    let rd = |b: &gpu_core::DeviceBuffer, n: usize| sam.gpu.read(b, n);
    let rows_c = (sam.cfg.rows() * sam.cfg.d_model) as usize;
    let mut r = Report::new(GATE);

    // ---- SAM tower ----
    r.against("sam_patch_embed", &rd(sam.patch_tokens(), rows_c), &g);
    r.against("sam_embed", &rd(sam.embedded_tokens(), rows_c), &g);
    for l in 0..cfg.sam.n_layers as usize {
        // `block_norm1` carries `rows + 1` rows: the last is `WindowPlan::padded`'s
        // zero sentinel, which is not a token and has no reference counterpart.
        r.against(&format!("sam_b{l}_norm1"), &rd(sam.block_norm1(l), rows_c), &g);
        r.against(&format!("sam_b{l}_attn_res"), &rd(sam.block_attn_res(l), rows_c), &g);
        r.against(&format!("sam_b{l}_out"), &rd(sam.block_out(l), rows_c), &g);
    }

    // ---- neck + compressor ----
    let stages = sam.neck_stages();
    for (tap, (_, buf, n)) in ["neck_conv1", "neck_ln1", "neck_conv2", "neck_ln2", "comp_net2", "compressor_out"].iter().zip(&stages) {
        r.against(tap, &rd(buf, *n), &g);
    }
    r.against("compressor_flat", &enc.read_compressor_flat(), &g);

    // ---- CLIP (patch embed bypassed) ----
    r.against("clip_patch_tokens", &enc.read_patch_tokens().expect("the fixture carries the bridge"), &g);
    r.against("clip_pos_full", &clip.read_pos_full(), &g);
    r.against("clip_tokens", &clip.read_x0(), &g);
    for l in 0..cfg.clip.layers() as usize {
        r.against(&format!("clip_b{l}_out"), &clip.read_block_out(l), &g);
    }
    r.against("clip_out", &clip.read_output(), &g);
    r.against("clip_spatial", &enc.read_clip_spatial(), &g);

    // ---- the concat, and the half-slice assertion that pins its ORDER ----
    let concat = enc.read_vision_concat();
    r.against("vision_concat", &concat, &g);
    let (w, cout, n) = (cfg.clip_width() as usize, cfg.compressor_out() as usize, cfg.image_tokens() as usize);
    let half = |lo: usize, hi: usize| -> Vec<f32> { (0..n).flat_map(|i| concat[i * (w + cout) + lo..i * (w + cout) + hi].to_vec()).collect() };
    let (c_low, _) = compare(&half(0, w), &g["clip_spatial"].data);
    let (c_high, _) = compare(&half(w, w + cout), &g["compressor_flat"].data);
    println!("  concat halves: low(=clip_spatial) cos {c_low:.10}   high(=compressor_flat) cos {c_high:.10}");
    assert!(c_low > GATE, "the concat's LOW half is not clip_spatial (cos {c_low}) -- the order is [clip, compressor]");
    assert!(c_high > GATE, "the concat's HIGH half is not compressor_flat (cos {c_high})");
    // And the swap really is distinguishable at these widths.
    assert_ne!(w, cout, "a swapped concat would not even typecheck if the widths were equal");

    r.against("projector_out", &enc.read_projector_out(), &g);

    // ---- decoder ----
    r.against("decoder_input", &m.read_decoder_input(), &g);
    for l in 0..cfg.decoder.n_layers() as usize {
        r.against(&format!("dec_l{l}_out"), &m.decoder().read_res(l + 1), &g);
    }
    r.against("decoder_final_norm", &m.decoder().read_final_norm(), &g);
    r.against("moe_router_logits", &m.decoder().read_router_logits(1).expect("layer 1 is MoE"), &g);
    assert!(m.decoder().read_router_logits(0).is_none(), "layer 0 must be dense");
    r.against("logits", &m.read_logits(), &g);

    // ---- the router gate is the RAW softmax probability, not the renormalized one ----
    let e = cfg.decoder.n_experts() as usize;
    let k = cfg.decoder.top_k() as usize;
    let idx: Vec<usize> = g["moe_topk_idx"].data.iter().map(|v| *v as usize).collect();
    let dense = |vals: &[f32]| -> Vec<f32> {
        let mut out = vec![0f32; seq as usize * e];
        for row in 0..seq as usize {
            for j in 0..k {
                out[row * e + idx[row * k + j]] = vals[row * k + j];
            }
        }
        out
    };
    let want_raw = dense(&g["moe_gate_raw"].data);
    let want_renorm = dense(&g["moe_gate_renorm"].data);
    let got = m.decoder().read_router_gate(1).expect("layer 1 is MoE");
    r.check("moe_gate_raw", &got, &want_raw);
    let (c_renorm, m_renorm) = compare(&got, &want_renorm);
    println!("  moe_gate (renormalized control) cos {c_renorm:.10}  max_abs {m_renorm:.3e}");
    assert!(m_renorm > 1e-3, "raw and renormalized gates are indistinguishable here -- the probe proves nothing");
    let (_, m_raw) = compare(&got, &want_raw);
    assert!(m_raw < 1e-4, "the decoder did not use the RAW gate (max_abs {m_raw:e} vs the renormalized control's {m_renorm:e})");

    r.finish("deepseek-ocr/tiny");
}

/// The composite's BACKWARD, end to end: the decoder's cross-entropy gradient
/// must reach every stage's parameters and the input pixels.
///
/// The three sub-models' own adjoints are gradient-checked in their own crates;
/// what is new here is the splice, the concat and the projector, so those are
/// what this checks numerically -- a directional finite difference of the
/// composite loss w.r.t. the glue parameters, plus the two properties a
/// finite-difference check alone cannot see: that the gradient reaches the far
/// end (SAM's first tensor, and the image), and that a descent step on it
/// actually lowers the loss.
#[test]
fn composite_backward_reaches_the_image_and_descends() {
    let ckpt = testdata("deepseek-ocr/tiny/ckpt/model.safetensors");
    let golden_path = testdata("deepseek-ocr/tiny/golden.safetensors");
    if !ckpt.exists() || !golden_path.exists() {
        eprintln!("SKIP: fixtures absent");
        return;
    }
    let ck = load(&ckpt);
    let g = load(&golden_path);
    let cfg = DeepseekOcrConfig::tiny();
    let init = build_init(&cfg, &ck);
    let ids: Vec<u32> = g["input_ids"].data.iter().map(|v| *v as u32).collect();
    let seq = ids.len() as u32;
    let row0 = ids.iter().position(|v| *v == 0).expect("placeholder") as u32;
    // Real (unmasked) next-token targets, so the loss is a real number the
    // gradient can be checked against.
    let mut targets: Vec<u32> = ids[1..].to_vec();
    targets.push(ids[0]);

    let m = DeepseekOcr::new(&|p| gpu_core::testgpu::dev(p), cfg.clone(), &init, 7, seq, row0, true);
    m.set_tokens(&ids, &targets);
    let image = &g["image"].data;

    let loss = |m: &DeepseekOcr| -> f64 {
        let l = m.forward(image);
        assert!(l.is_finite(), "loss is not finite");
        l as f64
    };
    let l0 = loss(&m);
    m.zero_grads();
    let _ = loss(&m);
    let d_image = m.backward();
    m.decoder().poll_wait();

    // The gradient reached the far end of the composite, through the splice, the
    // projector, the concat, CLIP's token seam and the whole SAM tower.
    assert_eq!(d_image.len(), image.len());
    assert!(d_image.iter().any(|v| v.abs() > 1e-9), "no gradient reached the input image");
    assert!(d_image.iter().all(|v| v.is_finite()));
    for name in ["vision.sam.patch_embed.weight", "vision.sam.blocks.0.attn.rel_pos_h"] {
        let gr = m.encoder().sam().read_grad(name);
        assert!(gr.iter().any(|v| v.abs() > 1e-9), "{name}: no gradient reached the SAM tower");
    }
    assert!(m.encoder().clip().read_grad("blocks.0.attn.qkv.weight").iter().any(|v| v.abs() > 1e-9), "no gradient reached CLIP");

    // The two learned image-block rows are declared by the glue manifest but
    // are NOT in this graph: `DeepseekOcr::new` splices the projector's 8 rows
    // contiguously, so no row reads `image_newline` or `view_separator` and
    // their gradients must be EXACTLY zero. Asserted rather than skipped --
    // "not in the graph" is a claim, and a nonzero here would mean the
    // contiguous path had quietly grown a layout.
    for name in [IMAGE_NEWLINE, VIEW_SEPARATOR] {
        let gr = m.encoder().read_glue_grad(name);
        assert!(gr.iter().all(|v| *v == 0.0), "{name}: the contiguous splice path must not touch it");
    }

    // ---- directional finite differences on the NEW parameters ----
    let mut rng = data::rng::Rng::new(0x51CE);
    let eps = 5e-4f32;
    for name in m.encoder().glue_param_names().into_iter().filter(|n| n != IMAGE_NEWLINE && n != VIEW_SEPARATOR) {
        let w0 = m.encoder().read_glue_weight(&name);
        let grad = m.encoder().read_glue_grad(&name);
        assert!(grad.iter().any(|v| *v != 0.0), "{name}: analytic gradient is exactly zero");
        let mut best = f32::INFINITY;
        for _ in 0..3 {
            let v: Vec<f32> = (0..w0.len()).map(|_| if rng.next_f32() < 0.5 { -1.0 } else { 1.0 }).collect();
            let analytic: f64 = grad.iter().zip(&v).map(|(a, b)| *a as f64 * *b as f64).sum();
            let step = |s: f32| -> f64 {
                let w: Vec<f32> = w0.iter().zip(&v).map(|(a, b)| a + s * eps * b).collect();
                m.encoder().write_glue_weight(&name, &w);
                loss(&m)
            };
            let numeric = (step(1.0) - step(-1.0)) / (2.0 * eps as f64);
            m.encoder().write_glue_weight(&name, &w0);
            let rel = ((analytic - numeric).abs() / analytic.abs().max(numeric.abs()).max(1e-6)) as f32;
            println!("  {name:<20} analytic {analytic:+.6e}  numeric {numeric:+.6e}  rel {rel:.3e}");
            best = best.min(rel);
        }
        assert!(best < 5e-2, "{name}: best relative gradient error {best:e}");
    }

    // ---- a descent step on the WHOLE composite lowers the loss ----
    let names: Vec<String> = m.encoder().glue_param_names().into_iter().filter(|n| n != IMAGE_NEWLINE && n != VIEW_SEPARATOR).collect();
    let gnorm: f32 = names.iter().map(|k| m.encoder().read_glue_grad(k).iter().map(|v| v * v).sum::<f32>()).sum::<f32>().sqrt();
    assert!(gnorm > 1e-6, "glue gradient is ~zero (norm {gnorm})");
    let lr = 1e-2 / gnorm;
    for k in &names {
        let w = m.encoder().read_glue_weight(k);
        let gr = m.encoder().read_glue_grad(k);
        let stepped: Vec<f32> = w.iter().zip(&gr).map(|(w, gv)| w - lr * gv).collect();
        m.encoder().write_glue_weight(k, &stepped);
    }
    let l1 = loss(&m);
    println!("  composite loss {l0:.6} -> {l1:.6} (glue grad norm {gnorm:.4})");
    assert!(l1 < l0, "a descent step did not decrease the composite loss: {l0} -> {l1}");
}

/// **LoRA descent smoke test.** Same fixture and the same shape of proof as
/// [`composite_backward_reaches_the_image_and_descends`] above, but the
/// decoder is built with `deepseek2::config::LoraCfg` set: its own base
/// weights (embeddings, norms, the four attention projections, the MoE
/// router/experts/shared expert, the untied head) are `Role::Frozen`, and the
/// ONLY trainable tensors anywhere in the composite are the decoder's
/// `.lora_a`/`.lora_b` adapter pairs on its four attention projections. A
/// plain gradient step on THOSE ALONE -- nothing else is ever written -- still
/// measurably lowers the composite loss, which is the whole point of Phase
/// 9's "LoRA finetune + descent smoke test": proving the parameter-efficient
/// training path is wired correctly end to end, not proving it trains to
/// convergence (see `crates/deepseekocr/src/train.rs` for the composite-level
/// merge helper this test drives, and `deepseek2::model`'s own doc for the
/// adapter mechanism -- composed entirely from the SAME matmul/axpy/grad_scale
/// kernels the base decoder's forward/backward already dispatch, no new
/// kernel).
///
/// **Why `build_init` runs against a NON-LoRA config.** `build_init`'s
/// two-way coverage check demands every declared tensor come from the
/// checkpoint and every checkpoint tensor be declared -- exactly right for
/// the base weights, and wrong for the adapters, which no checkpoint (real or
/// fixture) ever carries: LoRA is trained after import. So this test builds
/// the coverage-checked base map at `cfg_base` (lora: None, byte-identical to
/// every other test in this file) and only ADDS the adapter tensors on top,
/// through `deepseekocr::train::lora_init_map` -- the checkpoint coverage
/// guarantee is exactly as strong as the plain-training test's.
#[test]
fn composite_lora_backward_freezes_the_base_and_descends() {
    let ckpt = testdata("deepseek-ocr/tiny/ckpt/model.safetensors");
    let golden_path = testdata("deepseek-ocr/tiny/golden.safetensors");
    if !ckpt.exists() || !golden_path.exists() {
        eprintln!("SKIP: fixtures absent");
        return;
    }
    let ck = load(&ckpt);
    let g = load(&golden_path);
    let cfg_base = DeepseekOcrConfig::tiny();
    let base_init = build_init(&cfg_base, &ck);

    let mut cfg = cfg_base.clone();
    cfg.decoder.lora = Some(deepseek2::config::lora_cfg(2, 4.0));
    let init = deepseekocr::train::lora_init_map(&cfg, &base_init, 0x10_A_DEC);

    let ids: Vec<u32> = g["input_ids"].data.iter().map(|v| *v as u32).collect();
    let seq = ids.len() as u32;
    let row0 = ids.iter().position(|v| *v == 0).expect("placeholder") as u32;
    let mut targets: Vec<u32> = ids[1..].to_vec();
    targets.push(ids[0]);

    let m = DeepseekOcr::new(&|p| gpu_core::testgpu::dev(p), cfg.clone(), &init, 7, seq, row0, true);
    m.set_tokens(&ids, &targets);
    let image = &g["image"].data;

    let loss = |m: &DeepseekOcr| -> f64 {
        let l = m.forward(image);
        assert!(l.is_finite(), "loss is not finite");
        l as f64
    };

    // ---- warm up ONLY the decoder's own optimizer, a few steps ----
    //
    // A fresh LoRA adapter starts with `B == 0` (see `deepseek2::init::
    // init_adapters`'s doc: "so the adapter starts as an exact no-op delta"),
    // which makes `A`'s OWN gradient degenerate at step 0: `dA = (d_out ·
    // B)ᵀ · x` is EXACTLY zero whenever `B` is, regardless of whether the
    // adapter is wired correctly. That is not a bug this test should chase -
    // it is the same degeneracy `qwen3::gradcheck::check_qwen_lora`'s own doc
    // names ("a few AdamW steps run first so the zero-initialised B adapter
    // (and hence A's gradient) is non-trivial before the FD comparison") and
    // escapes the identical way: a few real optimizer steps BEFORE measuring
    // anything, through `m.decoder()`'s own `adamw_step` alone, so the
    // encoder is never touched by this warm-up either.
    for step in 1..=3u32 {
        m.zero_grads();
        let _ = loss(&m);
        let _ = m.backward();
        m.decoder().poll_wait();
        m.decoder().adamw_step(step, 5e-2, 0.0, Some(1.0), 1.0);
        m.decoder().poll_wait();
    }

    let l0 = loss(&m);
    m.zero_grads();
    let _ = loss(&m);
    let d_image = m.backward();
    m.decoder().poll_wait();
    assert!(d_image.iter().any(|v| v.abs() > 1e-9), "no gradient reached the input image through the LoRA-adapted decoder");

    // ---- the decoder base is truly Role::Frozen: its gradient buffer does not exist ----
    let base_has_no_grad = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| m.decoder().read_grad("blocks.0.self_attn.q_proj.weight")));
    assert!(base_has_no_grad.is_err(), "the decoder base's own gradient buffer must not exist under LoRA (it must be Role::Frozen)");

    // ---- the ONLY trainable tensors anywhere are the decoder's own adapters ----
    let names = m.decoder().param_names();
    assert!(!names.is_empty(), "no trainable parameters at all");
    assert!(
        names.iter().all(|n| n.ends_with(".lora_a") || n.ends_with(".lora_b")),
        "a non-adapter decoder tensor is trainable under LoRA: {names:?}"
    );

    // ---- every adapter got a real, nonzero gradient ----
    let mut gnorm_sq = 0f32;
    for n in &names {
        let gr = m.decoder().read_grad(n);
        assert!(gr.iter().any(|v| *v != 0.0), "{n}: analytic gradient is exactly zero");
        gnorm_sq += gr.iter().map(|v| v * v).sum::<f32>();
    }
    let gnorm = gnorm_sq.sqrt();
    assert!(gnorm > 1e-6, "adapter gradient is ~zero (norm {gnorm})");

    // ---- snapshot two weights this step must NOT touch: a decoder base tensor
    // (frozen by role) and an encoder tensor (never written by this test) ----
    let dec_base_before = m.decoder().read_weight("blocks.0.self_attn.q_proj.weight");
    let enc_before = m.encoder().sam().read_weight("vision.sam.patch_embed.weight");

    // ---- descend on the adapters ALONE ----
    let lr = 1e-1 / gnorm;
    for n in &names {
        let w = m.decoder().read_weight(n);
        let gr = m.decoder().read_grad(n);
        let stepped: Vec<f32> = w.iter().zip(&gr).map(|(w, gv)| w - lr * gv).collect();
        m.decoder().write_weight(n, &stepped);
    }
    let l1 = loss(&m);
    println!("  LoRA-only composite loss {l0:.6} -> {l1:.6} (adapter grad norm {gnorm:.4}, {} trainable tensors)", names.len());
    assert!(l1 < l0, "a LoRA-only descent step did not decrease the composite loss: {l0} -> {l1}");

    // ---- and the two untouched tensors really are byte-identical after it ----
    assert_eq!(m.decoder().read_weight("blocks.0.self_attn.q_proj.weight"), dec_base_before, "the decoder base moved under a LoRA-only step");
    assert_eq!(m.encoder().sam().read_weight("vision.sam.patch_embed.weight"), enc_before, "the vision encoder moved under a decoder-only LoRA step");
}

/// The **real interleaved layout** on the composite, checkpoint-free.
///
/// `DeepseekOcr::new_with_prompt` sizes the splice at the prompt's own
/// `n_rows` -- the whole image block, newline and separator rows included --
/// and fills it through `deepseekocr::layout::RowGather`. What this asserts is
/// the property that makes that path trustworthy and that no aggregate metric
/// can see: **every row of the spliced window is bit-identically its source**.
/// A projector row is the projector's own row, the newline rows are the learned
/// `vision.image_newline` vector, the separator row is
/// `vision.view_separator` -- exact equality, because these are copies and not
/// computed values.
///
/// The fixture's compressor grid is `4x2`, so its layout is written out here
/// rather than taken from `rows::row_plan` (which assumes the square views the
/// real model has). That is deliberate: `RowGather` is indifferent to how the
/// `Src` sequence was produced, and hand-writing it means this test also pins
/// the CONVENTION -- one newline after each token row, one separator ending the
/// view -- independently of `rows.rs`'s own unit tests.
///
/// The backward half is the other claim: the two learned rows, unreachable on
/// the contiguous path, are now in the graph and their gradients are the sum
/// over every row that read them.
#[test]
fn the_real_layout_splices_the_learned_rows_verbatim_and_trains_them() {
    let ckpt = testdata("deepseek-ocr/tiny/ckpt/model.safetensors");
    let golden_path = testdata("deepseek-ocr/tiny/golden.safetensors");
    if !ckpt.exists() || !golden_path.exists() {
        eprintln!("SKIP: fixtures absent");
        return;
    }
    let ck = load(&ckpt);
    let g = load(&golden_path);
    let cfg = DeepseekOcrConfig::tiny();
    let init = build_init(&cfg, &ck);
    let (gh, gw) = cfg.token_grid();
    assert_eq!((gh, gw), (4, 2));

    // ---- the layout: `gw` tokens then a newline, per token row, then one separator ----
    let mut plan_rows: Vec<Src> = Vec::new();
    for y in 0..gh {
        plan_rows.extend((0..gw).map(|x| Src::Projector(y * gw + x)));
        plan_rows.push(Src::Newline);
    }
    plan_rows.push(Src::Separator);
    assert_eq!(plan_rows.len(), (gh * (gw + 1) + 1) as usize);
    let n_rows = plan_rows.len() as u32;

    // A prompt around it. Ids are the fixture decoder's, not a tokenizer's --
    // `tokens_per_side` has no meaning for a non-square fixture grid and
    // `RowGather` never reads it, only `plan.rows`.
    let row0 = 1u32;
    let seq = row0 + n_rows + 2;
    let image_token_id = 0u32;
    let mut ids: Vec<u32> = vec![1];
    ids.extend(std::iter::repeat_n(image_token_id, n_rows as usize));
    ids.extend([2, 3]);
    assert_eq!(ids.len(), seq as usize);
    let prompt = Prompt {
        ids: ids.clone(),
        row0,
        n_rows,
        plan: RowPlan { rows: plan_rows.clone(), tokens_per_side: gw, grid: ViewGrid::global_only() },
    };

    let m = DeepseekOcr::new_with_prompt(&|p| gpu_core::testgpu::dev(p), cfg.clone(), &init, &init, 7, seq, &prompt, true);
    assert_eq!(m.image_run(), (row0, n_rows));
    assert_eq!(m.row_gather().expect("the layout path builds one").shared_row_counts(), (gh, 1));

    // Distinctive learned rows, so a swapped newline/separator cannot hide (the
    // fixture's own dump has neither tensor -- see `build_init`).
    let dm = cfg.projector_out() as usize;
    let newline: Vec<f32> = (0..dm).map(|c| -7.0 - c as f32).collect();
    let separator: Vec<f32> = (0..dm).map(|c| 13.0 + 2.0 * c as f32).collect();
    m.encoder().write_glue_weight(IMAGE_NEWLINE, &newline);
    m.encoder().write_glue_weight(VIEW_SEPARATOR, &separator);

    let mut targets: Vec<u32> = ids[1..].to_vec();
    targets.push(ids[0]);
    m.set_tokens(&ids, &targets);
    let image = &g["image"].data;
    let l0 = m.forward(image);
    assert!(l0.is_finite(), "loss is not finite");

    // ---- every spliced row is EXACTLY its source ----
    let res0 = m.read_decoder_input();
    let proj = m.encoder().read_projector_out();
    assert_eq!(proj.len(), (cfg.image_tokens() as usize) * dm);
    for (r, src) in plan_rows.iter().enumerate() {
        let at = (row0 as usize + r) * dm;
        let got = &res0[at..at + dm];
        let want: &[f32] = match *src {
            Src::Projector(i) => &proj[i as usize * dm..(i as usize + 1) * dm],
            Src::Newline => &newline,
            Src::Separator => &separator,
        };
        assert_eq!(got, want, "spliced row {r} ({src:?}) is not its source row");
    }
    // The rows OUTSIDE the block are still text embeddings, untouched.
    assert_ne!(&res0[..dm], &res0[row0 as usize * dm..(row0 as usize + 1) * dm], "the splice leaked past row0");
    println!("  {n_rows} spliced rows, all bit-identical to their source ({} projector, {gh} newline, 1 separator)", cfg.image_tokens());

    // ---- the two learned rows are now trained ----
    m.zero_grads();
    let _ = m.forward(image);
    let d_image = m.backward();
    m.decoder().poll_wait();
    assert!(d_image.iter().any(|v| v.abs() > 1e-9), "no gradient reached the input image");
    let d_nl = m.encoder().read_glue_grad(IMAGE_NEWLINE);
    let d_sep = m.encoder().read_glue_grad(VIEW_SEPARATOR);
    assert!(d_nl.iter().any(|v| v.abs() > 1e-9), "image_newline got no gradient on the layout path");
    assert!(d_sep.iter().any(|v| v.abs() > 1e-9), "view_separator got no gradient on the layout path");

    // Finite-difference them through the WHOLE composite -- the layout unit
    // tests (`deepseekocr::layout`) check the gather in isolation at rel ~1e-5;
    // what this adds is that it is wired into the REAL loss, with the decoder
    // and the encoder in between.
    //
    // `eps` is 5e-2, a hundred times the glue check above, and that is forced by
    // arithmetic rather than taste: the separator is ONE row of a 13-row block,
    // so its directional derivative is ~1e-3, and at eps = 5e-4 the loss moves
    // by ~1.4e-6 -- THREE ULP of an fp32 loss near 4.2. The measured numeric
    // derivative was then 1.4305e-3 against an analytic 1.1095e-3 (rel 2.2e-1),
    // i.e. the check was reading fp32 quantization, not curvature. A step big
    // enough to clear that quantum is the fix; the second-order error it buys
    // back is what the 5e-2 gate absorbs.
    let mut rng = data::rng::Rng::new(0xDEC0);
    let eps = 5e-2f32;
    for (name, grad) in [(IMAGE_NEWLINE, &d_nl), (VIEW_SEPARATOR, &d_sep)] {
        let w0 = m.encoder().read_glue_weight(name);
        let mut best = f32::INFINITY;
        for _ in 0..3 {
            let v: Vec<f32> = (0..w0.len()).map(|_| if rng.next_f32() < 0.5 { -1.0 } else { 1.0 }).collect();
            let analytic: f64 = grad.iter().zip(&v).map(|(a, b)| *a as f64 * *b as f64).sum();
            let step = |s: f32| -> f64 {
                let w: Vec<f32> = w0.iter().zip(&v).map(|(a, b)| a + s * eps * b).collect();
                m.encoder().write_glue_weight(name, &w);
                m.forward(image) as f64
            };
            let numeric = (step(1.0) - step(-1.0)) / (2.0 * eps as f64);
            m.encoder().write_glue_weight(name, &w0);
            let rel = ((analytic - numeric).abs() / analytic.abs().max(numeric.abs()).max(1e-6)) as f32;
            println!("  {name:<24} analytic {analytic:+.6e}  numeric {numeric:+.6e}  rel {rel:.3e}");
            assert!(analytic.abs() > 1e-6, "{name}: a ~zero directional derivative proves nothing");
            best = best.min(rel);
        }
        assert!(best < 5e-2, "{name}: best relative gradient error {best:e}");
    }
}

/// **Phase 8 split-device wiring gate**: `caps::Session::load` now builds the
/// vision encoder (SAM+CLIP+glue) on `Gpu::new_wgpu` and the decoder on
/// `Gpu::new_cpu` via [`DeepseekOcr::new_with_prompt_devices`], instead of one
/// `Gpu::new_cpu` factory for both (`crates/sam1`'s wgpu corruption at
/// 1024x1024/3+ blocks -- what forced the single CPU factory originally -- is
/// fixed and confirmed at real-weight scale, see `crates/sam1/tests/
/// wgpu_real_weight_parity.rs`). This is NOT a re-verification of wgpu's SAM
/// correctness -- that is that test's job. It is a wiring gate: build the SAME
/// tiny checkpoint-free fixture two ways, all-CPU
/// ([`DeepseekOcr::new_with_prompt`]) and split-device
/// ([`DeepseekOcr::new_with_prompt_devices`], vision on wgpu / decoder on
/// CPU), and assert the two forward outputs agree -- so a future edit that
/// crosses the wires (e.g. hands the decoder's `PIPELINES` to the vision
/// factory, or vice versa) fails loudly here instead of silently shipping.
#[test]
fn split_device_vision_wgpu_decoder_cpu_matches_all_cpu() {
    let ckpt = testdata("deepseek-ocr/tiny/ckpt/model.safetensors");
    let golden_path = testdata("deepseek-ocr/tiny/golden.safetensors");
    if !ckpt.exists() || !golden_path.exists() {
        eprintln!("SKIP: fixtures absent");
        return;
    }
    let ck = load(&ckpt);
    let g = load(&golden_path);
    let cfg = DeepseekOcrConfig::tiny();
    let init = build_init(&cfg, &ck);
    let (gh, gw) = cfg.token_grid();

    // Same interleaved real-layout construction as
    // `the_real_layout_splices_the_learned_rows_verbatim_and_trains_them`
    // above -- the shape `caps::Session::load` actually builds in production,
    // not the contiguous parity-only path.
    let mut plan_rows: Vec<Src> = Vec::new();
    for y in 0..gh {
        plan_rows.extend((0..gw).map(|x| Src::Projector(y * gw + x)));
        plan_rows.push(Src::Newline);
    }
    plan_rows.push(Src::Separator);
    let n_rows = plan_rows.len() as u32;

    let row0 = 1u32;
    let seq = row0 + n_rows + 2;
    let image_token_id = 0u32;
    let mut ids: Vec<u32> = vec![1];
    ids.extend(std::iter::repeat_n(image_token_id, n_rows as usize));
    ids.extend([2, 3]);
    let prompt = Prompt {
        ids: ids.clone(),
        row0,
        n_rows,
        plan: RowPlan { rows: plan_rows, tokens_per_side: gw, grid: ViewGrid::global_only() },
    };

    let cpu_dev = |p: &'static [(&'static str, &'static str)]| gpu_core::Gpu::new_cpu(p);
    let wgpu_dev = |p: &'static [(&'static str, &'static str)]| gpu_core::Gpu::new_wgpu(p);

    let m_cpu = DeepseekOcr::new_with_prompt(&cpu_dev, cfg.clone(), &init, &init, 7, seq, &prompt, false);
    m_cpu.set_tokens_unsupervised(&ids);
    let loss_cpu = m_cpu.forward(&g["image"].data);
    assert!(loss_cpu.is_finite());
    let logits_cpu = m_cpu.read_logits();
    let dec_in_cpu = m_cpu.read_decoder_input();

    let m_split = DeepseekOcr::new_with_prompt_devices(&wgpu_dev, &cpu_dev, cfg.clone(), &init, &init, 7, seq, &prompt, false);
    m_split.set_tokens_unsupervised(&ids);
    let loss_split = m_split.forward(&g["image"].data);
    assert!(loss_split.is_finite());
    let logits_split = m_split.read_logits();
    let dec_in_split = m_split.read_decoder_input();

    let (cos_logits, max_abs_logits) = compare(&logits_cpu, &logits_split);
    let (cos_dec_in, max_abs_dec_in) = compare(&dec_in_cpu, &dec_in_split);
    println!("split-device (vision=wgpu, decoder=cpu) vs all-cpu:");
    println!("  decoder_input cos {cos_dec_in:.10}  max_abs {max_abs_dec_in:.3e}");
    println!("  logits        cos {cos_logits:.10}  max_abs {max_abs_logits:.3e}");
    assert!(cos_dec_in > 0.9999, "the spliced decoder input diverges between backends (cos {cos_dec_in}) -- vision half disagrees");
    assert!(cos_logits > 0.9999, "final logits diverge between backends (cos {cos_logits})");
}
