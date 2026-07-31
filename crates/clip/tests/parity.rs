// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Stage-by-stage forward parity for the CLIP encoder family against the
//! goldens dumped by `tools/clip_dump_reference.py`.
//!
//! Fixtures live under `$BRAIN_TESTDATA` (default `<repo>/testdata`) in
//! `clip/`; the reference WEIGHTS are named by env var. Every test SKIPS itself
//! (never fails) when its fixture or its weights are absent:
//!
//! ```text
//! BRAIN_SDXL=/path/to/sdxl-base-1.0          # text_encoder/ + text_encoder_2/
//! BRAIN_EVA_CLIP=/path/to/EVA02_CLIP_L_336_psz14_s6B.pt
//! ```
//!
//! Gate: cosine >= 0.9999 per stage. Text stages are ALSO split by row
//! population (content rows vs right-pad rows) — pad rows are causally isolated
//! and would otherwise flatter the aggregate number.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use clip::config::{ClipTextConfig, EvaVisionConfig};
use clip::model::{ClipText, EvaVision, TextTap, VisionTap};

const GATE: f64 = 0.9999;

fn testdata(rel: &str) -> PathBuf {
    let root = std::env::var("BRAIN_TESTDATA")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata").to_string());
    Path::new(&root).join(rel)
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

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (&x, &y) in got.iter().zip(want) {
        num += (x as f64 - y as f64).powi(2);
        den += (y as f64).powi(2);
    }
    if den == 0.0 {
        return 0.0;
    }
    (num / den).sqrt()
}

/// Collected results so a run prints one table and fails once, at the end —
/// a single failing stage must not hide the twenty behind it.
#[derive(Default)]
struct Report {
    rows: Vec<(String, f64, f64, f64)>,
    failures: Vec<String>,
}

impl Report {
    fn check(&mut self, stage: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{stage}: len {} != golden {}", got.len(), want.len());
        let (c, m, r) = (cosine(got, want), max_abs(got, want), rel_l2(got, want));
        eprintln!("  {stage:<34} cosine={c:.8}  max_abs={m:.3e}  rel_l2={r:.3e}");
        self.rows.push((stage.to_string(), c, m, r));
        if !(c >= GATE) {
            self.failures.push(format!("{stage}: cosine {c:.8} < {GATE}"));
        }
    }

    /// Same stage, split by row population. `keep` selects rows.
    fn check_rows(
        &mut self,
        stage: &str,
        got: &[f32],
        want: &[f32],
        width: usize,
        keep: &dyn Fn(usize) -> bool,
    ) {
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
        eprintln!("{what}: {} stages checked, {} failed", self.rows.len(), self.failures.len());
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
            eprintln!("SKIP: golden {} absent", p.display());
            return None;
        }
        Some(Golden { t: checkpoint::safetensors::read(p.to_str().unwrap()).expect("read golden") })
    }
    fn get(&self, name: &str) -> &Vec<f32> {
        &self.t.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("golden tensor {name}")).data
    }
    fn ids(&self, name: &str) -> Vec<u32> {
        self.get(name).iter().map(|&x| x as u32).collect()
    }
    fn has(&self, name: &str) -> bool {
        self.t.iter().any(|t| t.name == name)
    }
}

fn env_path(var: &str) -> Option<PathBuf> {
    let v = std::env::var(var).ok().filter(|s| !s.is_empty())?;
    let p = PathBuf::from(v);
    if !p.exists() {
        eprintln!("SKIP: {var}={} not found", p.display());
        return None;
    }
    Some(p)
}

fn to_map(t: clip::import::Tensors) -> HashMap<String, Vec<f32>> {
    t.into_iter().map(|(k, (_, d))| (k, d)).collect()
}

/// The q or k region of brain's fused `[N, 3W]` qkv, un-permuted back into the
/// reference's interleaved head-channel order — the direct proof that
/// `EvaVisionConfig::head_perm` is the exact channel relabelling it claims.
fn unpermute_region(qkv: &[f32], rows: usize, w: usize, off: usize, heads: usize, perm: &[usize]) -> Vec<f32> {
    let hd = w / heads;
    let mut out = vec![0.0f32; rows * w];
    for t in 0..rows {
        for h in 0..heads {
            for d in 0..hd {
                out[t * w + h * hd + perm[d]] = qkv[t * 3 * w + off + h * hd + d];
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// text towers + the SDXL conditioning they compose into
// ---------------------------------------------------------------------------

fn run_text_tower(
    name: &str,
    cfg: ClipTextConfig,
    weights_dir: &Path,
    g: &Golden,
) -> (ClipText, Vec<f32>, Vec<f32>) {
    let ids = g.ids("input_ids");
    let mask = g.ids("attention_mask");
    let eos = g.ids("eos_index");
    let t = cfg.max_positions;
    let b = (ids.len() as u32) / t;

    let tensors = clip::import::read_text_encoder(weights_dir).expect("read text encoder");
    let src_count = tensors.len();
    let imported = clip::import::import_text(tensors, &cfg).expect("import_text");
    eprintln!("{name}: {src_count} source tensors -> {} parameters", imported.len());
    assert_eq!(imported.len(), cfg.tensor_manifest().len());

    let model = ClipText::new_on(
        gpu_core::testgpu::dev(clip::model::TEXT_PIPELINES),
        cfg.clone(),
        b,
        t,
        &to_map(imported),
    );
    model.set_tokens(&ids);
    model.forward();

    let h = cfg.hidden as usize;
    let inter = cfg.intermediate as usize;
    let n = (b * t) as usize;
    let mut rep = Report::default();
    eprintln!("{name} stages (B={b}, T={t}):");

    rep.check("pos.weight", &model.ps.read_weight(&model.gpu, "pos.weight"), g.get("pos_embed"));
    rep.check("tok_embed", &model.read_tok_embed(), g.get("tok_embed"));
    rep.check("embed", &model.read_x(0), g.get("embed"));

    // layer-0 internals
    let qkv = model.read_layer_tap(0, TextTap::Qkv);
    let slice = |off: usize| -> Vec<f32> {
        (0..n).flat_map(|r| qkv[r * 3 * h + off..r * 3 * h + off + h].to_vec()).collect()
    };
    rep.check("l0_ln1", &model.read_layer_tap(0, TextTap::Ln1), g.get("l0_ln1"));
    rep.check("l0_q", &slice(0), g.get("l0_q"));
    rep.check("l0_k", &slice(h), g.get("l0_k"));
    rep.check("l0_v", &slice(2 * h), g.get("l0_v"));
    rep.check("l0_attn_out", &model.read_layer_tap(0, TextTap::AttnOut), g.get("l0_attn_out"));
    rep.check("l0_ln2", &model.read_layer_tap(0, TextTap::Ln2), g.get("l0_ln2"));
    rep.check("l0_mlp_fc1", &model.read_layer_tap(0, TextTap::Fc1), g.get("l0_mlp_fc1"));
    rep.check("l0_mlp_out", &model.read_layer_tap(0, TextTap::MlpOut), g.get("l0_mlp_out"));
    rep.check("l0_out", &model.read_x(1), g.get("l0_out"));

    // every encoder layer
    for l in 0..cfg.layers as usize {
        rep.check(&format!("layer{l}_out"), &model.read_x(l + 1), g.get(&format!("layer{l}_out")));
    }

    let hidden = model.read_hidden();
    let pen = model.read_penultimate();
    rep.check("last_hidden_state", &hidden, g.get("last_hidden_state"));
    rep.check("penultimate", &pen, g.get("penultimate"));
    rep.check("pooled", &model.read_pooled(), g.get("pooled"));
    if let Some(te) = model.read_text_embeds() {
        rep.check("text_embeds", &te, g.get("text_embeds"));
    }

    // Row-population split: pad rows are causally isolated, so the aggregate
    // cosine is dominated by content rows unless they are separated.
    let content = |r: usize| mask[r] == 1;
    let pad = |r: usize| mask[r] == 0;
    rep.check_rows("last_hidden_state[content]", &hidden, g.get("last_hidden_state"), h, &content);
    rep.check_rows("last_hidden_state[pad]", &hidden, g.get("last_hidden_state"), h, &pad);
    rep.check_rows("penultimate[content]", &pen, g.get("penultimate"), h, &content);
    rep.check_rows("penultimate[pad]", &pen, g.get("penultimate"), h, &pad);
    let fc1 = model.read_layer_tap(0, TextTap::Fc1);
    rep.check_rows("l0_mlp_fc1[content]", &fc1, g.get("l0_mlp_fc1"), inter, &content);

    // The EOS pooling row the model derived must be the one the reference used.
    for s in 0..b as usize {
        let want = eos[s] as usize;
        let row = &ids[s * t as usize..(s + 1) * t as usize];
        let got = row.iter().position(|&x| x == cfg.eos_id).expect("eos in row");
        assert_eq!(got, want, "{name}: sample {s} eos row {got} != golden {want}");
    }

    rep.finish(name);
    (model, hidden, pen)
}

#[test]
fn sdxl_text_towers_and_conditioning_parity() {
    let Some(sdxl) = env_path("BRAIN_SDXL") else {
        eprintln!("SKIP: set BRAIN_SDXL to the sdxl-base-1.0 directory");
        return;
    };
    let (Some(gl), Some(gg)) =
        (Golden::open("clip/clip_l/text.safetensors"), Golden::open("clip/openclip_bigg/text.safetensors"))
    else {
        return;
    };

    let cfg_l = ClipTextConfig::clip_l();
    let cfg_g = ClipTextConfig::openclip_bigg();
    assert_eq!(cfg_l.penultimate_layer(), 10, "CLIP-L penultimate layer index");
    assert_eq!(cfg_g.penultimate_layer(), 30, "bigG penultimate layer index");

    let (_ml, _hl, pen_l) =
        run_text_tower("clip_l", cfg_l.clone(), &sdxl.join("text_encoder"), &gl);
    let (mg, _hg, pen_g) =
        run_text_tower("openclip_bigg", cfg_g.clone(), &sdxl.join("text_encoder_2"), &gg);

    // SDXL conditioning: prompt_embeds = concat(CLIP-L penultimate, bigG
    // penultimate) on the channel axis; pooled_prompt_embeds = bigG's PROJECTED
    // EOS pooling (CLIP-L's pooled output is unused by SDXL).
    let Some(gc) = Golden::open("clip/sdxl/cond.safetensors") else { return };
    let (hl, hg) = (cfg_l.hidden as usize, cfg_g.hidden as usize);
    let rows = pen_l.len() / hl;
    let mut cat = Vec::with_capacity(rows * (hl + hg));
    for r in 0..rows {
        cat.extend_from_slice(&pen_l[r * hl..(r + 1) * hl]);
        cat.extend_from_slice(&pen_g[r * hg..(r + 1) * hg]);
    }
    let mut rep = Report::default();
    eprintln!("sdxl conditioning:");
    rep.check("prompt_embeds", &cat, gc.get("prompt_embeds"));
    rep.check("pooled_prompt_embeds", &mg.read_text_embeds().unwrap(), gc.get("pooled_prompt_embeds"));
    rep.finish("sdxl");
}

// ---------------------------------------------------------------------------
// EVA02 image tower
// ---------------------------------------------------------------------------

#[test]
fn eva02_l336_image_tower_parity() {
    let Some(ckpt) = env_path("BRAIN_EVA_CLIP") else {
        eprintln!("SKIP: set BRAIN_EVA_CLIP to EVA02_CLIP_L_336_psz14_s6B.pt");
        return;
    };
    let Some(g) = Golden::open("clip/eva02_l336/image.safetensors") else { return };

    let cfg = EvaVisionConfig::eva02_l336();
    let tensors = checkpoint::torchpt::read(ckpt.to_str().unwrap()).expect("read eva .pt");
    let src = tensors.len();
    let (imported, rep_i) = clip::import::import_eva_visual(tensors, &cfg).expect("import_eva_visual");
    eprintln!(
        "eva02: {src} source tensors -> {} params, {} rope buffers recomputed-not-imported, {} non-visual",
        imported.len(),
        rep_i.skipped_rope_buffers,
        rep_i.skipped_non_visual
    );
    assert_eq!(rep_i.skipped_rope_buffers, 2 * (cfg.layers as usize + 1));
    assert_eq!(imported.len(), cfg.tensor_manifest().len());

    // The recomputed RoPE tables must reproduce the checkpoint's own fp16
    // buffers (which the reference discards) to fp16 resolution — an
    // independent check that the frequency construction is right. BOTH tables:
    // cos alone cannot distinguish an angle from its negation.
    {
        let (cos, sin) = cfg.rope_tables();
        let perm = cfg.head_perm();
        let half = cfg.rope_half() as usize;
        let np = cfg.num_patches() as usize;
        let hd = cfg.head_dim() as usize;
        for (tag, tab) in [("rope_freqs_cos", &cos), ("rope_freqs_sin", &sin)] {
            let want = g.get(tag);
            let mut got = vec![0.0f32; np * hd];
            for t in 0..np {
                for d in 0..half {
                    // permuted pair (d, d+half) == reference channels (perm[d], perm[d+half])
                    got[t * hd + perm[d]] = tab[t * half + d];
                    got[t * hd + perm[d + half]] = tab[t * half + d];
                }
            }
            let c = cosine(&got, want);
            eprintln!("  {tag} (recomputed vs reference)  cosine={c:.8} max_abs={:.3e}", max_abs(&got, want));
            assert!(c >= GATE, "{tag} table cosine {c:.8}");
        }
    }

    let weights = to_map(imported);
    let model = EvaVision::new_on(
        gpu_core::testgpu::dev(clip::model::VISION_PIPELINES),
        cfg.clone(),
        1,
        &weights,
    );
    model.set_pixels(g.get("pixel_values"));
    model.forward();

    let w = cfg.width as usize;
    let seq = cfg.seq_len() as usize;
    let np = cfg.num_patches() as usize;
    let heads = cfg.heads as usize;
    let hd = cfg.head_dim() as usize;
    let perm = cfg.head_perm();
    let mut rep = Report::default();
    eprintln!("eva02_l336 stages (B=1, T={seq}):");

    // `patch_embed`: the conv output is held in NCHW `[W, num_patches]` and
    // transposed into rows 1.. of the block input by `nchw_nlc`, so the golden
    // (NLC `[num_patches, W]`, pre-bias-free — the reference's PatchEmbed conv
    // HAS its bias) is compared against the NLC view PLUS the patch bias.
    {
        let nchw = model.read_patch_nchw();
        let bias = model.ps.read_weight(&model.gpu, "patch.bias");
        let mut nlc = vec![0.0f32; np * w];
        for ch in 0..w {
            for l in 0..np {
                nlc[l * w + ch] = nchw[ch * np + l] + bias[ch];
            }
        }
        rep.check("patch_embed", &nlc, g.get("patch_embed"));
    }
    rep.check("block_in", &model.read_x(0), g.get("block_in"));
    rep.check("b0_norm1", &model.read_block_tap(0, VisionTap::Norm1), g.get("b0_norm1"));

    // q/k live in brain's permuted head-channel order and are rotated IN PLACE,
    // so the un-permuted cls row proves the projection + permutation and the
    // un-permuted patch rows prove the RoPE.
    let qkv = model.read_block_tap(0, VisionTap::Qkv);
    let q_un = unpermute_region(&qkv, seq, w, 0, heads, &perm);
    let k_un = unpermute_region(&qkv, seq, w, w, heads, &perm);
    let v = unpermute_region(&qkv, seq, w, 2 * w, heads, &(0..hd).collect::<Vec<_>>());
    rep.check("b0_q[cls row, pre-rope]", &q_un[..w], &g.get("b0_q")[..w]);
    rep.check("b0_k[cls row, pre-rope]", &k_un[..w], &g.get("b0_k")[..w]);
    rep.check("b0_v", &v, g.get("b0_v"));
    // golden rope_{q,k}_out are [heads, num_patches, head_dim]
    for (tag, un, want) in
        [("b0_rope_q_out", &q_un, g.get("b0_rope_q_out")), ("b0_rope_k_out", &k_un, g.get("b0_rope_k_out"))]
    {
        let mut got = vec![0.0f32; heads * np * hd];
        for h in 0..heads {
            for t in 0..np {
                let src = (t + 1) * w + h * hd;
                got[(h * np + t) * hd..(h * np + t) * hd + hd].copy_from_slice(&un[src..src + hd]);
            }
        }
        rep.check(tag, &got, want);
    }

    rep.check("b0_attn_ctx", &model.read_block_tap(0, VisionTap::Ctx), g.get("b0_attn_ctx"));
    rep.check("b0_inner_ln", &model.read_block_tap(0, VisionTap::InnerLn), g.get("b0_inner_ln"));
    rep.check("b0_attn_proj", &model.read_block_tap(0, VisionTap::AttnProj), g.get("b0_attn_proj"));
    rep.check("b0_norm2", &model.read_block_tap(0, VisionTap::Norm2), g.get("b0_norm2"));
    rep.check("b0_mlp_w1", &model.read_block_tap(0, VisionTap::W1), g.get("b0_mlp_w1"));
    rep.check("b0_mlp_w2", &model.read_block_tap(0, VisionTap::W2), g.get("b0_mlp_w2"));
    rep.check("b0_mlp_ffn_ln", &model.read_block_tap(0, VisionTap::FfnLn), g.get("b0_mlp_ffn_ln"));
    rep.check("b0_mlp_out", &model.read_block_tap(0, VisionTap::MlpOut), g.get("b0_mlp_out"));

    for l in 0..cfg.layers as usize {
        rep.check(&format!("block{l}_out"), &model.read_x(l + 1), g.get(&format!("block{l}_out")));
    }
    rep.check("norm_out", &model.read_norm_out(), g.get("norm_out"));
    rep.check("head_out", &model.read_head_out(), g.get("head_out"));
    if g.has("cls_embed") {
        rep.check("cls_embed", &model.read_head_out(), g.get("cls_embed"));
    }
    rep.check("cls_embed_l2norm", &model.read_cls_embed_l2norm(), g.get("cls_embed_l2norm"));

    // PuLID's id_vit_hidden taps: outputs of blocks 3, 7, 11, 15, 19.
    for (i, &blk) in EvaVisionConfig::PULID_TAPS.iter().enumerate() {
        rep.check(&format!("pulid_hidden{i} (block{blk}_out)"), &model.read_x(blk as usize + 1), g.get(&format!("pulid_hidden{i}")));
    }

    rep.finish("eva02_l336");

    // ---- B=2 -----------------------------------------------------------
    // The stem (nchw_nlc / patch bias / cls copy), the RoPE regions and the
    // head are `step_sliced` dispatches whose binding offsets are computed per
    // sample; at B=1 every one of those offsets is 0, so the arithmetic is
    // completely ungated. Feeding the SAME image twice must reproduce the
    // golden in BOTH rows — a wrong offset, a wrong `pos_add` modulo or a
    // wrong `bsz` in the attention trio all break this and nothing else.
    drop(model);
    let batched =
        EvaVision::new_on(gpu_core::testgpu::dev(clip::model::VISION_PIPELINES), cfg.clone(), 2, &weights);
    let px = g.get("pixel_values");
    let mut px2 = Vec::with_capacity(px.len() * 2);
    px2.extend_from_slice(px);
    px2.extend_from_slice(px);
    batched.set_pixels(&px2);
    batched.forward();

    let mut rep2 = Report::default();
    eprintln!("eva02_l336 batched (B=2, both samples = the golden image):");
    let both = |r: &mut Report, tag: &str, got: &[f32], want: &[f32]| {
        assert_eq!(got.len(), 2 * want.len(), "{tag}: batched length");
        r.check(&format!("{tag}[s0]"), &got[..want.len()], want);
        r.check(&format!("{tag}[s1]"), &got[want.len()..], want);
    };
    both(&mut rep2, "block_in", &batched.read_x(0), g.get("block_in"));
    both(&mut rep2, "block0_out", &batched.read_x(1), g.get("block0_out"));
    both(&mut rep2, "block23_out", &batched.read_x(cfg.layers as usize), g.get("block23_out"));
    both(&mut rep2, "norm_out", &batched.read_norm_out(), g.get("norm_out"));
    both(&mut rep2, "head_out", &batched.read_head_out(), g.get("head_out"));
    both(&mut rep2, "cls_embed_l2norm", &batched.read_cls_embed_l2norm(), g.get("cls_embed_l2norm"));
    rep2.finish("eva02_l336 batched");
}
