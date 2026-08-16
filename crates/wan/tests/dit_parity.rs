// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Wan DiT parity, climbed in the order failures localise.
//!
//! 1. [`wan_dit_rope_tables_match_the_reference`] and
//!    [`wan_dit_runs_at_a_real_token_count`] - **no weights**, and they run
//!    first because they gate the two things that need nothing else: the
//!    three-axis RoPE tables, and that a 32,760-token graph can be built and
//!    submitted at all. The second is the one that would catch "correct at 320
//!    tokens, unallocatable at 480p".
//! 2. [`wan_dit_tiny_block_matches_reference`] - ONE block, replaying the
//!    reference's own hooked input.
//! 3. [`wan_dit_tiny_model_matches_reference`] /
//!    [`wan_dit_tiny_dev_matches_reference`] - the whole toy model, every
//!    block output tapped, on the host-orchestrated reference and on the
//!    device-resident engine.
//! 4. [`wan_dit_1_3b_matches_reference`] - the real 1.3B weights at 4,680
//!    tokens.
//!
//! Everything from (2) on needs `tools/goldens/wan_dit_dump_reference.py`'s
//! fixtures and skips loudly without them; (4) also needs the checkpoint.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brain_testutil::testdata;
use checkpoint::safetensors::StTensor;
use wan::block::WanBlock;
use wan::config::WanConfig;
use wan::dev::WanDitDev;
use wan::import::{dit_manifest, import_dit};
use wan::model::{Tensors, WanDit};

// ------------------------------------------------------------------ metrics

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    model::hostmath::cosine(a, b) as f64
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (&x, &y) in got.iter().zip(want) {
        num += (x as f64 - y as f64).powi(2);
        den += (y as f64).powi(2);
    }
    (num / den.max(f64::MIN_POSITIVE)).sqrt()
}

fn max_abs(got: &[f32], want: &[f32]) -> f32 {
    got.iter().zip(want).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max)
}

/// Report all three, always: cosine hides a scale error, rel_l2 hides a single
/// bad element, and max_abs hides a broad small bias.
fn report(label: &str, got: &[f32], want: &[f32], min_cos: f64, max_rel: f64) {
    assert_eq!(got.len(), want.len(), "{label}: {} values vs {}", got.len(), want.len());
    let (mut c, r, m) = (cosine(got, want), rel_l2(got, want), max_abs(got, want));
    // Bit-identical is a pass whatever the cosine says. It has to be spelled
    // out because cosine is 0/0 on two identically-zero populations, which is
    // exactly what a zero-initialised bias makes of the text encoding's pad
    // rows - reporting that as a failure would hide the case it was added for.
    let exact = m == 0.0;
    if exact {
        c = 1.0;
    }
    let note = if exact { "  (bit-identical)" } else { "" };
    eprintln!("{label}: cosine={c:.9}  rel_l2={r:.3e}  max_abs={m:.3e}{note}");
    assert!(c >= min_cos, "{label}: cosine {c:.9} < {min_cos}");
    assert!(r <= max_rel, "{label}: rel_l2 {r:.3e} > {max_rel:.0e}");
}

/// The text encoding has a real pad population - `text_len` is 512 and a prompt
/// is far shorter - so report the two row sets separately. A mask defect shows
/// up in the pad rows first and can be invisible in an all-rows average.
fn report_rows(label: &str, got: &[f32], want: &[f32], rows: usize, width: usize, content: usize, min_cos: f64, max_rel: f64) {
    let split = content * width;
    report(&format!("{label} [content rows 0..{content}]"), &got[..split], &want[..split], min_cos, max_rel);
    if content < rows {
        report(&format!("{label} [pad rows {content}..{rows}]"), &got[split..], &want[split..], min_cos, max_rel);
    }
}

// ------------------------------------------------------------------ fixtures

struct Fixture {
    t: Vec<StTensor>,
}

impl Fixture {
    fn open(rel: &str) -> Option<Fixture> {
        let p = testdata(rel);
        if !Path::new(&p).exists() {
            eprintln!("SKIP: fixture {p} absent - run tools/goldens/wan_dit_dump_reference.py");
            return None;
        }
        Some(Fixture { t: checkpoint::safetensors::read(&p).expect("read golden") })
    }
    fn get(&self, name: &str) -> &[f32] {
        &self.t.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("no golden {name}")).data
    }
    fn shape(&self, name: &str) -> &[usize] {
        &self.t.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("no golden {name}")).shape
    }
    fn has(&self, name: &str) -> bool {
        self.t.iter().any(|t| t.name == name)
    }
}

/// The toy config the dumper's `CFG_TINY` builds. Kept in sync by
/// [`wan_dit_tiny_weights_cover_the_manifest`], which fails if the fixture's
/// tensor set stops matching this config's manifest.
fn tiny_cfg() -> WanConfig {
    WanConfig {
        name: "tiny",
        dim: 64,
        ffn_dim: 128,
        num_heads: 2,
        num_layers: 3,
        in_channels: 4,
        out_channels: 4,
        freq_dim: 32,
        text_dim: 48,
        text_len: 16,
        ..WanConfig::t2v_1_3b()
    }
}

fn tiny_weights() -> Option<Tensors> {
    let p = testdata("golden/wan/dit/dit_tiny_weights.safetensors");
    if !Path::new(&p).exists() {
        eprintln!("SKIP: fixture {p} absent - run tools/goldens/wan_dit_dump_reference.py");
        return None;
    }
    let raw = checkpoint::safetensors::read(&p).expect("read tiny weights");
    Some(import_dit(raw, &tiny_cfg()).expect("import tiny weights"))
}

/// Deterministic synthetic weights covering a config's manifest exactly - no
/// file, and by construction no missing or extra name.
fn synthetic_weights(cfg: &WanConfig) -> Tensors {
    let mut t: Tensors = HashMap::new();
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    for (name, shape) in dit_manifest(cfg) {
        let n: usize = shape.iter().product();
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((state >> 33) as u32) as f32 / (1u64 << 31) as f32;
            v.push(0.2 * (u - 0.5));
        }
        // QK-norm gains near 1: a gain of ~0 would zero every downstream
        // activation and hide a real difference behind an all-zero compare.
        if name.contains("norm_q") || name.contains("norm_k") || name.ends_with("norm3.weight") {
            for x in v.iter_mut() {
                *x += 1.0;
            }
        }
        t.insert(name, (shape, v));
    }
    t
}

fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % 97) as f32 / 97.0 - 0.5) * 1.7).collect()
}

// ---------------------------------------------------------------- weight-free

/// The three-axis tables, against the reference's own `torch.polar` product.
#[test]
fn wan_dit_rope_tables_match_the_reference() {
    let Some(fx) = Fixture::open("golden/wan/dit/dit_tiny.safetensors") else { return };
    let g = fx.get("grid");
    let (f, h, w) = (g[0] as u32, g[1] as u32, g[2] as u32);
    let t = wan::rope::tables(&tiny_cfg(), f, h, w);
    report("rope cos", &t.cos, fx.get("rope_cos"), 0.9999999, 1e-6);
    report("rope sin", &t.sin, fx.get("rope_sin"), 0.9999999, 1e-6);
}

/// 81 frames at 480p: 32,760 tokens. A materialised score matrix there is
/// 51 GB across 12 heads against a 2047 MiB per-binding ceiling, so this is
/// the test that proves the real path can be built and submitted at all -
/// at toy widths, so it costs a second instead of an hour.
#[test]
fn wan_dit_runs_at_a_real_token_count() {
    let cfg = WanConfig {
        name: "tiny-480p",
        dim: 64,
        ffn_dim: 128,
        num_heads: 2,
        num_layers: 1,
        in_channels: 4,
        out_channels: 4,
        freq_dim: 32,
        text_dim: 48,
        ..WanConfig::t2v_1_3b()
    };
    // The real 480p latent extent: 81 frames -> 21 latent frames, 480x832 -> 60x104.
    let (f, h, w) = (21u32, 60u32, 104u32);
    let tokens = cfg.token_count(81, 832, 480).expect("81 frames is 1 + 4k");
    assert_eq!(tokens, 32_760);
    let wts = synthetic_weights(&cfg);
    let d = WanDitDev::build(&cfg, &wts, f, h, w, None, &[]);
    assert_eq!(d.tokens() as usize, tokens);
    eprintln!("built a {tokens}-token graph on {} ({:?})", d.gpu().kind(), wan::block::attn_mode(d.gpu()));
    d.set_context(&ramp(8 * cfg.text_dim), 8);
    let out = d.forward(&ramp((cfg.in_channels as u32 * f * h * w) as usize), 750.0);
    assert_eq!(out.len(), (cfg.out_channels as u32 * f * h * w) as usize);
    assert!(out.iter().all(|v| v.is_finite()), "the 32,760-token forward produced non-finite values");
    let (lo, hi) = out.iter().fold((f32::MAX, f32::MIN), |(l, hh), &v| (l.min(v), hh.max(v)));
    assert!(hi - lo > 1e-6, "output is constant ({lo}..{hi}) - the graph collapsed");
}

// --------------------------------------------------------------- tiny fixture

#[test]
fn wan_dit_tiny_weights_cover_the_manifest() {
    let Some(w) = tiny_weights() else { return };
    let cfg = tiny_cfg();
    assert_eq!(w.len(), dit_manifest(&cfg).len());
    assert_eq!(w.len(), 3 * 27 + 12 + 3);
}

/// One block, replaying the reference's own hooked input: the patch-embedded
/// token slab, the block's `e0`, the RoPE tables and the embedded text.
#[test]
fn wan_dit_tiny_block_matches_reference() {
    let Some(fx) = Fixture::open("golden/wan/dit/dit_tiny.safetensors") else { return };
    let Some(w) = tiny_weights() else { return };
    let cfg = tiny_cfg();
    let g = fx.get("grid");
    let (f, h, wd) = (g[0] as u32, g[1] as u32, g[2] as u32);
    let tokens = (f * h * wd) as usize;

    let blk = WanBlock::new(&cfg, &w, "blocks.0", tokens as u32, None);
    let out = blk.forward(fx.get("patch_embed"), fx.get("e0"), fx.get("rope_cos"), fx.get("rope_sin"), fx.get("text_embed"));
    report("block 0", &out, fx.get("block.0"), 0.999999, 1e-4);
}

fn run_tiny_model(device: Option<&str>) {
    let Some(fx) = Fixture::open("golden/wan/dit/dit_tiny.safetensors") else { return };
    let Some(w) = tiny_weights() else { return };
    let cfg = tiny_cfg();
    let ls = fx.shape("latent").to_vec();
    let (f, h, wd) = (ls[1] as u32, ls[2] as u32, ls[3] as u32);
    let ctx_rows = fx.shape("context")[0];
    let t = fx.get("timestep")[0];

    // Stage taps first, so a mismatch localises before the block stack runs.
    let (e, e0) = wan::model::timestep_cond(&cfg, &w, t);
    report("e (head modulation base)", &e, fx.get("e"), 0.9999999, 1e-5);
    report("e0 (block modulation)", &e0, fx.get("e0"), 0.9999999, 1e-5);
    let tokens = wan::model::embed_tokens(&cfg, &w, fx.get("latent"), f, h, wd);
    report("patch_embed", &tokens, fx.get("patch_embed"), 0.9999999, 1e-5);
    let ctx = wan::model::text_embed(&cfg, &w, fx.get("context"), ctx_rows);
    report_rows("text_embed", &ctx, fx.get("text_embed"), cfg.text_len, cfg.dim, ctx_rows, 0.9999999, 1e-5);

    let taps: Vec<usize> = (0..cfg.num_layers).collect();
    let m = WanDit::new(cfg.clone(), w, device);
    let (out, block_taps) = m.forward_taps(fx.get("latent"), f, h, wd, fx.get("context"), ctx_rows, t, &taps);
    for (l, v) in &block_taps {
        report(&format!("block.{l} ({})", device.unwrap_or("default")), v, fx.get(&format!("block.{l}")), 0.999999, 1e-4);
    }
    report(&format!("out ({})", device.unwrap_or("default")), &out, fx.get("out"), 0.999999, 1e-4);
}

#[test]
fn wan_dit_tiny_model_matches_reference() {
    run_tiny_model(None);
}

/// The CPU JIT cannot run the flash kernel's barriers, so this run takes the
/// query-chunked materialised path instead - the A/B that proves the two
/// attention implementations agree, without an environment variable that would
/// leak into every other test in the process.
#[test]
fn wan_dit_tiny_model_matches_reference_on_cpu() {
    run_tiny_model(Some("cpu"));
}

#[test]
fn wan_dit_tiny_dev_matches_reference() {
    let Some(fx) = Fixture::open("golden/wan/dit/dit_tiny.safetensors") else { return };
    let Some(w) = tiny_weights() else { return };
    let cfg = tiny_cfg();
    let ls = fx.shape("latent").to_vec();
    let (f, h, wd) = (ls[1] as u32, ls[2] as u32, ls[3] as u32);
    let ctx_rows = fx.shape("context")[0];

    let taps: Vec<usize> = (0..cfg.num_layers).collect();
    let d = WanDitDev::build(&cfg, &w, f, h, wd, None, &taps);
    d.set_context(fx.get("context"), ctx_rows);
    let out = d.forward(fx.get("latent"), fx.get("timestep")[0]);
    for l in 0..cfg.num_layers {
        report(&format!("dev block.{l}"), &d.read_tap(l).unwrap(), fx.get(&format!("block.{l}")), 0.999999, 1e-4);
    }
    report("dev out", &out, fx.get("out"), 0.999999, 1e-4);
}

// ------------------------------------------------------------ real 1.3B

/// The shipped transformer. `BRAIN_WAN_DIT` names a file or a directory of
/// shards; otherwise the same path inside `BRAIN_MODELS_DIR`. Both are
/// variables rather than a literal because a machine path baked into a test
/// passes on exactly one machine and skips silently on every other.
fn dit_shards() -> Option<Vec<PathBuf>> {
    let root = match std::env::var("BRAIN_WAN_DIT") {
        Ok(p) if !p.is_empty() => PathBuf::from(p),
        _ => {
            let store = std::env::var("BRAIN_MODELS_DIR").ok()?;
            let native = Path::new(&store).join("Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors");
            if native.exists() {
                native
            } else {
                Path::new(&store).join("Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer")
            }
        }
    };
    if root.is_file() {
        return Some(vec![root]);
    }
    if !root.is_dir() {
        return None;
    }
    let mut v: Vec<PathBuf> = std::fs::read_dir(&root)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    v.sort();
    (!v.is_empty()).then_some(v)
}

fn real_weights() -> Option<Tensors> {
    let Some(shards) = dit_shards() else {
        eprintln!("SKIP: set BRAIN_WAN_DIT (a file or a shard directory) or BRAIN_MODELS_DIR");
        return None;
    };
    let mut raw: Vec<StTensor> = Vec::new();
    for p in &shards {
        raw.extend(checkpoint::safetensors::read(&p.to_string_lossy()).unwrap_or_else(|e| panic!("read {}: {e}", p.display())));
    }
    let n = raw.len();
    let w = import_dit(raw, &WanConfig::t2v_1_3b()).expect("import DiT weights");
    eprintln!("imported {n} source tensors from {} shard(s) -> {} canonical", shards.len(), w.len());
    Some(w)
}

#[test]
fn wan_dit_import_covers_the_shipped_checkpoint() {
    let Some(w) = real_weights() else { return };
    let cfg = WanConfig::t2v_1_3b();
    assert_eq!(w.len(), dit_manifest(&cfg).len());
    assert_eq!(w.len(), 825);

    // Both directions, against the REAL names: drop one and the error must name
    // the canonical tensor it fed; add one the model never reads and it must be
    // reported rather than ignored.
    let stub = |skip: Option<&str>, add: Option<&str>| -> Vec<StTensor> {
        let mut v: Vec<StTensor> = dit_manifest(&cfg)
            .into_iter()
            .filter(|(n, _)| Some(n.as_str()) != skip)
            .map(|(n, s)| StTensor { name: wan::import::dit_native_to_diffusers(&n).expect("bijection"), shape: s.clone(), data: vec![0.0; s.iter().product()] })
            .collect();
        if let Some(a) = add {
            v.push(StTensor { name: a.into(), shape: vec![1], data: vec![0.0] });
        }
        v
    };
    let e = import_dit(stub(Some("blocks.11.cross_attn.norm_k.weight"), None), &cfg).unwrap_err();
    assert!(e.contains("blocks.11.cross_attn.norm_k.weight"), "{e}");
    let e = import_dit(stub(None, Some("blocks.0.norm1.weight")), &cfg).unwrap_err();
    assert!(e.contains("unmapped diffusers tensor"), "{e}");
}

/// The NATIVE path against the real checkpoint's tensor set.
///
/// `Wan-AI/Wan2.1-T2V-1.3B` is the registered default reference, so
/// `import_dit_native` is the canonical importer - but that repo ships the DiT
/// as one large file that need not be present for this test to mean something.
/// The shipped diffusers export's names, mapped into the reference name space,
/// ARE that file's tensor set, so importing them natively covers the same
/// ground: the sniffer must pick the native branch, and both error directions
/// must still name the offending tensor.
#[test]
fn wan_dit_native_import_covers_the_same_tensor_set() {
    let Some(w) = real_weights() else { return };
    let cfg = WanConfig::t2v_1_3b();
    // `real_weights` already mapped the shipped names into the reference name
    // space; the shapes come from the file, not from the manifest.
    let names: Vec<(String, Vec<usize>)> = w.iter().map(|(n, (s, _))| (n.clone(), s.clone())).collect();
    drop(w);
    assert_eq!(names.len(), 825);
    let make = |skip: Option<&str>, add: Option<&str>| -> Vec<StTensor> {
        let mut v: Vec<StTensor> = names
            .iter()
            .filter(|(n, _)| Some(n.as_str()) != skip)
            .map(|(n, s)| StTensor { name: n.clone(), shape: s.clone(), data: vec![0.0; s.iter().product()] })
            .collect();
        if let Some(a) = add {
            v.push(StTensor { name: a.into(), shape: vec![1], data: vec![0.0] });
        }
        v
    };
    // The sniffer must take the native branch (no `scale_shift_table` present).
    assert_eq!(import_dit(make(None, None), &cfg).expect("native import").len(), 825);
    let e = import_dit(make(Some("blocks.29.ffn.2.bias"), None), &cfg).unwrap_err();
    assert!(e.contains("blocks.29.ffn.2.bias"), "{e}");
    let e = import_dit(make(None, Some("blocks.0.norm2.weight")), &cfg).unwrap_err();
    assert!(e.contains("unused source tensors"), "{e}");
}

#[test]
fn wan_dit_1_3b_matches_reference() {
    let Some(fx) = Fixture::open("golden/wan/dit/dit_1_3b.safetensors") else { return };
    let Some(w) = real_weights() else { return };
    let cfg = WanConfig::t2v_1_3b();
    let ls = fx.shape("latent").to_vec();
    let (f, h, wd) = (ls[1] as u32, ls[2] as u32, ls[3] as u32);
    let ctx_rows = fx.shape("context")[0];
    let t = fx.get("timestep")[0];
    let g = fx.get("grid");
    let tokens = (g[0] * g[1] * g[2]) as usize;
    eprintln!("1.3B parity: latent {f}x{h}x{wd} -> {tokens} tokens, {ctx_rows} content text rows of {}", cfg.text_len);

    // Stage taps first: a host-side convention error localises here rather
    // than as a cosine deficit 30 blocks deep.
    let (e, e0) = wan::model::timestep_cond(&cfg, &w, t);
    report("e", &e, fx.get("e"), 0.9999999, 1e-5);
    report("e0", &e0, fx.get("e0"), 0.9999999, 1e-5);
    let toks = wan::model::embed_tokens(&cfg, &w, fx.get("latent"), f, h, wd);
    report("patch_embed", &toks, fx.get("patch_embed"), 0.9999999, 1e-5);
    let ctx = wan::model::text_embed(&cfg, &w, fx.get("context"), ctx_rows);
    report_rows("text_embed", &ctx, fx.get("text_embed"), cfg.text_len, cfg.dim, ctx_rows, 0.9999999, 1e-5);

    // Every fourth block plus the last: enough to bisect, at one extra
    // [tokens, dim] buffer each.
    let taps: Vec<usize> = (0..cfg.num_layers).filter(|l| l % 4 == 0 || *l == cfg.num_layers - 1).collect();
    let d = WanDitDev::build(&cfg, &w, f, h, wd, None, &taps);
    eprintln!("engine on {} ({:?})", d.gpu().kind(), wan::block::attn_mode(d.gpu()));
    d.set_context(fx.get("context"), ctx_rows);
    let out = d.forward(fx.get("latent"), t);
    for l in &taps {
        report(&format!("block.{l}"), &d.read_tap(*l).unwrap(), fx.get(&format!("block.{l}")), 0.99999, 1e-3);
    }
    report("out (vs the reference)", &out, fx.get("out"), 0.99999, 1e-3);
    if fx.has("out_diffusers") {
        // The same output against the INDEPENDENT diffusers implementation -
        // a different RoPE formulation and a different attention call, so it
        // gates the conventions rather than reproducing them.
        report("out (vs diffusers)", &out, fx.get("out_diffusers"), 0.99999, 1e-3);
    }
}

/// The same weights through the HOST-ORCHESTRATED reference: one block on the
/// device at a time, the token slab round-tripping through host memory between
/// them. The device-resident engine above keeps the residual on the device and
/// records the whole stack as one graph, so agreeing here is what says the two
/// compositions are the same model and not two models that happen to be close.
#[test]
fn wan_dit_1_3b_reference_forward_matches() {
    let Some(fx) = Fixture::open("golden/wan/dit/dit_1_3b.safetensors") else { return };
    let Some(w) = real_weights() else { return };
    let cfg = WanConfig::t2v_1_3b();
    let ls = fx.shape("latent").to_vec();
    let (f, h, wd) = (ls[1] as u32, ls[2] as u32, ls[3] as u32);
    let ctx_rows = fx.shape("context")[0];
    let taps = [0usize, cfg.num_layers - 1];
    let m = WanDit::new(cfg, w, None);
    let (out, block_taps) =
        m.forward_taps(fx.get("latent"), f, h, wd, fx.get("context"), ctx_rows, fx.get("timestep")[0], &taps);
    for (l, v) in &block_taps {
        report(&format!("host block.{l}"), v, fx.get(&format!("block.{l}")), 0.99999, 1e-3);
    }
    report("host out (vs the reference)", &out, fx.get("out"), 0.99999, 1e-3);
}
