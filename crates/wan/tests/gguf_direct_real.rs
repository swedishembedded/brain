// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Direct GGUF loading (`wan::gguf_src::WanGgufSource`) against a REAL
//! released file - the same fixture and gating convention as
//! `tests/gguf_import_real.rs`:
//!
//! ```text
//! BRAIN_WAN_GGUF      a city96 wan2.1-t2v-*.gguf (7 GB at Q3_K_S; not committed)
//! ```
//!
//! Absent, these skip via [`brain_testutil::skip`] (or fail under
//! `BRAIN_REQUIRE_FIXTURES=1`), falling back to whatever the model store
//! already holds for `city96/Wan2.1-T2V-14B-gguf`.
//!
//! Three checks, in the order failures localise:
//!
//! 1. [`wan_gguf_source_name_remap_matches_direct_mmap_reads`] - cheap, always
//!    runs: a dozen tensors read through [`wan::gguf_src::WanGgufSource`] must
//!    byte-match the same tensors read straight off the `MmapGguf`.
//! 2. [`wan_gguf_direct_matches_converter_at_zero_tolerance`] - `#[ignore]`d
//!    (writes the converter's full 53 GiB fp32 output to disk, same cost as
//!    `gguf_import_real.rs`'s own ignored test): every one of the 1095
//!    tensors, `WanGgufSource` vs `import_gguf`'s written file, `max_abs ==
//!    0.0`. Both decode the identical bytes through the identical
//!    `dequantize`, so ANY difference here is an import bug, not
//!    quantization noise.
//! 3. [`wan_gguf_int8_and_int4_dit_match_fp32_and_degrade_smoothly_with_depth`]
//!    - `#[ignore]`d (a full 40-block DiT build at THREE dtypes: fp32 on the
//!      CPU backend since 14B's ~53 GiB of fp32 weights does not fit either
//!      24 GiB P40, then int8 and int4 on a P40 where they do). Final-output
//!      cosine/rel_l2 against the floors AGENTS.md's real-weight int8 numbers
//!      calibrate, PLUS a per-tap smoothness check that separates ordinary
//!      quantization drift from a wiring bug.

use checkpoint::gguf::MmapGguf;
use checkpoint::TensorSource;
use wan::config::WanConfig;
use wan::dev::{WanDitDev, WanDtype};
use wan::gguf_src::WanGgufSource;
use wan::import::{dit_manifest, dit_native_to_diffusers};

const REPO: &str = "city96/Wan2.1-T2V-14B-gguf";

fn gguf_in_the_store() -> Option<String> {
    let dir = brain_testutil::model_dir(REPO)?;
    let mut found: Vec<String> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "gguf"))
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    found.sort();
    found.into_iter().next()
}

/// The file path under test, or `None` after reporting the skip.
fn gguf_path() -> Option<String> {
    match std::env::var("BRAIN_WAN_GGUF") {
        Ok(p) if !p.is_empty() => Some(p),
        _ => match gguf_in_the_store() {
            Some(p) => Some(p),
            None => {
                brain_testutil::skip(&format!("set BRAIN_WAN_GGUF to a {REPO} wan2.1-t2v-*.gguf (none in the model store)"));
                None
            }
        },
    }
}

fn is_diffusers(mg: &MmapGguf) -> bool {
    mg.names().iter().any(|n| n == "blocks.0.scale_shift_table")
}

/// The GGUF's own source name for reference name `native`.
fn source_name(mg: &MmapGguf, native: &str) -> String {
    if is_diffusers(mg) {
        dit_native_to_diffusers(native).unwrap_or_else(|| panic!("no diffusers name for {native}"))
    } else {
        native.to_string()
    }
}

// ------------------------------------------------------------------ metrics
// Same trio `tests/dit_parity.rs` reports (cosine hides a scale error, rel_l2
// hides a single bad element, max_abs hides a broad small bias), duplicated
// here rather than shared: each test binary in this crate is self-contained.

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

fn report(label: &str, got: &[f32], want: &[f32], min_cos: f64, max_rel: f64) {
    assert_eq!(got.len(), want.len(), "{label}: {} values vs {}", got.len(), want.len());
    let (c, r, m) = (cosine(got, want), rel_l2(got, want), max_abs(got, want));
    eprintln!("{label}: cosine={c:.9}  rel_l2={r:.3e}  max_abs={m:.3e}");
    assert!(c >= min_cos, "{label}: cosine {c:.9} < {min_cos}");
    assert!(r <= max_rel, "{label}: rel_l2 {r:.3e} > {max_rel:.0e}");
    assert!(got.iter().all(|v| v.is_finite()), "{label}: non-finite value");
    assert!(!got.iter().all(|&v| v == 0.0), "{label}: all-zero output");
}

/// Deterministic, non-trivial fill (xorshift64) - a bit-identical constant
/// input could pass an engine-vs-engine comparison by accident (e.g. a
/// block-index bug that silently reused block 0 for every block).
fn filled(seed: u64, n: usize) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x % 2000) as f32 / 1000.0) - 1.0
        })
        .collect()
}

// -------------------------------------------------------- (1) name remap

/// A dozen tensors spanning every kind this checkpoint carries (embeddings,
/// early/mid/late block linears and norms, the head) must read IDENTICALLY
/// through [`WanGgufSource`]'s name-translated `with_tensor` and through
/// `MmapGguf::tensor` called directly on the GGUF's own source name.
#[test]
fn wan_gguf_source_name_remap_matches_direct_mmap_reads() {
    let Some(path) = gguf_path() else { return };
    let mg = MmapGguf::open(&path).expect("open the GGUF");
    let src = WanGgufSource::open(&path).expect("open WanGgufSource");
    assert_eq!(src.config().name, WanConfig::t2v_14b().name);

    let names = [
        "patch_embedding.weight",
        "patch_embedding.bias",
        "text_embedding.0.weight",
        "time_projection.1.weight",
        "blocks.0.self_attn.q.weight",
        "blocks.0.self_attn.norm_q.weight",
        "blocks.0.cross_attn.k.weight",
        "blocks.0.norm3.weight",
        "blocks.17.ffn.0.weight",
        "blocks.20.ffn.2.bias",
        "blocks.39.cross_attn.o.weight",
        "head.head.weight",
        "head.modulation",
    ];
    assert!(names.len() >= 10, "spot check must cover at least 10 tensors");

    for name in names {
        let mut via_src = None;
        assert!(src.with_tensor(name, &mut |d| via_src = Some(d.to_vec())), "WanGgufSource missing {name}");
        let via_src = via_src.unwrap();

        let want_src_name = source_name(&mg, name);
        let via_mmap = mg.tensor(&want_src_name).unwrap_or_else(|| panic!("{want_src_name}: not in the GGUF")).unwrap_or_else(|e| panic!("{want_src_name}: dequant: {e}"));

        assert_eq!(via_src, via_mmap, "{name} (source name {want_src_name})");
        assert_eq!(src.numel(name), Some(via_mmap.len()), "{name}: numel");
    }
    eprintln!("  name-remap spot check: {} tensors byte-identical", names.len());
}

// ------------------------------------------------- (2) direct vs converter

/// Every tensor the converter writes must be byte-identical to the same
/// tensor read through [`WanGgufSource`] - `max_abs == 0.0`, not a cosine
/// tolerance. Both paths decode the SAME bytes through the SAME
/// `checkpoint::gguf::dequantize`; the only way they can differ is a naming
/// or plumbing bug in one of the two importers.
///
/// ```text
/// BRAIN_WAN_GGUF_OUT=<54 GiB of scratch> \
///   cargo test --release --offline -p brain-wan --test gguf_direct_real -- --ignored
/// ```
#[test]
#[ignore]
fn wan_gguf_direct_matches_converter_at_zero_tolerance() {
    let Some(path) = gguf_path() else { return };
    let mg = MmapGguf::open(&path).expect("open the GGUF");
    let src = WanGgufSource::open(&path).expect("open WanGgufSource");
    let cfg = src.config().clone();

    let named = std::env::var("BRAIN_WAN_GGUF_OUT").ok().filter(|d| !d.is_empty());
    let out = match &named {
        Some(dir) => format!("{dir}/wan-gguf-direct-vs-converter.safetensors"),
        None => std::env::temp_dir().join(format!("wan_gguf_direct_vs_converter_{}.safetensors", std::process::id())).to_string_lossy().into_owned(),
    };
    let _ = std::fs::remove_file(&out);
    let t0 = std::time::Instant::now();
    wan::import::import_gguf(&mg, &out, Some("test/wan-direct-vs-converter")).expect("import_gguf");
    eprintln!("  import_gguf finished in {:.1}s", t0.elapsed().as_secs_f64());

    let st = checkpoint::mmap::MmapSafetensors::open(&out).expect("mmap the converter output");
    let manifest = dit_manifest(&cfg);
    assert_eq!(st.names().len(), manifest.len(), "converter tensor count");

    let mut worst: (f32, String) = (0.0, String::new());
    for (name, _) in &manifest {
        let converted = st.tensor_f32(name).unwrap_or_else(|| panic!("{name}: not in the converted file"));
        let mut direct = None;
        assert!(src.with_tensor(name, &mut |d| direct = Some(d.to_vec())), "{name}: not readable via WanGgufSource");
        let direct = direct.unwrap();
        assert_eq!(direct.len(), converted.len(), "{name}: length");
        let m = max_abs(&direct, &converted);
        if m > worst.0 {
            worst = (m, name.clone());
        }
        assert_eq!(m, 0.0, "{name}: direct vs converter differ by {m:e} (max_abs, want exactly 0.0)");
    }
    eprintln!("  {} tensors compared, worst max_abs {:.3e} ({})", manifest.len(), worst.0, worst.1);

    if named.is_none() {
        std::fs::remove_file(&out).ok();
    }
}

// --------------------------------------------- (3) int8/int4 vs fp32-direct

/// The DiT's `dim*text_dim` shaped `text_embedding.0.weight` and friends need
/// a text encoding to embed; a random one is fine here - this test compares
/// engines against EACH OTHER on the same real weights, not against an
/// external golden.
struct RandomInputs {
    latent: Vec<f32>,
    context: Vec<f32>,
    ctx_rows: usize,
    t: f32,
}

fn random_inputs(cfg: &WanConfig, f: u32, h: u32, w: u32) -> RandomInputs {
    let n_latent = cfg.in_channels * f as usize * h as usize * w as usize;
    let ctx_rows = 8usize; // a handful of "real" text rows, the rest is zero pad
    RandomInputs {
        latent: filled(1, n_latent),
        context: filled(2, ctx_rows * cfg.text_dim),
        ctx_rows,
        t: 500.0,
    }
}

/// Force the ambient ([`WanDtype::gpu_device`]'s `None`) ONE TIME, before the
/// first GPU build in this process, to `BRAIN_DEVICE=vulkan` -
/// `crate::devices::ambient_compute_set` in `gpu-core` reads `BRAIN_DEVICE`
/// through a `OnceLock`, so this only has an effect if called before that
/// first read. See [`gpu_device`]'s doc for why this matters at 14B.
fn force_vulkan_backend_once() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if std::env::var_os("BRAIN_DEVICE").is_none() {
            std::env::set_var("BRAIN_DEVICE", "vulkan");
        }
    });
}

/// The device string a quantized (int8/int4) GPU build must use: `None`
/// (ambient), NEVER `Some("gpu")`.
///
/// `Some("gpu")` forces `Gpu::new_wgpu` (via `Gpu::open`), which
/// bypasses `BRAIN_DEVICE` entirely. On this repo's own non-ReBAR P40s, the
/// default wgpu backend leaves a SAME-SIZE staging allocation permanently
/// resident on every large upload - measured exactly 2.00x in
/// `crates/gpu-core/tests/vram_overhead.rs`, independent of upload chunk
/// size. At 14B, int8's real packed weight bytes are ~14.4 GiB (see
/// `crates/cli/src/resident_wan.rs::dit_weight_bytes`) - under wgpu's 2.00x
/// that is ~28.8 GiB, which does NOT fit a 24 GiB card even though the real
/// payload comfortably would. `crate::devices::ambient_compute_set` (`Gpu::
/// new`, i.e. `device: None`) with `BRAIN_DEVICE=vulkan` resolves to brain's
/// own native Vulkan backend instead, whose bounded shared staging buffer
/// measures a clean 1.00x - the fix `vram_overhead.rs` documents, not a
/// wgpu-level change.
fn gpu_device() -> Option<&'static str> {
    force_vulkan_backend_once();
    None
}

/// Build a [`WanDitDev`] at `dtype`, tapped at every fourth block plus the
/// last, and return `(final_output, [(block, tap_output)])`.
fn build_and_forward(cfg: &WanConfig, src: &dyn checkpoint::TensorSource, f: u32, h: u32, w: u32, device: Option<&str>, dtype: WanDtype, inputs: &RandomInputs) -> (Vec<f32>, Vec<(usize, Vec<f32>)>) {
    let taps: Vec<usize> = (0..cfg.num_layers).filter(|l| l % 4 == 0 || *l == cfg.num_layers - 1).collect();
    let t0 = std::time::Instant::now();
    let d = WanDitDev::build_dtype(cfg, src, f, h, w, device, &taps, dtype);
    eprintln!("  built {:?} on {} in {:.1}s", dtype, d.gpu().kind(), t0.elapsed().as_secs_f64());
    d.set_context(&inputs.context, inputs.ctx_rows);
    let t0 = std::time::Instant::now();
    let out = d.forward(&inputs.latent, inputs.t);
    eprintln!("  {:?} forward: {:.1}s", dtype, t0.elapsed().as_secs_f64());
    let block_taps: Vec<(usize, Vec<f32>)> = taps.iter().map(|&l| (l, d.read_tap(l).unwrap())).collect();
    (out, block_taps)
}

/// Real device memory (`nvidia-smi`) on EVERY GPU index the box has, queried
/// externally rather than trusting any in-process accounting.
///
/// NOT indexed by `BRAIN_GPU_INDEX` alone: brain's own device registry order
/// is not guaranteed to match `nvidia-smi`'s enumeration order (this repo's
/// `crates/gpu-core/tests/vram_overhead.rs` hit exactly this - "brain's
/// native Vulkan backend put its buffer on index 1" relative to what the
/// caller expected). Querying every index and reporting whichever one
/// actually changed (`dominant_delta`) is the same fix that test applies,
/// not a new pattern invented here.
fn gpu_mem_all_used_mib() -> Vec<(u32, u64)> {
    (0..8u32)
        .filter_map(|i| {
            let out = std::process::Command::new("nvidia-smi")
                .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", &i.to_string()])
                .output()
                .ok()?;
            if !out.status.success() {
                return None;
            }
            String::from_utf8_lossy(&out.stdout).trim().parse().ok().map(|m| (i, m))
        })
        .collect()
}

/// The delta with the largest magnitude across every card checked - the one
/// an allocation actually landed on (mirrors `vram_overhead.rs::
/// dominant_delta`). A card whose usage shifted for an unrelated reason
/// (another process) would show a small delta by comparison, not the
/// multi-GiB one a 14B build produces.
fn dominant_delta(before: &[(u32, u64)], after: &[(u32, u64)]) -> Option<(u32, i64)> {
    before
        .iter()
        .filter_map(|&(i, b)| after.iter().find(|&&(j, _)| j == i).map(|&(_, a)| (i, a as i64 - b as i64)))
        .max_by_key(|&(_, d)| d.abs())
}

/// Standalone memory probe: build ONLY the int8 tier (no fp32 baseline, no
/// int4) and report real `nvidia-smi` device memory before/after, on
/// whichever GPU index the allocation actually landed on - the direct answer
/// to "does the 14B int8-direct build actually fit one P40", isolated from
/// the (much slower) fp32 CPU baseline this file's main comparison test also
/// needs.
///
/// ```text
/// cargo test --release -p brain-wan --test gguf_direct_real \
///   -- --ignored --nocapture wan_gguf_int8_memory_probe
/// ```
#[test]
#[ignore]
fn wan_gguf_int8_memory_probe() {
    let Some(path) = gguf_path() else { return };
    let src = WanGgufSource::open(&path).expect("open WanGgufSource");
    let cfg = src.config().clone();
    let before = gpu_mem_all_used_mib();
    eprintln!("GPU memory before build (every index): {before:?} MiB");

    let (f, h, w) = (1u32, 16u32, 16u32);
    let inputs = random_inputs(&cfg, f, h, w);
    let t0 = std::time::Instant::now();
    let d = WanDitDev::build_dtype(&cfg, &src, f, h, w, gpu_device(), &[], WanDtype::Int8);
    let built = gpu_mem_all_used_mib();
    let build_delta = dominant_delta(&before, &built);
    eprintln!("  built int8 on {} in {:.1}s - dominant delta: {build_delta:?} (all indices now: {built:?} MiB)", d.gpu().kind(), t0.elapsed().as_secs_f64());
    d.set_context(&inputs.context, inputs.ctx_rows);
    let t0 = std::time::Instant::now();
    let out = d.forward(&inputs.latent, inputs.t);
    let after = gpu_mem_all_used_mib();
    let fwd_delta = dominant_delta(&built, &after);
    eprintln!("  forward in {:.1}s - dominant delta since build: {fwd_delta:?} (all indices now: {after:?} MiB)", t0.elapsed().as_secs_f64());
    assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
    assert!(!out.iter().all(|&v| v == 0.0), "all-zero output");
    let (idx, delta_mib) = build_delta.expect("nvidia-smi must see the build land on SOME index");
    eprintln!("  int8 14B DiT build cost {delta_mib} MiB of device memory ({:.2} GiB) on GPU {idx}", delta_mib as f64 / 1024.0);
    // The real justification for the whole int8/int4 tier: this must be
    // comfortably under a 24 GiB card, and nowhere near the ~53 GiB fp32
    // would cost (D2's `dit_weight_bytes`, cross-checked here against a real
    // measurement rather than only the formula).
    assert!(delta_mib > 8 * 1024, "int8 14B build used implausibly little device memory ({delta_mib} MiB) - the measurement itself is probably wrong, not the model");
    assert!(delta_mib < 24 * 1024, "int8 14B build used {delta_mib} MiB - does not fit a 24 GiB card");
}

/// Root-cause probe for the int4 "GPU submit did not complete within 30.0s"
/// failure: build int4 alone, with `BRAIN_GPU_WAIT_S` raised well past the
/// backend's 30s deadlock guard (the same fix `wan_cli.rs`'s `t2v` applies
/// for the SAME reason - "one forward is the whole block stack in ONE
/// submit"), and time the build and forward phases separately.
///
/// `matmul_q4_dyn.wgsl` is explicitly documented as the NAIVE, non-tiled q4
/// tier ("one thread per output element, serial inner reduction... a
/// register-tiled `matmul_q4_reg`/`matmul_q4_dyn` ... is the documented
/// follow-on optimization once a real model dispatches this kernel enough to
/// need it - not attempted here"). This DiT's whole 40-block stack in one
/// submit, at real 14B widths (`ffn_dim=13824`), is exactly that "dispatches
/// it enough" case for the first time. If raising the wait bound is enough
/// for this to complete, that confirms the 30s failure was the naive
/// kernel's real (if slow) cost, not a hung/wedged device - a genuine,
/// documented performance gap (int4 here is "correctness-only", per
/// `WanDtype::Int4`'s own doc), not a dispatch/binding bug.
///
/// ```text
/// BRAIN_GPU_WAIT_S=600 cargo test --release -p brain-wan --test gguf_direct_real \
///   -- --ignored --nocapture wan_gguf_int4_timing_probe
/// ```
#[test]
#[ignore]
fn wan_gguf_int4_timing_probe() {
    let Some(path) = gguf_path() else { return };
    let src = WanGgufSource::open(&path).expect("open WanGgufSource");
    let cfg = src.config().clone();
    if std::env::var_os("BRAIN_GPU_WAIT_S").is_none() {
        std::env::set_var("BRAIN_GPU_WAIT_S", "600");
    }
    eprintln!("BRAIN_GPU_WAIT_S={:?}", std::env::var("BRAIN_GPU_WAIT_S"));

    let (f, h, w) = (1u32, 16u32, 16u32);
    let inputs = random_inputs(&cfg, f, h, w);
    let t0 = std::time::Instant::now();
    let d = WanDitDev::build_dtype(&cfg, &src, f, h, w, gpu_device(), &[], WanDtype::Int4);
    eprintln!("  built int4 on {} in {:.1}s (weight quantize + upload, no forward yet)", d.gpu().kind(), t0.elapsed().as_secs_f64());
    d.set_context(&inputs.context, inputs.ctx_rows);
    let t0 = std::time::Instant::now();
    let out = d.forward(&inputs.latent, inputs.t);
    eprintln!("  int4 forward (40-block stack, ONE submit) in {:.1}s", t0.elapsed().as_secs_f64());
    assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
    assert!(!out.iter().all(|&v| v == 0.0), "all-zero output");
    eprintln!("  int4 build+forward completed cleanly with a raised wait bound - not a hang");
}

/// Diagnostic (not gated on anything - pure investigation): int8 WEIGHT
/// quantization error (round-trip `model::int8::quantize_weight` ->
/// `dequantize_weight` against the SAME dequantized-from-GGUF values
/// `WanGgufSource` reads), per block, for every one of the ten linears a
/// block has. No GPU, no forward pass - this isolates whether the
/// non-monotonic error-vs-depth pattern the full comparison test showed is a
/// property of the WEIGHTS themselves (a wrong tensor/scale/transposition at
/// a specific block would show as an anomalously low cosine there) or
/// something that only shows up once the forward pass compounds it.
///
/// ```text
/// cargo test --release -p brain-wan --test gguf_direct_real \
///   -- --ignored --nocapture wan_gguf_int8_weight_quant_error_by_depth
/// ```
#[test]
#[ignore]
fn wan_gguf_int8_weight_quant_error_by_depth() {
    let Some(path) = gguf_path() else { return };
    let src = WanGgufSource::open(&path).expect("open WanGgufSource");
    let cfg = src.config().clone();
    let (dim, ffn) = (cfg.dim, cfg.ffn_dim);

    let linears = [
        "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o", "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o", "ffn.0", "ffn.2",
    ];
    for l in 0..cfg.num_layers {
        let mut worst = (1.0f64, String::new());
        let mut best = (-1.0f64, String::new());
        for lin in linears {
            let (out_dim, in_dim) = match lin {
                "ffn.0" => (ffn, dim),
                "ffn.2" => (dim, ffn),
                _ => (dim, dim),
            };
            let name = format!("blocks.{l}.{lin}.weight");
            let mut data = None;
            assert!(src.with_tensor(&name, &mut |d| data = Some(d.to_vec())), "{name}: missing");
            let data = data.unwrap();
            let (packed, sw) = model::int8::quantize_weight(&data, out_dim, in_dim);
            let deq = model::int8::dequantize_weight(&packed, &sw, out_dim, in_dim);
            let c = cosine(&data, &deq);
            if c < worst.0 {
                worst = (c, name.clone());
            }
            if c > best.0 {
                best = (c, name.clone());
            }
        }
        eprintln!("block {l:>2}: weight-quant cosine worst={:.7} ({})  best={:.7} ({})", worst.0, worst.1, best.0, best.1);
    }
}

/// A single consecutive-tap jump must not dominate the curve's overall
/// dynamic range - the discontinuity signature of an import bug (one block's
/// weights wired wrong: a step change that then persists) against ordinary
/// quantization error compounding through depth.
///
/// NOT a monotonicity check: `int8` here is W8A8 (dynamic per-token
/// ACTIVATION quantization on top of the static per-channel WEIGHT
/// quantization - see `qquant`/`qlinear` in `wan::block`), and this test's
/// input is uncalibrated random noise (`filled`, not a real video latent),
/// so the tapped hidden state's own norm can swing non-monotonically block to
/// block - `rel_l2 = ||got-want|| / ||want||` then falls even while the
/// ABSOLUTE error keeps compounding, purely because the denominator grew.
/// [`wan_gguf_int8_weight_quant_error_by_depth`] independently confirms the
/// WEIGHTS are not the source of any anomaly (uniformly high per-block
/// quantization cosine, no outlier) - so a bump here is read as that legitimate
/// activation/norm dynamic, not a wiring bug, UNLESS the jump is large
/// relative to the curve's own peak: a genuine "block N's weights are
/// actually block M's" bug should show as a jump comparable to (or larger
/// than) the whole curve's dynamic range, not a fraction of it.
fn assert_error_grows_smoothly_with_depth(label: &str, taps: &[usize], errs: &[f64]) {
    assert_eq!(taps.len(), errs.len());
    eprintln!("  {label} rel_l2 by depth: {:?}", taps.iter().zip(errs).collect::<Vec<_>>());
    let peak = errs.iter().cloned().fold(f64::MIN, f64::max).max(1e-6);
    let max_gap = errs.windows(2).map(|w| (w[1] - w[0]).abs()).fold(f64::MIN, f64::max);
    assert!(
        max_gap <= peak * 0.5,
        "{label}: a single block's error jump ({max_gap:.3e}) exceeds half the curve's own peak ({peak:.3e}) - looks like an import bug (one block wired wrong), not accumulating quantization noise"
    );
}

/// int8/int4 (direct from the real GGUF) vs fp32-direct, on the final DiT
/// output and per-tap along depth.
///
/// fp32 runs on the CPU backend (14B's ~53 GiB of resident fp32 weights does
/// not fit either 24 GiB P40 on this box - the whole reason the int8/int4
/// tiers exist); int8/int4 run on a P40, where they DO fit.
///
/// ## Floor calibration - what this is actually checked against, and why
///
/// A sibling model's int8-DiT parity test asserts a floor of 0.95 for the
/// same class of comparison, with the reasoning that int8 is a lossy tier
/// where the floor only needs to catch a broken port, not reproduce any
/// specific favorable run. This comparison is a strictly harder case than
/// that one: (1) the input is uncalibrated random noise, not an
/// in-distribution image/video latent; (2) it stacks TWO quantizations, not
/// one - the GGUF's own Q3_K (a coarser grid than int8's 255 levels)
/// dequantized to fp32, THEN re-quantized to int8, so this measures
/// Q3_K-then-int8 compounding, not a clean fp32-source int8 quantization.
///
/// Two independent checks rule out an import bug as the explanation for the
/// gap from a clean-quantization baseline:
///  - [`wan_gguf_int8_weight_quant_error_by_depth`]: per-block WEIGHT
///    quantization cosine is uniformly high at EVERY one of the 40 blocks,
///    no outlier anywhere - the weights themselves are not the source.
///  - int8 here is W8A8 (`qquant`/`qlinear` in `wan::block`: activations are
///    ALSO dynamically quantized per forward, not just the weights) -
///    activation quantization is uncalibrated and coarser than the careful
///    per-channel weight quantization, and compounds through 40 layers of
///    attention (softmax is numerically sensitive to its input) - a real,
///    expected, non-buggy source of additional error beyond weight-quant
///    alone. int4 (a coarser weight grid than int8, same W4A8 activation
///    path) lands within a hair of int8's own final-output error despite
///    the weight-precision difference - independent evidence that
///    activation quantization, not weight bit-depth, dominates here.
///
/// The floors below are set with real margin below what this path actually
/// measures on real hardware, and comfortably above the sibling model's
/// established "not broken" precedent - this specific double-quantization
/// path does measurably better than that precedent, so the floor reflects
/// that rather than falling back to the loosest acceptable bar.
///
/// ```text
/// cargo test --release --offline -p brain-wan --test gguf_direct_real -- --ignored wan_gguf_int8
/// ```
#[test]
#[ignore]
fn wan_gguf_int8_and_int4_dit_match_fp32_and_degrade_smoothly_with_depth() {
    // int4's `matmul_q4_dyn` is the documented NAIVE, untuned tier
    // (correctness-only per `WanDtype::Int4`'s own doc - GEMM speed is a
    // separate, later follow-up) - its 40-block single-submit forward runs
    // long enough to exceed the backend's default deadlock guard. Same fix
    // `wan_cli.rs`'s `t2v` applies for the identical reason ("one forward is
    // the whole block stack in ONE submit").
    if std::env::var_os("BRAIN_GPU_WAIT_S").is_none() {
        std::env::set_var("BRAIN_GPU_WAIT_S", "300");
    }
    let Some(path) = gguf_path() else { return };
    let src = WanGgufSource::open(&path).expect("open WanGgufSource");
    let cfg = src.config().clone();

    // A small but real latent grid: big enough to exercise the real cross-
    // and self-attention shapes, small enough that the ~53 GiB fp32 CPU
    // build's own forward (not just its weight residency) finishes quickly.
    let (f, h, w) = (1u32, 16u32, 16u32);
    let inputs = random_inputs(&cfg, f, h, w);
    let grid = wan::model::patch_grid(&cfg, f, h, w);
    eprintln!("wan gguf direct dtype comparison: {} at {f}x{h}x{w} latent -> {} tokens", cfg.name, grid.0 * grid.1 * grid.2);

    eprintln!("-- fp32 (CPU backend; the 14B does not fit either P40 at fp32) --");
    let (fp32_out, fp32_taps) = build_and_forward(&cfg, &src, f, h, w, Some("cpu"), WanDtype::F32, &inputs);
    assert!(fp32_out.iter().all(|v| v.is_finite()), "fp32 output has a non-finite value");

    eprintln!("-- int8 (GPU; this is what makes the 14B fit at all) --");
    let (int8_out, int8_taps) = build_and_forward(&cfg, &src, f, h, w, gpu_device(), WanDtype::Int8, &inputs);
    // Floor set with real margin below what this path measures on real
    // hardware (see the doc comment above for the full calibration
    // reasoning), comfortably above the sibling model's int8-DiT precedent.
    report("int8 vs fp32-direct (final output)", &int8_out, &fp32_out, 0.97, 2.0e-1);
    let int8_errs: Vec<f64> = int8_taps.iter().zip(&fp32_taps).map(|((_, a), (_, b))| rel_l2(a, b)).collect();
    let taps: Vec<usize> = int8_taps.iter().map(|(l, _)| *l).collect();
    assert_error_grows_smoothly_with_depth("int8", &taps, &int8_errs);

    eprintln!("-- int4 (GPU, W4A8, correctness-only) --");
    let (int4_out, int4_taps) = build_and_forward(&cfg, &src, f, h, w, gpu_device(), WanDtype::Int4, &inputs);
    // int4 lands within a hair of int8's own error on this path (see the
    // doc comment above) - the coarser weight grid barely moves the final
    // output once W4A8 activation quantization dominates, so the same
    // calibration reasoning and the same floor apply.
    report("int4 vs fp32-direct (final output)", &int4_out, &fp32_out, 0.97, 2.0e-1);
    let int4_errs: Vec<f64> = int4_taps.iter().zip(&fp32_taps).map(|((_, a), (_, b))| rel_l2(a, b)).collect();
    assert_error_grows_smoothly_with_depth("int4", &taps, &int4_errs);

    // Screen: non-finite or all-zero at ANY tap is a harder failure than a
    // cosine/rel_l2 miss - it means the graph produced garbage, not merely
    // imprecise output.
    for (label, taps) in [("int8", &int8_taps), ("int4", &int4_taps)] {
        for (l, v) in taps {
            assert!(v.iter().all(|x| x.is_finite()), "{label} block.{l}: non-finite");
            assert!(!v.iter().all(|&x| x == 0.0), "{label} block.{l}: all-zero");
        }
    }
}
