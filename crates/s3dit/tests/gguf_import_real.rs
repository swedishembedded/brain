// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Z-Image (S3-DiT) GGUF importer against a REAL released file.
//!
//! Everything else that covers this path is synthetic - a two-element `dim`,
//! a tensor map the test itself built - which proves the rename rules are
//! self-consistent and nothing about the files that exist. That is exactly how
//! the same three-copies-of-the-checkpoint bug survived in `crates/wan` until a
//! real file was put in front of it.
//!
//! ```text
//! BRAIN_S3DIT_GGUF      an unsloth z-image*-*.gguf (3.4 GB at Q2_K; not committed)
//! BRAIN_S3DIT_GGUF_OUT  where the `#[ignore]`d conversion writes its ~25 GB of
//!                       fp32 output; defaults under the system temp dir
//! ```
//!
//! `BRAIN_S3DIT_GGUF` is a fixture, so absent it skips via
//! [`brain_testutil::skip`] and `BRAIN_REQUIRE_FIXTURES=1` turns that into a
//! failure. It falls back to whatever `*.gguf` the model store already holds
//! for `unsloth/Z-Image-Turbo-GGUF`, so a box that ran `brain fetch` needs
//! nothing exported.

use std::collections::{BTreeMap, HashMap, HashSet};

use checkpoint::gguf::MmapGguf;
use s3dit::import::{dit_config_from_shapes, dit_manifest, GGUF_ARCHITECTURE};
use s3dit::model::ZImageConfig;

/// The upstream repo the fallback scans. unsloth publishes both `Z-Image-GGUF`
/// and `Z-Image-Turbo-GGUF`; they are the same architecture and either one
/// exercises this path, so only one needs to be on the box.
const REPO: &str = "unsloth/Z-Image-Turbo-GGUF";

/// The first `*.gguf` in the model store's copy of [`REPO`], if there is one -
/// the repo ships one file per quantization, and any of them exercises this
/// path (they differ in which ggml types the projections carry).
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

/// The file under test, or `None` after reporting the skip.
fn open_gguf() -> Option<MmapGguf> {
    let path = match std::env::var("BRAIN_S3DIT_GGUF") {
        Ok(p) if !p.is_empty() => p,
        _ => match gguf_in_the_store() {
            Some(p) => p,
            None => {
                brain_testutil::skip(&format!("set BRAIN_S3DIT_GGUF to a {REPO} z-image-*.gguf (none in the model store)"));
                return None;
            }
        },
    };
    if !std::path::Path::new(&path).exists() {
        brain_testutil::skip(&format!("BRAIN_S3DIT_GGUF={path} not found"));
        return None;
    }
    println!("  gguf {path}");
    Some(MmapGguf::open(&path).expect("open the GGUF"))
}

/// Every tensor's name in brain's spelling, plus the shape it will be written
/// with - the same resolution `import_gguf` does, repeated here so a mismatch
/// is reported as a set difference instead of the importer's first-missing-name
/// error.
///
/// Only the fused `qkv.weight` becomes more than one output tensor, so this is
/// the rename rules plus that one split.
fn brain_names(mg: &MmapGguf, cfg: &ZImageConfig) -> HashMap<String, Vec<usize>> {
    let dim = cfg.dim as usize;
    let xk = format!("all_x_embedder.{}-{}.", cfg.patch_size, cfg.f_patch_size);
    let fk = format!("all_final_layer.{}-{}.", cfg.patch_size, cfg.f_patch_size);
    let mut out: HashMap<String, Vec<usize>> = HashMap::new();
    let mut put = |k: String, v: Vec<usize>| assert!(out.insert(k.clone(), v).is_none(), "two source tensors map to {k}");
    for n in mg.names() {
        let shape = mg.shape(n).expect("shape").to_vec();
        if let Some(base) = n.strip_suffix("qkv.weight") {
            for leaf in ["to_q", "to_k", "to_v"] {
                put(format!("{base}{leaf}.weight"), vec![dim, dim]);
            }
            continue;
        }
        let mut k = n.replace(".attention.out.", ".attention.to_out.0.");
        k = k.replace(".attention.k_norm.weight", ".attention.norm_k.weight");
        k = k.replace(".attention.q_norm.weight", ".attention.norm_q.weight");
        if let Some(rest) = k.strip_prefix("x_embedder.") {
            k = format!("{xk}{rest}");
        } else if let Some(rest) = k.strip_prefix("final_layer.") {
            k = format!("{fk}{rest}");
        }
        put(k, shape);
    }
    out
}

/// The header alone answers "is this the file the importer thinks it is":
/// architecture tag, the Z-Image-only discriminator that tells this apart from
/// a genuine Lumina2 GGUF sharing the tag, the config read off the shapes, and
/// the manifest covered in both directions - all off the tensor infos, before a
/// single block is decoded.
#[test]
fn z_image_gguf_names_the_dit_and_covers_the_manifest() {
    let Some(mg) = open_gguf() else { return };

    let arch = mg.kv().get("general.architecture").and_then(|v| v.as_str());
    assert_eq!(arch, Some(GGUF_ARCHITECTURE), "general.architecture");
    // The whole reason this importer carries its own guard: the tag above is
    // NOT Z-Image's, it is Lumina2's, and only this tensor separates them.
    assert!(mg.shape("cap_embedder.0.weight").is_some(), "the Z-Image discriminator tensor");

    let shapes: Vec<(String, Vec<usize>)> =
        mg.names().iter().map(|n| (n.clone(), mg.shape(n).expect("shape").to_vec())).collect();
    let cfg = dit_config_from_shapes(&shapes).expect("derive the config from the tensor shapes");
    println!("  {} source tensors, quant label {:?}, {} parameters", shapes.len(), mg.model_card().quant, mg.param_count());
    println!("  derived dim {} heads {} layers {} refiners {} cap_feat {}", cfg.dim, cfg.n_heads, cfg.n_layers, cfg.n_refiner_layers, cfg.cap_feat_dim);

    // Every field the weights can answer, checked against the shipped variant
    // rather than assumed to equal it.
    let want = ZImageConfig::turbo();
    assert_eq!((cfg.dim, cfg.n_heads, cfg.n_layers, cfg.n_refiner_layers, cfg.cap_feat_dim), (want.dim, want.n_heads, want.n_layers, want.n_refiner_layers, want.cap_feat_dim));
    assert_eq!((cfg.dim, cfg.n_layers, cfg.n_refiner_layers), (3840, 30, 2));
    // Derived, not read from a metadata field: `q_norm` is one head wide, so
    // the head count is a consequence of two tensor shapes.
    assert_eq!(mg.shape("layers.0.attention.q_norm.weight").expect("q_norm"), [(cfg.dim / cfg.n_heads) as usize]);
    // The fused qkv is the one tensor whose element count only makes sense
    // after the split; a repacker that pre-split it would land here.
    assert_eq!(mg.shape("layers.0.attention.qkv.weight").expect("qkv"), [3 * cfg.dim as usize, cfg.dim as usize]);

    let got = brain_names(&mg, &cfg);
    let want: BTreeMap<String, Vec<usize>> = dit_manifest(&cfg).into_iter().collect();
    let got_keys: HashSet<&str> = got.keys().map(String::as_str).collect();
    let want_keys: HashSet<&str> = want.keys().map(String::as_str).collect();
    let mut missing: Vec<&&str> = want_keys.difference(&got_keys).collect();
    let mut extra: Vec<&&str> = got_keys.difference(&want_keys).collect();
    missing.sort_unstable();
    extra.sort_unstable();
    assert!(missing.is_empty() && extra.is_empty(), "missing {missing:?}, unused {extra:?}");
    // 453 source tensors, 34 of them fused qkv that each become three.
    assert_eq!(mg.names().len(), 453);
    assert_eq!(got.len(), 521, "30 modulated blocks of 15 + 2 modulated + 2 unmodulated refiners + 15 wrapper");

    for (name, want_shape) in &want {
        let n: usize = want_shape.iter().product();
        let s = &got[name];
        assert_eq!(s.iter().product::<usize>(), n, "{name}: shape {s:?}, manifest {want_shape:?}");
    }
}

/// The manifest against the RELEASED diffusers transformer - a second real
/// file, and the one brain's own safetensors path already loads.
///
/// Worth having separately from the GGUF check above: that one certifies that
/// brain and unsloth agree, and two converters can agree on the same mistake.
/// This one certifies the manifest against Tongyi's own upload, whose 521
/// tensors are already in brain's spelling (the fused `qkv` is a Comfy/original
/// packing, not a diffusers one) - so it compares names with no rename step in
/// between, and reads `config.json` to pin the numbers the shapes are built
/// from.
///
/// Header-only: the shards are ~25 GB and nothing here needs a value.
#[test]
fn the_manifest_is_the_released_diffusers_transformers_own_tensor_set() {
    const REPO: &str = "Tongyi-MAI/Z-Image-Turbo";
    let Some(dir) = brain_testutil::model_dir(REPO) else {
        brain_testutil::skip(&format!("no model store to look for {REPO} in"));
        return;
    };
    let dir = format!("{dir}/transformer");
    if !std::path::Path::new(&format!("{dir}/config.json")).exists() {
        brain_testutil::skip(&format!("{REPO}'s transformer/ is not in the model store"));
        return;
    }

    // The config the manifest's shapes are computed from, against the release's
    // own declaration of it - so a drift in `ZImageConfig::turbo` is caught
    // here rather than as 521 wrong shapes.
    let cfg = ZImageConfig::turbo();
    let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/config.json")).expect("config.json")).expect("parse");
    for (key, want) in [
        ("dim", u64::from(cfg.dim)),
        ("n_heads", u64::from(cfg.n_heads)),
        ("n_layers", u64::from(cfg.n_layers)),
        ("n_refiner_layers", u64::from(cfg.n_refiner_layers)),
        ("cap_feat_dim", u64::from(cfg.cap_feat_dim)),
        ("in_channels", u64::from(cfg.in_channels)),
    ] {
        assert_eq!(json[key].as_u64(), Some(want), "transformer/config.json {key}");
    }
    // `patch_size`/`f_patch_size` are the ones the tensor NAMES are built from,
    // and the release spells them as one-element lists of supported sizes.
    assert_eq!(json["all_patch_size"], serde_json::json!([cfg.patch_size]));
    assert_eq!(json["all_f_patch_size"], serde_json::json!([cfg.f_patch_size]));

    let mut got: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut shards: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .expect("read transformer/")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    shards.sort();
    assert!(!shards.is_empty(), "no safetensors shards in {dir}");
    for s in &shards {
        let st = checkpoint::mmap::MmapSafetensors::open(s).expect("open a shard");
        for n in st.names() {
            got.insert(n.clone(), st.shape(n).expect("shape").to_vec());
        }
    }

    let want: BTreeMap<String, Vec<usize>> = dit_manifest(&cfg).into_iter().collect();
    let got_keys: HashSet<&str> = got.keys().map(String::as_str).collect();
    let want_keys: HashSet<&str> = want.keys().map(String::as_str).collect();
    let mut missing: Vec<&&str> = want_keys.difference(&got_keys).collect();
    let mut extra: Vec<&&str> = got_keys.difference(&want_keys).collect();
    missing.sort_unstable();
    extra.sort_unstable();
    assert!(missing.is_empty() && extra.is_empty(), "missing {missing:?}, unused {extra:?}");
    for (name, want_shape) in &want {
        assert_eq!(&got[name], want_shape, "{name}");
    }
    println!("  {} shards, {} tensors, all shapes exact", shards.len(), got.len());
}

/// The quant types the file ACTUALLY uses, decoded on the file's own data.
///
/// unsloth's "Q2_K" is not one type: it leaves every 1-D tensor (the biases,
/// the RMSNorm gains) at F32, keeps the refiner blocks and the wrapper linears
/// at BF16, and mixes Q2_K with Q4_K and Q5_K across the 30 main blocks.
/// `dequantize` returning a clean error for the IQ/TQ codebooks it does not
/// implement is only reassuring if the released file does not use them - which
/// is what the histogram checks.
#[test]
fn z_image_gguf_dequantizes_every_tensor_to_finite_values() {
    let Some(mg) = open_gguf() else { return };

    let mut dtypes: BTreeMap<&str, usize> = BTreeMap::new();
    for n in mg.names() {
        *dtypes.entry(mg.dtype(n).unwrap_or_else(|| panic!("{n}: no name for its ggml type"))).or_default() += 1;
    }
    println!("  quant types: {dtypes:?}");
    assert!(
        dtypes.keys().any(|t| t.starts_with('Q')),
        "a released quantization must actually carry quantized blocks, not F32/BF16 throughout: {dtypes:?}"
    );
    assert!(dtypes.contains_key("F32"), "the 1-D tensors are left at F32: {dtypes:?}");

    // Streaming, one tensor at a time: 6.15 G parameters is ~24 GB as fp32, so
    // a loop that kept them would need the whole model resident to check that
    // no single one is broken.
    let mut worst: (f32, String) = (f32::INFINITY, String::new());
    for name in mg.names() {
        let numel: usize = mg.shape(name).expect("shape").iter().product();
        let data = mg.tensor(name).expect("indexed").unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(data.len(), numel, "{name}: dequant produced {} values for a {numel}-element tensor", data.len());
        let bad = data.iter().position(|v| !v.is_finite());
        assert!(bad.is_none(), "{name}: non-finite at element {:?}", bad.unwrap());
        let mean_abs = data.iter().map(|v| v.abs() as f64).sum::<f64>() / numel as f64;
        if (mean_abs as f32) < worst.0 {
            worst = (mean_abs as f32, name.clone());
        }
    }
    // An all-zero tensor is what a dequantizer that mis-reads a scale field
    // produces, and it imports and validates perfectly.
    assert!(worst.0 > 0.0, "{} dequantized to all zeros", worst.1);
    println!("  smallest mean|w| {:.3e} at {} ({} tensors)", worst.0, worst.1, mg.names().len());
    brain_testutil::mem("after streaming every dequant");
}

/// A Q2_K tensor's first two super-blocks, against a hand-derivation of ggml's
/// `dequantize_row_q2_K` over this file's own bytes.
///
/// Every other Q2_K check in the workspace feeds the decoder a block the test
/// itself constructed, which pins the arithmetic and nothing about where the
/// bytes are: a Q2_K super-block is `scales[16] | qs[64] | d | dmin`, the
/// 16-element groups take their scale and their min from the two nibbles of one
/// `scales` byte, the quants are read two bits at a time across four shifts,
/// and the tensor's base is `align_up(header_end, general.alignment)` plus its
/// own declared offset. A decoder that got any of those wrong still returns
/// finite, plausibly-scaled weights. These values were produced from the GGML
/// block definition by a separate numpy implementation reading the same file,
/// and they are bit-exact rather than approximate because dequantization is a
/// multiply of an f16 scale by a small integer.
#[test]
fn z_image_gguf_q2_k_dequant_matches_a_hand_derivation_of_the_ggml_layout() {
    let Some(mg) = open_gguf() else { return };
    // The values below are this ONE quantization's bytes; every other file in
    // the repo re-quantizes the same weights differently.
    let quant = mg.model_card().quant;
    if quant.as_deref() != Some("Q2_K") {
        brain_testutil::skip_unavailable(&format!("the store's copy is {quant:?}, and these values are Q2_K's"));
        return;
    }
    const NAME: &str = "layers.1.attention.out.weight";
    assert_eq!(mg.dtype(NAME), Some("Q2_K"), "{NAME} carries the file's namesake type");
    // Elements 0..32 straddle the first super-block's first two 16-element
    // groups, which have different unpacked scales (0.0701 and 0.1002).
    #[rustfmt::skip]
    const HEAD32: [f32; 32] = [
        0.15480042, 0.014450073, 0.014450073, 0.014450073, 0.15480042, -0.055725098, -0.055725098, -0.055725098,
        -0.055725098, 0.014450073, 0.084625244, 0.014450073, -0.055725098, 0.084625244, -0.055725098, 0.014450073,
        -0.008468628, 0.091781616, 0.091781616, -0.008468628, 0.091781616, -0.10871887, -0.20896912, -0.008468628,
        0.091781616, -0.008468628, -0.008468628, 0.091781616, -0.10871887, -0.008468628, -0.008468628, -0.20896912,
    ];
    // The second super-block, which has its own `d`/`dmin` pair.
    #[rustfmt::skip]
    const BLOCK1: [f32; 8] = [
        -0.03517151, -0.03517151, 0.077438354, 0.19004822, 0.077438354, 0.077438354, -0.03517151, -0.03517151,
    ];
    let w = mg.tensor(NAME).expect("in the GGUF").expect("dequant");
    assert_eq!(&w[..32], &HEAD32, "first super-block");
    assert_eq!(&w[256..264], &BLOCK1, "second super-block");
}

/// The q|k boundary inside the fused `qkv.weight`, from the same independent
/// derivation - the one place a Z-Image import can be wrong in a way no shape
/// check can see.
///
/// `import_gguf` splits `[3*dim, dim]` into three `dim`-row blocks. Off by one
/// row-block and every attention projection in the model is another's, with
/// every shape still correct, every value still finite and the manifest still
/// covered. So the values pinned here are the LAST eight of `to_q` and the
/// FIRST eight of `to_k`, taken from the two Q4_K super-blocks that meet at
/// element `dim*dim` - the split lands on a super-block boundary exactly
/// (`3840*3840 / 256 = 57600`), so each side is a whole block's arithmetic.
///
/// Q4_K's scales are the six-bit kind, packed across 12 bytes by
/// `get_scale_min_k4`, and this tensor is Q4_K while the one the test above
/// pins is Q2_K - so between them the two cover both k-quant scale layouts the
/// released file uses on real bytes.
#[test]
fn z_image_gguf_splits_the_fused_qkv_at_the_right_row_block() {
    let Some(mg) = open_gguf() else { return };
    let quant = mg.model_card().quant;
    if quant.as_deref() != Some("Q2_K") {
        brain_testutil::skip_unavailable(&format!("the store's copy is {quant:?}, and these values are Q2_K's"));
        return;
    }
    const NAME: &str = "layers.0.attention.qkv.weight";
    assert_eq!(mg.dtype(NAME), Some("Q4_K"), "{NAME} is Q4_K in this file, not the file's namesake type");
    #[rustfmt::skip]
    const Q_TAIL8: [f32; 8] = [
        -0.07765961, 0.16706085, 0.0039138794, -0.11844635, 0.12627411, -0.036872864, -0.036872864, -0.32238007,
    ];
    #[rustfmt::skip]
    const K_HEAD8: [f32; 8] = [
        -0.005997658, -0.31621552, 0.028470993, -0.07493496, 0.097408295, -0.04046631, -0.04046631, -0.07493496,
    ];
    let dim = ZImageConfig::turbo().dim as usize;
    let dd = dim * dim;
    let w = mg.tensor(NAME).expect("in the GGUF").expect("dequant");
    assert_eq!(w.len(), 3 * dd, "the fused qkv is three dim-row blocks");
    assert_eq!(&w[dd - 8..dd], &Q_TAIL8, "the tail of to_q");
    assert_eq!(&w[dd..dd + 8], &K_HEAD8, "the head of to_k");
}

/// End to end: GGUF in, brain-native fp32 safetensors out, read back.
///
/// The conversion is the only part of the path the tests above do not reach,
/// and it is the part that used to be impossible on this file: 6.15 G
/// parameters is 24.6 GB as fp32 and the eager importer needed roughly three
/// copies of it at once. Streaming it costs one tensor of RAM but still WRITES
/// the whole 24.6 GB, so this is disk-bound and takes real wall-clock minutes -
/// hence `#[ignore]`, the same call `crates/wan`'s conversion test makes for
/// the same reason.
///
/// Measured on the Q2_K file, this test alone in its process (`--test-threads=1`
/// matters - the streaming-dequant test above reaches a higher peak of its own
/// and would be credited here): **peak RSS 3.55 GiB, 175 s**, of which 3.4 GiB
/// is the GGUF's own file-backed mapping walked end to end. Run it with:
///
/// ```text
/// BRAIN_S3DIT_GGUF_OUT=<26 GB of scratch> \
///   cargo test --release --offline -p brain-s3dit --test gguf_import_real -- --ignored
/// ```
#[test]
#[ignore]
fn z_image_gguf_converts_to_a_brain_safetensors_checkpoint() {
    let Some(mg) = open_gguf() else { return };
    let named = std::env::var("BRAIN_S3DIT_GGUF_OUT").ok().filter(|d| !d.is_empty());
    let out = match &named {
        Some(dir) => format!("{dir}/s3dit-gguf-import-test.safetensors"),
        None => std::env::temp_dir()
            .join(format!("s3dit_gguf_import_{}.safetensors", std::process::id()))
            .to_string_lossy()
            .into_owned(),
    };
    let _ = std::fs::remove_file(&out);
    brain_testutil::mem("before import_gguf");
    let t0 = std::time::Instant::now();
    s3dit::import::import_gguf(&mg, &out, Some("unsloth/z-image-turbo-gguf")).expect("import_gguf");
    println!("  import_gguf finished in {:.1}s", t0.elapsed().as_secs_f64());
    brain_testutil::mem("after import_gguf");

    let cfg = ZImageConfig::turbo();
    let manifest = dit_manifest(&cfg);
    let params: u64 = manifest.iter().map(|(_, s)| s.iter().product::<usize>() as u64).sum();
    let bytes = std::fs::metadata(&out).expect("stat the output").len();
    assert!(bytes > 4 * params, "output is {bytes} bytes, short of {params} fp32 values");

    let header = checkpoint::st::read_metadata(&out).expect("metadata");
    assert_eq!(header.get("family").map(String::as_str), Some("s3dit"));
    assert_eq!(header.get("id").map(String::as_str), Some("unsloth/z-image-turbo-gguf"));
    assert_eq!(header.get("param_count").map(String::as_str), Some(params.to_string().as_str()));

    // Read the written file back the way a loader does, one tensor at a time.
    let st = checkpoint::mmap::MmapSafetensors::open(&out).expect("mmap the output");
    assert_eq!(st.names().len(), manifest.len(), "tensor count");
    let dim = cfg.dim as usize;
    let dd = dim * dim;

    // The split, checked on the written file against the independently derived
    // values the test above pins on the source - so the row-block boundary is
    // certified end to end and not just inside the mmap.
    #[rustfmt::skip]
    const K_HEAD8: [f32; 8] = [
        -0.005997658, -0.31621552, 0.028470993, -0.07493496, 0.097408295, -0.04046631, -0.04046631, -0.07493496,
    ];
    if mg.model_card().quant.as_deref() == Some("Q2_K") {
        let k = st.tensor_f32("layers.0.attention.to_k.weight").expect("to_k readable");
        assert_eq!(&k[..8], &K_HEAD8, "to_k's first row-block came from qkv[dim*dim..]");
    }

    // Every renamed shape, plus the fused-qkv thirds, against a fresh dequant
    // of the source - which is also the check that each name landed in the
    // right slot.
    let spot: [(&str, &str, Option<usize>); 7] = [
        ("all_x_embedder.2-1.weight", "x_embedder.weight", None),
        ("all_final_layer.2-1.linear.weight", "final_layer.linear.weight", None),
        ("cap_embedder.0.weight", "cap_embedder.0.weight", None),
        ("layers.29.attention.to_out.0.weight", "layers.29.attention.out.weight", None),
        ("layers.17.attention.norm_q.weight", "layers.17.attention.q_norm.weight", None),
        ("context_refiner.1.attention.to_v.weight", "context_refiner.1.attention.qkv.weight", Some(2 * dd)),
        ("noise_refiner.0.adaLN_modulation.0.weight", "noise_refiner.0.adaLN_modulation.0.weight", None),
    ];
    for (name, from, start) in spot {
        let want_shape = &manifest.iter().find(|(n, _)| n == name).expect("in the manifest").1;
        assert_eq!(st.shape(name).expect("shape"), want_shape.as_slice(), "{name}: declared shape");
        let got = st.tensor_f32(name).unwrap_or_else(|| panic!("{name}: readable"));
        let src = mg.tensor(from).expect("in the GGUF").expect("dequant");
        let want: &[f32] = match start {
            Some(s) => &src[s..s + dd],
            None => &src[..],
        };
        assert_eq!(got, want, "{name}: round trip through the writer");
    }
    println!("  wrote {} tensors / {params} parameters ({bytes} bytes) to {out}", manifest.len());
    if named.is_none() {
        std::fs::remove_file(&out).expect("remove the converted checkpoint");
    }
}
