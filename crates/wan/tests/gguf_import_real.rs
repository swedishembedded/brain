// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Wan GGUF importer against a REAL released file.
//!
//! Everything else that covers this path is synthetic - a manifest turned back
//! into shapes, or a header written by the test itself - which proves the
//! derivation is self-consistent and nothing about the files that exist. The
//! only Wan GGUF anyone publishes is `city96/Wan2.1-*-gguf`, and it is the
//! **14B**, not the 1.3B the rest of this crate's parity suite uses, so the
//! shapes here are the 14B's (dim 5120, 40 blocks, 1095 tensors) and NOT the
//! 825-tensor 1.3B manifest.
//!
//! ```text
//! BRAIN_WAN_GGUF      a city96 wan2.1-t2v-*.gguf (7 GB at Q3_K_S; not committed)
//! BRAIN_WAN_GGUF_OUT  where the `#[ignore]`d conversion writes its 53 GiB of
//!                     fp32 output; defaults under the system temp dir
//! ```
//!
//! `BRAIN_WAN_GGUF` is a fixture, so absent it skips via
//! [`brain_testutil::skip`] and `BRAIN_REQUIRE_FIXTURES=1` turns that into a
//! failure. It falls back to whatever `*.gguf` the model store already holds
//! for `city96/Wan2.1-T2V-14B-gguf`, so a box that ran `brain fetch` needs
//! nothing exported.

use std::collections::{BTreeMap, HashMap, HashSet};

use checkpoint::gguf::MmapGguf;
use wan::config::WanConfig;
use wan::import::{dit_config_from_shapes, dit_diffusers_to_native, dit_manifest, dit_native_to_diffusers, GGUF_ARCHITECTURE};

/// The upstream repo the fallback scans, so a box that already ran
/// `brain fetch` does not also need the variable exported.
const REPO: &str = "city96/Wan2.1-T2V-14B-gguf";

/// The first `*.gguf` in the model store's copy of [`REPO`], if there is one -
/// the repo ships one file per quantization, and any of them exercises this
/// path (they differ in which ggml types the 400 projections carry).
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
    let path = match std::env::var("BRAIN_WAN_GGUF") {
        Ok(p) if !p.is_empty() => p,
        _ => match gguf_in_the_store() {
            Some(p) => p,
            None => {
                brain_testutil::skip(&format!("set BRAIN_WAN_GGUF to a {REPO} wan2.1-t2v-*.gguf (none in the model store)"));
                return None;
            }
        },
    };
    if !std::path::Path::new(&path).exists() {
        brain_testutil::skip(&format!("BRAIN_WAN_GGUF={path} not found"));
        return None;
    }
    println!("  gguf {path}");
    Some(MmapGguf::open(&path).expect("open the GGUF"))
}

/// Whether this file is in the diffusers spelling, by the one name only that
/// spelling has - the discriminator `import_dit` itself uses.
fn is_diffusers(mg: &MmapGguf) -> bool {
    mg.names().iter().any(|n| n == "blocks.0.scale_shift_table")
}

/// The source name carrying a given reference name, so a spot check reads the
/// same tensor out of both files whichever spelling the GGUF is in.
fn source_name(mg: &MmapGguf, native: &str) -> String {
    if is_diffusers(mg) {
        dit_native_to_diffusers(native).unwrap_or_else(|| panic!("no diffusers name for {native}"))
    } else {
        native.to_string()
    }
}

/// Every tensor's name in the reference spelling, plus its shape - the same
/// resolution `import_gguf` does, repeated here so a mismatch is reported as a
/// set difference instead of the importer's first-missing-name error.
fn native_names(mg: &MmapGguf) -> HashMap<String, Vec<usize>> {
    let diffusers = is_diffusers(mg);
    let mut out = HashMap::new();
    for n in mg.names() {
        let native = if diffusers {
            dit_diffusers_to_native(n).unwrap_or_else(|| panic!("unmapped diffusers tensor {n}"))
        } else {
            n.clone()
        };
        let shape = mg.shape(n).expect("shape").to_vec();
        assert!(out.insert(native.clone(), shape).is_none(), "two source tensors map to {native}");
    }
    out
}

/// The header alone answers "is this the file the importer thinks it is":
/// architecture tag, derived variant, and the manifest covered in both
/// directions - all off the tensor infos, before a single block is decoded.
#[test]
fn wan_gguf_header_names_the_14b_and_covers_the_manifest() {
    let Some(mg) = open_gguf() else { return };

    let arch = mg.kv().get("general.architecture").and_then(|v| v.as_str());
    assert_eq!(arch, Some(GGUF_ARCHITECTURE), "general.architecture");

    let shapes: Vec<(String, Vec<usize>)> =
        mg.names().iter().map(|n| (n.clone(), mg.shape(n).expect("shape").to_vec())).collect();
    let cfg = dit_config_from_shapes(&shapes).expect("derive the variant from the tensor shapes");
    println!("  {} tensors, derived variant {}", shapes.len(), cfg.name);
    println!("  quant label {:?}, {} parameters", mg.model_card().quant, mg.param_count());
    assert_eq!(cfg.name, WanConfig::t2v_14b().name, "city96 publishes the 14B, not the 1.3B");
    assert_eq!((cfg.dim, cfg.num_layers, cfg.ffn_dim), (5120, 40, 13824));

    // The 5-D `patch_embedding.weight` is the one tensor GGUF cannot express in
    // its own 4-dimension convention, and the variant derivation reads its
    // first two entries. A repacker that flattened it would resolve to a
    // different `dim`/`in_channels` pair entirely, so pin the rank.
    let patch = mg.shape("patch_embedding.weight").expect("patch_embedding.weight");
    assert_eq!(patch, [5120, 16, 1, 2, 2], "torch order, i.e. the GGUF `ne` reversed");

    let got = native_names(&mg);
    let want: BTreeMap<String, Vec<usize>> = dit_manifest(&cfg).into_iter().collect();
    let got_keys: HashSet<&str> = got.keys().map(String::as_str).collect();
    let want_keys: HashSet<&str> = want.keys().map(String::as_str).collect();
    let mut missing: Vec<&&str> = want_keys.difference(&got_keys).collect();
    let mut extra: Vec<&&str> = got_keys.difference(&want_keys).collect();
    missing.sort_unstable();
    extra.sort_unstable();
    assert!(missing.is_empty() && extra.is_empty(), "missing {missing:?}, unused {extra:?}");
    assert_eq!(got.len(), 1095, "40 blocks of 27 + 12 embedding + 3 head");

    for (name, want_shape) in &want {
        let n: usize = want_shape.iter().product();
        let s = &got[name];
        assert_eq!(s.iter().product::<usize>(), n, "{name}: shape {s:?}, manifest {want_shape:?}");
    }
}

/// The quant types the file ACTUALLY uses, decoded on the file's own data.
///
/// Q3_K_S is not one type: city96 leaves every 1-D tensor (biases, the RMSNorm
/// gains, the modulation tables) at F32 and squeezes only the 2-D projections,
/// so this file mixes F32 with Q3_K. `dequantize` returning a clean error for
/// the IQ/TQ codebooks it does not implement is only reassuring if the released
/// file does not use them - which is what the histogram checks.
#[test]
fn wan_gguf_dequantizes_every_tensor_to_finite_values() {
    let Some(mg) = open_gguf() else { return };
    let shapes: Vec<(String, Vec<usize>)> =
        mg.names().iter().map(|n| (n.clone(), mg.shape(n).expect("shape").to_vec())).collect();
    let cfg = dit_config_from_shapes(&shapes).expect("derive the variant");

    let mut dtypes: BTreeMap<&str, usize> = BTreeMap::new();
    for n in mg.names() {
        *dtypes.entry(mg.dtype(n).unwrap_or_else(|| panic!("{n}: no name for its ggml type"))).or_default() += 1;
    }
    println!("  quant types: {dtypes:?}");
    assert!(
        dtypes.keys().any(|t| t.starts_with('Q')),
        "a released quantization must actually carry quantized blocks, not F32/F16 throughout: {dtypes:?}"
    );
    assert!(dtypes.contains_key("F32"), "the 1-D tensors are left at F32: {dtypes:?}");

    // Streaming, one tensor at a time: 14.3 G parameters is 53 GiB as fp32, so
    // a loop that kept them would need the whole model resident to check that
    // no single one is broken.
    let mut worst: (f32, String) = (f32::INFINITY, String::new());
    for (name, shape) in &shapes {
        let numel: usize = shape.iter().product();
        let data = mg.tensor(name).expect("indexed").unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(data.len(), numel, "{name}: dequant produced {} values for shape {shape:?}", data.len());
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
    println!("  smallest mean|w| {:.3e} at {} ({} tensors)", worst.0, worst.1, shapes.len());
    brain_testutil::mem(&format!("after {} streamed dequants", cfg.num_layers));
}

/// The decoded VALUES, against a hand-derivation of ggml's `dequantize_row_q3_K`
/// over this file's own leading super-blocks.
///
/// Every other Q3_K check in the workspace feeds the decoder a block the test
/// itself constructed, which pins the arithmetic and nothing about where the
/// bytes are: the super-block scales are packed six bits at a time across 12
/// bytes, the tensor's base is `align_up(header_end, general.alignment)` plus
/// its own declared offset, and a decoder that got either wrong still returns
/// finite, plausibly-scaled weights. These 40 values were produced from the
/// GGML format definition by a separate numpy implementation reading the same
/// file, and they are bit-exact rather than approximate because dequantization
/// is a multiply of an f16 scale by a small integer.
///
/// The two windows are deliberate: elements 0..32 straddle the first
/// super-block's first two 16-element scale groups (two different unpacked
/// 6-bit scales, 2.29e-2 and 1.29e-2), and 256..264 is the second super-block,
/// which has its own `d`.
#[test]
fn wan_gguf_q3_k_dequant_matches_a_hand_derivation_of_the_ggml_layout() {
    let Some(mg) = open_gguf() else { return };
    const NAME: &str = "blocks.0.self_attn.q.weight";
    // The values below are this ONE quantization's bytes; every other file in
    // the repo re-quantizes the same weights differently.
    let quant = mg.model_card().quant;
    if quant.as_deref() != Some("Q3_K_S") {
        brain_testutil::skip_unavailable(&format!("the store's copy is {quant:?}, and these values are Q3_K_S's"));
        return;
    }
    assert_eq!(mg.dtype(NAME), Some("Q3_K"), "{NAME} carries the file's namesake type");
    #[rustfmt::skip]
    const HEAD32: [f32; 32] = [
        -0.022857666, 0.022857666, -0.0, -0.0, -0.0, -0.0, 0.022857666, 0.022857666,
        -0.0, -0.068573, -0.022857666, -0.0, -0.022857666, -0.045715332, 0.022857666, 0.091430664,
        -0.03857231, -0.0, 0.012857437, 0.05142975, -0.0, 0.012857437, 0.05142975, 0.012857437,
        -0.012857437, -0.0, -0.012857437, -0.0, 0.012857437, 0.012857437, -0.012857437, -0.012857437,
    ];
    #[rustfmt::skip]
    const BLOCK1: [f32; 8] = [
        -0.029125214, -0.0, 0.014562607, -0.0, 0.029125214, 0.058250427, 0.014562607, -0.029125214,
    ];
    let w = mg.tensor(&source_name(&mg, NAME)).expect("in the GGUF").expect("dequant");
    assert_eq!(&w[..32], &HEAD32, "first super-block");
    assert_eq!(&w[256..264], &BLOCK1, "second super-block");
}

/// End to end: GGUF in, brain-native fp32 safetensors out, header read back.
///
/// The conversion is the only part of the path the tests above do not reach,
/// and it is the part that used to be impossible on this file: a 14.3
/// G-parameter fp32 checkpoint is 53 GiB and the eager importer needed roughly
/// three copies of it at once. Streaming it costs one tensor of RAM but still
/// WRITES the whole 53 GiB, so this is disk-bound and takes real wall-clock
/// minutes - hence `#[ignore]`, the same call `qwen3omnimoe`'s full-import test
/// makes for the same reason. Run it with:
///
/// ```text
/// BRAIN_WAN_GGUF_OUT=<54 GiB of scratch> \
///   cargo test --release --offline -p brain-wan --test gguf_import_real -- --ignored
/// ```
#[test]
#[ignore]
fn wan_gguf_converts_to_a_brain_safetensors_checkpoint() {
    let Some(mg) = open_gguf() else { return };
    // Default under the system temp dir, like the other multi-GB import test in
    // this tree; a caller who names a directory gets the output left there.
    let named = std::env::var("BRAIN_WAN_GGUF_OUT").ok().filter(|d| !d.is_empty());
    let out = match &named {
        Some(dir) => format!("{dir}/wan-gguf-import-test.safetensors"),
        None => std::env::temp_dir()
            .join(format!("wan_gguf_import_{}.safetensors", std::process::id()))
            .to_string_lossy()
            .into_owned(),
    };
    let _ = std::fs::remove_file(&out);
    let t0 = std::time::Instant::now();
    wan::import::import_gguf(&mg, &out, Some("city96/wan2.1-t2v-14b-gguf")).expect("import_gguf");
    println!("  import_gguf finished in {:.1}s", t0.elapsed().as_secs_f64());
    brain_testutil::mem("after import_gguf");
    let bytes = std::fs::metadata(&out).expect("stat the output").len();
    assert!(bytes > 50_000_000_000, "output is {bytes} bytes, far short of the 14B's fp32 size");

    let cfg = WanConfig::t2v_14b();
    let manifest = dit_manifest(&cfg);
    let params: u64 = manifest.iter().map(|(_, s)| s.iter().product::<usize>() as u64).sum();

    let header = checkpoint::st::read_metadata(&out).expect("metadata");
    assert_eq!(header.get("family").map(String::as_str), Some("wan"));
    assert_eq!(header.get("id").map(String::as_str), Some("city96/wan2.1-t2v-14b-gguf"));
    assert_eq!(header.get("param_count").map(String::as_str), Some(params.to_string().as_str()));

    // Read the written file back the way a loader does, one tensor at a time,
    // and compare against a fresh dequant of the source - which is also the
    // check that the name remap landed each tensor in the right slot.
    let st = checkpoint::mmap::MmapSafetensors::open(&out).expect("mmap the output");
    assert_eq!(st.names().len(), manifest.len(), "tensor count");
    let spot = [
        "patch_embedding.weight",
        "blocks.0.self_attn.q.weight",
        "blocks.39.ffn.2.weight",
        "blocks.17.cross_attn.norm_k.weight",
        "head.head.weight",
        "head.modulation",
    ];
    for name in spot {
        let want_shape = &manifest.iter().find(|(n, _)| n == name).expect("in the manifest").1;
        assert_eq!(st.shape(name).expect("shape"), want_shape.as_slice(), "{name}: declared shape");
        let got = st.tensor_f32(name).unwrap_or_else(|| panic!("{name}: readable"));
        let from = source_name(&mg, name);
        let src = mg.tensor(&from).expect("in the GGUF").expect("dequant");
        assert_eq!(got, src, "{name}: round trip through the writer");
    }
    println!("  wrote {} tensors / {params} parameters ({bytes} bytes) to {out}", manifest.len());
    if named.is_none() {
        std::fs::remove_file(&out).expect("remove the converted checkpoint");
    }
}
