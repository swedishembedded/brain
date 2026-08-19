// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The generic GGUF import registry and its one CLI surface,
//! `brain import-gguf FILE [--out PATH] [--id NAME]`.
//!
//! brain reads GGUF generically already: [`checkpoint::gguf`] parses v2/v3
//! headers and dequantizes every mainstream GGML quant to fp32, and
//! [`crate::model_dir`] scans `$BRAIN_MODELS_DIR`, synthesizes a `ModelCard`
//! from each file's KV, and dispatches BY FAMILY. What was NOT generic was
//! architecture-specific *conversion*: `qwen35moe` had a hand-written importer
//! reachable only through its own bespoke `brain qwen35moe import` subcommand,
//! invisible to auto-discovery. Adding a second architecture that way would
//! mean inventing a second subcommand and separately remembering to wire it
//! into the scan - which is exactly what did not happen the first time.
//!
//! Here, an architecture is added by implementing
//! [`GgufArchitectureImporter`] and adding ONE line to [`IMPORTERS`]. The CLI
//! command, the `--list` output, the "unknown architecture" error message and
//! `model_dir`'s discovery hint all read that same table, so they cannot drift
//! apart - the same single-table discipline `model_dir::resident_for` applies
//! to family→resident dispatch.
//!
//! ## Design decision: explicit one-time import, NOT auto-conversion on scan
//!
//! Discovering a `.gguf` with a registered architecture does **not** silently
//! convert it. It logs an actionable line naming the exact command to run, and
//! auto-discovery picks up the resulting `.safetensors` on the next scan.
//! The reasons, in the order that decided it:
//!
//! 1. **Disk.** This importer's output is ALWAYS fp32 safetensors - brain's
//!    core-compute-only invariant is that ARITHMETIC stays fp32, not that
//!    on-disk storage does (some architectures, e.g. `wan`, load a GGUF
//!    DIRECTLY at inference and keep its weights quantized in device storage,
//!    dequantizing only inside the GEMM), but this command's whole contract
//!    is "produce a brain-native checkpoint the registry can discover
//!    uniformly", and that checkpoint format is fp32. So importing a
//!    quantized GGUF still materializes a much larger file here: a 22 GB
//!    Q4_K_M GGUF of a 35B model becomes roughly 140 GB of fp32 safetensors. A
//!    server-startup scan that can fill a disk is not a defensible default,
//!    and there is no honest way to ask for consent from inside a directory
//!    walk.
//! 2. **Startup time.** `discover` runs on every `brain serve` start and is
//!    expected to be a header-read-only pass (`read_card` / GGUF KV only, no
//!    tensor data). A conversion is a minutes-to-hours full dequantize+rewrite
//!    of every tensor. Making startup latency depend on what someone dropped
//!    into a directory turns a fast, predictable scan into an unbounded one.
//! 3. **Idempotence is not free.** "Convert only if the output is missing"
//!    sounds cheap but has to decide whether an existing output is stale,
//!    complete, or half-written by a crashed earlier run - real state to get
//!    wrong, silently, on a serving path. An explicit command fails visibly in
//!    the operator's terminal instead.
//! 4. **It matches the rest of the scan's contract.** `discover` is documented
//!    as non-fatal and side-effect-free: it logs and skips anything it cannot
//!    serve. Writing 140 GB to disk is not "logging and skipping".
//!
//! What the registry DOES buy over the old bespoke subcommand: one generic
//! command for every architecture, discovery that can *name* the fix instead
//! of reporting a dead end, and a single place to register the next one.

use checkpoint::gguf::MmapGguf;

/// Converts one GGUF architecture into a brain-native checkpoint.
///
/// Implementations are thin adapters over an architecture crate's existing
/// import logic (tensor-name mapping, config derivation) - the trait exists to
/// make that logic *pluggable*, not to re-home it.
pub trait GgufArchitectureImporter: Sync {
    /// The GGUF `general.architecture` metadata value this importer claims,
    /// matched EXACTLY (llama.cpp's own spelling, e.g. `"qwen35moe"`).
    fn architecture(&self) -> &'static str;

    /// One line for `brain import-gguf --list`.
    fn summary(&self) -> &'static str;

    /// Convert an already-open GGUF into a brain-native safetensors checkpoint
    /// at `out_path`, carrying a `ModelCard` whose family
    /// [`crate::model_dir::resident_for`] can dispatch. `id_override` names the
    /// card id (else the importer's own default).
    ///
    /// Takes the open [`MmapGguf`] rather than a path because the registry has
    /// to parse the header to find the architecture before it can pick an
    /// importer at all - passing the handle through avoids a second mmap.
    fn import(&self, gguf: &MmapGguf, out_path: &str, id_override: Option<&str>) -> Result<(), String>;
}

/// Qwen3.5-35B-A3B (`general.architecture = "qwen35moe"`). A registration, not
/// a reimplementation: the llama.cpp↔HF tensor-name mapping this delegates to
/// was hand-derived against a real checkpoint header and is tested in place -
/// see `qwen35moe::import`'s module doc.
struct Qwen35MoeImporter;

impl GgufArchitectureImporter for Qwen35MoeImporter {
    fn architecture(&self) -> &'static str {
        qwen35moe::import::GGUF_ARCHITECTURE
    }
    fn summary(&self) -> &'static str {
        "Qwen3.5-35B-A3B hybrid Gated-DeltaNet/GQA sparse-MoE decoder"
    }
    fn import(&self, gguf: &MmapGguf, out_path: &str, id_override: Option<&str>) -> Result<(), String> {
        qwen35moe::import::import_mmap(gguf, out_path, id_override)
    }
}

/// Z-Image (S³-DiT), DiT only (`general.architecture = "lumina2"` - shared
/// with real Lumina2 releases; `s3dit::import::import_gguf` refuses to guess
/// and checks for a Z-Image-only tensor before converting anything, since
/// this registry has no per-architecture discriminator the way
/// `crates/gguf::registry`'s `clip.projector_type` case does).
///
/// The dequantize -> remap -> safetensors path behind it is exercised against a
/// real `unsloth/Z-Image-Turbo-GGUF` file by `crates/s3dit`'s
/// `gguf_import_real` suite (`BRAIN_S3DIT_GGUF`), the same arrangement
/// [`WanImporter`] has: only the dispatch is covered here.
struct S3ditImporter;

impl GgufArchitectureImporter for S3ditImporter {
    fn architecture(&self) -> &'static str {
        s3dit::import::GGUF_ARCHITECTURE
    }
    fn summary(&self) -> &'static str {
        "Z-Image S3-DiT text-to-image (DiT only - VAE/text-encoder come from their own source)"
    }
    fn import(&self, gguf: &MmapGguf, out_path: &str, id_override: Option<&str>) -> Result<(), String> {
        s3dit::import::import_gguf(gguf, out_path, id_override)
    }
}

/// Wan2.1/2.2 video diffusion, DiT only (`general.architecture = "wan"`).
///
/// The contrast with [`S3ditImporter`] is the point: `wan`'s brain id IS its
/// GGUF spelling, so `brain_arch::by_gguf("wan")` resolves and no
/// ambiguous-tag exception is needed - nothing else in the ecosystem claims
/// the tag, and Wan2.1 and Wan2.2 share it deliberately (see the `wan` row in
/// `crates/arch`). The variant is read off the checkpoint's own tensor
/// shapes rather than any metadata field; see `wan::import::import_gguf`.
struct WanImporter;

impl GgufArchitectureImporter for WanImporter {
    fn architecture(&self) -> &'static str {
        wan::import::GGUF_ARCHITECTURE
    }
    fn summary(&self) -> &'static str {
        "Wan2.1/2.2 text-to-video transformer (DiT only - VAE/umT5/tokenizer come from their own source)"
    }
    fn import(&self, gguf: &MmapGguf, out_path: &str, id_override: Option<&str>) -> Result<(), String> {
        wan::import::import_gguf(gguf, out_path, id_override)
    }
}

/// LTX-2.5 two-stream audio+video diffusion transformer, DiT only
/// (`general.architecture = "ltxv"`).
///
/// Same shape as [`WanImporter`]: `ltxv`'s brain id IS its GGUF spelling
/// (`crates/arch`'s `ltxv` row, `gguf: None`), so `brain_arch::by_gguf
/// ("ltxv")` resolves and no ambiguous-tag exception is needed. Unlike
/// `wan::import::import_gguf`, there is no diffusers<->native name
/// remapping to choose between - the real checkpoint carries only one
/// tensor spelling, which already IS `crate::block::LtxBlock`/`LtxAvBlock`'s
/// own `tget` names (see `ltxv::gguf_src`'s module doc). The config comes
/// from the checkpoint's own embedded `config` KV (a JSON blob), not from
/// tensor shapes - see `ltxv::import::av_dit_config_from_kv`'s doc for why
/// that differs from Wan's shape-derived variant lookup.
struct LtxvImporter;

impl GgufArchitectureImporter for LtxvImporter {
    fn architecture(&self) -> &'static str {
        ltxv::import::GGUF_ARCHITECTURE
    }
    fn summary(&self) -> &'static str {
        "LTX-2.5 audio+video diffusion transformer (AV DiT only - VAEs/text-encoder/tokenizer come from their own source)"
    }
    fn import(&self, gguf: &MmapGguf, out_path: &str, id_override: Option<&str>) -> Result<(), String> {
        ltxv::import::import_gguf(gguf, out_path, id_override)
    }
}

/// Every registered architecture importer. ONE line per architecture - this is
/// the whole registration surface (see this module's doc).
const IMPORTERS: &[&dyn GgufArchitectureImporter] = &[&Qwen35MoeImporter, &S3ditImporter, &WanImporter, &LtxvImporter];

/// The importer claiming `architecture`, or `None` if none does.
pub fn importer_for(architecture: &str) -> Option<&'static dyn GgufArchitectureImporter> {
    IMPORTERS.iter().copied().find(|i| i.architecture() == architecture)
}

/// Every registered `general.architecture` value, for error messages and
/// `--list`.
pub fn architectures() -> Vec<&'static str> {
    IMPORTERS.iter().map(|i| i.architecture()).collect()
}

/// Read a GGUF's `general.architecture`, or `""` when the key is absent.
fn architecture_of(mg: &MmapGguf) -> &str {
    mg.kv().get("general.architecture").and_then(|v| v.as_str()).unwrap_or("")
}

/// The default output path for `gguf_path`: a sibling `<stem>.brain.safetensors`,
/// so the conversion lands in the SAME model-store directory the source came
/// from and the next `discover` scan finds it with no extra move.
pub fn default_out_path(gguf_path: &str) -> String {
    let p = std::path::Path::new(gguf_path);
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    p.with_file_name(format!("{stem}.brain.safetensors")).to_string_lossy().into_owned()
}

/// Open `gguf_path`, look its `general.architecture` up in the registry, and
/// run the matching importer. Returns the path actually written.
///
/// An unregistered architecture is a clear, actionable error naming what WAS
/// registered - never a panic, and never a silent no-op.
pub fn import_file(gguf_path: &str, out_path: Option<&str>, id: Option<&str>) -> Result<String, String> {
    let mg = MmapGguf::open(gguf_path)?;
    let arch = architecture_of(&mg);
    let importer = importer_for(arch).ok_or_else(|| {
        let known = architectures().join(", ");
        if arch.is_empty() {
            format!("{gguf_path}: no 'general.architecture' in the GGUF metadata (registered architectures: {known})")
        } else if let Some(a) = brain_arch::by_gguf(arch) {
            // brain names this architecture (it has a canonical id) but no
            // GgufArchitectureImporter claims it yet -- a real gap, not an
            // unrecognized file, so say so precisely instead of the generic
            // "no importer registered" wording.
            format!("{gguf_path}: architecture {arch:?} is brain's {:?} ({}), but has no GGUF importer yet (registered: {known})", a.id, a.display)
        } else {
            format!("{gguf_path}: no importer registered for GGUF architecture {arch:?} (registered: {known})")
        }
    })?;
    let out = out_path.map(str::to_string).unwrap_or_else(|| default_out_path(gguf_path));
    importer.import(&mg, &out, id)?;
    Ok(out)
}

/// `brain import-gguf FILE [--out PATH] [--id NAME]` / `brain import-gguf --list`:
/// the ONE generic conversion command, dispatching by the file's own
/// `general.architecture` through [`import_file`]. Replaces the per-model
/// `brain qwen35moe import` (which still works, as a thin wrapper).
pub fn run_import_gguf(args: &[String]) {
    let usage = "usage: brain import-gguf FILE [--out PATH] [--id VENDOR/REPO]\n       brain import-gguf --list";
    if args.iter().any(|a| a == "--list") {
        println!("registered GGUF architectures (general.architecture -> importer):");
        for i in IMPORTERS {
            println!("  {:<12} {}", i.architecture(), i.summary());
        }
        return;
    }
    let mut file: Option<String> = None;
    let mut out: Option<String> = None;
    let mut id: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" | "--gguf" | "--id" => {
                let flag = args[i].clone();
                i += 1;
                let Some(v) = args.get(i).cloned() else {
                    eprintln!("{flag} requires a value\n{usage}");
                    std::process::exit(2);
                };
                match flag.as_str() {
                    // `--gguf` is accepted as an alias for the positional FILE so
                    // the old `brain qwen35moe import --gguf F --out O` spelling
                    // keeps working verbatim through the generic command.
                    "--gguf" => file = Some(v),
                    "--out" => out = Some(v),
                    _ => id = Some(v),
                }
            }
            "-h" | "--help" => {
                println!("{usage}");
                return;
            }
            other if other.starts_with("--") => eprintln!("ignoring unknown flag {other:?}"),
            other if file.is_none() => file = Some(other.to_string()),
            other => eprintln!("ignoring extra argument {other:?}"),
        }
        i += 1;
    }
    let Some(file) = file else {
        eprintln!("{usage}");
        std::process::exit(2);
    };
    match import_file(&file, out.as_deref(), id.as_deref()) {
        Ok(out) => eprintln!("import-gguf: {file} -> {out}"),
        Err(e) => {
            eprintln!("brain import-gguf: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic GGUF carrying `general.architecture` and one tensor -
    /// enough to exercise DISPATCH (open → read architecture → pick importer).
    /// The tensor mapping itself is proven by `qwen35moe::import`'s own
    /// fixture-based tests; this file is about routing.
    fn write_gguf(path: &str, arch: &str) {
        use checkpoint::gguf::GgufValue;
        use checkpoint::gguf_write::{write, TensorOut};
        let kvs = vec![("general.architecture".to_string(), GgufValue::String(arch.to_string()))];
        let tensors = vec![TensorOut { name: "w".into(), shape: vec![4], ty: 0, data: (0..4u32).flat_map(|i| (i as f32).to_le_bytes()).collect() }];
        write(path, &kvs, &tensors, 32).unwrap();
    }

    fn tmp(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("brain-gguf-import-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn the_registry_is_keyed_by_the_real_gguf_architecture_string() {
        assert_eq!(importer_for("qwen35moe").map(|i| i.architecture()), Some("qwen35moe"));
        assert_eq!(importer_for("qwen35moe").map(|i| i.architecture()), Some(qwen35moe::import::GGUF_ARCHITECTURE));
        assert!(importer_for("llama").is_none());
        assert!(architectures().contains(&"qwen35moe"));
    }

    /// Every `GgufArchitectureImporter` claims a spelling `brain_arch` also
    /// knows -- this and `brain_arch`'s own table cannot drift apart, which is
    /// what lets `import_file`'s error message tell "brain has a name for
    /// this but no importer" apart from "brain has never heard of this".
    ///
    /// `s3dit`'s spelling ("lumina2") is the one documented exception: it is
    /// shared with real Lumina2 releases (see `s3dit::import::GGUF_ARCHITECTURE`'s
    /// doc), so `brain_arch` deliberately does NOT claim it as s3dit's `gguf:`
    /// spelling -- `import_gguf` discriminates by tensor presence instead,
    /// not by architecture string. This test asserts that non-resolution is
    /// intentional, not a silent gap.
    #[test]
    fn every_registered_importer_matches_a_brain_arch_row_or_is_a_documented_ambiguous_tag() {
        const AMBIGUOUS_TAG_EXCEPTIONS: &[&str] = &["lumina2"];
        for i in IMPORTERS {
            let arch = i.architecture();
            if AMBIGUOUS_TAG_EXCEPTIONS.contains(&arch) {
                assert!(brain_arch::by_gguf(arch).is_none(), "importer {arch:?} is a documented ambiguous-tag exception and must NOT resolve via by_gguf");
                continue;
            }
            assert!(brain_arch::by_gguf(arch).is_some(), "importer {arch:?} has no matching brain_arch row");
        }
    }

    #[test]
    fn a_named_but_unimported_architecture_gets_a_more_specific_error() {
        let dir = tmp("named-unimported");
        // deepseek2ocr's GGUF spelling ("deepseek2-ocr") is a real brain_arch
        // row with no GgufArchitectureImporter yet -- the error should say so
        // by name, not just "no importer registered".
        let src = dir.join("named.gguf").to_string_lossy().into_owned();
        write_gguf(&src, "deepseek2-ocr");
        let err = import_file(&src, None, None).unwrap_err();
        assert!(err.contains("deepseek2ocr"), "must name the known brain_arch id: {err}");
        assert!(err.contains("no GGUF importer"), "must say it is a known-but-unimported gap: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Dispatch routes a `general.architecture = "qwen35moe"` file to the
    /// qwen35moe importer: the importer runs and reports ITS OWN error (this
    /// synthetic file has an architecture but none of the required KV), which
    /// is only reachable if the lookup picked the right importer.
    #[test]
    fn a_registered_architecture_dispatches_to_its_importer() {
        let dir = tmp("dispatch");
        let src = dir.join("toy.gguf").to_string_lossy().into_owned();
        write_gguf(&src, qwen35moe::import::GGUF_ARCHITECTURE);
        let err = import_file(&src, None, None).unwrap_err();
        assert!(err.contains("qwen35"), "must reach the qwen35moe importer, got: {err}");
        assert!(!err.contains("no importer registered"), "must not fall through to the unknown-architecture error: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The whole point of the registry: a real end-to-end convert driven only
    /// by the file's own architecture string (no architecture named at the call
    /// site), whose output the model-dir scan then picks up on its own. Reuses
    /// `qwen35moe::import`'s synthetic fixture builder rather than a second copy.
    #[test]
    fn dispatch_converts_a_registered_architecture_and_the_scan_then_serves_it() {
        let dir = tmp("e2e");
        let src = dir.join("toy.gguf").to_string_lossy().into_owned();
        qwen35moe::import::testing::write_synthetic_gguf(&src);
        // The sibling tokenizer.json the qwen35 resident needs (never opened
        // here - only at activate()).
        std::fs::write(dir.join("tokenizer.json"), br#"{"model":{"vocab":{"a":0}}}"#).unwrap();

        let out = import_file(&src, None, Some("test/qwen35-tiny")).expect("registry dispatch must convert");
        assert_eq!(out, default_out_path(&src), "default output is a sibling .brain.safetensors");

        let reader = checkpoint::weightio::WeightReader::open(&out).unwrap();
        assert!(reader.tensor("tok.weight").is_some(), "the importer really ran");
        assert_eq!(reader.card().expect("a model card must be written").id, "test/qwen35-tiny");

        // The end of the loop: auto-discovery serves the conversion with no env
        // vars and no per-model wiring. The `.gguf` itself stays unregistered
        // (it needs the conversion, which is the whole design decision above).
        let ids: Vec<String> = crate::model_dir::discover(&dir).iter().map(|r| r.manifest().model).collect();
        assert!(ids.contains(&"test/qwen35-tiny".to_string()), "the imported checkpoint must be discovered: {ids:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Wan's registration, and an honest statement of its coverage.
    ///
    /// **What this proves.** The tag `"wan"` routes to [`WanImporter`] (the
    /// error is the wan importer's own, not the registry's "no importer
    /// registered"); the id needs no `AMBIGUOUS_TAG_EXCEPTIONS` row because
    /// brain's architecture id IS the GGUF spelling, unlike `s3dit`'s
    /// `"lumina2"`; and a file that carries the tag but is not a Wan
    /// transformer is refused by name rather than half-imported.
    ///
    /// **What this does NOT prove.** Only the dispatch. The dequantize ->
    /// `import_dit` -> safetensors path itself is exercised against a real
    /// `city96/Wan2.1-T2V-14B-gguf` file by `crates/wan`'s
    /// `gguf_import_real` suite (`BRAIN_WAN_GGUF`), which is where the
    /// manifest coverage, the Q3_K dequantization and the round-tripped
    /// checkpoint are actually certified on real quantized bytes.
    #[test]
    fn the_wan_tag_dispatches_to_its_importer_with_no_ambiguous_tag_exception() {
        assert_eq!(importer_for("wan").map(|i| i.architecture()), Some("wan"));
        assert_eq!(brain_arch::by_gguf("wan").map(|a| a.id), Some("wan"), "wan's id is its own GGUF spelling; an alias here means the id drifted");

        let dir = tmp("wan-dispatch");
        let src = dir.join("wan.gguf").to_string_lossy().into_owned();
        write_gguf(&src, wan::import::GGUF_ARCHITECTURE);
        let err = import_file(&src, None, None).unwrap_err();
        assert!(err.contains("patch_embedding.weight"), "must reach the wan importer's own check, got: {err}");
        assert!(!err.contains("no importer registered"), "must not fall through to the unknown-architecture error: {err}");
        assert!(!std::path::Path::new(&default_out_path(&src)).exists(), "a refused import must not leave an output file");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// ltxv's registration, dispatch AND a full convert - unlike the Wan test
    /// above (which only proves dispatch, because a real Wan fixture is too
    /// large to build here), a small-but-structurally-complete AV DiT
    /// manifest is cheap enough to build in-process, so this exercises the
    /// WHOLE convert path: registry dispatch -> `av_dit_config_from_kv` off
    /// the embedded `config` KV -> two-way manifest coverage -> streamed
    /// dequant+write, and reads the result back.
    ///
    /// **What this does NOT prove.** `model_dir::resident_for`/
    /// `resident_for_compound` serving the converted checkpoint - `ltxv` is
    /// a COMPOUND model (DiT + two VAEs + text encoder + tokenizer, like
    /// `wan`'s own `zimage`/`wan` arms in `resident_for_compound`), and
    /// wiring a compound residency arm for it is a distinct, later concern
    /// from this converter (this crate's `ltxv_cli`/`crate::pipeline`
    /// already loads real+random weights through its own `BRAIN_LTXV_*`
    /// env-var convention, independent of `model_dir`'s residency registry).
    #[test]
    fn the_ltxv_tag_dispatches_and_converts() {
        assert_eq!(importer_for("ltxv").map(|i| i.architecture()), Some("ltxv"));
        assert_eq!(brain_arch::by_gguf("ltxv").map(|a| a.id), Some("ltxv"), "ltxv's id is its own GGUF spelling; an alias here means the id drifted");

        use checkpoint::gguf::GgufValue;
        use checkpoint::gguf_write::{write, TensorOut};
        use ltxv::config::LtxAvDitConfig;
        use ltxv::dit::av_dit_tensor_manifest;

        let dir = tmp("ltxv-dispatch");
        let src = dir.join("ltxv.gguf").to_string_lossy().into_owned();

        let cfg = LtxAvDitConfig::tiny();
        let manifest = av_dit_tensor_manifest(&cfg);
        let mut seed = 0u64;
        let tensors: Vec<TensorOut> = manifest
            .iter()
            .map(|(name, shape)| {
                seed += 1;
                let n: usize = shape.iter().product();
                let data: Vec<u8> = (0..n).flat_map(|i| (((i as u64 + seed) % 997) as f32 * 0.01 - 5.0).to_le_bytes()).collect();
                TensorOut { name: name.clone(), shape: shape.clone(), ty: 0u32, data }
            })
            .collect();
        let config_kv = serde_json::json!({
            "transformer": {
                "num_attention_heads": cfg.video.num_heads,
                "attention_head_dim": cfg.video.head_dim(),
                "num_layers": cfg.video.num_layers,
                "in_channels": cfg.video.in_channels,
                "out_channels": cfg.video.out_channels,
                "cross_attention_dim": cfg.video.cross_attention_dim,
                "ff_bias": cfg.video.ff_bias,
                "cross_attention_adaln": cfg.video.cross_attention_adaln,
                "use_keyframes_abs_pos_embedding": cfg.video.use_keyframes_abs_pos_embedding,
                "norm_eps": cfg.video.norm_eps,
                "positional_embedding_theta": cfg.video.positional_embedding_theta,
                "positional_embedding_max_pos": cfg.video.positional_embedding_max_pos,
                "timestep_scale_multiplier": cfg.video.timestep_scale_multiplier,
                "use_middle_indices_grid": cfg.video.use_middle_indices_grid,
                "apply_gated_attention": cfg.video.apply_gated_attention,
                "connector_num_layers": cfg.video.connector_num_layers,
                "connector_num_attention_heads": cfg.video.connector_num_attention_heads,
                "connector_attention_head_dim": cfg.video.connector_attention_head_dim,
                "connector_num_learnable_registers": cfg.video.connector_num_learnable_registers,
                "connector_positional_embedding_max_pos": cfg.video.connector_positional_embedding_max_pos,
                "connector_apply_gated_attention": cfg.video.connector_apply_gated_attention,
                "connector_norm_output": cfg.video.connector_norm_output,
                "caption_proj_before_connector": cfg.video.caption_proj_before_connector,
                "audio_num_attention_heads": cfg.audio.num_heads,
                "audio_attention_head_dim": cfg.audio.head_dim(),
                "audio_out_channels": cfg.audio.out_channels,
                "audio_cross_attention_dim": cfg.audio.cross_attention_dim,
                "audio_positional_embedding_max_pos": cfg.audio.positional_embedding_max_pos,
                "audio_connector_num_attention_heads": cfg.audio.connector_num_attention_heads,
                "audio_connector_attention_head_dim": cfg.audio.connector_attention_head_dim,
                "av_ca_timestep_scale_multiplier": cfg.av_ca_timestep_scale_multiplier,
            },
        })
        .to_string();
        let kvs = vec![("general.architecture".to_string(), GgufValue::String("ltxv".to_string())), ("config".to_string(), GgufValue::String(config_kv))];
        write(&src, &kvs, &tensors, 32).unwrap();

        let out = import_file(&src, None, Some("test/ltxv-tiny")).expect("registry dispatch must convert a fully-covered fixture");
        assert_eq!(out, default_out_path(&src));

        let reader = checkpoint::weightio::WeightReader::open(&out).unwrap();
        assert!(reader.tensor("patchify_proj.weight").is_some(), "the importer really ran");
        assert_eq!(reader.card().expect("a model card must be written").id, "test/ltxv-tiny");
        for (name, shape) in &manifest {
            let n: usize = shape.iter().product();
            assert_eq!(reader.tensor(name).map(|d| d.len()), Some(n), "{name}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_unregistered_architecture_is_a_clear_error_not_a_panic_or_a_silent_no_op() {
        let dir = tmp("unknown");
        let src = dir.join("mystery.gguf").to_string_lossy().into_owned();
        write_gguf(&src, "llama");
        let err = import_file(&src, None, None).unwrap_err();
        assert!(err.contains("llama"), "the error must name the unsupported architecture: {err}");
        assert!(err.contains("qwen35moe"), "the error must list what IS registered: {err}");
        // Nothing was written.
        assert!(!std::path::Path::new(&default_out_path(&src)).exists(), "a failed dispatch must not leave an output file");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_gguf_with_no_architecture_key_errors_clearly() {
        let dir = tmp("noarch");
        let src = dir.join("bare.gguf").to_string_lossy().into_owned();
        use checkpoint::gguf_write::{write, TensorOut};
        let tensors = vec![TensorOut { name: "w".into(), shape: vec![1], ty: 0, data: 0f32.to_le_bytes().to_vec() }];
        write(&src, &[], &tensors, 32).unwrap();
        let err = import_file(&src, None, None).unwrap_err();
        assert!(err.contains("general.architecture"), "the error must name the missing key: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn default_out_path_is_a_sibling_brain_safetensors() {
        assert!(default_out_path("/a/b/Qwen3.5-35B-A3B-Q4_K_M.gguf").ends_with("/a/b/Qwen3.5-35B-A3B-Q4_K_M.brain.safetensors"));
    }
}
