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
//! ## One table, after there were two
//!
//! This table is the only one. `crates/gguf` used to carry a second
//! architecture table of its own, holding DeepSeek-OCR's two halves, that
//! nothing outside that crate's tests ever called. The cost was not
//! theoretical: `brain import-gguf` on a real DeepSeek-OCR file reported
//! "architecture 'deepseek2-ocr' ... has no GGUF importer yet" while the
//! importer sat two crates away. Those rows are now here, beside every other
//! architecture. What stayed in `crates/gguf` is `gguf::route`, which answers
//! the prior question - WHICH architecture is this file - once, against the
//! canonical registry (`brain_arch::by_gguf`), for every consumer.
//!
//! ## Conversion is not the only way to consume a GGUF
//!
//! Some architectures read a GGUF directly at inference, streaming one weight
//! matrix at a time out of the mapping and never materializing an fp32 model
//! (FLUX.2's Q8_0 tier, `wan::gguf_src`, `ltxv::gguf_src`, Qwen3-VL). Forcing
//! those through a dequantize-everything conversion would throw away the
//! saving that makes them work on one card, so the table has a column for it
//! rather than an opinion: [`GgufArchitectureImporter::loads_directly`]. A
//! caller holding a `.gguf` path reads that column and decides; an
//! architecture that loads directly and has no conversion answers with the
//! command that DOES work, never "unsupported".
//!
//! ## Two files, one model
//!
//! A vision-language checkpoint is a model GGUF plus an `mmproj-*.gguf`, and
//! every projector file ever produced declares `general.architecture =
//! "clip"`. So an entry may also claim a `clip.projector_type`
//! ([`GgufArchitectureImporter::projector`]), and lookup prefers the more
//! specific match - otherwise the first `clip` row would swallow every other
//! model's vision tower and import it with the wrong tensor map.
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

    /// An extra `clip.projector_type` this entry also requires, for the one
    /// architecture several models share: every multimodal projector file
    /// llama.cpp's mtmd tooling produces declares `general.architecture =
    /// "clip"` and identifies its real owner only in that second key. `None`
    /// for an ordinary model file, and an entry claiming `"clip"` without one
    /// would swallow every other model's projector - which a test below
    /// refuses.
    fn projector(&self) -> Option<&'static str> {
        None
    }

    /// Whether this architecture's own runtime ALSO reads a `.gguf` directly
    /// at inference, with no conversion step.
    ///
    /// The opt-in that keeps a fast path fast. A direct loader streams one
    /// weight matrix at a time out of the mapping and can feed a quantized
    /// tier without ever materializing the fp32 model, which is a real memory
    /// win and not something a generic "dequantize everything to safetensors"
    /// route can offer. Declaring it here is what lets a caller holding a
    /// `.gguf` path decide between handing it straight to the model and
    /// converting first, from the one table, rather than from a per-model
    /// branch.
    fn loads_directly(&self) -> bool {
        false
    }

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

/// Qwen3.8-27B dense hybrid decoder (`general.architecture = "qwen35"`) -
/// `qwen35moe`'s dense sibling. A registration, not a reimplementation: the
/// GDN/GQA/dense-MLP leaf vocabulary is `gguf::leaf`'s, shared with
/// `qwen35moe`'s own importer above; what is unique here is importing the MTP
/// head (`qwen35moe` drops its own MTP block, this one does not) - see
/// `qwen35::gguf_import`'s module doc.
struct Qwen35Importer;

impl GgufArchitectureImporter for Qwen35Importer {
    fn architecture(&self) -> &'static str {
        qwen35::gguf_import::GGUF_ARCHITECTURE
    }
    fn summary(&self) -> &'static str {
        "Qwen3.8-27B dense hybrid Gated-DeltaNet/GQA decoder + MTP"
    }
    fn import(&self, gguf: &MmapGguf, out_path: &str, id_override: Option<&str>) -> Result<(), String> {
        qwen35::gguf_import::import_mmap(gguf, out_path, id_override).map(|_| ())
    }
}

/// Dense Qwen3 (`general.architecture = "qwen3"`) - the decoder LM, and
/// FLUX.2's text encoder.
///
/// A registration, not a reimplementation. Unlike [`Qwen35MoeImporter`],
/// whose map was hand-derived from a real checkpoint header, this one is
/// transcribed from llama.cpp's own `tensor_mapping.py`/`constants.py` at a
/// named revision and gated bit-for-bit against the safetensors route - see
/// `qwen3::gguf_import`'s module doc.
struct Qwen3Importer;

impl GgufArchitectureImporter for Qwen3Importer {
    fn architecture(&self) -> &'static str {
        qwen3::gguf_import::GGUF_ARCHITECTURE
    }
    fn loads_directly(&self) -> bool {
        // `qwen3::import::source`/`shard_source` sniff the naming convention
        // from the file's own tensor names, so a `.gguf` is a first-class
        // weights path everywhere a Qwen3 is loaded.
        true
    }
    fn summary(&self) -> &'static str {
        "Qwen3 dense decoder (GQA + QK-norm + RoPE + SwiGLU); also FLUX.2's text encoder"
    }
    fn import(&self, gguf: &MmapGguf, out_path: &str, id_override: Option<&str>) -> Result<(), String> {
        qwen3::gguf_import::import_mmap(gguf, out_path, id_override).map(|_| ())
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

/// SUPIR photo-realistic restoration (`general.architecture = "sdxl"` -
/// borrowed, not real: no GGUF file carrying a SUPIR delta has ever been
/// observed, see `supir::import::GGUF_ARCHITECTURE`'s own doc for why this
/// spelling was chosen anyway).
///
/// A SECOND documented ambiguous-tag exception, same shape as
/// [`S3ditImporter`]'s `"lumina2"`: `"sdxl"` is exactly the tag a real,
/// vanilla SDXL GGUF conversion would also plausibly carry, and `sdxlunet`
/// itself claims no GGUF spelling today - so `brain_arch` deliberately does
/// NOT resolve `"sdxl"` to this importer either. `supir::import::import_gguf`
/// reflects the same "nothing to convert yet" honesty this module's own
/// design doc asks for (§ "explicit one-time import") rather than guessing a
/// tensor-name mapping against a file that has never been seen.
struct SupirImporter;

impl GgufArchitectureImporter for SupirImporter {
    fn architecture(&self) -> &'static str {
        supir::import::GGUF_ARCHITECTURE
    }
    fn summary(&self) -> &'static str {
        "SUPIR photo-realistic restoration (no real GGUF release observed yet - registered so one auto-dispatches the day it exists)"
    }
    fn import(&self, gguf: &MmapGguf, out_path: &str, id_override: Option<&str>) -> Result<(), String> {
        supir::import::import_gguf(gguf, out_path, id_override)
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
    fn loads_directly(&self) -> bool {
        // `wan::gguf_src::WanGgufSource` is a `checkpoint::TensorSource` over
        // the mapping itself.
        true
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
    fn loads_directly(&self) -> bool {
        // `ltxv::gguf_src::LtxvGgufSource`, the same shape as Wan's.
        true
    }
    fn summary(&self) -> &'static str {
        "LTX-2.5 audio+video diffusion transformer (AV DiT only - VAEs/text-encoder/tokenizer come from their own source)"
    }
    fn import(&self, gguf: &MmapGguf, out_path: &str, id_override: Option<&str>) -> Result<(), String> {
        ltxv::import::import_gguf(gguf, out_path, id_override)
    }
}

/// DeepSeek-OCR's decoder half (`general.architecture = "deepseek2-ocr"`).
///
/// This entry and [`DeepseekOcrVisionImporter`] below used to live in a
/// SECOND architecture table, in `crates/gguf`'s own `registry` module, which
/// nothing outside that crate's tests ever called. The consequence was
/// visible: `brain import-gguf` on a real DeepSeek-OCR file answered "has no
/// GGUF importer yet" while the importer sat two crates away. One table, so
/// that cannot recur.
struct Deepseek2OcrImporter;

impl GgufArchitectureImporter for Deepseek2OcrImporter {
    fn architecture(&self) -> &'static str {
        gguf::deepseek_ocr::GGUF_ARCHITECTURE
    }
    fn loads_directly(&self) -> bool {
        // `deepseek2ocr::import` opens both shipped GGUFs directly; this
        // architecture never needs a conversion to be served.
        true
    }
    fn summary(&self) -> &'static str {
        "DeepSeek-OCR decoder (DeepSeek-V2 MLA; the vision tower is its own mmproj file)"
    }
    fn import(&self, gguf: &MmapGguf, out_path: &str, id_override: Option<&str>) -> Result<(), String> {
        gguf::deepseek_ocr::import(gguf, out_path, id_override).map(|_| ())
    }
}

/// DeepSeek-OCR's vision half: a projector file, so it is claimed by
/// `general.architecture = "clip"` PLUS its own `clip.projector_type`.
struct DeepseekOcrVisionImporter;

impl GgufArchitectureImporter for DeepseekOcrVisionImporter {
    fn architecture(&self) -> &'static str {
        gguf::deepseek_ocr_vision::GGUF_ARCHITECTURE
    }
    fn projector(&self) -> Option<&'static str> {
        Some(gguf::deepseek_ocr_vision::PROJECTOR_TYPE)
    }
    fn loads_directly(&self) -> bool {
        true
    }
    fn summary(&self) -> &'static str {
        "DeepSeek-OCR vision tower (SAM + CLIP + projector), the mmproj half of the checkpoint"
    }
    fn import(&self, gguf: &MmapGguf, out_path: &str, id_override: Option<&str>) -> Result<(), String> {
        gguf::deepseek_ocr_vision::import(gguf, out_path, id_override).map(|_| ())
    }
}

/// Qwen3-VL: the language half. Its vision half is
/// [`Qwen3VlVisionImporter`] below, and neither is useful without the other.
struct Qwen3VlImporter;

impl GgufArchitectureImporter for Qwen3VlImporter {
    fn architecture(&self) -> &'static str {
        qwen3vl::gguf_import::GGUF_ARCHITECTURE
    }
    fn loads_directly(&self) -> bool {
        true
    }
    fn summary(&self) -> &'static str {
        "Qwen3-VL decoder (loaded directly with its mmproj vision tower; no conversion needed)"
    }
    fn import(&self, _gguf: &MmapGguf, _out: &str, _id: Option<&str>) -> Result<(), String> {
        Err(direct_only(qwen3vl::gguf_import::GGUF_ARCHITECTURE, "brain label images <dir> --weights <the language-half .gguf>"))
    }
}

/// Qwen3-VL's vision half (`clip` + `clip.projector_type = "qwen3vl_merger"`).
struct Qwen3VlVisionImporter;

impl GgufArchitectureImporter for Qwen3VlVisionImporter {
    fn architecture(&self) -> &'static str {
        gguf::deepseek_ocr_vision::GGUF_ARCHITECTURE
    }
    fn projector(&self) -> Option<&'static str> {
        Some(qwen3vl::gguf_import::PROJECTOR_TYPE)
    }
    fn loads_directly(&self) -> bool {
        true
    }
    fn summary(&self) -> &'static str {
        "Qwen3-VL vision tower (ViT + PatchMerger + DeepStack), the mmproj half of the checkpoint"
    }
    fn import(&self, _gguf: &MmapGguf, _out: &str, _id: Option<&str>) -> Result<(), String> {
        Err(direct_only(qwen3vl::gguf_import::GGUF_ARCHITECTURE, "brain label images <dir> --weights <the language-half .gguf>"))
    }
}

/// Qwen3-VL-30B-A3B (`general.architecture = "qwen3vlmoe"`) - the sparse-MoE
/// sibling of [`Qwen3VlImporter`]'s dense decoder, over a top-k-of-128 MoE
/// FFN with no shared expert (`crates/qwen3vlmoe`).
///
/// A REGISTRATION, not a working conversion, and the [`SupirImporter`]
/// pattern is followed deliberately: no `Qwen3-VL-30B-A3B` GGUF release was
/// available to inspect in this workspace, so there is no real tensor-name
/// mapping to derive or verify (a GGUF MoE decoder packs routed experts as 3D
/// `blk.N.ffn_*_exps.weight` tensors, llama.cpp's `LLM_TENSOR_FFN_*_EXPS`
/// convention - genuinely different from the dense per-layer 2D linears
/// `qwen3vl::gguf_import` already maps, so nothing there could be reused
/// blind). Registering the architecture NAME still matters on its own: before
/// this row existed, a `qwen3vlmoe` GGUF fell through to `by_gguf`'s `None`
/// and was refused with no name to act on at all (this model's own
/// user-facing documentation already described exactly that "refused by
/// name" behavior, which predates this crate). See `qwen3vlmoe::import`'s
/// own doc for the full account.
struct Qwen3VlMoeImporter;

impl GgufArchitectureImporter for Qwen3VlMoeImporter {
    fn architecture(&self) -> &'static str {
        qwen3vlmoe::import::GGUF_ARCHITECTURE
    }
    fn summary(&self) -> &'static str {
        "Qwen3-VL-30B-A3B decoder (sparse-MoE, top-8-of-128, no shared expert) - registered, not yet importable; no real GGUF release observed"
    }
    fn import(&self, gguf: &MmapGguf, out_path: &str, id_override: Option<&str>) -> Result<(), String> {
        qwen3vlmoe::import::import_gguf(gguf, out_path, id_override)
    }
}

/// The error for an architecture that is registered, and loads its GGUF
/// directly, but has no conversion to a brain-native checkpoint.
///
/// A dead end reported as "unsupported" would be wrong twice over: the file IS
/// supported, and the thing to do about it is a different command.
fn direct_only(arch: &str, how: &str) -> String {
    format!("architecture {arch:?} loads its GGUF directly at inference; there is no conversion to run. Use: {how}")
}

/// Every registered architecture importer. ONE line per architecture - this is
/// the whole registration surface (see this module's doc).
const IMPORTERS: &[&dyn GgufArchitectureImporter] = &[
    &Qwen3Importer,
    &Qwen35MoeImporter,
    &Qwen35Importer,
    &S3ditImporter,
    &WanImporter,
    &LtxvImporter,
    &SupirImporter,
    &Deepseek2OcrImporter,
    &DeepseekOcrVisionImporter,
    &Qwen3VlImporter,
    &Qwen3VlVisionImporter,
    &Qwen3VlMoeImporter,
];

/// The entry claiming `mg`, by the one architecture resolution
/// (`gguf::route`) every consumer shares.
///
/// A projector-discriminated entry wins over a bare architecture match, so a
/// file that says `clip` plus a `projector_type` reaches its own model rather
/// than whichever `clip` entry happens to be listed first.
pub fn importer_for_gguf(mg: &MmapGguf) -> Option<&'static dyn GgufArchitectureImporter> {
    let arch = architecture_of(mg);
    let projector = mg.kv().get(gguf::route::PROJECTOR_TYPE_KEY).and_then(|v| v.as_str());
    let mut fallback = None;
    for i in IMPORTERS.iter().copied().filter(|i| i.architecture() == arch) {
        match i.projector() {
            Some(want) => {
                if projector == Some(want) {
                    return Some(i);
                }
            }
            None => fallback = Some(i),
        }
    }
    fallback
}

/// The importer claiming `architecture` with no projector discriminator, or
/// `None`. For callers that hold only the architecture string (the model-dir
/// scan reads it off a `ModelCard`); a caller holding the open file should use
/// [`importer_for_gguf`], which can also resolve the projector case.
pub fn importer_for(architecture: &str) -> Option<&'static dyn GgufArchitectureImporter> {
    IMPORTERS.iter().copied().find(|i| i.architecture() == architecture && i.projector().is_none())
}

/// Every registered `general.architecture` value, in table order, for error
/// messages and `--list`.
///
/// Deduplicated, keeping the first occurrence: `clip` is claimed once per
/// projector-carrying model, and listing it N times says nothing extra. The
/// duplicates are not adjacent (each model's projector row sits beside its own
/// model row), so this cannot be `Vec::dedup`.
pub fn architectures() -> Vec<&'static str> {
    let mut seen: Vec<&'static str> = Vec::with_capacity(IMPORTERS.len());
    for a in IMPORTERS.iter().map(|i| i.architecture()) {
        if !seen.contains(&a) {
            seen.push(a);
        }
    }
    seen
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
    let importer = importer_for_gguf(&mg).ok_or_else(|| {
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
            let key = match i.projector() {
                Some(p) => format!("{}/{p}", i.architecture()),
                None => i.architecture().to_string(),
            };
            let how = if i.loads_directly() { "direct" } else { "convert" };
            println!("  {key:<22} [{how:>7}] {}", i.summary());
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

    /// A synthetic mmproj: the shared `clip` architecture plus the one key
    /// that says whose projector it is.
    fn write_projector_gguf(path: &str, projector: &str) {
        use checkpoint::gguf::GgufValue;
        use checkpoint::gguf_write::{write, TensorOut};
        let kvs = vec![
            ("general.architecture".to_string(), GgufValue::String("clip".to_string())),
            ("clip.projector_type".to_string(), GgufValue::String(projector.to_string())),
        ];
        let tensors = vec![TensorOut { name: "w".into(), shape: vec![1], ty: 0, data: 0f32.to_le_bytes().to_vec() }];
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
    /// `s3dit`'s spelling ("lumina2") and `supir`'s ("sdxl") are the two
    /// documented exceptions: each is shared with a real (or plausible)
    /// release under a DIFFERENT architecture's own name (see
    /// `s3dit::import::GGUF_ARCHITECTURE`'s and `supir::import::GGUF_ARCHITECTURE`'s
    /// own docs), so `brain_arch` deliberately does NOT claim either as that
    /// importer's `gguf:` spelling -- `s3dit::import::import_gguf` discriminates
    /// by tensor presence instead of by architecture string, and
    /// `supir::import::import_gguf` has nothing to discriminate yet (no real
    /// file has ever been observed). This test asserts that non-resolution is
    /// intentional, not a silent gap.
    #[test]
    fn every_registered_importer_matches_a_brain_arch_row_or_is_a_documented_ambiguous_tag() {
        const AMBIGUOUS_TAG_EXCEPTIONS: &[&str] = &["lumina2", "sdxl"];
        for i in IMPORTERS {
            let arch = i.architecture();
            if AMBIGUOUS_TAG_EXCEPTIONS.contains(&arch) {
                assert!(brain_arch::by_gguf(arch).is_none(), "importer {arch:?} is a documented ambiguous-tag exception and must NOT resolve via by_gguf");
                continue;
            }
            assert!(brain_arch::by_gguf(arch).is_some(), "importer {arch:?} has no matching brain_arch row");
        }
    }

    /// The `qwen3` tag dispatches AND converts, end to end through the
    /// generic command - so registering a second Qwen family really is the
    /// one line this module claims it is, and the resulting checkpoint carries
    /// a card the scan can dispatch on.
    #[test]
    fn the_qwen3_tag_dispatches_and_converts() {
        assert_eq!(importer_for("qwen3").map(|i| i.architecture()), Some("qwen3"));
        assert_eq!(brain_arch::by_gguf("qwen3").map(|a| a.id), Some("qwen3"));

        let dir = tmp("qwen3-dispatch");
        let src = dir.join("qwen3.gguf").to_string_lossy().into_owned();
        let out = dir.join("qwen3.brain.safetensors").to_string_lossy().into_owned();
        qwen3::gguf_import::testing::write_synthetic_gguf(&src, false);

        import_file(&src, Some(&out), Some("test/qwen3-tiny")).expect("the registry must convert a qwen3 GGUF");

        let card = checkpoint::st::read_card(&out).unwrap().expect("a converted checkpoint carries a card");
        assert_eq!(card.family, "qwen", "model_dir::resident_for dispatches on this exact string");
        assert_eq!(card.id, "test/qwen3-tiny");

        // ...and the output really is a loadable brain checkpoint, not just a
        // file that was written.
        let cfg = qwen3::QwenConfig::from_json(&checkpoint::read_config(&out));
        let r = checkpoint::weightio::WeightReader::open(&out).unwrap();
        for (name, numel) in cfg.param_list() {
            assert_eq!(r.tensor(&name).unwrap_or_else(|| panic!("missing {name}")).len(), numel, "{name}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_architecture_list_names_each_tag_once() {
        let list = architectures();
        let mut sorted = list.clone();
        sorted.sort_unstable();
        let n = sorted.len();
        sorted.dedup();
        assert_eq!(sorted.len(), n, "an architecture is listed twice: {list:?}");
        assert!(list.contains(&"clip"), "the shared projector tag is still listed once: {list:?}");
    }

    #[test]
    fn a_named_but_unimported_architecture_gets_a_more_specific_error() {
        let dir = tmp("named-unimported");
        // `t5encoder` is a real brain_arch row with no GgufArchitectureImporter
        // -- the error should say so by name, not just "no importer
        // registered". (This example was `deepseek2-ocr` until that
        // architecture's importer, which had been sitting in a second table in
        // `crates/gguf` that nothing called, was merged into IMPORTERS. It now
        // imports, so it can no longer stand for "named but unimported".)
        let src = dir.join("named.gguf").to_string_lossy().into_owned();
        write_gguf(&src, "t5encoder");
        let err = import_file(&src, None, None).unwrap_err();
        assert!(err.contains("t5encoder"), "must name the known brain_arch id: {err}");
        assert!(err.contains("no GGUF importer"), "must say it is a known-but-unimported gap: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The merge, asserted directly: DeepSeek-OCR's two halves are reachable
    /// through the ONE table, and the projector half is told apart from every
    /// other model's `clip` file by its projector_type.
    #[test]
    fn the_deepseek_ocr_halves_route_through_the_one_table() {
        assert!(importer_for("deepseek2-ocr").is_some(), "the decoder half must be registered here, not in a second table");

        let dir = tmp("mmproj-discriminator");
        for (projector, want) in [(gguf::deepseek_ocr_vision::PROJECTOR_TYPE, "DeepSeek-OCR vision"), (qwen3vl::gguf_import::PROJECTOR_TYPE, "Qwen3-VL vision")] {
            let src = dir.join(format!("mmproj-{projector}.gguf"));
            write_projector_gguf(src.to_str().unwrap(), projector);
            let mg = MmapGguf::open(src.to_str().unwrap()).unwrap();
            let i = importer_for_gguf(&mg).unwrap_or_else(|| panic!("{projector} must route"));
            assert_eq!(i.projector(), Some(projector));
            assert!(i.summary().starts_with(want), "{projector} routed to {:?}", i.summary());
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The fast-path opt-in is a real column, not decoration: the
    /// architectures whose own runtime streams a GGUF say so, and a caller
    /// holding a `.gguf` path reads that from the one table.
    #[test]
    fn architectures_with_a_direct_loader_declare_it() {
        for arch in ["qwen3", "wan", "ltxv", "qwen3vl", "deepseek2-ocr"] {
            assert!(importer_for(arch).is_some_and(|i| i.loads_directly()), "{arch} loads its GGUF directly and must declare it");
        }
        // qwen35moe genuinely has no direct loader: its real checkpoint is
        // ~140 GB at fp32, and the streaming import is the whole point.
        assert!(importer_for("qwen35moe").is_some_and(|i| !i.loads_directly()));
    }

    /// A direct-loading architecture with no conversion says what to run
    /// instead of reporting the file as unsupported.
    #[test]
    fn a_direct_only_architecture_names_the_command_to_use() {
        let dir = tmp("direct-only");
        let src = dir.join("qwen3vl.gguf").to_string_lossy().into_owned();
        write_gguf(&src, qwen3vl::gguf_import::GGUF_ARCHITECTURE);
        let err = import_file(&src, None, None).unwrap_err();
        assert!(err.contains("loads its GGUF directly"), "{err}");
        assert!(err.contains("brain label images"), "must name the command that does work: {err}");
        assert!(!std::path::Path::new(&default_out_path(&src)).exists());
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

    /// The dense sibling's own dispatch check, mirroring the qwen35moe test
    /// above: `general.architecture = "qwen35"` reaches `Qwen35Importer`, not
    /// the qwen35moe one two entries above it in the table.
    #[test]
    fn a_qwen35_dense_architecture_dispatches_to_its_own_importer() {
        let dir = tmp("dispatch-qwen35");
        let src = dir.join("toy.gguf").to_string_lossy().into_owned();
        write_gguf(&src, qwen35::gguf_import::GGUF_ARCHITECTURE);
        let err = import_file(&src, None, None).unwrap_err();
        assert!(err.contains("qwen35"), "must reach the qwen35 importer, got: {err}");
        assert!(!err.contains("no importer registered"), "must not fall through to the unknown-architecture error: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// End to end for the dense sibling, including the MTP head and the
    /// routing-collision fix in `model_dir::resident_for`: a raw
    /// `general.architecture = "qwen35"` file converts through the registry,
    /// and the CONVERTED (now brain-native, non-`.gguf`) checkpoint is what
    /// `discover` serves - the raw `.gguf` itself is never routed to the
    /// `qwen35` family resident, which cannot open it (see
    /// `model_dir::resident_for`'s GGUF gate).
    #[test]
    fn dispatch_converts_qwen35_dense_and_the_scan_then_serves_the_conversion() {
        let dir = tmp("e2e-qwen35");
        let src = dir.join("toy.gguf").to_string_lossy().into_owned();
        qwen35::gguf_import::testing::write_synthetic_gguf(&src);
        std::fs::write(dir.join("tokenizer.json"), br#"{"model":{"vocab":{"a":0}}}"#).unwrap();

        let out = import_file(&src, None, Some("test/qwen35-dense-tiny")).expect("registry dispatch must convert, MTP included");
        assert_eq!(out, default_out_path(&src), "default output is a sibling .brain.safetensors");

        let reader = checkpoint::weightio::WeightReader::open(&out).unwrap();
        assert!(reader.tensor("tok.weight").is_some(), "the importer really ran");
        assert!(reader.tensor("mtp.fc_e.weight").is_some(), "the MTP head must be imported, unlike qwen35moe's own GGUF route");
        assert_eq!(reader.card().expect("a model card must be written").id, "test/qwen35-dense-tiny");

        let ids: Vec<String> = crate::model_dir::discover(&dir).iter().map(|r| r.manifest().model).collect();
        assert!(ids.contains(&"test/qwen35-dense-tiny".to_string()), "the imported checkpoint must be discovered: {ids:?}");
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
