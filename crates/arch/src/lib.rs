// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The canonical model-architecture registry.
//!
//! brain used to have **four separate, drifting** answers to "which
//! architecture is this": the CLI's hand-written subcommand names, the
//! substring-matching HF-class-name scan in `modelstore::plan`, the GGUF
//! importer table in `cli::gguf_import`, and `ModelCard::family`. All four
//! read this crate's [`ARCHS`] table instead.
//!
//! ## The naming rule
//!
//! Every architecture gets one canonical `id`, restricted to `[a-z0-9]+` (no
//! underscores, no hyphens):
//!
//! - **[`Source::LlamaCpp`]** - llama.cpp's `LLM_ARCH_*` vocabulary, the enum
//!   name lowercased with the `LLM_ARCH_` prefix dropped and underscores
//!   removed (`LLM_ARCH_GLM_DSA` -> `glmdsa`).
//! - **[`Source::Brain`]** - architectures llama.cpp has no entry for (vision,
//!   diffusion, 3D, forecasting, world models): named after the upstream
//!   paper/repo's own architecture name, same `[a-z0-9]+` restriction
//!   (`yolov8`, `zipdepth`, `sdxlunet`, `s3dit`).
//! - **[`Source::Toy`]** - brain's own architectures with no upstream
//!   reference to parity-check against (the sparse-MoE toy task, the PID
//!   control transformer, …). Named, registered, gradient-checked and
//!   benchmarked like any other architecture, but excluded from `brain caps`,
//!   `brain --help` and the docs model list - see `Domain::Toy`.
//!
//! `id` is simultaneously: the crate directory name under `crates/`, the
//! package name's suffix (`brain-<id>`), the CLI word (`brain <id> <verb>` /
//! `brain <verb> <id>`), `ModelCard.architecture`, the GGUF importer key
//! ([`Arch::gguf`], when it differs from `id`), the fetch-recipe family key,
//! and the model's own docs page filename.
//!
//! A repo-wide gate keeps all of those in sync, and porting a new
//! architecture starts by adding its row here, before any other code.

/// The broad kind of task an architecture performs - used to group
/// `brain caps` output and the README quick-start, and to decide what a
/// generic `infer` verb should even mean for a given architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Domain {
    Text,
    Multimodal,
    Audio,
    Vision,
    Image,
    ThreeD,
    Forecast,
    World,
    /// brain's own architecture, no upstream reference. Real (gradient-checked,
    /// benchmarked) but excluded from `brain caps`, `brain --help` and the
    /// docs model list - see the [`Source::Toy`] naming rule above.
    Toy,
}

/// Where an architecture's canonical name comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// llama.cpp's `LLM_ARCH_*` vocabulary (lowercased, prefix dropped).
    LlamaCpp,
    /// brain-defined: llama.cpp has no entry for this architecture's domain
    /// (vision, diffusion, 3D, forecasting, world models).
    Brain,
    /// brain's own architecture, no upstream reference - see [`Domain::Toy`].
    Toy,
}

/// One row: a single architecture brain supports (or is named for, whether or
/// not the port is complete - an entry here is a name reservation, not a
/// completeness claim; each architecture's own status is tracked separately).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arch {
    /// The canonical id. `[a-z0-9]+`, unique across [`ARCHS`].
    pub id: &'static str,
    /// Human-readable name for `--help` / docs / `brain caps` output.
    pub display: &'static str,
    pub domain: Domain,
    pub source: Source,
    /// The crate's package name, e.g. `"brain-qwen3"`. Must name a real
    /// workspace member once the crate-rename migration reaches each row -
    /// some rows still name their pre-rename package while that migration is
    /// in flight, one commit at a time, one domain group per commit.
    pub package: &'static str,
    /// GGUF `general.architecture` value, when it differs from `id` (e.g.
    /// `deepseek2ocr`'s is `"deepseek2-ocr"` - llama.cpp's own spelling keeps
    /// the hyphen brain's id grammar forbids). `None` means `id` itself is
    /// the GGUF spelling, or this architecture has no GGUF importer.
    pub gguf: Option<&'static str>,
    /// Exact HF `config.json` `architectures[0]` class name(s) this id
    /// covers, plus the real `model_type` slug when a config lacking
    /// `architectures` needs that fallback spelling too (documented per
    /// entry). Exact string match only - no substring/prefix matching, which
    /// is the defect this table replaces (a naive scan matching `"qwen"`
    /// before `"omni"` would route `Qwen3OmniMoeForConditionalGeneration` to
    /// the dense Qwen3 importer). Empty when no HF-checkpoint fetch path
    /// exists yet for this architecture.
    pub hf: &'static [&'static str],
    /// The `<vendor>/<repo>` this architecture auto-fetches by default when a
    /// verb needs weights and none were named explicitly (`brain infer
    /// zipdepth --in image=x.jpg` with no `--weights`). `None` when no small,
    /// generally-useful default checkpoint is known, or when auto-fetch for
    /// this architecture is not wired yet (`crates/cli/src/supply.rs`'s
    /// `ensure_default_weights` is what actually resolves this - a `Some`
    /// here is a claim that repo is real and fetchable, not that every verb
    /// honors it yet).
    pub default_ref: Option<&'static str>,
    /// `(env var, manifest role)` pairs this architecture's OWN weights
    /// resolution reads (`BRAIN_SAM2_WEIGHTS`, `BRAIN_S3DIT_DIT`, ...) -- the
    /// capability-path counterpart to [`default_ref`](Self::default_ref):
    /// architectures reached via `brain <arch> <action>`'s generic capability
    /// dispatch (never through the `--weights`-flag injection the
    /// `ARCH_HANDLERS` path gets) resolve weights from environment variables
    /// their own provider's `from_env` reads, one role each when
    /// `default_ref` fetches a single file/directory, several when a compound
    /// checkpoint has named parts (`s3dit`'s DiT/VAE/text-encoder/tokenizer
    /// four). `crates/cli/src/supply.rs::ensure_env_weights` is what actually
    /// resolves this: for each listed var that is UNSET, it fetches
    /// `default_ref`, reads its role back from the store (a single-file
    /// [`LocalModel::weights`] path for a `"weights"` role, else
    /// [`LocalModel::roles`]), and sets the var -- never overriding one
    /// already set. Empty when this architecture has no env-var-driven
    /// weights resolution (an `ARCH_HANDLERS` architecture using `--weights`
    /// directly, or one with no auto-fetch story yet).
    pub weights_env: &'static [(&'static str, &'static str)],
}

/// Every field [`arch!`] does not set explicitly, for its trailing
/// functional-update `..DEFAULT`.
const DEFAULT: Arch = Arch { id: "", display: "", domain: Domain::Toy, source: Source::Toy, package: "", gguf: None, hf: &[], default_ref: None, weights_env: &[] };

macro_rules! arch {
    ($id:expr, $display:expr, $domain:expr, $source:expr, $package:expr $(, $key:ident : $val:expr)* $(,)?) => {
        Arch { id: $id, display: $display, domain: $domain, source: $source, package: $package, $($key: $val,)* ..DEFAULT }
    };
}

use Domain::*;
use Source::*;

/// The registry. Order is presentation order for `brain caps` / `--help` -
/// roughly the grouping `AGENTS.md` already uses (text decoders, multimodal,
/// audio, vision, image generation, 3D, forecasting, world models, toy).
///
/// `package` names the crate as it exists TODAY, and every row now satisfies
/// the naming rule (`crates/<id>`, package `brain-<id>`): the crate-rename
/// migration is complete, the last pair to land being `scrfd`/`arcface`, which
/// were one bundled crate until they were split into `crates/scrfd` and
/// `crates/arcface`. Each rename updates its row in the same commit that moves
/// the crate.
pub const ARCHS: &[Arch] = &[
    // -- Text decoders --------------------------------------------------
    arch!("gpt2", "GPT-2 (nanoGPT parity baseline)", Text, LlamaCpp, "brain-gpt2", hf: &["GPT2LMHeadModel"]),
    // "qwen3" is the real config.json `model_type` fallback value (used when
    // `architectures[0]` is absent), alongside the real `architectures[0]`
    // class name.
    arch!("qwen3", "Qwen3 dense decoder", Text, LlamaCpp, "brain-qwen3", hf: &["Qwen3ForCausalLM", "qwen3"], default_ref: Some("Qwen/Qwen3-0.6B")),
    arch!("qwen35moe", "Qwen3.5-35B-A3B hybrid GDN/GQA MoE decoder", Text, LlamaCpp, "brain-qwen35moe", gguf: Some("qwen35moe")),
    arch!("glmdsa", "GLM-5.2 (glm_moe_dsa: MLA + sigmoid noaux_tc MoE + DSA)", Text, LlamaCpp, "brain-glmdsa"),
    arch!("deepseek2", "DeepSeek-V2-family MoE decoder", Text, LlamaCpp, "brain-deepseek2"),
    arch!("lfm2", "LiquidAI LFM2.5-Encoder", Text, LlamaCpp, "brain-lfm2", hf: &["Lfm2ForCausalLM"], default_ref: Some("LiquidAI/LFM2.5-350M")),
    // -- Multimodal (VLM / omni / ASR) -----------------------------------
    arch!("qwen3omnimoe", "Qwen3-Omni-30B-A3B (Thinker+Talker+Code2Wav)", Multimodal, Brain, "brain-qwen3omnimoe", hf: &["Qwen3OmniMoeForConditionalGeneration"]),
    arch!("qwen3vl", "Qwen3-VL-4B (ViT+PatchMerger+DeepStack)", Multimodal, LlamaCpp, "brain-qwen3vl", hf: &["Qwen3VLForConditionalGeneration"], default_ref: Some("Qwen/Qwen3-VL-4B-Instruct"), weights_env: &[("BRAIN_QWEN3VL_WEIGHTS", "weights")]),
    arch!("fastvlm", "Apple FastVLM (FastViTHD + Qwen2 decoder)", Multimodal, Brain, "brain-fastvlm", hf: &["LlavaQwen2ForCausalLM"], default_ref: Some("apple/FastVLM-0.5B"), weights_env: &[("BRAIN_FASTVLM_WEIGHTS", "weights")]),
    arch!("moondream3", "Moondream 3 (SigLIP + MoE decoder)", Multimodal, Brain, "brain-moondream3"),
    // `default_ref` names the GGUF release repo (`ggml-org/DeepSeek-OCR-GGUF`),
    // not `deepseek-ai/DeepSeek-OCR` -- the latter is a `transformers`-shaped
    // repo with an empty `hf:` list here, so it would fall through to
    // `TransformersRecipe` and fail `UnsupportedArchitecture`, and this is
    // the checkpoint `crates/deepseek2ocr` actually loads (`BRAIN_DEEPSEEK_OCR_DIR`
    // wants the two-GGUF pair, not an HF safetensors dir). See
    // `crates/modelstore/src/recipe.rs`'s `deepseek2ocr-gguf` `FilesRecipe` row.
    arch!("deepseek2ocr", "DeepSeek-OCR (SAM+CLIP DeepEncoder + DeepSeek-V2 decoder)", Multimodal, LlamaCpp, "brain-deepseek2ocr", gguf: Some("deepseek2-ocr"), default_ref: Some("ggml-org/DeepSeek-OCR-GGUF"), weights_env: &[("BRAIN_DEEPSEEK_OCR_DIR", "dir")]),
    arch!("qwen3asr", "Qwen3-ASR-1.7B (Whisper-style encoder + Qwen3 decoder)", Audio, Brain, "brain-qwen3asr", hf: &["Qwen3ASRForConditionalGeneration"], default_ref: Some("Qwen/Qwen3-ASR-1.7B"), weights_env: &[("BRAIN_QWEN3ASR", "weights")]),
    arch!("nemotronasr", "Nemotron-3.5-ASR-Streaming (FastConformer + RNN-T)", Audio, Brain, "brain-nemotronasr", hf: &["Nemotron3_5AsrForRNNT"], default_ref: Some("nvidia/nemotron-3.5-asr-streaming-0.6b"), weights_env: &[("BRAIN_NEMOTRONASR", "weights")]),
    // -- Audio / TTS ------------------------------------------------------
    arch!("qwen3tts", "Qwen3-TTS (Talker + MTP code predictor)", Audio, LlamaCpp, "brain-qwen3tts", hf: &["Qwen3TTSForConditionalGeneration"], default_ref: Some("Qwen/Qwen3-TTS-12Hz-0.6B-Base"), weights_env: &[("BRAIN_QWEN3TTS_WEIGHTS", "weights_dir"), ("BRAIN_QWEN3TTS_CKPT", "ckpt")]),
    arch!("mimi", "Mimi/Moshi-style 12 Hz neural audio codec", Audio, Brain, "brain-mimi"),
    arch!("ecapatdnn", "ECAPA-TDNN speaker encoder", Audio, Brain, "brain-ecapatdnn"),
    // -- Vision: detection / segmentation / face / depth -------------------
    arch!("yolov8", "YOLOv8-style anchor-free detector", Vision, Brain, "brain-yolov8", default_ref: Some("Ultralytics/YOLOv8")),
    arch!("sam1", "SAM-1 / ViTDet ViT-B tower", Vision, Brain, "brain-sam1"),
    arch!("sam2", "SAM 2.1 promptable segmentation (image path)", Vision, Brain, "brain-sam2", default_ref: Some("facebook/sam2.1-hiera-tiny"), weights_env: &[("BRAIN_SAM2_WEIGHTS", "weights")]),
    arch!("scrfd", "SCRFD face detector", Vision, Brain, "brain-scrfd"),
    arch!("arcface", "ArcFace IResNet-100 face embedding", Vision, Brain, "brain-arcface"),
    arch!("clip", "CLIP-L / OpenCLIP-bigG / EVA-CLIP text+image towers", Vision, LlamaCpp, "brain-clip"),
    arch!("zipdepth", "ZipDepth monocular depth (pure-conv)", Vision, Brain, "brain-zipdepth"),
    // -- Image generation / restoration --------------------------------
    arch!("s3dit", "Z-Image S3-DiT text-to-image", Image, Brain, "brain-s3dit", default_ref: Some("Tongyi-MAI/Z-Image-Turbo"), weights_env: &[("BRAIN_S3DIT_DIT", "dit"), ("BRAIN_S3DIT_VAE", "vae"), ("BRAIN_S3DIT_QWEN", "text_encoder"), ("BRAIN_S3DIT_TOKENIZER", "tokenizer")]),
    arch!("flux2", "FLUX.2 Klein MMDiT text-to-image + editing", Image, Brain, "brain-flux2"),
    arch!("flux1", "FLUX.1 dev / Kontext / schnell MMDiT", Image, Brain, "brain-flux1"),
    arch!("t5encoder", "T5-XXL encoder (FLUX.1 text conditioning)", Text, LlamaCpp, "brain-t5encoder"),
    arch!("sdxlunet", "SDXL UNet2DConditionModel", Image, Brain, "brain-sdxlunet"),
    arch!("controlnet", "ControlNet (backbone-agnostic seam + SDXL producer)", Image, Brain, "brain-controlnet"),
    arch!("pulid", "PuLID-FLUX identity conditioning", Image, Brain, "brain-pulid"),
    arch!("instantid", "InstantID (SDXL + IP-Adapter-FaceID)", Image, Brain, "brain-instantid"),
    arch!("autoencoderkl", "diffusers AutoencoderKL (Z-Image/FLUX.2/SDXL VAE)", Image, Brain, "brain-vae"),
    arch!("vqgan", "VQGAN / CodeFormer VQ autoencoder", Image, Brain, "brain-vqgan"),
    arch!("codeformer", "CodeFormer blind face restoration", Image, Brain, "brain-codeformer"),
    arch!("rrdbnet", "Real-ESRGAN RRDBNet super-resolution", Image, Brain, "brain-rrdbnet", default_ref: Some("schwgHao/RealESRGAN_x4plus"), weights_env: &[("BRAIN_ESRGAN_WEIGHTS", "weights")]),
    // -- 3D -----------------------------------------------------------
    arch!("worldmirror2", "WorldMirror-2 multi-view 3D reconstruction", ThreeD, Brain, "brain-worldmirror2"),
    arch!("splat", "3D Gaussian Splatting rasterizer", ThreeD, Brain, "brain-splat"),
    // -- Forecasting ----------------------------------------------------
    arch!("chronos2", "Chronos-2 encoder-only patch transformer", Forecast, Brain, "brain-chronos2"),
    arch!("kronos", "Kronos BSQ-tokenizer candlestick model", Forecast, Brain, "brain-kronos"),
    arch!("fincast", "FinCast patched decoder + sparse MoE", Forecast, Brain, "brain-fincast"),
    // -- World models ---------------------------------------------------
    arch!("diamond", "DIAMOND EDM diffusion world model", World, Brain, "brain-diamond"),
    arch!("genieredux", "GenieRedux-G ST-transformer world model", World, Brain, "brain-genieredux"),
    // -- Toy (brain's own, no upstream reference) ------------------------
    arch!("toymoe", "Sparse-MoE toy task (64-symbol next-token rule)", Domain::Toy, Source::Toy, "brain-toymoe"),
    arch!("toypid", "PID event/effect control transformer", Domain::Toy, Source::Toy, "brain-toypid"),
    arch!("toyseq2seq", "Encoder-decoder toy task", Domain::Toy, Source::Toy, "brain-toyseq2seq"),
    arch!("toyautoencoder", "Bottleneck autoencoder toy task", Domain::Toy, Source::Toy, "brain-toyautoencoder"),
];

/// The [`Arch`] with this canonical `id`, or `None`.
pub fn by_id(id: &str) -> Option<&'static Arch> {
    ARCHS.iter().find(|a| a.id == id)
}

/// The [`Arch`] whose [`Arch::hf`] list contains this EXACT HF
/// `architectures[0]` class name, or `None`. Exact match only - see
/// [`Arch::hf`]'s doc for why a substring scan is the defect this replaces.
pub fn by_hf(class_name: &str) -> Option<&'static Arch> {
    ARCHS.iter().find(|a| a.hf.contains(&class_name))
}

/// The [`Arch`] whose GGUF `general.architecture` spelling is `architecture`.
/// Checks [`Arch::gguf`] first, then falls back to `id` for architectures
/// whose GGUF spelling equals their canonical id.
pub fn by_gguf(architecture: &str) -> Option<&'static Arch> {
    ARCHS.iter().find(|a| a.gguf.unwrap_or(a.id) == architecture)
}

/// Every non-toy architecture, in registry order - what `brain caps`,
/// `brain --help` and the docs model list show.
pub fn public() -> impl Iterator<Item = &'static Arch> {
    ARCHS.iter().filter(|a| a.domain != Domain::Toy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn every_default_ref_parses_as_a_non_reserved_two_segment_ref() {
        // A default_ref names a REAL upstream repo to auto-fetch, never a
        // reserved brain/local/test vendor (those are never fetched) and
        // never a stray third path segment.
        for a in ARCHS {
            let Some(r) = a.default_ref else { continue };
            let (vendor, repo) = r.split_once('/').unwrap_or_else(|| panic!("{:?}: default_ref {r:?} has no '/'", a.id));
            assert!(!repo.contains('/'), "{:?}: default_ref {r:?} has more than one '/'", a.id);
            assert!(
                !matches!(vendor, "brain" | "local" | "test"),
                "{:?}: default_ref {r:?} names a reserved vendor -- reserved vendors are never fetched",
                a.id
            );
        }
    }

    #[test]
    fn ids_are_lowercase_alphanumeric_only() {
        for a in ARCHS {
            assert!(!a.id.is_empty(), "empty id");
            assert!(a.id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit()), "{:?}: id must be [a-z0-9]+, no `-`/`_`", a.id);
        }
    }

    #[test]
    fn ids_are_unique() {
        let mut seen = HashSet::new();
        for a in ARCHS {
            assert!(seen.insert(a.id), "duplicate id {:?}", a.id);
        }
    }

    #[test]
    fn hf_class_names_are_unique_across_archs() {
        let mut seen = HashSet::new();
        for a in ARCHS {
            for hf in a.hf {
                assert!(seen.insert(*hf), "{:?}: hf class name {:?} claimed by more than one arch", a.id, hf);
            }
        }
    }

    #[test]
    fn gguf_spellings_are_unique_across_archs() {
        let mut seen = HashSet::new();
        for a in ARCHS {
            let spelling = a.gguf.unwrap_or(a.id);
            assert!(seen.insert(spelling), "{:?}: gguf spelling {:?} claimed by more than one arch", a.id, spelling);
        }
    }

    #[test]
    fn package_names_are_nonempty_and_prefixed() {
        for a in ARCHS {
            assert!(a.package.starts_with("brain-"), "{:?}: package {:?} should start with brain-", a.id, a.package);
        }
    }

    #[test]
    fn by_id_finds_a_known_row_and_none_for_unknown() {
        assert_eq!(by_id("qwen3").map(|a| a.id), Some("qwen3"));
        assert_eq!(by_id("totally-unknown"), None);
    }

    #[test]
    fn by_hf_omni_does_not_fall_through_to_dense_qwen3() {
        // The real defect this table replaces: a substring scan checking
        // "qwen" before "omni" would route Qwen3-Omni's real HF class name
        // (which CONTAINS "qwen" as a substring) to the dense qwen3 importer.
        // Exact match makes that class of bug structurally impossible.
        assert_eq!(by_hf("Qwen3OmniMoeForConditionalGeneration").map(|a| a.id), Some("qwen3omnimoe"));
        assert_eq!(by_hf("Qwen3ForCausalLM").map(|a| a.id), Some("qwen3"));
        assert_eq!(by_hf("totally-unknown"), None);
    }

    #[test]
    fn by_gguf_falls_back_to_id_when_no_explicit_spelling() {
        assert_eq!(by_gguf("qwen35moe").map(|a| a.id), Some("qwen35moe"));
        assert_eq!(by_gguf("deepseek2-ocr").map(|a| a.id), Some("deepseek2ocr"));
        assert_eq!(by_gguf("deepseek2ocr"), None); // the id itself is NOT the gguf spelling here
    }

    #[test]
    fn public_excludes_only_toy_domain() {
        let toy_ids: HashSet<&str> = ARCHS.iter().filter(|a| a.domain == Domain::Toy).map(|a| a.id).collect();
        assert_eq!(toy_ids, HashSet::from(["toymoe", "toypid", "toyseq2seq", "toyautoencoder"]));
        for a in public() {
            assert!(!toy_ids.contains(a.id));
        }
        assert_eq!(public().count() + toy_ids.len(), ARCHS.len());
    }
}
