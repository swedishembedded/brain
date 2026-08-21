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
//!   diffusion, video, 3D, forecasting, world models): named after the upstream
//!   paper/repo's own architecture name, same `[a-z0-9]+` restriction
//!   (`yolov8`, `zipdepth`, `sdxlunet`, `s3dit`). When the upstream family
//!   spans several releases under one architecture name, the id names the
//!   FAMILY, not the release (`wan` covers Wan2.1 and Wan2.2, as `qwen3`
//!   covers 0.6B through 32B) - the release is a config, and a per-release id
//!   would collide on the shared [`Arch::gguf`] spelling.
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
    /// Generates video (a temporal sequence of frames) rather than a single
    /// image - a separate domain because the generic `infer` verb has to mean
    /// something different here (frame count, fps and a video container are
    /// part of the request, not optional extras) and because the serving path
    /// carries `capability::Media::Video` blobs rather than `Image` ones.
    Video,
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
    /// Further `<vendor>/<repo>` checkpoints this architecture needs beyond
    /// [`default_ref`], for a model upstream publishes as SEVERAL repos.
    ///
    /// brain's fetch plan is one `ModelRef` -> one repo listing -> one
    /// `Plan`, and that stays true: this is not a second repo inside one plan,
    /// it is a second plan. `crates/cli/src/supply.rs::ensure_env_weights`
    /// fetches `default_ref` and then each of these in turn, and merges the
    /// roles of every resulting manifest into one map before resolving
    /// [`weights_env`](Self::weights_env) against it -- so role names must be
    /// DISJOINT across the set (a repo-wide test in this file enforces that).
    ///
    /// `wan` did not need this: choosing the native repo over `-Diffusers`
    /// found one listing carrying all four roles. `kronos` has no such
    /// option - upstream ships the BSQ tokenizer
    /// (`NeoQuasar/Kronos-Tokenizer-base`) and the decoder
    /// (`NeoQuasar/Kronos-base`) as two repos with no combined release - so
    /// the honest answer is to say so here rather than to teach the planner
    /// about compound fetches it would need for exactly one model.
    pub extra_refs: &'static [&'static str],
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
const DEFAULT: Arch =
    Arch { id: "", display: "", domain: Domain::Toy, source: Source::Toy, package: "", gguf: None, hf: &[], default_ref: None, extra_refs: &[], weights_env: &[] };

macro_rules! arch {
    ($id:expr, $display:expr, $domain:expr, $source:expr, $package:expr $(, $key:ident : $val:expr)* $(,)?) => {
        Arch { id: $id, display: $display, domain: $domain, source: $source, package: $package, $($key: $val,)* ..DEFAULT }
    };
}

use Domain::*;
use Source::*;

/// The registry. Order is presentation order for `brain caps` / `--help` -
/// roughly the grouping `AGENTS.md` already uses (text decoders, multimodal,
/// audio, vision, image generation, video generation, 3D, forecasting, world
/// models, toy).
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
    // The DENSE sibling of qwen35moe - llama.cpp registers the two as
    // separate architectures (`LLM_ARCH_QWEN35` vs `LLM_ARCH_QWEN35MOE`)
    // despite sharing one HF `model_type` ("qwen3_5"): same hybrid Gated
    // DeltaNet / gated-GQA mixer split, but a plain dense SwiGLU MLP instead
    // of qwen35moe's 256-expert MoE, plus a single-layer MTP head and a
    // spliced Qwen3-VL-style vision tower (no DeepStack) - hence `Multimodal`
    // here even though llama.cpp's own arch classifies it as text-only (its
    // GGUF conversion path drops the vision tower). `hf` carries both the
    // real `architectures[0]` class and the `model_type` fallback spelling,
    // same convention as `qwen3`'s row - a config lacking `architectures`
    // still resolves via `model_type: "qwen3_5"`.
    arch!("qwen35", "Qwen3.5/3.8-27B dense hybrid GDN/GQA decoder + MTP + ViT", Multimodal, LlamaCpp, "brain-qwen35", gguf: Some("qwen35"), hf: &["Qwen3_5ForConditionalGeneration", "qwen3_5"], default_ref: Some("Qwen/Qwen3.8-27B-FP8"), weights_env: &[("BRAIN_QWEN35_DIR", "dir")]),
    arch!("glmdsa", "GLM-5.2 (glm_moe_dsa: MLA + sigmoid noaux_tc MoE + DSA)", Text, LlamaCpp, "brain-glmdsa"),
    arch!("deepseek2", "DeepSeek-V2-family MoE decoder", Text, LlamaCpp, "brain-deepseek2"),
    arch!("lfm2", "LiquidAI LFM2.5-Encoder", Text, LlamaCpp, "brain-lfm2", hf: &["Lfm2ForCausalLM"], default_ref: Some("LiquidAI/LFM2.5-350M")),
    // The LTX-2.5 text encoder: Gemma4Unified's text tower (12B, 26 GB bf16 -
    // real-weight import out of scope until a machine that can hold it; see
    // `crates/gemma4`'s own doc). `transformers.models.gemma4_unified` is
    // itself dated 2026 in its own license header (a very recent addition, no
    // local llama.cpp checkout in this repo to range-check against), so this
    // is a brain-defined id per the naming rule's "no entry there yet" branch,
    // not a lowercased `LLM_ARCH_*` spelling - re-verify against a real
    // llama.cpp checkout before this assumption is load-bearing anywhere.
    // `hf` names the real gated checkpoint's class; no `default_ref`/
    // `weights_env` yet - this milestone never fetches or imports the real
    // checkpoint.
    arch!("gemma4", "Gemma-4 unified text tower (LTX-2.5's text encoder)", Text, Brain, "brain-gemma4", hf: &["Gemma4UnifiedForConditionalGeneration"]),
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
    arch!("campplus", "CAM++ D-TDNN speaker encoder (192-d x-vector)", Audio, Brain, "brain-campplus"),
    // No official llama.cpp/GGUF architecture entry exists for either
    // component; upstream ships the ONNX-only `speech_tokenizer_v2.onnx`
    // (CosyVoice 2) / `speech_tokenizer_v3.onnx` (CosyVoice 3). No single
    // `default_ref` names a whole repo carrying just this one file, so
    // `weights_env` is the only resolution path for now, one role per
    // codebook version - both variants share one 6561-entry FSQ codebook.
    arch!("s3tokenizer", "S3Tokenizer FSQ supervised-semantic speech tokenizer", Audio, Brain, "brain-s3tokenizer",
          weights_env: &[("BRAIN_S3TOKENIZER_V2", "v2"), ("BRAIN_S3TOKENIZER_V3", "v3")]),
    // The id names the FAMILY, not the release - CosyVoice 2 and CosyVoice 3
    // share one upstream product name and one LM backbone (a stock
    // Qwen2.5-0.5B), differing only in the flow decoder's estimator (UNet vs
    // DiT) and small vocoder causality deltas; the release is a config
    // (`Variant::CosyVoice2`/`CosyVoice3`), exactly as `wan` spans 2.1/2.2.
    // No official llama.cpp/GGUF architecture entry exists upstream, hence
    // `gguf: None`. `weights_env` names one role per component since
    // upstream ships `llm.pt`/`flow.pt`/`hift.pt` as three independent files
    // under one repo, not one combined checkpoint; `s3tokenizer` and
    // `campplus` are separate rows above, not roles here, because they are
    // independently useful architectures, not CosyVoice internals (the same
    // split `qwen3tts`/`mimi`/`ecapatdnn` already use).
    //
    // `default_ref` names ONLY the CosyVoice 2 repo, deliberately - not
    // `extra_refs`: CosyVoice 3's repo carries the SAME three role names
    // (`llm`/`flow`/`hift`), not additional ones, so it is a second variant
    // of one role set, not a compound checkpoint `extra_refs` merges roles
    // from (see `kronos`'s row for that shape). Fetching CosyVoice 3 means
    // pointing the `weights_env` vars at it explicitly, same as any other
    // non-default `wan`/`flux2` variant.
    arch!("cosyvoice", "CosyVoice 2/3 (LLM-based streaming zero-shot TTS)", Audio, Brain, "brain-cosyvoice",
          default_ref: Some("FunAudioLLM/CosyVoice2-0.5B"),
          weights_env: &[("BRAIN_COSYVOICE_LLM", "llm"),
                         ("BRAIN_COSYVOICE_FLOW", "flow"),
                         ("BRAIN_COSYVOICE_HIFT", "hift")]),
    // Five chained components, no single upstream checkpoint file: a real
    // Qwen3-8B "Global LLM" (`qwen_7B/qwen_7B/`, llama.cpp's own `qwen3`
    // architecture - reused via `crates/qwen3`, not reimplemented here), a
    // 4-layer causal "RVQ depth decoder", a small conv "condition encoder", a
    // 36-layer flow-matching DiT, and a DAC-style vocoder. No official GGUF
    // release exists upstream, hence `gguf: None`. `weights_env` names one
    // role per component since upstream ships them as five independent
    // checkpoint dirs under one repo, not one combined file.
    arch!("minimaxmusic3", "MiniMax Music 3 (lyrics+caption-conditioned music generation)", Audio, Brain, "brain-minimaxmusic3",
          hf: &["MiniMaxMusic3ForConditionalGeneration"],
          default_ref: Some("MiniMaxAI/MiniMax-Music3"),
          weights_env: &[("BRAIN_MINIMAXMUSIC3_LM", "language_model"),
                         ("BRAIN_MINIMAXMUSIC3_DEPTH", "depth_decoder"),
                         ("BRAIN_MINIMAXMUSIC3_CONDITION", "condition_encoder"),
                         ("BRAIN_MINIMAXMUSIC3_DIT", "transformer"),
                         ("BRAIN_MINIMAXMUSIC3_VOCODER", "vocoder"),
                         ("BRAIN_MINIMAXMUSIC3_TOKENIZER", "tokenizer")]),
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
    // `black-forest-labs/FLUX.2-klein-4B` ships the same shape
    // `ZimageRecipe` already matches generically (`model_index.json` +
    // transformer/vae/text_encoder/tokenizer role dirs) and even reuses
    // Z-Image's exact `vae/diffusion_pytorch_model.safetensors` filename, so
    // no new recipe is needed - confirmed against the real repo listing.
    arch!("flux2", "FLUX.2 Klein MMDiT text-to-image + editing", Image, Brain, "brain-flux2",
          default_ref: Some("black-forest-labs/FLUX.2-klein-4B"),
          weights_env: &[("BRAIN_FLUX2_DIT", "dit"), ("BRAIN_FLUX2_VAE", "vae"), ("BRAIN_FLUX2_TE", "text_encoder"), ("BRAIN_FLUX2_TOKENIZER", "tokenizer")]),
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
    // -- Video generation -----------------------------------------------
    // `wan` names the FAMILY, not the release: Wan2.1 and Wan2.2 share one
    // upstream architecture name, one HF class (`WanTransformer3DModel`) and
    // one GGUF `general.architecture` spelling (`"wan"`, ComfyUI-GGUF's
    // `IMG_ARCH_LIST`), so a per-release `wan21` id would have to claim that
    // shared spelling via `gguf:` and then collide with `wan22` under
    // `gguf_spellings_are_unique_across_archs`. The release is a config
    // (`WanConfig::{t2v_1_3b, t2v_14b, i2v_14b_480p, i2v_14b_720p}`), exactly
    // as `qwen3` spans 0.6B..32B. `gguf` stays `None` because the id IS the
    // spelling.
    //
    // `default_ref` names the NATIVE repo rather than `-Diffusers`: it is
    // self-contained (all four T2V roles in one listing, which is all brain's
    // one-ModelRef-per-plan fetch can express), and it ships the umT5-XXL
    // encoder as a single bf16 `.pth` (11.4 GB) where `-Diffusers` shards it
    // in fp32 (22.7 GB) -- 17.6 GB total against 28.9 GB for the same model.
    // `-Diffusers` remains the tensor-NAMING authority that `import_diffusers`
    // targets; it is just not what we make users download by default.
    //
    // `BRAIN_WAN_CLIP` (the I2V CLIP ViT-H/14 vision tower) is deliberately
    // NOT in `weights_env`: it does not exist in the T2V repo, and I2V's own
    // checkpoint is a different `default_ref` -- so it stays an explicitly-set
    // variable rather than something auto-fetch would fail to resolve.
    arch!("wan", "Wan2.1/2.2 video diffusion transformer (T2V/I2V)", Video, Brain, "brain-wan",
          hf: &["WanTransformer3DModel"],
          default_ref: Some("Wan-AI/Wan2.1-T2V-1.3B"),
          weights_env: &[("BRAIN_WAN_DIT", "dit"), ("BRAIN_WAN_VAE", "vae"),
                         ("BRAIN_WAN_T5", "text_encoder"), ("BRAIN_WAN_TOKENIZER", "tokenizer")]),
    // `id` IS the GGUF spelling: `general.architecture = "ltxv"` on every
    // LTX-2.x GGUF observed (confirmed by range-reading the header of both
    // `unsloth/LTX-2.3-GGUF` and `city96/LTX-Video-gguf`), so `gguf: None`.
    // `default_ref` names the split, Comfy-aligned HF repo directly -- brain's
    // `weights_env` resolution reads one role per file, which is exactly how
    // that repo is laid out (no single-file/monolith option exists upstream).
    // `BRAIN_LTXV_AUDIO_VAE` is a role separate from `_VAE` (video) because
    // upstream ships them as two independent checkpoints with unrelated
    // shapes, unlike Wan's single VAE role. The vocoder ships bundled inside
    // the audio-VAE checkpoint's own metadata, not as a fourth file.
    arch!("ltxv", "LTX-2.5 two-stream audio+video diffusion transformer", Video, Brain, "brain-ltxv",
          hf: &["AVTransformer3DModel"],
          default_ref: Some("Lightricks/LTX-2.5"),
          weights_env: &[("BRAIN_LTXV_DIT", "dit"), ("BRAIN_LTXV_VAE", "vae"),
                         ("BRAIN_LTXV_AUDIO_VAE", "audio_vae"),
                         ("BRAIN_LTXV_TEXT_ENCODER", "text_encoder"),
                         ("BRAIN_LTXV_TOKENIZER", "tokenizer")]),
    // -- 3D -----------------------------------------------------------
    arch!("worldmirror2", "WorldMirror-2 multi-view 3D reconstruction", ThreeD, Brain, "brain-worldmirror2"),
    arch!("splat", "3D Gaussian Splatting rasterizer", ThreeD, Brain, "brain-splat"),
    // -- Forecasting ----------------------------------------------------
    arch!("chronos2", "Chronos-2 encoder-only patch transformer", Forecast, Brain, "brain-chronos2"),
    // Two repos, one model: `Kronos-base` is the decoder and
    // `Kronos-Tokenizer-base` is the BSQ tokenizer, and upstream publishes no
    // combined release - see `extra_refs`. `-base` rather than `-small` as
    // the default because it is the tier the published RankIC results are
    // quoted for, and 391 MB is not a size worth trading accuracy for.
    arch!("kronos", "Kronos BSQ-tokenizer candlestick model", Forecast, Brain, "brain-kronos",
          default_ref: Some("NeoQuasar/Kronos-base"),
          extra_refs: &["NeoQuasar/Kronos-Tokenizer-base"],
          weights_env: &[("BRAIN_KRONOS_DECODER", "decoder"), ("BRAIN_KRONOS_TOKENIZER", "tokenizer")]),
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
    fn every_fetchable_ref_parses_as_a_non_reserved_two_segment_ref() {
        // A default_ref (or an extra_ref) names a REAL upstream repo to
        // auto-fetch, never a reserved brain/local/test vendor (those are
        // never fetched) and never a stray third path segment.
        for a in ARCHS {
            for r in a.default_ref.iter().chain(a.extra_refs.iter()) {
                let (vendor, repo) = r.split_once('/').unwrap_or_else(|| panic!("{:?}: ref {r:?} has no '/'", a.id));
                assert!(!repo.contains('/'), "{:?}: ref {r:?} has more than one '/'", a.id);
                assert!(!matches!(vendor, "brain" | "local" | "test"), "{:?}: ref {r:?} names a reserved vendor -- reserved vendors are never fetched", a.id);
            }
        }
    }

    #[test]
    fn extra_refs_are_distinct_from_the_default_and_imply_weights_env() {
        for a in ARCHS {
            if a.extra_refs.is_empty() {
                continue;
            }
            // An extra ref is only ever fetched by `ensure_env_weights`, which
            // exists to resolve `weights_env`. Listing one without the other
            // would download a checkpoint nothing then reads.
            assert!(a.default_ref.is_some(), "{:?}: extra_refs without a default_ref -- there is no first fetch to extend", a.id);
            assert!(!a.weights_env.is_empty(), "{:?}: extra_refs without weights_env -- nothing would ever read the extra checkpoint", a.id);
            let mut seen: HashSet<&str> = HashSet::new();
            for r in a.default_ref.iter().chain(a.extra_refs.iter()) {
                assert!(seen.insert(r), "{:?}: ref {r:?} listed twice -- it would be fetched twice", a.id);
            }
        }
    }

    #[test]
    fn weights_env_roles_are_unique_within_an_arch() {
        // `ensure_env_weights` merges the roles of every fetched ref into one
        // map, so two roles with the same name would silently resolve to
        // whichever repo happened to be fetched last.
        for a in ARCHS {
            let mut seen: HashSet<&str> = HashSet::new();
            for (var, role) in a.weights_env {
                assert!(seen.insert(role), "{:?}: role {role:?} claimed by more than one weights_env var (at {var})", a.id);
            }
        }
    }

    #[test]
    fn kronos_names_both_of_its_upstream_repos() {
        // The two-repo case the `extra_refs` field exists for. If someone ever
        // finds (or publishes) a single repo carrying both checkpoints, this
        // is the test that should send them to `Arch::extra_refs`'s doc before
        // they collapse the row.
        let k = by_id("kronos").expect("the kronos row exists");
        assert_eq!(k.default_ref, Some("NeoQuasar/Kronos-base"));
        assert_eq!(k.extra_refs, &["NeoQuasar/Kronos-Tokenizer-base"]);
        let roles: Vec<&str> = k.weights_env.iter().map(|(_, r)| *r).collect();
        assert_eq!(roles, ["decoder", "tokenizer"], "one role per repo, and the names the two FilesRecipe rows write");
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
    fn qwen35_and_qwen35moe_are_distinct_rows_despite_the_shared_prefix() {
        // "qwen35" is a literal prefix of "qwen35moe" - every lookup here
        // (`by_id`/`by_hf`/`by_gguf`) is `==`/exact-slice-contains, never
        // `starts_with`, so this is a regression guard against that ever
        // changing, not a fix for a live bug.
        assert_eq!(by_id("qwen35").map(|a| a.id), Some("qwen35"));
        assert_eq!(by_id("qwen35moe").map(|a| a.id), Some("qwen35moe"));
        assert_eq!(by_gguf("qwen35").map(|a| a.id), Some("qwen35"));
        assert_eq!(by_gguf("qwen35moe").map(|a| a.id), Some("qwen35moe"));
        assert_eq!(by_hf("Qwen3_5ForConditionalGeneration").map(|a| a.id), Some("qwen35"));
        assert_eq!(by_hf("qwen3_5").map(|a| a.id), Some("qwen35"));
    }

    #[test]
    fn wan_owns_the_bare_gguf_spelling_so_the_family_stays_one_row() {
        // Wan2.1 and Wan2.2 GGUFs both carry `general.architecture = "wan"`
        // (ComfyUI-GGUF's IMG_ARCH_LIST). Keeping the id itself equal to that
        // spelling -- rather than `wan21` + `gguf: Some("wan")` -- is what
        // lets one row cover the whole family: two release-pinned rows would
        // both have to claim "wan" and trip
        // `gguf_spellings_are_unique_across_archs`. If someone ever splits
        // this into per-release ids, this test is the thing that should stop
        // them and send them back to the module doc's naming rule.
        let wan = by_id("wan").expect("the wan row exists");
        assert_eq!(wan.gguf, None, "wan's id IS its GGUF spelling -- an explicit `gguf:` here means the id drifted off the upstream name");
        assert_eq!(by_gguf("wan").map(|a| a.id), Some("wan"));
        assert_eq!(by_hf("WanTransformer3DModel").map(|a| a.id), Some("wan"));
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
