// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The default-checkpoint auto-fetch policy, and the fetch/convert engine it
//! sits on -- moved out of `crates/cli/src/supply.rs`. What stays in
//! `crates/cli`: [`residency::supply::ModelSupplier`]'s serving-side
//! implementation (`StoreSupplier`, single-flight, background healing), the
//! `BRAIN_AUTO_FETCH` environment gate, and `ensure_env_weights` -- all of
//! which construct or register a CLI-local `ResidentModel`
//! (`crate::model_dir::resident_for_local`), the same "residency-adapter
//! glue" line `crates/catalog`'s own module doc draws.
//!
//! [`ensure_default_weights`] is the one piece named explicitly for this
//! move: `crate::resolve::maybe_inject_default_weights` calls it to fill in
//! `--weights`/`--tokenizer` for a flagless `brain infer <arch>`, and every
//! embedder wants the same "give me a runnable default checkpoint for this
//! architecture" primitive. It used to decide silently from one process-wide
//! environment variable (`BRAIN_AUTO_FETCH`); [`DownloadPolicy`] makes that
//! choice an explicit parameter instead, so a caller that must never touch
//! the network (`DownloadPolicy::Offline`) can say so without an env var.
//!
//! Swedish Embedded AB implements model distribution and weight-management
//! tooling for its clients. If your team needs expertise in fetching and
//! converting third-party checkpoints into a servable format, on demand and
//! with an explicit network policy, you can procure our services by sending
//! an email to info@swedishembedded.com.

use std::collections::BTreeMap;
use std::io::IsTerminal;

use brain_modelref::ModelRef;
use brain_modelstore::recipe::{WanRecipe, ZimageRecipe};
use brain_modelstore::{CompoundManifest, Hub, Step, Store, MANIFEST_FILE};

use crate::progress::{Mode, Reporter};

/// How eagerly [`ensure_default_weights`] is allowed to touch the network.
///
/// This is new, deliberate API surface: the CLI used to decide this
/// implicitly from one process-wide `BRAIN_AUTO_FETCH` environment variable
/// (`crate::supply::auto_fetch_enabled`, in `crates/cli`, still does for the
/// CLI's own default), with no way for a caller to say "never touch the
/// network, whatever the environment says". An embedder that must guarantee
/// offline operation now can, by passing [`DownloadPolicy::Offline`]
/// directly, independent of any process environment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DownloadPolicy {
    /// Fetch only when nothing local already resolves the reference - the
    /// sensible default for an embedder with no opinion: a checkpoint
    /// already on disk is never re-checked against the network, and a
    /// missing one is fetched once.
    #[default]
    IfMissing,
    /// Never touch the network. A checkpoint not already local is a clean
    /// error naming the remedy, never a download.
    Offline,
    /// Always go through the fetch/plan/execute sequence, even when a local
    /// copy might already resolve - what `BRAIN_AUTO_FETCH=1` has always
    /// meant for the CLI. The plan step itself skips whatever has already
    /// landed on disk, so this differs from `IfMissing` only in that it
    /// still consults the hub for a resolution/verification pass.
    AlwaysCheck,
}

/// [`ensure_default_weights`]'s result: the weights path every architecture
/// needs, plus the tokenizer path for the ones that also need one (a fetched
/// HF checkpoint's `tokenizer.json`, when present).
#[derive(Debug)]
pub struct DefaultWeights {
    pub weights: String,
    pub tokenizer: Option<String>,
}

/// Auto-fetch `arch`'s [`brain_arch::Arch::default_ref`] checkpoint into the
/// model store under `policy`, and return the path to its
/// `model.brain.safetensors`.
///
/// Not single-flight (unlike the serving-side `StoreSupplier`, built for
/// concurrent server requests sharing one long-lived process): a one-shot
/// caller has exactly one in-flight request, so the plain plan/execute/
/// convert sequence is enough - two callers racing the same cold fetch is a
/// real but rare case, and each just refetches into the same destination
/// independently rather than corrupting anything (`brain_modelstore::fetch`
/// writes via a temp file + atomic rename).
pub fn ensure_default_weights(arch: &str, policy: DownloadPolicy) -> Result<DefaultWeights, String> {
    let root = crate::model_dir::resolve(None).ok_or_else(|| "no models directory (no $HOME and no $BRAIN_MODELS_DIR)".to_string())?;
    ensure_default_weights_with(arch, &Store::new(root), &brain_modelstore::HfHub::new(), policy)
}

/// [`ensure_default_weights`]'s implementation, taking `store`/`hub`
/// explicitly so it is testable against [`brain_modelstore::FakeHub`] with no
/// real network or `$HOME`.
fn ensure_default_weights_with(arch: &str, store: &Store, hub: &dyn Hub, policy: DownloadPolicy) -> Result<DefaultWeights, String> {
    let a = brain_arch::by_id(arch).ok_or_else(|| format!("{arch}: not a registered architecture"))?;
    let default_ref = a.default_ref.ok_or_else(|| format!("{arch}: no default checkpoint known -- pass --weights explicitly"))?;

    let local_only = || -> Result<DefaultWeights, String> {
        let r = ModelRef::parse(default_ref).map_err(|e| format!("{default_ref}: {e}"))?;
        let Some(local) = store.local(&r) else {
            return Err(format!(
                "{arch}: {default_ref} is not pulled. Fetch with `brain pull {default_ref}`, or rerun with --autofetch (BRAIN_AUTO_FETCH=1)."
            ));
        };
        default_weights_from_local(arch, &local)
    };

    match policy {
        DownloadPolicy::Offline => local_only(),
        DownloadPolicy::IfMissing => match local_only() {
            Ok(w) => Ok(w),
            Err(_) => {
                let local = fetch_one_ref(default_ref, store, hub)?;
                default_weights_from_local(arch, &local)
            }
        },
        DownloadPolicy::AlwaysCheck => {
            let local = fetch_one_ref(default_ref, store, hub)?;
            default_weights_from_local(arch, &local)
        }
    }
}

fn default_weights_from_local(arch: &str, local: &brain_modelstore::LocalModel) -> Result<DefaultWeights, String> {
    // A model served from its downloaded directory names it as its
    // `weights` role, with the tokenizer that directory ships.
    let from_role = local.roles.as_ref().and_then(|r| r.get("weights"));
    let weights_path = from_role.unwrap_or(&local.weights);
    let weights = weights_path.to_str().map(str::to_string).ok_or_else(|| format!("{arch}: non-UTF8 store path"))?;
    let tokenizer = match from_role {
        Some(dir) => Some(dir.join("tokenizer.json")).filter(|t| t.is_file()),
        None => local.tokenizer.clone(),
    };
    let tokenizer = tokenizer.as_deref().and_then(|p| p.to_str()).map(str::to_string);
    Ok(DefaultWeights { weights, tokenizer })
}

/// Fetch ONE `<vendor>/<repo>` and run its recipe's finish step. Factored out
/// so an architecture whose checkpoint upstream publishes as several repos
/// (`brain_arch::Arch::extra_refs`) can drive this once per repo - one
/// `ModelRef` to one listing to one `Plan` each time, which is exactly the
/// shape `brain_modelstore::plan` supports.
pub fn fetch_one_ref(default_ref: &str, store: &Store, hub: &dyn Hub) -> Result<brain_modelstore::LocalModel, String> {
    let reference = ModelRef::parse(default_ref).map_err(|e| format!("{default_ref}: {e}"))?;
    let plan = brain_modelstore::plan(&reference, store, hub).map_err(|e| format!("{default_ref}: {e}"))?;
    // Progress on stderr: this runs inside a model command whose stdout
    // carries the command's own output.
    let mut err = std::io::stderr();
    let mode = Mode::of(err.is_terminal());
    let (local, moved, secs) = execute_plan_reported(store, hub, &plan, default_ref, mode, &mut err)?;
    eprintln!("brain: {default_ref}: fetched {} in {}", crate::progress::human_bytes(moved), crate::progress::human_secs(secs));
    Ok(local)
}

/// Run an already-built [`brain_modelstore::Plan`] to completion, rendering
/// the downloads through [`Reporter`] - the same progress shape `brain pull`
/// draws, because an auto-fetch download IS a pull, just one the caller did
/// not spell out by name. Returns the now-servable model plus what moved and
/// how long it took, so the caller states the outcome once, its own way.
/// `mode` and `out` are parameters rather than reaching for stderr here, so
/// the rendering is testable byte-for-byte.
///
/// `label` is what the reporter shows (the reference the caller typed, or the
/// one the command resolved).
pub fn execute_plan_reported(store: &Store, hub: &dyn Hub, plan: &brain_modelstore::Plan, label: &str, mode: Mode, out: &mut dyn std::io::Write) -> Result<(brain_modelstore::LocalModel, u64, f64), String> {
    let remaining = brain_modelstore::remaining_download(store, hub, plan).map_err(|e| format!("{label}: {e}"))?;
    let mut reporter = Reporter::new(mode, out, label, remaining);
    let model = execute_plan(store, hub, plan, label, &mut |name, got, total| reporter.on_bytes(name, got, total))?;
    let (moved, secs) = reporter.finish();
    Ok((model, moved, secs))
}

/// Run an already-built [`brain_modelstore::Plan`] to completion: download
/// every outstanding file, run whichever finish step the plan's recipe
/// deferred, and return the now-servable model.
///
/// The one implementation of "materialize this plan". Auto-fetch reaches it
/// through [`execute_plan_reported`]; `brain pull` (`crate::pull_cli`, in
/// `crates/cli`) reaches it directly with a closure that draws its own
/// progress bar. Making `brain pull` the explicit spelling of the operation
/// auto-fetch already performs is the whole point - two code paths that fetch
/// models would be two sets of bugs.
///
/// `label` is what the caller calls this model in messages (a `default_ref`
/// string, or the reference the user typed).
pub fn execute_plan(store: &Store, hub: &dyn Hub, plan: &brain_modelstore::Plan, label: &str, progress: &mut dyn FnMut(&str, u64, Option<u64>)) -> Result<brain_modelstore::LocalModel, String> {
    execute_plan_opt(store, hub, plan, label, progress)?.ok_or_else(|| format!("{label}: fetched but not found on disk (unexpected)"))
}

/// [`execute_plan`] without the requirement that the result be a servable
/// model. Every plan that materializes a whole repo produces one, which is
/// why [`execute_plan`] insists; a plan for ONE named artifact inside a repo
/// (`brain_modelstore::plan_file`, what a pasted file URL asks for) may not:
/// a lone `text_encoder/model.safetensors` is a file a `--text-encoder` flag
/// can be pointed at, not a checkpoint the store can serve by name. `None`
/// is that case, and the caller reports the path instead -- never a swallowed
/// failure, since every step still had to succeed to get here.
pub fn execute_plan_opt(store: &Store, hub: &dyn Hub, plan: &brain_modelstore::Plan, label: &str, progress: &mut dyn FnMut(&str, u64, Option<u64>)) -> Result<Option<brain_modelstore::LocalModel>, String> {
    let reference = &plan.reference;
    let deferred = brain_modelstore::execute(store, hub, plan, progress).map_err(|e| format!("{label}: {e}"))?;

    for step in &deferred {
        match step {
            Step::Convert { vendor, repo, recipe } => convert(store, vendor, repo, recipe).map_err(|e| format!("{label}: {e}"))?,
            other => {
                return Err(format!(
                    "{label}: needs an additional step ({other:?}) auto-fetch does not automate yet -- fetch and convert manually"
                ))
            }
        }
    }

    Ok(store.local(reference))
}

/// Dispatch a `Step::Convert { vendor, repo, recipe }` to the matching
/// family's finish logic. `recipe` is the `ArtifactRecipe::id` `modelstore::
/// plan` already picked (`brain_modelstore::recipe`) -- routing on it directly
/// rather than re-deriving the family from disk a second time, one
/// implementation of "which family this repo is", not a second guess that
/// could drift from the first.
pub fn convert(store: &Store, vendor: &str, repo: &str, recipe: &str) -> Result<(), String> {
    match recipe {
        "transformers" => convert_transformers(store, vendor, repo),
        "zimage" => convert_zimage(store, vendor, repo),
        // Special-cased ahead of the generic `files_recipe_roles` fallback
        // purely for `convert_diffusers_pipeline`'s `model_index.json`
        // safety net -- see its docs. The manifest it writes is otherwise
        // identical to what the fallback would have produced.
        "flux2" => convert_flux2(store, vendor, repo),
        "wan" => convert_wan(store, vendor, repo),
        "yolo" => convert_yolo(store, vendor, repo),
        "gguf" => convert_gguf(store, vendor, repo),
        other => match brain_modelstore::recipe::files_recipe_roles(other) {
            Some((family, roles)) => convert_files(store, vendor, repo, family, roles),
            None => Err(format!("{vendor}/{repo}: convert: unknown recipe {other:?} (bug: modelstore::recipe::recipes() and this dispatch have drifted)")),
        },
    }
}

/// The GGUF recipe's finish step. There is no tensor rewrite and no manifest
/// to write: a `<QUANT>.gguf` sitting in a repo directory is already exactly
/// what `Store::local` resolves a quantized reference to.
///
/// What it does instead is read each landed file's header back off disk and
/// report the architecture it declares. That is the FIRST moment that fact is
/// knowable: `brain_modelstore::Hub` exposes list / read-whole-file /
/// stream-to-disk and no range request, so a multi-gigabyte checkpoint's
/// `general.architecture` cannot be consulted while choosing which file to
/// fetch, and the choice upstream is therefore made on the filename's
/// quantization token alone. Reading the header here is what catches a
/// "download" that is not a GGUF at all -- an LFS pointer file, an HTML error
/// page -- before a model crate is handed it, and it costs one mmap of the
/// header, not a read of the weights.
fn convert_gguf(store: &Store, vendor: &str, repo: &str) -> Result<(), String> {
    let dir = store.repo_dir(&ModelRef::new(vendor, repo, None));
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| format!("{vendor}/{repo}: convert: {}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "gguf"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(format!("{vendor}/{repo}: convert: no .gguf file landed in {}", dir.display()));
    }
    for f in &files {
        let path = f.to_str().ok_or_else(|| format!("{vendor}/{repo}: convert: non-UTF8 path {}", f.display()))?;
        let g = checkpoint::gguf::MmapGguf::open(path).map_err(|e| format!("{vendor}/{repo}: convert: {path}: not a readable GGUF: {e}"))?;
        let arch = g
            .kv()
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("{vendor}/{repo}: convert: {path}: GGUF declares no general.architecture"))?;
        let name = f.file_name().and_then(|n| n.to_str()).unwrap_or("");
        eprintln!("brain: {vendor}/{repo}: {name}: GGUF architecture {arch:?}, {} tensors", g.names().len());
    }
    Ok(())
}

/// A [`brain_modelstore::recipe::FilesRecipe`]'s finish step: the files it
/// downloaded need no tensor rewrite at all -- write the
/// [`CompoundManifest`] naming their roles, reading the SAME
/// `(family, roles)` table [`brain_modelstore::recipe::files_recipe_roles`]
/// already exposes rather than a second copy of it living here (the
/// `ZimageRecipe::ROLES` pattern [`convert_zimage`] uses, generalised past
/// one hardcoded family).
fn convert_files(store: &Store, vendor: &str, repo: &str, family: &str, roles_table: &[(&str, &str)]) -> Result<(), String> {
    let dir = store.repo_dir(&ModelRef::new(vendor, repo, None));
    let mut roles = BTreeMap::new();
    for (role, rel) in roles_table {
        if !dir.join(rel).exists() {
            return Err(format!("{vendor}/{repo}: convert: role {role:?} ({rel}) did not download"));
        }
        roles.insert(role.to_string(), rel.to_string());
    }
    let manifest = CompoundManifest { id: format!("{vendor}/{repo}"), family: family.to_string(), roles };
    let bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| format!("{vendor}/{repo}: convert: encode manifest: {e}"))?;
    std::fs::write(dir.join(MANIFEST_FILE), bytes).map_err(|e| format!("{vendor}/{repo}: convert: write manifest: {e}"))
}

/// The yolo recipe: `YoloRecipe::artifacts` downloaded exactly one
/// `yolov8*.pt` file into the repo dir, which `yolov8::Yolo::load` reads as
/// it is; "finish" is a manifest naming it, with nothing converted and
/// nothing removed.
fn convert_yolo(store: &Store, vendor: &str, repo: &str) -> Result<(), String> {
    let dir = store.repo_dir(&ModelRef::new(vendor, repo, None));
    let pt = std::fs::read_dir(&dir)
        .map_err(|e| format!("{vendor}/{repo}: convert: {}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_str().map(str::to_string))
        .find(|n| n.starts_with("yolov8") && n.ends_with(".pt"))
        .ok_or_else(|| format!("{vendor}/{repo}: convert: no downloaded yolov8*.pt file in {}", dir.display()))?;
    convert_files(store, vendor, repo, "yolo", &[("weights", &pt)])
}

/// The finish step shared by every diffusers-pipeline family
/// (`model_index.json` at the repo root + role subdirectories): no tensor
/// rewrite is needed for any of them (each loader remaps names in memory at
/// load time), so "finish" is writing the `brain.manifest.json`
/// `Store::local` reads back, via [`convert_files`] -- but ONLY after
/// confirming `model_index.json`'s own `_class_name` actually names the
/// pipeline this recipe was written for.
///
/// That check exists because a file listing alone cannot always tell two
/// such families apart: an official `black-forest-labs/FLUX.2-klein-4B`
/// checkpoint matches `ZimageRecipe`'s shape signature byte-for-byte (see
/// `brain_modelstore::recipe`'s `flux2` `FilesRecipe` row), and was in fact
/// misclassified as `"zimage"` before that row's `repos` pin existed. A pin
/// can be missing or wrong for some family added later the same way; this is
/// the safety net that turns that mistake into a loud, named "wrong recipe
/// matched this repo" error at conversion time instead of a silently wrong
/// manifest, whether or not the registry's `matches()`/`select` ordering got
/// it right.
fn convert_diffusers_pipeline(store: &Store, vendor: &str, repo: &str, family: &str, roles_table: &'static [(&'static str, &'static str)], expected_class_name: &str) -> Result<(), String> {
    let dir = store.repo_dir(&ModelRef::new(vendor, repo, None));
    let index_bytes = std::fs::read(dir.join("model_index.json")).map_err(|e| format!("{vendor}/{repo}: convert: read model_index.json: {e}"))?;
    let index: serde_json::Value = serde_json::from_slice(&index_bytes).map_err(|e| format!("{vendor}/{repo}: convert: unparseable model_index.json: {e}"))?;
    let class_name = index
        .get("_class_name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("{vendor}/{repo}: convert: model_index.json has no _class_name"))?;
    if class_name != expected_class_name {
        return Err(format!(
            "{vendor}/{repo}: convert: model_index.json declares _class_name {class_name:?}, expected {expected_class_name:?} for the {family:?} recipe -- wrong recipe matched this repo"
        ));
    }
    convert_files(store, vendor, repo, family, roles_table)
}

/// The zimage recipe: [`ZimageRecipe::ROLES`] is z-image's own role layout
/// (one source of truth for what `ZimageRecipe::artifacts` downloaded and
/// what this manifest names, not a second guess of what landed on disk), and
/// `"ZImagePipeline"` is the `_class_name` every real Z-Image release
/// declares (`s3dit::pipeline`'s module docs name it directly).
fn convert_zimage(store: &Store, vendor: &str, repo: &str) -> Result<(), String> {
    convert_diffusers_pipeline(store, vendor, repo, "zimage", ZimageRecipe::ROLES, "ZImagePipeline")
}

/// The flux2 recipe: same shape, same roles ([`ZimageRecipe::ROLES`]) as
/// z-image, but a different pipeline class -- `"Flux2KleinPipeline"` is what
/// `black-forest-labs/FLUX.2-klein-4B`'s and `-9B`'s own `model_index.json`
/// declare.
fn convert_flux2(store: &Store, vendor: &str, repo: &str) -> Result<(), String> {
    convert_diffusers_pipeline(store, vendor, repo, "flux2", ZimageRecipe::ROLES, "Flux2KleinPipeline")
}

/// The wan recipe: like [`convert_zimage`], no tensor rewrite is needed
/// (`wan::import::import_dit` remaps names in memory at load time, and the
/// VAE/umT5 are read straight from their `.pth`), so "finish" is writing the
/// `brain.manifest.json` naming the four roles
/// [`WanRecipe::ROLES`](brain_modelstore::recipe::WanRecipe::ROLES) declares.
///
/// The one path that is decided HERE rather than in the table: the 1.3B tier
/// ships a single `diffusion_pytorch_model.safetensors` and the 14B tiers
/// ship a shard set plus an index, so the `dit` role is the file when it
/// exists and the repo directory otherwise -- `wan::pipeline`'s reader takes
/// either, and `checkpoint::safetensors::read_model_dir` follows the index
/// there rather than sweeping the two `.pth` siblings into the DiT.
fn convert_wan(store: &Store, vendor: &str, repo: &str) -> Result<(), String> {
    let dir = store.repo_dir(&ModelRef::new(vendor, repo, None));
    let mut roles = BTreeMap::new();
    for (role, rel) in WanRecipe::ROLES {
        let rel = if *role == "dit" && !dir.join(rel).exists() && dir.join("diffusion_pytorch_model.safetensors.index.json").exists() {
            WanRecipe::SHARDED_DIT
        } else {
            rel
        };
        if !dir.join(rel).exists() {
            return Err(format!("{vendor}/{repo}: convert: role {role:?} ({rel}) did not download"));
        }
        roles.insert(role.to_string(), rel.to_string());
    }
    let manifest = CompoundManifest { id: format!("{vendor}/{repo}"), family: "wan".to_string(), roles };
    let bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| format!("{vendor}/{repo}: convert: encode manifest: {e}"))?;
    std::fs::write(dir.join(MANIFEST_FILE), bytes).map_err(|e| format!("{vendor}/{repo}: convert: write manifest: {e}"))
}

/// The manifest role a family's resolver reads its downloaded directory
/// under, where that is not `weights`: `deepseek2ocr` composes four
/// checkpoints out of one directory and calls it `dir`
/// (`deepseek2ocr::spec::Deepseek2ocrSpec`), as does `decide`.
///
/// `fastvlm` and `qwen3asr` are not transformers-recipe families at all:
/// their upstream repos ship only `vocab.json` + `merges.txt` (no
/// `tokenizer.json`), so they need the WHOLE repo, which their own
/// `FilesRecipe` rows in `crates/modelstore/src/recipe.rs` fetch.
const DIRECTORY_ROLE: &[(&str, &str)] = &[("deepseek2ocr", "dir"), ("decide", "dir")];

/// The original (and still only) family: an HF `transformers`-shaped repo.
/// Every family brain serves from one reads the directory exactly as
/// downloaded - config, tokenizer and weights (safetensors or
/// `pytorch_model*.bin`), converted per tensor in memory as they load - so
/// "finish" is a `brain.manifest.json` naming the directory: nothing is
/// rewritten on disk and nothing downloaded is removed.
///
/// Reads `<dir>/config.json` to learn the family the same way
/// `modelstore::plan`'s `TransformersRecipe` already gated the download on
/// (`family_of_architecture`) -- one implementation of "which families brain
/// can serve", not a second guess that could drift from the first. The
/// manifest's `id` is the fully-qualified `vendor/repo` reference, so the
/// resident registers under what the client actually asked for.
fn convert_transformers(store: &Store, vendor: &str, repo: &str) -> Result<(), String> {
    let dir = store.repo_dir(&ModelRef::new(vendor, repo, None));
    let config_bytes = std::fs::read(dir.join("config.json")).map_err(|e| format!("{vendor}/{repo}: read config.json: {e}"))?;
    let config: serde_json::Value = serde_json::from_slice(&config_bytes).map_err(|e| format!("{vendor}/{repo}: config.json: {e}"))?;
    let arch = brain_modelstore::declared_architecture(&config).ok_or_else(|| format!("{vendor}/{repo}: config.json has no architecture"))?;
    let family = brain_modelstore::family_of_architecture(&arch).ok_or_else(|| format!("{vendor}/{repo}: unsupported architecture {arch:?}"))?;
    // gpt2 is nanogpt-style, trained from scratch -- brain has no reader for
    // an HF GPT-2 checkpoint (a Conv1D-transpose layout), so this fails
    // cleanly instead of registering a model nothing can load.
    if family == "gpt2" {
        return Err(format!("{vendor}/{repo}: gpt2 has no HF checkpoint reader yet -- fetch and convert manually"));
    }
    // qwen3tts's own repo (`speech_tokenizer/config.json` present) is claimed
    // by the `qwen3tts` `FilesRecipe` ahead of `TransformersRecipe` in
    // `recipes()`'s order, so `family == "qwen3tts"` never reaches here.
    let role = DIRECTORY_ROLE.iter().find(|(f, _)| *f == family).map_or("weights", |(_, role)| *role);
    convert_files(store, vendor, repo, family, &[(role, ".")])
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_modelstore::FakeHub;

    fn store(name: &str) -> Store {
        let dir = std::env::temp_dir().join(name);
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        Store::new(dir)
    }

    /// A tiny but real 1-layer tied-embedding Qwen3 HF checkpoint, reproduced
    /// as raw bytes for a [`FakeHub`] -- the same shape
    /// `crates/qwen3/src/import.rs`'s own `build_tiny_hf_dir` test fixture
    /// uses, since `ensure_default_weights_with`/`execute_plan_reported` must
    /// drive the whole plan -> download -> convert pipeline, not just call
    /// the importer directly.
    fn tiny_qwen3_hf_files() -> (Vec<u8>, Vec<u8>) {
        let config = br#"{"architectures":["Qwen3ForCausalLM"],
            "vocab_size":5,"hidden_size":6,"num_hidden_layers":1,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
            "intermediate_size":8,"rope_theta":1000000,"rms_norm_eps":1e-6,
            "tie_word_embeddings":true}"#
            .to_vec();
        fn seq(base: f32, n: usize) -> Vec<f32> {
            (0..n).map(|i| base + i as f32).collect()
        }
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = vec![
            ("model.embed_tokens.weight".into(), vec![30], seq(1_000_000.0, 30)),
            ("model.norm.weight".into(), vec![6], seq(2_000_000.0, 6)),
            ("model.layers.0.input_layernorm.weight".into(), vec![6], seq(10.0, 6)),
            ("model.layers.0.self_attn.q_proj.weight".into(), vec![48], seq(20.0, 48)),
            ("model.layers.0.self_attn.k_proj.weight".into(), vec![24], seq(70.0, 24)),
            ("model.layers.0.self_attn.v_proj.weight".into(), vec![24], seq(100.0, 24)),
            ("model.layers.0.self_attn.q_norm.weight".into(), vec![4], seq(130.0, 4)),
            ("model.layers.0.self_attn.k_norm.weight".into(), vec![4], seq(140.0, 4)),
            ("model.layers.0.self_attn.o_proj.weight".into(), vec![48], seq(150.0, 48)),
            ("model.layers.0.post_attention_layernorm.weight".into(), vec![6], seq(200.0, 6)),
            ("model.layers.0.mlp.gate_proj.weight".into(), vec![48], seq(210.0, 48)),
            ("model.layers.0.mlp.up_proj.weight".into(), vec![48], seq(260.0, 48)),
            ("model.layers.0.mlp.down_proj.weight".into(), vec![48], seq(310.0, 48)),
        ];
        static CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let out = std::env::temp_dir().join(format!("brain-loader-supply-tiny-qwen3-{}-{n}", std::process::id()));
        checkpoint::st::save_safetensors(out.to_str().unwrap(), &tensors, &serde_json::Value::Null, None).unwrap();
        let weights = std::fs::read(&out).unwrap();
        std::fs::remove_file(&out).ok();
        (config, weights)
    }

    /// A reported fetch renders progress the way `brain pull` does - one
    /// budgeted ladder of plain percentage lines piped - and hands back what
    /// moved, for the caller's own single outcome line. (Pipe mode plus a
    /// byte sink here: deterministic lines, no pty.)
    #[test]
    fn a_reported_fetch_renders_pull_style_progress_and_reports_what_moved() {
        let dir = std::env::temp_dir().join(format!(
            "brain-loader-supply-reported-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let store = Store::new(&dir);
        let (config, weights) = tiny_qwen3_hf_files();
        let mut hub = FakeHub::new();
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "config.json", config.clone());
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "model.safetensors", weights.clone());
        let reference = ModelRef::parse("Qwen/Qwen3-0.6B").unwrap();
        let plan = brain_modelstore::plan(&reference, &store, &hub).unwrap();
        let mut sink: Vec<u8> = Vec::new();
        let (_local, moved, _secs) = execute_plan_reported(&store, &hub, &plan, "Qwen/Qwen3-0.6B", Mode::Pipe, &mut sink).unwrap();
        // Only downloads move bytes; the deferred convert does not report.
        assert_eq!(moved, (config.len() + weights.len()) as u64);
        let out = String::from_utf8(sink).unwrap();
        assert!(out.contains("100%"), "the ladder reaches 100%: {out:?}");
        assert!(!out.contains('\r'), "pipe mode has no carriage returns: {out:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A Llama config variant of the qwen3 decoder, shipped the way
    /// deepseek-coder-1.3b-base is (a lone `pytorch_model.bin`), is fetched
    /// and served from the directory as downloaded: the pull writes only a
    /// manifest, the original weights stay, and the qwen3 decoder reads them
    /// at their own config (no QK-norm, untied head, linear RoPE scaling).
    #[test]
    fn a_llama_bin_checkpoint_is_served_from_its_downloaded_files() {
        let config = br#"{"architectures":["LlamaForCausalLM"],"vocab_size":5,"hidden_size":8,"num_hidden_layers":2,
            "num_attention_heads":2,"num_key_value_heads":2,"intermediate_size":12,"rope_theta":100000,
            "rms_norm_eps":1e-6,"tie_word_embeddings":false,"max_position_embeddings":64,
            "rope_scaling":{"type":"linear","factor":4.0}}"#
            .to_vec();
        let cfg = qwen3::hf::decoder_config(std::str::from_utf8(&config).unwrap()).unwrap();
        let names = qwen3::hf::HfNames::CAUSAL_LM;
        let init = qwen3::init_weights(&cfg, 3);
        let tensors: Vec<checkpoint::torchpt_write::TensorOut> = cfg
            .param_list()
            .into_iter()
            .map(|(n, numel)| checkpoint::torchpt_write::TensorOut { name: names.from_brain(&n).unwrap(), shape: vec![numel], data: init[&n].clone() })
            .collect();
        let bin = std::env::temp_dir().join(format!("brain-loader-llama-bin-{}", std::process::id()));
        checkpoint::torchpt_write::write(bin.to_str().unwrap(), &tensors).unwrap();
        let mut hub = FakeHub::new();
        hub.add_file("deepseek-ai", "tiny-llama", "main", "config.json", config);
        hub.add_file("deepseek-ai", "tiny-llama", "main", "pytorch_model.bin", std::fs::read(&bin).unwrap());
        std::fs::remove_file(&bin).ok();

        let store = store("loader-supply-test-llama-bin");
        let reference = ModelRef::parse("deepseek-ai/tiny-llama").unwrap();
        let plan = brain_modelstore::plan(&reference, &store, &hub).unwrap();
        execute_plan_reported(&store, &hub, &plan, "deepseek-ai/tiny-llama", Mode::Pipe, &mut Vec::new()).unwrap();

        let dir = store.repo_dir(&reference);
        assert!(dir.join("pytorch_model.bin").exists(), "the downloaded weights are kept");
        assert!(!dir.join("model.brain.safetensors").exists(), "nothing is converted on disk");
        let local = store.local(&reference).expect("the pulled model resolves");
        assert_eq!(local.card.as_ref().unwrap().family, "llama");
        let served = default_weights_from_local("llama", &local).unwrap();
        assert_eq!(std::path::Path::new(&served.weights), dir);
        let (got_cfg, src) = qwen3::open_checkpoint(&served.weights).unwrap();
        assert_eq!(got_cfg, cfg);
        for (n, _) in cfg.param_list() {
            let mut got = Vec::new();
            assert!(checkpoint::TensorSource::with_tensor(&*src, &n, &mut |t| got = t.to_vec()), "{n}");
            assert_eq!(got, init[&n], "{n}");
        }
    }

    /// Every transformers-shaped family finishes a pull by naming the
    /// downloaded directory: no converted copy, and the download untouched.
    #[test]
    fn a_pull_finishes_with_a_manifest_and_keeps_the_download() {
        for (class, family) in [("Lfm2ForCausalLM", "lfm2"), ("Qwen3OmniMoeForConditionalGeneration", "qwen3omnimoe"), ("LlamaForCausalLM", "llama")] {
            let store = store(&format!("loader-supply-test-manifest-{family}"));
            let reference = ModelRef::parse(&format!("test/{family}")).unwrap();
            let dir = store.repo_dir(&reference);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("config.json"), serde_json::json!({"architectures": [class]}).to_string()).unwrap();
            std::fs::write(dir.join("model.safetensors"), b"downloaded").unwrap();
            convert_transformers(&store, "test", family).unwrap();
            let manifest: CompoundManifest = serde_json::from_slice(&std::fs::read(dir.join(MANIFEST_FILE)).unwrap()).unwrap();
            assert_eq!(manifest.family, family);
            assert_eq!(manifest.roles.get("weights").map(String::as_str), Some("."));
            assert_eq!(std::fs::read(dir.join("model.safetensors")).unwrap(), b"downloaded", "{family}: the download is kept as is");
            assert!(!dir.join("model.brain.safetensors").exists(), "{family}: nothing is converted on disk");
        }
    }

    #[test]
    fn ensure_default_weights_always_check_fetches_and_returns_the_downloaded_directory() {
        let (config, weights) = tiny_qwen3_hf_files();
        let mut hub = FakeHub::new();
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "config.json", config);
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "model.safetensors", weights);
        let store = store("loader-supply-test-default-weights-qwen3");

        let got = ensure_default_weights_with("qwen3", &store, &hub, DownloadPolicy::AlwaysCheck).unwrap();
        assert!(got.weights.ends_with("Qwen/Qwen3-0.6B"), "{}", got.weights);
        assert!(std::path::Path::new(&got.weights).exists(), "{} must actually exist on disk", got.weights);
    }

    /// `IfMissing` is the whole point of the new policy: a checkpoint already
    /// resolved locally must not touch the hub at all, even a hub that would
    /// answer everything - the same "never re-check what is already there"
    /// contract `Offline` has, but without refusing to fetch when it truly
    /// is missing.
    #[test]
    fn if_missing_never_touches_the_hub_once_the_checkpoint_is_local() {
        let (config, weights) = tiny_qwen3_hf_files();
        let mut hub = FakeHub::new();
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "config.json", config);
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "model.safetensors", weights);
        let store = store("loader-supply-test-if-missing-local");
        ensure_default_weights_with("qwen3", &store, &hub, DownloadPolicy::AlwaysCheck).unwrap(); // seed the store

        let got = ensure_default_weights_with("qwen3", &store, &FakeHub::new(), DownloadPolicy::IfMissing).unwrap();
        assert!(got.weights.ends_with("Qwen/Qwen3-0.6B"), "{}", got.weights);
    }

    /// `IfMissing` still fetches once when nothing local resolves the
    /// reference - the default an embedder with no opinion should get.
    #[test]
    fn if_missing_fetches_once_when_nothing_is_local() {
        let (config, weights) = tiny_qwen3_hf_files();
        let mut hub = FakeHub::new();
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "config.json", config);
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "model.safetensors", weights);
        let store = store("loader-supply-test-if-missing-fetch");

        let got = ensure_default_weights_with("qwen3", &store, &hub, DownloadPolicy::IfMissing).unwrap();
        assert!(std::path::Path::new(&got.weights).exists());
    }

    /// `Offline` is the explicit, environment-independent "never touch the
    /// network" this whole enum exists to provide.
    #[test]
    fn offline_never_fetches_and_names_the_remedy_when_nothing_is_pulled() {
        let store = store("loader-supply-test-offline-missing");
        let err = ensure_default_weights_with("qwen3", &store, &FakeHub::new(), DownloadPolicy::Offline).unwrap_err();
        assert!(err.contains("brain pull Qwen/Qwen3-0.6B"), "{err}");
        assert!(err.contains("--autofetch"), "{err}");
    }

    #[test]
    fn ensure_default_weights_is_a_clean_error_for_an_arch_with_no_default_ref() {
        // t5encoder has no default_ref (no confirmed small upstream repo
        // yet) -- must fail with a clear reason, never panic or silently
        // pick something.
        let store = store("loader-supply-test-default-weights-no-ref");
        let hub = FakeHub::new();
        let err = ensure_default_weights_with("t5encoder", &store, &hub, DownloadPolicy::default()).unwrap_err();
        assert!(err.contains("no default checkpoint known"), "{err}");
    }

    #[test]
    fn ensure_default_weights_is_a_clean_error_for_an_unknown_arch() {
        let store = store("loader-supply-test-default-weights-unknown-arch");
        let hub = FakeHub::new();
        let err = ensure_default_weights_with("totally-bogus", &store, &hub, DownloadPolicy::default()).unwrap_err();
        assert!(err.contains("not a registered architecture"), "{err}");
    }

    /// The safety net a `repos` pin alone cannot guarantee: a recipe's shape
    /// match (`model_index.json` + the four role dirs) is not proof of which
    /// pipeline a repo really is -- `black-forest-labs/FLUX.2-klein-4B`
    /// matches `ZimageRecipe`'s shape byte-for-byte, and a missing or wrong
    /// `repos` pin on some family added later could route a repo to the
    /// wrong finish code the same way. `convert_zimage` must refuse rather
    /// than write a manifest when the downloaded `model_index.json` itself
    /// names a different pipeline (`"Flux2KleinPipeline"`, not
    /// `"ZImagePipeline"`).
    #[test]
    fn convert_zimage_refuses_when_model_index_names_a_different_pipeline() {
        let dir = store(&format!("loader-supply-test-zimage-wrong-class-{}", std::process::id())).root().to_path_buf();
        let repo_dir = dir.join("black-forest-labs").join("FLUX.2-klein-4B");
        for role_dir in ["transformer", "vae", "text_encoder", "tokenizer"] {
            std::fs::create_dir_all(repo_dir.join(role_dir)).unwrap();
        }
        std::fs::write(repo_dir.join("vae/diffusion_pytorch_model.safetensors"), b"stub").unwrap();
        std::fs::write(repo_dir.join("tokenizer/tokenizer.json"), b"stub").unwrap();
        std::fs::write(repo_dir.join("model_index.json"), br#"{"_class_name": "Flux2KleinPipeline"}"#).unwrap();

        let store = Store::new(dir);
        let err = convert_zimage(&store, "black-forest-labs", "FLUX.2-klein-4B").unwrap_err();
        assert!(err.contains("Flux2KleinPipeline"), "{err}");
        assert!(err.contains("ZImagePipeline"), "{err}");
    }
}
