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
use std::path::Path;

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
    let weights = local.weights.to_str().map(str::to_string).ok_or_else(|| format!("{arch}: non-UTF8 store path"))?;
    let tokenizer = local.tokenizer.as_deref().and_then(|p| p.to_str()).map(str::to_string);
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

/// What a build without a given family's importer returns.
///
/// It names the family AND the cargo feature that would supply it, because a
/// bare "unsupported family" here is indistinguishable from a genuinely
/// unknown architecture -- and those two need opposite fixes. A narrow build
/// (the `brain` SDK, a service, a sample) is a supported configuration, so its
/// failure mode has to be actionable rather than mysterious.
///
/// Compiled only when at least one importer is absent -- with `import-all` on
/// there is no arm that can call it, and an always-present helper would be an
/// unused-function warning in the configuration `brain-cli` actually ships.
#[cfg(not(all(
    feature = "import-qwen3",
    feature = "import-glmdsa",
    feature = "import-lfm2",
    feature = "import-qwen3omnimoe",
    feature = "import-qwen3tts",
    feature = "import-yolov8"
)))]
fn no_importer(vendor: &str, repo: &str, family: &str, feature: &str) -> String {
    format!(
        "{vendor}/{repo}: convert: this build of brain-loader has no {family} importer \
         (enable brain-loader/{feature}, or the `brain` SDK surface that selects it)"
    )
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
        // Real conversion (four output files, two roles), not a passthrough
        // manifest -- special-cased ahead of the generic `files_recipe_roles`
        // fallback, which would otherwise write the `FilesRecipe` row's
        // (deliberately empty) `roles` verbatim. See `convert_qwen3tts`.
        "qwen3tts" => convert_qwen3tts(store, vendor, repo),
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
/// `yolov8*.pt` file into the repo dir; run the pure-Rust importer
/// (`yolov8::import::import_yolov8n`, built on `checkpoint::torchpt`) and write
/// the remapped tensors as `model.brain.safetensors` -- the same single-file
/// convention every transformers-family model already uses, so no store or
/// `resident_for` changes were needed for this family.
#[cfg(not(feature = "import-yolov8"))]
fn convert_yolo(store: &Store, vendor: &str, repo: &str) -> Result<(), String> {
    let _ = store;
    Err(no_importer(vendor, repo, "yolov8", "import-yolov8"))
}

#[cfg(feature = "import-yolov8")]
fn convert_yolo(store: &Store, vendor: &str, repo: &str) -> Result<(), String> {
    let dir = store.repo_dir(&ModelRef::new(vendor, repo, None));
    let pt = std::fs::read_dir(&dir)
        .map_err(|e| format!("{vendor}/{repo}: convert: {}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("yolov8") && n.ends_with(".pt")))
        .ok_or_else(|| format!("{vendor}/{repo}: convert: no downloaded yolov8*.pt file in {}", dir.display()))?;
    let pt_str = pt.to_str().ok_or_else(|| format!("{vendor}/{repo}: convert: non-UTF8 path {}", pt.display()))?;

    let tensors = yolov8::import::import_yolov8n(pt_str)?;
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = tensors.into_iter().map(|(name, shape, data)| (name, shape.into_iter().map(|d| d as u64).collect(), data)).collect();
    let card = checkpoint::st::ModelCard::for_ref(&format!("{vendor}/{repo}"), vendor, repo, None, "yolo");
    let out = dir.join("model.brain.safetensors");
    checkpoint::st::save_safetensors(out.to_str().ok_or_else(|| format!("{vendor}/{repo}: convert: non-UTF8 store path"))?, &tensors, &yolov8::config::YoloConfig::yolov8n().to_json(), Some(&card))
        .map_err(|e| format!("{vendor}/{repo}: convert: write model.brain.safetensors: {e}"))?;
    // The upstream .pt is never read again -- Store::local/scan only ever load
    // model.brain.safetensors (see modelstore::BASE_WEIGHTS_FILE) -- so keeping
    // it around is pure disk waste. Best-effort: a failed cleanup must not fail
    // an otherwise-successful convert.
    std::fs::remove_file(&pt).ok();
    Ok(())
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

/// Families whose model crate loads the downloaded HF checkpoint directory
/// directly (`BRAIN_QWEN3VL_WEIGHTS`/`BRAIN_FASTVLM_WEIGHTS`/
/// `BRAIN_NEMOTRONASR`/`BRAIN_QWEN3ASR` each name a DIRECTORY, not a
/// brain-format file) rather than through a `model.brain.safetensors`
/// conversion -- see [`convert_transformers`]'s branch for why that changes
/// both what "finish" writes and whether the upstream weights get deleted.
/// Only families whose upstream repo ships a unified `tokenizer.json` --
/// `TransformersRecipe`'s curated fetch (config.json, tokenizer.json,
/// tokenizer_config.json, weights) is actually sufficient for those. `fastvlm`
/// and `qwen3asr` do NOT belong here even though their model crate ALSO reads
/// the directory verbatim: their upstream repos ship only `vocab.json`+
/// `merges.txt` (no `tokenizer.json`), so they need the WHOLE repo, which is
/// what their own `FilesRecipe` rows in `crates/modelstore/src/recipe.rs`
/// fetch instead -- confirmed the hard way for `fastvlm` (a checkpoint
/// converted through this curated path fails at load: "read .../vocab.json:
/// No such file or directory").
///
/// Each entry carries the ROLE name its model crate's resolver expects, since
/// that is not uniform: most want a single `weights` role pointing at the
/// directory, but `deepseek2ocr` composes four checkpoints out of one
/// directory and calls that role `dir`
/// (`deepseek2ocr::spec::Deepseek2ocrSpec`).
const PASSTHROUGH_TRANSFORMERS_FAMILIES: &[(&str, &str)] = &[("qwen3vl", "weights"), ("nemotronasr", "weights"), ("deepseek2ocr", "dir")];

/// The original (and still only) family: an HF `transformers`-shaped repo.
/// Reads `<dir>/config.json` to pick the specific qwen/glm/lfm/gpt importer
/// the same way `modelstore::plan`'s `TransformersRecipe` already gated the
/// download on (`family_of_architecture`) -- one implementation of "which
/// families brain can serve", not a second guess that could drift from the
/// first. The produced card's `id` is overridden to `vendor/repo` (each
/// importer otherwise derives it from the output filename) so the resident
/// registers under the fully-qualified reference the client actually asked
/// for, not `"model.brain"`.
fn convert_transformers(store: &Store, vendor: &str, repo: &str) -> Result<(), String> {
    let dir = store.repo_dir(&ModelRef::new(vendor, repo, None));
    let config_bytes = std::fs::read(dir.join("config.json")).map_err(|e| format!("{vendor}/{repo}: read config.json: {e}"))?;
    let config: serde_json::Value = serde_json::from_slice(&config_bytes).map_err(|e| format!("{vendor}/{repo}: config.json: {e}"))?;
    let arch = brain_modelstore::declared_architecture(&config).ok_or_else(|| format!("{vendor}/{repo}: config.json has no architecture"))?;
    let family = brain_modelstore::family_of_architecture(&arch).ok_or_else(|| format!("{vendor}/{repo}: unsupported architecture {arch:?}"))?;

    // A handful of families read the downloaded HF directory VERBATIM at
    // load time (own config.json + model.safetensors[.index.json] +
    // tokenizer.json, HF tensor names used as-is by the model crate's own
    // loader) -- there is no brain-format tensor rewrite to do, so "finish"
    // is a manifest naming the directory itself as the `weights` role,
    // never `convert_transformers`'s `model.brain.safetensors` step below.
    // Critically this means `remove_upstream_weights` must NOT run either:
    // for every other family the upstream `model.safetensors` is dead
    // weight once the brain-format file exists; for these it IS what gets
    // served.
    if let Some((_, role)) = PASSTHROUGH_TRANSFORMERS_FAMILIES.iter().find(|(f, _)| *f == family) {
        return convert_files(store, vendor, repo, family, &[(role, ".")]);
    }
    // qwen3tts's own repo (`speech_tokenizer/config.json` present) is claimed
    // by the `qwen3tts` `FilesRecipe` ahead of `TransformersRecipe` in
    // `recipes()`'s order, so `family == "qwen3tts"` is never actually
    // reachable here -- `hf: &["Qwen3TTSForConditionalGeneration"]` on its
    // `Arch` row exists for `family_of_architecture` completeness/documentation,
    // not because this path converts it. See `convert_qwen3tts` (dispatched
    // from `convert`'s `"qwen3tts"` recipe-id arm instead).

    let hf_dir = dir.to_str().ok_or_else(|| format!("{vendor}/{repo}: non-UTF8 store path"))?;
    let out_path = dir.join("model.brain.safetensors");
    let out = out_path.to_str().ok_or_else(|| format!("{vendor}/{repo}: non-UTF8 store path"))?;
    let id = format!("{vendor}/{repo}");

    // Every consumer of these three is behind an `import-*` feature. In a build
    // with none of them on, the match below is nothing but diagnostic arms and
    // the bindings are genuinely unused -- said here rather than by prefixing
    // them with `_`, which would also silence a real unused binding later.
    #[cfg(not(any(
        feature = "import-qwen3",
        feature = "import-glmdsa",
        feature = "import-lfm2",
        feature = "import-qwen3omnimoe"
    )))]
    let _ = (hf_dir, out, &id);

    let result = match family {
        #[cfg(feature = "import-qwen3")]
        "qwen3" => qwen3::import::import_as(hf_dir, out, None, Some(&id)),
        #[cfg(not(feature = "import-qwen3"))]
        "qwen3" => Err(no_importer(vendor, repo, "qwen3", "import-qwen3")),
        #[cfg(feature = "import-glmdsa")]
        "glmdsa" => glmdsa::import::import_as(hf_dir, out, Some(&id)),
        #[cfg(not(feature = "import-glmdsa"))]
        "glmdsa" => Err(no_importer(vendor, repo, "glmdsa", "import-glmdsa")),
        #[cfg(feature = "import-lfm2")]
        "lfm2" => lfm2::import::import_as(hf_dir, out, Some(&id)),
        #[cfg(not(feature = "import-lfm2"))]
        "lfm2" => Err(no_importer(vendor, repo, "lfm2", "import-lfm2")),
        // gpt2 is nanogpt-style, trained from scratch -- brain has never had
        // an HF importer for it (unlike glmdsa/qwen3/lfm2, all
        // production-tested). Writing one is real new-crate work, not "wire
        // the dispatch", so this fails cleanly instead of guessing at a
        // Conv1D-transpose import.
        "gpt2" => Err("gpt2 has no HF import path yet -- fetch and convert manually".to_string()),
        // qwen3omnimoe (Qwen3-Omni) is recognized via an exact HF class-name
        // match, so it is never mis-routed to the dense qwen3 importer even
        // though its class name contains "qwen" as a substring. The importer
        // itself streams from the sharded HF dir fine -- what is NOT yet
        // true is that the resulting unified checkpoint is directly loadable
        // by qwen3tts::mtp::MtpModel/mimi::Codec for the Talker/Code2Wav pieces
        // (two open naming gaps); Thinker-only generation is unaffected by
        // either gap.
        #[cfg(feature = "import-qwen3omnimoe")]
        "qwen3omnimoe" => qwen3omnimoe::import::import_as(hf_dir, out, Some(&id)),
        #[cfg(not(feature = "import-qwen3omnimoe"))]
        "qwen3omnimoe" => Err(no_importer(vendor, repo, "qwen3omnimoe", "import-qwen3omnimoe")),
        other => Err(format!("architecture {other:?} matched but has no dispatch arm (bug: family_of_architecture and this match have drifted)")),
    };
    result.map_err(|e| format!("{vendor}/{repo}: convert: {e}"))?;
    // The upstream weights (single model.safetensors, or a model-*-of-*.safetensors
    // shard set + its index) are never read again once model.brain.safetensors
    // exists -- Store::local/scan only ever load BASE_WEIGHTS_FILE -- so keeping
    // them is pure disk waste (often larger than the converted file itself, e.g.
    // a bf16 upstream vs. brain's fp32-only format). Best-effort: a failed
    // cleanup must not fail an otherwise-successful convert.
    remove_upstream_weights(&dir);
    Ok(())
}

/// Qwen3-TTS's finish step: unlike every other `convert_transformers` family,
/// this one produces FOUR converted files (not one `model.brain.safetensors`)
/// via the exact same three importers `brain qwen3tts import` runs by hand
/// (`qwen3tts::import::{import_talker,import_mtp}`, `mimi::import::import`
/// for the codec, `ecapatdnn::import::import` for the speaker encoder) --
/// this is that command's logic, reused, not reimplemented. The speaker
/// encoder is best-effort: CustomVoice/VoiceDesign checkpoints ship none
/// (`tts_model_type != "base"`), so a failure there is a warning, matching
/// `tts_cli.rs::import`'s own policy, never a hard error for the whole fetch.
///
/// Two roles, both kept (no `remove_upstream_weights` -- the downloaded
/// checkpoint dir doubles as `ckpt`, still needed for tokenizer/config at
/// serve time): `ckpt` -> the repo dir itself, `weights_dir` -> the new
/// `brain_tts/` subdirectory holding the four converted files.
#[cfg(not(feature = "import-qwen3tts"))]
fn convert_qwen3tts(store: &Store, vendor: &str, repo: &str) -> Result<(), String> {
    let _ = store;
    Err(no_importer(vendor, repo, "qwen3tts", "import-qwen3tts"))
}

#[cfg(feature = "import-qwen3tts")]
fn convert_qwen3tts(store: &Store, vendor: &str, repo: &str) -> Result<(), String> {
    let dir = store.repo_dir(&ModelRef::new(vendor, repo, None));
    let ckpt = dir.to_str().ok_or_else(|| format!("{vendor}/{repo}: non-UTF8 store path"))?;
    let out_dir = dir.join("brain_tts");
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("{vendor}/{repo}: create {}: {e}", out_dir.display()))?;
    let path = |name: &str| out_dir.join(name).to_str().map(str::to_string).ok_or_else(|| format!("{vendor}/{repo}: non-UTF8 store path"));

    qwen3tts::import::import_talker(ckpt, &path("talker.safetensors")?).map_err(|e| format!("{vendor}/{repo}: import talker: {e}"))?;
    qwen3tts::import::import_mtp(ckpt, &path("mtp.safetensors")?).map_err(|e| format!("{vendor}/{repo}: import mtp: {e}"))?;
    // The speech tokenizer (codec) ships nested inside the Talker's own
    // checkpoint dir, same default `tts_cli.rs::import` uses.
    let codec_ckpt = dir.join("speech_tokenizer");
    let codec_ckpt = codec_ckpt.to_str().ok_or_else(|| format!("{vendor}/{repo}: non-UTF8 store path"))?;
    mimi::import::import(codec_ckpt, &path("codec.safetensors")?).map_err(|e| format!("{vendor}/{repo}: import codec: {e}"))?;
    if let Err(e) = ecapatdnn::import::import(ckpt, &path("speaker.safetensors")?) {
        residency::log::info(&format!("{vendor}/{repo}: import speaker: skipped ({e}) -- fine for CustomVoice/VoiceDesign checkpoints"));
    }

    convert_files(store, vendor, repo, "qwen3tts", &[("ckpt", "."), ("weights_dir", "brain_tts")])
}

/// See [`convert_transformers`]'s cleanup note. Handles both shapes
/// `TransformersRecipe::artifacts` can have downloaded: a single
/// `model.safetensors`, or a `model.safetensors.index.json` + its
/// `model-NNNNN-of-NNNNN.safetensors` shard set.
fn remove_upstream_weights(dir: &Path) {
    let single = dir.join("model.safetensors");
    if single.exists() {
        std::fs::remove_file(&single).ok();
        return;
    }
    std::fs::remove_file(dir.join("model.safetensors.index.json")).ok();
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for name in entries.filter_map(|e| e.ok()).map(|e| e.file_name()) {
        let name = name.to_string_lossy();
        if name.starts_with("model-") && name.ends_with(".safetensors") {
            std::fs::remove_file(dir.join(&*name)).ok();
        }
    }
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

    #[test]
    fn ensure_default_weights_always_check_fetches_converts_and_returns_the_brain_safetensors_path() {
        let (config, weights) = tiny_qwen3_hf_files();
        let mut hub = FakeHub::new();
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "config.json", config);
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "model.safetensors", weights);
        let store = store("loader-supply-test-default-weights-qwen3");

        let got = ensure_default_weights_with("qwen3", &store, &hub, DownloadPolicy::AlwaysCheck).unwrap();
        assert!(got.weights.ends_with("Qwen/Qwen3-0.6B/model.brain.safetensors"), "{}", got.weights);
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
        assert!(got.weights.ends_with("Qwen/Qwen3-0.6B/model.brain.safetensors"), "{}", got.weights);
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
