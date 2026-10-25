// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The global model directory scan — brain's catalog source of truth.
//!
//! The directory itself is found by `loader::model_dir::resolve`
//! (`--models-dir` / `BRAIN_MODELS_DIR`, else `$XDG_DATA_HOME/brain/models`,
//! else `$HOME/.local/share/brain/models` - no absolute-path literal, always
//! computed from env) - that is pure "which directory" logic and lives there;
//! [`discover`] scans it and turns every servable weight file into its
//! OWN [`ResidentModel`], keyed by its
//! model-card id, so a base model and a finetune/LoRA sitting side by side are
//! two distinct selectable models.
//!
//! Non-fatal throughout: an unreadable file, a cardless safetensors, an unknown
//! `card.family`, or a missing directory is logged and skipped — the scan never
//! fails serving. HF-sharded safetensors register once (the index / first shard
//! carries the card; the `-00001-of-*` shards are grouped, not re-registered).
//! GGUF embeds its tokenizer in KV; a safetensors model reuses a sibling
//! `tokenizer.json` (HF convention) and registers even without one.
//!
//! Two layouts are scanned, preferred-first: the `<vendor>/<repo>` store
//! layout ([`brain_modelstore::Store::scan`]) — each entry already knows its
//! own directory, so its sibling `tokenizer.json` is never shared across
//! models — and, for back-compat, the original flat single-level directory of
//! `*.safetensors`/`*.gguf` files. A flat-layout hit logs a one-time-per-process
//! warning recommending migration to the store layout.
//!
//! [`resident_for`] dispatches by `card.family`: `qwen`, `gpt`, `glm`, `lfm`,
//! `yolo`, `depth`, `qwen35` today. A raw `.gguf` reaches that dispatch at all
//! only for the families whose resident reads GGUF itself
//! ([`family_reads_gguf`]) - every other `.gguf` is answered BEFORE the match
//! with [`gguf_advice`], because a GGUF card's family is its
//! `general.architecture` verbatim and so may spell any family's name. A
//! `.gguf` whose architecture has a registered importer
//! ([`crate::gguf_import`]) but is not itself servable is reported with the
//! one-time `brain import-gguf` command that makes it servable - see that
//! module's doc for why the scan does not convert it in place.
//!
//! A released `qwen35` GGUF is the exception to both: it is served straight
//! from the file, INT8 and layer-sharded across the cards the process may use
//! (`catalog::resident_qwen35::gguf_resident`), each file its own model under
//! its store id. Those residents span several devices, so they come back in
//! [`Discovered::multi`] for `Executor::register_multi` rather than among the
//! single-device residents. A file that could never load (no card to place it
//! on, weights that do not fit, no embedded tokenizer) is not listed at all;
//! the reason is logged.
//!
//! Each family's own checkpoint-writing code must
//! attach a real [`ModelCard`] (`checkpoint::save_carded`, not the plain
//! `checkpoint::save`, which never carries one) for a checkpoint to be
//! reachable here at all — a `from_card` resident constructor alone is not
//! enough (`gpt`/`glm` had working dispatch arms for a while with no carded
//! checkpoint ever able to reach them).
//!
//! A model that is FOUR distinct-role files (DiT/VAE/text-encoder/tokenizer)
//! has no single weights path, so it registers through
//! [`resident_for_compound`] instead, keyed on the roles a
//! `brain.manifest.json` names: `zimage`, `wan` and `minimaxh3` (five roles)
//! today. `flux2` is the same
//! shape and still has no arm there -- it is reachable only through its own
//! `BRAIN_FLUX2_*` variables.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Once};

use catalog::adapter_registry::{AdapterBankSettings, AdapterHost, AdapterTargets};
use catalog::adapter_release::verify_adapter;
use catalog::resident_llm::QwenResident;
use catalog::resident_qwen35::Qwen35GgufServed;
use checkpoint::st::{self, ModelCard};
use residency::multi::MultiDeviceResidentModel;
use residency::ResidentModel;

/// A [`LocalModel`](brain_modelstore::LocalModel) [`discover`] could not turn
/// into a resident -- today, only a compound (`roles.is_some()`) model,
/// since a single-file model's failure stays the pre-existing
/// `register`-internal `eprintln` (see that function's doc). Named by repo
/// dir and family so a caller can act on it instead of it vanishing into
/// stderr.
#[derive(Debug, Clone)]
pub struct DiscoveryError {
    pub dir: PathBuf,
    pub family: String,
    pub reason: String,
}

/// How a scan builds the residents it finds: the per-family serving knobs, and
/// the cards a model that spans devices may be placed on.
#[derive(Clone, Default)]
pub struct ScanSettings {
    /// The Qwen3 paged engine's knobs.
    pub qwen: catalog::resident_llm::QwenServeConfig,
    /// The Qwen3.8 knobs: `fp32` sizes the brain-checkpoint resident, `gguf`
    /// the released-GGUF residents.
    pub qwen35: catalog::resident_qwen35::Qwen35ServeConfig,
    /// `(index, TOTAL bytes)` of every GPU the process may use - already
    /// narrowed by `--device`. Empty means no model that needs a GPU can be
    /// served, and none is listed.
    pub gpus: Vec<(u32, u64)>,
    /// Bytes kept free on each of [`Self::gpus`].
    pub reserved: u64,
}

/// What [`discover`] found.
pub struct Discovered {
    /// Single-device residents, for `Executor::start`/`register`.
    pub residents: Vec<Arc<dyn ResidentModel>>,
    /// Residents that occupy several devices at once, for
    /// `Executor::register_multi` (a plain `register` would budget only one of
    /// the cards they occupy). Every entry places on at least one budgeted card.
    pub multi: Vec<Arc<dyn MultiDeviceResidentModel>>,
    /// The Qwen3 bases among `residents`, as concrete handles: adapters may be
    /// offered on them, and an env-selected resident serving the same
    /// checkpoint is merged into them (see `build_executor`).
    pub qwen: Vec<Arc<QwenResident>>,
    /// The released Qwen3.8 GGUFs among `multi`, as concrete handles: adapters
    /// are offered on them ([`catalog::adapter_registry::AdapterHost`]).
    pub qwen35: Vec<Arc<Qwen35GgufServed>>,
    /// Every [`DiscoveryError`] a compound model produced -- a partial scan
    /// still serves what it can.
    pub errors: Vec<DiscoveryError>,
}

/// Families whose catalog entry registers their resident itself (a `spec`
/// that resolves the checkpoint, and a `resident`), which the directory scan
/// therefore leaves alone.
fn catalog_served_families() -> std::collections::BTreeSet<&'static str> {
    catalog::models().into_iter().filter(|e| e.resident.is_some()).filter_map(|e| e.spec.map(|(family, _)| family)).collect()
}

/// Scan `dir` and build one resident per discovered model-card id. Deduplicates
/// by id (first wins, store layout before flat layout). See the module docs
/// for the layout precedence and the skip/warn policy.
pub fn discover(dir: &Path, settings: &ScanSettings) -> Discovered {
    let mut errors: Vec<DiscoveryError> = Vec::new();

    let mut locals = brain_modelstore::Store::new(dir.to_path_buf()).scan();
    locals.sort_by(|a, b| a.weights.cmp(&b.weights)); // deterministic catalog order
    let (adapters, bases): (Vec<_>, Vec<_>) = locals.iter().partition(|l| l.adapter.is_some());
    let mut scan = Scan::new(settings, &adapters);

    let released_per_repo = released_qwen35_per_repo(&bases);
    let catalog_served = catalog_served_families();
    for local in bases {
        if local.card.as_ref().is_some_and(|c| catalog_served.contains(c.family.as_str())) {
            // Registered by its own catalog entry from the resolved assembly;
            // building it here too would double it, or report it unservable.
            continue;
        }
        if local.roles.is_some() {
            // A compound model has no single weights path (`local.weights` is
            // the manifest file itself) -- `register`/`resident_for` is the
            // single-file dispatcher and must never see one of these; go
            // through the compound-aware dispatch instead.
            register_compound(local, &mut scan, &mut errors);
            continue;
        }
        if is_released_qwen35(local) {
            let shares_repo = released_per_repo.get(&local.dir).copied().unwrap_or(0) > 1;
            let card = local.card.clone().map(|c| ModelCard { id: released_qwen35_id(&local.reference, shares_repo), ..c });
            scan.register(&local.weights, &card, &local.dir);
            continue;
        }
        scan.register(&local.weights, &local.card, &local.dir);
    }

    discover_flat(dir, &mut scan);
    // Adapters last: each is a view over a Qwen3 base the passes above built.
    for local in adapters {
        scan.register_adapter(local);
    }
    let qwen35 = scan.qwen35_bases.values().cloned().collect();
    let qwen = scan.qwen_bases.values().cloned().collect();
    Discovered { residents: scan.out, multi: scan.multi, qwen, qwen35, errors }
}

/// Whether `local` is a released Qwen3.8 GGUF: served from the file itself by
/// the multi-card INT8 resident, not converted and not a safetensors engine.
fn is_released_qwen35(local: &brain_modelstore::LocalModel) -> bool {
    local.weights.to_string_lossy().ends_with(".gguf") && local.card.as_ref().is_some_and(|c| brain_family(&c.family) == "qwen35")
}

/// How many released Qwen3.8 GGUFs each repo directory holds.
fn released_qwen35_per_repo(bases: &[&brain_modelstore::LocalModel]) -> BTreeMap<PathBuf, usize> {
    let mut counts = BTreeMap::new();
    for local in bases.iter().filter(|l| is_released_qwen35(l)) {
        *counts.entry(local.dir.clone()).or_insert(0) += 1;
    }
    counts
}

/// The id a released GGUF is served under: its repo, `<vendor>/<repo>`, which
/// is what a client knows the download by. A repo that holds several quants
/// (`Q8_0.gguf`, `Q4_K_M.gguf`) would otherwise name them all alike, so each
/// then carries its quant: `<vendor>/<repo>-<QUANT>`, the reference grammar
/// `brain pull` writes.
fn released_qwen35_id(reference: &brain_modelref::ModelRef, shares_repo: bool) -> String {
    if shares_repo {
        reference.to_string()
    } else {
        reference.base().to_string()
    }
}

/// What one [`discover`] accumulates: the residents, and the settings the
/// next resident is built with.
struct Scan {
    /// Card ids registered so far (first wins).
    seen: BTreeSet<String>,
    out: Vec<Arc<dyn ResidentModel>>,
    /// The Qwen3 bases built so far, by card id: the adapter found beside one
    /// is served as a view over it ([`Scan::register_adapter`]).
    qwen_bases: BTreeMap<String, Arc<QwenResident>>,
    /// The released Qwen3.8 GGUFs built so far, by served id: an adapter found
    /// beside one is served as a view over it ([`Scan::register_adapter`]).
    qwen35_bases: BTreeMap<String, Arc<Qwen35GgufServed>>,
    /// Multi-device residents ([`Discovered::multi`]).
    multi: Vec<Arc<dyn MultiDeviceResidentModel>>,
    /// The adapter bank every released Qwen3.8 GGUF is built with: the
    /// operator's, else sized to hold every Qwen3.8 adapter the store has,
    /// else none.
    qwen35_bank: Option<AdapterBankSettings>,
    settings: ScanSettings,
    /// The adapter bank each base that has adapters in the store is built
    /// with, by base card id.
    banks: BTreeMap<String, AdapterBankSettings>,
}

impl Scan {
    fn new(settings: &ScanSettings, adapters: &[&brain_modelstore::LocalModel]) -> Scan {
        let qwen_cfg = settings.qwen.clone();
        let mut banks: BTreeMap<String, AdapterBankSettings> = BTreeMap::new();
        let mut qwen35_bank = qwen_cfg.adapter_bank.clone();
        // What the store's adapters may grow. A bank no flag sized is theirs to
        // size; one sized by `--adapter-slots` / `--adapter-rank-cap` alone keeps
        // those but still takes the targets the adapters need, since the
        // operator named none; `--adapter-targets` makes the bank the operator's.
        let widen = Widen { rank: qwen_cfg.adapter_bank.is_none(), targets: qwen_cfg.adapter_bank.is_none() || !qwen_cfg.adapter_targets_named };
        for local in adapters {
            if let (true, Some(card)) = (widen.any(), local.card.as_ref().filter(|c| brain_family(&c.family) == "qwen35")) {
                qwen35_bank = Some(widen_to_hold(qwen35_bank.take().unwrap_or_default(), card, widen));
                continue;
            }
            let Some((base, card)) = local.card.as_ref().and_then(|c| Some((base_id_of(&c.id)?, c))) else { continue };
            // The bank an operator sized with flags is theirs; otherwise it
            // is sized to hold every adapter the store has for the base, so
            // an adapter on disk is never refused for a limit nobody chose.
            let bank = banks.entry(base).or_insert_with(|| qwen_cfg.adapter_bank.clone().unwrap_or_default());
            if widen.any() {
                *bank = widen_to_hold(bank.clone(), card, widen);
            }
        }
        Scan { seen: BTreeSet::new(), out: Vec::new(), qwen_bases: BTreeMap::new(), qwen35_bases: BTreeMap::new(), multi: Vec::new(), qwen35_bank, settings: settings.clone(), banks }
    }

    /// The serving config for the Qwen3 base `id`: the operator's, plus a
    /// bank when the store holds adapters for it.
    fn qwen_cfg_for(&self, id: &str) -> catalog::resident_llm::QwenServeConfig {
        let card_budget_bytes = self.settings.qwen.card_budget_bytes.or_else(|| self.settings.gpus.iter().map(|&(_, total)| total.saturating_sub(self.settings.reserved)).max());
        catalog::resident_llm::QwenServeConfig { adapter_bank: self.banks.get(id).cloned().or_else(|| self.settings.qwen.adapter_bank.clone()), card_budget_bytes, ..self.settings.qwen.clone() }
    }

    /// Serve the adapter `local` as a model of its own over the Qwen3 base it
    /// rides on. An adapter whose base is not served, that does not verify
    /// against it, or that the bank cannot hold is skipped with the reason.
    fn register_adapter(&mut self, local: &brain_modelstore::LocalModel) {
        let (Some(card), Some(path)) = (local.card.as_ref(), local.adapter.as_ref()) else { return };
        let shown = path.display();
        if let Err(e) = checkpoint::license::redistributable(card.license.as_deref()) {
            eprintln!("brain: skip {shown} ({e})");
            return;
        }
        let named = base_id_of(&card.id);
        let on_qwen = named.as_ref().and_then(|b| self.qwen_bases.get(b).cloned());
        let on_qwen35 = named.as_ref().and_then(|b| self.qwen35_bases.get(b).cloned()).or_else(|| self.released_base_of(card));
        if on_qwen.is_none() && on_qwen35.is_none() {
            eprintln!("brain: skip {shown} (its base is not served as a Qwen3 or Qwen3.8 model, so adapter {} has nothing to ride on)", card.id);
            return;
        }
        if !self.seen.insert(card.id.clone()) {
            eprintln!("brain: skip {shown} (duplicate model id '{}')", card.id);
            return;
        }
        let kept = match (on_qwen, on_qwen35) {
            (Some(base), _) => verify_adapter(path, &base.served_base()).and_then(|loaded| base.adapter_view(loaded)).map(|view| self.out.push(Arc::new(view))),
            (None, Some(base)) => verify_adapter(path, &base.served_base()).and_then(|loaded| base.adapter_view(loaded)).map(|view| self.multi.push(Arc::new(view))),
            (None, None) => unreachable!("checked above"),
        };
        if let Err(e) = kept {
            self.seen.remove(&card.id);
            eprintln!("brain: skip {shown} ({e})");
        }
    }

    /// The released Qwen3.8 GGUF a carded adapter rides on when its id does
    /// not name one exactly (a quant-qualified id names the file, a served id
    /// of a repo with one quant does not): the GGUF whose served id is the
    /// card's base, else the one that quantises the full-precision release the
    /// card names. `None` when none matches or several do (the card's base
    /// must then name the one it rides on).
    fn released_base_of(&self, card: &ModelCard) -> Option<Arc<Qwen35GgufServed>> {
        let named = card.adapter.as_ref()?.base.as_deref()?;
        if let Some(served) = self.qwen35_bases.get(named) {
            return Some(served.clone());
        }
        let mut matching = self.qwen35_bases.values().filter(|b| b.served_base().quantizes(named));
        let first = matching.next()?.clone();
        matching.next().is_none().then_some(first)
    }

    /// Dispatch one carded file to its family's resident and keep it (deduped
    /// by id). Adapters are not registered here: see [`Self::register_adapter`].
    fn register(&mut self, weights: &Path, card: &Option<ModelCard>, dir: &Path) {
        let wp = weights.to_string_lossy();
        let card = match card {
            Some(c) => c,
            None => {
                eprintln!("brain: skip {wp} (no model card)");
                return;
            }
        };
        if let Err(e) = checkpoint::license::redistributable(card.license.as_deref()) {
            // Publishing a model into the served catalog IS redistribution: every
            // registered model is reachable over the API. The refusal names the
            // file, because the card that carries the licence is inside it.
            eprintln!("brain: skip {wp} ({e})");
            return;
        }
        if !self.seen.insert(card.id.clone()) {
            eprintln!("brain: skip {wp} (duplicate model id '{}')", card.id);
            return;
        }
        if wp.ends_with(".gguf") && brain_family(&card.family) == "qwen35" {
            self.register_released_qwen35(&wp, &card.id);
            return;
        }
        // GGUF carries its tokenizer in KV; a safetensors model reuses a sibling
        // tokenizer.json (HF convention). Absent it, register anyway (limited chat).
        let sibling = dir.join("tokenizer.json");
        let tokenizer: Option<String> = if wp.ends_with(".gguf") {
            None
        } else if sibling.exists() {
            Some(sibling.to_string_lossy().into_owned())
        } else {
            eprintln!("brain: {wp}: no sibling tokenizer.json ({} chat-templating may be limited)", card.id);
            None
        };

        match resident_for(&wp, card, tokenizer.as_deref(), self.qwen_cfg_for(&card.id), self.settings.qwen35.fp32) {
            Some(built) => self.keep(&card.id, built),
            None => {
                self.seen.remove(&card.id); // free the id if the family declined
            }
        }
    }

    /// Serve the released Qwen3.8 GGUF at `weights` as the multi-card model
    /// `id`, or leave it out with the reason: only a model that can load is
    /// listed, and the id is freed for a later duplicate to claim.
    fn register_released_qwen35(&mut self, weights: &str, id: &str) {
        let ScanSettings { gpus, reserved, qwen35, .. } = &self.settings;
        match catalog::resident_qwen35::gguf_resident_banked(weights, id, gpus, *reserved, &qwen35.gguf, self.qwen35_bank.clone()) {
            Ok(resident) => {
                for line in resident.chosen_defaults() {
                    eprintln!("brain: {id}: default {line}");
                }
                let resident = Arc::new(resident);
                self.qwen35_bases.insert(id.to_string(), resident.clone());
                self.multi.push(resident);
            }
            Err(reason) => {
                eprintln!("brain: skip {id} ({weights}: {reason})");
                self.seen.remove(id);
            }
        }
    }

    fn keep(&mut self, id: &str, built: Built) {
        if let Built::Qwen(qwen) = &built {
            self.qwen_bases.insert(id.to_string(), qwen.clone());
        }
        self.out.push(built.erased());
    }
}

/// The card id of the base an adapter card id (`<base>:<owner>:<name>:<tag>`)
/// rides on.
fn base_id_of(adapter_id: &str) -> Option<String> {
    let r = brain_modelref::ModelRef::parse(adapter_id).ok()?;
    r.adapter()?;
    Some(r.without_adapter().to_string())
}

/// Which settings of a bank the adapters of the store may grow.
#[derive(Clone, Copy)]
struct Widen {
    rank: bool,
    targets: bool,
}

impl Widen {
    fn any(self) -> bool {
        self.rank || self.targets
    }
}

/// `bank`, grown just enough to hold the adapter `card` describes: its rank
/// and the linears it targets, as its card records them, for what `widen` lets
/// the store grow.
fn widen_to_hold(bank: AdapterBankSettings, card: &ModelCard, widen: Widen) -> AdapterBankSettings {
    let Some(adapter) = card.adapter.as_ref() else { return bank };
    let rank_cap = if widen.rank { bank.rank_cap.max(adapter.rank.unwrap_or(0)) } else { bank.rank_cap };
    let named = adapter.targets.as_ref().and_then(|t| AdapterTargets::parse(&t.join(",")).ok());
    let targets = match named {
        Some(named) if widen.targets => AdapterTargets::union(bank.targets.clone(), named),
        _ => bank.targets.clone(),
    };
    AdapterBankSettings { rank_cap, targets, ..bank }
}

/// A resident the scan built: the concrete handle of a Qwen3 base is kept,
/// because an adapter found beside it is served over it.
enum Built {
    Qwen(Arc<QwenResident>),
    Other(Arc<dyn ResidentModel>),
}

impl Built {
    fn erased(&self) -> Arc<dyn ResidentModel> {
        match self {
            Built::Qwen(q) => q.clone(),
            Built::Other(o) => o.clone(),
        }
    }
}

/// [`register`]'s dedup-then-construct-then-free-on-failure dance, for a
/// compound model dispatched through [`resident_for_local`] instead of
/// [`resident_for`] -- kept a separate function (not folded into `register`,
/// which takes a bare weights path a compound model doesn't have) so the two
/// dispatchers stay independently readable.
fn register_compound(local: &brain_modelstore::LocalModel, scan: &mut Scan, errors: &mut Vec<DiscoveryError>) {
    let family = local.card.as_ref().map(|c| c.family.clone()).unwrap_or_default();
    let id = match local.card.as_ref() {
        Some(c) => c.id.clone(),
        None => {
            errors.push(DiscoveryError { dir: local.dir.clone(), family, reason: "no model card".to_string() });
            return;
        }
    };
    if let Err(e) = checkpoint::license::redistributable(local.card.as_ref().and_then(|c| c.license.as_deref())) {
        errors.push(DiscoveryError { dir: local.dir.clone(), family, reason: e });
        return;
    }
    if !scan.seen.insert(id.clone()) {
        errors.push(DiscoveryError { dir: local.dir.clone(), family, reason: format!("duplicate model id '{id}'") });
        return;
    }
    match build_for_local(local, scan.qwen_cfg_for(&id), scan.settings.qwen35.fp32) {
        Ok(built) => scan.keep(&id, built),
        Err(reason) => {
            scan.seen.remove(&id); // free the id if the family declined, same as `register`
            errors.push(DiscoveryError { dir: local.dir.clone(), family, reason });
        }
    }
}

static FLAT_LAYOUT_WARNING: Once = Once::new();

/// The original single-level flat scan, predating the `<vendor>/<repo>` store
/// layout. Kept for back-compat; a hit warns once per process.
fn discover_flat(dir: &Path, scan: &mut Scan) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("brain: models dir {} not scanned ({e})", dir.display());
            return;
        }
    };

    // Enumerate weight files. Plain `*.safetensors` and `*.gguf` are single
    // models; a sharded `*-NNNNN-of-MMMMM.safetensors` or split
    // `*-NNNNN-of-MMMMM.gguf` set groups by base and registers once (the
    // lowest-index part carries the card - for safetensors, the
    // `.index.json` handle is preferred when present; for GGUF, part 1's
    // path is what `MmapGguf::open` needs to find every sibling). Everything
    // else (incl. bare `.index.json`, `tokenizer.json`) is ignored as a
    // model file. `checkpoint::split::split_name` is the one parser for
    // both extensions' identical `-NNNNN-of-MMMMM` convention.
    let mut singles: Vec<PathBuf> = Vec::new();
    let mut shards: BTreeMap<String, Vec<(u32, PathBuf)>> = BTreeMap::new();
    let mut gguf_splits: BTreeMap<String, Vec<(u32, PathBuf)>> = BTreeMap::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let fname = match path.file_name().and_then(|s| s.to_str()) {
            Some(f) => f.to_string(),
            None => continue,
        };
        if fname.ends_with(".safetensors") {
            match checkpoint::split::split_name(&fname, "safetensors") {
                Some((base, part, _count, _width)) => shards.entry(base.to_string()).or_default().push((part, path)),
                None => singles.push(path),
            }
        } else if fname.ends_with(".gguf") {
            match checkpoint::split::split_name(&fname, "gguf") {
                Some((base, part, _count, _width)) => gguf_splits.entry(base.to_string()).or_default().push((part, path)),
                None => singles.push(path),
            }
        }
    }

    if !singles.is_empty() || !shards.is_empty() || !gguf_splits.is_empty() {
        FLAT_LAYOUT_WARNING.call_once(|| {
            eprintln!(
                "brain: {} holds models directly (the flat legacy layout); \
                 migrate to <vendor>/<repo>/ -- \
                 this warning prints once per process",
                dir.display()
            );
        });
    }

    // Sorted for a deterministic catalog order.
    singles.sort();
    for path in &singles {
        scan.register(path, &card_of(path), dir);
    }
    for (base, mut group) in shards {
        group.sort_by_key(|(idx, _)| *idx);
        let first = &group[0].1; // lowest-index shard carries the card
        // Prefer the `<base>.safetensors.index.json` handle if it exists.
        let index = dir.join(format!("{base}.safetensors.index.json"));
        let handle = if index.exists() { index } else { first.clone() };
        scan.register(&handle, &card_of(first), dir);
    }
    for (_base, mut group) in gguf_splits {
        group.sort_by_key(|(part, _)| *part);
        let part1 = &group[0].1; // MmapGguf::open finds every sibling from part 1's path
        scan.register(part1, &card_of(part1), dir);
    }
}

/// Read a file's card: `read_card` (metadata-only) for safetensors, the KV-
/// synthesized card for GGUF (always `Some`). `None` on I/O error or a cardless
/// safetensors.
fn card_of(path: &Path) -> Option<ModelCard> {
    let p = path.to_string_lossy();
    if p.ends_with(".gguf") {
        match checkpoint::weightio::WeightReader::open(&p) {
            Ok(r) => r.card(),
            Err(e) => {
                eprintln!("brain: skip {p} ({e})");
                None
            }
        }
    } else {
        match st::read_card(&p) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("brain: skip {p} ({e})");
                None
            }
        }
    }
}

/// GGUF cards carry `general.architecture` verbatim as `family` (the KV's own
/// reported name, e.g. llama.cpp's `"qwen3"`) — brain's own family names
/// (`catalog::resident_llm`) differ, so alias the ones brain's implementation
/// actually matches. brain's `qwen` crate targets the Qwen3 architecture
/// specifically (QK-norm + GQA); a plain `"qwen2"` GGUF is a different
/// architecture and stays unaliased rather than silently mis-served.
fn brain_family(reported: &str) -> &str {
    match reported {
        "qwen3" => "qwen",
        other => other,
    }
}

/// Whether a family's resident can open a raw `.gguf` weights path ITSELF.
///
/// The opt-in half of [`resident_for`]'s GGUF gate, and deliberately an
/// allowlist rather than a list of exclusions: a GGUF card's `family` is
/// `general.architecture` verbatim (see [`card_of`]), so ANY architecture
/// string is free to collide with ANY family name here, and the safe answer
/// for a family nobody has checked is "no" - the caller then prints the
/// `brain import-gguf` advice instead of handing GGUF bytes to a resident
/// that cannot parse them.
///
/// The three listed all reach their weights through
/// `checkpoint::weightio::WeightReader`, which sniffs the container and reads
/// safetensors and GGUF alike; `qwen` additionally has a documented `.gguf`
/// decode path and an embedded-tokenizer fallback
/// (`catalog::resident_llm::QwenResident::activate`). Every other family loads
/// through `checkpoint::load` / `checkpoint::torchpt`, which are
/// safetensors-only: `qwen35`/`qwen35moe`'s serving `Engine`, yolo's
/// `yolov8::Yolo::load`, depth's `zipdepth::import::load`.
fn family_reads_gguf(family: &str) -> bool {
    matches!(family, "gpt" | "glm" | "qwen")
}

/// What to do with a raw `.gguf` no family resident can open - the one line
/// [`resident_for`] logs before declining it, keyed on the file's own
/// `general.architecture`.
///
/// A `.gguf` whose architecture IS in the importer table is not a dead end -
/// name what to do with it instead of reporting it as unservable. Which advice
/// is right comes from the table's own direct-load column: an architecture
/// that streams its GGUF at inference has no conversion to run, and telling
/// someone to convert one would send them to a command that refuses. See
/// `crate::gguf_import`'s module doc for why discovery does not run a
/// conversion itself.
fn gguf_advice(weights: &str, architecture: &str) -> String {
    match crate::gguf_import::importer_for(architecture) {
        Some(entry) if entry.loads_directly() => {
            format!("{weights}: GGUF architecture '{architecture}' loads directly through its own verb, not the model-dir scan")
        }
        Some(_) => format!("{weights}: GGUF architecture '{architecture}' needs a one-time conversion -- run `brain import-gguf {weights}`"),
        None => format!("family '{architecture}' not servable from the model dir yet"),
    }
}

/// Construct the resident matching `card.family` (normalized via
/// [`brain_family`]), or log + `None` for an unknown / not-yet-dispatchable
/// family.
fn resident_for(weights: &str, card: &ModelCard, tokenizer: Option<&str>, qwen_cfg: catalog::resident_llm::QwenServeConfig, qwen35_sizing: catalog::resident_qwen35::Qwen35Sizing) -> Option<Built> {
    let family = brain_family(&card.family);
    // The GGUF gate, BEFORE the family match: a GGUF card's family is its
    // `general.architecture` copied verbatim, so an architecture string is
    // free to spell a brain-native family name and fall into that family's arm
    // below - handing a resident that only reads safetensors a file it cannot
    // parse, which surfaces as a low-level error at activate() instead of the
    // actionable import advice. Gating here rather than per-arm is what makes
    // the whole collision class impossible: an arm is reached by a `.gguf`
    // only if its family opted in via `family_reads_gguf`.
    if weights.ends_with(".gguf") && family == "qwen35" {
        eprintln!("brain: skip {} ({weights}: a released qwen35 GGUF is placed across the cards when `brain serve` starts; restart it to serve this one)", card.id);
        return None;
    }
    if weights.ends_with(".gguf") && !family_reads_gguf(family) {
        eprintln!("brain: skip {} ({})", card.id, gguf_advice(weights, &card.family));
        return None;
    }
    if family == "qwen" {
        return Some(Built::Qwen(Arc::new(QwenResident::from_card_configured(weights, card, tokenizer, qwen_cfg))));
    }
    let other: Option<Arc<dyn ResidentModel>> = match family {
        "gpt" => Some(Arc::new(catalog::resident_llm::GptResident::from_card(weights, card, tokenizer))),
        "glm" => Some(Arc::new(catalog::resident_llm::GlmResident::from_card(weights, card, tokenizer))),
        "lfm" => match catalog::resident_lfm::LfmResident::from_card(weights, card, tokenizer) {
            Ok(l) => Some(Arc::new(l)),
            Err(e) => {
                eprintln!("brain: skip {} ({e})", card.id);
                None
            }
        },
        "ouro" => match catalog::resident_ouro::OuroResident::from_card(weights, card, tokenizer) {
            Ok(o) => Some(Arc::new(o.with_sizing(qwen35_sizing.ctx, catalog::resident_ouro::DEFAULT_BLOCK, qwen35_sizing.max_batch))),
            Err(e) => {
                eprintln!("brain: skip {} ({e})", card.id);
                None
            }
        },
        "huginn" => match catalog::resident_huginn::HuginnResident::from_card(weights, card, tokenizer) {
            Ok(h) => Some(Arc::new(h.with_sizing(qwen35_sizing.ctx, catalog::resident_huginn::DEFAULT_BLOCK, qwen35_sizing.max_batch))),
            Err(e) => {
                eprintln!("brain: skip {} ({e})", card.id);
                None
            }
        },
        "yolo" => Some(Arc::new(catalog::resident_yolo::YoloResident::from_card(weights, card, tokenizer))),
        "depth" => Some(Arc::new(catalog::resident_depth::DepthResident::from_card(weights, card, tokenizer))),
        // What `crate::gguf_import`'s registry-driven conversion stamps on the
        // checkpoint it writes - so an imported GGUF is picked up by the very
        // next scan with no env vars and no per-model wiring.
        "qwen35moe" => match catalog::resident_qwen35moe::Qwen35Resident::from_card(weights, card, tokenizer) {
            Ok(q) => Some(Arc::new(q)),
            Err(e) => {
                eprintln!("brain: skip {} ({e})", card.id);
                None
            }
        },
        // What `Qwen35::save` (`crates/qwen35`, the dense sibling) stamps on
        // its own checkpoints - a distinct family from "qwen35moe" above
        // (a pre-existing collision between the two crates, fixed in the
        // same commit that added this arm).
        "qwen35" if qwen35_sizing.refused => None,
        "qwen35" => match catalog::resident_qwen35::Qwen35Resident::from_card_sized(weights, card, tokenizer, qwen35_sizing) {
            Ok(q) => Some(Arc::new(q)),
            Err(e) => {
                eprintln!("brain: skip {} ({e})", card.id);
                None
            }
        },
        // Only a brain-native (safetensors) checkpoint reaches here: the gate
        // above already answered every `.gguf` whose family has no arm.
        other => {
            eprintln!("brain: skip {} (family '{other}' not servable from the model dir yet)", card.id);
            None
        }
    };
    other.map(Built::Other)
}

/// [`resident_for`] over a [`brain_modelstore::LocalModel`] -- the same
/// family dispatch [`discover`] uses for the store's scan, reused by the
/// auto-fetch supplier (`supply` in `crates/cli`) so there is exactly one
/// "weights file -> resident" mapping regardless of how the file was found.
/// A compound (multi-file, `local.roles.is_some()`) model has no single
/// weights path, so it dispatches separately, by family, before the
/// single-path case below even applies.
///
/// `Err` names WHY, rather than printing and returning `None` -- both
/// callers decide for themselves what to do with the reason (`discover`
/// collects it as a [`DiscoveryError`]; `supply::StoreSupplier::ensure`
/// folds it into the error it already returns to ITS caller).
///
/// An adapter is not a resident of its own: it is served as a view over the
/// base it rides on (`discover`), so `local` must be a base.
pub fn resident_for_local(local: &brain_modelstore::LocalModel, qwen_cfg: catalog::resident_llm::QwenServeConfig, qwen35_sizing: catalog::resident_qwen35::Qwen35Sizing) -> Result<Arc<dyn ResidentModel>, String> {
    build_for_local(local, qwen_cfg, qwen35_sizing).map(|built| built.erased())
}

fn build_for_local(local: &brain_modelstore::LocalModel, qwen_cfg: catalog::resident_llm::QwenServeConfig, qwen35_sizing: catalog::resident_qwen35::Qwen35Sizing) -> Result<Built, String> {
    let card = local.card.as_ref().ok_or_else(|| "no model card".to_string())?;
    if local.adapter.is_some() {
        return Err(format!("{}: an adapter is served by the base it rides on, not built as a model of its own", card.id));
    }
    if let Some(roles) = &local.roles {
        return resident_for_compound(card, roles, qwen_cfg, qwen35_sizing);
    }
    let weights = local.weights.to_str().ok_or_else(|| "weights path is not valid UTF-8".to_string())?;
    let tokenizer = local.tokenizer.as_deref().and_then(|p| p.to_str());
    resident_for(weights, card, tokenizer, qwen_cfg, qwen35_sizing).ok_or_else(|| format!("family '{}' not servable from the model dir yet", card.family))
}

/// The compound-model counterpart of [`resident_for`]: family dispatch keyed
/// on named roles (a directory or file each) rather than one weights path.
///
/// A checkpoint of a family whose model reads a downloaded Hugging Face
/// directory (the qwen3 decoder and its `llama`/`qwen2` rows, glmdsa, lfm2)
/// is served straight from the directory its `weights` role names.
fn resident_for_compound(card: &ModelCard, roles: &std::collections::BTreeMap<String, PathBuf>, qwen_cfg: catalog::resident_llm::QwenServeConfig, qwen35_sizing: catalog::resident_qwen35::Qwen35Sizing) -> Result<Built, String> {
    let implementation = brain_arch::by_id(&card.family).map(|a| a.implementation().id);
    if matches!(implementation, Some("qwen3" | "glmdsa" | "lfm2" | "qwen3omnimoe")) {
        let dir = roles.get("weights").ok_or("compound manifest missing role \"weights\"")?;
        let weights = dir.to_str().ok_or("weights path is not valid UTF-8")?;
        let tokenizer = dir.join("tokenizer.json");
        return match implementation {
            // An embedding checkpoint of the family is served for embeddings, not chat.
            Some("qwen3") if catalog::resident_qwen_embed::is_embedding_id(&card.id) => Ok(Built::Other(catalog::resident_qwen_embed::embedding_resident(weights, card, tokenizer.to_str()))),
            Some("qwen3") => Ok(Built::Qwen(Arc::new(QwenResident::from_card_configured(weights, card, tokenizer.to_str(), qwen_cfg)))),
            Some("glmdsa") => Ok(Built::Other(Arc::new(catalog::resident_llm::GlmResident::from_card(weights, card, None)))),
            Some("lfm2") => Ok(Built::Other(Arc::new(catalog::resident_lfm::LfmResident::from_card(weights, card, tokenizer.to_str())?))),
            // Sharded across several GPUs, so it is registered on its own
            // multi-device path rather than from the model directory.
            _ => Err(format!("{} is served across GPUs: set BRAIN_QWEN3OMNIMOE_INT8_CHECKPOINT={weights}", card.id)),
        };
    }
    // A Qwen3.5/3.6 MoE release is served from its published directory, every tensor in the format it is stored in.
    if implementation == Some("qwen35moe") {
        let dir = roles.get("weights").and_then(|p| p.to_str()).ok_or("compound manifest missing role \"weights\"")?;
        return catalog::resident_qwen35moe::Qwen35Resident::from_hf_dir(dir, card).map(|q| Built::Other(Arc::new(q)));
    }
    // A pulled YOLOv8 is served from its downloaded `.pt`.
    if card.family == "yolo" {
        let pt = roles.get("weights").and_then(|p| p.to_str()).ok_or("compound manifest missing role \"weights\"")?;
        return Ok(Built::Other(Arc::new(catalog::resident_yolo::YoloResident::from_card(pt, card, None))));
    }
    let other: Result<Arc<dyn ResidentModel>, String> = match brain_family(&card.family) {
        // A looped decoder's HF directory is the model: config + tokenizer +
        // safetensors, read as pulled.
        "ouro" => {
            let dir = roles.get("weights").and_then(|p| p.to_str()).ok_or("compound manifest missing role \"weights\"")?;
            catalog::resident_ouro::OuroResident::from_card(dir, card, None)
                .map(|o| Arc::new(o.with_sizing(qwen35_sizing.ctx, catalog::resident_ouro::DEFAULT_BLOCK, qwen35_sizing.max_batch)) as Arc<dyn ResidentModel>)
        }
        "huginn" => {
            let dir = roles.get("weights").and_then(|p| p.to_str()).ok_or("compound manifest missing role \"weights\"")?;
            catalog::resident_huginn::HuginnResident::from_card(dir, card, None)
                .map(|h| Arc::new(h.with_sizing(qwen35_sizing.ctx, catalog::resident_huginn::DEFAULT_BLOCK, qwen35_sizing.max_batch)) as Arc<dyn ResidentModel>)
        }
        "zimage" => {
            let paths = zimage_paths_from_roles(roles)?;
            catalog::resident_zimage::ZImageResident::from_paths(card.id.clone(), paths).map(|z| Arc::new(z) as Arc<dyn ResidentModel>)
        }
        // Unlike zimage's, this construction cannot fail: the four roles ARE
        // the model, and the weights are read lazily at activate().
        "wan" => wan_paths_from_roles(roles).map(|paths| Arc::new(catalog::resident_wan::WanResident::from_paths(card.id.clone(), paths)) as Arc<dyn ResidentModel>),
        // Same shape as wan's: five roles ARE the model (a 33B fp32 DiT is
        // loaded lazily at the first request - see `catalog::
        // resident_minimaxh3`'s own doc for why one hot transformer serves
        // every request shape).
        "minimaxh3" => minimaxh3_paths_from_roles(roles).map(|paths| Arc::new(catalog::resident_minimaxh3::MiniMaxH3Resident::from_paths(card.id.clone(), paths)) as Arc<dyn ResidentModel>),
        other => Err(format!("compound family '{other}' not servable from the model dir yet")),
    };
    other.map(Built::Other)
}

/// Build a `s3dit::pipeline::Paths` from a compound manifest's roles --
/// reads the SAME role names `brain_modelstore::recipe::ZimageRecipe::ROLES`
/// writes, from a `brain.manifest.json` `loader::supply::convert`
/// (`"zimage"` arm) produced.
fn zimage_paths_from_roles(roles: &std::collections::BTreeMap<String, PathBuf>) -> Result<s3dit::pipeline::Paths, String> {
    let get = |role: &str| roles.get(role).and_then(|p| p.to_str()).map(str::to_string).ok_or_else(|| format!("compound manifest missing role {role:?}"));
    Ok(s3dit::pipeline::Paths { dit: get("dit")?, vae: get("vae")?, qwen: get("text_encoder")?, tokenizer: get("tokenizer")? })
}

/// Build a `wan::Paths` from a compound manifest's roles -- the SAME role
/// names `brain_modelstore::recipe::WanRecipe::ROLES` writes, from a
/// `brain.manifest.json` `loader::supply::convert_wan` produced. The
/// `text_encoder` role is umT5-XXL rather than z-image's Qwen, but the role
/// NAMES are deliberately the same four as zimage's, so the two compound
/// families read alike.
fn wan_paths_from_roles(roles: &std::collections::BTreeMap<String, PathBuf>) -> Result<wan::Paths, String> {
    let get = |role: &str| roles.get(role).and_then(|p| p.to_str()).map(str::to_string).ok_or_else(|| format!("compound manifest missing role {role:?}"));
    Ok(wan::Paths { dit: get("dit")?, vae: get("vae")?, t5: get("text_encoder")?, tokenizer: get("tokenizer")? })
}

/// Build a `minimaxh3::caps::Paths` from a compound manifest's roles -- the
/// SAME five role names `brain_modelstore::recipe::H3Recipe`'s finish step
/// (`loader::supply::convert_h3`) writes and `minimaxh3::spec::H3Spec`
/// classifies on, from a `brain.manifest.json` the pull produced. Each role
/// is one of the release's own role directories
/// (`transformer`/`vae`/`audio_vae`/`text_encoder`/`tokenizer`).
fn minimaxh3_paths_from_roles(roles: &std::collections::BTreeMap<String, PathBuf>) -> Result<minimaxh3::caps::Paths, String> {
    let get = |role: &str| roles.get(role).and_then(|p| p.to_str()).map(str::to_string).ok_or_else(|| format!("compound manifest missing role {role:?}"));
    Ok(minimaxh3::caps::Paths { dit: get("dit")?, video_vae: get("video_vae")?, vocoder: get("vocoder")?, text_encoder: get("text_encoder")?, tokenizer: get("tokenizer")? })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("brain-modeldir-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A tiny carded safetensors: one 1-element tensor + a `ModelCard`.
    fn write_st(dir: &Path, file: &str, id: &str, family: &str) {
        let card = ModelCard::new(id, family);
        let p = dir.join(file);
        st::save_safetensors(p.to_str().unwrap(), &[("w".into(), vec![1u64], vec![0.0f32])], &serde_json::json!({}), Some(&card)).unwrap();
    }

    /// A pulled checkpoint whose family its own catalog entry serves
    /// (MiniCPM-V, registered from its resolved assembly) is neither built a
    /// second time by the scan nor reported as a family it cannot serve.
    #[test]
    fn a_catalog_served_checkpoint_is_left_to_its_catalog_entry() {
        let dir = tmp_dir("catalog-served");
        let repo = dir.join("openbmb").join("MiniCPM-V-4.6");
        std::fs::create_dir_all(&repo).unwrap();
        std::fs::write(repo.join("config.json"), br#"{"architectures": ["MiniCPMV4_6ForConditionalGeneration"], "model_type": "minicpmv4_6"}"#).unwrap();
        st::save_safetensors(repo.join("model.safetensors").to_str().unwrap(), &[("w".into(), vec![1u64], vec![0.0f32])], &serde_json::json!({}), None).unwrap();
        write_tokenizer(&repo);
        assert!(brain_modelstore::Store::new(dir.clone()).scan().iter().any(|l| l.card.as_ref().is_some_and(|c| c.family == "minicpmv")), "the store recognizes the checkpoint");
        let found = discover(&dir, &ScanSettings::default());
        assert!(found.errors.is_empty(), "{:?}", found.errors.iter().map(|e| &e.reason).collect::<Vec<_>>());
        assert!(found.residents.is_empty() && found.multi.is_empty(), "the scan built it too");
        std::fs::remove_dir_all(&dir).ok();
    }

    fn write_tokenizer(dir: &Path) {
        std::fs::write(dir.join("tokenizer.json"), br#"{"model":{"vocab":{"a":0}}}"#).unwrap();
    }

    fn put_str(v: &mut Vec<u8>, s: &str) {
        v.extend((s.len() as u64).to_le_bytes());
        v.extend(s.as_bytes());
    }

    /// A minimal GGUF (one f32 tensor) whose KV synthesizes a card: family =
    /// `general.architecture`, id = `general.name`.
    fn write_gguf(dir: &Path, file: &str, arch: &str, name: &str) {
        let mut h: Vec<u8> = Vec::new();
        h.extend(b"GGUF");
        h.extend(3u32.to_le_bytes()); // version
        h.extend(1u64.to_le_bytes()); // tensor count
        h.extend(2u64.to_le_bytes()); // kv count
        put_str(&mut h, "general.architecture");
        h.extend(8u32.to_le_bytes()); // value type: string
        put_str(&mut h, arch);
        put_str(&mut h, "general.name");
        h.extend(8u32.to_le_bytes());
        put_str(&mut h, name);
        // tensor info: "w", 1 dim [4], type F32, offset 0
        put_str(&mut h, "w");
        h.extend(1u32.to_le_bytes());
        h.extend(4u64.to_le_bytes());
        h.extend(0u32.to_le_bytes());
        h.extend(0u64.to_le_bytes());
        let data_start = h.len().div_ceil(32) * 32;
        h.resize(data_start, 0);
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            h.extend(v.to_le_bytes());
        }
        std::fs::write(dir.join(file), &h).unwrap();
    }

    /// A minimal GGUF whose KV synthesizes a `qwen` card AND embeds a gpt2
    /// byte-level BPE tokenizer (`tokenizer.ggml.*`) — the shape a real Qwen3
    /// GGUF ships, minus the weights. Enough for the discovery/registration path
    /// to build a chat-capable Qwen entry with no sibling tokenizer.json.
    fn write_gguf_qwen(dir: &Path, file: &str, name: &str) {
        fn put_str_arr(v: &mut Vec<u8>, key: &str, items: &[&str]) {
            put_str(v, key);
            v.extend(9u32.to_le_bytes()); // array
            v.extend(8u32.to_le_bytes()); // element type: string
            v.extend((items.len() as u64).to_le_bytes());
            for s in items {
                put_str(v, s);
            }
        }
        fn put_i32_arr(v: &mut Vec<u8>, key: &str, items: &[i32]) {
            put_str(v, key);
            v.extend(9u32.to_le_bytes()); // array
            v.extend(5u32.to_le_bytes()); // element type: int32
            v.extend((items.len() as u64).to_le_bytes());
            for x in items {
                v.extend(x.to_le_bytes());
            }
        }
        fn put_u32_kv(v: &mut Vec<u8>, key: &str, x: u32) {
            put_str(v, key);
            v.extend(4u32.to_le_bytes());
            v.extend(x.to_le_bytes());
        }

        let mut kv: Vec<u8> = Vec::new();
        put_str(&mut kv, "general.architecture");
        kv.extend(8u32.to_le_bytes());
        // The real value llama.cpp writes for Qwen3 (verified against an
        // actual downloaded Qwen3-0.6B-GGUF) -- brain's own family name is
        // "qwen"; see `brain_family`'s alias.
        put_str(&mut kv, "qwen3");
        put_str(&mut kv, "general.name");
        kv.extend(8u32.to_le_bytes());
        put_str(&mut kv, name);
        put_str(&mut kv, "tokenizer.ggml.model");
        kv.extend(8u32.to_le_bytes());
        put_str(&mut kv, "gpt2");
        put_str_arr(&mut kv, "tokenizer.ggml.tokens", &["<|endoftext|>", "<|im_start|>", "<|im_end|>", "h", "i", "hi"]);
        put_str_arr(&mut kv, "tokenizer.ggml.merges", &["h i"]);
        put_i32_arr(&mut kv, "tokenizer.ggml.token_type", &[3, 3, 3, 1, 1, 1]);
        put_u32_kv(&mut kv, "tokenizer.ggml.bos_token_id", 0);
        put_u32_kv(&mut kv, "tokenizer.ggml.eos_token_id", 2);

        let mut h: Vec<u8> = Vec::new();
        h.extend(b"GGUF");
        h.extend(3u32.to_le_bytes()); // version
        h.extend(1u64.to_le_bytes()); // tensor count
        h.extend(8u64.to_le_bytes()); // kv count
        h.extend(&kv);
        // tensor info: "w", 1 dim [4], type F32, offset 0
        put_str(&mut h, "w");
        h.extend(1u32.to_le_bytes());
        h.extend(4u64.to_le_bytes());
        h.extend(0u32.to_le_bytes());
        h.extend(0u64.to_le_bytes());
        let data_start = h.len().div_ceil(32) * 32;
        h.resize(data_start, 0);
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            h.extend(v.to_le_bytes());
        }
        std::fs::write(dir.join(file), &h).unwrap();
    }

    /// A 3-part split GGUF carrying the same "qwen" card shape
    /// `write_gguf_qwen` writes as one file - so this test proves the split
    /// path registers a real, chat-capable resident, not just that
    /// `discover_flat`'s grouping runs without panicking. Each part gets its
    /// own distinctly-named tensor (`MmapGguf::open`'s cross-part merge
    /// refuses a repeated name), and only part 1 carries the KV a card is
    /// synthesized from - split.no/split.count/split.tensors.count come
    /// from `write_split` itself.
    fn write_gguf_qwen_split(dir: &Path, base: &str, name: &str) -> String {
        use checkpoint::gguf::GgufValue as V;
        let kv = vec![
            ("general.architecture".to_string(), V::String("qwen3".to_string())),
            ("general.name".to_string(), V::String(name.to_string())),
            ("tokenizer.ggml.model".to_string(), V::String("gpt2".to_string())),
            (
                "tokenizer.ggml.tokens".to_string(),
                V::Array(["<|endoftext|>", "<|im_start|>", "<|im_end|>", "h", "i", "hi"].into_iter().map(|s| V::String(s.to_string())).collect()),
            ),
            ("tokenizer.ggml.merges".to_string(), V::Array(vec![V::String("h i".to_string())])),
            ("tokenizer.ggml.token_type".to_string(), V::Array([3, 3, 3, 1, 1, 1].into_iter().map(V::I32).collect())),
            ("tokenizer.ggml.bos_token_id".to_string(), V::U32(0)),
            ("tokenizer.ggml.eos_token_id".to_string(), V::U32(2)),
        ];
        let tensor = |n: &str| checkpoint::gguf_write::TensorOut {
            name: n.to_string(),
            shape: vec![4],
            ty: checkpoint::gguf::GgmlType::F32.id(),
            data: [1.0f32, 2.0, 3.0, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
        };
        let parts = vec![vec![tensor("w0")], vec![tensor("w1")], vec![tensor("w2")]];
        checkpoint::gguf_write::write_split(dir.to_str().unwrap(), base, &kv, &parts, 32).unwrap()
    }

    /// The split path must register exactly ONE resident from a 3-part
    /// split, chat-capable exactly like `gguf_qwen_with_embedded_tokenizer_
    /// registers_chat_capable`'s single-file case.
    #[test]
    fn a_split_gguf_registers_once_as_a_chat_capable_qwen_resident() {
        let dir = tmp_dir("ggufqwensplit");
        write_gguf_qwen_split(&dir, "qwen3-split", "toy-qwen-gguf-split");

        let residents = discover(&dir, &ScanSettings::default()).residents;
        let matching: Vec<&Arc<dyn ResidentModel>> = residents.iter().filter(|r| r.manifest().model == "toy-qwen-gguf-split").collect();
        assert_eq!(matching.len(), 1, "the 3-part split must register exactly once, not once per part");

        let m = matching[0].manifest();
        let gen = m.actions.iter().find(|a| a.name == "generate").expect("generate action");
        assert!(gen.streaming, "qwen generate must stream");
        assert!(gen.params.iter().any(|p| p.name == "messages"), "qwen generate must accept chat messages");
        std::fs::remove_dir_all(&dir).ok();
    }

    fn ids(residents: &[Arc<dyn ResidentModel>]) -> Vec<String> {
        residents.iter().map(|r| r.manifest().model).collect()
    }

    #[test]
    fn discovers_each_file_as_a_distinct_catalog_entry() {
        let dir = tmp_dir("variants");
        // Two checkpoints of ONE family → two distinct selectable models.
        write_st(&dir, "base.safetensors", "toy-base", "gpt");
        write_st(&dir, "ft.safetensors", "toy-ft", "gpt");
        // Unknown family → skipped (not fatal).
        write_st(&dir, "mystery.safetensors", "toy-unknown", "mystery");
        // A sibling tokenizer.json lets the encoder register (it embeds no vocab).
        write_tokenizer(&dir);
        write_st(&dir, "enc.safetensors", "toy-enc", "lfm");
        // GGUF card is synthesized from KV and dispatched by family.
        write_gguf(&dir, "toy.gguf", "gpt", "toy-gguf");

        let got = ids(&discover(&dir, &ScanSettings::default()).residents);
        assert!(got.contains(&"toy-base".to_string()), "base missing: {got:?}");
        assert!(got.contains(&"toy-ft".to_string()), "ft missing: {got:?}");
        assert!(!got.contains(&"toy-unknown".to_string()), "unknown family not skipped: {got:?}");
        assert!(got.contains(&"toy-enc".to_string()), "sibling tokenizer.json not picked up (lfm missing): {got:?}");
        assert!(got.contains(&"toy-gguf".to_string()), "gguf card not discovered: {got:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// gpt was already proven by `discovers_each_file_as_a_distinct_catalog_entry`
    /// above (its `write_st(..., "gpt")` cases) -- this covers the three
    /// families that joined `resident_for`'s dispatch table alongside gpt's
    /// checkpoint-save fix (glm, yolo, depth): each constructs with nothing
    /// but a path + card (no real weight loading happens until `activate()`),
    /// so a bare `write_st` is enough to prove discovery end to end, exactly
    /// like the "unknown family" contrast case above proves the negative.
    #[test]
    fn discovers_glm_yolo_and_depth_checkpoints() {
        let dir = tmp_dir("glm-yolo-depth");
        write_st(&dir, "glm.safetensors", "toy-glm", "glm");
        write_st(&dir, "yolo.safetensors", "toy-yolo", "yolo");
        write_st(&dir, "depth.safetensors", "toy-depth", "depth");

        let residents = discover(&dir, &ScanSettings::default()).residents;
        let got = ids(&residents);
        assert!(got.contains(&"toy-glm".to_string()), "glm missing: {got:?}");
        assert!(got.contains(&"toy-yolo".to_string()), "yolo missing: {got:?}");
        assert!(got.contains(&"toy-depth".to_string()), "depth missing: {got:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn lfm_skipped_without_sibling_tokenizer() {
        // No tokenizer.json → the encoder cannot construct → skipped (not fatal).
        let dir = tmp_dir("notok");
        write_st(&dir, "enc.safetensors", "toy-enc", "lfm");
        let got = ids(&discover(&dir, &ScanSettings::default()).residents);
        assert!(!got.contains(&"toy-enc".to_string()), "lfm registered without a tokenizer: {got:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn qwen3_gguf_architecture_aliases_to_the_qwen_family() {
        assert_eq!(brain_family("qwen3"), "qwen");
        assert_eq!(brain_family("qwen"), "qwen");
        assert_eq!(brain_family("llama"), "llama"); // unaliased families pass through
    }

    #[test]
    fn gguf_qwen_with_embedded_tokenizer_registers_chat_capable() {
        // A Qwen3 GGUF (general.architecture = "qwen3", llama.cpp's own naming)
        // carrying its own gpt2 tokenizer registers with NO sibling
        // tokenizer.json, and advertises the streaming chat `generate` action.
        let dir = tmp_dir("ggufqwen");
        write_gguf_qwen(&dir, "qwen3.gguf", "toy-qwen-gguf");
        let residents = discover(&dir, &ScanSettings::default()).residents;
        let got = ids(&residents);
        assert!(got.contains(&"toy-qwen-gguf".to_string()), "qwen gguf not registered: {got:?}");

        // The registered entry is chat-capable: its `generate` action streams and
        // takes chat `messages` (generate_spec(chat=true)).
        let r = residents.iter().find(|r| r.manifest().model == "toy-qwen-gguf").unwrap();
        let m = r.manifest();
        let gen = m.actions.iter().find(|a| a.name == "generate").expect("generate action");
        assert!(gen.streaming, "qwen generate must stream");
        assert!(gen.params.iter().any(|p| p.name == "messages"), "qwen generate must accept chat messages");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A raw `.gguf` must never be handed to a brain-native family resident
    /// just because its `general.architecture` happens to spell that family's
    /// name. `card_of` copies the architecture into `card.family` VERBATIM, so
    /// a `qwen35moe`/`qwen35` GGUF collides with the two arms serving those
    /// families' brain-native checkpoints -- residents whose `Engine` reads
    /// `checkpoint::load` (safetensors ONLY) and cannot open GGUF bytes at
    /// all. The collision must resolve to the `brain import-gguf` guidance,
    /// not to a resident that dies with a low-level parse error at activate().
    /// (A released `qwen35` GGUF is answered earlier still: the scan serves it
    /// through the multi-card resident, so the single-device dispatch declines
    /// it.) A GGUF architecture whose resident DOES read GGUF (`qwen3` ->
    /// `qwen`) is unaffected.
    #[test]
    fn a_raw_gguf_never_reaches_a_safetensors_only_family_resident() {
        let dir = tmp_dir("gguf-family-collision");
        // The store layout hands a sibling tokenizer.json over, so the two
        // residents below would construct happily -- the tokenizer is not what
        // is supposed to be stopping them.
        write_tokenizer(&dir);
        let tok_path = dir.join("tokenizer.json");
        let tok = tok_path.to_str();

        for arch in ["qwen35moe", "qwen35"] {
            let file = format!("{arch}.gguf");
            write_gguf(&dir, &file, arch, &format!("toy-{arch}-gguf"));
            let p = dir.join(&file);
            let card = card_of(&p).expect("gguf card");
            assert_eq!(card.family, arch, "a GGUF card's family IS general.architecture, verbatim");
            assert!(
                resident_for(p.to_str().unwrap(), &card, tok, catalog::resident_llm::QwenServeConfig::default(), catalog::resident_qwen35::Qwen35Sizing::default()).is_none(),
                "raw {arch}.gguf routed into the brain-native '{arch}' resident, which cannot read GGUF bytes"
            );
        }

        // ...and the guidance is the actionable one, not a bare "unservable":
        // an architecture WITH a registered importer names the exact command.
        assert!(gguf_advice("m.gguf", "qwen35moe").contains("run `brain import-gguf m.gguf`"), "{}", gguf_advice("m.gguf", "qwen35moe"));

        // The GGUF path that DOES work must keep working: qwen3 aliases to the
        // `qwen` family, whose resident streams a `.gguf` directly.
        write_gguf_qwen(&dir, "qwen3.gguf", "toy-qwen-gguf");
        let p = dir.join("qwen3.gguf");
        let card = card_of(&p).expect("gguf card");
        assert!(resident_for(p.to_str().unwrap(), &card, None, catalog::resident_llm::QwenServeConfig::default(), catalog::resident_qwen35::Qwen35Sizing::default()).is_some(), "a qwen3 GGUF must still serve from the scan");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn store_layout_scans_vendor_repo_dirs_with_independent_tokenizers() {
        // Regression test for the shared-tokenizer bug: two different
        // <vendor>/<repo> dirs each own their sibling tokenizer.json, so one
        // repo's tokenizer must never leak into another's.
        let dir = tmp_dir("store-layout");
        let a = dir.join("VendorA").join("RepoX");
        std::fs::create_dir_all(&a).unwrap();
        let card_a = ModelCard::new("VendorA/RepoX", "lfm");
        st::save_safetensors(
            a.join("model.brain.safetensors").to_str().unwrap(),
            &[("w".into(), vec![1u64], vec![0.0f32])],
            &serde_json::json!({}),
            Some(&card_a),
        )
        .unwrap();
        std::fs::write(a.join("tokenizer.json"), br#"{"model":{"vocab":{"a":0}}}"#).unwrap();

        // A second repo with NO tokenizer of its own: it must NOT register (lfm
        // requires one), proving VendorA's tokenizer wasn't picked up for it.
        let b = dir.join("VendorB").join("RepoY");
        std::fs::create_dir_all(&b).unwrap();
        let card_b = ModelCard::new("VendorB/RepoY", "lfm");
        st::save_safetensors(
            b.join("model.brain.safetensors").to_str().unwrap(),
            &[("w".into(), vec![1u64], vec![0.0f32])],
            &serde_json::json!({}),
            Some(&card_b),
        )
        .unwrap();

        let got = ids(&discover(&dir, &ScanSettings::default()).residents);
        assert!(got.contains(&"VendorA/RepoX".to_string()), "A missing: {got:?}");
        assert!(
            !got.contains(&"VendorB/RepoY".to_string()),
            "B registered without its own tokenizer -- tokenizer leaked across repos: {got:?}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A base Qwen model and a named LoRA adapter trained on top of it must
    /// register as TWO distinct, independently selectable catalog entries
    /// (mirrors `brain_modelstore::Store`'s own
    /// `scan_finds_a_base_model_and_its_adapter_as_two_distinct_entries`,
    /// but as the CLI-facing gate: this is what `brain caps`/`brain do`
    /// actually walk). The adapter is a view over the base: the base's one
    /// instance key, its own id, and the adapter's digest as its identity.
    #[test]
    fn discover_surfaces_a_base_and_its_named_adapter_as_two_distinct_entries() {
        let dir = tmp_dir("adapter-catalog");
        let repo = dir.join("Qwen").join("Qwen3-Toy");
        std::fs::create_dir_all(&repo).unwrap();
        let base_card = ModelCard::new("Qwen/Qwen3-Toy", "qwen");
        st::save_safetensors(
            repo.join("model.brain.safetensors").to_str().unwrap(),
            &[("w".into(), vec![1u64], vec![0.0f32])],
            &serde_json::json!({}),
            Some(&base_card),
        )
        .unwrap();
        write_tokenizer(&repo);

        let adapter_id = "Qwen/Qwen3-Toy:swedishembedded-com:generic-sft:latest";
        let adapter_dir = repo.join("adapters").join("swedishembedded-com").join("generic-sft").join("latest");
        std::fs::create_dir_all(&adapter_dir).unwrap();
        let mut adapter_card = ModelCard::new(adapter_id, "qwen");
        adapter_card.variant_of = Some("Qwen/Qwen3-Toy".to_string());
        adapter_card.adapter = Some(st::Adapter { kind: "lora".into(), rank: Some(4), base: Some("Qwen/Qwen3-Toy".to_string()), alpha: Some(8.0), targets: Some(vec!["wq".into()]), dataset_id: None, per_target: None });
        let adapter_file = adapter_dir.join("adapter.brain.safetensors");
        st::save_safetensors(
            adapter_file.to_str().unwrap(),
            &[("blocks.0.attn.wq.weight.lora_a".into(), vec![4, 2], vec![0.1f32; 8]), ("blocks.0.attn.wq.weight.lora_b".into(), vec![2, 4], vec![0.1f32; 8])],
            &serde_json::json!({"rank": 4, "alpha": 8.0}),
            Some(&adapter_card),
        )
        .unwrap();

        let residents = discover(&dir, &ScanSettings::default()).residents;
        let got = ids(&residents);
        assert!(got.contains(&"Qwen/Qwen3-Toy".to_string()), "base missing: {got:?}");
        assert!(got.contains(&adapter_id.to_string()), "adapter missing: {got:?}");
        assert_eq!(got.len(), 2, "expected exactly base + adapter, got: {got:?}");

        let by_id = |id: &str| residents.iter().find(|r| r.manifest().model == id).unwrap_or_else(|| panic!("{id} not registered"));
        let (base, adapter) = (by_id("Qwen/Qwen3-Toy"), by_id(adapter_id));
        let inv = capability::Invocation::new();
        assert_eq!(adapter.instance_key("generate", &inv), base.instance_key("generate", &inv), "base and adapter are one instance, so they batch");
        assert_eq!(adapter.adapter_identity().map(|a| (a.id, a.digest)), Some((adapter_id.to_string(), brain_modelstore::fetch::file_digest(&adapter_file).unwrap())));
        assert_eq!(base.adapter_identity(), None, "the base names no adapter");
        let mut stamped = capability::Invocation::new();
        adapter.stamp(&mut stamped);
        assert_eq!(stamped.get_str("__adapter").as_deref(), Some(adapter_id));
        let mut forged = capability::Invocation::new().set("__adapter", serde_json::json!(adapter_id));
        base.stamp(&mut forged);
        assert_eq!(forged.get_str("__adapter"), None, "a caller reaching the base cannot select an adapter");
        assert!(base.estimate(&base.instance_key("generate", &inv)).vram == adapter.estimate(&adapter.instance_key("generate", &inv)).vram, "the view is budgeted as the base");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Adapters ride on a released Qwen3.8 GGUF the way they ride on a Qwen3:
    /// each is a view of the multi-card base (its instance key, its placement),
    /// found beside the base by its id or, when its card names the full-precision
    /// release the GGUF quantises, by that; a Qwen3 adapter does not attach, and
    /// the bank the adapters need is part of the base's placement.
    #[test]
    fn a_released_qwen35_gguf_serves_its_adapters_as_views_of_one_multi_card_model() {
        let dir = tmp_dir("released-qwen35-adapters");
        let gguf = put_released_qwen35(&dir, "unsloth", "Qwen3.8-27B-GGUF", "Q8_0.gguf");
        let plain = discover(&dir, &two_p40s());
        let plain_key = plain.multi[0].instance_key("generate", &capability::Invocation::new());
        let plain_bytes: u64 = plain.multi[0].estimate_multi(&plain_key).devices().map(|d| plain.multi[0].estimate_multi(&plain_key).on(d)).sum();
        assert!(plain.qwen35.len() == 1 && plain.multi.len() == 1);
        let repo = gguf.parent().unwrap().to_path_buf();

        let put_adapter = |id: &str, family: &str, slot: &str, base: &str| -> PathBuf {
            use brain_testutil::adapters::{write_lora_adapter_of, Linear};
            let adapter_dir = repo.join("adapters").join("lab").join(slot).join("v1");
            std::fs::create_dir_all(&adapter_dir).unwrap();
            let file = adapter_dir.join("adapter.brain.safetensors");
            let leaf = if family == "qwen35" { "linear_attn.in_proj_qkv" } else { "attn.wq" };
            write_lora_adapter_of(family, &file, id, base, 4, &[Linear { base: format!("blocks.0.{leaf}.weight"), out: 6, inn: 3 }], 3);
            file
        };
        // The store files an adapter under the GGUF's repo and names its
        // quant; the card's base says which model it was trained for.
        let by_gguf_id = "unsloth/Qwen3.8-27B-GGUF-Q8_0:lab:by-id:v1";
        let by_release = "unsloth/Qwen3.8-27B-GGUF-Q8_0:lab:by-release:v1";
        let wrong_family = "unsloth/Qwen3.8-27B-GGUF-Q8_0:lab:wrong:v1";
        let file = put_adapter(by_gguf_id, "qwen35", "by-id", "unsloth/Qwen3.8-27B-GGUF");
        put_adapter(by_release, "qwen35", "by-release", "Qwen/Qwen3.8-27B-FP8");
        put_adapter(wrong_family, "qwen", "wrong", "unsloth/Qwen3.8-27B-GGUF");

        let found = discover(&dir, &two_p40s());
        let mut want = vec![by_release.to_string(), by_gguf_id.to_string(), "unsloth/Qwen3.8-27B-GGUF".to_string()];
        want.sort();
        assert_eq!(multi_ids(&found), want, "the base and its two Qwen3.8 adapters; the Qwen3 adapter is not served");
        assert!(found.residents.is_empty(), "views of a multi-card model are multi-card models");
        let base = found.multi.iter().find(|m| m.manifest().model == "unsloth/Qwen3.8-27B-GGUF").unwrap();
        let inv = capability::Invocation::new();
        let key = base.instance_key("generate", &inv);
        for id in [by_gguf_id, by_release] {
            let view = found.multi.iter().find(|m| m.manifest().model == id).unwrap_or_else(|| panic!("{id} not served"));
            assert_eq!(view.instance_key("generate", &inv), key, "base and adapter are one instance, so they batch");
            assert_eq!(view.estimate_multi(&key).devices().collect::<Vec<_>>(), base.estimate_multi(&key).devices().collect::<Vec<_>>(), "the view is placed as the base");
            let mut stamped = capability::Invocation::new();
            view.stamp(&mut stamped);
            assert_eq!(stamped.get_str("__adapter").as_deref(), Some(id));
        }
        let view = found.multi.iter().find(|m| m.manifest().model == by_gguf_id).unwrap();
        assert_eq!(view.adapter_identity().map(|a| (a.id, a.digest)), Some((by_gguf_id.to_string(), brain_modelstore::fetch::file_digest(&file).unwrap())));
        let banked_bytes: u64 = base.estimate_multi(&key).devices().map(|d| base.estimate_multi(&key).on(d)).sum();
        assert!(banked_bytes > plain_bytes, "the adapter bank the store's adapters need is part of the base's placement ({banked_bytes} vs {plain_bytes})");
        assert!(found.errors.is_empty(), "{:?}", found.errors);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// An adapter whose rank or linears exceed the default bank widens the
    /// bank built for ITS base (the store holds it, so it is served) unless the
    /// operator sized the bank, in which case it is skipped with the flag to
    /// change.
    #[test]
    fn the_bank_grows_to_hold_the_stores_adapters_unless_the_operator_sized_it() {
        let dir = tmp_dir("adapter-bank-size");
        let repo = dir.join("Qwen").join("Qwen3-Toy");
        let adapter_dir = repo.join("adapters").join("lab").join("wide").join("v1");
        std::fs::create_dir_all(&adapter_dir).unwrap();
        let base_card = ModelCard::new("Qwen/Qwen3-Toy", "qwen");
        st::save_safetensors(repo.join("model.brain.safetensors").to_str().unwrap(), &[("w".into(), vec![1u64], vec![0.0f32])], &serde_json::json!({}), Some(&base_card)).unwrap();
        write_tokenizer(&repo);
        let mut card = ModelCard::new("Qwen/Qwen3-Toy:lab:wide:v1", "qwen");
        card.adapter = Some(st::Adapter { kind: "lora".into(), rank: Some(32), base: Some("Qwen/Qwen3-Toy".to_string()), alpha: Some(64.0), targets: Some(vec!["wq".into(), "down".into()]), dataset_id: None, per_target: None });
        let (a, b) = (vec![0.1f32; 32 * 2], vec![0.1f32; 2 * 32]);
        st::save_safetensors(
            adapter_dir.join("adapter.brain.safetensors").to_str().unwrap(),
            &[("blocks.0.attn.wq.weight.lora_a".into(), vec![32, 2], a.clone()), ("blocks.0.attn.wq.weight.lora_b".into(), vec![2, 32], b.clone()), ("blocks.0.mlp.down.weight.lora_a".into(), vec![32, 2], a), ("blocks.0.mlp.down.weight.lora_b".into(), vec![2, 32], b)],
            &serde_json::json!({"rank": 32, "alpha": 64.0}),
            Some(&card),
        )
        .unwrap();

        let defaults = catalog::resident_llm::QwenServeConfig::default();
        let residents = discover(&dir, &ScanSettings { qwen: defaults.clone(), ..Default::default() }).residents;
        assert!(ids(&residents).contains(&"Qwen/Qwen3-Toy:lab:wide:v1".to_string()), "rank 32 on down is above the defaults, so the bank grew: {:?}", ids(&residents));

        // `--adapter-targets` makes the bank the operator's: not widened.
        let named = catalog::resident_llm::QwenServeConfig { adapter_bank: Some(AdapterBankSettings::default()), adapter_targets_named: true, ..defaults.clone() };
        let residents = discover(&dir, &ScanSettings { qwen: named, ..Default::default() }).residents;
        assert_eq!(ids(&residents), vec!["Qwen/Qwen3-Toy".to_string()], "an operator-named bank is not widened for the store");

        // `--adapter-slots` / `--adapter-rank-cap` alone size the bank but name
        // no targets: the store's adapters still widen the targets (down is not
        // an attention linear), as with no bank flag at all, while the rank cap
        // stays the operator's.
        let roomy = catalog::resident_llm::QwenServeConfig { adapter_bank: Some(AdapterBankSettings { slots: 2, rank_cap: 32, ..AdapterBankSettings::default() }), ..defaults.clone() };
        let residents = discover(&dir, &ScanSettings { qwen: roomy, ..Default::default() }).residents;
        assert!(ids(&residents).contains(&"Qwen/Qwen3-Toy:lab:wide:v1".to_string()), "slots and rank cap given, targets not: the targets grew: {:?}", ids(&residents));
        let tight = catalog::resident_llm::QwenServeConfig { adapter_bank: Some(AdapterBankSettings { slots: 2, rank_cap: 8, ..AdapterBankSettings::default() }), ..defaults };
        let residents = discover(&dir, &ScanSettings { qwen: tight, ..Default::default() }).residents;
        assert_eq!(ids(&residents), vec!["Qwen/Qwen3-Toy".to_string()], "a rank cap the operator gave is not widened");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A `brain.manifest.json` naming `family` with the four wan-shaped roles
    /// (dit/vae/text_encoder/tokenizer), same layout `loader::supply::convert_wan`
    /// writes -- content is never opened at construction (`WanResident::
    /// from_paths` reads lazily at `activate()`), so stub bytes are enough.
    fn write_compound_fixture(dir: &Path, vendor: &str, repo: &str, family: &str) {
        let repo_dir = dir.join(vendor).join(repo);
        std::fs::create_dir_all(&repo_dir).unwrap();
        let mut roles = BTreeMap::new();
        for role in ["dit", "vae", "text_encoder", "tokenizer"] {
            std::fs::write(repo_dir.join(format!("{role}.bin")), b"stub").unwrap();
            roles.insert(role.to_string(), format!("{role}.bin"));
        }
        let manifest = brain_modelstore::CompoundManifest { id: format!("{vendor}/{repo}"), family: family.to_string(), roles };
        std::fs::write(repo_dir.join(brain_modelstore::MANIFEST_FILE), serde_json::to_vec(&manifest).unwrap()).unwrap();
    }

    /// A qwen3-decoder checkpoint pulled as downloaded (a manifest whose
    /// `weights` role is the repo directory itself) is served by the Qwen
    /// resident under its own id, whatever variant row names it.
    #[test]
    fn discover_serves_a_downloaded_llama_directory() {
        let root = tmp_dir("discover-llama-dir");
        let repo_dir = root.join("deepseek-ai").join("tiny-llama");
        std::fs::create_dir_all(&repo_dir).unwrap();
        let manifest = brain_modelstore::CompoundManifest { id: "deepseek-ai/tiny-llama".into(), family: "llama".into(), roles: [("weights".to_string(), ".".to_string())].into() };
        std::fs::write(repo_dir.join(brain_modelstore::MANIFEST_FILE), serde_json::to_vec(&manifest).unwrap()).unwrap();
        let Discovered { residents, errors, .. } = discover(&root, &ScanSettings::default());
        assert!(errors.is_empty(), "unexpected discovery errors: {errors:?}");
        assert_eq!(ids(&residents), ["deepseek-ai/tiny-llama"]);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A downloaded embedding checkpoint of the qwen3 family is an embedding
    /// model, not a chat model: it is served with the `embed` action
    /// (last-token pooling) so `/v1/embeddings` can reach it, and offers no
    /// `generate`.
    #[test]
    fn discover_serves_a_qwen3_embedding_directory_as_an_embedding_model() {
        let root = tmp_dir("discover-qwen3-embedding-dir");
        let repo_dir = root.join("Qwen").join("Qwen3-Embedding-0.6B");
        std::fs::create_dir_all(&repo_dir).unwrap();
        let manifest = brain_modelstore::CompoundManifest { id: "Qwen/Qwen3-Embedding-0.6B".into(), family: "qwen3".into(), roles: [("weights".to_string(), ".".to_string())].into() };
        std::fs::write(repo_dir.join(brain_modelstore::MANIFEST_FILE), serde_json::to_vec(&manifest).unwrap()).unwrap();
        let Discovered { residents, errors, .. } = discover(&root, &ScanSettings::default());
        assert!(errors.is_empty(), "unexpected discovery errors: {errors:?}");
        let resident = residents.iter().find(|r| r.manifest().model == "Qwen/Qwen3-Embedding-0.6B").expect("discovered");
        let actions: Vec<String> = resident.manifest().actions.iter().map(|a| a.name.clone()).collect();
        assert_eq!(actions, ["embed"]);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A pulled GLM-DSA directory is served by the GLM resident from the
    /// download, like the qwen3 decoder's.
    #[test]
    fn discover_serves_a_downloaded_glm_directory() {
        let root = tmp_dir("discover-glm-dir");
        let repo_dir = root.join("zai-org").join("tiny-glm");
        std::fs::create_dir_all(&repo_dir).unwrap();
        let manifest = brain_modelstore::CompoundManifest { id: "zai-org/tiny-glm".into(), family: "glmdsa".into(), roles: [("weights".to_string(), ".".to_string())].into() };
        std::fs::write(repo_dir.join(brain_modelstore::MANIFEST_FILE), serde_json::to_vec(&manifest).unwrap()).unwrap();
        let Discovered { residents, errors, .. } = discover(&root, &ScanSettings::default());
        assert!(errors.is_empty(), "unexpected discovery errors: {errors:?}");
        assert_eq!(ids(&residents), ["zai-org/tiny-glm"]);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn discover_registers_a_wan_compound_model_via_resident_for_local() {
        let dir = tmp_dir("compound-wan");
        write_compound_fixture(&dir, "Wan-AI", "Wan2.1-T2V-1.3B", "wan");

        let Discovered { residents, errors, .. } = discover(&dir, &ScanSettings::default());
        assert!(errors.is_empty(), "unexpected discovery errors: {errors:?}");
        let got = ids(&residents);
        assert!(got.contains(&"Wan-AI/Wan2.1-T2V-1.3B".to_string()), "wan compound model not discovered: {got:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn discover_reports_an_unrecognized_compound_family_as_a_named_error_not_a_panic() {
        let dir = tmp_dir("compound-unknown-family");
        write_compound_fixture(&dir, "Some-Vendor", "Mystery-Compound", "flux2");

        let Discovered { residents, errors, .. } = discover(&dir, &ScanSettings::default());
        let got = ids(&residents);
        assert!(!got.contains(&"Some-Vendor/Mystery-Compound".to_string()), "unrecognized compound family must not register: {got:?}");
        assert_eq!(errors.len(), 1, "expected exactly one discovery error, got: {errors:?}");
        assert_eq!(errors[0].family, "flux2");
        assert_eq!(errors[0].dir, dir.join("Some-Vendor").join("Mystery-Compound"));
        std::fs::remove_dir_all(&dir).ok();
    }

    // ---- released Qwen3.8 GGUFs: every one is its own multi-card model

    const GIB: u64 = 1 << 30;

    /// A complete synthetic Qwen3.8 release at `<root>/<vendor>/<repo>/<file>`.
    fn put_released_qwen35(root: &Path, vendor: &str, repo: &str, file: &str) -> PathBuf {
        let dir = root.join(vendor).join(repo);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(file);
        qwen35::gguf_import::testing::write_synthetic_servable_gguf(path.to_str().unwrap());
        path
    }

    fn two_p40s() -> ScanSettings {
        ScanSettings { gpus: vec![(0, 24 * GIB), (1, 24 * GIB)], reserved: 2 * GIB, ..Default::default() }
    }

    fn multi_ids(found: &Discovered) -> Vec<String> {
        let mut ids: Vec<String> = found.multi.iter().map(|m| m.manifest().model).collect();
        ids.sort();
        ids
    }

    /// The point of the scan: two released quants in the store are two served
    /// models under their store ids, neither disabling the other, and neither
    /// is a single-device resident.
    #[test]
    fn every_released_qwen35_gguf_in_the_store_is_its_own_served_model() {
        let dir = tmp_dir("released-qwen35");
        put_released_qwen35(&dir, "unsloth", "Qwen3.8-27B-GGUF", "Q8_0.gguf");
        put_released_qwen35(&dir, "Abiray", "Qwen3.8-27B-Q4_K_M-GGUF", "Q4_K_M.gguf");

        let found = discover(&dir, &two_p40s());
        assert_eq!(multi_ids(&found), vec!["Abiray/Qwen3.8-27B-Q4_K_M-GGUF".to_string(), "unsloth/Qwen3.8-27B-GGUF".to_string()]);
        assert!(found.residents.is_empty(), "a multi-card model must not also be registered as a single-device one: {:?}", ids(&found.residents));
        assert!(found.errors.is_empty(), "{:?}", found.errors);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Two quants of one repo would collide on `<vendor>/<repo>`, so each then
    /// carries its quant instead of one being dropped as a duplicate.
    #[test]
    fn quants_sharing_a_repo_are_told_apart_by_their_quant() {
        let dir = tmp_dir("released-qwen35-quants");
        put_released_qwen35(&dir, "unsloth", "Qwen3.8-27B-GGUF", "Q8_0.gguf");
        put_released_qwen35(&dir, "unsloth", "Qwen3.8-27B-GGUF", "Q4_K_M.gguf");

        let found = discover(&dir, &two_p40s());
        assert_eq!(multi_ids(&found), vec!["unsloth/Qwen3.8-27B-GGUF-Q4_K_M".to_string(), "unsloth/Qwen3.8-27B-GGUF-Q8_0".to_string()]);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A listed model always loads: every entry places on at least one budgeted
    /// card, and only cards the process may use. A file that could never load
    /// is not listed, and does not take a loadable neighbour down with it.
    #[test]
    fn a_listed_qwen35_gguf_is_always_placeable_on_the_allowed_cards() {
        let dir = tmp_dir("released-qwen35-listed");
        put_released_qwen35(&dir, "unsloth", "Qwen3.8-27B-GGUF", "Q8_0.gguf");
        // No embedded vocabulary: weights fine, but no request could tokenize.
        let bare = dir.join("Abiray").join("Qwen3.8-27B-Q4_K_M-GGUF");
        std::fs::create_dir_all(&bare).unwrap();
        qwen35::gguf_import::testing::write_synthetic_gguf(bare.join("Q4_K_M.gguf").to_str().unwrap());

        let only_gpu1 = ScanSettings { gpus: vec![(1, 24 * GIB)], reserved: 2 * GIB, ..Default::default() };
        let found = discover(&dir, &only_gpu1);
        assert_eq!(multi_ids(&found), vec!["unsloth/Qwen3.8-27B-GGUF".to_string()]);
        for m in &found.multi {
            let key = m.instance_key("generate", &capability::Invocation::new());
            let devices: Vec<residency::Device> = m.estimate_multi(&key).devices().collect();
            assert_eq!(devices, vec![residency::Device::Gpu(1)], "{} must place only on the allowed card", m.manifest().model);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// With no GPU the process may use, nothing that needs one is listed.
    #[test]
    fn without_an_allowed_gpu_no_released_qwen35_gguf_is_listed() {
        let dir = tmp_dir("released-qwen35-nogpu");
        put_released_qwen35(&dir, "unsloth", "Qwen3.8-27B-GGUF", "Q8_0.gguf");
        let found = discover(&dir, &ScanSettings::default());
        assert!(found.multi.is_empty());
        assert!(found.residents.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Weights that do not fit the allowed cards are not listed either.
    #[test]
    fn a_released_qwen35_gguf_that_does_not_fit_the_cards_is_not_listed() {
        let dir = tmp_dir("released-qwen35-toobig");
        put_released_qwen35(&dir, "unsloth", "Qwen3.8-27B-GGUF", "Q8_0.gguf");
        let tiny = ScanSettings { gpus: vec![(0, 3 * GIB)], reserved: 3 * GIB - 64, ..Default::default() };
        assert!(discover(&dir, &tiny).multi.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }
}
