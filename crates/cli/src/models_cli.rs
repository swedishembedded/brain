// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain models {list,list-adapters,info,profile}` - the model store's own
//! view of itself: which architectures have weights on disk, which official
//! quantizations exist but are not pulled, what a checkpoint's tensors
//! actually are, and what it will cost to run on THIS machine.
//!
//! Every data source already existed - `brain_modelstore::Store::scan`,
//! `brain_arch::Arch::variants` (the declared official-quant registry),
//! `checkpoint::weightio::WeightReader` (per-tensor dtype/shape, header
//! only), `modelcost`'s shared pricing cache - this module is the join.
//!
//! Rendering follows `brain pull`'s own precedent
//! (`Mode::of(stdout.is_terminal())`): a terminal gets the interactive
//! `crate::tree` browser, a pipe/redirect gets plain box-drawing lines whose
//! LEAF rows are self-contained (the full canonical id baked in), so `brain
//! models list | grep Q4_K_M` returns complete, useful lines - see
//! `crate::tree`'s own module doc.
//!
//! Swedish Embedded AB implements model-fleet observability tooling for its
//! clients. If your team needs to know what is actually on your inference
//! boxes - not what a deploy manifest claims - you can procure our services
//! by sending an email to info@swedishembedded.com.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use brain_modelref::ModelRef;
use brain_modelstore::{Format, LocalModel, Store};
use checkpoint::st::ModelCard;
use serde_json::Value;

use crate::args::Args;
use crate::tree::{self, Node};

const USAGE: &str = "\
usage: brain models list [--arch ID] [--local] [--plain|--tui] [--json]
                          [--reprofile] [--models-dir DIR]
       brain models list-adapters [--arch ID] [--plain|--tui] [--json] [--models-dir DIR]
       brain models info <model-ref-or-url> [--json] [--models-dir DIR]
       brain models profile <model-ref-or-url> [--measure [--reps N]] [--models-dir DIR]

  list             architecture -> provider repo -> quantization, local +
                   declared-but-not-pulled, with size/fit/cost from the
                   shared `modelcost` cache. --reprofile re-measures this
                   device's roofline and re-prices every local model.
  list-adapters    architecture -> base variant -> LoRA adapter + metadata.
  info             one checkpoint's real tensor tree (name/dtype/shape/size),
                    adapter tensors merged in and marked with a leading '+'.
  profile          price ONE named local model now and cache the result -
                    errors if it is not pulled; never fetches. --measure
                    instead builds the model for real and TIMES it: load
                    time separate from a forward pass's own time, the cold
                    (first) pass separate from the best of --reps (default 5)
                    hot passes, plus a per-layer FLOP breakdown - derived
                    exactly for a uniform stack (qwen3, gpt2), averaged for a
                    hybrid one (lfm2's per-layer conv/attention mix). Not
                    cached - a timing describes this machine right now.

  --arch ID        only this architecture (brain_arch id, e.g. qwen3)
  --local          declared-but-not-pulled rows are omitted
  --plain          force the plain box-drawing renderer (default off-TTY)
  --tui            force the interactive browser (default on a TTY)
  --json           emit the same tree as data instead of either renderer
  --measure        `profile` only: real execution + timing, see above
  --reps N         `profile --measure` only: hot passes to time (default 5)
  --models-dir DIR override the model store root (see brain pull --help)
";

pub fn run_models(args: &[String]) -> i32 {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return 0;
    }
    match args.first().map(String::as_str) {
        Some("list") => run_list(&args[1..]),
        Some("list-adapters") => run_list_adapters(&args[1..]),
        Some("info") => run_info(&args[1..]),
        Some("profile") => run_profile(&args[1..]),
        Some(other) => {
            eprintln!("brain models: unknown subcommand {other:?}\n{USAGE}");
            2
        }
        None => {
            eprintln!("{USAGE}");
            2
        }
    }
}

fn open_store(a: &mut Args) -> Option<Store> {
    let dir = a.take_str("--models-dir");
    crate::model_dir::resolve(dir.as_deref()).map(Store::new)
}

fn resolve_arch(l: &LocalModel) -> Option<&'static brain_arch::Arch> {
    let card = l.card.as_ref()?;
    match l.format {
        Format::Gguf => brain_arch::by_gguf(&card.family),
        _ => brain_arch::by_id(&card.family),
    }
}

fn human_bytes(n: u64) -> String {
    crate::pull_cli::human_bytes(n)
}

/// Bytes on disk for whatever `local` actually points at - a single file for
/// `Safetensors`/`Gguf`, or the sum of every role's file (or, for a role that
/// is a directory - a sharded HF checkpoint - every file under it) for
/// `Compound`. `None` when nothing can be statted (a role path that does not
/// exist, which `Store::local`'s own contract already treats as "not really
/// there" - see `LocalModel::roles`'s doc).
fn size_of(l: &LocalModel) -> Option<u64> {
    match &l.roles {
        Some(roles) => {
            let mut total = 0u64;
            for path in roles.values() {
                total += dir_or_file_size(path)?;
            }
            Some(total)
        }
        None => std::fs::metadata(&l.weights).ok().map(|m| m.len()),
    }
}

fn dir_or_file_size(path: &Path) -> Option<u64> {
    let meta = std::fs::metadata(path).ok()?;
    if !meta.is_dir() {
        return Some(meta.len());
    }
    let mut total = 0u64;
    for entry in std::fs::read_dir(path).ok()? {
        let entry = entry.ok()?;
        total += dir_or_file_size(&entry.path()).unwrap_or(0);
    }
    Some(total)
}

fn fitting_gpus(bytes: u64) -> Vec<String> {
    gpu_core::devices::registry().devices().iter().filter(|d| d.identity.vram_bytes >= bytes).map(|d| format!("gpu{}", d.index)).collect()
}

fn fit_str(bytes: Option<u64>) -> String {
    match bytes {
        None => "?".to_string(),
        Some(b) => {
            let fits = fitting_gpus(b);
            if fits.is_empty() {
                "no gpu fits".to_string()
            } else {
                format!("{} fits", fits.join(","))
            }
        }
    }
}

/// GGUF's KV metadata uses llama.cpp's own `{arch}.*` naming convention
/// (`qwen3.block_count`, `qwen3.attention.head_count`, ...), never brain's
/// canonical shape-key schema (`n_layers`, `n_heads`, ...) that
/// `{Qwen,Qwen35,...}Config::from_json_checked` validates against - so unlike
/// `Format::Safetensors` (whose `brain.config` blob IS already in that
/// schema, written by brain's own converter), a GGUF's raw KV map cannot be
/// handed to `modelcost` as-is. Each row below is the one architecture-aware
/// translator that knows how to read that architecture's REAL llama.cpp keys
/// (plus, where the KV can disagree with the checkpoint's own tensors - e.g.
/// vocab size - the tensor shape, which is ground truth) and re-emit brain's
/// canonical schema via that architecture's own `Config::to_json`. Adding
/// GGUF support for a new architecture means adding its own
/// `config_from_gguf` next to its importer, then one row here - the same
/// spirit as `modelcost::registry()`'s own per-architecture rows.
fn qwen3_gguf_config(mg: &checkpoint::gguf::MmapGguf) -> Result<Value, String> {
    qwen3::gguf_import::config_from_gguf(mg).map(|c| c.to_json())
}

fn qwen35moe_gguf_config(mg: &checkpoint::gguf::MmapGguf) -> Result<Value, String> {
    qwen35moe::import::config_from_gguf(mg).map(|c| c.to_json())
}

type GgufConfigReader = fn(&checkpoint::gguf::MmapGguf) -> Result<Value, String>;

const GGUF_CONFIG_READERS: &[(&str, GgufConfigReader)] = &[("qwen3", qwen3_gguf_config), ("qwen35moe", qwen35moe_gguf_config)];

/// Header-only config read for a local (non-compound, non-adapter) entry -
/// what both the cost lookup and a `--reprofile` re-price need. `None` for a
/// compound model (no single config to key a cost entry by), an unreadable
/// file, or a GGUF whose architecture has no [`GGUF_CONFIG_READERS`] row yet
/// (an honest "no cost model", not a fabricated one - see that table's doc).
fn read_config(l: &LocalModel, arch_id: &str) -> Option<Value> {
    if matches!(l.format, Format::Compound) {
        return None;
    }
    let path = l.weights.to_str()?;
    match l.format {
        Format::Gguf => {
            let mg = checkpoint::gguf::MmapGguf::open(path).ok()?;
            GGUF_CONFIG_READERS.iter().find(|(id, _)| *id == arch_id)?.1(&mg).ok()
        }
        _ => checkpoint::weightio::WeightReader::open(path).ok().map(|r| r.config()),
    }
}

fn cost_str(arch_id: &str, ref_str: &str, l: &LocalModel) -> String {
    let Some(config) = read_config(l, arch_id) else { return "not profiled".to_string() };
    match modelcost::cached(arch_id, ref_str, &config) {
        Some(c) => format_cached(&c),
        None => "not profiled".to_string(),
    }
}

fn eng(x: f64, unit: &str) -> String {
    for (t, s) in [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")] {
        if x >= t {
            return format!("{:.2} {s}{unit}", x / t);
        }
    }
    format!("{x:.0} {unit}")
}

fn format_cached(c: &modelcost::CachedCost) -> String {
    match c.tier {
        modelcost::Tier::Exact => format!("exact  {} ({:.0}% covered)", eng(c.summary.flops as f64, "FLOP"), c.summary.coverage * 100.0),
        modelcost::Tier::Bandwidth => format!("~bandwidth  {} weight", human_bytes(c.summary.bytes)),
    }
}

// -------------------------------------------------------------------- list --

fn run_list(args: &[String]) -> i32 {
    let mut a = Args::new(args);
    let arch_filter = a.take_str("--arch");
    let local_only = a.take_flag("--local");
    let plain = a.take_flag("--plain");
    let force_tui = a.take_flag("--tui");
    let json = a.take_flag("--json");
    let reprofile = a.take_flag("--reprofile");
    let Some(store) = open_store(&mut a) else {
        a.finish();
        return render(Vec::new(), plain, force_tui, json, None);
    };
    a.finish();

    let locals = store.scan();
    if reprofile {
        run_reprofile(&locals);
    }

    let roots = build_list_tree(&locals, arch_filter.as_deref(), local_only);
    // `Enter` on a local leaf opens its tensor tree in a second screen - the
    // same data `brain models info <ref>` prints, built fresh per open (a
    // new `Store` from the same root; `Store` is cheap - just a `PathBuf`)
    // so it reflects whatever is on disk NOW, not a snapshot from when
    // `list` started.
    let root_path = store.root().to_path_buf();
    let on_enter = move |reference: &str| -> Vec<Node> {
        let store = Store::new(root_path.clone());
        match build_info_nodes(&store, reference) {
            Ok(nodes) => nodes,
            Err(e) => vec![Node::leaf(format!("error: {e}"))],
        }
    };
    render(roots, plain, force_tui, json, Some(&on_enter))
}

fn run_reprofile(locals: &[LocalModel]) {
    // Still does the real remeasure - every local model's price below needs
    // fresh roofline numbers - but no longer prints its OWN GFLOP/s+GB/s
    // summary of it: `brain roofline` is now the canonical place to see the
    // full hardware picture (every accelerator, every dtype), so a second,
    // differently-formatted dump of the same GPU numbers here would just be
    // a duplicate that can drift out of sync with it.
    let gpu = gpu_core::Gpu::new(&[]);
    match gpu_core::roof::reprofile(&gpu) {
        Some(_) => println!("hardware roofline refreshed - see 'brain roofline' for the full picture"),
        None => println!("hardware roofline: unavailable on this device/backend (BRAIN_NO_ROOF, or an unprobeable backend)"),
    }
    let mut priced = 0;
    for l in locals {
        if l.reference.adapter().is_some() {
            continue;
        }
        let Some(arch) = resolve_arch(l) else { continue };
        let Some(config) = read_config(l, arch.id) else { continue };
        // Bandwidth tier only, deliberately - see that function's own doc for
        // why an unattended walk over the whole store must never trigger the
        // exact tier's real (if zero-init) weight-shaped allocation.
        match modelcost::price_and_cache_bandwidth_only(arch.id, &l.reference.to_string(), &config) {
            Ok(_) => priced += 1,
            // Surfaced, never swallowed - a config failing its own shape
            // validation (see `qwen3::QwenConfig::from_json_checked`) is a
            // real problem with that ONE checkpoint, not something a bulk
            // reprofile should hide by silently skipping it.
            Err(e) => eprintln!("brain models list --reprofile: {} not priced: {e}", l.reference),
        }
    }
    println!("priced {priced} local model(s) (bandwidth tier; `brain flops`/`brain models profile <ref>` for an exact number on one model)");
}

fn build_list_tree(locals: &[LocalModel], arch_filter: Option<&str>, local_only: bool) -> Vec<Node> {
    let mut by_arch: BTreeMap<&'static str, Vec<&LocalModel>> = BTreeMap::new();
    let mut unknown: Vec<&LocalModel> = Vec::new();
    for l in locals {
        if l.reference.adapter().is_some() {
            continue;
        }
        match resolve_arch(l) {
            Some(arch) => by_arch.entry(arch.id).or_default().push(l),
            None => unknown.push(l),
        }
    }

    let mut roots = Vec::new();
    for arch in brain_arch::ARCHS {
        if let Some(filter) = arch_filter {
            if arch.id != filter {
                continue;
            }
        }
        let arch_locals = by_arch.get(arch.id).cloned().unwrap_or_default();
        if arch_locals.is_empty() && arch.variants.is_empty() {
            continue;
        }
        if local_only && arch_locals.is_empty() {
            continue;
        }
        roots.push(build_arch_node(arch, &arch_locals, local_only));
    }

    if arch_filter.is_none() && !local_only && !unknown.is_empty() {
        let children = unknown.iter().map(|l| Node::leaf(unresolved_leaf(l))).collect();
        roots.push(Node::branch(format!("local  (custom GGUF/checkpoints of an architecture not in brain_arch, {} found)", unknown.len()), children));
    }

    roots
}

fn unresolved_leaf(l: &LocalModel) -> String {
    let family = l.card.as_ref().map(|c| c.family.as_str()).unwrap_or("?");
    let size = size_of(l).map(human_bytes).unwrap_or_else(|| "?".to_string());
    format!("{}  family={family:?}  {size}  {}", l.reference, l.dir.display())
}

fn build_arch_node(arch: &'static brain_arch::Arch, locals: &[&LocalModel], local_only: bool) -> Node {
    let mut by_repo: BTreeMap<String, Vec<&LocalModel>> = BTreeMap::new();
    for l in locals {
        by_repo.entry(l.reference.base().to_string()).or_default().push(l);
    }
    let mut repo_keys: Vec<String> = by_repo.keys().cloned().collect();
    for v in arch.variants {
        if !by_repo.contains_key(v.reference) {
            repo_keys.push(v.reference.to_string());
        }
    }
    repo_keys.sort();
    repo_keys.dedup();

    let local_repo_count = by_repo.len();
    let mut children = Vec::new();
    for repo in &repo_keys {
        if local_only && !by_repo.contains_key(repo) {
            continue;
        }
        let declared = arch.variants.iter().find(|v| v.reference == repo);
        let repo_locals = by_repo.get(repo).cloned().unwrap_or_default();
        children.push(build_repo_node(arch, repo, declared, &repo_locals, local_only));
    }

    Node::branch(format!("{}  {}  ({} repo{}, {} local)", arch.id, arch.display, repo_keys.len(), if repo_keys.len() == 1 { "" } else { "s" }, local_repo_count), children)
}

fn build_repo_node(arch: &'static brain_arch::Arch, repo: &str, declared: Option<&brain_arch::Variant>, locals: &[&LocalModel], local_only: bool) -> Node {
    let base_local = locals.iter().find(|l| l.reference.quant().is_none()).copied();
    let params = declared.filter(|v| v.params > 0).map(|v| v.params);
    let params_str = params.map(|p| format!("{:.2}B params", p as f64 / 1e9)).unwrap_or_else(|| "params unknown".to_string());
    let pulled = base_local.is_some() || locals.iter().any(|l| l.reference.quant().is_some());
    let header = format!("{repo}  {params_str}  {}", if pulled { "local" } else { "not pulled" });

    let mut leaves = Vec::new();
    if let Some(bl) = base_local {
        leaves.push(leaf_for_local(arch, bl, "base"));
    } else if declared.is_some() && !local_only {
        leaves.push(Node::leaf(not_pulled_leaf(arch.id, repo, "base")));
    }

    let mut seen: HashSet<String> = HashSet::new();
    for l in locals.iter().filter(|l| l.reference.quant().is_some()) {
        let q = l.reference.quant().unwrap();
        seen.insert(q.as_str().to_string());
        leaves.push(leaf_for_local(arch, l, q.as_str()));
    }
    if let Some(v) = declared {
        if !local_only {
            for q in v.quants {
                if seen.contains(*q) {
                    continue;
                }
                leaves.push(Node::leaf(not_pulled_leaf(arch.id, &format!("{repo}-{q}"), q)));
            }
        }
    }

    Node::branch(header, leaves)
}

/// A declared-but-absent row: `ref_str` is the FULL canonical id (repo, or
/// repo-quant), never just the quant token - the same self-contained-leaf
/// rule `leaf_for_local` follows, so a piped `grep` finds a complete line
/// here too, not a fragment that needs its parent for context.
fn not_pulled_leaf(arch_id: &str, ref_str: &str, quant_label: &str) -> String {
    format!("{arch_id} {ref_str:<48} {quant_label:<8} not pulled")
}

fn leaf_for_local(arch: &'static brain_arch::Arch, l: &LocalModel, quant_label: &str) -> Node {
    let ref_str = l.reference.to_string();
    let size = size_of(l);
    let size_str = size.map(human_bytes).unwrap_or_else(|| "?".to_string());
    let cost = cost_str(arch.id, &ref_str, l);
    let line = format!("{} {ref_str:<48} {quant_label:<8} local  {size_str:>10}  {:<14}  {cost}", arch.id, fit_str(size));
    // A pulled model has real tensors to open in the interactive browser's
    // detail view; a declared-but-absent row (`not_pulled_leaf`, no
    // `LocalModel` at all) never gets a `detail_ref` and Enter on it does
    // nothing, which is correct - there is nothing on disk to show.
    Node::leaf_with_detail(line, ref_str)
}

/// `on_enter`, forwarded to [`tree::run_tui`] unchanged - see that
/// function's own doc. Only `list` passes one (built in [`run_list`]);
/// `list-adapters` has nothing further to drill into.
fn render(roots: Vec<Node>, plain: bool, force_tui: bool, json: bool, on_enter: Option<&tree::DetailFn>) -> i32 {
    if json {
        println!("{}", serde_json::to_string_pretty(&node_to_json(&roots)).unwrap_or_default());
        return 0;
    }
    let interactive = force_tui || (!plain && tree::stdout_is_terminal());
    if interactive {
        if let Err(e) = tree::run_tui(roots, on_enter) {
            eprintln!("brain models: {e}");
            return 1;
        }
    } else {
        for line in tree::render_plain(&roots) {
            println!("{line}");
        }
    }
    0
}

fn node_to_json(nodes: &[Node]) -> Value {
    Value::Array(nodes.iter().map(|n| serde_json::json!({"line": n.line, "children": node_to_json(&n.children)})).collect())
}

// -------------------------------------------------------- list-adapters --

fn run_list_adapters(args: &[String]) -> i32 {
    let mut a = Args::new(args);
    let arch_filter = a.take_str("--arch");
    let plain = a.take_flag("--plain");
    let force_tui = a.take_flag("--tui");
    let json = a.take_flag("--json");
    let Some(store) = open_store(&mut a) else {
        a.finish();
        return render(Vec::new(), plain, force_tui, json, None);
    };
    a.finish();

    let locals = store.scan();
    let mut by_arch: BTreeMap<&'static str, BTreeMap<String, Vec<&LocalModel>>> = BTreeMap::new();
    for l in &locals {
        if l.reference.adapter().is_none() {
            continue;
        }
        let Some(arch) = resolve_arch(l) else { continue };
        if let Some(filter) = &arch_filter {
            if arch.id != filter {
                continue;
            }
        }
        by_arch.entry(arch.id).or_default().entry(l.reference.base().to_string()).or_default().push(l);
    }

    let roots = by_arch
        .into_iter()
        .map(|(arch_id, by_base)| {
            let children = by_base
                .into_iter()
                .map(|(base, adapters)| {
                    let leaves = adapters.iter().map(|l| Node::leaf(adapter_leaf(arch_id, l))).collect();
                    Node::branch(base, leaves)
                })
                .collect();
            Node::branch(arch_id.to_string(), children)
        })
        .collect();

    render(roots, plain, force_tui, json, None)
}

fn adapter_leaf(arch_id: &str, l: &LocalModel) -> String {
    let ref_str = l.reference.to_string();
    let size = l.adapter.as_deref().and_then(|p| std::fs::metadata(p).ok()).map(|m| human_bytes(m.len())).unwrap_or_else(|| "?".to_string());
    let card = l.card.as_ref();
    let adapter_meta = card.and_then(|c| c.adapter.as_ref());
    let mut meta = Vec::new();
    if let Some(ad) = adapter_meta {
        meta.push(format!("kind={}", ad.kind));
        if let Some(r) = ad.rank {
            meta.push(format!("rank={r}"));
        }
        if let Some(a) = ad.alpha {
            meta.push(format!("alpha={a}"));
        }
        if let Some(targets) = &ad.targets {
            meta.push(format!("targets={}", targets.join(",")));
        }
        if let Some(ds) = &ad.dataset_id {
            meta.push(format!("dataset={ds}"));
        }
    }
    format!("{arch_id} {ref_str:<48} {size:>10}  {}", meta.join(" "))
}

// ------------------------------------------------------------------- info --

fn run_info(args: &[String]) -> i32 {
    let mut a = Args::new(args);
    let json = a.take_flag("--json");
    let Some(reference) = a.positional() else {
        eprintln!("{USAGE}");
        return 2;
    };
    let Some(store) = open_store(&mut a) else {
        eprintln!("brain models info: no model store configured (see --models-dir / BRAIN_MODELS_DIR)");
        return 1;
    };
    a.finish();

    let nodes = match build_info_nodes(&store, &reference) {
        Ok(nodes) => nodes,
        Err(e) => {
            eprintln!("brain models info: {e}");
            return 1;
        }
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&node_to_json(&nodes)).unwrap_or_default());
        return 0;
    }
    for line in tree::render_plain(&nodes) {
        println!("{line}");
    }
    0
}

/// The guts of `brain models info`: resolve `reference` against `store`,
/// read its tensors header-only, overlay any pulled adapter for the same
/// base repo. Shared with `run_list`'s interactive `on_enter` (see that
/// function's own doc) so the printed command and the browser's detail view
/// can never show different information for the same model.
fn build_info_nodes(store: &Store, reference: &str) -> Result<Vec<Node>, String> {
    let model_ref = brain_modelstore::refurl::parse_model_arg(reference).map_err(|e| e.to_string())?;
    let Some(local) = store.local(&model_ref) else {
        return Err(format!("{model_ref} is not pulled; run `brain pull {model_ref}` first"));
    };
    let path = local.weights.to_str().ok_or_else(|| "non-UTF-8 weights path".to_string())?;
    let reader = checkpoint::weightio::WeightReader::open(path).map_err(|e| format!("{path}: {e}"))?;

    // Any adapter already pulled for this SAME base repo is overlaid onto the
    // tree it targets, colored distinctly - `Store::scan` finds it the same
    // way `list-adapters` does.
    let adapters: Vec<(ModelRef, checkpoint::weightio::WeightReader)> = store
        .scan()
        .into_iter()
        .filter(|l| l.reference.adapter().is_some() && l.reference.base() == model_ref.base())
        .filter_map(|l| {
            let p = l.adapter.as_ref()?.to_str()?.to_string();
            let r = checkpoint::weightio::WeightReader::open(&p).ok()?;
            Some((l.reference.clone(), r))
        })
        .collect();

    let card = local.card.clone().unwrap_or_else(|| ModelCard::new(model_ref.to_string(), "?"));
    let param_count = card.param_count.unwrap_or_else(|| reader.names().filter_map(|n| reader.shape(n)).map(|s| s.iter().product::<u64>()).sum());
    let total_bytes = reader.names().filter_map(|n| reader.nbytes(n)).sum::<u64>();
    let mut header = format!(
        "{}   {:?}   {} tensors   {:.2}B params   {}",
        model_ref,
        local.format,
        reader.names().count(),
        param_count as f64 / 1e9,
        human_bytes(total_bytes)
    );
    if !adapters.is_empty() {
        let names: Vec<String> = adapters.iter().map(|(r, _)| r.adapter().map(|a| a.to_string()).unwrap_or_default()).collect();
        header.push_str(&format!("   adapters: {}", names.join(", ")));
    }

    let mut nodes = vec![Node::leaf(header)];
    nodes.extend(build_tensor_tree(&reader, &adapters));
    Ok(nodes)
}

/// Group tensor names by `.`-separated segment into a tree, each leaf a real
/// tensor with its OWN dtype (a GGUF routinely mixes precision per tensor -
/// see `WeightReader::dtype`'s own doc). Adapter tensors are merged in at the
/// same path their base tensor sits under (or their own path, if the adapter
/// names one the base does not have - a LoRA on a fused/renamed projection),
/// marked with a leading `+` and cyan, matching `caps_cli`'s existing
/// `\x1b[36m` convention for a secondary/overlay element.
fn build_tensor_tree(reader: &checkpoint::weightio::WeightReader, adapters: &[(ModelRef, checkpoint::weightio::WeightReader)]) -> Vec<Node> {
    #[derive(Default)]
    struct Branch {
        children: BTreeMap<String, Branch>,
        leaves: Vec<String>,
    }
    let mut root = Branch::default();
    let insert = |root: &mut Branch, name: &str, line: String| {
        let segs: Vec<&str> = name.split('.').collect();
        let mut cur = root;
        for seg in &segs[..segs.len().saturating_sub(1)] {
            cur = cur.children.entry((*seg).to_string()).or_default();
        }
        cur.leaves.push(line);
    };

    for name in reader.names() {
        let shape = reader.shape(name).map(|s| format!("{s:?}")).unwrap_or_else(|| "?".to_string());
        let dtype = reader.dtype(name).unwrap_or("?");
        let size = reader.nbytes(name).map(human_bytes).unwrap_or_else(|| "?".to_string());
        insert(&mut root, name, format!("{name}  {dtype}  {shape}  {size}"));
    }
    for (aref, areader) in adapters {
        let tag = aref.adapter().map(|a| a.to_string()).unwrap_or_default();
        for name in areader.names() {
            let shape = areader.shape(name).map(|s| format!("{s:?}")).unwrap_or_else(|| "?".to_string());
            let dtype = areader.dtype(name).unwrap_or("?");
            let size = areader.nbytes(name).map(human_bytes).unwrap_or_else(|| "?".to_string());
            insert(&mut root, name, format!("\x1b[36m+ {name}  {dtype}  {shape}  {size}  {tag}\x1b[0m"));
        }
    }

    fn to_nodes(name: &str, b: Branch) -> Node {
        let mut children: Vec<Node> = b.children.into_iter().map(|(k, v)| to_nodes(&k, v)).collect();
        children.extend(b.leaves.into_iter().map(Node::leaf));
        Node::branch(name.to_string(), children)
    }
    root.children.into_iter().map(|(k, v)| to_nodes(&k, v)).chain(root.leaves.into_iter().map(Node::leaf)).collect()
}

// ---------------------------------------------------------------- profile --

fn run_profile(args: &[String]) -> i32 {
    let mut a = Args::new(args);
    let measure = a.take_flag("--measure");
    let reps = a.usize_or("--reps", 5);
    let Some(reference) = a.positional() else {
        eprintln!("{USAGE}");
        return 2;
    };
    let Some(store) = open_store(&mut a) else {
        eprintln!("brain models profile: no model store configured (see --models-dir / BRAIN_MODELS_DIR)");
        return 1;
    };
    a.finish();

    let model_ref = match brain_modelstore::refurl::parse_model_arg(&reference) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("brain models profile: {e}");
            return 2;
        }
    };
    let Some(local) = store.local(&model_ref) else {
        eprintln!("brain models profile: {model_ref} is not pulled; run `brain pull {model_ref}` first");
        return 1;
    };
    let Some(arch) = resolve_arch(&local) else {
        eprintln!("brain models profile: {model_ref}'s architecture is not registered in brain_arch");
        return 1;
    };
    let Some(config) = read_config(&local, arch.id) else {
        eprintln!("brain models profile: {model_ref} is a compound (multi-file) model - no single config to price");
        return 1;
    };

    if measure {
        return run_measure(arch.id, &model_ref.to_string(), &config, reps);
    }

    match modelcost::price_and_cache(arch.id, &model_ref.to_string(), &config) {
        Ok(c) => {
            println!("{model_ref}: {}", format_cached(&c));
            0
        }
        Err(e) => {
            eprintln!("brain models profile: {model_ref}: {e}");
            1
        }
    }
}

/// `--measure`: builds the REAL model and runs it - see
/// `modelcost::Measurement`'s own doc for exactly what each number means
/// (cold vs. hot, why load time is separate, why lfm2's per-layer figure is
/// an average and every other registered architecture's is derived).
fn run_measure(arch_id: &str, ref_str: &str, config: &serde_json::Value, reps: usize) -> i32 {
    let m = match modelcost::measure(arch_id, config, reps) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("brain models profile --measure: {ref_str}: {e}");
            return 1;
        }
    };
    println!("{ref_str}");
    println!("  load:  {:.3} ms  (weight upload + pipeline build, once)", m.load_seconds * 1e3);
    println!("  cold:  {:.3} ms  (first forward pass)", m.cold_seconds * 1e3);
    println!("  hot:   {:.3} ms  (best of {reps} subsequent pass{})", m.hot_seconds * 1e3, if reps == 1 { "" } else { "es" });
    println!(
        "  total: {}  ({:.0}% covered)  ->  {} cold, {} hot",
        eng(m.total.flops as f64, "FLOP"),
        m.total.coverage * 100.0,
        eng(m.total.flops as f64 / m.cold_seconds.max(f64::MIN_POSITIVE), "FLOP/s"),
        eng(m.total.flops as f64 / m.hot_seconds.max(f64::MIN_POSITIVE), "FLOP/s"),
    );
    println!("  per layer: {}", eng(m.per_layer.flops as f64, "FLOP"));
    0
}
