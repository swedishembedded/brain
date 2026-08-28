// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The one shared model-pricing registry and on-disk cache.
//!
//! `brain flops` was the first version of this: real dry-recorded pricing
//! (`gpu_core::cost::Recording`), just private to `crates/cli/src/flops_cli.rs`
//! and wired for a handful of architectures by hand, with nothing else able to
//! reuse it. This crate pulls the pricing itself out from under that CLI
//! module - mirroring how `crates/catalog` already centralises per-model
//! constructors above `capability` - so `brain flops` and `brain models
//! list`/`brain models profile` become front ends over ONE engine and ONE
//! cache, instead of two independent implementations that could disagree.
//!
//! ## Two tiers
//!
//! **Exact** ([`CostEntry::price`]): builds a real (zero-init, no weights
//! needed - cost is a function of shape, never buffer contents) model at the
//! architecture's OWN config and reports its forward-pass [`CostReport`] via
//! `Model::cost_fwd`. Needs that architecture to have wired a `Recording`
//! through its forward pass; today that is qwen3/gpt2/lfm2, the three
//! `brain flops` already covered without a generation-shape parameter
//! (width/height/steps) a bare model config cannot supply.
//!
//! **Bandwidth** ([`CostEntry::tensor_manifest`]): sums tensor byte sizes from
//! an architecture's own shape manifest - no device, no model build,
//! microseconds. This is deliberately NOT a FLOP estimate: a GGUF quant
//! dequantizes to fp32 at kernel launch in brain's engine, so quantization
//! changes *bytes moved*, not floating-point operation count, and a `2 x
//! params` FLOP guess would be wrong for a conv/diffusion/MoE architecture's
//! real op mix in a way `crates/gpu-core`'s own "coverage is honest, an
//! uncovered kernel is excluded, never counted as zero" discipline exists to
//! prevent. This tier is what makes a size/fit column - not a FLOPs column -
//! available for every architecture whose config already exposes
//! `param_list`/`tensor_manifest`, whether or not it has an exact pricer.
//!
//! Registering an exact pricer for one more architecture is a one-file,
//! additive change to [`registry`] - the same shape as `brain_arch`'s own
//! "adding a model means adding its row here."
//!
//! ## The cache
//!
//! `gpu_core::cache_dir()/models/cost.json` (reusing the one resolver every
//! other on-disk cache in this workspace already shares - never re-derived).
//! Keyed by `(arch, variant_ref)`; each entry carries a hash of the config it
//! was priced from, so a changed shape invalidates it rather than serving a
//! stale number. An unknown schema or a hash mismatch drops the entry and
//! reports "not profiled" - the same rule `gpu_core::roof`'s own persisted
//! store follows for a corrupt or pre-schema record. Written via a temp file
//! plus atomic rename.
//!
//! Device roofline itself is NOT re-cached here - `gpu_core::roof::known`/
//! `ensure`/`reprofile` already do exactly that, and combining a cached
//! [`CostSummary`] with a cached `Roofs` (bytes/gbs, flops/gflops) is a pure
//! computation a caller does at render time. Two caches would only invite the
//! two disagreeing.

use std::collections::BTreeMap;
use std::path::PathBuf;

use gpu_core::cost::CostReport;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Exact tier - see the module doc. Takes the model's own config (as read
/// header-only from a real checkpoint, or a declared variant's synthetic
/// stand-in), returns one forward pass's cost at batch 1, the config's own
/// context length, inference precision.
pub type PriceFn = fn(&Value) -> Result<CostReport, String>;

/// Bandwidth tier - see the module doc. Returns `(tensor name, element
/// count)` pairs; this crate multiplies by 4 bytes/element (every registered
/// manifest reports fp32 shapes, matching brain's own dequantize-on-load
/// engine) to get the byte total.
pub type TensorManifestFn = fn(&Value) -> Result<Vec<(String, usize)>, String>;

/// Real, on-device measurement - see [`Measurement`]. Unlike [`PriceFn`]
/// (dry, no device, cacheable indefinitely against a config hash), this
/// ACTUALLY builds the model and runs it `reps` times; never cached, because
/// a timing is a fact about THIS machine right now, not about the model.
pub type MeasureFn = fn(&Value, usize) -> Result<Measurement, String>;

/// One architecture's pricing capability. Every field is independently
/// optional - an architecture may have any subset registered.
pub struct CostEntry {
    /// Must match `brain_arch::Arch::id` - this crate does not depend on
    /// `brain-arch` (a plain string keeps `modelcost` a leaf the CLI composes,
    /// not another spoke that has to agree with arch's own dependency shape),
    /// so that agreement is a convention checked at the CLI layer, where both
    /// crates are already in scope.
    pub arch: &'static str,
    pub price: Option<PriceFn>,
    pub tensor_manifest: Option<TensorManifestFn>,
    pub measure: Option<MeasureFn>,
}

/// A real, timed measurement of one forward pass - see [`CostEntry::measure`].
///
/// `cold_seconds` is the FIRST pass after the model is built (pipeline
/// specialisation / first-touch allocation still pending); `hot_seconds` is
/// the best of the passes after that, at steady state - the two are kept
/// separate because conflating them either overstates steady-state
/// throughput (if cold pollutes the average) or hides a real specialisation
/// cost a user paying for exactly one pass (a cold start) would still incur.
/// `load_seconds` is measured separately again: weight upload + model/pipeline
/// construction is not part of EITHER pass's own cost, and folding it in
/// would make "seconds per forward pass" answer a different question
/// ("seconds including a one-time setup that amortises over many requests")
/// than the one it names.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Measurement {
    pub load_seconds: f64,
    pub cold_seconds: f64,
    pub hot_seconds: f64,
    /// The whole model's forward-pass cost - identical to what the exact
    /// tier's [`PriceFn`] would report for the same config.
    pub total: CostSummary,
    /// One transformer layer's own cost. For a UNIFORM stack (qwen3, gpt2 -
    /// every layer identical), this is DERIVED, never measured or guessed:
    /// dry probes at 0/1/2 layers, verified affine at the point outside that
    /// basis (`pricers::per_unit_cost`), so `total.flops / per_layer.flops`
    /// recovers the exact layer count. For a HYBRID stack whose layers are
    /// not interchangeable (lfm2 - a per-layer choice of conv vs attention;
    /// see `pricers::measure_lfm2`'s own doc for why probing depth would
    /// silently mix layer types), it is the AVERAGE (`total / n_layers`)
    /// instead - a real number, just a coarser one, and never presented as
    /// the derived kind.
    pub per_layer: CostSummary,
}

mod pricers;

/// The registry, in no particular order (`by_arch` is a linear scan over a
/// short list, same as `brain_arch::by_id`).
pub fn registry() -> &'static [CostEntry] {
    &[
        CostEntry { arch: "qwen3", price: Some(pricers::price_qwen3), tensor_manifest: Some(pricers::manifest_qwen3), measure: Some(pricers::measure_qwen3) },
        CostEntry { arch: "gpt2", price: Some(pricers::price_gpt2), tensor_manifest: Some(pricers::manifest_gpt2), measure: Some(pricers::measure_gpt2) },
        CostEntry { arch: "lfm2", price: Some(pricers::price_lfm2), tensor_manifest: Some(pricers::manifest_lfm2), measure: Some(pricers::measure_lfm2) },
        // qwen35/qwen35moe have no wired `Recording` through their forward
        // pass yet (no `cost_fwd`, unlike qwen3/gpt2/lfm2) - bandwidth tier
        // only until one of them gains it.
        CostEntry { arch: "qwen35", price: None, tensor_manifest: Some(pricers::manifest_qwen35), measure: None },
        CostEntry { arch: "qwen35moe", price: None, tensor_manifest: Some(pricers::manifest_qwen35moe), measure: None },
    ]
}

pub fn by_arch(id: &str) -> Option<&'static CostEntry> {
    registry().iter().find(|e| e.arch == id)
}

/// A cached pricing result's numeric core - deliberately NOT the full
/// [`CostReport`] (its per-kernel `by_kernel`/`uncovered` maps are a `--
/// per-kernel` presentation detail `brain flops` still computes fresh; a
/// `models list` row only ever renders the totals).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CostSummary {
    pub flops: u64,
    pub int_ops: u64,
    pub bytes: u64,
    /// `covered / steps`, `1.0` when `steps == 0` - see `CostReport::coverage`.
    pub coverage: f64,
}

impl From<&CostReport> for CostSummary {
    fn from(r: &CostReport) -> CostSummary {
        CostSummary { flops: r.total.flops, int_ops: r.total.int_ops, bytes: r.total.bytes, coverage: r.coverage() }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Tier {
    /// A real dry-recorded forward pass - [`CostEntry::price`].
    Exact,
    /// Tensor bytes summed from a shape manifest, no FLOPs - see the module
    /// doc for why this is never presented as one. [`CostEntry::tensor_manifest`].
    Bandwidth,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CachedCost {
    pub tier: Tier,
    pub summary: CostSummary,
    /// A hash of the config this was priced from - see [`config_hash`]. A live
    /// config whose hash no longer matches invalidates the entry rather than
    /// serving a stale number for a repo whose shape has since changed.
    pub config_hash: u64,
}

/// A cheap, deterministic hash of the config a pricing was derived from -
/// FNV-1a over the config's canonical (sorted-key) JSON bytes, matching
/// `gpu_core::tune::source_fingerprint`'s algorithm (no new hashing
/// convention introduced for this one cache).
pub fn config_hash(config: &Value) -> u64 {
    // `serde_json::to_vec` on a `Value` built from a `BTreeMap`/object walk is
    // already key-sorted for a `Value::Object` (serde_json's default map is
    // itself a `BTreeMap` unless the `preserve_order` feature is enabled,
    // which nothing in this workspace turns on), so this is stable across
    // process runs regardless of the source's own key order.
    let bytes = serde_json::to_vec(config).unwrap_or_default();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Read-only: whatever is cached for `(arch, variant_ref)`, without pricing
/// anything. `None` on a cache miss, a corrupt/unknown-schema store, OR a
/// config-hash mismatch (a live config that no longer matches what was
/// priced) - all three collapse to the same "not profiled" a caller renders,
/// deliberately: none of them is a case worth distinguishing over a silently
/// stale number.
pub fn cached(arch: &str, variant_ref: &str, config: &Value) -> Option<CachedCost> {
    let store = load_store()?;
    let entry = store.get(&cache_key(arch, variant_ref))?;
    (entry.config_hash == config_hash(config)).then_some(*entry)
}

/// Price `(arch, variant_ref)` now (exact tier if registered, else bandwidth,
/// else neither) and persist the result.
///
/// `Err` distinguishes two callers must not conflate: no pricer of either
/// tier is registered for `arch` at all (a coverage gap - "add a row to
/// `registry()`"), vs. a pricer IS registered but `config` failed its own
/// validation (e.g. `QwenConfig::from_json_checked` refusing a config
/// missing a shape key - "this specific config is wrong", not "this
/// architecture isn't supported"). Conflating the two is exactly how a
/// silently-mis-keyed config used to get priced as an unrelated model with
/// `brain models profile` reporting total confidence in the wrong number.
pub fn price_and_cache(arch: &str, variant_ref: &str, config: &Value) -> Result<CachedCost, String> {
    let entry = by_arch(arch).ok_or_else(|| format!("no cost model registered for architecture {arch:?}"))?;
    let cached = if let Some(price) = entry.price {
        let report = price(config)?;
        CachedCost { tier: Tier::Exact, summary: CostSummary::from(&report), config_hash: config_hash(config) }
    } else {
        return price_and_cache_bandwidth_only(arch, variant_ref, config);
    };
    save(arch, variant_ref, cached);
    Ok(cached)
}

/// The bandwidth tier ONLY, even when an exact pricer is registered - never
/// builds a model, so its cost is a few tensor-shape multiplications
/// regardless of how large the real checkpoint is.
///
/// This is the one an unattended BULK walk over every local model must call
/// (`brain models list --reprofile`'s pass over the whole store) - the exact
/// tier's `price` materializes a real zero-init weight set at the config's
/// own shape (no data is read, but the buffer itself is real host memory),
/// which is a bounded, deliberate cost for ONE model a human explicitly named
/// (`brain flops`, `brain models profile <ref>`) and an unbounded, surprising
/// one multiplied across however many multi-billion-parameter checkpoints
/// happen to be sitting in the store - exactly the "surprising use of
/// memory/time during an unattended scan" this workspace's own conventions
/// avoid elsewhere, in the GGUF-import scan for the same reason.
pub fn price_and_cache_bandwidth_only(arch: &str, variant_ref: &str, config: &Value) -> Result<CachedCost, String> {
    let entry = by_arch(arch).ok_or_else(|| format!("no cost model registered for architecture {arch:?}"))?;
    let manifest = entry.tensor_manifest.ok_or_else(|| format!("architecture {arch:?} has no bandwidth-tier pricer registered"))?;
    let bytes: u64 = manifest(config)?.into_iter().map(|(_, numel)| numel as u64 * 4).sum();
    let cached = CachedCost { tier: Tier::Bandwidth, summary: CostSummary { bytes, ..CostSummary::default() }, config_hash: config_hash(config) };
    save(arch, variant_ref, cached);
    Ok(cached)
}

/// Cache an ALREADY-COMPUTED exact-tier report, without recomputing it -
/// what `brain flops` calls after pricing a real on-disk model itself (it has
/// its own richer flag surface - `--train`/`--i8`/`--stages` - that this
/// crate's own [`CostEntry::price`] deliberately does not replicate; see the
/// module doc), so the two commands share ONE cache without a second, slower
/// derivation.
pub fn cache_report(arch: &str, variant_ref: &str, config: &Value, report: &CostReport) {
    save(arch, variant_ref, CachedCost { tier: Tier::Exact, summary: CostSummary::from(report), config_hash: config_hash(config) });
}

/// Build `arch`'s model for real and time it - see [`Measurement`] and
/// [`CostEntry::measure`]. `reps` is how many HOT passes to time (the best of
/// them is kept); the cold (first) pass always runs once regardless. Never
/// cached - a timing describes THIS run on THIS machine, not the model.
pub fn measure(arch: &str, config: &Value, reps: usize) -> Result<Measurement, String> {
    let f = by_arch(arch).ok_or_else(|| format!("no cost model registered for architecture {arch:?}"))?.measure.ok_or_else(|| format!("architecture {arch:?} has no real-execution measurement registered"))?;
    f(config, reps)
}

// ------------------------------------------------------------------ store --

#[derive(Default, Serialize, Deserialize)]
struct Store {
    schema: u32,
    entries: BTreeMap<String, CachedCost>,
}

const SCHEMA: u32 = 1;

fn cache_key(arch: &str, variant_ref: &str) -> String {
    format!("{arch}\t{variant_ref}")
}

fn cache_path() -> Option<PathBuf> {
    let root = CACHE_DIR_OVERRIDE.with(|c| c.borrow().clone()).or_else(gpu_core::cache_dir)?;
    Some(root.join("models").join("cost.json"))
}

fn load_store() -> Option<BTreeMap<String, CachedCost>> {
    let path = cache_path()?;
    let text = std::fs::read_to_string(path).ok()?;
    let store: Store = serde_json::from_str(&text).ok()?;
    // An unknown schema is dropped, never migrated-on-the-fly - the same rule
    // `roof::persist::RoofStore::load` applies to its own file.
    (store.schema == SCHEMA).then_some(store.entries)
}

fn save(arch: &str, variant_ref: &str, entry: CachedCost) {
    let Some(path) = cache_path() else { return };
    let mut entries = load_store().unwrap_or_default();
    entries.insert(cache_key(arch, variant_ref), entry);
    let store = Store { schema: SCHEMA, entries };
    let Ok(body) = serde_json::to_string_pretty(&store) else { return };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let tmp = path.with_extension("tmp");
    if std::fs::write(&tmp, body).is_ok() {
        let _ = std::fs::rename(&tmp, &path);
    }
}

/// Redirect where the cache persists (`None` restores the default
/// resolution). A TEST seam, matching `gpu_core::roof::set_cache_dir` - tests
/// point this at a temp dir so they never touch the developer's real
/// `~/.cache/brain`.
pub fn set_cache_dir_override(dir: Option<PathBuf>) {
    CACHE_DIR_OVERRIDE.with(|c| *c.borrow_mut() = dir);
}

thread_local! {
    static CACHE_DIR_OVERRIDE: std::cell::RefCell<Option<PathBuf>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Isolate one test's cache from the developer's real `~/.cache/brain`
    /// AND from every other test in this file (each gets its own scratch
    /// dir; `set_cache_dir_override` is thread-local, so this is safe under
    /// parallel `cargo test` without a lock, unlike `gpu_core::roof`'s
    /// process-global override).
    struct Scratch(PathBuf);
    impl Scratch {
        fn new(name: &str) -> Scratch {
            let dir = std::env::temp_dir().join(format!("brain-modelcost-test-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            set_cache_dir_override(Some(dir.clone()));
            Scratch(dir)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            set_cache_dir_override(None);
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn registry_arch_ids_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for e in registry() {
            assert!(seen.insert(e.arch), "duplicate arch id {:?} in registry()", e.arch);
        }
    }

    #[test]
    fn by_arch_finds_a_known_row_and_none_for_unknown() {
        assert_eq!(by_arch("qwen3").map(|e| e.arch), Some("qwen3"));
        assert!(by_arch("totally-unknown").is_none());
    }

    #[test]
    fn every_registered_manifest_reports_at_least_one_tensor_on_its_own_tiny_config() {
        // Not empty, and not degenerate (a config-parsing bug that silently
        // produced a zero-tensor model would still "succeed" here otherwise).
        let cases: &[(&str, Value)] = &[
            ("qwen3", qwen3::QwenConfig::tiny().to_json()),
            ("gpt2", gpt2::GptConfig::tiny().to_json()),
            ("lfm2", lfm2::config::LfmConfig::tiny().to_json()),
            ("qwen35", qwen35::config::Qwen35Config::tiny().to_json()),
            ("qwen35moe", qwen35moe::config::Qwen35Config::tiny().to_json()),
        ];
        for (arch, cfg) in cases {
            let entry = by_arch(arch).unwrap_or_else(|| panic!("{arch} is not registered"));
            let manifest = entry.tensor_manifest.unwrap_or_else(|| panic!("{arch} has no tensor_manifest registered"));
            let tensors = manifest(cfg).unwrap_or_else(|e| panic!("{arch}: {e}"));
            assert!(!tensors.is_empty(), "{arch}: tensor_manifest reported zero tensors on its own tiny() config");
            assert!(tensors.iter().all(|(_, n)| *n > 0), "{arch}: every tensor should have a positive element count");
        }
    }

    #[test]
    fn measure_qwen3_reports_positive_timings_and_a_consistent_per_layer_split() {
        let cfg = qwen3::QwenConfig::tiny().to_json();
        let m = measure("qwen3", &cfg, 2).expect("qwen3 has a registered measure fn");
        assert!(m.load_seconds >= 0.0);
        assert!(m.cold_seconds > 0.0, "a real forward pass must take measurable time");
        assert!(m.hot_seconds > 0.0);
        assert!(m.total.flops > 0);
        assert!(m.per_layer.flops > 0);
        // Homogeneous stack: `n_layers` copies of one layer's cost must fit
        // inside the whole model's cost, with the remainder being embed/head
        // (the 0-layer base) - never more than the total (that would mean
        // the derivation double-counted something).
        let n_layers = qwen3::QwenConfig::tiny().n_layers as u64;
        assert!(m.per_layer.flops * n_layers <= m.total.flops, "n_layers * per_layer must not exceed the whole model's cost");
    }

    #[test]
    fn measure_lfm2_per_layer_is_the_average_not_a_fabricated_exact_value() {
        let cfg = lfm2::config::LfmConfig::tiny();
        let n_layers = cfg.layer_types.len() as u64;
        let m = measure("lfm2", &cfg.to_json(), 1).expect("lfm2 has a registered measure fn");
        assert_eq!(m.per_layer.flops, m.total.flops / n_layers, "lfm2's hybrid stack must report an average, computed exactly as total/n_layers");
        assert!(m.cold_seconds > 0.0);
    }

    #[test]
    fn measure_errors_for_an_unregistered_architecture() {
        let err = measure("totally-unknown", &serde_json::json!({}), 1).expect_err("no measure fn is registered for this arch");
        assert!(err.contains("totally-unknown"));
    }

    #[test]
    fn measure_errors_for_an_architecture_with_no_measure_fn_registered() {
        // qwen35 has a bandwidth-tier manifest but no `cost_fwd`/measure path.
        let cfg = qwen35::config::Qwen35Config::tiny().to_json();
        let err = measure("qwen35", &cfg, 1).expect_err("qwen35 has no measure fn registered");
        assert!(err.contains("qwen35"));
    }

    #[test]
    fn measure_propagates_a_config_validation_error() {
        let bad = serde_json::json!({"vocab": 16});
        let err = measure("qwen3", &bad, 1).expect_err("a config missing shape keys must be refused before any device work");
        assert!(err.contains("vocab_size"));
    }

    #[test]
    fn bandwidth_tier_byte_total_matches_a_hand_summed_manifest() {
        let cfg = qwen35::config::Qwen35Config::tiny().to_json();
        let manifest = by_arch("qwen35").unwrap().tensor_manifest.unwrap();
        let tensors = manifest(&cfg).unwrap();
        let expect_bytes: u64 = tensors.iter().map(|(_, n)| *n as u64 * 4).sum();

        let _scratch = Scratch::new("bandwidth-byte-total");
        let priced = price_and_cache("qwen35", "test/qwen35-tiny", &cfg).expect("qwen35 has a bandwidth pricer");
        assert_eq!(priced.tier, Tier::Bandwidth);
        assert_eq!(priced.summary.bytes, expect_bytes);
        assert_eq!(priced.summary.flops, 0, "the bandwidth tier must never report a fabricated FLOP count");
    }

    #[test]
    fn exact_tier_reports_nonzero_flops_for_a_real_forward_pass() {
        let cfg = qwen3::QwenConfig::tiny().to_json();
        let _scratch = Scratch::new("exact-nonzero-flops");
        let priced = price_and_cache("qwen3", "test/qwen3-tiny", &cfg).expect("qwen3 has an exact pricer");
        assert_eq!(priced.tier, Tier::Exact);
        assert!(priced.summary.flops > 0, "a real transformer forward pass must cost more than zero FLOPs");
        assert!(priced.summary.bytes > 0);
    }

    #[test]
    fn price_and_cache_errors_for_an_unregistered_architecture() {
        let _scratch = Scratch::new("unregistered-arch");
        let err = price_and_cache("totally-unknown", "test/x", &serde_json::json!({})).expect_err("no pricer is registered for this arch");
        assert!(err.contains("totally-unknown"));
    }

    #[test]
    fn price_and_cache_errors_distinctly_for_a_config_that_fails_its_own_validation() {
        // A REGISTERED architecture, but a config missing shape keys - this
        // must not be conflated with "architecture not supported": it is
        // exactly the failure mode that let a silently-mis-keyed config get
        // priced as a different model with total reported confidence.
        let _scratch = Scratch::new("bad-config");
        let bad = serde_json::json!({"vocab": 16, "block_size": 32});
        let err = price_and_cache("qwen3", "test/x", &bad).expect_err("a config missing shape keys must be refused, not silently defaulted");
        assert!(err.contains("vocab_size"), "the error must name the real missing key: {err:?}");
    }

    #[test]
    fn cached_round_trips_what_price_and_cache_just_wrote() {
        let cfg = qwen3::QwenConfig::tiny().to_json();
        let _scratch = Scratch::new("round-trip");
        assert_eq!(cached("qwen3", "test/qwen3-tiny", &cfg), None, "nothing primed yet");
        let priced = price_and_cache("qwen3", "test/qwen3-tiny", &cfg).unwrap();
        assert_eq!(cached("qwen3", "test/qwen3-tiny", &cfg), Some(priced));
    }

    #[test]
    fn a_config_hash_mismatch_invalidates_the_cache_entry() {
        // A model's shape changed since it was priced (more layers, a wider
        // d_model, ...) - the stale entry must not be served as if it still
        // described the live config.
        let cfg_a = qwen3::QwenConfig::tiny().to_json();
        let mut cfg_b = qwen3::QwenConfig::tiny();
        cfg_b.n_layers += 1;
        let cfg_b = cfg_b.to_json();
        assert_ne!(config_hash(&cfg_a), config_hash(&cfg_b), "the two configs must actually differ for this test to mean anything");

        let _scratch = Scratch::new("hash-mismatch");
        price_and_cache("qwen3", "test/qwen3-tiny", &cfg_a).unwrap();
        assert!(cached("qwen3", "test/qwen3-tiny", &cfg_a).is_some());
        assert_eq!(cached("qwen3", "test/qwen3-tiny", &cfg_b), None, "a changed config must not read back the old entry's numbers");
    }

    #[test]
    fn a_corrupt_or_unknown_schema_cache_file_is_ignored_not_trusted() {
        let cfg = qwen3::QwenConfig::tiny().to_json();
        let scratch = Scratch::new("corrupt-cache");
        let path = scratch.0.join("models").join("cost.json");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        std::fs::write(&path, b"not json at all").unwrap();
        assert_eq!(cached("qwen3", "test/qwen3-tiny", &cfg), None, "garbage bytes must read back as a miss, never panic or fabricate a value");
        // price_and_cache must still succeed by recomputing, not propagate the
        // corruption forward.
        let priced = price_and_cache("qwen3", "test/qwen3-tiny", &cfg);
        assert!(priced.is_ok());

        std::fs::write(&path, br#"{"schema":9999,"entries":{}}"#).unwrap();
        assert_eq!(cached("qwen3", "test/qwen3-tiny", &cfg), None, "an unrecognised schema version must be dropped, never partially trusted");
    }

    #[test]
    fn cache_report_stores_an_already_computed_report_without_recomputing() {
        // `brain flops` calls this with a report it already built under its
        // own richer flags (--stages/--weights/...) - `cached()` must read it
        // back exactly, keyed the same way `price_and_cache` keys its own
        // writes.
        let cfg = qwen3::QwenConfig::tiny().to_json();
        let _scratch = Scratch::new("cache-report");
        let report = pricers::price_qwen3(&cfg).unwrap();
        cache_report("qwen3", "Qwen/Qwen3-0.6B", &cfg, &report);
        let got = cached("qwen3", "Qwen/Qwen3-0.6B", &cfg).expect("cache_report must have written a readable entry");
        assert_eq!(got.tier, Tier::Exact);
        assert_eq!(got.summary, CostSummary::from(&report));
    }

    #[test]
    fn bandwidth_only_never_reports_exact_even_when_a_price_fn_is_registered() {
        // qwen3 has BOTH tiers registered - this entry point must still only
        // ever touch tensor_manifest, never `price` (which would materialize
        // a real zero-init weight buffer at the config's shape).
        let cfg = qwen3::QwenConfig::tiny().to_json();
        let _scratch = Scratch::new("bandwidth-only-forces-tier");
        let priced = price_and_cache_bandwidth_only("qwen3", "test/qwen3-tiny", &cfg).unwrap();
        assert_eq!(priced.tier, Tier::Bandwidth);
        assert_eq!(priced.summary.flops, 0);
        assert!(priced.summary.bytes > 0);
    }

    #[test]
    fn config_hash_is_stable_regardless_of_source_key_order() {
        let a = serde_json::json!({"a": 1, "b": 2});
        let b = serde_json::json!({"b": 2, "a": 1});
        assert_eq!(config_hash(&a), config_hash(&b));
    }
}
