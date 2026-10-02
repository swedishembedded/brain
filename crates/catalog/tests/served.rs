// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What the catalog promises about the models it serves: every model declares
//! how it finds its weights, every residency adapter belongs to a listed model,
//! and no model is registered through both of the scheduler's claim paths.


use capability::Assembly;
use catalog::{manifests, multi_residents, provider_from_assembly, residents};

/// A placeholder [`Assembly`] for an entry that reads none.
fn empty_assembly() -> Assembly {
Assembly { id: String::new(), arch: String::new(), variant: None, roles: Default::default(), provenance: Vec::new() }
}

/// A resolved [`Assembly`]'s `weights` role, as a loader-ready path -
/// no `id`/`variant`/`provenance` beyond what the assertion below reads.
fn assembly_with_weights(arch: &str, weights: &str) -> Assembly {
    Assembly {
        id: format!("local/{arch}"),
        arch: arch.to_string(),
        variant: None,
        roles: std::collections::BTreeMap::from([("weights".to_string(), std::path::PathBuf::from(weights))]),
        provenance: Vec::new(),
    }
}

/// The SERVED half of "resolver-migrated", and the one a CLI test cannot
/// see.
///
/// `crate::resolve::dispatch_arch` hands `provider_from_assembly` a REAL
/// assembly it resolved itself, so a migrated architecture works from the
/// command line as soon as its `ModelEntry.provider` reads one. Every
/// other caller - D-Bus, HTTP, `build_executor` - goes through
/// [`provider`] instead, which resolves via [`catalog::resolver_spec_for`]
/// and falls back to [`empty_assembly`] for anything absent from it.
///
/// So an entry whose provider reads a role while its model carries no
/// `catalog::ModelEntry::spec` is broken in exactly one direction: fine on
/// the CLI, "assembly 'local/…' has no <role> role" when served. This
/// asserts the two agree, by CONSTRUCTING every listed model against an
/// empty assembly and requiring that anything which needs a role is
/// registered to get one.
#[test]
fn every_assembly_reading_provider_is_registered_for_the_served_path() {
    let empty = empty_assembly();
    let mut unregistered = Vec::new();
    for m in manifests() {
        let id = m.model;
        // Does this entry's provider actually need a resolved role?
        let needs_role = match provider_from_assembly(&id, &empty) {
            Err(e) => e.contains("has no") && e.contains("role"),
            Ok(_) => false,
        };
        if needs_role && catalog::resolver_spec_for(&id).is_none() {
            unregistered.push(id);
        }
    }
    assert!(
        unregistered.is_empty(),
        "these models build their provider from an Assembly but their catalog entry carries no spec, so every served surface hands them an empty one: {unregistered:?}"
    );
}

/// Every catalog model is in exactly ONE of three declared states for
/// how it finds its weights, and a model in none of them fails here
/// rather than at a user's first run.
///
/// 1. **Resolver-backed** - the provider reads a role off the `Assembly`,
///    and its catalog entry's `spec` supplies a real one. Nothing to configure:
///    the checkpoint is found by scanning the model store.
/// 2. **Weights as an action parameter** - the provider needs none, and
///    the action declares a `host_env` param the caller may pass
///    explicitly or let `ActionSpec::validate` fill from that variable.
/// 3. **Env-only provider** - listed by name below, with the reason it
///    cannot classify from disk.
///
/// The point is that state 3 is a CHOICE someone made and wrote down,
/// not a default a new model falls into by being added. Together with
/// `every_assembly_reading_provider_is_registered_for_the_served_path`
/// (which catches a HALF-migrated model), this makes "weights stopped
/// being findable and nobody noticed" a compile-time-adjacent failure
/// instead of a support question.
#[test]
fn every_model_declares_how_it_finds_its_weights() {
    /// Models that legitimately still take their weights from the
    /// environment, each with the reason it cannot classify from disk.
    const ENV_ONLY: &[(&str, &str)] = &[
        (
            "deepseek-ai/DeepSeek-OCR-2",
            "no vendor-published GGUF exists; community conversions vary in layout and carry no \
             declaration to classify on, so there is nothing stable for an ArchSpec to key off yet",
        ),
        (
            "brain/ltxv",
            "`ltxv::spec` exists and the CLI resolves through it, but the served manifest declares no \
             weights param at all - t2v's default is the tiny random-weight smoke config, and the real \
             22B path is reached by `--dit`/$BRAIN_LTXV_DIT on the dedicated CLI only. Registering it \
             here means giving `LtxvProvider` the roles first, which belongs with making the real-weight \
             path servable rather than with this gate",
        ),
    ];

    /// Models that need no weights at all: pure composition over OTHER
    /// models' weights, or a deterministic mock.
    const NO_WEIGHTS: &[&str] = &[
        "brain/imgpipe",
        "brain/imageops",
        "brain/demo",
        "brain/mock",
        // A rasterizer, not a learned model: `render` takes the scene
        // itself as an `--in scene=` PLY blob and `fit` optimizes one.
        "brain/splat",
    ];

    let empty = empty_assembly();
    let mut undeclared = Vec::new();
    for m in manifests() {
        let id = m.model.clone();
        if NO_WEIGHTS.contains(&id.as_str()) || ENV_ONLY.iter().any(|(n, _)| *n == id) {
            continue;
        }
        // State 1: the provider needs a resolved role, and gets one.
        if catalog::resolver_spec_for(&id).is_some() {
            continue;
        }
        // State 2: some action names a host-side weights param.
        if m.actions.iter().any(|a| a.params.iter().any(|p| p.host_env.is_some() || p.host_resolved)) {
            continue;
        }
        // A provider that cannot even be built from an empty assembly is
        // half-migrated; that is the sibling test's business, not this one.
        if provider_from_assembly(&id, &empty).is_err() {
            continue;
        }
        undeclared.push(id);
    }
    assert!(
        undeclared.is_empty(),
        "these models are in none of the three declared weight-acquisition states - make one true, \
         or add the model to ENV_ONLY with the reason: {undeclared:?}"
    );
}

/// Every residency adapter reachable from this file advertises an id this
/// file also lists, so a model cannot be schedulable but undiscoverable.
#[test]
fn every_residency_adapter_here_is_also_listed_here() {
    let catalog: std::collections::HashSet<String> = manifests().into_iter().map(|m| m.model).collect();
    // Both claim paths, or the multi-device half is exactly as unguarded as
    // the single-device half was before this test existed.
    let store = loader::model_dir::resolve(None);
    let single = residents(store.as_deref()).into_iter().map(|r| r.manifest().model);
    let multi = multi_residents(store.as_deref(), &[(0, 24u64 << 30)], 2u64 << 30).into_iter().map(|r| r.manifest().model);
    for id in single.chain(multi) {
        assert!(catalog.contains(&id), "residency adapter '{id}' is not in the catalog");
    }
}

/// A model registered through BOTH claim paths would have its budget
/// charged twice and its `activate` reachable through a path it correctly
/// refuses - see `residency::Executor::register_multi`'s own doc. The two
/// lists are derived from one field precisely so this cannot happen, and
/// this test is what says so out loud.
#[test]
fn no_model_is_registered_through_both_claim_paths() {
    let store = loader::model_dir::resolve(None);
    let single: std::collections::HashSet<String> = residents(store.as_deref()).into_iter().map(|r| r.manifest().model).collect();
    for r in multi_residents(store.as_deref(), &[(0, 24u64 << 30)], 2u64 << 30) {
        let id = r.manifest().model;
        assert!(!single.contains(&id), "'{id}' is registered as BOTH a single- and a multi-device resident");
    }
}

/// `crates/supir` links no VLM, so its optional caption auto-fill names
/// LLaVA by STRING (`supir::caps::LLAVA_MODEL`) - this is the other half
/// of that decision: the CLI sees both real constants, so it asserts the
/// string still names the real catalog entry - the same drift class
/// `imgpipe_stage_ids_match_the_catalog` guards against.
#[test]
fn supir_llava_model_id_matches_the_catalog() {
    assert_eq!(supir::caps::LLAVA_MODEL, llava::caps::MODEL);
}

/// `sam2`/`qwen3asr`/`nemotronasr` all defer their real checkpoint load
/// past construction (`Sam2Provider::new`/[`LazyProvider`] both load lazily
/// on the first action run - see their own doc comments) - so with the
/// weights role coming from a resolved [`Assembly`], construction alone
/// SUCCEEDS even for a made-up path. A regression back to `from_env!`
/// would instead answer `Err("set BRAIN_…")` here, because the relevant
/// variable is (deliberately, via the `remove_var` calls below) unset -
/// that flip from `Err` to `Ok` is exactly what this asserts.
#[test]
fn deferred_load_providers_construct_from_the_assembly_with_no_env_set() {
    let _serial = brain_testutil::env_lock();
    for var in ["BRAIN_QWEN3ASR", "BRAIN_NEMOTRONASR", "BRAIN_SAM2_WEIGHTS"] {
        std::env::remove_var(var);
    }
    for (model, arch) in [(qwen3asr::caps::MODEL, "qwen3asr"), (nemotronasr::caps::MODEL, "nemotronasr"), (sam2::caps::MODEL, "sam2")] {
        let assembly = assembly_with_weights(arch, "/nonexistent/brain-catalog-test-weights");
        provider_from_assembly(model, &assembly).unwrap_or_else(|e| panic!("{arch}: construction from a resolved assembly must succeed (the real load is deferred): {e}"));
    }
}

/// `rrdbnet`'s provider loads EAGERLY at construction (unlike the three
/// above), so a made-up path is a real, immediate load error - the
/// regression this guards is the SAME one as above, just visible as a
/// different `Err` rather than an `Ok`/`Err` flip: `from_env!` would
/// answer with `BRAIN_ESRGAN_WEIGHTS`'s own "set BRAIN_…" message
/// regardless of the assembly, which this checks is no longer the
/// failure mode.
#[test]
fn rrdbnets_eager_provider_reads_the_assembly_not_env() {
    let _serial = brain_testutil::env_lock();
    std::env::remove_var("BRAIN_ESRGAN_WEIGHTS");
    let assembly = assembly_with_weights("rrdbnet", "/nonexistent/brain-catalog-test-weights");
    // `Arc<dyn Provider>` (the `Ok` side) isn't `Debug`, so `expect_err` -
    // which needs that to format its own panic message - doesn't apply.
    let err = match provider_from_assembly(rrdbnet::caps::MODEL, &assembly) {
        Err(e) => e,
        Ok(_) => panic!("a nonexistent path must fail to load, not silently succeed"),
    };
    assert!(!err.contains("unknown model"), "{err}");
    assert!(!err.contains("BRAIN_ESRGAN_WEIGHTS"), "rrdbnet provider still reads BRAIN_ESRGAN_WEIGHTS instead of the assembly: {err}");
}
