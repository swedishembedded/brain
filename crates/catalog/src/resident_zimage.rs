// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Z-Image behind the residency scheduler.

use std::sync::Arc;

use capability::{ActionResult, Invocation, Manifest, Outcome, Progress};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel, Tier};
use s3dit::pipeline::{HotPipeline, Image, Paths};
use serde_json::{json, Value};

use loader::served::RoleEnv;

/// z-image behind the scheduler. A resident instance is a built [`HotPipeline`] for a
/// `(width, height, precision, adapter)` key - the DiT (and its encoder) on the GPU;
/// dropping it frees the VRAM. `text2image` runs on the resident pipeline; the
/// image-editing actions build fresh per call (they take a variable-size input) via a
/// held provider, so the full manifest still works over the bus.
pub struct ZImageResident {
    id: String,
    paths: Paths,
    provider: Arc<s3dit::caps::ZImageProvider>,
}

impl ZImageResident {
    pub fn from_env() -> Result<ZImageResident, String> {
        Self::from_paths(s3dit::caps::MODEL, Paths::from_env()?)
    }

    /// Each of the four components independently: its own `BRAIN_S3DIT_*`
    /// variable if the operator set that one, else whatever the model-store
    /// resolver finds for that role through `s3dit::spec::S3ditSpec` - the
    /// SAME spec/scan/candidate rules a one-shot resolver-backed command
    /// would use, rather than [`Self::from_env`]'s four-variables-or-nothing
    /// mechanism. Mirrors `crate::resident_flux2::Flux2Resident::from_env`
    /// exactly - the same `served_assembly` seam, the other architecture
    /// already migrated onto it.
    ///
    /// `None` when nothing resolves, or the outcome is ambiguous (logged) -
    /// never a hard startup failure for the whole daemon. [`Self::from_env`]
    /// stays as it is: a lower-level escape hatch other callers (`brain perf
    /// run zimage`, the `#[ignore]`d real-checkpoint tests) still use when
    /// they deliberately want every path named outright.
    ///
    /// `models_dir` is the serving process's resolved models directory (see
    /// `loader::served::served_assembly`).
    pub fn from_store(models_dir: Option<&std::path::Path>) -> Option<ZImageResident> {
        let assembly = loader::served::served_assembly(
            models_dir,
            "s3dit",
            &s3dit::spec::S3ditSpec,
            &[
                RoleEnv { role: "dit", var: "BRAIN_S3DIT_DIT" },
                RoleEnv { role: "vae", var: "BRAIN_S3DIT_VAE" },
                RoleEnv { role: "text_encoder", var: "BRAIN_S3DIT_QWEN" },
                RoleEnv { role: "tokenizer", var: "BRAIN_S3DIT_TOKENIZER" },
            ],
        )?;
        // `Paths::from_assembly` owns the role-name -> field mapping already
        // (including `text_encoder` -> `qwen`); this must not carry a second
        // copy of it.
        let paths = match Paths::from_assembly(&assembly) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("brain: z-image not served over the scheduler ({e})");
                return None;
            }
        };
        match Self::from_paths(s3dit::caps::MODEL, paths) {
            Ok(r) => Some(r),
            Err(e) => {
                eprintln!("brain: z-image not served over the scheduler ({e})");
                None
            }
        }
    }

    /// Built from an already-resolved [`Paths`] rather than the environment,
    /// under `id` rather than the compiled-in `s3dit::caps::MODEL` -- what
    /// `serving::model_dir::resident_for_local` uses for a compound
    /// (multi-file) model found on disk or just auto-fetched, whose four
    /// component paths come from a `brain.manifest.json`'s roles and whose id
    /// is the fully-qualified ref it was fetched as (e.g.
    /// `Tongyi-MAI/Z-Image-Turbo`) -- registering under the compiled-in
    /// constant instead would silently strand the request that triggered the
    /// fetch (it named the fetched ref, not `brain/s3dit`).
    pub fn from_paths(id: impl Into<String>, paths: Paths) -> Result<ZImageResident, String> {
        let provider = Arc::new(s3dit::caps::ZImageProvider::from_paths(paths.clone()));
        Ok(ZImageResident { id: id.into(), paths, provider })
    }

    /// The caption capacity a request asks for. An absent, negative or
    /// out-of-`u32` value falls back to the default rather than wrapping,
    /// because `instance_key` cannot return a `Result` and `activate` reads
    /// this back out of the key it produced - the two must never disagree
    /// about which build a request names (`resident_qwen3vl::precision_of`
    /// resolves the same tension the same way). A capacity that is in range
    /// but unbuildable for this DiT is refused by name inside
    /// `HotPipeline::build_adapted`, where the config is known.
    fn cap_len_of(inv: &Invocation) -> u32 {
        inv.get_i64("cap_len").and_then(|v| u32::try_from(v).ok()).unwrap_or(s3dit::pipeline::DEFAULT_CAP_LEN)
    }
}

impl ResidentModel for ZImageResident {
    fn manifest(&self) -> Manifest {
        let mut m = s3dit::caps::manifest();
        m.model = self.id.clone();
        m
    }

    fn instance_key(&self, action: &str, inv: &Invocation) -> InstanceKey {
        if action == "text2image" {
            let w = inv.get_i64("width").unwrap_or(1024);
            let h = inv.get_i64("height").unwrap_or(1024);
            let prec = if inv.get_str("precision").as_deref() == Some("fp32") { "fp32" } else { "int8" };
            let adapter = inv.get_str("adapter").unwrap_or_default();
            // `cap_len` is a build-shape param like the other four: it fixes
            // the caption capacity every graph in the pipeline is recorded
            // over. `activate` gets nothing but this key, so leaving it out
            // meant activate had to hardcode the default - a caller who set
            // `cap_len=1024` got a pipeline built for 512 and was then refused
            // a 700-token prompt "because this pipeline was built for cap_len
            // 512", with no sign the param had been dropped. Same rule
            // `resident_qwen3vl` follows for `max_pixels`/`precision`.
            let cap_len = Self::cap_len_of(inv);
            InstanceKey::new(&self.id, format!("{w}x{h}:{prec}:{cap_len}:{adapter}"))
        } else {
            // Editing/training actions build fresh per call - one transient instance.
            InstanceKey::new(&self.id, format!("edit:{action}"))
        }
    }

    fn estimate(&self, key: &InstanceKey) -> MemCost {
        // int8 DiT (~13 GB); edit builds are transient and small-footprint
        // (they build + drop within the call). fp32 delegates to
        // `s3dit::pipeline::hifi_cost_bytes`, which picks between the
        // 2-GPU-shard estimate and the real windowed-engine estimate from
        // the SAME machine-shape decision (`gpu_core::devices::schedulable_gpu_count()`)
        // `DitEngine::build_from_source` itself makes - the number budgeted
        // here and the number the code actually allocates must be the same
        // expression, or this estimate silently outlives whichever engine
        // it was written for.
        if key.config.contains(":fp32:") {
            let (vram, ram) = s3dit::pipeline::hifi_cost_bytes(gpu_core::devices::schedulable_gpu_count());
            return MemCost::new(vram, ram);
        }
        let vram = if key.config.starts_with("edit:") { 2u64 << 30 } else { 14u64 << 30 };
        MemCost::new(vram, 0)
    }

    fn estimate_at(&self, key: &InstanceKey, tier: Tier) -> MemCost {
        // The shape is `ZImageConfig::turbo()` because that IS the only config
        // the pipeline ever builds (`s3dit::pipeline` hardcodes turbo() at
        // every build site); deriving it per-checkpoint belongs with the model
        // crate growing a second config, not here.
        let cache_ram = || s3dit::pipeline::int8_cache_bytes_estimate(&s3dit::ZImageConfig::turbo());
        match tier {
            // A cache-retaining build holds the multi-GB host `DitI8Cache`
            // ALONGSIDE the hot pipeline - charging Hot only the VRAM left
            // those bytes invisible to every budget exactly while they
            // coexist with the device copy (the residency contract this
            // adapter exists to keep honest). Only the shape that can
            // actually retain a cache pays it: fp32/adapter/edit builds
            // keep the plain estimate.
            Tier::Hot => {
                let mut cost = self.estimate(key);
                let (_, _, hifi, _, adapter) = parse_key(&key.config);
                let adapter = if adapter.is_empty() { None } else { Some(adapter.as_str()) };
                if !key.config.starts_with("edit:") && retains_int8_cache(hifi, adapter) {
                    cost.ram += cache_ram();
                }
                cost
            }
            // Real, not `0`: only the plain int8 build (see
            // `retains_int8_cache`) ever actually has a Warm state (every
            // other shape's `demote` refuses, so the manager never
            // consults this for those) -- but when it does, the retained
            // `DitI8Cache` genuinely holds several GB, and claiming
            // otherwise is precisely the kind of budgeting lie this whole
            // workstream exists to avoid.
            Tier::Warm | Tier::Cold => MemCost::new(0, cache_ram()),
        }
    }

    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        if key.config.starts_with("edit:") {
            // No persistent pipeline - the provider builds fresh per call.
            return Ok(Box::new(ZImageInstance { pipe: None, dit_cache: None, provider: self.provider.clone(), paths: self.paths.clone(), width: 0, height: 0, cap_len: 0 }));
        }
        let (w, h, hifi, cap_len, adapter) = parse_key(&key.config);
        let adapter = if adapter.is_empty() { None } else { Some(adapter.as_str()) };
        // Place the DiT on the assigned card (scoped registry selection); the
        // encoder card is z-image's own (BRAIN_S3DIT_ENCODER_GPU) and left as
        // configured.
        let (pipe, dit_cache) = if retains_int8_cache(hifi, adapter) {
            let (pipe, cache) = crate::resident_llm::on_device(device, || HotPipeline::build_adapted_with_cache(&self.paths, w, h, cap_len, |_| {}))??;
            (pipe, Some(cache))
        } else {
            let pipe = crate::resident_llm::on_device(device, || HotPipeline::build_adapted(&self.paths, w, h, cap_len, hifi, adapter, |_| {}))??;
            (pipe, None)
        };
        Ok(Box::new(ZImageInstance { pipe: Some(pipe), dit_cache, provider: self.provider.clone(), paths: self.paths.clone(), width: w, height: h, cap_len }))
    }
}

/// Whether an `activate` for `(hifi, adapter)` should retain a
/// [`s3dit::DitI8Cache`] alongside the built pipeline - real, permanent
/// extra host RAM (see [`ZImageDitI8::build_from_source_with_cache`]'s
/// doc), so opt-in only (`BRAIN_S3DIT_RETAIN_INT8_CACHE=1`) and only for
/// the one shape a cache can even be built for: plain int8, no adapter.
/// fp32 and LoRA-folded builds always return `false` regardless of the env
/// var - `demote` for those stays the manager's unmodified default
/// (`Err("unsupported")`, today's full drop-and-rebuild), not a silent lie.
fn retains_int8_cache(hifi: bool, adapter: Option<&str>) -> bool {
    !hifi && adapter.is_none() && std::env::var("BRAIN_S3DIT_RETAIN_INT8_CACHE").ok().as_deref() == Some("1")
}

/// A resident z-image instance: `pipe` when a text2image pipeline is built; the
/// `provider` handles the fresh-build editing/training actions. `dit_cache`
/// is `Some` only for an instance that opted into [`retains_int8_cache`] -
/// what makes `demote`/`promote` real instead of the default `Err`.
struct ZImageInstance {
    pipe: Option<HotPipeline>,
    dit_cache: Option<s3dit::DitI8Cache>,
    provider: Arc<s3dit::caps::ZImageProvider>,
    paths: Paths,
    width: u32,
    height: u32,
    cap_len: u32,
}

impl Instance for ZImageInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        if action == "text2image" {
            let pipe = self.pipe.as_ref().ok_or("z-image: text2image instance has no pipeline")?;
            let prompt = inv.get_str("prompt").unwrap_or_default();
            let seed = inv.get_i64("seed").map(|s| s.max(0) as u64).unwrap_or_else(data::rng::random_seed);
            let steps = inv.get_i64("steps").unwrap_or(8).max(1) as u32;
            let img = pipe.generate(&prompt, seed, steps, &inv.cancel, |s, t, m| progress(Progress::step(s, t, m.to_string())))?;
            return Ok(emit_image(img));
        }
        // Editing / training: delegate to the provider's action (fresh build).
        use capability::Provider;
        let act = self.provider.action(action).ok_or_else(|| format!("z-image: unknown action '{action}'"))?;
        let inv = act.spec().validate(inv.clone())?;
        act.run(&inv, progress)
    }

    fn metrics(&self) -> Vec<(String, Value)> {
        self.pipe.as_ref().map(|p| p.metrics()).unwrap_or_default()
    }

    /// Real only when `activate` retained a `dit_cache` (plain int8, no
    /// adapter, `BRAIN_S3DIT_RETAIN_INT8_CACHE=1`): drops the whole
    /// resident pipeline - encoder, DiT, VAE, every device buffer - while
    /// the cache (already held separately) survives, ready for `promote`.
    /// Refuses for everything else (fp32, an adapter build, or a plain
    /// int8 build that didn't opt in), so the manager falls back to its
    /// default full evict+rebuild exactly as it does for every model that
    /// never overrides this.
    fn demote(&mut self, tier: Tier) -> Result<(), String> {
        if tier == Tier::Hot {
            return Err("z-image: Hot is not a demotion target".to_string());
        }
        if self.dit_cache.is_none() {
            return Err("z-image: this instance retained no demote/promote cache".to_string());
        }
        self.pipe = None;
        Ok(())
    }

    /// The inverse: rebuild the pipeline from `dit_cache` on `device` - no
    /// DiT checkpoint read, no re-quantization (see
    /// `s3dit::ZImageDitI8::rebuild_from_cache`'s doc). Only reachable
    /// after a successful `demote`, so `dit_cache` is always `Some` here.
    fn promote(&mut self, device: Device) -> Result<(), String> {
        let cache = self.dit_cache.as_ref().ok_or("z-image: promote called on an instance with no retained cache")?;
        let pipe = crate::resident_llm::on_device(device, || HotPipeline::build_from_dit_cache(&self.paths, self.width, self.height, self.cap_len, cache, |_| {}))??;
        self.pipe = Some(pipe);
        Ok(())
    }
}

/// Parse a `"WxH:precision:cap_len:adapter"` instance key. The adapter is last
/// because it is a filesystem path and may itself contain a ':'.
fn parse_key(config: &str) -> (u32, u32, bool, u32, String) {
    let mut parts = config.splitn(4, ':');
    let wh = parts.next().unwrap_or("1024x1024");
    let prec = parts.next().unwrap_or("int8");
    let cap_len = parts.next().and_then(|s| s.parse().ok()).unwrap_or(s3dit::pipeline::DEFAULT_CAP_LEN);
    let adapter = parts.next().unwrap_or("").to_string();
    let (w, h) = wh.split_once('x').unwrap_or(("1024", "1024"));
    (w.parse().unwrap_or(1024), h.parse().unwrap_or(1024), prec == "fp32", cap_len, adapter)
}

/// Wrap a generated [`Image`] as an image-output [`Outcome`] (the shared
/// `capability::blob` wire format).
fn emit_image(img: Image) -> Outcome {
    Outcome::new()
        .set("width", json!(img.w))
        .set("height", json!(img.h))
        .blob("image", capability::blob::image_blob(&img.hwc, img.w as u32, img.h as u32, 3))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `cap_len` is a BUILD-SHAPE param: it fixes the caption capacity every
    /// graph in the pipeline is recorded over, exactly as `width`/`height`/
    /// `precision`/`adapter` do. It was missing from the instance key, so
    /// `activate` had nothing to read it from and hardcoded the default - a
    /// served caller who explicitly asked for `cap_len=1024` got a pipeline
    /// built for 512 and was then refused a 700-token prompt "because this
    /// pipeline was built for cap_len 512", with no acknowledgement the param
    /// had been ignored. Same rule `resident_qwen3vl` follows for `max_pixels`.
    #[test]
    fn cap_len_is_part_of_the_text2image_instance_identity() {
        let r = ZImageResident::from_paths(s3dit::caps::MODEL, unresolvable_paths()).expect("no weights are read to build a resident");
        let key = |inv: Invocation| r.instance_key("text2image", &inv);
        let k_default = key(Invocation::new());
        let k_big = key(Invocation::new().set("cap_len", json!(1024)));
        assert_ne!(k_default, k_big, "two capacities are two different builds");
        assert_eq!(parse_key(&k_default.config).3, s3dit::pipeline::DEFAULT_CAP_LEN);
        assert_eq!(parse_key(&k_big.config).3, 1024, "activate must build for what the caller asked, not a constant");
        // An out-of-range value falls back to the default rather than wrapping,
        // so `instance_key` and `activate` can never disagree about the build.
        assert_eq!(key(Invocation::new().set("cap_len", json!(1i64 << 40))), k_default);
        assert_eq!(key(Invocation::new().set("cap_len", json!(-5))), k_default);
        // The other build-shape params still key apart, and an adapter path
        // (which may itself contain a ':') still round-trips out of the key.
        assert_ne!(key(Invocation::new().set("width", json!(512))), k_default);
        assert_ne!(key(Invocation::new().set("precision", json!("fp32"))), k_default);
        let k_ad = key(Invocation::new().set("adapter", json!("a:b/c.brain")));
        assert_eq!(parse_key(&k_ad.config), (1024, 1024, false, s3dit::pipeline::DEFAULT_CAP_LEN, "a:b/c.brain".to_string()));
    }

    fn unresolvable_paths() -> Paths {
        let p = |role: &str| format!("no-such-z-image-{role}");
        Paths { dit: p("dit"), vae: p("vae"), qwen: p("qwen"), tokenizer: p("tokenizer") }
    }

    // -------- ZImageResident::from_store: a synthetic store, no env vars --------
    //
    // Mirrors `s3dit::spec`'s own `turbo_fixture` test helper (same tensor
    // names/shapes `S3ditSpec::classify`/`validate` actually read) and
    // `crates/flux2/tests/resolve_layout.rs`'s HF-checkpoint fixture (a real
    // `config.json` + one shard + its index - `hfdir_record` in
    // `brain_modelstore::inventory` refuses to collapse a directory with no
    // real shard file into an `HfDir` record at all), so the resolver's own
    // real `brain_modelstore::inventory::scan` recognizes every role exactly
    // as it would on a real fetched store.

    const FROM_STORE_TOY_VOCAB: usize = 100;

    fn scratch_store(tag: &str) -> std::path::PathBuf {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("brain-cli-resident-{tag}-{}-{n}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Only the tensors `s3dit::import::dit_config_from_shapes` actually
    /// reads, at real `ZImageConfig::turbo()` dimensions.
    fn write_from_store_dit(path: &std::path::Path) {
        let cfg = s3dit::model::ZImageConfig::turbo();
        let (dim, cap_feat_dim, head_dim) = (cfg.dim as usize, cfg.cap_feat_dim as usize, (cfg.dim / cfg.n_heads) as usize);
        let patch_dim = (cfg.in_channels * cfg.patch_size * cfg.patch_size * cfg.f_patch_size) as usize;
        let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = vec![
            ("cap_embedder.0.weight".to_string(), vec![1], vec![0.0f32]),
            ("cap_embedder.1.weight".to_string(), vec![dim as u64, cap_feat_dim as u64], vec![0.0f32; dim * cap_feat_dim]),
            ("layers.0.attention.q_norm.weight".to_string(), vec![head_dim as u64], vec![0.0f32; head_dim]),
            ("x_embedder.weight".to_string(), vec![dim as u64, patch_dim as u64], vec![0.0f32; dim * patch_dim]),
        ];
        for prefix in ["layers", "noise_refiner", "context_refiner"] {
            let n = if prefix == "layers" { cfg.n_layers } else { cfg.n_refiner_layers };
            for l in 0..n {
                tensors.push((format!("{prefix}.{l}.attention.qkv.weight"), vec![1], vec![0.0f32]));
            }
        }
        checkpoint::st::save_safetensors(path.to_str().unwrap(), &tensors, &json!({}), None).unwrap();
    }

    /// A canonical `<vendor>/<repo>` HF text-encoder directory: `config.json`
    /// declaring `Qwen3ForCausalLM` at `hidden`, plus a real (tiny) shard and
    /// its index - the shape `brain_modelstore::inventory::scan` collapses to
    /// one `HfDir` record.
    fn write_from_store_text_encoder(dir: &std::path::Path, hidden: u64) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&json!({"architectures": ["Qwen3ForCausalLM"], "hidden_size": hidden, "vocab_size": FROM_STORE_TOY_VOCAB})).unwrap()).unwrap();
        checkpoint::st::save_safetensors(dir.join("model-00001-of-00001.safetensors").to_str().unwrap(), &[("w".to_string(), vec![1], vec![0.0f32])], &json!({}), None).unwrap();
        std::fs::write(dir.join("model.safetensors.index.json"), serde_json::to_vec(&json!({"weight_map": {"w": "model-00001-of-00001.safetensors"}})).unwrap()).unwrap();
    }

    fn write_from_store_vae(path: &std::path::Path) {
        checkpoint::st::save_safetensors(
            path.to_str().unwrap(),
            &[
                ("decoder.conv_in.weight".to_string(), vec![512, 32, 3, 3], vec![0.0f32; 512 * 32 * 3 * 3]),
                ("encoder.conv_in.weight".to_string(), vec![128, 3, 3, 3], vec![0.0f32; 128 * 3 * 3 * 3]),
            ],
            &json!({}),
            None,
        )
        .unwrap();
    }

    fn write_from_store_tokenizer(path: &std::path::Path) {
        let vocab: serde_json::Map<String, serde_json::Value> = (0..FROM_STORE_TOY_VOCAB).map(|i| (format!("t{i}"), json!(i))).collect();
        std::fs::write(path, serde_json::to_vec(&json!({"version": "1.0", "model": {"vocab": vocab}, "added_tokens": []})).unwrap()).unwrap();
    }

    /// Milestone C's own acceptance case: with every `BRAIN_S3DIT_*` variable
    /// unset, `ZImageResident::from_store` must still resolve a complete
    /// assembly purely from local discovery against a synthetic model-store
    /// fixture - the whole point of routing the static resident registration
    /// through the resolver instead of [`ZImageResident::from_env`]'s
    /// four-variables-or-nothing mechanism.
    #[test]
    fn from_store_resolves_a_synthetic_store_with_no_brain_s3dit_env_vars_set() {
        let _serial = brain_testutil::env_lock();
        let root = scratch_store("s3dit-from-store");
        let vendor = root.join("Tongyi-MAI");
        std::fs::create_dir_all(&vendor).unwrap();
        write_from_store_dit(&vendor.join("dit.safetensors"));
        let cap_feat_dim = s3dit::model::ZImageConfig::turbo().cap_feat_dim as u64;
        write_from_store_text_encoder(&vendor.join("Qwen3-4B"), cap_feat_dim);
        write_from_store_vae(&vendor.join("vae.safetensors"));
        // A vendor-flat loose `tokenizer.json` (sitting directly under the
        // vendor directory, not a `<vendor>/<repo>` walk) is never scanned
        // for the tokenizer role at all - `brain_modelstore::inventory::
        // walk_vendor_dir` only recognizes `tokenizer.json` by filename one
        // level deeper, inside a repo-shaped directory (see
        // `crates/flux2/tests/resolve_layout.rs`'s own tokenizer fixture,
        // which notes the same rule). A real fetched tokenizer always sits
        // in its own repo directory, so this is what makes the fixture real.
        let tok_dir = vendor.join("Qwen3-4B-tokenizer");
        std::fs::create_dir_all(&tok_dir).unwrap();
        write_from_store_tokenizer(&tok_dir.join("tokenizer.json"));
        // A second, unrelated top-level vendor directory - without it every
        // record lives under `root/Tongyi-MAI/...` alone, so `resolve()`'s
        // own `common_root` (the deepest ancestor common to every record)
        // collapses all the way down to the vendor directory itself, and
        // `classify_tokenizer_role`'s vendor-co-location check then computes
        // a DIFFERENT "vendor" for the tokenizer file than for the
        // text_encoder directory (see `vendor_dir`'s doc) - exactly the
        // shape `s3dit::spec`'s own `turbo_fixture` test helper works around
        // the same way, with its own `other-vendor/unrelated.bin`. A real
        // model store always holds more than one vendor, so this is not an
        // artifact of the test - it is what makes the fixture a real
        // multi-vendor store rather than a store of exactly one model.
        std::fs::create_dir_all(root.join("other-vendor")).unwrap();
        checkpoint::st::save_safetensors(root.join("other-vendor").join("unrelated.safetensors").to_str().unwrap(), &[("w".to_string(), vec![1], vec![0.0f32])], &json!({}), None).unwrap();

        for v in ["BRAIN_S3DIT_DIT", "BRAIN_S3DIT_VAE", "BRAIN_S3DIT_QWEN", "BRAIN_S3DIT_TOKENIZER"] {
            std::env::remove_var(v);
        }
        let resident = ZImageResident::from_store(Some(&root));
        std::fs::remove_dir_all(&root).ok();

        assert!(resident.is_some(), "must resolve purely from local discovery with no BRAIN_S3DIT_* set");
    }

    /// The served half of the same contract, on real weights: a caller who
    /// asks for a bigger capacity must actually GET one. With `cap_len`
    /// missing from the instance key, `activate` built at the compiled-in
    /// default and this prompt - deliberately longer than that default -
    /// came back refused "because this pipeline was built for cap_len 512",
    /// contradicting the request the resident was keyed on.
    #[test]
    #[ignore = "slow: real checkpoint + GPU; set BRAIN_S3DIT_* and run with --ignored"]
    fn a_served_request_that_raises_cap_len_gets_a_pipeline_built_for_it() {
        let paths = match Paths::from_env() {
            Ok(p) => p,
            Err(e) => return brain_testutil::skip(&format!("Z-Image checkpoint paths not set: {e}")),
        };
        let model = ZImageResident::from_paths(s3dit::caps::MODEL, paths).expect("BRAIN_S3DIT_* all resolved");
        // ~700 caption tokens: over the 512 default, under the 1024 asked for.
        let prompt = "a red fox in deep snow at dawn, long telephoto photograph, soft rim light, ".repeat(40);
        let inv = Invocation::new()
            .set("prompt", json!(prompt))
            .set("cap_len", json!(1024))
            .set("width", json!(256))
            .set("height", json!(256))
            .set("steps", json!(2))
            .set("seed", json!(42));

        let key = model.instance_key("text2image", &inv);
        assert_eq!(parse_key(&key.config).3, 1024, "the request's capacity must reach activate through the key");
        let mut inst = model.activate(&key, Device::Gpu(0)).expect("activate at the requested capacity");
        let out = inst.run("text2image", &inv, &mut |_| {}).expect("a prompt past the DEFAULT capacity must generate at the capacity that was asked for");
        assert!(out.blobs.contains_key("image"));
    }

    /// Real end-to-end proof that demote/promote works against the actual
    /// ~31 GB Z-Image checkpoint, not a synthetic model: activate (builds
    /// fresh, retains a DitI8Cache), demote (drops the whole pipeline --
    /// GPU AND host -- keeping only the cache), promote (rebuilds from the
    /// cache: no checkpoint read, no re-quantization), then run a real
    /// generation and confirm it produces a real image. Times both
    /// activate() and promote() so "promote is faster" is a measured
    /// number, not an assertion resting on the design alone.
    #[test]
    #[ignore = "slow: real checkpoint + GPU; set BRAIN_S3DIT_* and run with --ignored"]
    fn zimage_demote_then_promote_produces_a_real_image_and_promote_is_faster() {
        std::env::set_var("BRAIN_S3DIT_RETAIN_INT8_CACHE", "1");
        // Two different failures used to share one skip: "the checkpoint paths
        // are not on this box" (a fixture that is legitimately absent) and
        // "the paths resolved but the provider would not load" (a real
        // failure). Resolve them separately so only the first is a skip.
        let paths = match s3dit::pipeline::Paths::from_env() {
            Ok(p) => p,
            Err(e) => return brain_testutil::skip(&format!("Z-Image checkpoint paths not set: {e}")),
        };
        let model = ZImageResident::from_paths(s3dit::caps::MODEL, paths)
            .expect("BRAIN_S3DIT_* all resolved, so the Z-Image provider must load");
        let key = InstanceKey::new(s3dit::caps::MODEL, "256x256:int8:");

        let t0 = std::time::Instant::now();
        let mut inst = model.activate(&key, Device::Gpu(0)).expect("activate");
        let activate_secs = t0.elapsed().as_secs_f64();

        inst.demote(Tier::Warm).expect("a cache-retaining instance must demote successfully");

        let t1 = std::time::Instant::now();
        inst.promote(Device::Gpu(0)).expect("promote must rebuild from the cache");
        let promote_secs = t1.elapsed().as_secs_f64();

        let inv = Invocation::new().set("prompt", json!("a red fox in snow, photograph")).set("seed", json!(42)).set("steps", json!(4));
        let result = inst.run("text2image", &inv, &mut |_| {}).expect("run after promote must succeed");
        assert!(result.blobs.contains_key("image"), "a promoted instance must still be able to generate a real image");

        eprintln!("activate: {activate_secs:.1}s, promote: {promote_secs:.1}s");
        assert!(promote_secs < activate_secs, "promote (cache-based) must be faster than the fresh activate() it followed -- it skips the checkpoint read AND the quantization activate() just did");
    }
}
