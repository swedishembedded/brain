// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Wan2.1 text-to-video behind the residency scheduler
//! (`resident::build_executor`).
//!
//! A resident instance is a built transformer for one
//! `(variant, frames, WxH)` fingerprint - the only things that fix the DiT's
//! graphs and buffer sizes - held across requests by
//! `wan::pipeline::HotDit`; dropping the instance frees it. All action
//! execution goes through the shared helpers in `wan::caps`, so this adapter
//! and `WanProvider` cannot decode a request differently.
//!
//! The umT5 encoder and the VAE decoder are deliberately NOT resident: umT5
//! is 22.72 GB in fp32 and provably does not fit next to the DiT on a 24 GB
//! card (it defaults to the CPU for exactly that reason), and the VAE is 508
//! MB against the DiT's 5.7 GB. What residency buys here is the transformer
//! load and upload, which is what a second request actually re-pays.

use capability::{ActionResult, Invocation, Manifest, Progress};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

/// Wan resident model family, gated on the four weight env vars
/// (`BRAIN_WAN_{DIT,VAE,T5,TOKENIZER}`).
pub struct WanResident {
    /// The id this registers under. `wan::caps::MODEL` for the env-configured
    /// build, but a FETCHED checkpoint registers under its own reference
    /// (`Wan-AI/Wan2.1-T2V-1.3B`) - a client that asked the supplier for that
    /// name has to find it afterwards, and the compiled-in constant would
    /// leave the request that triggered the fetch looking at nothing. Same
    /// reason `ZImageResident::from_paths` takes an id.
    id: String,
    paths: wan::Paths,
}

impl WanResident {
    /// `None` (not registered) unless all four `BRAIN_WAN_*` vars are set.
    pub fn from_env() -> Option<WanResident> {
        wan::Paths::from_env().ok().map(|paths| WanResident { id: wan::caps::MODEL.to_string(), paths })
    }

    /// The same adapter over explicitly-named role paths - what a fetched
    /// compound checkpoint's `brain.manifest.json` supplies
    /// (`crate::model_dir::wan_paths_from_roles`), with no `BRAIN_WAN_*`
    /// variable involved anywhere. `id` is the checkpoint's own model-card id.
    pub fn from_paths(id: String, paths: wan::Paths) -> WanResident {
        WanResident { id, paths }
    }
}

/// `(variant, frames, width, height, adapter, dtype)` from an instance key
/// (`"{variant}:{frames}:{w}x{h}[:{adapter}][@{dtype}]"`). `None` if the key
/// does not parse.
///
/// The dtype rides a trailing `@{dtype}` SUFFIX on the whole string, not a
/// fifth `:`-separated field, and is stripped first: an adapter PATH may
/// itself contain `:` (the existing `splitn(4, ':')` swallows everything
/// remaining into the adapter for exactly that reason), so a dtype field
/// packed in among the colons would be ambiguous with the tail of such a
/// path. `@` cannot appear in a `WanDtype::key()` spelling, and a key with no
/// `@{dtype}` suffix at all defaults to `F32` - the original key format
/// (`"{variant}:{frames}:{w}x{h}[:{adapter}]"`) still parses unchanged, so
/// every key produced before this dtype support existed reads the same way.
fn parse_key(config: &str) -> Option<(String, usize, usize, usize, Option<String>, wan::WanDtype)> {
    let (rest, dtype) = match config.rsplit_once('@') {
        Some((r, d)) if wan::WanDtype::from_name(d).is_ok() => (r, wan::WanDtype::from_name(d).unwrap()),
        _ => (config, wan::WanDtype::F32),
    };
    let mut it = rest.splitn(4, ':');
    let variant = it.next()?.to_string();
    let frames: usize = it.next()?.parse().ok()?;
    let (w, h) = it.next()?.split_once('x')?;
    let adapter = it.next().filter(|s| !s.is_empty()).map(str::to_string);
    Some((variant, frames, w.parse().ok()?, h.parse().ok()?, adapter, dtype))
}

/// Real per-dtype byte count for the DiT's block-stack weights (the ten
/// quantizable linears per block: self/cross attention's q/k/v/o plus the two
/// FFN linears - exactly `model::int8::quantize_weight`/`model::int4::
/// quantize_weight_q4`'s domain) plus the one always-fp32 host tensor big
/// enough to matter (`text_embedding.0.weight`, `dim x text_dim`, never
/// quantized - `WanDitDev` keeps it and every other embedding/norm/bias
/// tensor on the host as plain fp32). Norms and biases are dropped as a
/// rounding error, same as the pre-existing `params` closed form did.
fn dit_weight_bytes(cfg: &wan::WanConfig, dtype: wan::WanDtype) -> u64 {
    let (dim, ffn, layers, text_dim) = (cfg.dim as u64, cfg.ffn_dim as u64, cfg.num_layers as u64, cfg.text_dim as u64);
    // Element count of the ten quantizable linears, one block's worth: eight
    // dim*dim projections (self+cross attention q/k/v/o) plus ffn.0
    // (ffn*dim) and ffn.2 (dim*ffn).
    let quant_elems_per_layer = 8 * dim * dim + 2 * dim * ffn;
    // Sum of OUTPUT rows across those same ten linears - the per-row scale
    // count `quantize_weight`/`quantize_weight_q4` produce one f32 per.
    let rows_per_layer = 8 * dim + ffn + dim;
    let always_fp32_bytes = dim * text_dim * 4;
    match dtype {
        // Unchanged from before this dtype support existed.
        wan::WanDtype::F32 | wan::WanDtype::F16 => (quant_elems_per_layer * layers) * 4 + always_fp32_bytes,
        // 1 byte/element packed (4 int8 lanes per u32 = 4 bytes = 4 elements)
        // plus one f32 scale per output row.
        wan::WanDtype::Int8 => quant_elems_per_layer * layers + rows_per_layer * layers * 4 + always_fp32_bytes,
        // 0.5 byte/element packed (8 int4 lanes per u32) plus the same
        // per-row f32 scale int8 uses.
        wan::WanDtype::Int4 => (quant_elems_per_layer * layers) / 2 + rows_per_layer * layers * 4 + always_fp32_bytes,
    }
}

impl ResidentModel for WanResident {
    fn manifest(&self) -> Manifest {
        // The shared schema, under THIS instance's own id -- see `id`'s doc.
        Manifest { model: self.id.clone(), ..wan::caps::manifest() }
    }

    fn instance_key(&self, action: &str, inv: &Invocation) -> InstanceKey {
        let variant = inv.get_str("variant").unwrap_or_else(|| "t2v-1.3B".into());
        if action == "lora_train" {
            // The host f32 trainer builds and drops its own encoders + DiT
            // per run - no resident DiT graph to key on, unlike `t2v`.
            return InstanceKey::new(&self.id, format!("train:{variant}"));
        }
        // Exactly the fields `wan::pipeline::DitKey` is built from: the
        // variant, the latent extent and the folded adapter. Steps, seed,
        // guidance, solver and the prompts are per-call and must NOT split
        // the instance - a different seed re-uploading 5.7 GB would defeat
        // residency entirely, but a different (or absent) adapter changes the
        // weights actually baked into the uploaded graph and MUST split it.
        let d = wan::GenOpts::from_config(&wan::WanConfig::t2v_1_3b());
        let frames = inv.get_i64("frames").unwrap_or(d.frames as i64);
        let w = inv.get_i64("width").unwrap_or(d.width as i64);
        let h = inv.get_i64("height").unwrap_or(d.height as i64);
        let adapter = inv.get_str("adapter").filter(|s| !s.is_empty());
        // Same "does this instance's uploaded graph change" question as the
        // adapter: a different weight dtype is a different resident build
        // (see `wan::pipeline::DitKey`'s doc), so it must split the instance
        // too. Falling back to `d.dit_dtype` (F32) on an absent/unparseable
        // value means an omitted `dit_dtype` param keys exactly like it did
        // before this field existed - no `@{dtype}` suffix at all.
        let dtype = match inv.get_str("dit_dtype").filter(|s| !s.is_empty()) {
            Some(s) => wan::WanDtype::from_name(&s).unwrap_or(d.dit_dtype),
            None => d.dit_dtype,
        };
        let mut config = match adapter {
            Some(a) => format!("{variant}:{frames}:{w}x{h}:{a}"),
            None => format!("{variant}:{frames}:{w}x{h}"),
        };
        if dtype != wan::WanDtype::F32 {
            config.push('@');
            config.push_str(dtype.key());
        }
        InstanceKey::new(&self.id, config)
    }

    fn estimate(&self, key: &InstanceKey) -> MemCost {
        // Derived, not measured-and-hardcoded: the two terms that actually
        // scale are the fp32 weights (a function of the variant) and the
        // resident token slabs (a function of the latent extent), and both
        // are computable from the key.
        //
        // The measured reference point this is calibrated against is the
        // 480p/81-frame 1.3B run: 5.2 GB of weights, 32,760 tokens.
        if let Some(variant) = key.config.strip_prefix("train:") {
            // The host f32 LoRA trainer (`model::hostmath`) is RAM, not VRAM.
            let cfg = wan::caps::config_from_name(variant).unwrap_or_else(|_| wan::WanConfig::t2v_1_3b());
            let params = cfg.num_layers as u64 * (8 * cfg.dim as u64 * cfg.dim as u64 + 2 * cfg.dim as u64 * cfg.ffn_dim as u64);
            return MemCost::new(0, params * 4 + (4u64 << 30));
        }
        let Some((variant, frames, w, h, _adapter, dtype)) = parse_key(&key.config) else {
            // An unparseable key must not read as "free"; charge the 1.3B
            // default rather than 0.
            return MemCost::new(8u64 << 30, 4u64 << 30);
        };
        let cfg = wan::caps::config_from_name(&variant).unwrap_or_else(|_| wan::WanConfig::t2v_1_3b());
        let tokens = cfg.token_count(frames, w, h).unwrap_or(0) as u64;
        // The real per-dtype byte count for the block-stack weights: `params
        // * 4` (unconditionally fp32) was D2 - an int8/int4 instance's actual
        // device footprint is a fraction of that, and this is the whole
        // reason int8/int4 storage is worth having (14B is ~53 GiB at fp32,
        // does not fit a 24 GiB card; ~14.4 GiB at int8, does).
        let dim = cfg.dim as u64;
        let weights = dit_weight_bytes(&cfg, dtype);
        // Two alternating `[tokens, dim]` residual slabs plus the block
        // scratch the engine keeps live; the VAE decoder's own activations
        // are the other large allocation and are charged here too, since
        // `generate_hot` holds the DiT across the decode.
        let activations = tokens * dim * 4 * 8;
        let vae = (frames as u64 * w as u64 * h as u64 * 3 * 4) * 4;
        // Host side: the imported checkpoint is materialized before upload,
        // plus staging. The umT5 encoder's own ~22.72 GB fp32 host peak is
        // deliberately NOT charged here: it is per-request and transient (the
        // encoder is built, run and dropped before the DiT is touched), and
        // `MemCost` has no way to say "transient peak" - reserving it for the
        // instance's whole life would make this model unplaceable on any box
        // that can actually run it, which is a worse failure than not knowing
        // about a peak the pipeline itself is structured to survive.
        MemCost::new(weights + activations + vae, weights + (2u64 << 30))
    }

    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        if let Some(variant) = key.config.strip_prefix("train:") {
            wan::caps::config_from_name(variant)?;
            return Ok(Box::new(WanInstance { paths: self.paths.clone(), device, hot: None }));
        }
        let (variant, _, _, _, _, _) = parse_key(&key.config).ok_or_else(|| format!("wan: bad instance key {:?}", key.config))?;
        // Validate the variant at activation, not at the first run: a key
        // naming a variant this build cannot construct should fail placement
        // rather than a request.
        wan::caps::config_from_name(&variant)?;
        // The DiT itself is built lazily, on the first run, because building
        // it needs the request's own latent extent AND its text conditioning
        // ordering (the text encoder must run and be dropped first - see
        // `wan::pipeline`'s memory note). What `activate` fixes is the device.
        Ok(Box::new(WanInstance { paths: self.paths.clone(), device, hot: None }))
    }
}

/// A resident Wan instance: the four weight paths, the assigned device, and
/// the transformer once a request has built it.
struct WanInstance {
    paths: wan::Paths,
    device: Device,
    hot: Option<wan::HotDit>,
}

impl WanInstance {
    /// The `device` string `wan::GenOpts` takes for the scheduler's placement.
    /// A CPU placement is a real (very slow) answer, not an error; an NPU has
    /// no Wan path, so it falls back to the ambient default rather than
    /// claiming one.
    fn device_name(&self) -> Option<String> {
        match self.device {
            Device::Cpu => Some("cpu".to_string()),
            Device::Gpu(_) => Some("gpu".to_string()),
            Device::Npu(_) => None,
        }
    }
}

impl Instance for WanInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        match action {
            "t2v" => {
                let mut p = wan::caps::gen_params_from(inv)?;
                p.opts.device = self.device_name();
                crate::resident_llm::on_device(self.device, || wan::caps::generate_on(&self.paths, &mut self.hot, inv, &p, progress))?
            }
            "lora_train" => wan::caps::train_action(&self.paths, inv, progress),
            other => Err(format!("wan: unknown action '{other}'")),
        }
    }

    /// Sequential, deliberately - and this override exists to say so rather
    /// than to inherit the default silently.
    ///
    /// What a batch CAN share here is the expensive thing: every job at this
    /// key runs against the SAME resident `HotDit`, so N requests pay one
    /// load and one 5.7 GB upload between them. What it cannot share is the
    /// forward. `wan::dev::WanDitDev` records one graph for one latent volume
    /// and holds ONE context buffer, which the CFG loop already has to
    /// re-upload between its own two forwards
    /// (`wan::pipeline::denoise`'s bracketing note) - so two jobs with
    /// different prompts cannot be in flight against it at once, and a real
    /// batched forward means a batch axis through the engine, the RoPE tables
    /// and the flash-attention slabs. At 32,760 tokens and ~46 s per forward
    /// that is a measurement-led change, not a wiring one.
    ///
    /// Per-request cancellation still works: each job's own `inv.cancel` is
    /// polled inside its own denoise loop, so cancelling job 2 does not
    /// disturb job 1.
    fn run_batch(&mut self, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        invs.iter().enumerate().map(|(i, inv)| self.run(action, inv, &mut |p| progress(i, p))).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn resident() -> WanResident {
        WanResident::from_paths(wan::caps::MODEL.to_string(), wan::Paths { dit: "/dit".into(), vae: "/vae".into(), t5: "/t5".into(), tokenizer: "/tok".into() })
    }

    /// The key must fix the built graphs and NOTHING else: two requests that
    /// differ only in seed/steps/guidance/prompt share one instance, which is
    /// the whole point of holding 5.7 GB resident.
    #[test]
    fn the_instance_key_is_exactly_the_variant_and_the_latent_extent() {
        let r = resident();
        let base = Invocation::new().set("prompt", json!("a"));
        let a = r.instance_key("t2v", &base);
        assert_eq!(a.config, "t2v-1.3B:81:832x480", "the defaults must key on the manifest's own defaults");
        let b = r.instance_key("t2v", &Invocation::new().set("prompt", json!("something else")).set("seed", json!(7)).set("steps", json!(3)).set("guidance", json!(1.0)).set("solver", json!("dpm++")));
        assert_eq!(a.config, b.config, "per-call params must not split the instance");
        // Anything that resizes the latent volume, or changes the weights, must.
        for (k, v) in [("frames", json!(9)), ("width", json!(256)), ("height", json!(256)), ("variant", json!("t2v-14B"))] {
            assert_ne!(a.config, r.instance_key("t2v", &base.clone().set(k, v)).config, "{k} must split the instance");
        }
        // An adapter changes the weights baked into the uploaded graph, so it
        // must split the instance too - and a different adapter path must
        // split from a first one, not just from "no adapter".
        let with_adapter = r.instance_key("t2v", &base.clone().set("adapter", json!("/adapters/a.brain")));
        assert_ne!(a.config, with_adapter.config, "an adapter must split the instance");
        let other_adapter = r.instance_key("t2v", &base.clone().set("adapter", json!("/adapters/b.brain")));
        assert_ne!(with_adapter.config, other_adapter.config, "different adapters must not collide");
        // `lora_train` keys on the variant alone - no resident DiT graph.
        let train = r.instance_key("lora_train", &Invocation::new().set("variant", json!("t2v-1.3B")));
        assert_eq!(train.config, "train:t2v-1.3B");
    }

    #[test]
    fn key_parsing_round_trips_and_a_bad_key_still_costs_something() {
        assert_eq!(parse_key("t2v-1.3B:81:832x480"), Some(("t2v-1.3B".to_string(), 81, 832, 480, None, wan::WanDtype::F32)));
        assert_eq!(
            parse_key("t2v-1.3B:81:832x480:/adapters/a.brain"),
            Some(("t2v-1.3B".to_string(), 81, 832, 480, Some("/adapters/a.brain".to_string()), wan::WanDtype::F32))
        );
        assert_eq!(parse_key("garbage"), None);
        // An unparseable key must never estimate as free -- a 0-cost model is
        // placeable anywhere, including where it does not fit.
        let cost = resident().estimate(&InstanceKey::new(wan::caps::MODEL, "garbage".to_string()));
        assert!(cost.vram > 0 && cost.ram > 0, "{cost:?}");
    }

    /// The `@{dtype}` suffix round-trips through both key shapes (with and
    /// without an adapter) and DEFAULTS to F32 when absent - a key produced
    /// before this dtype support existed must still parse exactly as it did.
    #[test]
    fn dtype_suffix_round_trips_and_defaults_to_f32() {
        assert_eq!(
            parse_key("t2v-14B:81:832x480@int8"),
            Some(("t2v-14B".to_string(), 81, 832, 480, None, wan::WanDtype::Int8))
        );
        assert_eq!(
            parse_key("t2v-14B:81:832x480:/adapters/a.brain@int4"),
            Some(("t2v-14B".to_string(), 81, 832, 480, Some("/adapters/a.brain".to_string()), wan::WanDtype::Int4))
        );
        // No suffix at all -> F32, same as every key from before this existed.
        assert_eq!(parse_key("t2v-1.3B:81:832x480").unwrap().5, wan::WanDtype::F32);
    }

    /// A dtype change must split the instance - a resident int8 build's
    /// weights and recorded kernel graph are not the same object as an f32
    /// one at the same size.
    #[test]
    fn a_dit_dtype_change_splits_the_instance() {
        let r = resident();
        let base = Invocation::new().set("prompt", json!("a"));
        let f32_key = r.instance_key("t2v", &base);
        let int8_key = r.instance_key("t2v", &base.clone().set("dit_dtype", json!("int8")));
        assert_ne!(f32_key.config, int8_key.config);
        assert!(int8_key.config.ends_with("@int8"), "{}", int8_key.config);
        // Explicitly asking for f32 must key IDENTICALLY to omitting it.
        let explicit_f32 = r.instance_key("t2v", &base.clone().set("dit_dtype", json!("f32")));
        assert_eq!(f32_key.config, explicit_f32.config);
    }

    /// A `lora_train` key must cost RAM (the host f32 trainer), never charge
    /// VRAM as if it were a resident DiT graph.
    #[test]
    fn a_training_key_costs_ram_not_vram() {
        let cost = resident().estimate(&InstanceKey::new(wan::caps::MODEL, "train:t2v-1.3B".to_string()));
        assert_eq!(cost.vram, 0);
        assert!(cost.ram > 0);
    }

    /// `activate` on a training key must not need a resident DiT and must
    /// still validate the variant.
    #[test]
    fn activate_accepts_a_training_key() {
        assert!(resident().activate(&InstanceKey::new(wan::caps::MODEL, "train:t2v-1.3B".to_string()), Device::Cpu).is_ok());
        let e = resident().activate(&InstanceKey::new(wan::caps::MODEL, "train:i2v-14B".to_string()), Device::Cpu).err().expect("unknown variant must not activate");
        assert!(e.contains("unknown wan variant"), "{e}");
    }

    /// The estimate has to land on the real parameter counts, or the budget
    /// means nothing. The closed form is checked against the numbers the
    /// variants are NAMED for: 1.3 G and 14 G parameters.
    #[test]
    fn the_estimate_tracks_the_real_parameter_counts() {
        let r = resident();
        // Isolate the weight term by asking for the smallest representable
        // clip, where activations and the VAE decode are negligible.
        let params = |variant: &str| r.estimate(&InstanceKey::new(wan::caps::MODEL, format!("{variant}:1:16x16"))).vram / 4;
        let g = 1_000_000_000u64;
        let small = params("t2v-1.3B");
        assert!((1_250_000_000..1_500_000_000).contains(&small), "t2v-1.3B must land near 1.3 G parameters, got {small}");
        let big = params("t2v-14B");
        assert!((13 * g..15 * g).contains(&big), "t2v-14B must land near 14 G parameters, got {big}");

        // At a real clip size the total is still single-digit GB for 1.3B.
        let gb = 1u64 << 30;
        let full = r.estimate(&InstanceKey::new(wan::caps::MODEL, "t2v-1.3B:81:832x480".to_string()));
        assert!((5 * gb..12 * gb).contains(&full.vram), "1.3B at 480p should be single-digit GB, got {} GB", full.vram / gb);
        // A smaller clip costs less at the same weights.
        let tiny = r.estimate(&InstanceKey::new(wan::caps::MODEL, "t2v-1.3B:9:256x256".to_string()));
        assert!(tiny.vram < full.vram);
    }

    /// D2: an int8/int4 instance must estimate STRICTLY less VRAM than the
    /// same shape at f32 - the entire justification for the quantized
    /// storage tier is that it fits where fp32 does not (14B: ~53 GiB fp32
    /// vs ~14.4 GiB int8) - and neither may ever report 0 (the deepseek2ocr
    /// `vram==0` mistake this repo documents elsewhere: a 0-cost model is
    /// placeable anywhere, including where it does not fit).
    #[test]
    fn quantized_dtypes_estimate_strictly_less_vram_than_f32_and_never_zero() {
        let r = resident();
        for variant in ["t2v-1.3B", "t2v-14B"] {
            let f32_cost = r.estimate(&InstanceKey::new(wan::caps::MODEL, format!("{variant}:81:832x480")));
            let f16_cost = r.estimate(&InstanceKey::new(wan::caps::MODEL, format!("{variant}:81:832x480@f16")));
            let int8_cost = r.estimate(&InstanceKey::new(wan::caps::MODEL, format!("{variant}:81:832x480@int8")));
            let int4_cost = r.estimate(&InstanceKey::new(wan::caps::MODEL, format!("{variant}:81:832x480@int4")));
            for c in [&f32_cost, &f16_cost, &int8_cost, &int4_cost] {
                assert!(c.vram > 0 && c.ram > 0, "{variant}: {c:?}");
            }
            assert_eq!(f32_cost.vram, f16_cost.vram, "{variant}: f16 storage is the same byte count as f32 here");
            assert!(int8_cost.vram < f32_cost.vram, "{variant}: int8 {} must be < f32 {}", int8_cost.vram, f32_cost.vram);
            assert!(int4_cost.vram < int8_cost.vram, "{variant}: int4 {} must be < int8 {}", int4_cost.vram, int8_cost.vram);
        }

        // At 14B specifically: the WEIGHTS alone (isolated the same way
        // `the_estimate_tracks_the_real_parameter_counts` does, via the
        // smallest representable clip so activations/VAE are negligible) do
        // not fit a 24 GiB card at fp32, but do at int8 - the actual
        // justification for the tier (see the module doc on `WanDtype`).
        let gb = 1u64 << 30;
        let fp32_14b = r.estimate(&InstanceKey::new(wan::caps::MODEL, "t2v-14B:1:16x16".to_string()));
        let int8_14b = r.estimate(&InstanceKey::new(wan::caps::MODEL, "t2v-14B:1:16x16@int8".to_string()));
        assert!(fp32_14b.vram > 24 * gb, "14B fp32 weights should exceed a 24 GiB card, got {} GB", fp32_14b.vram / gb);
        assert!(int8_14b.vram < 24 * gb, "14B int8 weights should fit a 24 GiB card, got {} GB", int8_14b.vram / gb);
    }

    /// A key naming a variant this build cannot construct must fail placement,
    /// not a request an hour later.
    #[test]
    fn activate_rejects_an_unknown_variant() {
        // `.err()`, not `.unwrap_err()`: the Ok arm is a `Box<dyn Instance>`,
        // which has no `Debug`.
        let e = resident().activate(&InstanceKey::new(wan::caps::MODEL, "i2v-14B:81:832x480".to_string()), Device::Cpu).err().expect("an unknown variant must not activate");
        assert!(e.contains("unknown wan variant"), "{e}");
    }

    /// The adapter and the provider must advertise the SAME manifest -- a
    /// client that discovers over D-Bus and one that runs `brain do` are
    /// looking at one model.
    #[test]
    fn the_adapter_advertises_the_shared_manifest() {
        let m = resident().manifest();
        assert_eq!(m.model, wan::caps::MODEL);
        assert_eq!(m.actions.len(), wan::caps::manifest().actions.len());
        // A FETCHED checkpoint advertises the same actions under its own
        // reference: the request that triggered the fetch asked for that name,
        // and the compiled-in constant would leave it finding nothing.
        let fetched = WanResident::from_paths("Wan-AI/Wan2.1-T2V-1.3B".to_string(), wan::Paths { dit: "d".into(), vae: "v".into(), t5: "t".into(), tokenizer: "k".into() });
        assert_eq!(fetched.manifest().model, "Wan-AI/Wan2.1-T2V-1.3B");
        assert_eq!(fetched.manifest().actions.len(), m.actions.len());
        assert_eq!(fetched.instance_key("t2v", &Invocation::new()).model, "Wan-AI/Wan2.1-T2V-1.3B");
    }
}
