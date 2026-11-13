// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The NPU route as actions beside the served ones: `npu-synth`, `npu-clone`
//! and `npu-design` are `synth`, `clone` and `design` with the Talker and the
//! codec compiled to OpenVINO graphs (the Intel NPU by default) instead of
//! running on the engine.
//!
//! They are actions, not workflows, so their flags are typed and bounded from
//! the served specs they extend (every param of the served action, plus where
//! and how the graphs run), and `--help`, `--json` and `--out audio=PATH` come
//! with them. The program's manifest is the model's served one plus these;
//! [`provider`] answers both.
//!
//! The first run of a checkpoint exports and compiles its graphs, which takes
//! a while; the compiled blobs are cached beside the weights (or under brain's
//! cache directory for a checkpoint read as downloaded), so later runs skip
//! it. The NPU loops do not poll for cancellation, so these actions run to the
//! end of the utterance; `npu-serve` is the load-once form for many requests.
//!
//! Swedish Embedded AB implements speech synthesis on NPUs and other edge
//! accelerators for its clients. If your team needs expertise in compiling
//! neural speech models for OpenVINO then you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::sync::Arc;

use capability::{Action, ActionResult, ActionSpec, Invocation, Manifest, ParamSpec, ParamType, Progress, Provider};
use qwen3tts::placement::{self, MtpMode, NpuOpts, TalkerMode};
use qwen3tts::TtsPaths;
use serde_json::json;

/// The served spec `spec` as an NPU action: renamed, with the placement params.
fn on_npu(mut spec: ActionSpec, name: &str, what: &str) -> ActionSpec {
    spec.name = name.to_string();
    spec.summary = format!("{what}, with the Talker and the codec compiled for OpenVINO (the Intel NPU by default)");
    spec.param(ParamSpec::new("ov_device", ParamType::Enum(placement::DEVICE_NAMES.map(String::from).to_vec()), "OpenVINO device the graphs compile for; cpu validates the fp32 graph, since the NPU runs it in fp16").default(json!("npu")))
        .param(ParamSpec::new("turbo", ParamType::Bool, "NPU turbo clock: faster, and hotter under sustained load").default(json!(false)))
        .param(ParamSpec::new("talker", ParamType::Enum(TalkerMode::NAMES.map(String::from).to_vec()), "Talker graph and weight precision (npu-kv is the resident INT8 KV-cache decode graph); omit for the KV-cache graph, INT8 for the large model and fp32 for the small"))
        .param(ParamSpec::new("mtp", ParamType::Enum(MtpMode::NAMES.map(String::from).to_vec()), "engine that fills the residual codebooks; omit for the device graph on the large model and the host on the small one"))
        .param(ParamSpec::new("parity", ParamType::Bool, "also report the Talker prefix's max-abs difference to the CPU decoder").default(json!(false)))
}

pub fn synth_spec() -> ActionSpec {
    on_npu(qwen3tts::caps::synth_spec(), "npu-synth", "speaker-free text-to-speech")
}

pub fn clone_spec() -> ActionSpec {
    on_npu(qwen3tts::caps::clone_spec(), "npu-clone", "voice cloning from a reference wav")
}

pub fn design_spec() -> ActionSpec {
    on_npu(qwen3tts::caps::design_spec(), "npu-design", "VoiceDesign and/or CustomVoice preset speakers")
}

/// The model's served manifest plus the NPU actions.
pub fn manifest() -> Manifest {
    let mut m = qwen3tts::caps::manifest();
    m.actions.extend([synth_spec(), clone_spec(), design_spec()]);
    m
}

/// The served provider, extended with the NPU actions over the same checkpoint.
pub fn provider(assembly: &capability::Assembly) -> Result<Arc<dyn Provider>, String> {
    let served = qwen3tts::caps::provider(assembly)?;
    Ok(Arc::new(WithNpu { served, paths: TtsPaths::from_assembly(assembly).ok() }))
}

struct WithNpu {
    served: Arc<dyn Provider>,
    paths: Option<TtsPaths>,
}

impl Provider for WithNpu {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        let kind = match name {
            "npu-synth" => Kind::Synth,
            "npu-clone" => Kind::Clone,
            "npu-design" => Kind::Design,
            served => return self.served.action(served),
        };
        Some(Arc::new(NpuAction { kind, paths: self.paths.clone() }))
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Synth,
    Clone,
    Design,
}

struct NpuAction {
    kind: Kind,
    paths: Option<TtsPaths>,
}

/// The placement an invocation asks for.
fn npu_opts(inv: &Invocation) -> Result<NpuOpts, String> {
    Ok(NpuOpts {
        device: placement::parse_device(&inv.get_str("ov_device").unwrap_or_else(|| "npu".to_string()))?,
        turbo: inv.get_bool("turbo").unwrap_or(false),
        talker: inv.get_str("talker").map(|n| TalkerMode::parse(&n)).transpose()?,
        mtp: inv.get_str("mtp").map(|n| MtpMode::parse(&n)).transpose()?,
        parity: inv.get_bool("parity").unwrap_or(false),
    })
}

impl Action for NpuAction {
    fn spec(&self) -> ActionSpec {
        match self.kind {
            Kind::Synth => synth_spec(),
            Kind::Clone => clone_spec(),
            Kind::Design => design_spec(),
        }
    }

    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let name = self.spec().name;
        let (paths, text, lang, opts) = qwen3tts::caps::speech_request(&self.paths, inv, &name, matches!(self.kind, Kind::Clone))?;
        let npu = npu_opts(inv).map_err(|e| format!("{name}: {e}"))?;
        let cache = paths.npu_cache_dir().map(|p| p.to_string_lossy().into_owned());
        let cache = cache.as_deref();
        let wav = match self.kind {
            Kind::Synth => qwen3tts::pipeline::synth_npu(&paths, &opts, &text, &lang, cache, &npu),
            Kind::Clone => {
                // The Talker and the codec run on the device; the speaker
                // x-vector encoder is the one engine model left, and it is
                // small enough for the CPU backend.
                gpu_core::set_default_backend(gpu_core::Backend::Cpu);
                let (reference, ref_codes) = qwen3tts::caps::reference_from(inv, &name)?;
                let ref_text = inv.get_str("ref_text").unwrap_or_default();
                qwen3tts::pipeline::clone_npu(&paths, &opts, &text, &reference, &ref_text, &lang, ref_codes, cache, &npu)
            }
            Kind::Design => {
                let instruct = inv.get_str("instruct").unwrap_or_default();
                let speaker = inv.get_str("speaker").filter(|s| !s.is_empty());
                qwen3tts::pipeline::design_npu(&paths, &opts, &text, &lang, &instruct, speaker.as_deref(), cache, &npu)
            }
        }
        .map_err(|e| format!("{name}: {e}"))?;
        qwen3tts::caps::audio_outcome(wav, opts.seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::Registry;

    /// Every served action stays, and each NPU action is its served one plus
    /// the placement params - so nothing a flag of `synth`/`clone`/`design`
    /// does is lost on the NPU.
    #[test]
    fn the_npu_actions_extend_the_served_ones_with_the_placement_params() {
        let m = manifest();
        let served = qwen3tts::caps::manifest();
        assert_eq!(m.actions[..served.actions.len()].iter().map(|a| a.name.as_str()).collect::<Vec<_>>(), served.actions.iter().map(|a| a.name.as_str()).collect::<Vec<_>>());
        for (npu, base) in [("npu-synth", "synth"), ("npu-clone", "clone"), ("npu-design", "design")] {
            let (n, b) = (m.actions.iter().find(|a| a.name == npu).unwrap(), served.actions.iter().find(|a| a.name == base).unwrap());
            for p in &b.params {
                assert!(n.params.iter().any(|q| q.name == p.name), "{npu} lacks {base}'s {}", p.name);
            }
            assert_eq!(n.inputs.len(), b.inputs.len(), "{npu}: same input blobs as {base}");
            for p in ["ov_device", "turbo", "talker", "mtp", "parity"] {
                assert!(n.params.iter().any(|q| q.name == p), "{npu}: {p}");
            }
            // The placement choices are unset unless named, so the model's size decides.
            for p in ["talker", "mtp"] {
                assert_eq!(n.params.iter().find(|q| q.name == p).unwrap().default, None, "{npu}: {p} follows the checkpoint");
            }
        }
    }

    #[test]
    fn placement_params_become_the_npu_options() {
        let o = npu_opts(&Invocation::new()).unwrap();
        assert_eq!(o, NpuOpts::default());
        let inv = Invocation::new().set("ov_device", json!("cpu")).set("turbo", json!(true)).set("talker", json!("npu-kv-int4")).set("mtp", json!("fused")).set("parity", json!(true));
        let o = npu_opts(&inv).unwrap();
        assert_eq!((o.turbo, o.talker, o.mtp, o.parity), (true, Some(TalkerMode::NpuKvI4), Some(MtpMode::Fused), true));
        assert_eq!(o.device, placement::parse_device("cpu").unwrap());
        assert!(npu_opts(&Invocation::new().set("talker", json!("gpu"))).is_err());
    }

    /// A request is validated (text, the checkpoint) before any graph is
    /// compiled, so a bad one costs nothing; no checkpoint says how to name one.
    #[test]
    fn a_request_without_text_or_a_checkpoint_is_refused_before_any_compile() {
        let mut reg = Registry::new();
        reg.register(provider(&invoke::resolve::empty_assembly()).unwrap());
        let m = qwen3tts::caps::MODEL;
        let err = reg.run(m, "npu-synth", Invocation::new().set("text", json!(" ")), &mut |_| {}).unwrap_err();
        assert!(err.contains("non-empty"), "got: {err}");
        let err = reg.run(m, "npu-synth", Invocation::new().set("text", json!("hi")), &mut |_| {}).unwrap_err();
        assert!(err.contains("--weights-dir"), "got: {err}");
        // The served actions answer through the same provider.
        assert!(reg.run(m, "synth", Invocation::new().set("text", json!("hi")), &mut |_| {}).unwrap_err().contains("--weights-dir"));
    }
}
