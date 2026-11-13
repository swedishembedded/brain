// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `npu-serve`: the resident NPU speech server. The compiled OpenVINO graphs
//! of each enabled engine load once, on its first request, and requests then
//! stream their audio back over a Unix socket as JSON lines (the protocol is
//! documented in `qwen3tts::socket`).
//!
//! Every engine is one checkpoint: the one the model store resolves (or
//! `--weights-dir`/`--ckpt` name), unless its own flags name another - a
//! clone engine over a Base checkpoint beside a design engine over a
//! VoiceDesign one, say. An engine is hosted when it is `--enable`d, or, with
//! no `--enable`, when it can run: `synth` always, `clone` when `--clone-ref`
//! names its voice, `design`/`customvoice` when their checkpoint flags are
//! given.
//!
//! Swedish Embedded AB implements low-latency speech synthesis on NPUs for its
//! clients. If your team needs expertise in serving neural speech from an
//! accelerator with the weights kept hot then you can procure our services by
//! sending an email to info@swedishembedded.com.

use qwen3tts::placement::{self, CodecMode, MtpMode};
use qwen3tts::socket::{CloneVoice, EngineName, ModelDirs, ServerCfg, TalkerQuant};

use crate::flags::{take_count, take_flag, take_value};

/// Codec frames per streamed chunk and per sliding window when none is named.
const DEFAULT_STREAM_CHUNK: usize = qwen3tts::engine::DEFAULT_CHUNK_FRAMES;
const DEFAULT_STREAM_WINDOW: usize = 32;

#[cfg(unix)]
pub fn run(call: &invoke::WorkflowCall) -> Result<(), String> {
    let default = ModelDirs { weights_dir: call.assembly.role_path("weights_dir")?, ckpt_dir: call.assembly.role_path("ckpt")? };
    let cfg = config(call.args.to_vec(), &default)?;
    qwen3tts::socket::serve(&cfg)
}

#[cfg(not(unix))]
pub fn run(_call: &invoke::WorkflowCall) -> Result<(), String> {
    Err("npu-serve listens on a Unix domain socket, which this system does not have".into())
}

/// An engine's checkpoint: its own `--<prefix>-weights`/`--<prefix>-ckpt` pair,
/// both or neither, else the default one.
fn dirs(args: &mut Vec<String>, prefix: &str, default: &ModelDirs) -> Result<(ModelDirs, bool), String> {
    let (wf, cf) = (format!("--{prefix}-weights"), format!("--{prefix}-ckpt"));
    match (take_value(args, &wf)?, take_value(args, &cf)?) {
        (Some(weights_dir), Some(ckpt_dir)) => Ok((ModelDirs { weights_dir, ckpt_dir }, true)),
        (None, None) => Ok((default.clone(), false)),
        _ => Err(format!("{wf} and {cf} name one checkpoint together")),
    }
}

/// The server a command line describes, `default` being the resolved checkpoint.
pub fn config(mut args: Vec<String>, default: &ModelDirs) -> Result<ServerCfg, String> {
    let socket = take_value(&mut args, "--socket")?.map_or_else(|| std::env::temp_dir().join("brain-tts.sock"), Into::into);
    let cap = take_count(&mut args, "--cap")?;
    let talker_quant = take_value(&mut args, "--talker-quant")?.map_or(Ok(TalkerQuant::Int8), |v| TalkerQuant::parse(&v))?;
    let mtp = take_value(&mut args, "--mtp")?.map(|v| MtpMode::parse(&v)).transpose()?;
    let codec = take_value(&mut args, "--codec")?.map_or(Ok(CodecMode::default()), |v| CodecMode::parse(&v))?;
    let stream_chunk = take_count(&mut args, "--stream-chunk")?.unwrap_or(DEFAULT_STREAM_CHUNK);
    let stream_window = take_count(&mut args, "--stream-window")?.unwrap_or(DEFAULT_STREAM_WINDOW);
    let device = take_value(&mut args, "--npu-device")?.map_or(Ok(npu::openvino::NpuDevice::Npu), |v| placement::parse_device(&v))?;
    let turbo = take_flag(&mut args, "--turbo");
    let profile = take_flag(&mut args, "--profile");
    let enable = take_value(&mut args, "--enable")?
        .map(|list| list.split(',').map(|n| EngineName::parse(n.trim())).collect::<Result<Vec<_>, _>>())
        .transpose()?;

    let clone_ref = take_value(&mut args, "--clone-ref")?;
    let clone_ref_text = take_value(&mut args, "--clone-ref-text")?;
    let (clone_dirs, _) = dirs(&mut args, "clone", default)?;
    let (design, design_named) = dirs(&mut args, "design", default)?;
    let (custom_voice, cv_named) = dirs(&mut args, "cv", default)?;
    let (synth, _) = dirs(&mut args, "synth", default)?;
    if let Some(unknown) = args.first() {
        return Err(format!("unknown flag {unknown:?}; npu-serve --help lists them"));
    }

    let enabled = |e: EngineName, runnable: bool| enable.as_ref().map_or(runnable, |list| list.contains(&e));
    let clone = if enabled(EngineName::Clone, clone_ref.is_some()) {
        let ref_wav = clone_ref.ok_or("the clone engine needs its voice: --clone-ref WAV")?;
        let ref_text = match clone_ref_text {
            Some(path) => std::fs::read_to_string(&path).map_err(|e| format!("read {path}: {e}"))?.trim().to_string(),
            None => String::new(),
        };
        Some(CloneVoice { model: clone_dirs, ref_wav, ref_text })
    } else {
        None
    };
    Ok(ServerCfg {
        socket,
        target: qwen3tts::npu_gen::Target::new(device, turbo),
        cap,
        talker_quant,
        mtp,
        codec,
        stream_chunk,
        stream_window,
        profile,
        clone,
        design: enabled(EngineName::Design, design_named).then_some(design),
        custom_voice: enabled(EngineName::CustomVoice, cv_named).then_some(custom_voice),
        synth: enabled(EngineName::Synth, true).then_some(synth),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default() -> ModelDirs {
        ModelDirs { weights_dir: "/w".into(), ckpt_dir: "/c".into() }
    }

    fn cfg(args: &[&str]) -> Result<ServerCfg, String> {
        config(args.iter().map(|a| a.to_string()).collect(), &default())
    }

    /// A bare command line serves speaker-free synthesis from the resolved
    /// checkpoint: the one engine that needs nothing else.
    #[test]
    fn a_bare_command_line_serves_synth_from_the_resolved_checkpoint() {
        let c = cfg(&[]).unwrap();
        assert!(c.clone.is_none() && c.design.is_none() && c.custom_voice.is_none());
        assert_eq!(c.synth.as_ref().map(|m| m.weights_dir.as_str()), Some("/w"));
        assert_eq!((c.talker_quant, c.codec, c.stream_chunk, c.stream_window, c.cap), (TalkerQuant::Int8, CodecMode::NpuStream, 16, 32, None));
        assert!(c.socket.ends_with("brain-tts.sock") && !c.target.turbo && !c.profile && c.mtp.is_none());
    }

    #[test]
    fn engines_are_hosted_when_named_or_when_their_flags_make_them_runnable() {
        let c = cfg(&["--clone-ref", "/v.wav", "--design-weights", "/dw", "--design-ckpt", "/dc"]).unwrap();
        assert_eq!(c.clone.as_ref().map(|v| (v.ref_wav.as_str(), v.model.weights_dir.as_str())), Some(("/v.wav", "/w")));
        assert_eq!(c.design.as_ref().map(|m| (m.weights_dir.as_str(), m.ckpt_dir.as_str())), Some(("/dw", "/dc")));
        assert!(c.custom_voice.is_none() && c.synth.is_some());

        // Naming engines hosts exactly those, over the default checkpoint when they name none.
        let c = cfg(&["--enable", "customvoice, design"]).unwrap();
        assert!(c.synth.is_none() && c.clone.is_none());
        assert_eq!(c.custom_voice.as_ref().map(|m| m.ckpt_dir.as_str()), Some("/c"));
        assert!(c.design.is_some());
    }

    #[test]
    fn every_knob_of_the_server_is_a_flag() {
        let c = cfg(&[
            "--socket", "/tmp/s.sock", "--cap", "256", "--talker-quant", "int4", "--mtp", "cpu", "--codec", "windowed", "--stream-chunk", "8", "--stream-window", "24", "--npu-device", "cpu", "--turbo", "--profile",
            "--synth-weights", "/sw", "--synth-ckpt", "/sc",
        ])
        .unwrap();
        assert_eq!((c.socket.to_str(), c.cap, c.talker_quant, c.mtp, c.codec, c.stream_chunk, c.stream_window), (Some("/tmp/s.sock"), Some(256), TalkerQuant::Int4, Some(MtpMode::Cpu), CodecMode::Windowed, 8, 24));
        assert!(c.target.turbo && c.profile);
        assert_eq!(c.target.device, placement::parse_device("cpu").unwrap());
        assert_eq!(c.synth.as_ref().map(|m| m.weights_dir.as_str()), Some("/sw"));
    }

    #[test]
    fn a_bad_command_line_is_refused_with_the_reason() {
        assert!(cfg(&["--enable", "clone"]).unwrap_err().contains("--clone-ref"));
        assert!(cfg(&["--design-weights", "/dw"]).unwrap_err().contains("together"));
        assert!(cfg(&["--talker-quant", "int2"]).unwrap_err().contains("int8"));
        assert!(cfg(&["--enable", "tenor"]).unwrap_err().contains("customvoice"));
        assert!(cfg(&["--cap", "0"]).unwrap_err().contains("positive"));
        assert!(cfg(&["--bogus"]).unwrap_err().contains("--bogus"));
    }

    #[test]
    fn the_clone_voice_transcript_is_read_from_its_file() {
        let dir = std::env::temp_dir().join(format!("qwen3tts-serve-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let text = dir.join("ref.txt");
        std::fs::write(&text, "  words spoken in the reference \n").unwrap();
        let c = cfg(&["--clone-ref", "/v.wav", "--clone-ref-text", text.to_str().unwrap()]).unwrap();
        assert_eq!(c.clone.unwrap().ref_text, "words spoken in the reference");
        assert!(cfg(&["--clone-ref", "/v.wav", "--clone-ref-text", "/nonexistent/ref.txt"]).unwrap_err().contains("read"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
