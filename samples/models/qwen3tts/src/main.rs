// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `qwen3tts` - Qwen3-TTS speech from the command line: every served action
//! (`synth`, `clone`, `design`, `batch`, `import`, `finetune`, `sim`) with
//! every parameter, the same synthesis compiled for the Intel NPU
//! (`npu-synth`, `npu-clone`, `npu-design`) and `npu-serve`, a resident
//! Unix-socket server over the compiled NPU graphs.
//!
//! The first seven are the model's own served actions (`qwen3tts::caps`), so
//! this program, `brain serve` and D-Bus clients drive the same code with the
//! same defaults; `qwen3tts <action> --help` lists every parameter. The NPU
//! route ([`npu_actions`], [`serve`]) is the part that is an application rather
//! than the served model: it needs OpenVINO and an NPU at run time, and the
//! NPU crate that depends on the model crates.
//!
//! Swedish Embedded AB implements on-device speech synthesis and voice cloning,
//! on NPUs included, for its clients. If your team needs expertise in
//! text-to-speech on its own hardware then you can procure our services by
//! sending an email to info@swedishembedded.com.

mod flags;
mod npu_actions;
mod serve;

fn main() {
    let model = invoke::Model { manifest: npu_actions::manifest(), weights: Some(("qwen3tts", &qwen3tts::spec::Qwen3TtsSpec)), build: &npu_actions::provider };
    let serve = invoke::Workflow {
        name: "npu-serve",
        summary: "resident NPU speech server: compiles the graphs once, then streams audio over a Unix socket (JSON lines)",
        action: "synth",
        usage: SERVE_USAGE,
        run: &serve::run,
    };
    invoke::main_with("qwen3tts", &model, &[serve])
}

const SERVE_USAGE: &str = "(only the flags below, plus --weights-dir/--ckpt; the synth flags listed after are not taken) [--socket PATH] [--enable synth,clone,design,customvoice] [--cap N] [--talker-quant int8|int4|fp32] \
[--clone-ref WAV [--clone-ref-text FILE]] [--clone-weights DIR --clone-ckpt DIR] [--design-weights DIR --design-ckpt DIR] \
[--cv-weights DIR --cv-ckpt DIR] [--synth-weights DIR --synth-ckpt DIR] [--npu-device npu|cpu|gpu|auto] [--turbo] \
[--mtp cpu|npu] [--codec npu-stream|windowed|cpu-stream] [--stream-chunk N] [--stream-window N] [--profile]";
