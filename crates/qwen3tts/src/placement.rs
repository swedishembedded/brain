// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Where the compiled OpenVINO/NPU route puts each stage of a synthesis, as
//! values a caller passes in: the target device, the Talker's weight
//! precision and graph kind, the MTP's engine, the codec's streaming mode.
//!
//! Every choice has a name the command line and a manifest spell the same way
//! (`parse`/`NAMES`), and an unset choice resolves from the model's own size
//! ([`TalkerMode::default_for`], [`MtpMode::default_for`]) rather than from
//! process-global state.
//!
//! Swedish Embedded AB implements speech synthesis on NPUs and other edge
//! accelerators for its clients. If your team needs expertise in placing
//! neural audio models on a device's accelerator then you can procure our
//! services by sending an email to info@swedishembedded.com.

use npu::openvino::NpuDevice;

/// The Talker's size from which the large-model defaults apply (the 1.7B
/// checkpoints; the 0.6B is below it).
const LARGE_D_MODEL: u32 = 2048;

/// The OpenVINO device by name (`npu`, `cpu`, `gpu`, `auto`).
pub fn parse_device(name: &str) -> Result<NpuDevice, String> {
    NpuDevice::parse(name).ok_or_else(|| format!("unknown OpenVINO device {name:?} (one of {})", DEVICE_NAMES.join(", ")))
}

pub const DEVICE_NAMES: [&str; 4] = ["npu", "cpu", "gpu", "auto"];

/// Where and how the Talker decodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TalkerMode {
    /// CPU KV-cache decoder (the codec still runs on the device).
    Cpu,
    /// fp32 cache-free graph.
    NpuF32,
    /// Weight-only INT8 cache-free graph.
    NpuI8,
    /// INT8 resident KV-cache decode graph (O(1) projection per frame).
    NpuKvI8,
    /// fp32 resident KV-cache decode graph.
    NpuKvF32,
    /// INT4 weight-compressed resident KV-cache decode graph.
    NpuKvI4,
}

impl TalkerMode {
    pub const NAMES: [&'static str; 6] = ["cpu", "npu", "npu-int8", "npu-kv", "npu-kv-fp32", "npu-kv-int4"];

    pub fn parse(name: &str) -> Result<TalkerMode, String> {
        Ok(match name {
            "cpu" => TalkerMode::Cpu,
            "npu" | "npu-fp32" => TalkerMode::NpuF32,
            "npu-int8" | "int8" => TalkerMode::NpuI8,
            "npu-kv" | "kv" | "npu-kv-int8" => TalkerMode::NpuKvI8,
            "npu-kv-int4" | "int4" => TalkerMode::NpuKvI4,
            "npu-kv-fp32" => TalkerMode::NpuKvF32,
            other => return Err(format!("unknown talker mode {other:?} (one of {})", Self::NAMES.join(", "))),
        })
    }

    /// The resident KV-cache decode graph (measured an order of magnitude
    /// faster per frame than cache-free): INT8 for the large Talker, fp32 for
    /// the small one.
    pub fn default_for(d_model: u32) -> TalkerMode {
        if d_model >= LARGE_D_MODEL {
            TalkerMode::NpuKvI8
        } else {
            TalkerMode::NpuKvF32
        }
    }
}

/// Which engine fills the residual codebooks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MtpMode {
    /// The host decoder.
    Cpu,
    /// The per-substep INT8 KV-cache graph.
    Npu,
    /// All substeps of a frame in one inference.
    Fused,
}

impl MtpMode {
    pub const NAMES: [&'static str; 3] = ["cpu", "npu", "fused"];

    pub fn parse(name: &str) -> Result<MtpMode, String> {
        match name {
            "cpu" => Ok(MtpMode::Cpu),
            "npu" => Ok(MtpMode::Npu),
            "fused" => Ok(MtpMode::Fused),
            other => Err(format!("unknown mtp mode {other:?} (one of {})", Self::NAMES.join(", "))),
        }
    }

    /// The per-substep device graph for the large model, whose host MTP
    /// re-streams its fp32 weights sixteen times a frame; the host for the
    /// small one, whose MTP is cheap.
    pub fn default_for(d_model: u32) -> MtpMode {
        if d_model >= LARGE_D_MODEL {
            MtpMode::Npu
        } else {
            MtpMode::Cpu
        }
    }
}

/// How the resident server decodes codes into audio.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CodecMode {
    /// Stateful streaming graph on the device: each chunk decodes only its new
    /// frames.
    #[default]
    NpuStream,
    /// Sliding-window re-decode on the device.
    Windowed,
    /// The pure-CPU stateful streaming decoder.
    CpuStream,
}

impl CodecMode {
    pub const NAMES: [&'static str; 3] = ["npu-stream", "windowed", "cpu-stream"];

    pub fn parse(name: &str) -> Result<CodecMode, String> {
        match name {
            "npu-stream" => Ok(CodecMode::NpuStream),
            "windowed" => Ok(CodecMode::Windowed),
            "cpu-stream" => Ok(CodecMode::CpuStream),
            other => Err(format!("unknown codec mode {other:?} (one of {})", Self::NAMES.join(", "))),
        }
    }
}

/// One NPU-routed synthesis: the target device and every placement choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NpuOpts {
    pub device: NpuDevice,
    /// Request the NPU's turbo clock: faster, and hotter under sustained load.
    pub turbo: bool,
    /// `None`: [`TalkerMode::default_for`] the checkpoint.
    pub talker: Option<TalkerMode>,
    /// `None`: [`MtpMode::default_for`] the checkpoint.
    pub mtp: Option<MtpMode>,
    /// Also report the Talker prefix's max-abs difference to the CPU decoder.
    pub parity: bool,
}

impl Default for NpuOpts {
    fn default() -> NpuOpts {
        NpuOpts { device: NpuDevice::Npu, turbo: false, talker: None, mtp: None, parity: false }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn talker_modes_parse_by_every_documented_spelling() {
        for (name, mode) in [("cpu", TalkerMode::Cpu), ("npu", TalkerMode::NpuF32), ("npu-fp32", TalkerMode::NpuF32), ("int8", TalkerMode::NpuI8), ("kv", TalkerMode::NpuKvI8), ("npu-kv-int8", TalkerMode::NpuKvI8), ("int4", TalkerMode::NpuKvI4), ("npu-kv-fp32", TalkerMode::NpuKvF32)] {
            assert_eq!(TalkerMode::parse(name), Ok(mode), "{name}");
        }
        for name in TalkerMode::NAMES {
            assert!(TalkerMode::parse(name).is_ok(), "{name} is advertised and must parse");
        }
        assert!(TalkerMode::parse("gpu").unwrap_err().contains("npu-kv"));
    }

    #[test]
    fn unset_modes_follow_the_checkpoint_size() {
        assert_eq!(TalkerMode::default_for(1024), TalkerMode::NpuKvF32);
        assert_eq!(TalkerMode::default_for(2048), TalkerMode::NpuKvI8);
        assert_eq!(MtpMode::default_for(1024), MtpMode::Cpu);
        assert_eq!(MtpMode::default_for(2048), MtpMode::Npu);
    }

    #[test]
    fn mtp_codec_and_device_names_round_trip() {
        for name in MtpMode::NAMES {
            assert!(MtpMode::parse(name).is_ok());
        }
        for name in CodecMode::NAMES {
            assert!(CodecMode::parse(name).is_ok());
        }
        for name in DEVICE_NAMES {
            assert!(parse_device(name).is_ok());
        }
        assert!(MtpMode::parse("gpu").is_err() && CodecMode::parse("x").is_err() && parse_device("tpu").is_err());
        assert_eq!(CodecMode::default(), CodecMode::NpuStream);
    }
}
