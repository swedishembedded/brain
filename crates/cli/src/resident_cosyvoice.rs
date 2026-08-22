// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Resident-model adapter putting CosyVoice behind the residency
//! [`Executor`](residency::Executor), mirroring
//! [`crate::resident_minimaxmusic3::MinimaxMusic3Resident`] - the "load per
//! call, nothing kept warm" shape, not `wan`'s hot-DiT cache.
//!
//! Config is env-only, matching `crates/arch`'s own `weights_env`
//! registration for the `cosyvoice`/`s3tokenizer`/`campplus` architectures
//! and `cosyvoice::pipeline::CosyVoicePaths::from_env`'s own six roles:
//!   * `BRAIN_COSYVOICE_LLM` - speech-token LM (`llm.pt`)
//!   * `BRAIN_COSYVOICE_FLOW` - flow decoder (`flow.pt`)
//!   * `BRAIN_COSYVOICE_HIFT` - HiFT vocoder (`hift.pt`)
//!   * `BRAIN_S3TOKENIZER_V2` - S3Tokenizer FSQ speech tokenizer
//!   * `BRAIN_CAMPPLUS_DIR` - CAM++ speaker encoder
//!   * `BRAIN_COSYVOICE_TOKENIZER` - text BPE tokenizer identity
//!     (`CosyVoice-BlankEN`)
//!
//! One action, `synth`, dispatched straight through
//! `cosyvoice::caps::synth_action` - the SAME param-decode + generation +
//! outcome-shaping implementation `cosyvoice::caps::CosyVoiceProvider` (the
//! direct/`brain do` path) uses, so this file adds no second copy of that
//! logic.
//!
//! # The `MemCost`/sequential-stage-drop tension, and the call made here
//!
//! `residency::ResidentModel::estimate` is consulted BEFORE
//! `activate` and is meant to describe the bytes reserved on the device
//! for as long as the built [`Instance`] stays Hot - the manager reserves
//! that budget once and expects it to remain a fair description of what the
//! instance is holding until it is evicted. `cosyvoice::pipeline::generate`
//! does not match that shape at all: it holds NO checkpoint open across the
//! whole call. Each of its four stages (CAM++ + S3Tokenizer analysis, the
//! speech-token LM, the flow decoder, HiFT) imports its own weights into a
//! block-scoped local, uses them, and drops them before the next stage's
//! import even runs - see that function's own module doc for why (this
//! box's 30 GB RAM and no discrete GPU cannot hold `llm.pt` (~2 GB),
//! `flow.pt` (451 MB CosyVoice 2 / 1.33 GB CosyVoice 3), `hift.pt` (~83 MB),
//! `speech_tokenizer_v2.onnx` (~496 MB) and `campplus.onnx` (~28 MB) all
//! resident at once, or even want to). And [`CosyVoiceInstance`] itself,
//! between calls, holds nothing but a handful of path `String`s - its real
//! idle footprint is negligible, not the number [`estimate`] reports.
//!
//! So the byte figure this file reports is not "what is currently
//! allocated" (nothing is, most of the time) - it is an ADMISSION-CONTROL
//! budget: the largest single stage `generate` will ever hold open at once,
//! which is the peak concurrent pressure one in-flight `synth` call can put
//! on the device, even though it is never held for the instance's whole
//! Hot lifetime. Reporting the SUM of all five checkpoints would
//! over-reserve against a peak that never actually occurs (no stage ever
//! overlaps another); reporting a per-instance-idle number near zero would
//! under-reserve and let the scheduler admit more concurrent `synth` calls
//! than the box can actually survive during the (brief) window a stage
//! really is loaded. Taking the max of the four stages is the same
//! judgment call `crate::resident_minimaxmusic3::MinimaxMusic3Resident::
//! estimate` already makes for the identical reason (its own AR/denoise
//! stages never overlap either) - this is not a new problem this milestone
//! invented a bespoke answer to, it is the same expressiveness gap in the
//! current `MemCost` contract (which models "resident until evicted" costs,
//! not "peaks once per call, stage by stage" costs), answered the same way
//! for the same reason.
//!
//! Unlike `MinimaxMusic3Resident::estimate`, no known promotion factor
//! (e.g. a bf16-on-disk checkpoint silently doubling to fp32 in RAM) applies
//! here: `cosyvoice::llm_import`/`flow_import`/`hift_import` all decode
//! their checkpoints straight into `Vec<f32>`, and `CosyVoiceLm` builds a
//! decode-only `qwen3::Qwen` (`from_tensors_decode`, no backward buffers)
//! over that same `f32` backbone - so a real checkpoint's own file size,
//! not a multiplied estimate, is the honest number for each stage.

use capability::{ActionResult, Invocation, Manifest, Progress};
use cosyvoice::pipeline::CosyVoicePaths;
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

/// CosyVoice behind the scheduler. Holds only resolved PATHS - every real
/// `synth` call loads (and drops) all five components fresh; see
/// `cosyvoice::pipeline::generate`'s own module doc for why a warm cache
/// would be actively wrong here.
pub struct CosyVoiceResident {
    paths: CosyVoicePaths,
}

impl CosyVoiceResident {
    /// Configure from the environment. Returns `None` (not served) when any
    /// of the six roles is unset, like
    /// [`crate::resident::YoloResident::from_env`].
    pub fn from_env() -> Option<CosyVoiceResident> {
        CosyVoicePaths::from_env().ok().map(|paths| CosyVoiceResident { paths })
    }

    /// The size of ONE named checkpoint file inside `dir` - not the whole
    /// directory's total, because `CosyVoicePaths`' six roles may all point
    /// at the SAME directory (the released "one folder holds everything"
    /// layout `cosyvoice::pipeline`'s own module doc and
    /// `crates/cosyvoice/examples/synth.rs`'s fallback both support): summing
    /// whole-directory sizes across roles that alias one directory would
    /// count `llm.pt`/`flow.pt`/`hift.pt` five times over.
    fn file_bytes(dir: &str, filename: &str) -> u64 {
        std::fs::metadata(std::path::Path::new(dir).join(filename)).map(|m| m.len()).unwrap_or(0)
    }
}

impl ResidentModel for CosyVoiceResident {
    fn manifest(&self) -> Manifest {
        // The spec lives in cosyvoice::caps, next to the catalog's own synth
        // spec, so the two surfaces cannot silently diverge.
        cosyvoice::caps::resident_manifest()
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(cosyvoice::caps::MODEL, "default")
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // Four stages, never resident together (see this module's own doc) -
        // budget the LARGEST one, not their sum.
        let analysis = Self::file_bytes(&self.paths.campplus, "campplus.onnx") + Self::file_bytes(&self.paths.s3tokenizer, "speech_tokenizer_v2.onnx");
        let llm = Self::file_bytes(&self.paths.llm, "llm.pt");
        let flow = Self::file_bytes(&self.paths.flow, "flow.pt");
        let hift = Self::file_bytes(&self.paths.hift, "hift.pt");
        let peak = analysis.max(llm).max(flow).max(hift);
        // `llm.pt`'s own real size (~2 GB, this stack's largest single
        // checkpoint) when the files cannot be statted at all (e.g. paths
        // pointing at a directory that does not exist yet).
        let ram = if peak > 0 { peak } else { 2u64 << 30 };
        MemCost::new(0, ram)
    }

    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        // `cosyvoice::pipeline::generate` loads every component internally
        // per call, so the resident's only job here is to fail fast when the
        // configured directories are missing.
        for (dir, role) in [
            (&self.paths.llm, "speech-token LM"),
            (&self.paths.flow, "flow decoder"),
            (&self.paths.hift, "HiFT vocoder"),
            (&self.paths.s3tokenizer, "S3Tokenizer"),
            (&self.paths.campplus, "CAM++"),
            (&self.paths.tokenizer, "text tokenizer"),
        ] {
            if !std::path::Path::new(dir).exists() {
                return Err(format!("cosyvoice: {role} weights not found at {dir} (set the matching BRAIN_COSYVOICE_*/BRAIN_S3TOKENIZER_V2/BRAIN_CAMPPLUS_DIR var)"));
            }
        }
        Ok(Box::new(CosyVoiceInstance { paths: self.paths.clone() }))
    }
}

struct CosyVoiceInstance {
    paths: CosyVoicePaths,
}

impl Instance for CosyVoiceInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        if action != "synth" {
            return Err(format!("cosyvoice: unsupported action '{action}' (this resident declares: synth)"));
        }
        cosyvoice::caps::synth_action(&self.paths, inv, progress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resident_at(dir: &str) -> CosyVoiceResident {
        CosyVoiceResident {
            paths: CosyVoicePaths { llm: dir.to_string(), flow: dir.to_string(), hift: dir.to_string(), s3tokenizer: dir.to_string(), campplus: dir.to_string(), tokenizer: dir.to_string() },
        }
    }

    #[test]
    fn file_bytes_is_zero_for_a_missing_file() {
        assert_eq!(CosyVoiceResident::file_bytes("/definitely/not/a/real/dir", "llm.pt"), 0);
    }

    #[test]
    fn estimate_falls_back_to_llms_own_real_size_when_nothing_can_be_statted() {
        let r = resident_at("/definitely/not/a/real/dir");
        let key = InstanceKey::new(cosyvoice::caps::MODEL, "default");
        assert_eq!(r.estimate(&key).ram, 2u64 << 30);
    }

    #[test]
    fn estimate_picks_the_largest_stage_not_the_sum_even_when_every_role_shares_one_directory() {
        let dir = std::env::temp_dir().join(format!("brain-cosyvoice-resident-estimate-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        // Real relative sizes (llm.pt is this stack's biggest single
        // checkpoint) so the "biggest wins" behaviour is pinned, not just
        // "some deterministic number".
        std::fs::write(dir.join("llm.pt"), vec![0u8; 2_000]).unwrap();
        std::fs::write(dir.join("flow.pt"), vec![0u8; 450]).unwrap();
        std::fs::write(dir.join("hift.pt"), vec![0u8; 80]).unwrap();
        std::fs::write(dir.join("speech_tokenizer_v2.onnx"), vec![0u8; 490]).unwrap();
        std::fs::write(dir.join("campplus.onnx"), vec![0u8; 28]).unwrap();

        let r = resident_at(dir.to_str().unwrap());
        let key = InstanceKey::new(cosyvoice::caps::MODEL, "default");
        let cost = r.estimate(&key);
        // Every role points at the SAME directory here - a naive "sum every
        // directory's total size across roles" estimator would count each
        // file five times over; this must still be the single llm.pt size.
        assert_eq!(cost.ram, 2_000, "expected the single largest stage (llm.pt), not a sum across the aliased roles");
        assert_eq!(cost.vram, 0, "cosyvoice runs host-CPU only");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn instance_key_is_stable_regardless_of_the_invocation() {
        let r = resident_at("/nonexistent");
        let a = r.instance_key("synth", &Invocation::new());
        let b = r.instance_key("synth", &Invocation::new().set("variant", serde_json::json!("cosyvoice3")));
        assert_eq!(a, b);
    }

    #[test]
    fn activate_fails_fast_with_a_named_role_when_weights_are_missing() {
        let r = resident_at("/definitely/not/a/real/dir");
        let key = InstanceKey::new(cosyvoice::caps::MODEL, "default");
        let err = match r.activate(&key, Device::Cpu) {
            Err(e) => e,
            Ok(_) => panic!("activate should have failed against a nonexistent weights dir"),
        };
        assert!(err.contains("speech-token LM"), "unexpected error: {err}");
    }

    #[test]
    fn instance_rejects_an_action_other_than_synth() {
        let mut inst = CosyVoiceInstance {
            paths: CosyVoicePaths { llm: String::new(), flow: String::new(), hift: String::new(), s3tokenizer: String::new(), campplus: String::new(), tokenizer: String::new() },
        };
        let mut progress = |_: Progress| {};
        let err = inst.run("generate", &Invocation::new(), &mut progress).unwrap_err();
        assert!(err.contains("unsupported action"), "unexpected error: {err}");
    }
}
