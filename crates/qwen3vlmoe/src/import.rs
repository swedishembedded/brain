// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GGUF architecture recognition for `qwen3vlmoe`.
//!
//! [`GGUF_ARCHITECTURE`] registers the name so `brain import-gguf` (and
//! `crates/cli/src/gguf_import.rs`'s registry) can *identify* a
//! `Qwen3-VL-30B-A3B` GGUF the moment one exists, rather than routing it to
//! the dense `qwen3vl` importer by name-prefix confusion (the exact defect
//! `brain_arch::by_hf`'s exact-match design exists to rule out - see that
//! crate's `by_hf_omni_does_not_fall_through_to_dense_qwen3` test). This is
//! the SAME sanctioned pattern `crates/supir/src/import.rs`'s own module
//! comment names verbatim: "no real GGUF release exists yet to test against
//! - this importer is registered so one auto-dispatches the day it exists."
//!
//! **What this module does NOT do, and why**: convert a real file. No
//! `Qwen3-VL-30B-A3B` GGUF release was available to inspect in this sandboxed
//! environment (unlike `qwen3vl::gguf_import`'s dense-decoder importer, which
//! was written against a real downloaded `Qwen3-VL-4B-Instruct-Q8_0.gguf` +
//! `mmproj-F16.gguf` pair - see that module's own doc). A llama.cpp/GGUF
//! tensor-name mapping for a 128-expert top-8 sparse MoE decoder is NOT the
//! same shape as the dense `qwen3vl::gguf_import::GGUF_ARCHITECTURE` map (a
//! GGUF MoE decoder packs its routed experts as 3D tensors -
//! `blk.N.ffn_gate_exps.weight` etc., llama.cpp's `LLM_TENSOR_FFN_*_EXPS`
//! convention - a genuinely different leaf vocabulary from a dense decoder's
//! flat 2D per-layer linears), and guessing at that mapping with no real file
//! to verify a single tensor name or shape against is exactly the kind of
//! unverifiable claim this workspace's culture (`AGENTS.md`'s "a finding is a
//! hypothesis until checked") treats as worse than an honest gap. This
//! crate's own vision-language roadmap log records the open item.
//!
//! `general.architecture` for the real release is expected to be
//! `"qwen3vlmoe"` per this workspace's naming rule
//! (`crates/arch/src/lib.rs`'s `Source::LlamaCpp` convention: llama.cpp's
//! `LLM_ARCH_QWEN3VLMOE` lowercased with the prefix dropped) - the same
//! spelling this model's own user-facing documentation already named as "the
//! MoE architecture name" before this crate existed; not independently
//! re-verified against a real llama.cpp checkout in this pass (no such
//! checkout, and no file carrying the tag, were available here).

/// The GGUF `general.architecture` value this architecture is expected to
/// carry. See this module's doc for what is and is not confirmed about it.
pub const GGUF_ARCHITECTURE: &str = "qwen3vlmoe";

/// A GGUF architecture importer's `import` body needs a real file to convert;
/// none has ever been observed for this architecture (see [`GGUF_ARCHITECTURE`]'s
/// doc), so this states that plainly instead of guessing at a tensor-name
/// mapping nothing can test - the same shape as `supir::import::import_gguf`.
pub fn import_gguf(_gguf: &checkpoint::gguf::MmapGguf, _out_path: &str, _id_override: Option<&str>) -> Result<(), String> {
    Err(format!(
        "qwen3vlmoe: no GGUF file with general.architecture={GGUF_ARCHITECTURE:?} (Qwen3-VL-30B-A3B, a \
         128-expert top-8 sparse-MoE decoder) has been available to inspect in this workspace - this \
         importer is registered so one auto-dispatches the day a real release (or a real llama.cpp \
         tensor_mapping.py MoE-expert convention to derive one from) exists, but there is nothing to \
         convert yet. Use a native HuggingFace safetensors checkpoint instead, once `qwen3vlmoe`'s \
         safetensors import path exists."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn architecture_matches_the_naming_rule() {
        assert_eq!(GGUF_ARCHITECTURE, "qwen3vlmoe");
        assert!(GGUF_ARCHITECTURE.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit()), "must be [a-z0-9]+, the arch id grammar");
    }
}
