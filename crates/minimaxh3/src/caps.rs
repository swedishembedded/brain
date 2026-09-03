// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniMax-H3's capability surface - the [`capability`] `Provider`/`Action`
//! wiring lands here in full once the model itself exists. For now this
//! module holds the one piece that has to exist before ANY other H3 code
//! runs: the license gate.
//!
//! ## Why this gate exists
//!
//! The MiniMax H3 Community License Agreement is not Apache-2.0. It carries:
//! a territorial carve-out (the ordinary grant excludes the EU, UK, South
//! Korea and the US - an organization in one of those regions needs
//! MiniMax's separate authorization), a >$20M/yr revenue registration
//! clause, and a mandatory "MiniMax H3" UI attribution requirement for any
//! commercial product or service. This is a stricter license than
//! `minimaxmusic3`'s (which has no commercial-use restriction, so that
//! crate carries only a prose note in its own user-facing docs page, no
//! runtime gate) - modeled instead on the one runtime license gate that
//! exists in this workspace, `flux2::caps::check_license`, byte-for-byte.
//!
//! This crate's Rust *implementation* is Apache-2.0, ported from the
//! Apache-2.0 `diffusers`/`transformers` reference (see this crate's module
//! doc). What this gate protects is the *weights*: brain never vendors or
//! auto-fetches them (`arch::ARCHS`'s `minimaxh3` row has `default_ref:
//! None`), and running against a checkpoint the operator obtained themself
//! still requires an explicit opt-in - the same "the code is Apache, the
//! weights are the operator's problem to clear" split `supir`/`flux2`'s 9B
//! variant already establish in this tree.

use std::sync::Once;

/// Refuse to run against MiniMax-H3 weights unless the operator has
/// confirmed they are authorized under the MiniMax H3 Community License
/// Agreement (including, if applicable, its territorial restrictions) -
/// then print the attribution notice once per process.
///
/// Called from every entry point that would touch real H3 weights (the
/// eventual `gen_params_from`-equivalent in this module, and each CLI
/// handler), the same call-site discipline `flux2::caps::check_license`
/// uses so no served surface can bypass it.
pub fn check_license() -> Result<(), String> {
    if std::env::var("BRAIN_MINIMAXH3_ALLOW_COMMUNITY").ok().as_deref() != Some("1") {
        return Err(
            "MiniMax-H3 weights are released under the MiniMax H3 Community License Agreement \
             (territorial restrictions apply - the ordinary grant excludes the EU, UK, South \
             Korea and the US; a >$20M/yr revenue registration clause and a mandatory \"MiniMax \
             H3\" UI attribution requirement also apply). Set \
             BRAIN_MINIMAXH3_ALLOW_COMMUNITY=1 to confirm you are authorized to use these \
             weights under that license."
                .into(),
        );
    }
    static NOTICE: Once = Once::new();
    NOTICE.call_once(|| {
        eprintln!(
            "minimaxh3: MiniMax H3 Community License weights enabled - territorial and \
             revenue-cap restrictions apply"
        );
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gated_unless_opted_in() {
        // The gate reads the env var per call; only assert the refusing path
        // here (setting env vars in tests races other tests in the binary -
        // same discipline as `flux2::caps`'s equivalent test).
        if std::env::var("BRAIN_MINIMAXH3_ALLOW_COMMUNITY").ok().as_deref() != Some("1") {
            let err = check_license().unwrap_err();
            assert!(err.contains("BRAIN_MINIMAXH3_ALLOW_COMMUNITY"), "error must name the opt-in: {err}");
            assert!(err.contains("Community License"), "error must name the license: {err}");
        }
    }
}
