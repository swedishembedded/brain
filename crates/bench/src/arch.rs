// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The **architecture registry** — the turn-key seam that lets the *same*
//! benchmark battery (see [`crate::registry`]) be run against *any* architecture,
//! and the results compared apples-to-apples.
//!
//! Each entry is an [`Arch`]: a name, a one-line description, a size descriptor
//! ([`Size`] — the depth/width/heads that determine its parameter count), and a
//! **factory** that produces a [`DecoderLm`] (the trainable causal-decoder seam
//! every benchmark already speaks). The eval harness ([`crate::eval`]) takes an
//! arch name, builds its `DecoderLm`, and drives every benchmark through
//! [`Benchmark::evaluate_with`](crate::Benchmark::evaluate_with).
//!
//! ## Adding a NEW architecture (the 3-step recipe)
//! 1. Implement [`DecoderLm`] (and a [`Scorer`](crate::Scorer)) for your model —
//!    one `train_decoder` + one `load_scorer`. Nothing else changes.
//! 2. Add **one line** to [`arch_registry`] registering it (name + size +
//!    `factory`).
//! 3. `make bench/eval ARCH=<name>` — runs the whole battery, writes
//!    `results/<name>-<seed>.json`. Then `make bench/compare` to diff it against
//!    every prior architecture.
//!
//! The GPT baseline is registered as `gpt`; `gpt-small` and `gpt-wide` are size
//! variants registered today so `compare` is demonstrable immediately. Because
//! benchmarks set their own requested depth/width, a size variant is a
//! [`DecoderLm`] that **overrides** the requested size to a fixed shape — see
//! [`ScaledGpt`].

use std::path::Path;

use gpt::model::Gpt;
use gpt::GptConfig;
use model::FitOpts;

use crate::model::{DecoderLm, Scorer, TrainConfig};

/// A size descriptor for an architecture: the hyperparameters that dominate its
/// parameter count. Kept architecture-neutral (depth / width / heads); the
/// concrete model maps them in its own terms. `None` fields mean "let the
/// benchmark choose" (the baseline `gpt` does this).
#[derive(Clone, Copy, Debug, Default)]
pub struct Size {
    pub n_layers: Option<u32>,
    pub d_model: Option<u32>,
    pub n_heads: Option<u32>,
}

impl Size {
    /// A fixed shape that overrides whatever the benchmark requests.
    pub fn fixed(n_layers: u32, d_model: u32, n_heads: u32) -> Self {
        Size { n_layers: Some(n_layers), d_model: Some(d_model), n_heads: Some(n_heads) }
    }

    /// Human-readable shape, e.g. `"L2xD64xH4"` or `"(bench-default)"`.
    pub fn label(&self) -> String {
        match (self.n_layers, self.d_model, self.n_heads) {
            (Some(l), Some(d), Some(h)) => format!("L{l}xD{d}xH{h}"),
            _ => "(bench-default)".to_string(),
        }
    }
}

/// One registered architecture: identity + size + a [`DecoderLm`] factory.
pub struct Arch {
    /// Stable identifier used on the CLI (`--arch <name>`) and in the artifact /
    /// results filename. Lowercase, no spaces.
    pub name: &'static str,
    /// One-line human description.
    pub description: &'static str,
    /// Size descriptor (param-count-bearing; see [`Arch::param_count`]).
    pub size: Size,
    /// Build a fresh [`DecoderLm`] for this architecture. The harness calls this
    /// once per eval run.
    pub factory: fn() -> Box<dyn DecoderLm>,
}

impl Arch {
    /// Total trainable parameter count for this architecture at the given
    /// `vocab` / `block_size`, using the same `GptConfig::param_list` the trainer
    /// allocates from. For a GPT-family arch with a fixed size this is exact; for
    /// an arch that defers sizing to the benchmark it falls back to a reference
    /// `(n_layers=2, d_model=64, n_heads=4)` shape so the artifact still carries a
    /// representative number (documented in the artifact's `param_count_basis`).
    pub fn param_count(&self, vocab: u32, block_size: u32) -> u64 {
        let n_layers = self.size.n_layers.unwrap_or(2);
        let d_model = self.size.d_model.unwrap_or(64);
        let cfg = GptConfig {
            vocab,
            block_size,
            n_layers,
            d_model,
            n_heads: self.size.n_heads.unwrap_or(4),
            d_ff: d_model * 4,
        };
        cfg.param_list().iter().map(|(_, n)| *n as u64).sum()
    }
}

/// The architecture registry: every [`Arch`] the eval harness can score.
///
/// **Adding a NEW architecture is one line here** (after implementing
/// [`DecoderLm`] for it). The baseline `gpt` uses the benchmark's own requested
/// size; `gpt-small` / `gpt-wide` are fixed-shape variants so `compare` shows a
/// real spread today.
pub fn arch_registry() -> Vec<Arch> {
    vec![
        Arch {
            name: "gpt",
            description: "dense GPT decoder baseline (size per benchmark)",
            size: Size::default(),
            factory: || Box::new(crate::GptDecoder),
        },
        Arch {
            name: "gpt-small",
            description: "GPT, fixed small shape (1 layer / d_model 32)",
            size: Size::fixed(1, 32, 2),
            factory: || Box::new(ScaledGpt(Size::fixed(1, 32, 2))),
        },
        Arch {
            name: "gpt-wide",
            description: "GPT, fixed wide shape (2 layers / d_model 96)",
            size: Size::fixed(2, 96, 4),
            factory: || Box::new(ScaledGpt(Size::fixed(2, 96, 4))),
        },
    ]
}

/// Look up an [`Arch`] by name.
pub fn get_arch(name: &str) -> Option<Arch> {
    arch_registry().into_iter().find(|a| a.name == name)
}

/// Names of all registered architectures.
pub fn arch_names() -> Vec<&'static str> {
    arch_registry().into_iter().map(|a| a.name).collect()
}

/// A GPT [`DecoderLm`] that **overrides** the depth/width/heads a benchmark
/// requests with a fixed [`Size`], so a registered size variant trains at *its*
/// shape regardless of the benchmark's defaults. (The baseline [`GptDecoder`]
/// honors the benchmark's requested size.) This is what makes `gpt-small` /
/// `gpt-wide` distinct architectures the same battery can score.
#[derive(Clone, Copy, Debug)]
pub struct ScaledGpt(pub Size);

impl DecoderLm for ScaledGpt {
    fn arch_name(&self) -> &'static str {
        // Variants share the GPT engine; the registry name distinguishes them in
        // artifacts. A static name keeps the trait object simple.
        "gpt"
    }

    fn train_decoder(
        &self,
        dir: &Path,
        block_size: u32,
        cfg: &TrainConfig,
        weights_out: &Path,
    ) -> std::io::Result<(f32, f32)> {
        let n_layers = self.0.n_layers.unwrap_or(cfg.n_layers);
        let d_model = self.0.d_model.unwrap_or(cfg.d_model);
        let n_heads = self.0.n_heads.unwrap_or(cfg.n_heads);
        let gcfg = GptConfig {
            vocab: 0, // inferred from meta.json
            block_size,
            n_layers,
            d_model,
            n_heads,
            d_ff: d_model * 4,
        };
        let opts = FitOpts {
            steps: cfg.steps,
            batch_size: cfg.batch_size,
            block_size,
            lr: cfg.lr,
            warmup: 20,
            decay_iters: cfg.steps * 2,
            eval_interval: 0,
            seed: cfg.seed,
            mask_before: cfg.mask_before,
            mask_per_line: cfg.mask_per_line,
            align_to_lines: cfg.align_to_lines,
            ..Default::default()
        };
        model::train::fit::<Gpt>(dir, gcfg, &opts, Some(weights_out))
    }

    fn load_scorer(&self, weights: &Path, block_size: u32) -> Box<dyn Scorer> {
        // Reuse the GPT baseline's loader (shape is read from the checkpoint).
        crate::GptDecoder.load_scorer(weights, block_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_gpt_and_variants() {
        let names = arch_names();
        assert!(names.contains(&"gpt"));
        assert!(names.contains(&"gpt-small"));
        assert!(names.contains(&"gpt-wide"));
    }

    #[test]
    fn get_arch_builds_a_decoder() {
        let a = get_arch("gpt-wide").expect("gpt-wide registered");
        assert_eq!(a.size.d_model, Some(96));
        let lm = (a.factory)();
        assert_eq!(lm.arch_name(), "gpt");
    }

    #[test]
    fn param_count_grows_with_width() {
        let small = get_arch("gpt-small").unwrap().param_count(32, 32);
        let wide = get_arch("gpt-wide").unwrap().param_count(32, 32);
        assert!(wide > small, "wider arch must have more params: {wide} vs {small}");
    }

    #[test]
    fn unknown_arch_is_none() {
        assert!(get_arch("does-not-exist").is_none());
    }
}
