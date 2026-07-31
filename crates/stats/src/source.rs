// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The [`StatsSource`] contract and the [`Assembler`] that walks all registered
//! sources into one [`StatsSnapshot`].
//!
//! A source owns one (or a few) subtree(s) and fills them in; sources are additive
//! and independent, so components contribute metrics without any central hardcoded
//! switchboard. To add a whole new data provider, implement [`StatsSource`] and
//! register it; to add a single metric, prefer extending an existing typed section
//! or emitting into an `extra` map (see [`crate::snapshot`]).

use crate::snapshot::StatsSnapshot;

/// A contributor of stats. Each `contribute` fills the section(s) it owns into the
/// shared, in-progress snapshot. Kept object-safe so the assembler can hold a
/// heterogeneous list.
pub trait StatsSource {
    fn contribute(&self, snap: &mut StatsSnapshot);
}

/// Collects [`StatsSource`]s and builds a snapshot by walking them in registration
/// order. Later sources see (and may extend) what earlier ones wrote.
#[derive(Default)]
pub struct Assembler {
    sources: Vec<Box<dyn StatsSource>>,
}

impl Assembler {
    pub fn new() -> Assembler {
        Assembler::default()
    }

    /// Register a source (builder style).
    pub fn register(mut self, source: impl StatsSource + 'static) -> Assembler {
        self.sources.push(Box::new(source));
        self
    }

    /// Register a source in place.
    pub fn add(&mut self, source: impl StatsSource + 'static) -> &mut Self {
        self.sources.push(Box::new(source));
        self
    }

    /// Number of registered sources.
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// Build one snapshot by contributing every registered source into a fresh tree.
    pub fn build(&self) -> StatsSnapshot {
        let mut snap = StatsSnapshot::new();
        for s in &self.sources {
            s.contribute(&mut snap);
        }
        snap
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::{Accelerator, ModelStat};

    /// A fake source that adds `n` accelerators — proves the assembler is purely
    /// data-driven (the count comes from the source, nothing is hardcoded).
    struct FakeAccels(u32);
    impl StatsSource for FakeAccels {
        fn contribute(&self, snap: &mut StatsSnapshot) {
            for i in 0..self.0 {
                snap.accelerators.push(Accelerator { id: format!("gpu{i}"), kind: "gpu".into(), index: i, ..Default::default() });
            }
        }
    }

    struct FakeModels(Vec<&'static str>);
    impl StatsSource for FakeModels {
        fn contribute(&self, snap: &mut StatsSnapshot) {
            for m in &self.0 {
                snap.models.push(ModelStat { id: (*m).into(), ..Default::default() });
            }
        }
    }

    #[test]
    fn assembler_walks_all_registered_sources() {
        let snap = Assembler::new().register(FakeAccels(4)).register(FakeModels(vec!["a", "b"])).build();
        assert_eq!(snap.accelerators.len(), 4);
        assert_eq!(snap.models.len(), 2);
        assert_eq!(snap.schema, crate::snapshot::SCHEMA_VERSION);
    }

    #[test]
    fn sources_are_additive_and_order_preserving() {
        let mut asm = Assembler::new();
        asm.add(FakeAccels(1)).add(FakeAccels(2));
        assert_eq!(asm.len(), 2);
        // Two sources each contribute — the snapshot accumulates both (1 + 2).
        assert_eq!(asm.build().accelerators.len(), 3);
    }
}
