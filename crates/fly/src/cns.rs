// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Loading the animal's nervous system: a cord, or a cord with a brain on it.
//!
//! One function, because the choice between them is one decision and it should
//! be made in one place. Everything downstream - the motor map, the
//! proprioceptors, the gait analysis - keys on published ANNOTATIONS rather
//! than on indices, so it does not care which of the two it was handed. That
//! is the property that makes the brain addable at all, and it is why this
//! module is ten lines of policy rather than a second `Fly`.

use std::path::{Path, PathBuf};

use connectome::bridge::{self, Policy};
use connectome::Connectome;

/// Which nervous system to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cns {
    /// MANC alone: the ventral nerve cord, and nothing that decides anything.
    /// A descending command has to be supplied from outside, which is the
    /// caller standing in for the missing half of the animal.
    Cord,
    /// BANC's brain joined to MANC's cord at the cells that cross between
    /// them. The descending population is then DRIVEN BY the brain rather
    /// than by the caller, and the caller's handle on the animal moves to its
    /// senses. See `connectome::bridge`.
    BrainAndCord,
}

/// Where a dataset's two files and the bridge live, under one root.
///
/// `$BRAIN_CONNECTOME_DIR` in practice: the same directory
/// `tools/buzzfly/collect.sh` assembles, holding `manc/` and, for
/// [`Cns::BrainAndCord`], `banc/` with the `bridge_manc.csv.gz` that
/// `tools/convert/banc_codex.py` writes beside it.
pub fn load(root: impl AsRef<Path>, cns: Cns) -> Result<Connectome, String> {
    let root = root.as_ref();
    let one = |name: &str| -> Result<Connectome, String> {
        let (neurons, edges) = connectome::find(root, name)?;
        connectome::load(name, &neurons, &edges)
    };
    let cord = one("manc")?;
    match cns {
        Cns::Cord => Ok(cord),
        Cns::BrainAndCord => {
            let brain = one("banc")?;
            let path = bridge_path(root)?;
            let crossings = connectome::read_bridge(&path)?;
            let (cns, report) = bridge::join(&brain, &cord, &crossings, Policy::default(), bridge::is_brain)?;
            // A bridge that resolves nothing produces a valid graph in two
            // disconnected halves and every measurement downstream still runs,
            // which is the one failure here that has to be loud.
            if report.merged == 0 {
                return Err(format!(
                    "{}: not one of {} published crossings resolved against both datasets; \
                     the brain and the cord would be two separate networks in one file",
                    path.display(),
                    crossings.len()
                ));
            }
            Ok(cns)
        }
    }
}

/// The bridge file that goes with a BANC export.
fn bridge_path(root: &Path) -> Result<PathBuf, String> {
    let mut tried = Vec::new();
    for dir in [root.join("banc"), root.join("banc-codex"), root.to_path_buf()] {
        let p = dir.join("bridge_manc.csv.gz");
        if p.is_file() {
            return Ok(p);
        }
        tried.push(p.display().to_string());
    }
    Err(format!(
        "no bridge_manc.csv.gz found, so BANC and MANC cannot be joined. \
         tools/convert/banc_codex.py writes it beside the BANC export. Looked in: {}",
        tried.join(", ")
    ))
}
