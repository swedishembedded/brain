// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for seeing which models a server
// holds, builds and evicts while it serves, for its clients. If your team
// needs expertise in observable multi-model serving then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Each instance's life as an observer reads it ([`ResidencyManager::lives`]): whether it is resident, being built,
//! demoted or evicted, and when its last build and eviction happened and how long they took.
//!
//! Recorded at the manager's own transitions (claim of a cold or warm instance, adoption, a failed build, every way an
//! instance leaves its device), so it cannot disagree with the residency it describes. A workload's instances
//! ([`ResidentModel::transient`](crate::ResidentModel::transient)) are not recorded: each run builds its own and
//! drops it, and the job's own record says what it did.

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::ResidencyManager;
use crate::{Device, InstanceKey, Tier};

/// Milliseconds since the Unix epoch now.
fn unix_ms_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

/// An instance's last build: when it ended, how long it took, and whether it succeeded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BuildRecord {
    pub ended_unix_ms: u64,
    pub duration: Duration,
    pub succeeded: bool,
}

/// An instance's last eviction: when, how long it had been resident, and whether it was dropped or demoted to the
/// warm tier (kept off its card, rebuilt from host memory).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvictionRecord {
    pub at_unix_ms: u64,
    /// Since it was last built; `None` when it was never seen built (adopted before the manager recorded).
    pub resident_for: Option<Duration>,
    pub demoted: bool,
}

/// What the manager records of one instance between its transitions.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Life {
    building_since: Option<Instant>,
    resident_since: Option<Instant>,
    last_build: Option<BuildRecord>,
    last_eviction: Option<EvictionRecord>,
}

/// Where an instance is now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LifeState {
    /// Built and hot on its device(s).
    Resident,
    /// Claimed and being built (or promoted from the warm tier).
    Building,
    /// Demoted to the warm tier: off its card, promoted back on its next claim.
    Warm,
    /// Built once and evicted since.
    Evicted,
    /// Claimed and never built (its builds failed).
    NotLoaded,
}

impl LifeState {
    /// The state's stable name in a metrics document.
    pub fn slug(self) -> &'static str {
        match self {
            LifeState::Resident => "resident",
            LifeState::Building => "building",
            LifeState::Warm => "warm",
            LifeState::Evicted => "evicted",
            LifeState::NotLoaded => "not_loaded",
        }
    }
}

/// One instance's life as [`ResidencyManager::lives`] reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstanceLife {
    pub key: InstanceKey,
    pub state: LifeState,
    /// The devices it occupies (resident, building or warm); empty when evicted.
    pub devices: Vec<Device>,
    /// How long the build in progress has run.
    pub building_for: Option<Duration>,
    pub last_build: Option<BuildRecord>,
    pub last_eviction: Option<EvictionRecord>,
}

impl ResidencyManager {
    /// Whether `key` is a workload's instance, which is not recorded (see the module doc).
    fn transient_key(&self, key: &InstanceKey) -> bool {
        self.models.get(&key.model).is_some_and(|m| m.transient())
    }

    /// `key` was claimed cold or warm: its build starts now.
    pub(super) fn life_building(&mut self, key: &InstanceKey) {
        if !self.transient_key(key) {
            self.lives.entry(key.clone()).or_default().building_since = Some(Instant::now());
        }
    }

    /// `key`'s build ended, built or not.
    pub(super) fn life_built(&mut self, key: &InstanceKey, succeeded: bool) {
        let Some(life) = self.lives.get_mut(key) else { return };
        let Some(since) = life.building_since.take() else { return };
        life.last_build = Some(BuildRecord { ended_unix_ms: unix_ms_now(), duration: since.elapsed(), succeeded });
        if succeeded {
            life.resident_since = Some(Instant::now());
        }
    }

    /// `key` left its device: dropped, abandoned, or `demoted` to the warm tier.
    pub(super) fn life_evicted(&mut self, key: &InstanceKey, demoted: bool) {
        let Some(life) = self.lives.get_mut(key) else { return };
        life.building_since = None;
        let resident_for = life.resident_since.take().map(|t| t.elapsed());
        life.last_eviction = Some(EvictionRecord { at_unix_ms: unix_ms_now(), resident_for, demoted });
    }

    /// Forget the lives of `model`'s instances (it was retired).
    pub(super) fn life_forget(&mut self, model: &str) {
        self.lives.retain(|k, _| k.model != model);
    }

    /// Every recorded instance's life now, ordered by key.
    pub fn lives(&self) -> Vec<InstanceLife> {
        let mut lives: Vec<InstanceLife> = self
            .lives
            .iter()
            .map(|(key, life)| {
                let single = self.residents.get(key).map(|e| (vec![e.device], e.tier));
                let multi = self.multi_residents.get(key).map(|e| (e.devices.clone(), Tier::Hot));
                let placed = single.or(multi);
                let state = match (&placed, life.building_since) {
                    (_, Some(_)) => LifeState::Building,
                    (Some((_, Tier::Hot)), None) if self.instances.contains_key(key) => LifeState::Resident,
                    (Some(_), None) => LifeState::Warm,
                    (None, None) if life.last_eviction.is_some() => LifeState::Evicted,
                    (None, None) => LifeState::NotLoaded,
                };
                InstanceLife {
                    key: key.clone(),
                    state,
                    devices: placed.map(|(d, _)| d).unwrap_or_default(),
                    building_for: life.building_since.map(|t| t.elapsed()),
                    last_build: life.last_build,
                    last_eviction: life.last_eviction,
                }
            })
            .collect();
        lives.sort_by(|a, b| (&a.key.model, &a.key.config).cmp(&(&b.key.model, &b.key.config)));
        lives
    }
}
