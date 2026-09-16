// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Recorded fly walking, and the reward that scores against it.

use std::path::Path;

const MAGIC: &[u8; 8] = b"BRNFLYW1";

/// Reference trajectories: real flies walking, recorded and retargeted onto
/// this body.
///
/// Written by `tools/convert/flybody_walking_reference.py` from the published
/// HDF5, so that reading them needs no hdf5 dependency in the engine.
///
/// `Debug` prints the shape and not the frames: a reference holds tens of
/// millions of floats and a derived Debug would make any assertion that
/// mentions one unreadable.
pub struct Reference {
    timestep: f64,
    nq: usize,
    nv: usize,
    lengths: Vec<usize>,
    offsets: Vec<usize>,
    data: Vec<f32>,
    moving: Vec<u32>,
}

impl std::fmt::Debug for Reference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let frames: usize = self.lengths.iter().sum();
        f.debug_struct("Reference")
            .field("snippets", &self.lengths.len())
            .field("frames", &frames)
            .field("nq", &self.nq)
            .field("nv", &self.nv)
            .field("timestep", &self.timestep)
            .finish()
    }
}

impl Reference {
    pub fn load(path: impl AsRef<Path>) -> Result<Reference, String> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::parse(&bytes).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn parse(bytes: &[u8]) -> Result<Reference, String> {
        // 8 magic + 8 timestep + 3 * 4 counts.
        if bytes.len() < 28 || &bytes[..8] != MAGIC {
            return Err("not a brain fly-walking reference (bad magic)".to_string());
        }
        let rd_u32 = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()) as usize;
        let timestep = f64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let (nq, nv, count) = (rd_u32(16), rd_u32(20), rd_u32(24));
        if !(timestep.is_finite() && timestep > 0.0) {
            return Err(format!("timestep {timestep} is not a positive duration"));
        }
        let head = 28 + count * 4;
        if bytes.len() < head {
            return Err("truncated before the length table".to_string());
        }
        let lengths: Vec<usize> = (0..count).map(|i| rd_u32(28 + i * 4)).collect();

        let stride = nq + nv;
        let mut offsets = Vec::with_capacity(count);
        let mut frames = 0usize;
        for &l in &lengths {
            offsets.push(frames);
            frames += l;
        }
        let want = head + frames * stride * 4;
        if bytes.len() != want {
            return Err(format!(
                "expected {want} bytes for {frames} frames of {stride} floats, file has {}",
                bytes.len()
            ));
        }
        let data: Vec<f32> = bytes[head..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let moving = moving_dofs(&data, nq, nv, stride);
        Ok(Reference { timestep, nq, nv, lengths, offsets, data, moving })
    }

    /// Seconds per frame. Measured 2.000 ms on the published dataset, which is
    /// exactly the 500 Hz control period flybody's own walking tasks use - so
    /// one reference frame is one control tick and nothing is resampled.
    pub fn timestep(&self) -> f64 {
        self.timestep
    }
    pub fn nq(&self) -> usize {
        self.nq
    }
    pub fn nv(&self) -> usize {
        self.nv
    }
    pub fn snippets(&self) -> usize {
        self.lengths.len()
    }
    pub fn len(&self, snippet: usize) -> usize {
        self.lengths.get(snippet).copied().unwrap_or(0)
    }
    pub fn is_empty(&self) -> bool {
        self.lengths.is_empty()
    }

    /// `(qpos, qvel)` for one frame, or `None` past the end of the snippet.
    pub fn frame(&self, snippet: usize, i: usize) -> Option<(&[f32], &[f32])> {
        if i >= self.len(snippet) {
            return None;
        }
        let stride = self.nq + self.nv;
        let base = (self.offsets[snippet] + i) * stride;
        Some((&self.data[base..base + self.nq], &self.data[base + self.nq..base + stride]))
    }

    /// Velocity indices that actually vary somewhere in the recording.
    ///
    /// Measured on the published walking data: 42 of 102 joints move, and they
    /// are exactly the seven leg DoF on each of six legs. The other 60 - head,
    /// rostrum, haustellum, labrum, antennae, wings, halteres, abdomen - are
    /// not tracked by a walking recording and sit at zero for every frame.
    ///
    /// This matters for the reward rather than being trivia. Scoring velocity
    /// over ALL degrees of freedom makes the reference's zeros a target, so a
    /// creature is penalised for moving its wings or neck at all - and this
    /// connectome drives 66 wing and 24 neck motor neurons, so that penalty is
    /// real and is not what "learn to walk" should mean. flybody scores every
    /// DoF and adds a separate wing-retraction term; scoring only what was
    /// recorded says the same thing without asking the reward to mean two
    /// things at once.
    pub fn moving_dofs(&self) -> &[u32] {
        &self.moving
    }

    /// Check the reference describes the model it will be scored against.
    ///
    /// A reference with a different joint count is a reference for a different
    /// animal, and every reward computed from it would be a comparison against
    /// the wrong body while looking perfectly well-formed.
    pub fn check_matches(&self, nq: usize, nv: usize) -> Result<(), String> {
        if self.nq != nq || self.nv != nv {
            return Err(format!(
                "reference is for a body with nq={} nv={}, this model has nq={nq} nv={nv}",
                self.nq, self.nv
            ));
        }
        Ok(())
    }
}

/// Velocity indices whose value differs from the first frame anywhere.
fn moving_dofs(data: &[f32], nq: usize, nv: usize, stride: usize) -> Vec<u32> {
    let frames = if stride == 0 { 0 } else { data.len() / stride };
    let mut moving = vec![false; nv];
    if frames == 0 {
        return Vec::new();
    }
    for f in 1..frames {
        for k in 0..nv {
            if !moving[k] && (data[f * stride + nq + k] - data[nq + k]).abs() > 1e-6 {
                moving[k] = true;
            }
        }
    }
    (0..nv as u32).filter(|&k| moving[k as usize]).collect()
}

/// The DeepMimic-style imitation reward, in the reduced form this binding can
/// compute.
///
/// flybody scores walking with four multiplicative Gaussian factors
/// (`flybody/tasks/rewards.py`): centre of mass, joint velocities, egocentric
/// end-effector vectors, and joint orientation quaternions. The first two come
/// out of MuJoCo's flat state API; the last two need site positions and joint
/// axes from mjData/mjModel, which `crates/mujoco` does not mirror. So this is
/// two factors of four, with the published standard deviations and weights
/// kept exactly rather than re-tuned - a re-tuned constant would make the
/// number incomparable with the reference implementation for no gain.
///
/// Why imitation at all, rather than rewarding forward speed: a displacement
/// reward scores a single coordinated lunge as highly as a gait, and measuring
/// the ceiling under one showed exactly that - the optimiser suppressed the
/// cord's recurrent circuitry and drove sensory input straight to the muscles,
/// which is a reflex. Tracking a recorded gait asks for the gait.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImitationReward {
    pub com_std: f64,
    pub com_weight: f64,
    pub qvel_std: f64,
    pub qvel_weight: f64,
}

impl Default for ImitationReward {
    /// flybody's own constants for the fruit-fly walking imitation task.
    fn default() -> Self {
        ImitationReward { com_std: 0.078487, com_weight: 20.0, qvel_std: 53.7801, qvel_weight: 1.0 }
    }
}

impl ImitationReward {
    /// `(com_factor, qvel_factor)`, each a weighted unnormalised Gaussian of
    /// the summed squared difference, exactly as `reward_factors_deep_mimic`
    /// computes them.
    pub fn factors(&self, qpos: &[f64], qvel: &[f64], ref_qpos: &[f32], ref_qvel: &[f32]) -> (f64, f64) {
        // Centre of mass is the root free joint's translation, the first three
        // entries of qpos.
        let com: f64 = (0..3)
            .map(|i| {
                let d = qpos.get(i).copied().unwrap_or(0.0) - ref_qpos.get(i).copied().unwrap_or(0.0) as f64;
                d * d
            })
            .sum();
        let vel: f64 = qvel
            .iter()
            .zip(ref_qvel)
            .map(|(a, b)| {
                let d = a - *b as f64;
                d * d
            })
            .sum();
        (
            self.com_weight * (-0.5 / (self.com_std * self.com_std) * com).exp(),
            self.qvel_weight * (-0.5 / (self.qvel_std * self.qvel_std) * vel).exp(),
        )
    }

    /// The same, scoring velocity only over the given DoF indices.
    ///
    /// See [`Reference::moving_dofs`] for why restricting it is the right
    /// default here: the untracked two thirds of the reference are zeros, and
    /// scoring against them turns the reward into "hold still" for every joint
    /// a walking recording did not capture.
    pub fn factors_over(
        &self,
        qpos: &[f64],
        qvel: &[f64],
        ref_qpos: &[f32],
        ref_qvel: &[f32],
        dofs: &[u32],
    ) -> (f64, f64) {
        let com: f64 = (0..3)
            .map(|i| {
                let d = qpos.get(i).copied().unwrap_or(0.0) - ref_qpos.get(i).copied().unwrap_or(0.0) as f64;
                d * d
            })
            .sum();
        let vel: f64 = dofs
            .iter()
            .map(|&k| {
                let a = qvel.get(k as usize).copied().unwrap_or(0.0);
                let b = ref_qvel.get(k as usize).copied().unwrap_or(0.0) as f64;
                (a - b) * (a - b)
            })
            .sum();
        (
            self.com_weight * (-0.5 / (self.com_std * self.com_std) * com).exp(),
            self.qvel_weight * (-0.5 / (self.qvel_std * self.qvel_std) * vel).exp(),
        )
    }

    /// [`Self::factors_over`], multiplied. See [`Self::total`].
    pub fn total_over(
        &self,
        qpos: &[f64],
        qvel: &[f64],
        ref_qpos: &[f32],
        ref_qvel: &[f32],
        dofs: &[u32],
    ) -> f64 {
        let (a, b) = self.factors_over(qpos, qvel, ref_qpos, ref_qvel, dofs);
        a * b
    }

    /// The factors MULTIPLIED, which is the scalar reward.
    ///
    /// A product, not a sum, and the difference is the whole character of the
    /// objective: `flybody/tasks/base.py` returns `np.prod(...)` over the
    /// factors its walking task produces. Under a sum, a creature that gets
    /// its joint velocities roughly right collects most of what the velocity
    /// term can pay while its centre of mass drifts anywhere it likes. Under a
    /// product, a dead factor kills the reward outright, so there is no
    /// partial credit for matching one feature while abandoning another.
    ///
    /// The cost of a product is that it is zero over most of the space, which
    /// is exactly why the imitation literature pairs it with early termination
    /// and reference-state initialisation - see [`Self::com_distance`], which
    /// is what an episode terminates on.
    pub fn total(&self, qpos: &[f64], qvel: &[f64], ref_qpos: &[f32], ref_qvel: &[f32]) -> f64 {
        let (a, b) = self.factors(qpos, qvel, ref_qpos, ref_qvel);
        a * b
    }

    /// Distance between the body's centre of mass and the reference's, in the
    /// model's own length units.
    ///
    /// The termination criterion. flybody ends a walking episode once this
    /// exceeds 0.33 cm, which is about 1.3 body lengths, and that is what
    /// keeps a product reward from spending an entire episode at zero.
    pub fn com_distance(qpos: &[f64], ref_qpos: &[f32]) -> f64 {
        (0..3)
            .map(|i| {
                let d = qpos.get(i).copied().unwrap_or(0.0) - ref_qpos.get(i).copied().unwrap_or(0.0) as f64;
                d * d
            })
            .sum::<f64>()
            .sqrt()
    }

    /// The largest value [`Self::total`] can return: a perfect match.
    ///
    /// Worth having explicitly, because a reward whose scale is unknown makes
    /// "it improved by 0.3" uninterpretable.
    pub fn max(&self) -> f64 {
        self.com_weight * self.qvel_weight
    }
}
