// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! World-model core: the playable [`WorldModel`] trait, the deterministic,
//! GPU-free [`FakeWorldModel`] test model, and host-side dispatch helpers for
//! the world-model kernel families ([`gn`] GroupNorm, [`film`] FiLM/adaLN).
//!
//! Specs: docs/world-models/specs/P1.worldmodel-trait.md, P1.gn.md, P1.film.md

pub mod film;
pub mod gn;
pub mod vq;

/// A playable world model: reset with context, then step one action at a
/// time, receiving one frame per step.
///
/// Frames are CHW-ordered `f32` in `[0, 1]`:
/// `frame[c*H*W + y*W + x]` with `(C, H, W) = frame_shape()`, `c in 0..C`
/// (channel), `y in 0..H` (row, top = 0), `x in 0..W` (column, left = 0).
/// Every [`WorldModel::step`] returns a `Vec<f32>` of length exactly `C*H*W`.
///
/// The trait is object safe: `Box<dyn WorldModel>` is the intended handle in
/// display/CLI code.
pub trait WorldModel {
    /// `(C, H, W)` of every frame this model consumes and produces.
    fn frame_shape(&self) -> (u32, u32, u32);

    /// Number of discrete actions; valid actions are `0..num_actions()`.
    fn num_actions(&self) -> u32;

    /// Re-seed the model from context frames (concatenated CHW frames, may
    /// be empty) and context actions (may be empty). Deterministic: the same
    /// `(ctx_frames, ctx_actions)` always produce the same state.
    fn reset(&mut self, ctx_frames: &[f32], ctx_actions: &[u32]);

    /// Advance one step with `action` (must be `< num_actions()`, panics
    /// otherwise) and return the next frame: CHW `f32` in `[0, 1]`, length
    /// `C*H*W`, freshly allocated.
    fn step(&mut self, action: u32) -> Vec<f32>;

    /// Diffusion-sampler quality knob (number of function evaluations).
    /// Models without an NFE concept ignore it; default is a no-op.
    fn set_nfe(&mut self, _n: u32) {}
}

/// Deterministic, GPU-free test model: a bright 8x8 square (value `1.0` in
/// all channels) moving over a fixed gradient background, 64x64 RGB
/// (`frame_shape() == (3, 64, 64)`), 5 actions
/// (0 = noop, 1 = W/up, 2 = A/left, 3 = S/down, 4 = D/right).
///
/// * Motion and the rendered footprint wrap toroidally at the edges.
/// * [`WorldModel::reset`] seeds the square position from a 64-bit FNV-1a
///   hash of the context **bit patterns** (lengths prepended; NaN payloads
///   legal; `-0.0` and `+0.0` seed differently) — identical
///   `(ctx_frames, ctx_actions)` plus an identical action sequence yield
///   byte-identical frames (`f32::to_bits` equality), across runs and
///   machines.
/// * Background: `R = x/63`, `G = y/63`, `B = 0.25`; the blue channel is
///   `1.0` exactly on the square footprint, so the square position is always
///   recoverable from a frame.
///
/// Exact math, index conventions and hand-computed reference values:
/// docs/world-models/specs/P1.worldmodel-trait.md §4–§5.
pub struct FakeWorldModel {
    /// Square top-left corner, `0..64` each (the only mutable state).
    px: u32,
    py: u32,
}

/// Frame channels (RGB).
const C: u32 = 3;
/// Frame height (rows).
const H: u32 = 64;
/// Frame width (columns).
const W: u32 = 64;
/// Square side length.
const S: u32 = 8;

/// FNV-1a 64 offset basis.
const FNV_OFFSET: u64 = 0xcbf29ce484222325;
/// FNV-1a 64 prime.
const FNV_PRIME: u64 = 0x100000001b3;

impl FakeWorldModel {
    /// Starts at square position `(px, py) = (0, 0)`; usable without
    /// [`WorldModel::reset`].
    pub fn new() -> Self {
        Self { px: 0, py: 0 }
    }
}

impl Default for FakeWorldModel {
    fn default() -> Self {
        Self::new()
    }
}

impl WorldModel for FakeWorldModel {
    fn frame_shape(&self) -> (u32, u32, u32) {
        (C, H, W)
    }

    fn num_actions(&self) -> u32 {
        5
    }

    /// Seeds `(px, py)` from a 64-bit FNV-1a hash over the context byte
    /// stream (spec §4.1): LE `u64` lengths prepended for domain separation,
    /// `f32`s contributed by BIT PATTERN (`to_bits`, little-endian) — NaN
    /// payloads are legal input and `-0.0` vs `+0.0` seed differently —
    /// and `u32` actions little-endian. Then `px = h & 63`,
    /// `py = (h >> 32) & 63`.
    fn reset(&mut self, ctx_frames: &[f32], ctx_actions: &[u32]) {
        let mut h = FNV_OFFSET;
        let mut eat = |bytes: &[u8]| {
            for &b in bytes {
                h = (h ^ b as u64).wrapping_mul(FNV_PRIME);
            }
        };
        eat(&(ctx_frames.len() as u64).to_le_bytes());
        for f in ctx_frames {
            eat(&f.to_bits().to_le_bytes());
        }
        eat(&(ctx_actions.len() as u64).to_le_bytes());
        for a in ctx_actions {
            eat(&a.to_le_bytes());
        }
        self.px = (h & 63) as u32;
        self.py = ((h >> 32) & 63) as u32;
    }

    /// Move-then-render (spec §4.2–§4.3): asserts `action < 5`, moves the
    /// square with toroidal wrap, then renders a fresh frame at the NEW
    /// position.
    fn step(&mut self, action: u32) -> Vec<f32> {
        assert!(
            action < 5,
            "FakeWorldModel::step: action {action} out of range (num_actions = 5)"
        );
        // (dx, dy) encoded as +63 ≡ −1 (mod 64) so nothing underflows in u32.
        let (dx, dy) = match action {
            0 => (0, 0),          // noop
            1 => (0, W - 1),      // W: up (dy = −1)
            2 => (W - 1, 0),      // A: left (dx = −1)
            3 => (0, 1),          // S: down
            4 => (1, 0),          // D: right
            _ => unreachable!(),
        };
        self.px = (self.px + dx) % W;
        self.py = (self.py + dy) % H;

        // Render (pure function of (px, py), spec §4.3).
        let mut frame = vec![0.0f32; (C * H * W) as usize];
        for c in 0..C {
            for y in 0..H {
                for x in 0..W {
                    let in_square =
                        (x + W - self.px) % W < S && (y + H - self.py) % H < S;
                    let v = if in_square {
                        1.0
                    } else {
                        match c {
                            0 => x as f32 / 63.0, // R: left→right ramp
                            1 => y as f32 / 63.0, // G: top→bottom ramp
                            _ => 0.25,            // B: constant
                        }
                    };
                    frame[(c * H * W + y * W + x) as usize] = v;
                }
            }
        }
        frame
    }
}
