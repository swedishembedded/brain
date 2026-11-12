// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Where the viewer's eye is, for sources that are not a camera: a fixed
//! point, the mouse, or a scripted orbit. They exist so the projection and the
//! shaders can be judged on their own, with no tracking noise in the picture,
//! and they always say which they are - none of them is ever presented as
//! tracking.

/// A screen-frame eye, metres.
pub type Eye = [f64; 3];

/// The rest position every source starts from: centred, 55 cm out.
pub const REST: Eye = [0.0, 0.0, 0.55];

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Source {
    /// One eye position for the whole run.
    Fixed(Eye),
    /// The cursor steers the eye across the glass, the wheel moves it nearer
    /// or further, the arrow keys nudge it.
    Manual,
    /// A slow Lissajous orbit around [`REST`].
    Synthetic,
}

impl Source {
    pub fn parse(s: &str) -> Result<Source, String> {
        match s {
            "manual" => Ok(Source::Manual),
            "synthetic" => Ok(Source::Synthetic),
            other => Err(format!("--pose-source must be manual or synthetic, got {other:?}")),
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Source::Fixed(_) => "fixed eye",
            Source::Manual => "manual eye (mouse, wheel, arrows)",
            Source::Synthetic => "synthetic orbit",
        }
    }
}

/// The scripted orbit at time `t` seconds: wide enough sideways to show the
/// parallax, with a little height and depth so neither is mistaken for a
/// pan.
pub fn synthetic(t: f64) -> Eye {
    [0.13 * (0.55 * t).sin(), 0.045 * (0.83 * t + 0.6).sin(), REST[2] + 0.09 * (0.37 * t).sin()]
}

/// The manual eye: the cursor's place on the glass, scaled out so a modest
/// mouse move sweeps a head-sized range, at distance `z`.
pub fn manual(cursor_on_glass: [f64; 2], z: f64) -> Eye {
    [cursor_on_glass[0] * 1.6, cursor_on_glass[1] * 1.6, z]
}

/// Parse `x,y,z` in centimetres.
pub fn parse_cm(s: &str) -> Result<Eye, String> {
    let v: Vec<f64> = s.split(',').map(|p| p.trim().parse::<f64>()).collect::<Result<_, _>>().map_err(|e| format!("--eye {s:?}: {e}"))?;
    match v.as_slice() {
        [x, y, z] if *z > 1.0 => Ok([x / 100.0, y / 100.0, z / 100.0]),
        [_, _, _] => Err(format!("--eye {s:?}: the eye must be in front of the glass (z > 1 cm)")),
        _ => Err(format!("--eye {s:?}: expected x,y,z in centimetres")),
    }
}
