// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The scene's one dimension the host decides: the embrasure behind the
//! glass is exactly the display's size, so its front edge IS the glass and
//! its walls converge correctly only when the projection is right.
//! Everything else (the objects, the far landscape, the light) is laid out
//! in `shaders/scene.wgsl`, in the screen frame (metres, origin at the
//! display centre, +Z toward the viewer), so it stays fixed to the laptop
//! while the viewer moves.

/// The embrasure: it fills the display and runs back `depth` metres.
#[derive(Clone, Copy, Debug)]
pub struct Tank {
    pub half_w: f32,
    pub half_h: f32,
    pub depth: f32,
}

impl Tank {
    pub fn for_display(width_m: f64, height_m: f64) -> Tank {
        Tank { half_w: width_m as f32 / 2.0, half_h: height_m as f32 / 2.0, depth: (width_m * 0.9) as f32 }
    }
}
