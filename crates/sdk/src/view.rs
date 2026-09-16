// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`View`]: a window onto a [`Creature`].
//!
//! MuJoCo renders offscreen on the GPU and the frame is blitted into an SDL
//! window on the CPU, which is this workspace's existing arrangement for a
//! realtime display: presentation stays off the compute device so the whole of
//! it belongs to the model.
//!
//! The window is not required. [`View::open`] fails with a readable error when
//! there is no display, and [`View::frame`] hands back the last rendered
//! pixels either way, so a headless run can write them to a file and still be
//! checking the same path a human would be looking at.

use crate::{Creature, Error};

use wm_display::keymap::{Key, KeySet, UxKey};
use wm_display::sink::{FrameSink, Hud};

/// What the keyboard said this frame.
///
/// The MOUSE is not in here, because nothing outside this module has to act on
/// it: dragging and scrolling move the camera, and the camera belongs to the
/// [`View`]. See [`View::steering`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Steering {
    /// The window was closed, or Escape was pressed.
    pub quit: bool,
    /// W and S: increase and decrease the descending command.
    pub faster: bool,
    pub slower: bool,
    /// A and D.
    pub left: bool,
    pub right: bool,
    /// R.
    pub reset: bool,
    /// C, HELD: lesion the proprioceptive channel. Held rather than toggled,
    /// because a lesion you have to keep your finger on is one you cannot
    /// forget you left enabled.
    pub lesion: bool,
    /// Space, HELD: more wing power. Held for the same reason a throttle is
    /// held - a fly that keeps flying because you tapped a key once is a fly
    /// you are not controlling.
    pub throttle: bool,
    /// Up and Down arrows: trim, whatever the caller uses trim for.
    pub up: bool,
    pub down: bool,
}

/// How far a gesture moves the camera.
///
/// Rates rather than fitted constants: an orbit is degrees per pixel dragged,
/// a zoom is a ratio per wheel click, and a pan is a fraction of the viewing
/// DISTANCE per pixel - so a drag covers the same span on screen however far
/// out the camera is, which is what makes dragging feel like dragging.
const ORBIT_DEG_PER_PIXEL: f64 = 0.35;
const ZOOM_PER_CLICK: f64 = 0.88;
const PAN_DISTANCES_PER_PIXEL: f64 = 0.0022;
/// Straight down either pole is a singularity in an azimuth/elevation camera:
/// the scene spins about the view axis and the controls stop meaning anything,
/// so the orbit stops just short of both.
const MAX_ELEVATION: f64 = 89.0;

/// A window showing a creature.
pub struct View {
    win: wm_display::window::SdlWindow,
    renderer: mujoco::Renderer,
    width: u32,
    height: u32,
    rgb: Vec<u8>,
    /// Where the camera looks RELATIVE to the animal, in the model's own
    /// length units. Zero is a camera locked on it, which is the default and
    /// what [`View::follow`] restores every frame; a right-button drag moves
    /// this and it then persists, so somebody who deliberately looked at the
    /// ground beside the fly keeps looking there while the fly walks off it.
    offset: [f64; 3],
}

impl View {
    /// Open a window sized `width` x `height` showing `creature`.
    ///
    /// The creature is needed here, not only at [`View::show`], because
    /// MuJoCo's renderer is built against one model and sizes its scene
    /// buffers from it.
    pub fn open(creature: &Creature, title: &str, width: u32, height: u32) -> Result<View, Error> {
        // The RENDERER first, and the order is load-bearing in both
        // directions. Building the GL context after SDL has brought up X11
        // fails outright on a Mesa stack - EGL cannot get a screen once the
        // display server owns one. Building it first and LEAVING IT CURRENT
        // fails the other way, because SDL's X11 setup then has to switch the
        // thread off a foreign context and the server refuses with a BadAccess
        // on X_GLXMakeCurrent.
        //
        // So: build it first, and have it hand the context back the moment it
        // is done. Nothing holds a current context except the frame itself.
        let (model, _) = creature.body_handles();
        let renderer = mujoco::Renderer::for_model(creature.mujoco(), model, width, height).map_err(Error::Backend)?;
        let win = wm_display::window::SdlWindow::new(title, width, height, 1).map_err(Error::Backend)?;
        Ok(View {
            win,
            renderer,
            width,
            height,
            rgb: vec![0; (width as usize) * (height as usize) * 3],
            offset: [0.0; 3],
        })
    }

    /// Drain the keyboard.
    pub fn steering(&mut self) -> Steering {
        let input = self.win.pump();
        self.mouse(&input);
        Steering {
            quit: input.quit,
            faster: input.pressed.contains(KeySet::of(&[Key::W])),
            slower: input.pressed.contains(KeySet::of(&[Key::S])),
            left: input.pressed.contains(KeySet::of(&[Key::A])),
            right: input.pressed.contains(KeySet::of(&[Key::D])),
            reset: input.ux.contains(&UxKey::Reset),
            lesion: input.pressed.contains(KeySet::of(&[Key::C])),
            throttle: input.pressed.contains(KeySet::of(&[Key::Space])),
            up: input.pressed.contains(KeySet::of(&[Key::Up])),
            down: input.pressed.contains(KeySet::of(&[Key::Down])),
        }
    }

    /// Apply this pump's mouse gestures to the camera.
    ///
    /// Drag to orbit, drag with the right button to pan, wheel to zoom - the
    /// three gestures every 3D viewer has, in the arrangement MuJoCo's own
    /// `simulate` uses, so anybody who has driven one of those already knows
    /// this one.
    ///
    /// The camera is written directly rather than nudged through
    /// `mjv_moveCamera`, because the thing being asked for here is absolute -
    /// "look at the fly" - and a relative nudge can only approximate an
    /// absolute request through a shadow copy of the state it is nudging. See
    /// `mujoco::Renderer::camera`.
    fn mouse(&mut self, input: &wm_display::window::Input) {
        let (cam, offset) = gesture(self.renderer.camera(), self.offset, input);
        self.offset = offset;
        // Infallible in practice and still checked: every value `gesture`
        // produces is a finite number times a finite rate. A pose that is NOT
        // finite renders an empty frame rather than failing, and an empty
        // frame is indistinguishable from a window that never presented.
        let _ = self.renderer.set_camera(cam);
    }

    /// Keep the camera pointed at the creature.
    ///
    /// Without this a flying fly leaves the frame in about a second, and what
    /// is on screen after that is an empty floor - which reads as the fly
    /// having failed rather than as the camera having been left behind.
    ///
    /// `lag` in `[0, 1]` is how much of the error to take out per call: `1.0`
    /// pins the animal to the centre and makes the world slide about, and
    /// something small follows it the way a camera operator would.
    pub fn follow(&mut self, creature: &Creature, lag: f64) {
        let lag = lag.clamp(0.0, 1.0);
        let want = creature.position();
        let mut cam = self.renderer.camera();
        for ((look, at), off) in cam.lookat.iter_mut().zip(want).zip(self.offset) {
            *look += (at + off - *look) * lag;
        }
        let _ = self.renderer.set_camera(cam);
    }

    /// Render the creature as it is now and put it on the screen.
    ///
    /// `status` goes in the window title, which is where this workspace's
    /// display puts a HUD.
    pub fn show(&mut self, creature: &Creature, status: &str) -> Result<(), Error> {
        let (model, data) = creature.body_handles();
        self.renderer.render(model, data).map_err(Error::Backend)?;
        self.rgb = self.renderer.rgb_top_down();
        let hud = Hud { model: status.to_string(), ..Default::default() };
        self.win.frame(&self.rgb, self.width, self.height, &hud);
        Ok(())
    }

    /// The last frame [`View::show`] put on the screen, RGB8, top row first.
    ///
    /// This is what was HANDED to the window. See [`View::window_frame`] for
    /// what the window actually has, which is not the same question.
    pub fn frame(&self) -> &[u8] {
        &self.rgb
    }

    /// Capture the next presented frame as it goes through the blit path.
    ///
    /// The difference between this and [`View::frame`] is texture upload,
    /// pixel-format conversion, pitch and scaling - so a mismatch localises
    /// the fault to the blit and a match clears it.
    ///
    /// It does NOT say the window is showing anything. SDL has no call that
    /// reads a window back off the display server, so presentation is simply
    /// not observable from inside the process, and an instrument here that
    /// claimed otherwise sent a black-window report down the wrong path once
    /// already. `cargo run -p brain-wm-display --example window_smoke` is the
    /// test that answers the presentation question, by putting an
    /// unmistakable pattern on the screen with nothing else running.
    pub fn capture_next_frame(&mut self) {
        self.win.capture_next_frame();
    }

    /// The frame captured by [`View::capture_next_frame`], once one has been
    /// presented.
    pub fn captured(&self) -> Option<&[u8]> {
        self.win.captured()
    }

    /// How many frames have been presented.
    pub fn presented(&self) -> u64 {
        self.win.presented()
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}

/// One pump's mouse gestures applied to a camera pose and a look-at offset.
///
/// Pure, and separate from [`View`] for one reason: the signs are the whole
/// content of it, they cannot be derived (MuJoCo documents neither the
/// handedness nor the zero of `mjvCamera`'s azimuth), and a function that
/// needs a window and a GPU to run cannot be asserted about. The convention
/// this is built on is MEASURED, in
/// `crates/mujoco/tests/render.rs::moving_the_lookat_towards_plus_y_sweeps_the_subject_right`:
/// at azimuth 0 the camera is on -x looking towards +x, world +y is on the
/// LEFT of the frame, and pushing the look-at point towards +y sweeps the
/// subject right.
///
/// Everything here is GRAB-AND-DRAG: whatever is under the cursor goes where
/// the cursor goes. Dragging right sweeps the subject right, which means the
/// look-at point moves with the drag and, for an orbit, the camera goes the
/// other way.
fn gesture(
    mut cam: mujoco::CameraPose,
    mut offset: [f64; 3],
    input: &wm_display::window::Input,
) -> (mujoco::CameraPose, [f64; 3]) {
    let (dx, dy) = (input.mouse_dx as f64, input.mouse_dy as f64);
    if input.wheel != 0 {
        cam.distance *= ZOOM_PER_CLICK.powi(input.wheel);
    }
    if input.buttons.right || input.buttons.middle {
        // In the camera's OWN screen plane, not in world axes, so the scene
        // moves with the cursor from wherever it is being watched from. Both
        // vectors come out of the pose the camera already holds, which is the
        // whole reason for reading it back rather than shadowing it.
        let (az, el) = (cam.azimuth.to_radians(), cam.elevation.to_radians());
        let screen_right = [-az.sin(), az.cos(), 0.0];
        let screen_up = [-el.sin() * az.cos(), -el.sin() * az.sin(), el.cos()];
        let step = cam.distance * PAN_DISTANCES_PER_PIXEL;
        for i in 0..3 {
            offset[i] += (screen_right[i] * dx + screen_up[i] * dy) * step;
        }
    } else if input.buttons.left {
        cam.azimuth -= dx * ORBIT_DEG_PER_PIXEL;
        cam.elevation = (cam.elevation + dy * ORBIT_DEG_PER_PIXEL).clamp(-MAX_ELEVATION, MAX_ELEVATION);
    }
    (cam, offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wm_display::window::{Buttons, Input};

    const POSE: mujoco::CameraPose =
        mujoco::CameraPose { lookat: [0.0; 3], distance: 2.0, azimuth: 0.0, elevation: 0.0 };

    fn drag(buttons: Buttons, dx: i32, dy: i32) -> Input {
        Input { mouse_dx: dx, mouse_dy: dy, buttons, ..Default::default() }
    }

    #[test]
    fn the_wheel_zooms_both_ways_and_nothing_else_moves() {
        let (closer, off) = gesture(POSE, [0.0; 3], &Input { wheel: 1, ..Default::default() });
        assert!(closer.distance < POSE.distance, "a click towards the screen did not zoom in");
        let (further, _) = gesture(POSE, [0.0; 3], &Input { wheel: -1, ..Default::default() });
        assert!(further.distance > POSE.distance, "a click back did not zoom out");
        assert_eq!((closer.azimuth, closer.elevation, off), (POSE.azimuth, POSE.elevation, [0.0; 3]));
    }

    /// The camera goes the other way from the drag, which is what makes the
    /// subject follow it.
    #[test]
    fn dragging_right_orbits_the_camera_the_other_way() {
        let left = Buttons { left: true, ..Default::default() };
        let (turned, off) = gesture(POSE, [0.0; 3], &drag(left, 100, 0));
        assert!(turned.azimuth < POSE.azimuth, "dragging right did not orbit");
        assert_eq!(off, [0.0; 3], "an orbit moved the look-at point");
    }

    #[test]
    fn the_orbit_stops_short_of_both_poles() {
        let left = Buttons { left: true, ..Default::default() };
        let (down, _) = gesture(POSE, [0.0; 3], &drag(left, 0, 100_000));
        let (up, _) = gesture(POSE, [0.0; 3], &drag(left, 0, -100_000));
        assert_eq!(down.elevation, MAX_ELEVATION);
        assert_eq!(up.elevation, -MAX_ELEVATION);
    }

    /// The measured convention, restated as an assertion on the sign that
    /// depends on it: at azimuth 0, +y is screen-left, so a rightward drag has
    /// to push the look-at point towards +y for the subject to follow the
    /// cursor. Getting this backwards is not a crash - it is a camera that
    /// runs away from the mouse.
    #[test]
    fn dragging_right_pans_the_lookat_towards_plus_y() {
        let right = Buttons { right: true, ..Default::default() };
        let (cam, off) = gesture(POSE, [0.0; 3], &drag(right, 100, 0));
        assert!(off[1] > 0.0, "a rightward pan moved the look-at point to {off:?}");
        assert!(off[0].abs() < 1e-12 && off[2].abs() < 1e-12, "a horizontal pan moved something else: {off:?}");
        assert_eq!((cam.azimuth, cam.elevation, cam.distance), (POSE.azimuth, POSE.elevation, POSE.distance));
    }

    /// Dragging DOWN has to move the subject down, which means the aim point
    /// goes up: the far more obvious "aim lower" is the inversion users report
    /// as the camera fighting them.
    #[test]
    fn dragging_down_pans_the_lookat_up() {
        let right = Buttons { right: true, ..Default::default() };
        let (_, off) = gesture(POSE, [0.0; 3], &drag(right, 0, 100));
        assert!(off[2] > 0.0, "a downward pan moved the look-at point to {off:?}");
    }

    /// A pan scaled in world units alone would crawl when zoomed out and fly
    /// when zoomed in. It is a fraction of the viewing DISTANCE instead, so
    /// the same drag covers the same span of the picture either way.
    #[test]
    fn a_pan_covers_the_same_screen_distance_at_any_zoom() {
        let right = Buttons { right: true, ..Default::default() };
        let (_, near) = gesture(POSE, [0.0; 3], &drag(right, 100, 0));
        let far = mujoco::CameraPose { distance: POSE.distance * 10.0, ..POSE };
        let (_, out) = gesture(far, [0.0; 3], &drag(right, 100, 0));
        assert!((out[1] - near[1] * 10.0).abs() < 1e-9, "{out:?} is not ten times {near:?}");
    }
}
