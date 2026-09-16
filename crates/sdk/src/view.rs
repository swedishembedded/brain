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

/// A window showing a creature.
/// World distance the camera's look-at point travels per unit of pan, in
/// multiples of the scene's extent - one constant per axis.
///
/// Measured rather than read off the API: `mjv_moveCamera` takes "a fraction
/// of the window", which is not a distance, and the conversion is exactly what
/// a tracking shot needs. `crates/mujoco/examples/pan_calib.rs` puts markers
/// at known world positions, pans by known amounts and inverts the screen
/// displacement; the relationship is linear to three significant figures over
/// the range tested.
///
/// The TWO NUMBERS ARE DIFFERENT, and assuming they were equal is what made
/// the first tracking camera point at empty floor: a pan's two components run
/// along the camera's right and forward vectors, and forward is tilted out of
/// the horizontal plane, so the same fraction of the window covers more ground
/// along it. Sideways is negative because a positive drag pulls the scene one
/// way and therefore the look-at point the other.
const PAN_X_PER_EXTENT: f64 = -1.262;
const PAN_Y_PER_EXTENT: f64 = 1.469;

pub struct View {
    win: wm_display::window::SdlWindow,
    renderer: mujoco::Renderer,
    width: u32,
    height: u32,
    rgb: Vec<u8>,
    /// Where the camera is currently looking, tracked here because MuJoCo will
    /// not say: the field is inside `mjvCamera` and this binding reads none of
    /// them. Every pan this type issues updates the estimate, so it stays true
    /// as long as nothing else moves the camera.
    lookat: [f64; 2],
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
            lookat: [0.0, 0.0],
        })
    }

    /// Drain the keyboard.
    pub fn steering(&mut self) -> Steering {
        let input = self.win.pump();
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
        let [x, y, _] = creature.position();
        let extent = creature.scene_extent();
        if extent <= 0.0 {
            return;
        }
        // Aim slightly SHORT of the animal rather than straight at it. The
        // camera looks down at the floor, so a look-at point level with the
        // ground puts a flying fly low in the frame and a climbing one out of
        // it; pulling the aim point back along the camera's forward axis lifts
        // the subject to where a camera operator would hold it.
        let target_y = y - 0.3 * extent;
        let (dx, dy) = ((x - self.lookat[0]) * lag, (target_y - self.lookat[1]) * lag);
        self.renderer.move_camera(
            mujoco::Camera::PanH,
            dx / (PAN_X_PER_EXTENT * extent),
            dy / (PAN_Y_PER_EXTENT * extent),
        );
        self.lookat = [self.lookat[0] + dx, self.lookat[1] + dy];
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
