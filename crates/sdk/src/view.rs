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
}

/// A window showing a creature.
pub struct View {
    win: wm_display::window::SdlWindow,
    renderer: mujoco::Renderer,
    width: u32,
    height: u32,
    rgb: Vec<u8>,
}

impl View {
    /// Open a window sized `width` x `height` showing `creature`.
    ///
    /// The creature is needed here, not only at [`View::show`], because
    /// MuJoCo's renderer is built against one model and sizes its scene
    /// buffers from it.
    pub fn open(creature: &Creature, title: &str, width: u32, height: u32) -> Result<View, Error> {
        let (model, _) = creature.body_handles();
        let renderer = mujoco::Renderer::for_model(creature.mujoco(), model, width, height).map_err(Error::Backend)?;
        let win = wm_display::window::SdlWindow::new(title, width, height, 1).map_err(Error::Backend)?;
        Ok(View { win, renderer, width, height, rgb: vec![0; (width as usize) * (height as usize) * 3] })
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
        }
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
    pub fn frame(&self) -> &[u8] {
        &self.rgb
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}
