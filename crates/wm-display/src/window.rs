// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SDL2 window: software-renderer streaming texture (the iGPU's whole compute
//! budget stays with the model), nearest-neighbor scaling, poll-based input.
//! HUD goes in the window title (no in-frame font yet).

use crate::keymap::{Key, KeySet, UxKey};
use crate::sink::{FrameSink, Hud};
use crate::sys;
use std::ffi::CString;

/// Input state snapshot from one pump: latest pressed-set + UX commands.
#[derive(Clone, Debug, Default)]
pub struct Input {
    pub pressed: KeySet,
    pub ux: Vec<UxKey>,
    pub quit: bool,
}

pub struct SdlWindow {
    win: *mut sys::SDL_Window,
    ren: *mut sys::SDL_Renderer,
    tex: *mut sys::SDL_Texture,
    fw: u32,
    fh: u32,
    pressed: KeySet,
    last_title: String,
}

impl SdlWindow {
    /// Create a window scaled `scale`x from the model's `fw x fh` frames.
    /// Fails with the SDL error string when no video driver is available
    /// (headless CI: run with SDL_VIDEODRIVER=dummy or skip).
    pub fn new(title: &str, fw: u32, fh: u32, scale: u32) -> Result<SdlWindow, String> {
        unsafe {
            if sys::SDL_Init(sys::SDL_INIT_VIDEO) != 0 {
                return Err(sdl_error("SDL_Init"));
            }
            // Nearest-neighbor scaling for crisp low-res frames.
            let hint = CString::new("SDL_RENDER_SCALE_QUALITY").unwrap();
            let zero = CString::new("0").unwrap();
            sys::SDL_SetHint(hint.as_ptr(), zero.as_ptr());

            let t = CString::new(title).unwrap();
            let win = sys::SDL_CreateWindow(
                t.as_ptr(),
                sys::SDL_WINDOWPOS_CENTERED,
                sys::SDL_WINDOWPOS_CENTERED,
                (fw * scale.max(1)) as i32,
                (fh * scale.max(1)) as i32,
                sys::SDL_WINDOW_SHOWN,
            );
            if win.is_null() {
                return Err(sdl_error("SDL_CreateWindow"));
            }
            let ren = sys::SDL_CreateRenderer(win, -1, sys::SDL_RENDERER_SOFTWARE);
            if ren.is_null() {
                return Err(sdl_error("SDL_CreateRenderer"));
            }
            let tex = sys::SDL_CreateTexture(
                ren,
                sys::SDL_PIXELFORMAT_RGB24,
                sys::SDL_TEXTUREACCESS_STREAMING,
                fw as i32,
                fh as i32,
            );
            if tex.is_null() {
                return Err(sdl_error("SDL_CreateTexture"));
            }
            Ok(SdlWindow {
                win,
                ren,
                tex,
                fw,
                fh,
                pressed: KeySet::empty(),
                last_title: String::new(),
            })
        }
    }

    /// Drain pending events into an [`Input`] snapshot.
    pub fn pump(&mut self) -> Input {
        let mut input = Input { pressed: self.pressed, ..Default::default() };
        unsafe {
            let mut ev = sys::SDL_Event::zeroed();
            while sys::SDL_PollEvent(&mut ev) != 0 {
                match ev.kind() {
                    sys::SDL_QUIT => input.quit = true,
                    sys::SDL_KEYDOWN if !ev.is_repeat() => {
                        match keycode_to_key(ev.keycode()) {
                            Mapped::Action(k) => input.pressed.press(k),
                            Mapped::Ux(u) => input.ux.push(u),
                            Mapped::None => {}
                        }
                    }
                    sys::SDL_KEYUP => {
                        if let Mapped::Action(k) = keycode_to_key(ev.keycode()) {
                            input.pressed.release(k);
                        }
                    }
                    _ => {}
                }
            }
        }
        if input.ux.contains(&UxKey::Quit) {
            input.quit = true;
        }
        self.pressed = input.pressed;
        input
    }
}

fn sdl_error(what: &str) -> String {
    unsafe {
        let e = sys::SDL_GetError();
        let msg = if e.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(e).to_string_lossy().into_owned()
        };
        format!("{what} failed: {msg}")
    }
}

enum Mapped {
    Action(Key),
    Ux(UxKey),
    None,
}

/// SDL keycode -> chord key or UX key.
fn keycode_to_key(sym: i32) -> Mapped {
    match sym {
        119 => Mapped::Action(Key::W),      // w
        97 => Mapped::Action(Key::A),       // a
        115 => Mapped::Action(Key::S),      // s
        100 => Mapped::Action(Key::D),      // d
        32 => Mapped::Action(Key::Space),   // space
        0x4000_0052 => Mapped::Action(Key::Up),
        0x4000_0051 => Mapped::Action(Key::Down),
        0x4000_0050 => Mapped::Action(Key::Left),
        0x4000_004F => Mapped::Action(Key::Right),
        27 => Mapped::Ux(UxKey::Quit),      // esc
        13 => Mapped::Ux(UxKey::Reset),     // return
        46 => Mapped::Ux(UxKey::Pause),     // .
        101 => Mapped::Ux(UxKey::StepOnce), // e
        91 => Mapped::Ux(UxKey::QualityDown), // [
        93 => Mapped::Ux(UxKey::QualityUp),   // ]
        _ => Mapped::None,
    }
}

impl FrameSink for SdlWindow {
    fn frame(&mut self, rgb: &[u8], w: u32, h: u32, hud: &Hud) {
        debug_assert_eq!((w, h), (self.fw, self.fh));
        debug_assert_eq!(rgb.len(), (w * h * 3) as usize);
        unsafe {
            sys::SDL_UpdateTexture(
                self.tex,
                std::ptr::null(),
                rgb.as_ptr() as *const _,
                (w * 3) as i32,
            );
            sys::SDL_RenderClear(self.ren);
            sys::SDL_RenderCopy(self.ren, self.tex, std::ptr::null(), std::ptr::null());
            sys::SDL_RenderPresent(self.ren);

            let title = format!(
                "brain wm — {} | {:.1}/{} fps | step {}{}{}",
                hud.model,
                hud.fps,
                hud.target_fps,
                hud.step,
                if hud.paused { " | PAUSED" } else { "" },
                if hud.quality > 0 { format!(" | q{}", hud.quality) } else { String::new() },
            );
            if title != self.last_title {
                if let Ok(t) = CString::new(title.clone()) {
                    sys::SDL_SetWindowTitle(self.win, t.as_ptr());
                }
                self.last_title = title;
            }
        }
    }
}

impl Drop for SdlWindow {
    fn drop(&mut self) {
        unsafe {
            sys::SDL_DestroyTexture(self.tex);
            sys::SDL_DestroyRenderer(self.ren);
            sys::SDL_DestroyWindow(self.win);
            sys::SDL_Quit();
        }
    }
}
