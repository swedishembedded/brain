// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SDL2 window: software-renderer streaming texture (the iGPU's whole compute
//! budget stays with the model), nearest-neighbor scaling, poll-based input.
//! HUD goes in the window title (no in-frame font yet).

use crate::keymap::{Key, KeySet, UxKey};
use crate::sink::{FrameSink, Hud};
use crate::sys;
use std::ffi::CString;

/// Input state snapshot from one pump: latest pressed-set + UX commands +
/// relative mouse motion accumulated over the drained events.
#[derive(Clone, Debug, Default)]
pub struct Input {
    pub pressed: KeySet,
    pub ux: Vec<UxKey>,
    pub quit: bool,
    pub mouse_dx: i32,
    pub mouse_dy: i32,
}

pub struct SdlWindow {
    win: *mut sys::SDL_Window,
    ren: *mut sys::SDL_Renderer,
    tex: *mut sys::SDL_Texture,
    fw: u32,
    fh: u32,
    pressed: KeySet,
    last_title: String,
    /// Blit failures already reported. SDL's presentation calls all return a
    /// code and all of them were being discarded, which turns a broken blit
    /// into a black window and nothing else - the single hardest kind of fault
    /// to diagnose from the outside. They are reported now, and latched,
    /// because a failure that recurs every frame would otherwise bury the
    /// first one under thirty copies a second.
    reported: std::collections::BTreeSet<&'static str>,
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
                reported: std::collections::BTreeSet::new(),
            })
        }
    }

    /// Read back the renderer's current output as RGB24 (post-present).
    /// For self-tests: proves the texture format/pitch path is faithful.
    pub fn read_back(&mut self, w: u32, h: u32) -> Result<Vec<u8>, String> {
        let mut buf = vec![0u8; (w * h * 3) as usize];
        let rc = unsafe {
            sys::SDL_RenderReadPixels(
                self.ren,
                std::ptr::null(),
                sys::SDL_PIXELFORMAT_RGB24,
                buf.as_mut_ptr() as *mut _,
                (w * 3) as i32,
            )
        };
        if rc != 0 {
            return Err(sdl_error("SDL_RenderReadPixels"));
        }
        Ok(buf)
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
                    sys::SDL_MOUSEMOTION => {
                        input.mouse_dx += ev.motion_xrel();
                        input.mouse_dy += ev.motion_yrel();
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

    /// Capture (or release) the mouse for relative look: hides the cursor and
    /// streams unbounded `xrel/yrel` deltas into [`Input::mouse_dx`]/`dy`.
    pub fn set_relative_mouse(&mut self, on: bool) {
        unsafe {
            sys::SDL_SetRelativeMouseMode(on as i32);
        }
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
        99 => Mapped::Action(Key::C),       // c
        0x4000_00E1 => Mapped::Action(Key::Shift), // left shift
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
        118 => Mapped::Ux(UxKey::CycleView),  // v
        112 => Mapped::Ux(UxKey::Screenshot), // p
        109 => Mapped::Ux(UxKey::ToggleMouse), // m
        _ => Mapped::None,
    }
}

impl SdlWindow {
    /// Complain, once, about an SDL call that failed.
    fn blit_failed(&mut self, what: &'static str) {
        if self.reported.insert(what) {
            eprintln!("wm-display: {}", sdl_error(what));
        }
    }
}

impl FrameSink for SdlWindow {
    fn frame(&mut self, rgb: &[u8], w: u32, h: u32, hud: &Hud) {
        // Checked rather than asserted: a debug assertion is compiled out of
        // the release build that people actually run, and a frame of the wrong
        // size is a buffer overrun inside SDL rather than a wrong picture.
        if (w, h) != (self.fw, self.fh) || rgb.len() != (w * h * 3) as usize {
            if self.reported.insert("size") {
                eprintln!(
                    "wm-display: refusing a {w}x{h} frame of {} bytes for a {}x{} window",
                    rgb.len(),
                    self.fw,
                    self.fh
                );
            }
            return;
        }
        unsafe {
            if sys::SDL_UpdateTexture(self.tex, std::ptr::null(), rgb.as_ptr() as *const _, (w * 3) as i32) != 0 {
                self.blit_failed("SDL_UpdateTexture");
            }
            if sys::SDL_RenderClear(self.ren) != 0 {
                self.blit_failed("SDL_RenderClear");
            }
            if sys::SDL_RenderCopy(self.ren, self.tex, std::ptr::null(), std::ptr::null()) != 0 {
                self.blit_failed("SDL_RenderCopy");
            }
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
