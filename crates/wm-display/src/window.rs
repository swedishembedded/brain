// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SDL2 window: software-renderer streaming texture (the iGPU's whole compute
//! budget stays with the model), nearest-neighbor scaling, poll-based input.
//! HUD goes in the window title (no in-frame font yet).

use crate::keymap::{Key, KeySet, UxKey};
use crate::sink::{FrameSink, Hud};
use crate::sys;
use std::ffi::CString;

/// Which mouse buttons are down.
///
/// Latched across pumps the same way [`KeySet`] is, because a drag is a state
/// that spans frames while the events that start and end it are instants: a
/// consumer that only saw the events would have to keep this itself, and every
/// consumer would keep it slightly differently.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Buttons {
    pub left: bool,
    pub middle: bool,
    pub right: bool,
}

/// Input state snapshot from one pump: latest pressed-set + UX commands +
/// mouse motion, buttons and wheel accumulated over the drained events.
#[derive(Clone, Debug, Default)]
pub struct Input {
    pub pressed: KeySet,
    pub ux: Vec<UxKey>,
    pub quit: bool,
    pub mouse_dx: i32,
    pub mouse_dy: i32,
    /// Buttons held at the end of this pump.
    pub buttons: Buttons,
    /// Wheel clicks since the last pump, positive away from the user.
    pub wheel: i32,
}

pub struct SdlWindow {
    win: *mut sys::SDL_Window,
    ren: *mut sys::SDL_Renderer,
    tex: *mut sys::SDL_Texture,
    fw: u32,
    fh: u32,
    pressed: KeySet,
    buttons: Buttons,
    last_title: String,
    /// Blit failures already reported. SDL's presentation calls all return a
    /// code and all of them were being discarded, which turns a broken blit
    /// into a black window and nothing else - the single hardest kind of fault
    /// to diagnose from the outside. They are reported now, and latched,
    /// because a failure that recurs every frame would otherwise bury the
    /// first one under thirty copies a second.
    reported: std::collections::BTreeSet<&'static str>,
    /// Pending/completed single-frame capture of the blit path.
    capture: Option<Vec<u8>>,
    presented: u64,
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
            set_hint("SDL_RENDER_SCALE_QUALITY", "0");

            // WHICH RENDERER depends on the video driver, and getting it wrong
            // is invisible from inside the program.
            //
            // The software renderer is this crate's deliberate default:
            // presentation stays on the CPU and the whole GPU compute budget
            // belongs to the model. It does not work on Wayland, where SDL's
            // software path needs a window framebuffer the compositor does not
            // provide, and the result is a window that stays BLACK while every
            // call involved returns success (libsdl-org/sdl2-compat issue 266,
            // "Window framebuffer support not available"). So on Wayland, and
            // only there, ask for an accelerated renderer: it costs the model a
            // textured quad per frame, which is nothing next to not being able
            // to see it.
            //
            // The driver has to be known BEFORE the window is created, because
            // the framebuffer hint below is read at that moment.
            let driver = sys::SDL_GetCurrentVideoDriver();
            let driver = if driver.is_null() {
                String::from("unknown")
            } else {
                std::ffi::CStr::from_ptr(driver).to_string_lossy().into_owned()
            };
            let wayland = driver == "wayland";
            // `BRAIN_WM_RENDERER=software|accelerated` overrides the choice.
            // Which backend can present depends on the display server, the
            // driver and what else in the process has already touched the GPU,
            // and none of that is knowable from in here - so the choice is
            // reachable from outside without a rebuild.
            let forced = std::env::var("BRAIN_WM_RENDERER").unwrap_or_default();
            let want = match forced.as_str() {
                "software" => sys::SDL_RENDERER_SOFTWARE,
                "accelerated" => sys::SDL_RENDERER_ACCELERATED,
                _ if wayland => sys::SDL_RENDERER_ACCELERATED,
                _ => sys::SDL_RENDERER_SOFTWARE,
            };

            // A SOFTWARE renderer presents into the window's framebuffer - and
            // SDL builds that framebuffer, by default, on top of an internal
            // TEXTURE renderer, which on X11 means an OpenGL context SDL
            // creates behind the window and never mentions.
            //
            // In a process that already holds its own GL context for offscreen
            // rendering, that hidden context is a second claimant on the same
            // thread, and the cost is every frame: SDL_RenderPresent returns
            // - it returns void, so there is nothing to check - having drawn
            // nothing, set no error, and sent NOT ONE request to the display
            // server. The window is black and every call reports success.
            //
            // Asking for the plain shared-memory framebuffer instead is what a
            // software renderer was always meant to present into. The
            // difference is measurable from outside the process: one
            // XShmPutImage per frame against none at all, and the window's
            // SDL_WINDOW_OPENGL flag never being set behind our back.
            if want == sys::SDL_RENDERER_SOFTWARE {
                set_hint("SDL_FRAMEBUFFER_ACCELERATION", "0");
            }

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
            let mut ren = sys::SDL_CreateRenderer(win, -1, want);
            if ren.is_null() {
                // Whichever was asked for is unavailable; the other is better
                // than no window, and the reason it was wanted is recorded
                // above rather than lost.
                eprintln!("wm-display: {}", sdl_error("SDL_CreateRenderer"));
                let fallback = if wayland { sys::SDL_RENDERER_SOFTWARE } else { sys::SDL_RENDERER_ACCELERATED };
                ren = sys::SDL_CreateRenderer(win, -1, fallback);
            }
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
            // Shown and raised explicitly. SDL_WINDOW_SHOWN asks for a mapped
            // window and is not the same as the compositor having actually
            // mapped it; on a display server that defers the map until
            // something asks, presenting into an unmapped window is a black
            // rectangle and no error anywhere.
            sys::SDL_ShowWindow(win);
            sys::SDL_RaiseWindow(win);

            // Report what SDL ACTUALLY gave us, not what was asked for.
            // `SDL_CreateRenderer` substitutes a backend of its own choosing
            // and still returns success, so echoing the requested flag is an
            // assertion dressed up as a measurement - and one of those already
            // cost a debugging round trip here.
            let mut info = sys::SDL_RendererInfo::zeroed();
            let backend = if sys::SDL_GetRendererInfo(ren, &mut info) == 0 && !info.name.is_null() {
                std::ffi::CStr::from_ptr(info.name).to_string_lossy().into_owned()
            } else {
                String::from("?")
            };
            let (mut ww, mut wh) = (0, 0);
            sys::SDL_GetWindowSize(win, &mut ww, &mut wh);
            let flags = sys::SDL_GetWindowFlags(win);
            // A software renderer over a GL-backed window framebuffer is the
            // configuration that presents nothing, so say so where it can be
            // read rather than leaving a black window to be interpreted.
            if want == sys::SDL_RENDERER_SOFTWARE && flags & sys::SDL_WINDOW_OPENGL != 0 {
                eprintln!(
                    "wm-display: WARNING: the window framebuffer is OpenGL-backed despite the \
                     software renderer. If this process also renders offscreen with OpenGL the \
                     window will stay black. Set SDL_FRAMEBUFFER_ACCELERATION=0."
                );
            }
            eprintln!(
                "wm-display: {driver} driver, \"{backend}\" renderer (flags {:#x}), {fw}x{fh} frames in a {ww}x{wh} window{}{}",
                info.flags,
                if flags & sys::SDL_WINDOW_SHOWN == 0 { " HIDDEN" } else { "" },
                if wayland { " [wayland: software does not present, asked for accelerated]" } else { "" }
            );

            Ok(SdlWindow {
                win,
                ren,
                tex,
                fw,
                fh,
                pressed: KeySet::empty(),
                buttons: Buttons::default(),
                last_title: String::new(),
                reported: std::collections::BTreeSet::new(),
                capture: None,
                presented: 0,
            })
        }
    }

    /// Ask for the next presented frame to be captured on its way through.
    ///
    /// The capture is taken after the texture has been copied into the
    /// backbuffer and BEFORE the present, which is the only point SDL
    /// documents `SDL_RenderReadPixels` as meaningful on the main target.
    /// What it proves is that the blit path - texture upload, pixel-format
    /// conversion, pitch, scaling - carried the frame intact.
    ///
    /// What it CANNOT prove is that anything reached the screen. There is no
    /// call in SDL that reads a window back off the display server, so a
    /// capture that matches the source frame is consistent with both a
    /// correct window and a completely black one. Read it as clearing the
    /// blit of suspicion, never as clearing the presentation.
    pub fn capture_next_frame(&mut self) {
        self.capture = Some(Vec::new());
    }

    /// The frame captured by [`SdlWindow::capture_next_frame`], if one has
    /// been presented since.
    pub fn captured(&self) -> Option<&[u8]> {
        match &self.capture {
            Some(buf) if !buf.is_empty() => Some(buf),
            _ => None,
        }
    }

    /// How many frames have been handed to `SDL_RenderPresent`.
    ///
    /// A black window with a present count of one means the loop stalled
    /// after the first frame; a black window with a rising count means the
    /// frames are going somewhere that is not the screen. Those are different
    /// faults and the count is the cheapest way to tell them apart.
    pub fn presented(&self) -> u64 {
        self.presented
    }

    /// Drain pending events into an [`Input`] snapshot.
    pub fn pump(&mut self) -> Input {
        let mut input = Input { pressed: self.pressed, buttons: self.buttons, ..Default::default() };
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
                    sys::SDL_MOUSEBUTTONDOWN | sys::SDL_MOUSEBUTTONUP => {
                        let down = ev.kind() == sys::SDL_MOUSEBUTTONDOWN;
                        match ev.button() {
                            sys::SDL_BUTTON_LEFT => input.buttons.left = down,
                            sys::SDL_BUTTON_MIDDLE => input.buttons.middle = down,
                            sys::SDL_BUTTON_RIGHT => input.buttons.right = down,
                            _ => {}
                        }
                    }
                    sys::SDL_MOUSEWHEEL => input.wheel += ev.wheel_y(),
                    _ => {}
                }
            }
        }
        if input.ux.contains(&UxKey::Quit) {
            input.quit = true;
        }
        self.pressed = input.pressed;
        self.buttons = input.buttons;
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

/// Set an SDL hint by name.
fn set_hint(name: &str, value: &str) {
    let (n, v) = (CString::new(name).unwrap(), CString::new(value).unwrap());
    // SAFETY: both pointers are NUL-terminated and live across the call.
    unsafe {
        sys::SDL_SetHint(n.as_ptr(), v.as_ptr());
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
            if matches!(&self.capture, Some(buf) if buf.is_empty()) {
                let mut buf = vec![0u8; (w * h * 3) as usize];
                let rc = sys::SDL_RenderReadPixels(
                    self.ren,
                    std::ptr::null(),
                    sys::SDL_PIXELFORMAT_RGB24,
                    buf.as_mut_ptr() as *mut _,
                    (w * 3) as i32,
                );
                if rc != 0 {
                    self.blit_failed("SDL_RenderReadPixels");
                } else {
                    self.capture = Some(buf);
                }
            }
            // SDL_RenderPresent RETURNS VOID, which is why a present that
            // does nothing is invisible - but it still sets the error string
            // on the way out. Clearing it first makes whatever is there
            // afterwards attributable to this call and nothing earlier.
            sys::SDL_ClearError();
            sys::SDL_RenderPresent(self.ren);
            self.presented += 1;
            let e = sys::SDL_GetError();
            if !e.is_null() {
                let msg = std::ffi::CStr::from_ptr(e).to_string_lossy();
                if !msg.is_empty() && self.reported.insert("present") {
                    eprintln!("wm-display: SDL_RenderPresent: {msg}");
                }
            }

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
