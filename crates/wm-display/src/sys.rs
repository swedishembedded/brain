// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Minimal SDL2 FFI - exactly the calls the world-model window
//! needs, loaded from the system libSDL2 at run time ([`load`]) when a window
//! opens. No sdl2/sdl2-sys crate: brain hand-rolls small FFI surfaces (cf. the
//! vendored ONNX protobuf) and needs ~25 functions.
//!
//! ABI notes: SDL2's `SDL_Event` is a 56-byte C union; we reserve 64 bytes and
//! read fields at their documented offsets (type @0; key events: keycode/`sym`
//! @20, `repeat` @13). Pixel format RGB24 matches the engine's interleaved
//! RGB8 frames byte-for-byte.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::os::raw::{c_char, c_int, c_void};

pub type SDL_Window = c_void;
pub type SDL_Renderer = c_void;
pub type SDL_Texture = c_void;

pub const SDL_INIT_VIDEO: u32 = 0x0000_0020;
pub const SDL_WINDOWPOS_CENTERED: c_int = 0x2FFF_0000u32 as c_int;
pub const SDL_WINDOW_SHOWN: u32 = 0x0000_0004;
/// Set on a window whose framebuffer SDL decided to back with OpenGL.
pub const SDL_WINDOW_OPENGL: u32 = 0x0000_0002;
pub const SDL_RENDERER_SOFTWARE: u32 = 0x0000_0001;
pub const SDL_RENDERER_ACCELERATED: u32 = 0x0000_0002;
// SDL_PIXELFORMAT_RGB24: tightly packed interleaved R,G,B bytes.
// SDL_DEFINE_PIXELFORMAT(ARRAYU8=7, ARRAYORDER_RGB=1, layout 0, 24 bits,
// 3 bytes) = 0x17101803 — printed from SDL2/SDL_pixels.h by a C program,
// NOT recomputed by hand (a hand-derived value shipped wrong once and
// garbled the whole window; tests/sdl_roundtrip.rs now guards this).
pub const SDL_PIXELFORMAT_RGB24: u32 = 0x1710_1803;
pub const SDL_TEXTUREACCESS_STREAMING: c_int = 1;

/// `SDL_RendererInfo`. The name and flags say which backend SDL ACTUALLY
/// picked, which is not the same question as which one was asked for -
/// `SDL_CreateRenderer` falls back on its own and reports success either way,
/// so a message that echoes the requested flag is not a measurement.
#[repr(C)]
pub struct SDL_RendererInfo {
    pub name: *const c_char,
    pub flags: u32,
    pub num_texture_formats: u32,
    pub texture_formats: [u32; 16],
    pub max_texture_width: c_int,
    pub max_texture_height: c_int,
}

impl SDL_RendererInfo {
    pub fn zeroed() -> SDL_RendererInfo {
        SDL_RendererInfo {
            name: std::ptr::null(),
            flags: 0,
            num_texture_formats: 0,
            texture_formats: [0; 16],
            max_texture_width: 0,
            max_texture_height: 0,
        }
    }
}

// Event types.
pub const SDL_QUIT: u32 = 0x100;
pub const SDL_KEYDOWN: u32 = 0x300;
pub const SDL_KEYUP: u32 = 0x301;
pub const SDL_MOUSEMOTION: u32 = 0x400;
pub const SDL_MOUSEBUTTONDOWN: u32 = 0x401;
pub const SDL_MOUSEBUTTONUP: u32 = 0x402;
pub const SDL_MOUSEWHEEL: u32 = 0x403;

/// SDL's mouse button numbers, as `SDL_MouseButtonEvent.button` reports them.
pub const SDL_BUTTON_LEFT: u8 = 1;
pub const SDL_BUTTON_MIDDLE: u8 = 2;
pub const SDL_BUTTON_RIGHT: u8 = 3;

/// 64-byte buffer covering SDL2's 56-byte SDL_Event union.
#[repr(C, align(8))]
#[derive(Clone, Copy)]
pub struct SDL_Event(pub [u8; 64]);

impl SDL_Event {
    pub fn zeroed() -> SDL_Event {
        SDL_Event([0u8; 64])
    }
    pub fn kind(&self) -> u32 {
        u32::from_ne_bytes([self.0[0], self.0[1], self.0[2], self.0[3]])
    }
    /// SDL_KeyboardEvent.keysym.sym (SDL_Keycode, i32 at byte offset 20).
    pub fn keycode(&self) -> i32 {
        i32::from_ne_bytes([self.0[20], self.0[21], self.0[22], self.0[23]])
    }
    /// SDL_KeyboardEvent.repeat (u8 at byte offset 13): key-repeat events.
    pub fn is_repeat(&self) -> bool {
        self.0[13] != 0
    }
    /// SDL_MouseMotionEvent.xrel (i32 at byte offset 28: type 0, timestamp 4,
    /// windowID 8, which 12, state 16, x 20, y 24, xrel 28, yrel 32).
    pub fn motion_xrel(&self) -> i32 {
        i32::from_ne_bytes([self.0[28], self.0[29], self.0[30], self.0[31]])
    }
    /// SDL_MouseMotionEvent.yrel (i32 at byte offset 32).
    pub fn motion_yrel(&self) -> i32 {
        i32::from_ne_bytes([self.0[32], self.0[33], self.0[34], self.0[35]])
    }
    /// SDL_MouseButtonEvent.button (u8 at byte offset 16: type 0, timestamp 4,
    /// windowID 8, which 12, button 16, state 17, clicks 18, padding 19).
    pub fn button(&self) -> u8 {
        self.0[16]
    }
    /// SDL_MouseWheelEvent.y (i32 at byte offset 20: type 0, timestamp 4,
    /// windowID 8, which 12, x 16, y 20). Positive is away from the user.
    pub fn wheel_y(&self) -> i32 {
        i32::from_ne_bytes([self.0[20], self.0[21], self.0[22], self.0[23]])
    }
}

/// Declares SDL's entry points once: the table [`load`] binds them into, and
/// a same-named wrapper for each that calls through it.
macro_rules! sdl_api {
    ($(fn $name:ident($($arg:ident: $ty:ty),*) $(-> $ret:ty)?;)*) => {
        /// The loaded library and every entry point, resolved once.
        struct Api {
            // Dropping this unloads the library, so it must outlive every
            // pointer read out of it. Held, never read.
            _lib: libloading::Library,
            $($name: unsafe extern "C" fn($($ty),*) $(-> $ret)?,)*
        }

        impl Api {
            fn bind(lib: libloading::Library, path: &str) -> Result<Api, String> {
                // A `Symbol` borrows `lib`; dereferencing copies the bare
                // pointer out, valid while `lib` (moved in last) is loaded.
                Ok(Api {
                    $($name: *unsafe { lib.get::<unsafe extern "C" fn($($ty),*) $(-> $ret)?>(concat!(stringify!($name), "\0").as_bytes()) }
                        .map_err(|e| format!("{path}: missing symbol {}: {e}", stringify!($name)))?,)*
                    _lib: lib,
                })
            }
        }

        $(
            /// # Safety
            /// As the SDL function of the same name; [`load`] must have succeeded.
            pub unsafe fn $name($($arg: $ty),*) $(-> $ret)? {
                unsafe { (api().$name)($($arg),*) }
            }
        )*
    };
}

sdl_api! {
    fn SDL_Init(flags: u32) -> c_int;
    fn SDL_Quit();
    fn SDL_GetError() -> *const c_char;
    fn SDL_ClearError();
    fn SDL_SetHint(name: *const c_char, value: *const c_char) -> c_int;
    fn SDL_CreateWindow(title: *const c_char, x: c_int, y: c_int, w: c_int, h: c_int, flags: u32) -> *mut SDL_Window;
    fn SDL_DestroyWindow(win: *mut SDL_Window);
    fn SDL_SetWindowTitle(win: *mut SDL_Window, title: *const c_char);
    fn SDL_CreateRenderer(win: *mut SDL_Window, index: c_int, flags: u32) -> *mut SDL_Renderer;
    fn SDL_DestroyRenderer(r: *mut SDL_Renderer);
    fn SDL_CreateTexture(r: *mut SDL_Renderer, format: u32, access: c_int, w: c_int, h: c_int) -> *mut SDL_Texture;
    fn SDL_DestroyTexture(t: *mut SDL_Texture);
    fn SDL_UpdateTexture(t: *mut SDL_Texture, rect: *const c_void, pixels: *const c_void, pitch: c_int) -> c_int;
    fn SDL_RenderClear(r: *mut SDL_Renderer) -> c_int;
    fn SDL_RenderCopy(r: *mut SDL_Renderer, t: *mut SDL_Texture, src: *const c_void, dst: *const c_void) -> c_int;
    fn SDL_RenderPresent(r: *mut SDL_Renderer);
    fn SDL_GetRendererInfo(r: *mut SDL_Renderer, info: *mut SDL_RendererInfo) -> c_int;
    fn SDL_GetWindowSize(w: *mut SDL_Window, cw: *mut c_int, ch: *mut c_int);
    fn SDL_GetCurrentVideoDriver() -> *const std::ffi::c_char;
    fn SDL_GetWindowFlags(w: *mut SDL_Window) -> u32;
    fn SDL_ShowWindow(w: *mut SDL_Window);
    fn SDL_RaiseWindow(w: *mut SDL_Window);
    fn SDL_PollEvent(ev: *mut SDL_Event) -> c_int;
    fn SDL_SetRelativeMouseMode(enabled: c_int) -> c_int;
    fn SDL_RenderReadPixels(r: *mut SDL_Renderer, rect: *const c_void, format: u32, pixels: *mut c_void, pitch: c_int) -> c_int;
}

/// The library names to try, most specific first: the runtime soname a
/// distribution or SDL's own installer ships, then the development name.
/// Homebrew installs outside the macOS loader's default search path, so on
/// macOS Homebrew is asked where its sdl2 lives (`brew --prefix sdl2`).
fn candidates() -> Vec<String> {
    let names: &[&str] = if cfg!(target_os = "windows") {
        &["SDL2.dll"]
    } else if cfg!(target_os = "macos") {
        &["libSDL2-2.0.0.dylib", "libSDL2.dylib"]
    } else {
        &["libSDL2-2.0.so.0", "libSDL2.so"]
    };
    let mut out: Vec<String> = names.iter().map(|n| n.to_string()).collect();
    if cfg!(target_os = "macos") {
        let prefix = std::process::Command::new("brew").args(["--prefix", "sdl2"]).output().ok().filter(|o| o.status.success());
        if let Some(prefix) = prefix.and_then(|o| String::from_utf8(o.stdout).ok()).map(|p| p.trim().to_string()).filter(|p| !p.is_empty()) {
            out.push(format!("{prefix}/lib/libSDL2-2.0.0.dylib"));
        }
    }
    out
}

static API: std::sync::OnceLock<Result<Api, String>> = std::sync::OnceLock::new();

/// Load SDL2, once per process. A run that never opens a window never calls
/// this, so `brain` neither links SDL2 nor needs it installed to run headless.
pub fn load() -> Result<(), String> {
    API.get_or_init(|| {
        let mut tried = Vec::new();
        for path in candidates() {
            // SAFETY: loading SDL2 runs its initialisers, which only set up
            // the library's own state.
            match unsafe { libloading::Library::new(&path) } {
                Ok(lib) => return Api::bind(lib, &path),
                Err(e) => tried.push(format!("{path}: {e}")),
            }
        }
        Err(format!(
            "SDL2 is not installed, and the window needs it (tried {}). Install it with your package manager: \
             `apt install libsdl2-2.0-0` on Debian/Ubuntu, `brew install sdl2` on macOS, or put SDL2.dll beside brain.exe on Windows.",
            tried.join("; ")
        ))
    })
    .as_ref()
    .map(|_| ())
    .map_err(Clone::clone)
}

fn api() -> &'static Api {
    match API.get() {
        Some(Ok(api)) => api,
        _ => panic!("wm-display: an SDL function was called before sys::load() succeeded"),
    }
}
