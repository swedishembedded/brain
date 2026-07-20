// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Minimal hand-rolled SDL2 FFI — exactly the calls the world-model window
//! needs, linked against the system libSDL2 (see build.rs). No sdl2/sdl2-sys
//! crate: brain hand-rolls small FFI surfaces (cf. the vendored ONNX protobuf)
//! and needs ~15 functions.
//!
//! ABI notes: SDL2's `SDL_Event` is a 56-byte C union; we reserve 64 bytes and
//! read fields at their documented offsets (type @0; key events: keycode/`sym`
//! @20, `repeat` @13). Pixel format RGB24 matches the engine's interleaved
//! RGB8 frames byte-for-byte.

#![allow(non_camel_case_types, dead_code)]

use std::os::raw::{c_char, c_int, c_void};

pub type SDL_Window = c_void;
pub type SDL_Renderer = c_void;
pub type SDL_Texture = c_void;

pub const SDL_INIT_VIDEO: u32 = 0x0000_0020;
pub const SDL_WINDOWPOS_CENTERED: c_int = 0x2FFF_0000u32 as c_int;
pub const SDL_WINDOW_SHOWN: u32 = 0x0000_0004;
pub const SDL_RENDERER_SOFTWARE: u32 = 0x0000_0001;
// SDL_PIXELFORMAT_RGB24: tightly packed interleaved R,G,B bytes.
// SDL_DEFINE_PIXELFORMAT(ARRAYU8=7, ARRAYORDER_RGB=1, layout 0, 24 bits,
// 3 bytes) = 0x17101803 — printed from SDL2/SDL_pixels.h by a C program,
// NOT recomputed by hand (a hand-derived value shipped wrong once and
// garbled the whole window; tests/sdl_roundtrip.rs now guards this).
pub const SDL_PIXELFORMAT_RGB24: u32 = 0x1710_1803;
pub const SDL_TEXTUREACCESS_STREAMING: c_int = 1;

// Event types.
pub const SDL_QUIT: u32 = 0x100;
pub const SDL_KEYDOWN: u32 = 0x300;
pub const SDL_KEYUP: u32 = 0x301;
pub const SDL_MOUSEMOTION: u32 = 0x400;

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
}

extern "C" {
    pub fn SDL_Init(flags: u32) -> c_int;
    pub fn SDL_Quit();
    pub fn SDL_GetError() -> *const c_char;
    pub fn SDL_SetHint(name: *const c_char, value: *const c_char) -> c_int;
    pub fn SDL_CreateWindow(
        title: *const c_char,
        x: c_int,
        y: c_int,
        w: c_int,
        h: c_int,
        flags: u32,
    ) -> *mut SDL_Window;
    pub fn SDL_DestroyWindow(win: *mut SDL_Window);
    pub fn SDL_SetWindowTitle(win: *mut SDL_Window, title: *const c_char);
    pub fn SDL_CreateRenderer(win: *mut SDL_Window, index: c_int, flags: u32)
        -> *mut SDL_Renderer;
    pub fn SDL_DestroyRenderer(r: *mut SDL_Renderer);
    pub fn SDL_CreateTexture(
        r: *mut SDL_Renderer,
        format: u32,
        access: c_int,
        w: c_int,
        h: c_int,
    ) -> *mut SDL_Texture;
    pub fn SDL_DestroyTexture(t: *mut SDL_Texture);
    pub fn SDL_UpdateTexture(
        t: *mut SDL_Texture,
        rect: *const c_void,
        pixels: *const c_void,
        pitch: c_int,
    ) -> c_int;
    pub fn SDL_RenderClear(r: *mut SDL_Renderer) -> c_int;
    pub fn SDL_RenderCopy(
        r: *mut SDL_Renderer,
        t: *mut SDL_Texture,
        src: *const c_void,
        dst: *const c_void,
    ) -> c_int;
    pub fn SDL_RenderPresent(r: *mut SDL_Renderer);
    pub fn SDL_PollEvent(ev: *mut SDL_Event) -> c_int;
    pub fn SDL_SetRelativeMouseMode(enabled: c_int) -> c_int;
    pub fn SDL_RenderReadPixels(
        r: *mut SDL_Renderer,
        rect: *const c_void,
        format: u32,
        pixels: *mut c_void,
        pitch: c_int,
    ) -> c_int;
}
