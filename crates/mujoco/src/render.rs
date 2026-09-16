// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Rendering a model to a pixel buffer.
//!
//! ## Why the visual structs are opaque blobs
//!
//! MuJoCo's renderer takes four caller-allocated structs - `mjvScene`,
//! `mjvCamera`, `mjvOption` and `mjrContext` - whose layouts depend on
//! compile-time maxima (`mjMAXLIGHT` alone is 100 lights inline) and on
//! several nested struct definitions. Mirroring them would reintroduce exactly
//! the risk this binding avoids for `mjData`: a field read at the wrong offset
//! produces a plausible value rather than a crash.
//!
//! It turns out none of their fields need to be read or written. MuJoCo
//! provides an initialiser for each and `mjv_defaultFreeCamera`, which frames
//! the model sensibly without any field being touched. So each struct here is
//! an over-allocated, zeroed byte buffer that only ever travels back into
//! MuJoCo, and the ONLY property this binding depends on is that the buffer is
//! at least as large as the real struct.
//!
//! That property is checked rather than assumed: every buffer carries a canary
//! pattern in its tail and [`Renderer::new`] verifies the canary is intact
//! after MuJoCo has initialised the struct, so an undersized buffer fails
//! loudly at construction instead of corrupting the heap. It is the same
//! discipline as [`crate::Model`]'s `mj_stateSize` cross-check.
//!
//! Swedish Embedded AB implements headless rendering pipelines for clients who
//! need simulation output on a server rather than at a desk. If your team needs
//! GPU rendering that works over SSH and in CI, you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::ffi::{c_int, c_void};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::sys::Rect;
use crate::{Data, Model, MuJoCo};

mod egl;
pub use egl::EglContext;

/// Bytes allocated for each visual struct.
///
/// `mjvScene`'s largest inline member is an array of 100 `mjvLight`, and
/// `mjrContext` is of comparable size, so the real structs are tens of
/// kilobytes. A megabyte is far beyond either and costs nothing next to the
/// framebuffers; the canary below is what turns "far beyond" from a belief
/// into a checked fact.
const BLOB: usize = 1 << 20;
const CANARY: u8 = 0xA5;
/// Bytes of canary at the tail of each blob.
const CANARY_LEN: usize = 4096;

/// A zeroed, over-allocated buffer standing in for a MuJoCo visual struct.
struct Blob {
    bytes: Vec<u8>,
}

impl Blob {
    fn new() -> Blob {
        let mut bytes = vec![0u8; BLOB];
        bytes[BLOB - CANARY_LEN..].fill(CANARY);
        Blob { bytes }
    }

    fn ptr(&mut self) -> *mut c_void {
        self.bytes.as_mut_ptr() as *mut c_void
    }

    /// Whether MuJoCo stayed inside the space this buffer assumed it needed.
    fn intact(&self) -> bool {
        self.bytes[BLOB - CANARY_LEN..].iter().all(|&b| b == CANARY)
    }
}

/// Where `mjvCamera`'s fields sit, in bytes from the start of the struct.
///
/// This is the ONE MuJoCo struct this binding reads and writes by offset
/// rather than passing back untouched, and it earns the exception: an
/// interactive camera has to be pointed at a moving animal every frame, and
/// the only alternative the C API offers - `mjv_moveCamera` gestures - is a
/// relative nudge whose effect depends on state this binding would then have
/// to shadow and keep in sync. It was shadowed that way, with two fitted
/// constants converting a pan into a distance, and it drifted the moment
/// anything else moved the camera.
///
/// The layout is from `mujoco/mjvisualize.h` at the pinned version, with
/// `mjtNum` as `double`: three `int`s, four bytes of padding to align the
/// doubles, then `lookat[3]`, `distance`, `azimuth`, `elevation`.
/// [`Renderer::check_camera_layout`] verifies it against a freshly initialised
/// camera rather than trusting this comment.
mod cam {
    pub const TYPE: usize = 0;
    pub const FIXEDCAMID: usize = 4;
    pub const TRACKBODYID: usize = 8;
    pub const LOOKAT: usize = 16;
    pub const DISTANCE: usize = 40;
    pub const AZIMUTH: usize = 48;
    pub const ELEVATION: usize = 56;
    /// `mjCAMERA_FREE`.
    pub const FREE: i32 = 0;
}

/// The abstract camera's pose: where it looks, from how far, and from where.
///
/// Angles in degrees, distances in the model's own length units, exactly as
/// `mjvCamera` stores them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraPose {
    pub lookat: [f64; 3],
    pub distance: f64,
    pub azimuth: f64,
    pub elevation: f64,
}

/// Whether a [`Renderer`] is live anywhere in this process.
///
/// There is one GL context per process (see [`egl`]), EGL allows it to be
/// current on at most one thread at a time, and a renderer holds it current
/// for its whole lifetime. A second renderer would therefore either steal the
/// context from the first or fail to acquire it, depending on which thread it
/// was built on - so it is refused with a message that says why instead.
static LIVE: AtomicBool = AtomicBool::new(false);

/// Claims [`LIVE`] for as long as it exists.
struct Exclusive;

impl Exclusive {
    fn claim() -> Result<Exclusive, String> {
        match LIVE.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => Ok(Exclusive),
            Err(_) => Err("a Renderer is already live in this process, and it holds the \
                           process's only GL context current. Drop the first one before \
                           creating a second."
                .to_string()),
        }
    }
}

impl Drop for Exclusive {
    fn drop(&mut self) {
        LIVE.store(false, Ordering::Release);
    }
}

/// An offscreen renderer for one model.
///
/// At most one may be live per PROCESS - see [`LIVE`]. Creating a second while
/// the first is alive is an error rather than undefined rendering.
pub struct Renderer {
    mj: Arc<MuJoCo>,
    /// The process's context, current on this renderer's thread for as long as
    /// it lives. Released in [`Renderer::drop`], never destroyed.
    egl: &'static EglContext,
    // Released last, so the next renderer cannot start until this one has
    // finished freeing its GL resources and released the context.
    _exclusive: Exclusive,
    scene: Blob,
    camera: Blob,
    option: Blob,
    context: Blob,
    width: u32,
    height: u32,
    rgb: Vec<u8>,
    /// GL errors seen while drawing, so the complaint is made once rather than
    /// thirty times a second.
    gl_errors_reported: usize,
}

// Every pointer this holds is borrowed from MuJoCo and only ever handed back
// to it; the renderer is already single-threaded by construction.
unsafe impl Send for Renderer {}

impl Renderer {
    /// Build an offscreen renderer at `width` x `height`.
    ///
    /// `max_geom` bounds the scene's geometry buffer. A model whose scene
    /// exceeds it renders a truncated scene rather than failing, so it is
    /// sized generously by [`Renderer::for_model`].
    pub fn new(mj: &Arc<MuJoCo>, model: &Model, width: u32, height: u32, max_geom: u32) -> Result<Renderer, String> {
        if width == 0 || height == 0 {
            return Err("a renderer needs a non-empty viewport".to_string());
        }
        // Claimed first, so every failure path below releases it on unwind.
        let exclusive = Exclusive::claim()?;
        let egl = EglContext::get()?;
        egl.make_current()?;
        egl.drain_gl_errors();
        let r = mj
            .lib()
            .render
            .as_ref()
            .ok_or("this MuJoCo build exposes no GL renderer (no mjr_* symbols)")?;

        let mut scene = Blob::new();
        let mut camera = Blob::new();
        let mut option = Blob::new();
        let mut context = Blob::new();

        // SAFETY: each pointer addresses BLOB bytes of zeroed, writable memory,
        // which is more than the struct MuJoCo initialises. "More than" is a
        // checked claim, not an assumption - see the canary loop below. The GL
        // context created above is current on this thread, which is what
        // mjr_makeContext requires.
        unsafe {
            (r.default_scene)(scene.ptr());
            (r.default_option)(option.ptr());
            (r.default_context)(context.ptr());
            (r.default_free_camera)(model.ptr(), camera.ptr());
            (r.make_scene)(model.ptr(), scene.ptr(), max_geom as c_int);
            (r.make_context)(model.ptr(), context.ptr(), 150);
            (r.set_buffer)(egl::MJFB_OFFSCREEN, context.ptr());
            // The offscreen framebuffer is sized from the model's
            // `<visual><global offwidth/offheight>`, which defaults to 640x480
            // - so a larger viewport would silently read undefined pixels for
            // every row and column past it. Resizing here makes the buffer
            // follow the request instead of the model file, and the check
            // below confirms it took.
            (r.resize_offscreen)(width as c_int, height as c_int, context.ptr());
        }

        for (name, b) in [("mjvScene", &scene), ("mjvCamera", &camera), ("mjvOption", &option), ("mjrContext", &context)] {
            if !b.intact() {
                return Err(format!(
                    "MuJoCo {} wrote past {BLOB} bytes while initialising {name}; this binding's \
                     over-allocation is too small for this build and nothing it renders can be trusted",
                    mj.version()
                ));
            }
        }

        Renderer::check_camera_layout(&camera, mj.version())?;

        // Ask MuJoCo how big the active buffer actually is, rather than
        // trusting that the resize above succeeded. This is the same shape of
        // check as Model::validate_layout: the library's own answer, compared
        // against what this binding intends to do.
        let max = unsafe { (r.max_viewport)(context.ptr()) };
        if max.width < width as c_int || max.height < height as c_int {
            return Err(format!(
                "MuJoCo's offscreen buffer is {}x{} but {width}x{height} was requested. \
                 Rendering would read undefined pixels outside it.",
                max.width, max.height
            ));
        }

        // Handed back immediately. A GL context left current on the thread is
        // not inert: anything else that wants one - an SDL window bringing up
        // X11, for instance - has to switch the thread away from it, and the
        // EGL-to-GLX transition is refused by the X server with a BadAccess on
        // X_GLXMakeCurrent. The context is therefore taken only around the
        // calls that actually need it, which is `render` and the teardown.
        egl.release();

        Ok(Renderer {
            mj: mj.clone(),
            egl,
            _exclusive: exclusive,
            scene,
            camera,
            option,
            context,
            gl_errors_reported: 0,
            width,
            height,
            rgb: vec![0u8; (width as usize) * (height as usize) * 3],
        })
    }

    /// A renderer sized for `model`, with a geometry budget that does not
    /// truncate a mesh-heavy scene like the fly's.
    pub fn for_model(mj: &Arc<MuJoCo>, model: &Model, width: u32, height: u32) -> Result<Renderer, String> {
        Renderer::new(mj, model, width, height, 20_000)
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Render the current state, returning the frame as RGB8 in OpenGL's own
    /// bottom-up row order.
    ///
    /// See [`Self::rgb_top_down`] for the order a screen blitter wants. The
    /// flip is not done here because a caller writing a PPM or feeding a
    /// texture may not want to pay for it.
    pub fn render(&mut self, model: &Model, data: &Data) -> Result<&[u8], String> {
        let r = self.mj.lib().render.as_ref().ok_or("this MuJoCo build exposes no GL renderer")?;
        // Held for the duration of the frame and no longer - see `new`.
        self.egl.make_current()?;
        let viewport = Rect { left: 0, bottom: 0, width: self.width as c_int, height: self.height as c_int };
        // SAFETY: all four blobs were initialised by MuJoCo in `new` and are
        // only ever passed back to it; the pixel buffer is exactly w*h*3 bytes,
        // which is what mjr_readPixels writes for this viewport, and the
        // viewport was verified in `new` to fit the offscreen buffer. A null
        // perturbation and a null depth buffer are both documented as "none".
        unsafe {
            (r.update_scene)(
                model.ptr(),
                data.ptr(),
                self.option.ptr(),
                std::ptr::null_mut(),
                self.camera.ptr(),
                egl::MJCAT_ALL,
                self.scene.ptr(),
            );
            (r.render)(viewport, self.scene.ptr(), self.context.ptr());
            (r.read_pixels)(self.rgb.as_mut_ptr(), std::ptr::null_mut(), viewport, self.context.ptr());
        }
        // Whatever the frame raised, said once. A GL error during rendering
        // is how a blank or corrupt frame announces itself, and discarding it
        // leaves a black window as the only symptom.
        let errors = self.egl.drain_gl_errors();
        if errors > 0 && self.gl_errors_reported == 0 {
            eprintln!("render: OpenGL reported {errors} error(s) while drawing a frame; the image may be wrong");
        }
        self.gl_errors_reported += errors;
        self.egl.release();
        Ok(&self.rgb)
    }

    /// Confirm this binding's `mjvCamera` offsets against a camera MuJoCo has
    /// just initialised.
    ///
    /// `mjv_defaultFreeCamera` leaves a signature that is very unlikely to
    /// appear at the wrong offsets: the type is `mjCAMERA_FREE` (0) and BOTH
    /// id fields are -1, which is three specific words in a row, and the
    /// distance it frames the model at is a positive finite double. A build
    /// whose struct moved fails here, loudly, instead of pointing the camera
    /// with whatever happened to be at byte 40.
    fn check_camera_layout(camera: &Blob, version: i32) -> Result<(), String> {
        let i = |at: usize| i32::from_ne_bytes(camera.bytes[at..at + 4].try_into().unwrap_or([0; 4]));
        let f = |at: usize| f64::from_ne_bytes(camera.bytes[at..at + 8].try_into().unwrap_or([0; 8]));
        let (ty, fixed, track, dist) = (i(cam::TYPE), i(cam::FIXEDCAMID), i(cam::TRACKBODYID), f(cam::DISTANCE));
        if ty != cam::FREE || fixed != -1 || track != -1 || !dist.is_finite() || dist <= 0.0 {
            return Err(format!(
                "MuJoCo {version}'s mjvCamera is not laid out the way this binding reads it: a freshly \
                 initialised free camera should read (type 0, fixedcamid -1, trackbodyid -1, distance > 0) \
                 and reads ({ty}, {fixed}, {track}, {dist})"
            ));
        }
        Ok(())
    }

    /// Where the camera is looking, from how far, and from what angle.
    ///
    /// Read out of `mjvCamera` by offset. That is the exception this binding
    /// otherwise refuses to make, and [`Self::check_camera_layout`] is the
    /// price of making it: pointing a camera at a moving animal is an ABSOLUTE
    /// request, and `mjv_moveCamera`'s relative nudges can only serve one
    /// through a shadow copy of the state they are nudging - which this code
    /// had, with two fitted constants in it, and which drifted.
    pub fn camera(&self) -> CameraPose {
        let f = |at: usize| f64::from_ne_bytes(self.camera.bytes[at..at + 8].try_into().unwrap_or([0; 8]));
        CameraPose {
            lookat: [f(cam::LOOKAT), f(cam::LOOKAT + 8), f(cam::LOOKAT + 16)],
            distance: f(cam::DISTANCE),
            azimuth: f(cam::AZIMUTH),
            elevation: f(cam::ELEVATION),
        }
    }

    /// Point the camera. See [`CameraPose`].
    ///
    /// A non-finite field is refused rather than written: it does not crash
    /// MuJoCo, it renders an EMPTY frame, and an empty frame is
    /// indistinguishable from a window that never presented.
    pub fn set_camera(&mut self, pose: CameraPose) -> Result<(), String> {
        let all = [pose.lookat[0], pose.lookat[1], pose.lookat[2], pose.distance, pose.azimuth, pose.elevation];
        if all.iter().any(|v| !v.is_finite()) {
            return Err(format!("a camera pose has to be finite, got {pose:?}"));
        }
        let mut put = |at: usize, v: f64| self.camera.bytes[at..at + 8].copy_from_slice(&v.to_ne_bytes());
        put(cam::LOOKAT, pose.lookat[0]);
        put(cam::LOOKAT + 8, pose.lookat[1]);
        put(cam::LOOKAT + 16, pose.lookat[2]);
        put(cam::DISTANCE, pose.distance.max(f64::MIN_POSITIVE));
        put(cam::AZIMUTH, pose.azimuth);
        put(cam::ELEVATION, pose.elevation);
        Ok(())
    }

    /// The last rendered frame with rows in top-down order, which is what a
    /// screen blitter expects.
    pub fn rgb_top_down(&self) -> Vec<u8> {
        let row = self.width as usize * 3;
        let mut out = vec![0u8; self.rgb.len()];
        for y in 0..self.height as usize {
            let src = (self.height as usize - 1 - y) * row;
            out[y * row..(y + 1) * row].copy_from_slice(&self.rgb[src..src + row]);
        }
        out
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        // SAFETY: both were made by MuJoCo in `new`, are freed exactly once,
        // and are freed while the context is still current - which is the only
        // state in which deleting GL objects does anything at all.
        if let Some(r) = self.mj.lib().render.as_ref() {
            // Deleting GL objects with no context current does nothing at all,
            // silently, so the context is taken back for the teardown.
            if self.egl.make_current().is_ok() {
                unsafe {
                    (r.free_scene)(self.scene.ptr());
                    (r.free_context)(self.context.ptr());
                }
            }
        }
        // Only now may another thread take the context. `_exclusive` is
        // declared after this field, so it is released after this runs.
        self.egl.release();
    }
}
