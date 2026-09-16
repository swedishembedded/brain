// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A headless OpenGL context, so rendering needs no X server.
//!
//! ## Why this crate has to load OpenGL at all
//!
//! `libmujoco.so` declares its `gl*` symbols undefined and links against no GL
//! library of its own - by design, so that an application picks the windowing
//! stack. Because MuJoCo is itself `dlopen`ed here with lazy binding, those
//! symbols stay unresolved until the first `mjr_*` call, and they resolve
//! against the process's GLOBAL scope. So the only thing needed to satisfy
//! them is to `dlopen` libGL with `RTLD_GLOBAL` before the first frame, which
//! is what [`EglContext::new`] does. Loading it `RTLD_LOCAL` would leave
//! MuJoCo's symbols unresolvable and abort the process at the first render.
//!
//! ## Why EGL and not GLX
//!
//! EGL's device platform creates a context bound to a GPU rather than to a
//! display server, which is the difference between a renderer that works over
//! SSH, in a container and in CI, and one that works only at a logged-in seat.
//! A display-bound path is kept as the fallback for the case where the EGL
//! device enumeration extension is missing.
//!
//! ## Why the context is created once and never destroyed
//!
//! MuJoCo resolves its OpenGL entry points on first use and keeps them, so a
//! context per renderer buys nothing and costs a driver-side setup each time.
//! There is therefore exactly ONE context per process, built on first use and
//! kept for the process's lifetime; a renderer makes it current on its own
//! thread and releases it on drop, which is how a renderer built on a second
//! thread can use it after the first has finished. A GL context held for the
//! life of a process is the ordinary arrangement for a renderer.

use std::ffi::{c_int, c_void};

use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};

pub(crate) const MJFB_OFFSCREEN: c_int = 1;
pub(crate) const MJCAT_ALL: c_int = 7;

// EGL enumerants, from egl.h. Written out because the alternative is a
// build-time dependency on EGL headers, which would make a headless renderer a
// compile-time requirement for a crate whose whole point is that its native
// dependencies are optional.
const EGL_NONE: i32 = 0x3038;
const EGL_RED_SIZE: i32 = 0x3024;
const EGL_GREEN_SIZE: i32 = 0x3023;
const EGL_BLUE_SIZE: i32 = 0x3022;
const EGL_ALPHA_SIZE: i32 = 0x3021;
const EGL_DEPTH_SIZE: i32 = 0x3025;
const EGL_STENCIL_SIZE: i32 = 0x3026;
const EGL_SURFACE_TYPE: i32 = 0x3033;
const EGL_PBUFFER_BIT: i32 = 0x0001;
const EGL_RENDERABLE_TYPE: i32 = 0x3040;
const EGL_OPENGL_BIT: i32 = 0x0008;
const EGL_WIDTH: i32 = 0x3057;
const EGL_HEIGHT: i32 = 0x3056;
const EGL_OPENGL_API: u32 = 0x30A2;
const EGL_PLATFORM_DEVICE_EXT: u32 = 0x313F;

type Display = *mut c_void;
type Config = *mut c_void;
type Surface = *mut c_void;
type Context = *mut c_void;
type Device = *mut c_void;

struct Egl {
    _egl: Library,
    _gl: Library,
    get_display: unsafe extern "C" fn(*mut c_void) -> Display,
    get_platform_display: Option<unsafe extern "C" fn(u32, *mut c_void, *const i32) -> Display>,
    query_devices: Option<unsafe extern "C" fn(i32, *mut Device, *mut i32) -> u32>,
    initialize: unsafe extern "C" fn(Display, *mut i32, *mut i32) -> u32,
    terminate: unsafe extern "C" fn(Display) -> u32,
    choose_config: unsafe extern "C" fn(Display, *const i32, *mut Config, i32, *mut i32) -> u32,
    bind_api: unsafe extern "C" fn(u32) -> u32,
    create_pbuffer: unsafe extern "C" fn(Display, Config, *const i32) -> Surface,
    destroy_surface: unsafe extern "C" fn(Display, Surface) -> u32,
    create_context: unsafe extern "C" fn(Display, Config, Context, *const i32) -> Context,
    destroy_context: unsafe extern "C" fn(Display, Context) -> u32,
    make_current: unsafe extern "C" fn(Display, Surface, Surface, Context) -> u32,
    get_error: unsafe extern "C" fn() -> i32,
    gl_get_error: unsafe extern "C" fn() -> u32,
}

/// Load libEGL and libGL into the process's GLOBAL scope.
fn open_gl_stack() -> Result<(Library, Library), String> {
    let flags = RTLD_NOW | RTLD_GLOBAL;
    // SAFETY: loading a shared library runs its initialisers; both of these are
    // ordinary system graphics libraries. RTLD_GLOBAL is required, not merely
    // convenient - see the module documentation.
    let egl = unsafe { Library::open(Some("libEGL.so.1"), flags) }
        .map_err(|e| format!("libEGL.so.1: {e}. Offscreen rendering needs an EGL implementation installed."))?;
    let gl = unsafe { Library::open(Some("libGL.so.1"), flags) }
        .map_err(|e| format!("libGL.so.1: {e}. MuJoCo's renderer resolves its gl* symbols against this."))?;
    Ok((egl, gl))
}

impl Egl {
    fn load() -> Result<Egl, String> {
        let (egl, gl) = open_gl_stack()?;
        macro_rules! sym {
            ($name:literal, $ty:ty) => {{
                let s: libloading::os::unix::Symbol<$ty> = unsafe { egl.get($name) }
                    .map_err(|e| format!("libEGL.so.1: missing {}: {e}", String::from_utf8_lossy($name)))?;
                *s
            }};
        }
        // Extension entry points are NOT exported symbols of libEGL under
        // libglvnd - they are reachable only through eglGetProcAddress, which
        // is the whole reason a plain dlsym for them comes back empty and the
        // headless device path silently degrades to the display-bound one.
        let get_proc: unsafe extern "C" fn(*const u8) -> *mut c_void = {
            let s: libloading::os::unix::Symbol<unsafe extern "C" fn(*const u8) -> *mut c_void> =
                unsafe { egl.get(b"eglGetProcAddress\0") }.map_err(|e| format!("libEGL.so.1: missing eglGetProcAddress: {e}"))?;
            *s
        };
        macro_rules! ext {
            ($name:literal, $ty:ty) => {{
                // SAFETY: eglGetProcAddress returns either null or a pointer to
                // the entry point named by this NUL-terminated string, whose
                // signature is fixed by the EGL extension specification.
                let p = unsafe { get_proc($name.as_ptr()) };
                if p.is_null() {
                    None
                } else {
                    Some(unsafe { std::mem::transmute::<*mut c_void, $ty>(p) })
                }
            }};
        }
        Ok(Egl {
            get_display: sym!(b"eglGetDisplay\0", unsafe extern "C" fn(*mut c_void) -> Display),
            get_platform_display: ext!(b"eglGetPlatformDisplayEXT\0", unsafe extern "C" fn(u32, *mut c_void, *const i32) -> Display),
            query_devices: ext!(b"eglQueryDevicesEXT\0", unsafe extern "C" fn(i32, *mut Device, *mut i32) -> u32),
            initialize: sym!(b"eglInitialize\0", unsafe extern "C" fn(Display, *mut i32, *mut i32) -> u32),
            terminate: sym!(b"eglTerminate\0", unsafe extern "C" fn(Display) -> u32),
            choose_config: sym!(b"eglChooseConfig\0", unsafe extern "C" fn(Display, *const i32, *mut Config, i32, *mut i32) -> u32),
            bind_api: sym!(b"eglBindAPI\0", unsafe extern "C" fn(u32) -> u32),
            create_pbuffer: sym!(b"eglCreatePbufferSurface\0", unsafe extern "C" fn(Display, Config, *const i32) -> Surface),
            destroy_surface: sym!(b"eglDestroySurface\0", unsafe extern "C" fn(Display, Surface) -> u32),
            create_context: sym!(b"eglCreateContext\0", unsafe extern "C" fn(Display, Config, Context, *const i32) -> Context),
            destroy_context: sym!(b"eglDestroyContext\0", unsafe extern "C" fn(Display, Context) -> u32),
            make_current: sym!(b"eglMakeCurrent\0", unsafe extern "C" fn(Display, Surface, Surface, Context) -> u32),
            get_error: sym!(b"eglGetError\0", unsafe extern "C" fn() -> i32),
            gl_get_error: {
                let s: libloading::os::unix::Symbol<unsafe extern "C" fn() -> u32> =
                    unsafe { gl.get(b"glGetError\0") }.map_err(|e| format!("libGL.so.1: missing glGetError: {e}"))?;
                *s
            },
            _egl: egl,
            _gl: gl,
        })
    }

    /// Every display worth trying, GPU devices first and the default last.
    fn displays(&self) -> Vec<(String, Display)> {
        let mut out = Vec::new();
        if let (Some(query), Some(platform)) = (self.query_devices, self.get_platform_display) {
            let mut devices = [std::ptr::null_mut(); 16];
            let mut n = 0i32;
            // SAFETY: the array holds 16 entries and 16 is what is passed as the
            // capacity; EGL writes at most that many and reports how many.
            if unsafe { query(devices.len() as i32, devices.as_mut_ptr(), &mut n) } != 0 {
                for (i, d) in devices.iter().take(n.max(0) as usize).enumerate() {
                    // SAFETY: `d` came from eglQueryDevicesEXT above.
                    let dpy = unsafe { platform(EGL_PLATFORM_DEVICE_EXT, *d, std::ptr::null()) };
                    if !dpy.is_null() {
                        out.push((format!("EGL device {i}"), dpy));
                    }
                }
            }
        }
        // SAFETY: EGL_DEFAULT_DISPLAY is the null native display handle.
        let dpy = unsafe { (self.get_display)(std::ptr::null_mut()) };
        if !dpy.is_null() {
            out.push(("EGL default display".to_string(), dpy));
        }
        out
    }
}

/// The process's headless OpenGL context.
///
/// Built once by [`EglContext::get`] and never destroyed. Made current on a
/// thread by [`EglContext::make_current`] and released by
/// [`EglContext::release`], which is how a renderer on a second thread can use
/// it after a renderer on the first has finished.
pub struct EglContext {
    egl: Egl,
    display: Display,
    surface: Surface,
    context: Context,
}

// SAFETY: the three handles are immutable after construction, and EGL permits
// a context to be made current on any thread provided it is current on no
// other. That exclusion is enforced above this layer: at most one `Renderer`
// exists per process, and it releases the context when it is dropped.
unsafe impl Send for EglContext {}
unsafe impl Sync for EglContext {}

static CONTEXT: std::sync::OnceLock<Result<EglContext, String>> = std::sync::OnceLock::new();

impl EglContext {
    /// The process's context, built on first call.
    ///
    /// The failure is cached along with the success: a box with no GPU should
    /// report the same clear reason every time rather than re-probing EGL on
    /// every frame loop that asks.
    pub fn get() -> Result<&'static EglContext, String> {
        match CONTEXT.get_or_init(EglContext::create) {
            Ok(c) => Ok(c),
            Err(e) => Err(e.clone()),
        }
    }

    fn create() -> Result<EglContext, String> {
        let egl = Egl::load()?;
        let mut tried = Vec::new();
        for (what, display) in egl.displays() {
            // The pbuffer is NOT what MuJoCo renders into - it renders into its
            // own framebuffer object, sized by `mjr_resizeOffscreen`. EGL just
            // needs some drawable to make a context current on this path, so
            // the smallest legal one is the right size for it.
            match EglContext::on(&egl, display, 16, 16) {
                Ok((surface, context)) => return Ok(EglContext { egl, display, surface, context }),
                Err(e) => tried.push(format!("{what}: {e}")),
            }
        }
        Err(format!(
            "no usable EGL display for offscreen rendering. Tried: {}",
            if tried.is_empty() { "none, EGL reported no displays at all".to_string() } else { tried.join("; ") }
        ))
    }

    /// Bind this context to the calling thread.
    pub fn make_current(&self) -> Result<(), String> {
        // SAFETY: all three handles were produced by `create` and outlive the
        // process; EGL allows this exactly while the context is current on no
        // other thread, which the caller guarantees.
        if unsafe { (self.egl.make_current)(self.display, self.surface, self.surface, self.context) } == 0 {
            return Err(format!(
                "eglMakeCurrent failed (EGL error {:#x}). Another thread may still hold the context.",
                unsafe { (self.egl.get_error)() }
            ));
        }
        Ok(())
    }

    /// Unbind this context from the calling thread, so another may take it.
    pub fn release(&self) {
        // SAFETY: unbinding takes only the display handle and EGL's documented
        // "no object" nulls; it is valid whether or not anything is current.
        unsafe {
            (self.egl.make_current)(self.display, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
        }
    }

    fn on(egl: &Egl, display: Display, width: u32, height: u32) -> Result<(Surface, Context), String> {
        // SAFETY (whole block): every pointer is either null - EGL's documented
        // "no object" / "no attributes" value - or addresses a live local whose
        // lifetime spans the call, and each attribute list is EGL_NONE
        // terminated as the API requires. Every failure path releases exactly
        // what it created, innermost first.
        unsafe {
            let fail = |what: &str| format!("{what} failed (EGL error {:#x})", (egl.get_error)());
            if (egl.initialize)(display, std::ptr::null_mut(), std::ptr::null_mut()) == 0 {
                return Err(fail("eglInitialize"));
            }
            // MuJoCo's renderer is desktop OpenGL, not GLES. Binding the wrong
            // API yields a context whose creation succeeds and whose rendering
            // is wrong, which is the expensive way to find out.
            if (egl.bind_api)(EGL_OPENGL_API) == 0 {
                (egl.terminate)(display);
                return Err(fail("eglBindAPI(EGL_OPENGL_API)"));
            }
            let attrs = [
                EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
                EGL_DEPTH_SIZE, 24, EGL_STENCIL_SIZE, 8,
                EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                EGL_NONE,
            ];
            let mut config: Config = std::ptr::null_mut();
            let mut n = 0i32;
            if (egl.choose_config)(display, attrs.as_ptr(), &mut config, 1, &mut n) == 0 || n < 1 {
                (egl.terminate)(display);
                return Err(fail("eglChooseConfig (no RGBA8 + depth24 pbuffer config)"));
            }
            let pb = [EGL_WIDTH, width as i32, EGL_HEIGHT, height as i32, EGL_NONE];
            let surface = (egl.create_pbuffer)(display, config, pb.as_ptr());
            if surface.is_null() {
                (egl.terminate)(display);
                return Err(fail("eglCreatePbufferSurface"));
            }
            let context = (egl.create_context)(display, config, std::ptr::null_mut(), std::ptr::null());
            if context.is_null() {
                (egl.destroy_surface)(display, surface);
                (egl.terminate)(display);
                return Err(fail("eglCreateContext"));
            }
            if (egl.make_current)(display, surface, surface, context) == 0 {
                (egl.destroy_context)(display, context);
                (egl.destroy_surface)(display, surface);
                (egl.terminate)(display);
                return Err(fail("eglMakeCurrent"));
            }
            Ok((surface, context))
        }
    }
}

impl EglContext {
    /// Empty OpenGL's error queue, returning how many errors were pending.
    ///
    /// GL reports failure by queueing rather than by returning, so an error
    /// raised during context creation is still pending when the next library
    /// looks. MuJoCo looks, and reports what it finds as "in or before
    /// `mjr_makeContext`" - an ambiguity that makes the warning unactionable.
    /// Draining here removes the "or before": whatever MuJoCo then reports is
    /// MuJoCo's own.
    ///
    /// Measured on an EGL pbuffer context: this finds nothing to drain, and
    /// MuJoCo still reports `GL_INVALID_OPERATION`, so that error is raised
    /// inside its context creation. It is not inherited from here, and the
    /// rendered output is verified correct by `tests/render.rs` regardless.
    pub(crate) fn drain_gl_errors(&self) -> usize {
        let mut n = 0;
        // SAFETY: the context is current on this thread, and glGetError takes
        // no arguments and returns GL_NO_ERROR once the queue is empty. The
        // bound is belt and braces against a driver that never clears.
        while n < 64 && unsafe { (self.egl.gl_get_error)() } != 0 {
            n += 1;
        }
        n
    }
}
