// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The raw symbols, and where the library is looked for.

use std::ffi::{c_char, c_int, c_void};

/// MuJoCo's object-type enum, for `mj_name2id` / `mj_id2name`.
///
/// Values are positional in `mjtObj` and only the ones this crate uses are
/// listed. They are written out rather than counted from a partial list
/// because an off-by-one here does not fail: it looks up a name in the wrong
/// table and returns `None`, which reads as "no such actuator".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum ObjType {
    Body = 1,
    Joint = 3,
    Geom = 5,
    Site = 6,
    Actuator = 19,
}

/// MuJoCo's `mjtState` bit-spec, for the flat state API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
#[allow(clippy::upper_case_acronyms)]
pub enum StateSpec {
    TIME = 1,
    QPOS = 1 << 1,
    QVEL = 1 << 2,
    CTRL = 1 << 6,
}

type LoadXml = unsafe extern "C" fn(*const c_char, *const c_void, *mut c_char, c_int) -> *mut c_void;
type ModelFn = unsafe extern "C" fn(*mut c_void);
type MakeData = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
type StepFn = unsafe extern "C" fn(*mut c_void, *mut c_void);
type StateSize = unsafe extern "C" fn(*mut c_void, c_int) -> c_int;
type GetState = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut f64, c_int);
type SetState = unsafe extern "C" fn(*mut c_void, *mut c_void, *const f64, c_int);
type Name2Id = unsafe extern "C" fn(*mut c_void, c_int, *const c_char) -> c_int;
type Id2Name = unsafe extern "C" fn(*mut c_void, c_int, c_int) -> *const c_char;

/// The dlopened library plus the entry points, resolved once.
pub struct Lib {
    // Dropping this unloads the library, so it must outlive every pointer
    // obtained through it. Held, never read.
    _lib: libloading::Library,
    pub version: unsafe extern "C" fn() -> c_int,
    pub load_xml: LoadXml,
    pub delete_model: ModelFn,
    pub make_data: MakeData,
    pub delete_data: ModelFn,
    pub step: StepFn,
    pub forward: StepFn,
    pub reset: StepFn,
    pub state_size: StateSize,
    pub get_state: GetState,
    pub set_state: SetState,
    pub name2id: Name2Id,
    pub id2name: Id2Name,
}

/// Candidate paths, most specific first. A `*_DIR` variable names a MuJoCo
/// INSTALL root (the directory holding `lib/` and `include/`), matching how
/// MuJoCo's own distribution is laid out and what its other bindings expect.
fn candidates() -> Vec<String> {
    let mut out = Vec::new();
    for var in ["BRAIN_MUJOCO_DIR", "MUJOCO_DIR", "MUJOCO_PATH"] {
        if let Ok(root) = std::env::var(var) {
            if !root.is_empty() {
                out.push(format!("{root}/lib/libmujoco.so"));
                out.push(format!("{root}/libmujoco.so"));
            }
        }
    }
    // Bare soname last, so the dynamic loader's own search path applies for a
    // system-installed MuJoCo.
    out.push("libmujoco.so".to_string());
    out
}

impl Lib {
    pub fn open() -> Result<Lib, String> {
        let tried = candidates();
        let mut last = String::new();
        for path in &tried {
            // SAFETY: loading a shared library runs its initialisers. MuJoCo
            // is a well-behaved C library with no constructors of consequence.
            match unsafe { libloading::Library::new(path) } {
                Ok(lib) => return Self::bind(lib, path),
                Err(e) => last = format!("{path}: {e}"),
            }
        }
        Err(format!(
            "MuJoCo not found. Tried: {}. Last error: {last}. \
             Install MuJoCo and point $BRAIN_MUJOCO_DIR at the directory holding lib/libmujoco.so.",
            tried.join(", ")
        ))
    }

    fn bind(lib: libloading::Library, path: &str) -> Result<Lib, String> {
        // A `Symbol` borrows from `lib`; dereferencing copies the bare
        // function pointer out. That pointer stays valid exactly as long as
        // the library is loaded, which is why `lib` is moved into the struct
        // below and never dropped earlier.
        macro_rules! sym {
            ($name:literal, $ty:ty) => {{
                let s: libloading::Symbol<$ty> = unsafe { lib.get($name) }
                    .map_err(|e| format!("{path}: missing symbol {}: {e}", String::from_utf8_lossy($name)))?;
                *s
            }};
        }
        Ok(Lib {
            version: sym!(b"mj_version\0", unsafe extern "C" fn() -> c_int),
            load_xml: sym!(b"mj_loadXML\0", LoadXml),
            delete_model: sym!(b"mj_deleteModel\0", ModelFn),
            make_data: sym!(b"mj_makeData\0", MakeData),
            delete_data: sym!(b"mj_deleteData\0", ModelFn),
            step: sym!(b"mj_step\0", StepFn),
            forward: sym!(b"mj_forward\0", StepFn),
            reset: sym!(b"mj_resetData\0", StepFn),
            state_size: sym!(b"mj_stateSize\0", StateSize),
            get_state: sym!(b"mj_getState\0", GetState),
            set_state: sym!(b"mj_setState\0", SetState),
            name2id: sym!(b"mj_name2id\0", Name2Id),
            id2name: sym!(b"mj_id2name\0", Id2Name),
            _lib: lib,
        })
    }
}
