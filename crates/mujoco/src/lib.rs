// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A minimal binding to the MuJoCo physics library.
//!
//! Hand-rolled and `dlopen`ed, which is this workspace's established shape for
//! an optional native dependency: `backend-cuda` reaches `libcuda.so.1` the
//! same way, `crates/capture` hand-rolls V4L2 ioctls, `crates/wm-display`
//! hand-rolls SDL2. The consequence that matters is that **this crate builds
//! and tests green on a machine with no MuJoCo installed** - absence is a
//! runtime skip, never a build failure, so it can be an ordinary member of the
//! default build.
//!
//! ## Why there is no `mjData` struct in here
//!
//! The obvious way to bind MuJoCo is to mirror `mjModel` and `mjData` as Rust
//! structs and read their fields. `mjData` makes that a bad trade: its layout
//! depends on `mjNISLAND`, `mjNSOLVER`, `mjNWARNING` and `mjNTIMER`, and on the
//! sizes of three stat structs, so a mirror is a version-specific guess whose
//! failure mode is reading a physics quantity from the wrong offset and
//! getting a plausible number.
//!
//! MuJoCo ships a flat state API precisely to avoid this - `mj_stateSize`,
//! `mj_getState`, `mj_setState` over a documented bit-spec - so this binding
//! uses it and mirrors no `mjData` at all. Only `mjModel`'s first three size
//! fields are read directly, and even those are **self-validating**: MuJoCo
//! must agree that `mj_stateSize(mjSTATE_QPOS) == nq`, or [`Model::from_xml`]
//! refuses to return. A wrong layout therefore fails loudly at load instead of
//! quietly during simulation.
//!
//! Swedish Embedded AB implements native library bindings for clients who need
//! an optional dependency to stay optional - loaded at run time, validated on
//! arrival, and impossible to mistake for a build-time requirement. If your
//! team needs that discipline applied to its own native stack, you can procure
//! our services by sending an email to info@swedishembedded.com.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::Path;
use std::sync::Arc;

mod sys;
pub use sys::{ObjType, StateSpec};

/// The loaded MuJoCo shared library and the entry points this crate uses.
pub struct MuJoCo {
    inner: sys::Lib,
}

impl MuJoCo {
    /// `dlopen` MuJoCo and check its version.
    ///
    /// Searched in order: `$BRAIN_MUJOCO_DIR/lib`, `$MUJOCO_DIR/lib`,
    /// `$MUJOCO_PATH/lib`, then the plain soname so the dynamic loader's own
    /// search path applies. The error names every place that was tried,
    /// because "not found" without a list is the least actionable message a
    /// native dependency can produce.
    pub fn load() -> Result<Arc<MuJoCo>, String> {
        let inner = sys::Lib::open()?;
        let v = unsafe { (inner.version)() };
        // Not pinned to one build: the layout risk this binding takes is
        // limited to three leading size fields, and `Model::from_xml`
        // validates those against MuJoCo's own answer. A major-version change
        // is a different matter and is refused.
        if !(3_000_000..4_000_000).contains(&v) {
            return Err(format!(
                "MuJoCo reports version {v}, which is not a 3.x release; this binding is written against the 3.x API"
            ));
        }
        Ok(Arc::new(MuJoCo { inner }))
    }

    /// The loaded library's version, in MuJoCo's `MMmmpp` integer form
    /// (`3012000` is 3.12.0).
    pub fn version(&self) -> i32 {
        unsafe { (self.inner.version)() }
    }
}

/// A compiled MuJoCo model. Freed on drop.
pub struct Model {
    mj: Arc<MuJoCo>,
    ptr: *mut c_void,
    nq: usize,
    nv: usize,
    nu: usize,
}

// The pointer is owned exclusively by this handle and MuJoCo's model is
// immutable once compiled; `Data` carries the mutable state.
unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    /// Compile an MJCF file.
    pub fn from_xml(mj: &Arc<MuJoCo>, path: impl AsRef<Path>) -> Result<Model, String> {
        let path = path.as_ref();
        let c = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let mut err = vec![0 as c_char; 1024];
        let ptr = unsafe { (mj.inner.load_xml)(c.as_ptr(), std::ptr::null(), err.as_mut_ptr(), err.len() as c_int) };
        if ptr.is_null() {
            let msg = unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy().trim().to_string();
            return Err(format!("{}: {}", path.display(), if msg.is_empty() { "mj_loadXML failed".into() } else { msg }));
        }
        // SAFETY: mjModel begins with a run of int64 size fields; nq, nv and
        // nu are the first three. Validated immediately below, so a layout
        // change cannot pass silently.
        let sizes = ptr as *const i64;
        let (nq, nv, nu) = unsafe { (*sizes as usize, *sizes.add(1) as usize, *sizes.add(2) as usize) };

        let model = Model { mj: mj.clone(), ptr, nq, nv, nu };
        model.validate_layout()?;
        Ok(model)
    }

    /// Cross-check the three fields this binding reads directly against
    /// MuJoCo's own answer for the same quantities.
    ///
    /// This is the whole reason the binding can decline to pin a version. If
    /// `mjModel`'s leading layout ever changes, `nq` read from the struct and
    /// `mj_stateSize(mjSTATE_QPOS)` computed by the library stop agreeing, and
    /// this fails at load with a message that says so.
    fn validate_layout(&self) -> Result<(), String> {
        for (what, ours, spec) in [
            ("nq", self.nq, StateSpec::QPOS),
            ("nv", self.nv, StateSpec::QVEL),
            ("nu", self.nu, StateSpec::CTRL),
        ] {
            let theirs = unsafe { (self.mj.inner.state_size)(self.ptr, spec as c_int) } as usize;
            if ours != theirs {
                return Err(format!(
                    "MuJoCo {} mjModel layout mismatch: this binding read {what} = {ours}, \
                     but mj_stateSize says {theirs}. The struct layout changed; do not trust \
                     any physics from this build.",
                    self.mj.version()
                ));
            }
        }
        Ok(())
    }

    pub fn nq(&self) -> usize {
        self.nq
    }
    pub fn nv(&self) -> usize {
        self.nv
    }
    pub fn nu(&self) -> usize {
        self.nu
    }

    /// Index of a named object, or `None`.
    pub fn id_of(&self, kind: ObjType, name: &str) -> Option<usize> {
        let c = CString::new(name).ok()?;
        let id = unsafe { (self.mj.inner.name2id)(self.ptr, kind as c_int, c.as_ptr()) };
        (id >= 0).then_some(id as usize)
    }

    /// Name of an object by index, or `None` for an unnamed or out-of-range one.
    pub fn name_of(&self, kind: ObjType, id: usize) -> Option<String> {
        let p = unsafe { (self.mj.inner.id2name)(self.ptr, kind as c_int, id as c_int) };
        if p.is_null() {
            return None;
        }
        Some(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
    }

    /// Every actuator name, in index order. `None` where an actuator is
    /// unnamed, which MJCF permits.
    pub fn actuator_names(&self) -> Vec<Option<String>> {
        (0..self.nu).map(|i| self.name_of(ObjType::Actuator, i)).collect()
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { (self.mj.inner.delete_model)(self.ptr) }
    }
}

/// Simulation state for a [`Model`]. Freed on drop.
pub struct Data {
    mj: Arc<MuJoCo>,
    ptr: *mut c_void,
}

unsafe impl Send for Data {}

impl Data {
    pub fn new(model: &Model) -> Result<Data, String> {
        let ptr = unsafe { (model.mj.inner.make_data)(model.ptr) };
        if ptr.is_null() {
            return Err("mj_makeData returned null (out of memory?)".to_string());
        }
        Ok(Data { mj: model.mj.clone(), ptr })
    }

    /// Advance the simulation by one `timestep`.
    pub fn step(&mut self, model: &Model) {
        unsafe { (self.mj.inner.step)(model.ptr, self.ptr) }
    }

    /// Forward dynamics without integrating: what you call after setting a
    /// pose, to make the derived quantities consistent with it.
    pub fn forward(&mut self, model: &Model) {
        unsafe { (self.mj.inner.forward)(model.ptr, self.ptr) }
    }

    pub fn reset(&mut self, model: &Model) {
        unsafe { (self.mj.inner.reset)(model.ptr, self.ptr) }
    }

    /// Read a state component. The returned length is MuJoCo's own
    /// `mj_stateSize` for that spec, so a caller never has to know it.
    pub fn get(&self, model: &Model, spec: StateSpec) -> Vec<f64> {
        let n = unsafe { (self.mj.inner.state_size)(model.ptr, spec as c_int) } as usize;
        let mut out = vec![0.0f64; n];
        if n > 0 {
            unsafe { (self.mj.inner.get_state)(model.ptr, self.ptr, out.as_mut_ptr(), spec as c_int) };
        }
        out
    }

    /// Write a state component. Errors rather than truncating on a length
    /// mismatch: `mj_setState` reads exactly `mj_stateSize` doubles from the
    /// pointer, so a short slice is an out-of-bounds read inside MuJoCo.
    pub fn set(&mut self, model: &Model, spec: StateSpec, values: &[f64]) -> Result<(), String> {
        let n = unsafe { (self.mj.inner.state_size)(model.ptr, spec as c_int) } as usize;
        if values.len() != n {
            return Err(format!("{spec:?} takes {n} values, got {}", values.len()));
        }
        if n > 0 {
            unsafe { (self.mj.inner.set_state)(model.ptr, self.ptr, values.as_ptr(), spec as c_int) };
        }
        Ok(())
    }

    /// Simulated time, in seconds.
    pub fn time(&self, model: &Model) -> f64 {
        self.get(model, StateSpec::TIME).first().copied().unwrap_or(0.0)
    }
}

impl Drop for Data {
    fn drop(&mut self) {
        unsafe { (self.mj.inner.delete_data)(self.ptr) }
    }
}
