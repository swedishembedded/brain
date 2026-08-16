// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Block primitives for **3D causal** video autoencoders - the sibling of
//! [`crate::blocks`] for `[C, T, H, W]` volumes.
//!
//! This is a separate module rather than a widening of [`crate::blocks`] on
//! purpose. Those builders take `(prefix, c, h, w, x)` and five consumers
//! (AutoencoderKL, VQGAN, CodeFormer, RRDBNet, the SDXL UNet) are built on that
//! exact signature; threading a time axis through all of them would destabilise
//! every one of them for no benefit, because none of them has a time axis. What
//! IS shared is the kernel set - almost every op below dispatches the same
//! kernel the 2D builder does, at 3D `Params`.
//!
//! # What a 3D causal VAE actually needs
//!
//! Reading the Wan-VAE reference removed most of the "video means 3D
//! everywhere" work up front:
//!
//! * **Only the main convs are genuinely 3D.** Spatial resampling and the
//!   attention block fold time into the batch and use plain 2D convs. Both are
//!   expressible as `conv3d` with `KT = 1`, which is why this module dispatches
//!   exactly one convolution kernel for every case: a `(3,3,3)` residual conv, a
//!   `(3,1,1)` temporal conv, a per-frame `(1,3,3)` spatial conv and a
//!   `(1,1,1)` projection differ only in [`Conv3d`].
//! * **The norm is a channel-axis L2 normalisation with a learnable gain**
//!   (`F.normalize(x, dim=1) * sqrt(dim) * gamma`), not GroupNorm and not
//!   RMSNorm - see [`Builder3d::rms_norm`].
//! * **Spatial upsampling is a per-frame `nearest-exact` at scale 2**, which for
//!   an exact integer 2x is provably identical to plain nearest
//!   (`floor(d/2 + 0.25) == floor(d/2)` for integer `d`), so `upsample2` is
//!   bit-correct. [`Builder3d::upsample2`] is the only 2x in this module; a
//!   non-integer scale would break that equivalence and has no path here.
//!
//! # The time axis is the channel axis of a reshaped view
//!
//! `[C, T, H, W]` viewed as `[N=C, C'=T, H, W]` turns every time-axis slice,
//! placement and concatenation into the existing NCHW channel kernels
//! (`concat_split`, `chan_place`, `concat2`). No new kernel exists for any of
//! them, and none is needed.
//!
//! # `feat_cache` is an SSA value, not device state
//!
//! The chunked causal forward carries the last [`CACHE_T`] frames of every
//! `CausalConv3d`'s input across chunk boundaries. Because the whole clip is
//! recorded as ONE static graph before a single submit, that cache is just a
//! buffer flowing from one chunk's sub-graph into the next - see [`FeatCache`].
//! Nothing is mutated on the device between chunks and nothing is read back.
//!
//! # Performance
//!
//! `conv3d` is the direct one-thread-per-output kernel. The 2D builder's
//! `im2col_at` + `matmul_reg3` lowering has no 3D twin (a 3D im2col operand is
//! the classic way past `max_storage_buffer_binding_size`), and the per-frame
//! spatial convs that COULD use it are the minority of the FLOPs here - the
//! `(3,3,3)` residual convs dominate. Correctness first: this module exists to
//! be parity-gated, and a lowering is a later, separately-measured change.

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use std::collections::HashMap;

use crate::blocks::Tensors;

// Kernel-table indices (order matches KERNELS).
const K_CONV3D: usize = 0;
const K_SILU: usize = 1;
const K_ADD2: usize = 2;
const K_UPSAMPLE2: usize = 3;
const K_NCHW_NLC: usize = 4;
const K_NLC_NCHW: usize = 5;
const K_L2NORM_SCALE: usize = 6;
const K_ATTN_SCORES: usize = 7;
const K_ATTN_SOFTMAX: usize = 8;
const K_ATTN_APPLY: usize = 9;
const K_CONCAT2: usize = 10;
const K_CONCAT_SPLIT: usize = 11;
const K_CHAN_PLACE: usize = 12;
const K_SCALE_CHAN: usize = 13;
const K_ADD_CHAN_BCAST: usize = 14;

/// Frames of cross-chunk history every causal conv keeps (upstream `CACHE_T`).
/// A `(3,1,1)`/`(3,3,3)` kernel with a one-sided pad of 2 needs exactly the
/// previous two input frames to reproduce the whole-clip result.
pub const CACHE_T: u32 = 2;

/// The 3D builder's kernel set, in slot order.
pub const KERNELS: [(&str, &str); 15] = [
    ("conv3d", kernels::CONV3D),
    ("silu", kernels::SILU),
    ("add2", kernels::ADD2),
    ("upsample2", kernels::UPSAMPLE2),
    ("nchw_nlc", kernels::NCHW_NLC),
    ("nlc_nchw", kernels::NLC_NCHW),
    ("l2norm_scale", kernels::L2NORM_SCALE),
    ("attn_scores_bidir", kernels::ATTN_SCORES_BIDIR),
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
    ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR),
    ("concat2", kernels::CONCAT2),
    ("concat_split", kernels::CONCAT_SPLIT),
    ("chan_place", kernels::CHAN_PLACE),
    ("scale_chan", kernels::SCALE_CHAN),
    ("add_chan_bcast", kernels::ADD_CHAN_BCAST),
];

/// Slot index the first caller-supplied kernel gets when a set is built with
/// [`kernels_with`].
pub const NEXT_SLOT: usize = KERNELS.len();

/// Copy [`KERNELS`] into the front of a fixed-size kernel set whose remaining
/// slots the caller fills - the same idiom as [`crate::blocks::kernels_with`],
/// so a crate needing these blocks plus its own kernels never restates the list.
pub const fn kernels_with<const N: usize>() -> [(&'static str, &'static str); N] {
    let mut out = [("", ""); N];
    let mut i = 0;
    while i < KERNELS.len() {
        out[i] = KERNELS[i];
        i += 1;
    }
    out
}

/// A device tensor with its `[C, T, H, W]` shape (batch 1 throughout).
///
/// The shape travels with the buffer because every op here is a different view
/// of the same four numbers - a time slice is a channel slice of `[N=C, T, H, W]`,
/// a per-frame conv is a `KT=1` 3D conv - and passing the dims separately is how
/// a caller ends up convolving over the wrong axis with shapes that still fit.
#[derive(Clone)]
pub struct T3 {
    pub buf: DeviceBuffer,
    pub c: u32,
    pub t: u32,
    pub h: u32,
    pub w: u32,
}

/// Shape only - `DeviceBuffer` is opaque, and the shape is what every assertion
/// message here is about.
impl std::fmt::Debug for T3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{},{},{},{}]", self.c, self.t, self.h, self.w)
    }
}

impl T3 {
    /// Element count.
    pub fn len(&self) -> u64 {
        self.c as u64 * self.t as u64 * self.h as u64 * self.w as u64
    }

    /// Whether the tensor is empty (never true for a recorded activation; the
    /// clippy companion of [`T3::len`]).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Frames as a flat element count (`C*T*H*W` fits u32 for every shape here).
    fn n(&self) -> u32 {
        self.len() as u32
    }
}

/// A convolution's per-axis geometry.
///
/// `pt` is the **already-doubled** low temporal pad, exactly as `conv3d.wgsl`
/// and `dwconv3d.wgsl` define it: upstream's `CausalConv3d(..., padding=1)`
/// becomes `pt = 2`, and the high side of time gets nothing. Passing `pad_t`
/// here instead of `2*pad_t` produces a model that is off by one frame
/// everywhere and still runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv3d {
    pub kt: u32,
    pub kh: u32,
    pub kw: u32,
    pub st: u32,
    pub sh: u32,
    pub sw: u32,
    pub pt: u32,
    pub ph: u32,
    pub pw: u32,
}

impl Conv3d {
    /// `CausalConv3d(cin, cout, 3, padding=1)` - the only true 3D conv in the
    /// Wan-VAE: `(3,3,3)`, stride 1, symmetric spatial pad 1, causal `pt = 2`.
    pub const fn causal3() -> Conv3d {
        Conv3d { kt: 3, kh: 3, kw: 3, st: 1, sh: 1, sw: 1, pt: 2, ph: 1, pw: 1 }
    }

    /// `CausalConv3d(cin, cout, 1)` - a pointwise projection. `padding` is
    /// `(0,0,0)`, so `pt = 0` and no cross-frame history exists at all.
    pub const fn point() -> Conv3d {
        Conv3d { kt: 1, kh: 1, kw: 1, st: 1, sh: 1, sw: 1, pt: 0, ph: 0, pw: 0 }
    }

    /// `CausalConv3d(c, ..., (3,1,1), padding=(1,0,0))` - the temporal conv of
    /// an `upsample3d` resample. Time only, causal `pt = 2`.
    pub const fn time_causal() -> Conv3d {
        Conv3d { kt: 3, kh: 1, kw: 1, st: 1, sh: 1, sw: 1, pt: 2, ph: 0, pw: 0 }
    }

    /// `CausalConv3d(c, c, (3,1,1), stride=(2,1,1), padding=(0,0,0))` - the
    /// temporal conv of a `downsample3d` resample. NO pad: the frame it needs
    /// for history is concatenated in by the caller.
    pub const fn time_down() -> Conv3d {
        Conv3d { kt: 3, kh: 1, kw: 1, st: 2, sh: 1, sw: 1, pt: 0, ph: 0, pw: 0 }
    }

    /// A per-frame 2D conv: `KT = 1`, so each output frame reads only its own
    /// input frame. This is how `nn.Conv2d` under
    /// `rearrange('b c t h w -> (b t) c h w')` is expressed here.
    pub const fn spatial(k: u32, stride: u32, pad: u32) -> Conv3d {
        Conv3d { kt: 1, kh: k, kw: k, st: 1, sh: stride, sw: stride, pt: 0, ph: pad, pw: pad }
    }

    /// Output extents for an input `[t, h, w]`.
    fn out_dims(&self, t: u32, h: u32, w: u32) -> (u32, u32, u32) {
        (
            (t + self.pt - self.kt) / self.st + 1,
            (h + 2 * self.ph - self.kh) / self.sh + 1,
            (w + 2 * self.pw - self.kw) / self.sw + 1,
        )
    }
}

/// One `CausalConv3d` call site's cross-chunk state.
#[derive(Clone, Debug)]
pub enum CacheSlot {
    /// No chunk has been through this site yet (upstream `None`).
    Empty,
    /// Upstream's literal `'Rep'` sentinel, used only by `upsample3d`: the
    /// first chunk passed through untouched and the temporal conv's history
    /// restarts from zeros rather than from that chunk.
    Rep,
    /// The previous chunk's last [`CACHE_T`] frames of this site's input.
    Frames(T3),
}

/// The per-chunk `feat_cache` / `feat_idx` pair.
///
/// Slots are addressed by the order call sites are reached inside ONE chunk,
/// which upstream implements as a `[0]` list it mutates while walking the
/// module tree. [`FeatCache::claim`] is that counter; [`FeatCache::rewind`] is
/// the `feat_idx = [0]` at the top of every chunk.
pub struct FeatCache {
    slots: Vec<CacheSlot>,
    idx: usize,
}

impl FeatCache {
    /// A cache with `n` slots, all [`CacheSlot::Empty`]. `n` may exceed the
    /// number of sites actually reached (upstream sizes it with
    /// `count_conv3d`, which also counts the shortcut convs that never consume
    /// a slot); extra slots are inert.
    pub fn new(n: usize) -> FeatCache {
        FeatCache { slots: vec![CacheSlot::Empty; n], idx: 0 }
    }

    /// Restart the per-chunk slot counter.
    pub fn rewind(&mut self) {
        self.idx = 0;
    }

    /// Claim the next slot, returning its index and current contents.
    pub fn claim(&mut self) -> (usize, CacheSlot) {
        let i = self.idx;
        assert!(i < self.slots.len(), "feat_cache overflow: {} slots", self.slots.len());
        self.idx += 1;
        (i, self.slots[i].clone())
    }

    /// Overwrite a slot (after its previous contents' last read).
    pub fn set(&mut self, i: usize, s: CacheSlot) {
        self.slots[i] = s;
    }
}

/// Graph-construction state for a 3D causal autoencoder (borrows the device +
/// host tensors), mirroring [`crate::blocks::Builder`].
///
/// Inference only: there is no training tape here, because the 3D backward
/// kernels (`conv3d_dx` / `conv3d_dw` / `l2norm_scale_dx` / `l2norm_scale_dg`)
/// exist but nothing trains this graph yet. Adding one is a `tape`/`Op` pair
/// exactly like the 2D builder's, not a redesign.
pub struct Builder3d<'a> {
    gpu: &'a Gpu,
    t: &'a Tensors,
    steps: Vec<Step>,
    taps: Vec<(String, DeviceBuffer, usize)>,
    taps_on: bool,
    wmemo: HashMap<String, DeviceBuffer>,
    pool: HashMap<u64, Vec<DeviceBuffer>>,
    pooling: bool,
    uploaded: u64,
}

impl<'a> Builder3d<'a> {
    /// New builder over `gpu` (built with a kernel set whose first
    /// [`KERNELS`]`.len()` slots are [`KERNELS`]) and the host `tensors`.
    /// `taps_on` records named intermediates and pins their buffers.
    pub fn new(gpu: &'a Gpu, tensors: &'a Tensors, taps_on: bool) -> Builder3d<'a> {
        Builder3d {
            gpu,
            t: tensors,
            steps: Vec::new(),
            taps: Vec::new(),
            taps_on,
            wmemo: HashMap::new(),
            pool: HashMap::new(),
            // Activation reuse is bit-exact (the graph runs in order with
            // barriers, and a buffer is only freed after its last consumer step
            // is recorded), but a mis-placed `free` in a caller is silent and
            // looks exactly like a parity bug. `BRAIN_VAE3D_NOPOOL=1` turns
            // reuse off so that hypothesis can be killed in one run.
            pooling: !taps_on && std::env::var("BRAIN_VAE3D_NOPOOL").is_err(),
            uploaded: 0,
        }
    }

    /// The device the graph is being recorded on.
    pub fn gpu(&self) -> &'a Gpu {
        self.gpu
    }

    /// A host tensor by name; panics naming the tensor if absent (import
    /// validates coverage up front, so this only fires on a schedule bug).
    pub fn get(&self, name: &str) -> &(Vec<usize>, Vec<f32>) {
        self.t.get(name).unwrap_or_else(|| panic!("vae::blocks3d: missing tensor {name}"))
    }

    /// Whether a tensor is present (a resample block has a `time_conv` only in
    /// its 3D modes).
    pub fn has(&self, name: &str) -> bool {
        self.t.contains_key(name)
    }

    /// Upload a host tensor to the device by name, memoized so one tensor is
    /// one device buffer however many times it is read.
    pub fn dev(&mut self, name: &str) -> DeviceBuffer {
        if let Some(b) = self.wmemo.get(name) {
            return b.clone();
        }
        let t = self.t;
        let data = &t.get(name).unwrap_or_else(|| panic!("vae::blocks3d: missing tensor {name}")).1;
        let buf = self.upload(data);
        self.wmemo.insert(name.to_string(), buf.clone());
        buf
    }

    /// Upload host data the builder synthesised, under a name of its own.
    pub fn dev_owned(&mut self, name: &str, data: &[f32]) -> DeviceBuffer {
        if let Some(b) = self.wmemo.get(name) {
            return b.clone();
        }
        let buf = self.upload(data);
        self.wmemo.insert(name.to_string(), buf.clone());
        buf
    }

    /// Upload one weight tensor, non-ReBAR-safe - `storage()` + `write()` with
    /// a periodic drain, for the reasons [`crate::blocks::Builder`] documents at
    /// length (mapped-at-creation forces an inefficient memory type on a card
    /// without resizable BAR, and wgpu only reclaims staging on a real submit).
    fn upload(&mut self, data: &[f32]) -> DeviceBuffer {
        let buf = self.gpu.storage(data.len() as u64);
        let bits: Vec<u32> = data.iter().map(|v| v.to_bits()).collect();
        self.gpu.write(&buf, &bits);
        self.gpu.poll_wait();
        self.uploaded += 4 * data.len() as u64;
        if self.uploaded > (1 << 30) {
            let _ = self.gpu.read(&buf, 1);
            self.uploaded = 0;
        }
        buf
    }

    /// Allocate an activation buffer of `len` words, reusing a freed one of the
    /// same length when available.
    pub fn act(&mut self, len: u64) -> DeviceBuffer {
        if let Some(b) = self.pool.get_mut(&len).and_then(Vec::pop) {
            return b;
        }
        self.gpu.storage(len)
    }

    /// An all-zero volume, uploaded once per distinct shape and never pooled.
    ///
    /// `upsample3d`'s `'Rep'` chunk stores `cat([zeros_like(x), x])` as its
    /// history, so the zero frame is a real operand of the NEXT chunk's
    /// temporal conv - it is not a padding shortcut, and skipping it changes
    /// that chunk's output.
    pub fn zeros(&mut self, c: u32, t: u32, h: u32, w: u32) -> T3 {
        let n = c as usize * t as usize * h as usize * w as usize;
        let buf = self.dev_owned(&format!("__zeros.{c}x{t}x{h}x{w}"), &vec![0.0f32; n]);
        T3 { buf, c, t, h, w }
    }

    /// A fresh activation shaped `[c, t, h, w]`.
    fn act3(&mut self, c: u32, t: u32, h: u32, w: u32) -> T3 {
        let buf = self.act(c as u64 * t as u64 * h as u64 * w as u64);
        T3 { buf, c, t, h, w }
    }

    /// Return an activation buffer for reuse. MUST be called only after its last
    /// read step has been pushed.
    pub fn free(&mut self, x: T3) {
        if self.pooling {
            self.pool.entry(x.len()).or_default().push(x.buf);
        }
    }

    /// Record a named intermediate for later readback. No-op unless `taps_on`.
    pub fn tap(&mut self, name: &str, x: &T3) {
        if self.taps_on {
            self.taps.push((name.to_string(), x.buf.clone(), x.n() as usize));
        }
    }

    /// Append a caller-recorded step (for kernels outside this module).
    pub fn push_step(&mut self, step: Step) {
        self.steps.push(step);
    }

    /// Consume the builder, yielding the recorded steps and taps.
    pub fn finish(self) -> (Vec<Step>, Vec<(String, DeviceBuffer, usize)>) {
        (self.steps, self.taps)
    }

    // ---------------------------------------------------------------- convs

    /// Convolution `prefix.{weight,bias}` over `x`, at geometry `spec`.
    ///
    /// `conv3d`'s `Params` is 19 words in the order the kernel declares them:
    /// `N, Cin, T, H, W, Cout, KT, KH, KW, st, sh, sw, pt, ph, pw, groups, To,
    /// Ho, Wo` - the three output extents are computed by the host and passed
    /// in, not derived in the kernel.
    pub fn conv(&mut self, prefix: &str, cout: u32, spec: Conv3d, x: &T3) -> T3 {
        let wgt = self.dev(&format!("{prefix}.weight"));
        let bias = self.dev(&format!("{prefix}.bias"));
        self.conv_raw(&wgt, &bias, cout, spec, x)
    }

    /// Convolution with **caller-chosen** output extents, for the asymmetric
    /// `F.pad(x, (0,1,0,1))` + stride-2 downsample.
    ///
    /// `conv3d` bounds-checks every tap and skips the ones that fall outside
    /// the volume, which is exactly a zero border - so asking for one more
    /// output row/column than the symmetric formula gives reproduces the
    /// right/bottom-only pad without a separate `pad2d` pass. Same trick
    /// [`crate::blocks::Builder::conv_down`] uses in 2D.
    pub fn conv_sized(
        &mut self,
        prefix: &str,
        cout: u32,
        spec: Conv3d,
        out: (u32, u32, u32),
        x: &T3,
    ) -> T3 {
        let wgt = self.dev(&format!("{prefix}.weight"));
        let bias = self.dev(&format!("{prefix}.bias"));
        let (to, ho, wo) = out;
        let y = self.act3(cout, to, ho, wo);
        self.steps.push(self.gpu.step(
            K_CONV3D,
            &[&x.buf, &wgt, &bias, &y.buf],
            &[
                1, x.c, x.t, x.h, x.w, cout, spec.kt, spec.kh, spec.kw, spec.st, spec.sh, spec.sw,
                spec.pt, spec.ph, spec.pw, 1, to, ho, wo,
            ],
            y.n(),
        ));
        y
    }

    /// [`Builder3d::conv`] over already-uploaded weight/bias buffers.
    pub fn conv_raw(
        &mut self,
        wgt: &DeviceBuffer,
        bias: &DeviceBuffer,
        cout: u32,
        spec: Conv3d,
        x: &T3,
    ) -> T3 {
        let (to, ho, wo) = spec.out_dims(x.t, x.h, x.w);
        assert!(to > 0 && ho > 0 && wo > 0, "conv3d: empty output for {spec:?} on {x:?}");
        let y = self.act3(cout, to, ho, wo);
        self.steps.push(self.gpu.step(
            K_CONV3D,
            &[&x.buf, wgt, bias, &y.buf],
            &[
                1, x.c, x.t, x.h, x.w, cout, spec.kt, spec.kh, spec.kw, spec.st, spec.sh, spec.sw,
                spec.pt, spec.ph, spec.pw, 1, to, ho, wo,
            ],
            y.n(),
        ));
        y
    }

    /// A `CausalConv3d` at a `feat_cache` call site: the previous chunk's last
    /// [`CACHE_T`] frames of THIS conv's input are concatenated in front of `x`
    /// and subtracted from the low pad, and this chunk's last [`CACHE_T`] frames
    /// replace the slot.
    ///
    /// This is the whole reason a chunked causal decode reproduces a whole-clip
    /// one: the concatenated history is literally the frames the whole-clip
    /// conv would have read. Two details are load-bearing and both come
    /// straight from upstream:
    ///
    /// * the cache is cloned from the conv's **input**, BEFORE the conv;
    /// * when this chunk has fewer than [`CACHE_T`] frames (every decoder chunk
    ///   does - they are one latent frame each), the stored cache is the
    ///   previous cache's LAST frame followed by this chunk, so the two frames
    ///   are still the two most recent frames of the running sequence.
    pub fn conv_cached(
        &mut self,
        prefix: &str,
        cout: u32,
        spec: Conv3d,
        x: &T3,
        cache: &mut FeatCache,
    ) -> T3 {
        let (idx, slot) = cache.claim();
        // cache_x = x[:, :, -CACHE_T:]
        let keep = CACHE_T.min(x.t);
        let mut cache_x = self.time_slice(x, x.t - keep, keep);
        if cache_x.t < CACHE_T {
            if let CacheSlot::Frames(prev) = &slot {
                let prev = prev.clone();
                let tail = self.time_slice(&prev, prev.t - 1, 1);
                let joined = self.time_cat(&tail, &cache_x);
                self.free(tail);
                self.free(cache_x);
                cache_x = joined;
            }
        }
        // The conv itself, over [prev_cache ++ x] with the low pad reduced by
        // however many history frames were supplied. `_padding[4] > 0` upstream:
        // a pointwise conv has no temporal pad and ignores the cache entirely.
        let y = match &slot {
            CacheSlot::Frames(prev) if spec.pt > 0 => {
                assert!(prev.t <= spec.pt, "cache of {} frames exceeds pt={}", prev.t, spec.pt);
                let prev = prev.clone();
                let xin = self.time_cat(&prev, x);
                let mut s = spec;
                s.pt -= prev.t;
                let y = self.conv(prefix, cout, s, &xin);
                self.free(xin);
                y
            }
            _ => self.conv(prefix, cout, spec, x),
        };
        if let CacheSlot::Frames(prev) = slot {
            self.free(prev); // last read was the concat above
        }
        cache.set(idx, CacheSlot::Frames(cache_x));
        y
    }

    // ---------------------------------------------------------------- norms

    /// The Wan-VAE `RMS_norm`: `F.normalize(x, dim=1) * sqrt(C) * gamma`.
    ///
    /// Despite the upstream class name this is **not** an RMSNorm. It is a
    /// plain L2 normalisation over the **channel** axis with a learnable
    /// per-channel gain and no epsilon inside the mean - i.e. exactly
    /// `l2norm_scale`, which normalises the last axis of an `[N, D]` view. The
    /// tensor is therefore permuted to `[T*H*W, C]` rows, normalised, and
    /// permuted back. brain's `rmsnorm*` (`x / sqrt(mean(x^2) + eps)` over the
    /// last axis) and `gn_*` are both the wrong operator here, and both would
    /// produce plausible output.
    ///
    /// `sqrt(C)` is folded into the uploaded gain: `gamma' = sqrt(C) * gamma`
    /// is one host multiply per channel and saves a whole pass over the volume.
    /// The reference multiplies in the other order
    /// (`normalize(x) * scale * gamma`), so this differs by at most one f32
    /// rounding on a per-channel constant.
    ///
    /// The epsilon is `1e-24` under the square root, not `0`: upstream's
    /// `F.normalize` divides by `max(||x||, 1e-12)`, and this is the same
    /// guard against an all-zero row expressed inside `rsqrt`. On any row with
    /// a normal-magnitude norm it changes nothing in f32.
    pub fn rms_norm(&mut self, prefix: &str, x: &T3) -> T3 {
        let gname = format!("{prefix}.gamma");
        let (shape, gamma) = self.get(&gname);
        assert_eq!(
            gamma.len(),
            x.c as usize,
            "{gname}: {shape:?} does not match {} channels",
            x.c
        );
        let scaled: Vec<f32> = gamma.iter().map(|g| g * (x.c as f32).sqrt()).collect();
        let g = self.dev_owned(&format!("{prefix}.gamma.scaled"), &scaled);

        let thw = x.t * x.h * x.w;
        let rows = self.act(x.len());
        self.steps.push(self.gpu.step(
            K_NCHW_NLC,
            &[&x.buf, &rows],
            &[x.n(), x.c, thw],
            x.n(),
        ));
        let normed = self.act(x.len());
        self.steps.push(self.gpu.step(
            K_L2NORM_SCALE,
            &[&rows, &g, &normed],
            &[thw, x.c, f(1e-24)],
            x.n(),
        ));
        let y = self.act3(x.c, x.t, x.h, x.w);
        self.steps.push(self.gpu.step(
            K_NLC_NCHW,
            &[&normed, &y.buf],
            &[x.n(), x.c, thw],
            x.n(),
        ));
        self.free(T3 { buf: rows, ..x.clone() });
        self.free(T3 { buf: normed, ..x.clone() });
        y
    }

    // ------------------------------------------------------------ pointwise

    /// SiLU/swish (`x * sigmoid(x)`), elementwise.
    pub fn silu(&mut self, x: &T3) -> T3 {
        let y = self.act3(x.c, x.t, x.h, x.w);
        self.steps.push(self.gpu.step(K_SILU, &[&x.buf, &y.buf], &[x.n()], x.n()));
        y
    }

    /// Elementwise sum of two identically-shaped volumes.
    pub fn add(&mut self, a: &T3, b: &T3) -> T3 {
        assert_eq!(a.len(), b.len(), "add: {a:?} vs {b:?}");
        let y = self.act3(a.c, a.t, a.h, a.w);
        self.steps.push(self.gpu.step(K_ADD2, &[&a.buf, &b.buf, &y.buf], &[a.n()], a.n()));
        y
    }

    /// Per-channel scale (`y[c] = x[c] * s[c]`), `s` a host vector of length `C`.
    pub fn scale_chan(&mut self, name: &str, s: &[f32], x: &T3) -> T3 {
        assert_eq!(s.len(), x.c as usize, "scale_chan: {} values for {} channels", s.len(), x.c);
        let sb = self.dev_owned(name, s);
        let y = self.act3(x.c, x.t, x.h, x.w);
        let inner = x.t * x.h * x.w;
        self.steps.push(self.gpu.step(
            K_SCALE_CHAN,
            &[&x.buf, &sb, &y.buf],
            &[x.n(), x.c, inner],
            x.n(),
        ));
        y
    }

    /// Per-channel shift (`y[c] = x[c] + v[c]`), `v` a host vector of length `C`.
    pub fn add_chan(&mut self, name: &str, v: &[f32], x: &T3) -> T3 {
        assert_eq!(v.len(), x.c as usize, "add_chan: {} values for {} channels", v.len(), x.c);
        let vb = self.dev_owned(name, v);
        let y = self.act3(x.c, x.t, x.h, x.w);
        self.steps.push(self.gpu.step(
            K_ADD_CHAN_BCAST,
            &[&x.buf, &vb, &y.buf],
            &[1, x.c, x.t * x.h * x.w],
            x.n(),
        ));
        y
    }

    /// Per-frame nearest-neighbour 2x spatial upsample.
    ///
    /// The reference uses `mode='nearest-exact'`, which differs from `nearest`
    /// only in index rounding (`floor((d+0.5)*in/out)` vs `floor(d*in/out)`).
    /// At an exact integer 2x the two agree for every integer `d`
    /// (`floor(d/2 + 0.25) == floor(d/2)`), so `upsample2` is bit-correct - and
    /// ONLY at an exact 2x, which is why this method has no scale parameter.
    pub fn upsample2(&mut self, x: &T3) -> T3 {
        let y = self.act3(x.c, x.t, x.h * 2, x.w * 2);
        // [C,T,H,W] is C*T planes of (H,W): the kernel's own N is unused here.
        self.steps.push(self.gpu.step(
            K_UPSAMPLE2,
            &[&x.buf, &y.buf],
            &[1, x.c * x.t, x.h, x.w],
            y.n(),
        ));
        y
    }

    // ------------------------------------------------------------- reshapes

    /// Frames `[t0, t0+n)` of `x`.
    ///
    /// A time slice of `[C,T,H,W]` is a CHANNEL slice of the `[N=C, T, H, W]`
    /// view, so `concat_split` does it with no new kernel.
    pub fn time_slice(&mut self, x: &T3, t0: u32, n: u32) -> T3 {
        assert!(t0 + n <= x.t, "time_slice [{t0},{}) of {} frames", t0 + n, x.t);
        let y = self.act3(x.c, n, x.h, x.w);
        self.steps.push(self.gpu.step(
            K_CONCAT_SPLIT,
            &[&x.buf, &y.buf],
            &[x.c, x.t, n, t0, x.h, x.w],
            y.n(),
        ));
        y
    }

    /// Channels `[c0, c0+n)` of `x` (`chunk(2, dim=1)` and friends).
    pub fn chan_slice(&mut self, x: &T3, c0: u32, n: u32) -> T3 {
        assert!(c0 + n <= x.c, "chan_slice [{c0},{}) of {} channels", c0 + n, x.c);
        let y = self.act3(n, x.t, x.h, x.w);
        self.steps.push(self.gpu.step(
            K_CONCAT_SPLIT,
            &[&x.buf, &y.buf],
            &[1, x.c, n, c0, x.t, x.h * x.w],
            y.n(),
        ));
        y
    }

    /// `torch.cat([a, b], dim=2)` - concatenation on TIME, again as the channel
    /// axis of the `[N=C, T, H, W]` view.
    pub fn time_cat(&mut self, a: &T3, b: &T3) -> T3 {
        assert!(a.c == b.c && a.h == b.h && a.w == b.w, "time_cat: {a:?} vs {b:?}");
        let y = self.act3(a.c, a.t + b.t, a.h, a.w);
        self.steps.push(self.gpu.step(
            K_CONCAT2,
            &[&a.buf, &b.buf, &y.buf],
            &[a.c, a.t, b.t, a.h, a.w],
            y.n(),
        ));
        y
    }

    /// Write `src`'s frames into `dst[:, t0 .. t0+src.t]`, where `dst` holds
    /// `t_tot` frames. Used to assemble a chunked forward's output without a
    /// left-fold of growing concatenations.
    pub fn time_place(&mut self, src: &T3, dst: &DeviceBuffer, t_tot: u32, t0: u32) {
        assert!(t0 + src.t <= t_tot, "time_place [{t0},{}) into {t_tot}", t0 + src.t);
        self.steps.push(self.gpu.step(
            K_CHAN_PLACE,
            &[&src.buf, dst],
            &[src.c, t_tot, src.t, t0, src.h, src.w],
            src.n(),
        ));
    }

    /// The `upsample3d` channel-to-time fold: `[2C, T, H, W] -> [C, 2T, H, W]`
    /// with `y[c, 2i] = x[c, i]` and `y[c, 2i+1] = x[C+c, i]`.
    ///
    /// Upstream writes it as `reshape(b,2,c,t,h,w)` then
    /// `stack((x[:,0], x[:,1]), 3)`. Split into the two channel halves, the
    /// result is `concat2` over the `[N = C*T, 1, H, W]` view - the two halves
    /// interleave frame-by-frame because the concatenated axis sits directly
    /// above `(H,W)` in that view.
    pub fn time_interleave(&mut self, x: &T3) -> T3 {
        assert_eq!(x.c % 2, 0, "time_interleave needs an even channel count, got {}", x.c);
        let c = x.c / 2;
        let lo = self.chan_slice(x, 0, c);
        let hi = self.chan_slice(x, c, c);
        let y = self.act3(c, x.t * 2, x.h, x.w);
        self.steps.push(self.gpu.step(
            K_CONCAT2,
            &[&lo.buf, &hi.buf, &y.buf],
            &[c * x.t, 1, 1, x.h, x.w],
            y.n(),
        ));
        self.free(lo);
        self.free(hi);
        y
    }

    // ------------------------------------------------------------ attention

    /// Per-frame single-head self-attention over the `H*W` spatial positions:
    /// `x + proj(attn(qkv(norm(x))))`.
    ///
    /// `norm`/`qkv`/`proj` are full tensor prefixes. The reference folds time
    /// into the batch (`rearrange('b c t h w -> (b t) c h w')`) and attends with
    /// one head of width `C` over `h*w` keys, so this dispatches the shared
    /// bidirectional trio with `bsz = T` and `n_heads = 1` - the frames are
    /// independent batch items, and no token ever attends across a frame
    /// boundary.
    pub fn attn(&mut self, norm: &str, qkv: &str, proj: &str, x: &T3) -> T3 {
        let (c, t, hw) = (x.c, x.t, x.h * x.w);
        let normed = self.rms_norm(norm, x);
        // to_qkv is nn.Conv2d(dim, 3*dim, 1) - a per-frame 1x1, i.e. a linear.
        let qkv_chw = self.conv(qkv, 3 * c, Conv3d::point(), &normed);
        self.free(normed);

        // `[3C, T, H, W]` -> rows `[T*HW, 3C]`, the layout the trio indexes as
        // `qkv[(b*tcols + i)*stride + off + d]` with `b` the frame.
        //
        // The permute's `hw` argument is **T*H*W, not H*W**. Time is the axis
        // BELOW the channel here (`conv3d` writes `[Cout, To, Ho, Wo]`), so the
        // volume is one `[3C, T*H*W]` matrix, not `T` separate `[3C, H*W]`
        // ones. The two agree exactly when `T == 1` - which every chunk of the
        // reference's own decode happens to be, so a golden alone cannot catch
        // this. `wan_vae_encoder_is_chunk_size_invariant` can, and did.
        let rows = self.act(qkv_chw.len());
        self.steps.push(self.gpu.step(
            K_NCHW_NLC,
            &[&qkv_chw.buf, &rows],
            &[qkv_chw.n(), 3 * c, t * hw],
            qkv_chw.n(),
        ));
        self.free(qkv_chw);

        let scores = self.act(t as u64 * hw as u64 * hw as u64);
        self.steps.push(self.gpu.step(
            K_ATTN_SCORES,
            &[&rows, &scores],
            &[t, 1, hw, c, 3 * c, 0, c],
            t * hw * hw,
        ));
        let probs = self.act(t as u64 * hw as u64 * hw as u64);
        self.steps.push(self.gpu.step(K_ATTN_SOFTMAX, &[&scores, &probs], &[t, 1, hw], t * hw));
        let ctx = self.act(x.len());
        self.steps.push(self.gpu.step(
            K_ATTN_APPLY,
            &[&probs, &rows, &ctx],
            &[t, 1, hw, c, 3 * c, 2 * c, c],
            t * hw * c,
        ));
        self.pool_raw(t as u64 * hw as u64 * hw as u64, scores);
        self.pool_raw(t as u64 * hw as u64 * hw as u64, probs);
        self.pool_raw(3 * x.len(), rows);

        // rows `[T*HW, C]` -> `[C, T, H, W]` (same `T*HW` reasoning as above).
        let chw = self.act3(c, t, x.h, x.w);
        self.steps.push(self.gpu.step(
            K_NLC_NCHW,
            &[&ctx, &chw.buf],
            &[x.n(), c, t * hw],
            x.n(),
        ));
        self.pool_raw(x.len(), ctx);

        let out = self.conv(proj, c, Conv3d::point(), &chw);
        self.free(chw);
        let y = self.add(x, &out);
        self.free(out);
        y
    }

    /// Return a raw buffer of known length to the pool (for scratch that has no
    /// natural [`T3`] shape, e.g. the attention score slab).
    fn pool_raw(&mut self, len: u64, buf: DeviceBuffer) {
        if self.pooling {
            self.pool.entry(len).or_default().push(buf);
        }
    }
}
