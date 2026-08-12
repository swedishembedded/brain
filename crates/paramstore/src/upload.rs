// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The generic **disk → VRAM** weight mover: pull one named tensor from a
//! [`checkpoint::TensorSource`] straight onto a device, with peak *host*
//! allocation bounded to one [`crate::UPLOAD_CHUNK_WORDS`] chunk no matter
//! how large the tensor is.
//!
//! Any code path that puts model weights on a device should go through here
//! - [`crate::ParamStore`] included - because the two things such a path has
//! to get right are easy to get wrong independently, and getting either
//! wrong is invisible until it OOMs a card mid-load:
//!
//! 1. **Bounded host staging.** Prefer [`checkpoint::TensorSource::raw_words`]
//!    (an mmap whose on-disk dtype already matches what the device binds
//!    lends its bytes with no copy at all), else decode
//!    `UPLOAD_CHUNK_WORDS` elements at a time into a scratch the source
//!    reuses. Never `tensor()`/`tensor_u32()`, which materialize the whole
//!    thing - a single real Omni `lm_head` is 1.2 GB as f32.
//! 2. **Bounded DEVICE staging.** wgpu only frees a `write_buffer` staging
//!    allocation on `poll_wait`, and with no submitted compute even that can
//!    be a no-op - so [`Uploader`] polls after every tensor and forces a real
//!    submit + drain with a one-element readback roughly every GiB. Without
//!    it a multi-GB load accrues a shadow copy of everything it has uploaded
//!    and OOMs a 24 GB card partway through (measured on a non-ReBAR P40).
//!
//! Neither is a property of one tensor, which is why they live in a type a
//! caller holds across a whole load rather than in a free function each
//! caller has to remember to imitate.
//!
//! Note the separate, *backend-level* cost this cannot touch: on the default
//! wgpu backend a non-ReBAR Pascal card holds ~2x each uploaded buffer
//! resident (`crates/gpu-core/tests/vram_overhead.rs`). Chunking the write
//! CALLS does not change that; only `--device vulkan` does.

use checkpoint::TensorSource;
use gpu_core::{DeviceBuffer, Gpu};

use crate::dtype::DeviceLimits;
use crate::UPLOAD_CHUNK_WORDS;

/// Bytes uploaded between forced drains. A one-element readback costs a
/// round trip; amortized over a GiB it is negligible, and it is what actually
/// reclaims wgpu's write staging (see this module's doc).
const DRAIN_EVERY_BYTES: u64 = 1 << 30;

/// A bounded weight uploader bound to one device. Hold ONE across a whole
/// model load (not one per tensor) so the drain accounting spans the load -
/// that is the part that keeps device-side staging from accruing.
pub struct Uploader<'g> {
    gpu: &'g Gpu,
    since_drain: u64,
    limits: DeviceLimits,
}

impl<'g> Uploader<'g> {
    pub fn new(gpu: &'g Gpu) -> Uploader<'g> {
        Uploader { gpu, since_drain: 0, limits: DeviceLimits::of(gpu) }
    }

    /// The device this uploader targets.
    pub fn gpu(&self) -> &'g Gpu {
        self.gpu
    }

    /// This device's queried allocation/binding ceilings - see
    /// [`crate::dtype::DeviceLimits`]. Read from the device at construction,
    /// so a caller sizing its own buffers asks the same source this uploader
    /// enforces against rather than hardcoding a second number.
    pub fn limits(&self) -> DeviceLimits {
        self.limits
    }

    /// Force a real submit + device wait now, whatever the drain accounting
    /// says, so wgpu retires the recorded work and releases every buffer that
    /// has been dropped since the last one.
    ///
    /// The seam a loader needs when it streams a WORKING SET rather than a
    /// resident model: buffers dropped host-side are not freed on the card
    /// until the commands referencing them have completed, and with no
    /// submitted compute a bare `poll_wait` can be a no-op. A loop that
    /// allocates a multi-GiB group per iteration and drops it therefore
    /// accumulates every iteration's group on the device until it OOMs - the
    /// exact failure this exists to prevent - even though its live set is one
    /// group. Costs one round trip.
    pub fn drain(&mut self, buf: &DeviceBuffer) {
        let _ = self.gpu.read(buf, 1);
        self.since_drain = 0;
    }

    /// Allocate a `numel`-word device buffer and stream `name`'s f32 data
    /// into it. `Err` if the source does not have the tensor or its element
    /// count disagrees with `numel` - a size mismatch means the caller's
    /// shape model and the checkpoint disagree, which must not be papered
    /// over with a partially-written buffer.
    pub fn tensor(&mut self, source: &dyn TensorSource, name: &str, numel: usize) -> Result<DeviceBuffer, String> {
        // `storage()` + `write*` rather than `storage_init`: create_buffer_init's
        // mapped-at-creation path forces weights into an inefficient memory type
        // on a non-ReBAR GPU (a 16.8 GB encoder ballooned to ~30 GB and OOM'd).
        self.limits.check_alloc(name, numel)?;
        let buf = self.gpu.storage(numel as u64);
        self.tensor_into(&buf, source, name, numel)?;
        Ok(buf)
    }

    /// [`Self::tensor`] into a buffer the caller already allocated.
    pub fn tensor_into(&mut self, buf: &DeviceBuffer, source: &dyn TensorSource, name: &str, numel: usize) -> Result<(), String> {
        let mut written = 0usize;
        let found = if let Some(words) = source.raw_words(name) {
            if words.len() != numel {
                return Err(format!("upload '{name}': source has {} words, expected {numel}", words.len()));
            }
            for (i, part) in words.chunks(UPLOAD_CHUNK_WORDS).enumerate() {
                self.gpu.write_at(buf, (i * UPLOAD_CHUNK_WORDS) as u64, part);
                written += part.len();
            }
            true
        } else {
            source.with_tensor_chunks(name, UPLOAD_CHUNK_WORDS, &mut |off, chunk| {
                self.gpu.write_f32_at(buf, off, chunk);
                written += chunk.len();
            })
        };
        self.finish_tensor(name, found, written, numel)
    }

    /// [`Self::tensor`] for a **packed int8** tensor (brain's `U32` storage
    /// convention). Uses the zero-copy `raw_words` view when the mapping
    /// allows it, else the bounded
    /// [`TensorSource::with_tensor_u32_chunks`] decode - never
    /// `WeightReader::tensor_u32`, which materializes the whole packed tensor
    /// on the host first.
    pub fn packed(&mut self, source: &dyn TensorSource, name: &str, numel: usize) -> Result<DeviceBuffer, String> {
        self.limits.check_alloc(name, numel)?;
        let buf = self.gpu.storage(numel as u64);
        self.packed_into(&buf, source, name, numel)?;
        Ok(buf)
    }

    /// [`Self::packed`] into a caller-allocated buffer.
    pub fn packed_into(&mut self, buf: &DeviceBuffer, source: &dyn TensorSource, name: &str, numel: usize) -> Result<(), String> {
        let mut written = 0usize;
        let found = source.with_tensor_u32_chunks(name, UPLOAD_CHUNK_WORDS, &mut |off, chunk| {
            self.gpu.write_at(buf, off, chunk);
            written += chunk.len();
        });
        self.finish_tensor(name, found, written, numel)
    }

    /// Upload an already-host-resident f32 slice in bounded write calls. For
    /// the case where the data was *computed* rather than read (a dequantized
    /// tensor, a synthesized table) - the host copy already exists, so only
    /// the device-side staging discipline is left to apply.
    pub fn host_f32(&mut self, data: &[f32]) -> DeviceBuffer {
        let buf = self.gpu.storage(data.len() as u64);
        for (i, part) in data.chunks(UPLOAD_CHUNK_WORDS).enumerate() {
            self.gpu.write_f32_at(&buf, (i * UPLOAD_CHUNK_WORDS) as u64, part);
        }
        self.drain_accounting(&buf, 4 * data.len() as u64);
        buf
    }

    /// Shared tail: validate the transfer, then apply the staging-reclaim
    /// discipline. Split out so `tensor_into`/`packed_into` cannot drift.
    fn finish_tensor(&mut self, name: &str, found: bool, written: usize, numel: usize) -> Result<(), String> {
        if !found {
            return Err(format!("upload '{name}': not present in this source"));
        }
        if written != numel {
            return Err(format!("upload '{name}': wrote {written} words, expected {numel}"));
        }
        // Reclaim THIS tensor's write staging before the next one starts.
        // The periodic forced drain is left to the caller's `maybe_drain`
        // (it needs a buffer to read back from, and only the caller still
        // holds one here) - see this module's doc for why both steps exist.
        self.account(4 * numel as u64);
        Ok(())
    }

    /// Record `bytes` as uploaded and reclaim this tensor's write staging.
    /// The seam for a caller that writes into a device buffer through
    /// `Gpu::write*_at` ITSELF (e.g. a dequantize-while-streaming loader) but
    /// still wants this uploader's cross-tensor staging discipline - pair it
    /// with [`Self::maybe_drain`].
    pub fn account(&mut self, bytes: u64) {
        self.gpu.poll_wait();
        self.since_drain += bytes;
    }

    /// Force a real submit + drain if enough has been uploaded since the last
    /// one. `poll_wait` alone is not always enough - with no submitted compute
    /// the poll can be a no-op and wgpu keeps holding the write staging.
    fn drain_accounting(&mut self, buf: &DeviceBuffer, bytes: u64) {
        self.account(bytes);
        self.maybe_drain(buf);
    }

    /// Force the periodic drain against `buf` if this uploader has moved
    /// enough since the last one. Callers that upload through
    /// `tensor`/`packed` should call this once per tensor with the buffer
    /// they just filled (the store loops do); it is separate from
    /// `finish_tensor` only because that method does not own the buffer.
    pub fn maybe_drain(&mut self, buf: &DeviceBuffer) {
        if self.since_drain > DRAIN_EVERY_BYTES {
            let _ = self.gpu.read(buf, 1);
            self.since_drain = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A source that records the largest slice it ever hands out and whether
    /// the unbounded whole-tensor path was taken - the two facts that
    /// distinguish "streamed in bounded chunks" from "materialized", with no
    /// dependence on any allocator.
    struct Probe {
        f32s: HashMap<String, Vec<f32>>,
        packed: HashMap<String, Vec<u32>>,
        /// Names for which `raw_words` is allowed to succeed (simulating an
        /// aligned mmap range); everything else must go the chunked route.
        zero_copyable: Vec<String>,
        max_chunk: std::cell::Cell<usize>,
        whole_tensor_calls: std::cell::RefCell<Vec<String>>,
    }

    impl checkpoint::TensorSource for Probe {
        fn with_tensor(&self, name: &str, f: &mut dyn FnMut(&[f32])) -> bool {
            self.whole_tensor_calls.borrow_mut().push(name.to_string());
            match self.f32s.get(name) {
                Some(v) => {
                    self.max_chunk.set(self.max_chunk.get().max(v.len()));
                    f(v);
                    true
                }
                None => false,
            }
        }
        fn raw_words(&self, name: &str) -> Option<&[u32]> {
            if !self.zero_copyable.iter().any(|n| n == name) {
                return None;
            }
            self.packed.get(name).map(|v| v.as_slice())
        }
        fn with_tensor_chunks(&self, name: &str, max_elems: usize, f: &mut dyn FnMut(u64, &[f32])) -> bool {
            let Some(v) = self.f32s.get(name) else { return false };
            let chunk = if max_elems == 0 { v.len().max(1) } else { max_elems };
            for (i, part) in v.chunks(chunk).enumerate() {
                self.max_chunk.set(self.max_chunk.get().max(part.len()));
                f((i * chunk) as u64, part);
            }
            true
        }
        fn with_tensor_u32_chunks(&self, name: &str, max_elems: usize, f: &mut dyn FnMut(u64, &[u32])) -> bool {
            if let Some(words) = self.raw_words(name) {
                let chunk = if max_elems == 0 { words.len().max(1) } else { max_elems };
                for (i, part) in words.chunks(chunk).enumerate() {
                    self.max_chunk.set(self.max_chunk.get().max(part.len()));
                    f((i * chunk) as u64, part);
                }
                return true;
            }
            let Some(v) = self.packed.get(name) else { return false };
            let chunk = if max_elems == 0 { v.len().max(1) } else { max_elems };
            for (i, part) in v.chunks(chunk).enumerate() {
                self.max_chunk.set(self.max_chunk.get().max(part.len()));
                f((i * chunk) as u64, part);
            }
            true
        }
    }

    fn probe(f32s: Vec<(&str, Vec<f32>)>, packed: Vec<(&str, Vec<u32>)>, zero_copyable: &[&str]) -> Probe {
        Probe {
            f32s: f32s.into_iter().map(|(n, v)| (n.to_string(), v)).collect(),
            packed: packed.into_iter().map(|(n, v)| (n.to_string(), v)).collect(),
            zero_copyable: zero_copyable.iter().map(|s| s.to_string()).collect(),
            max_chunk: std::cell::Cell::new(0),
            whole_tensor_calls: std::cell::RefCell::new(Vec::new()),
        }
    }

    static KERNELS: &[(&str, &str)] = &[("add2", kernels::ADD2)];

    #[test]
    fn f32_upload_is_chunk_bounded_and_bit_exact() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let gpu = gpu_core::testgpu::dev(KERNELS);
        // Deliberately not a multiple of the chunk, and several chunks long.
        let n = UPLOAD_CHUNK_WORDS * 3 + 17;
        let vals: Vec<f32> = (0..n).map(|i| (i % 617) as f32 * 0.125 - 7.0).collect();
        let src = probe(vec![("w", vals.clone())], vec![], &[]);

        let mut up = Uploader::new(&gpu);
        let buf = up.tensor(&src, "w", n).expect("upload");

        assert!(src.whole_tensor_calls.borrow().is_empty(), "the unbounded with_tensor path must not be used");
        assert!(src.max_chunk.get() <= UPLOAD_CHUNK_WORDS, "chunk of {} exceeds the {UPLOAD_CHUNK_WORDS}-word bound", src.max_chunk.get());
        assert_eq!(gpu.read(&buf, n), vals, "chunked upload must be bit-identical to the source");
    }

    #[test]
    fn packed_upload_is_chunk_bounded_and_bit_exact_without_zero_copy() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let gpu = gpu_core::testgpu::dev(KERNELS);
        let n = UPLOAD_CHUNK_WORDS * 2 + 5;
        let words: Vec<u32> = (0..n).map(|i| (i as u32).wrapping_mul(2_654_435_761)).collect();
        // NOT zero-copyable: forces the bounded chunked packed decode, the
        // path a misaligned safetensors byte range would take.
        let src = probe(vec![], vec![("packed", words.clone())], &[]);

        let mut up = Uploader::new(&gpu);
        let buf = up.packed(&src, "packed", n).expect("upload");

        assert!(src.max_chunk.get() <= UPLOAD_CHUNK_WORDS, "chunk of {} exceeds the bound", src.max_chunk.get());
        let got: Vec<u32> = gpu.read(&buf, n).iter().map(|f| f.to_bits()).collect();
        assert_eq!(got, words, "packed upload must round-trip bit-exact");
    }

    #[test]
    fn packed_upload_prefers_the_zero_copy_view_when_available() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let gpu = gpu_core::testgpu::dev(KERNELS);
        let words: Vec<u32> = (0..4096u32).collect();
        let src = probe(vec![], vec![("packed", words.clone())], &["packed"]);
        let mut up = Uploader::new(&gpu);
        let buf = up.packed(&src, "packed", words.len()).expect("upload");
        let got: Vec<u32> = gpu.read(&buf, words.len()).iter().map(|f| f.to_bits()).collect();
        assert_eq!(got, words);
    }

    #[test]
    fn a_missing_or_mis_sized_tensor_is_a_clean_error_not_a_partial_buffer() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let gpu = gpu_core::testgpu::dev(KERNELS);
        let src = probe(vec![("w", vec![1.0f32, 2.0, 3.0])], vec![], &[]);
        let mut up = Uploader::new(&gpu);
        let missing = up.tensor(&src, "nope", 3).err().expect("a missing tensor must be an Err");
        assert!(missing.contains("not present"), "{missing}");
        let mis_sized = up.tensor(&src, "w", 9).err().expect("a size mismatch must be an Err");
        assert!(mis_sized.contains("expected 9"), "{mis_sized}");
    }
}
