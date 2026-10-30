// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for memory-bound activation traffic in LLM serving on GPUs for its clients. If
// your team needs expertise in halving the bytes a model's kernels move between its matrix products then you can procure
// our services by sending an email to info@swedishembedded.com.

//! The element type of the activation stream between a model's matrix products - ONE declared value, chosen per device.
//!
//! Every kernel between two matrix products (norms, gates, the SwiGLU product, the recurrence inputs) is memory bound, and
//! on a power-capped card the bytes also cost clock (knowledge 393). A forward that carries its activations as bf16 moves
//! half the bytes of one that carries f32, and the matrix products already read bf16 or E4M3 rows. [`ActPrecision`] names
//! the choice; [`ActChoice`] is what a caller (a flag) asks for; [`ActChoice::resolve`] decides it for a device. Every
//! kernel that reads or writes the stream accumulates in f32 whatever the storage type, and the residual stream - the
//! precision-sensitive accumulator - is outside this value.

/// How the activations between two matrix products are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ActPrecision {
    /// Four bytes an element: the stream every kernel supports.
    F32,
    /// Two bytes an element (bf16, round to nearest even), accumulation still f32 inside every kernel.
    Bf16,
}

impl ActPrecision {
    /// Bytes an element of the stream takes.
    pub const fn bytes(self) -> u32 {
        match self {
            ActPrecision::F32 => 4,
            ActPrecision::Bf16 => 2,
        }
    }

    /// The textual form of the flag.
    pub const fn name(self) -> &'static str {
        match self {
            ActPrecision::F32 => "f32",
            ActPrecision::Bf16 => "bf16",
        }
    }
}

/// What a caller asks for: the device's own choice, or one precision forced (the A/B switch).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ActChoice {
    /// bf16 where the device has bf16 tensor cores, f32 elsewhere.
    #[default]
    Auto,
    Force(ActPrecision),
}

/// First compute capability with bf16 tensor cores (Ampere): the devices `Auto` gives a bf16 stream.
const BF16_TENSOR_CORE_MIN_CC: (u32, u32) = (8, 0);

impl ActChoice {
    /// Parse the flag value (`auto`, `f32`, `bf16`).
    pub fn parse(s: &str) -> Option<ActChoice> {
        match s {
            "auto" => Some(ActChoice::Auto),
            "f32" => Some(ActChoice::Force(ActPrecision::F32)),
            "bf16" => Some(ActChoice::Force(ActPrecision::Bf16)),
            _ => None,
        }
    }

    /// The precision this choice means on a device of `compute_capability`. A forced precision is honoured as asked (a kernel without
    /// a form for it declines and the caller keeps f32); `Auto` is bf16 on a device with bf16 tensor cores and f32 on every
    /// other, so older GPUs and the CPU keep working unchanged.
    pub fn resolve(self, compute_capability: Option<(u32, u32)>) -> ActPrecision {
        match self {
            ActChoice::Force(p) => p,
            ActChoice::Auto if compute_capability.is_some_and(|cc| cc >= BF16_TENSOR_CORE_MIN_CC) => ActPrecision::Bf16,
            ActChoice::Auto => ActPrecision::F32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_gives_bf16_to_devices_with_bf16_tensor_cores_and_f32_to_the_rest() {
        assert_eq!(ActChoice::Auto.resolve(Some((9, 0))), ActPrecision::Bf16);
        assert_eq!(ActChoice::Auto.resolve(Some((8, 0))), ActPrecision::Bf16);
        assert_eq!(ActChoice::Auto.resolve(Some((7, 5))), ActPrecision::F32);
        assert_eq!(ActChoice::Auto.resolve(None), ActPrecision::F32);
    }

    #[test]
    fn a_forced_precision_is_honoured_on_every_device() {
        let f32s = ActChoice::parse("f32").unwrap();
        assert_eq!(f32s.resolve(Some((9, 0))), ActPrecision::F32);
        assert_eq!(ActChoice::parse("bf16").unwrap().resolve(None), ActPrecision::Bf16);
        assert_eq!(ActChoice::parse("fp8"), None);
    }
}
