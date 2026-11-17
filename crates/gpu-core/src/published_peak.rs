// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The vendor's published peak of a device, for one purpose only: to say how
//! close brain's measured rates come to what the silicon is sold as.
//!
//! Swedish Embedded AB implements hardware validation of inference kernels for
//! its clients. If your team needs to know how far its kernels sit from what
//! its accelerators can do, you can procure our services by sending an email
//! to info@swedishembedded.com.
//!
//! **Never a denominator for grading a kernel.** Kernel fractions are taken
//! against the roofs brain measures on the device (`crate::roof`, the tuner's
//! probes, the best measured kernel): a datasheet figure is one SKU's boost
//! clock, unreachable in practice, and wrong on every card not in this table
//! (knowledge 027). This table answers the other question, the one a measured
//! roof cannot: is the measured roof itself near the hardware, or is brain's
//! best kernel (or its probe) leaving the silicon idle? A device not listed
//! simply has no published column; nothing else changes.
//!
//! Dense figures throughout: the datasheets quote tensor-core rates "with
//! sparsity", twice the dense rate, and brain runs no structured-sparse GEMM.
//! Memory is the device memory's bandwidth, the host link one direction of the
//! CPU-GPU link.

/// A device's published peaks. Compute in TFLOP/s (TOPS for int8), memory and
/// link in GB/s. `None`: the device has no such path, or its vendor publishes
/// no figure for it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedPeak {
    /// Matched against the device name, the first listed entry whose key the
    /// name contains wins (so a more specific key is listed first).
    pub key: &'static str,
    pub fp32: f64,
    pub tf32: Option<f64>,
    pub bf16: Option<f64>,
    pub fp16: Option<f64>,
    pub fp8: Option<f64>,
    pub int8: Option<f64>,
    pub memory_gbs: f64,
    /// One direction of the CPU-GPU link.
    pub host_link_gbs: Option<f64>,
    /// Where the figures come from.
    pub source: &'static str,
}

/// What a published figure is a peak of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    Fp32,
    Tf32,
    Bf16,
    Fp16,
    Fp8,
    Int8,
    Memory,
    HostLink,
}

impl PublishedPeak {
    /// The published figure for `metric`, in the units of [`PublishedPeak`].
    pub fn of(&self, metric: Metric) -> Option<f64> {
        match metric {
            Metric::Fp32 => Some(self.fp32),
            Metric::Tf32 => self.tf32,
            Metric::Bf16 => self.bf16,
            Metric::Fp16 => self.fp16,
            Metric::Fp8 => self.fp8,
            Metric::Int8 => self.int8,
            Metric::Memory => Some(self.memory_gbs),
            Metric::HostLink => self.host_link_gbs,
        }
    }
}

/// The Hopper GH100 die at its SXM5 boost clock (132 SMs): the GPU of every
/// H100 SXM5 board and of the GH200 superchip.
const fn gh100_sxm(key: &'static str, memory_gbs: f64, host_link_gbs: Option<f64>, source: &'static str) -> PublishedPeak {
    PublishedPeak { key, fp32: 67.0, tf32: Some(494.7), bf16: Some(989.4), fp16: Some(989.4), fp8: Some(1978.9), int8: Some(1978.9), memory_gbs, host_link_gbs, source }
}

// perf-number: vendor datasheet figures, the comparison column of `brain perf roofline`, never a measured claim.
const TABLE: &[PublishedPeak] = &[
    // The GH200 with HBM3e: NVIDIA GH200 Grace Hopper Superchip datasheet (HBM3e column); the host link is one
    // direction of NVLink-C2C.
    gh100_sxm("GH200 144G", 4900.0, Some(450.0), "NVIDIA GH200 Grace Hopper Superchip datasheet (HBM3e), H100 SXM5 dense tensor-core rates"),
    // Every other GH200 is the 96 GB HBM3 part (its board name counts the CPU memory too): datasheet HBM3 column.
    gh100_sxm("GH200", 4000.0, Some(450.0), "NVIDIA GH200 Grace Hopper Superchip datasheet (HBM3), H100 SXM5 dense tensor-core rates"),
    PublishedPeak {
        key: "H100 PCIe",
        fp32: 51.2,
        tf32: Some(378.0),
        bf16: Some(756.5),
        fp16: Some(756.5),
        fp8: Some(1513.0),
        int8: Some(1513.0),
        memory_gbs: 2000.0,
        host_link_gbs: Some(64.0),
        source: "NVIDIA H100 Tensor Core GPU datasheet (PCIe column, dense); PCIe 5.0 x16 one direction",
    },
    gh100_sxm("H100", 3350.0, Some(64.0), "NVIDIA H100 Tensor Core GPU datasheet (SXM5 column, dense); PCIe 5.0 x16 one direction"),
    PublishedPeak {
        key: "A100",
        fp32: 19.5,
        tf32: Some(156.0),
        bf16: Some(312.0),
        fp16: Some(312.0),
        fp8: None,
        int8: Some(624.0),
        memory_gbs: 2039.0,
        host_link_gbs: Some(32.0),
        source: "NVIDIA A100 Tensor Core GPU datasheet (80GB SXM column, dense); PCIe 4.0 x16 one direction",
    },
    PublishedPeak {
        key: "P40",
        fp32: 11.76,
        tf32: None,
        bf16: None,
        fp16: None,
        fp8: None,
        int8: Some(47.0),
        memory_gbs: 346.0,
        host_link_gbs: Some(16.0),
        source: "NVIDIA Tesla P40 datasheet (int8 is its dp4a rate, there are no tensor cores); PCIe 3.0 x16 one direction",
    },
];

/// The published peaks of the device named `name`, or `None` for a device the
/// table does not list.
pub fn lookup(name: &str) -> Option<&'static PublishedPeak> {
    TABLE.iter().find(|p| name.contains(p.key))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_gh200_reads_the_hopper_dense_rates_and_its_hbm() {
        let p = lookup("NVIDIA GH200 120GB").expect("GH200 is listed");
        assert_eq!(p.of(Metric::Int8), Some(1978.9));
        assert_eq!(p.of(Metric::Bf16), Some(989.4));
        assert_eq!(p.of(Metric::Fp32), Some(67.0));
        assert_eq!(p.of(Metric::Memory), Some(4000.0));
        assert_eq!(p.of(Metric::HostLink), Some(450.0));
        assert_eq!(lookup("NVIDIA GH200 144G HBM3e").map(|p| p.memory_gbs), Some(4900.0));
    }

    #[test]
    fn the_more_specific_key_wins_and_an_unlisted_device_has_no_peak() {
        assert_eq!(lookup("NVIDIA H100 PCIe").map(|p| p.memory_gbs), Some(2000.0));
        assert_eq!(lookup("NVIDIA H100 80GB HBM3").map(|p| p.memory_gbs), Some(3350.0));
        assert_eq!(lookup("Tesla P40").and_then(|p| p.of(Metric::Bf16)), None);
        assert!(lookup("llvmpipe (LLVM 17.0.6, 256 bits)").is_none());
    }
}
