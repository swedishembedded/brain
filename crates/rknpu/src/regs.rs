// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running neural networks on the
// NPUs of low-cost embedded Linux SoCs for its clients. If your team needs
// expertise in embedded NPU bring-up then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Register command programs: the words a task's program counter feeds the
//! NPU's engines.
//!
//! A command word is `target << 48 | value << 16 | register`. Every task writes
//! the full [`PROFILE`] (so no state leaks from the previous task) with its own
//! fields overridden, then a four-word tail that links to the next task.
//!
//! The profile values are transcribed from open-rknpu's
//! `src/open_rknpu/register_profile.py` (MIT, measured on RV1103/RV1106; see
//! `/NOTICE.md`). Register names follow Mesa's Rocket `registers.xml` (MIT),
//! which open-rknpu re-checked on this NPU; a field neither project decoded keeps
//! its measured default and is not named here.

/// Engine a command word targets (bits 63:48).
pub const TARGET_CNA: u16 = 0x0201;
pub const TARGET_CORE: u16 = 0x0801;
pub const TARGET_DPU: u16 = 0x1001;
pub const TARGET_DPU_RDMA: u16 = 0x2001;
pub const TARGET_DDMA: u16 = 0x0401;
const TARGET_PC: u16 = 0x0101;
const TARGET_PC_OP: u16 = 0x0081;
const TARGET_NONE: u16 = 0x0041;

// CNA: convolution input, weights and geometry.
pub const CNA_CONV_CON1: u16 = 0x100c;
pub const CNA_CONV_CON2: u16 = 0x1010;
pub const CNA_CONV_CON3: u16 = 0x1014;
pub const CNA_DATA_SIZE0: u16 = 0x1020;
pub const CNA_DATA_SIZE1: u16 = 0x1024;
pub const CNA_DATA_SIZE2: u16 = 0x1028;
pub const CNA_DATA_SIZE3: u16 = 0x102c;
pub const CNA_WEIGHT_SIZE0: u16 = 0x1030;
pub const CNA_WEIGHT_SIZE1: u16 = 0x1034;
pub const CNA_WEIGHT_SIZE2: u16 = 0x1038;
/// Not in Mesa's table; open-rknpu measured it as the data entries (high half).
pub const CNA_DATA_ENTRIES_HI: u16 = 0x103c;
pub const CNA_CBUF_CON1: u16 = 0x1044;
/// Undecoded; open-rknpu measured it as the input extent in 2048-atom units.
pub const CNA_INPUT_EXTENT: u16 = 0x1048;
pub const CNA_CVT_CON0: u16 = 0x104c;
pub const CNA_CVT_CON1: u16 = 0x1050;
pub const CNA_CVT_CON2: u16 = 0x1054;
pub const CNA_CVT_CON3: u16 = 0x1058;
pub const CNA_CVT_CON4: u16 = 0x105c;
pub const CNA_PAD_CON0: u16 = 0x1068;
pub const CNA_FEATURE_DATA_ADDR: u16 = 0x1070;
pub const CNA_DMA_CON1: u16 = 0x107c;
pub const CNA_DMA_CON2: u16 = 0x1080;
pub const CNA_FC_DATA_SIZE0: u16 = 0x1084;
pub const CNA_FC_DATA_SIZE1: u16 = 0x1088;
/// Undecoded; open-rknpu's dense convolution writes 0.
pub const CNA_UNDECODED_108C: u16 = 0x108c;
/// Undecoded; open-rknpu's dense convolution writes 0.
pub const CNA_UNDECODED_101C: u16 = 0x101c;
pub const CNA_DCOMP_ADDR0: u16 = 0x1110;
pub const CNA_PAD_CON1: u16 = 0x1184;
/// Weight-table stride (open-rknpu: `8*k*k*tiles`).
pub const CNA_WEIGHT_STRIDE: u16 = 0x1188;
/// Feature width minus one, duplicated in both halves.
pub const CNA_FEATURE_WIDTH: u16 = 0x118c;

// CORE: the MAC array's output cube.
pub const CORE_DATAOUT_SIZE_0: u16 = 0x3014;
pub const CORE_DATAOUT_SIZE_1: u16 = 0x3018;
pub const CORE_CLIP_TRUNCATE: u16 = 0x301c;

// DPU: post-processing and write-back.
pub const DPU_FEATURE_MODE_CFG: u16 = 0x400c;
pub const DPU_DST_BASE_ADDR: u16 = 0x4020;
pub const DPU_DST_SURF_STRIDE: u16 = 0x4024;
pub const DPU_DATA_CUBE_WIDTH: u16 = 0x4030;
pub const DPU_DATA_CUBE_HEIGHT: u16 = 0x4034;
pub const DPU_DATA_CUBE_CHANNEL: u16 = 0x403c;
pub const DPU_BS_CFG: u16 = 0x4040;
pub const DPU_BS_MUL_CFG: u16 = 0x4048;
pub const DPU_BS_OW_CFG: u16 = 0x4050;
pub const DPU_BS_OW_OP: u16 = 0x4054;
pub const DPU_WDMA_SIZE_0: u16 = 0x4058;
pub const DPU_WDMA_SIZE_1: u16 = 0x405c;
pub const DPU_BN_CFG: u16 = 0x4060;
pub const DPU_BN_RELUX_CMP_VALUE: u16 = 0x406c;
pub const DPU_EW_CFG: u16 = 0x4070;
pub const DPU_EW_CVT_OFFSET_VALUE: u16 = 0x4074;
pub const DPU_EW_CVT_SCALE_VALUE: u16 = 0x4078;
pub const DPU_OUT_CVT_OFFSET: u16 = 0x4080;
pub const DPU_OUT_CVT_SCALE: u16 = 0x4084;
pub const DPU_OUT_CVT_SHIFT: u16 = 0x4088;
pub const DPU_SURFACE_ADD: u16 = 0x40c0;
/// Second activation lower clamp, mirroring [`DPU_BN_RELUX_CMP_VALUE`].
pub const DPU_RELUX_CMP_VALUE_2: u16 = 0x40e0;

// DPU_RDMA: the DPU's own reads (bias block, element-wise operands).
pub const RDMA_DATA_CUBE_WIDTH: u16 = 0x500c;
pub const RDMA_DATA_CUBE_HEIGHT: u16 = 0x5010;
pub const RDMA_DATA_CUBE_CHANNEL: u16 = 0x5014;
pub const RDMA_SRC_BASE_ADDR: u16 = 0x5018;
pub const RDMA_BRDMA_CFG: u16 = 0x501c;
pub const RDMA_BS_BASE_ADDR: u16 = 0x5020;
pub const RDMA_ERDMA_CFG: u16 = 0x5034;
pub const RDMA_EW_BASE_ADDR: u16 = 0x5038;
pub const RDMA_EW_SURF_STRIDE: u16 = 0x5040;
pub const RDMA_FEATURE_MODE_CFG: u16 = 0x5044;

/// Every register a task writes, with its measured default and target.
pub const PROFILE: [(u16, u32, u16); 126] = [
    (0x1004, 0x0000000e, 0x0201),
    (0x3004, 0x0000000e, 0x0801),
    (0x1040, 0x00000008, 0x0201),
    (0x4004, 0x0000000e, 0x1001),
    (0x5004, 0x0000000e, 0x2001),
    (0x100c, 0x2000a000, 0x0201),
    (0x1010, 0x000003ff, 0x0201),
    (0x1014, 0x00000009, 0x0201),
    (0x101c, 0xffe00000, 0x0201),
    (0x1020, 0x00080008, 0x0201),
    (0x1024, 0x00020010, 0x0201),
    (0x1028, 0x00000008, 0x0201),
    (0x102c, 0x00000040, 0x0201),
    (0x1030, 0x00000010, 0x0201),
    (0x1034, 0x00000004, 0x0201),
    (0x1038, 0x01010004, 0x0201),
    (0x103c, 0x00100000, 0x0201),
    (0x1040, 0x00000008, 0x0201),
    (0x1044, 0x00100002, 0x0201),
    (0x1048, 0x04000000, 0x0201),
    (0x104c, 0x000e38e0, 0x0201),
    (0x1050, 0x40004000, 0x0201),
    (0x1054, 0x00014000, 0x0201),
    (0x1058, 0xffe00000, 0x0201),
    (0x105c, 0xffe00000, 0x0201),
    (0x1060, 0x00000000, 0x0201),
    (0x1064, 0x00000000, 0x0201),
    (0x1068, 0x00000000, 0x0201),
    (0x106c, 0x00000000, 0x0201),
    (0x1070, 0x00002000, 0x0201),
    (0x1074, 0x00000000, 0x0201),
    (0x1078, 0x400f400f, 0x0201),
    (0x107c, 0x00000018, 0x0201),
    (0x1080, 0x00000018, 0x0201),
    (0x1084, 0x00180001, 0x0201),
    (0x1088, 0x00000010, 0x0201),
    (0x108c, 0x000000ff, 0x0201),
    (0x1094, 0x00000000, 0x0201),
    (0x1100, 0x00000000, 0x0201),
    (0x1104, 0x00000000, 0x0201),
    (0x1110, 0x00000ac0, 0x0201),
    (0x1140, 0x00000000, 0x0201),
    (0x1144, 0x00000000, 0x0201),
    (0x1184, 0x0000ff80, 0x0201),
    (0x1188, 0x00000002, 0x0201),
    (0x118c, 0x000f000f, 0x0201),
    (0x3010, 0x00000001, 0x0801),
    (0x3014, 0x00070007, 0x0801),
    (0x3018, 0x0000000f, 0x0801),
    (0x301c, 0x00000020, 0x0801),
    (0x400c, 0x000001e4, 0x1001),
    (0x4010, 0x00800000, 0x1001),
    (0x4014, 0x00000000, 0x1001),
    (0x4018, 0x00000000, 0x1001),
    (0x401c, 0x7fffffff, 0x1001),
    (0x4020, 0x00003000, 0x1001),
    (0x4024, 0x00000400, 0x1001),
    (0x4028, 0x7fffffff, 0x1001),
    (0x402c, 0x7fffffff, 0x1001),
    (0x4030, 0x00000007, 0x1001),
    (0x4034, 0x00000007, 0x1001),
    (0x4038, 0x00000000, 0x1001),
    (0x403c, 0x0003000f, 0x1001),
    (0x4040, 0x00121d80, 0x1001),
    (0x4044, 0x00000000, 0x1001),
    (0x4048, 0x00000001, 0x1001),
    (0x404c, 0x80000000, 0x1001),
    (0x4050, 0x30000001, 0x1001),
    (0x4054, 0x8e000000, 0x1001),
    (0x4058, 0x00000003, 0x1001),
    (0x405c, 0x00070007, 0x1001),
    (0x4060, 0x00000013, 0x1001),
    (0x4064, 0x00000000, 0x1001),
    (0x4068, 0x00000000, 0x1001),
    (0x406c, 0x80000000, 0x1001),
    (0x4070, 0x010041c1, 0x1001),
    (0x4074, 0x00000000, 0x1001),
    (0x4078, 0x00000001, 0x1001),
    (0x407c, 0x80000000, 0x1001),
    (0x4080, 0xffffff80, 0x1001),
    (0x4084, 0x00004040, 0x1001),
    (0x4088, 0x00000016, 0x1001),
    (0x4090, 0x00000000, 0x1001),
    (0x4094, 0x00000000, 0x1001),
    (0x4098, 0x00000000, 0x1001),
    (0x409c, 0x00000000, 0x1001),
    (0x40a0, 0x00000000, 0x1001),
    (0x40c0, 0x00000400, 0x1001),
    (0x40c4, 0x00000000, 0x1001),
    (0x40d8, 0x80000000, 0x1001),
    (0x40dc, 0x7fffffff, 0x1001),
    (0x40e0, 0x80000000, 0x1001),
    (0x40e4, 0x7fffffff, 0x1001),
    (0x40e8, 0x80000000, 0x1001),
    (0x40ec, 0x7fffffff, 0x1001),
    (0x4100, 0x00000000, 0x1001),
    (0x4104, 0x00000000, 0x1001),
    (0x4108, 0x00000000, 0x1001),
    (0x410c, 0x00000000, 0x1001),
    (0x4110, 0x00000000, 0x1001),
    (0x4114, 0x00000000, 0x1001),
    (0x4118, 0x00000000, 0x1001),
    (0x411c, 0x00000000, 0x1001),
    (0x4120, 0x00000000, 0x1001),
    (0x4124, 0x00000000, 0x1001),
    (0x4128, 0x00000000, 0x1001),
    (0x412c, 0x00000000, 0x1001),
    (0x500c, 0x00000007, 0x2001),
    (0x5010, 0x00000007, 0x2001),
    (0x5014, 0x0000000f, 0x2001),
    (0x5018, 0x00000000, 0x2001),
    (0x501c, 0x0000000e, 0x2001),
    (0x5020, 0x00000b00, 0x2001),
    (0x5028, 0x00000000, 0x2001),
    (0x502c, 0x00000000, 0x2001),
    (0x5034, 0x00000001, 0x2001),
    (0x5038, 0x00000000, 0x2001),
    (0x5040, 0x00000000, 0x2001),
    (0x5044, 0x00007810, 0x2001),
    (0x5048, 0x00000000, 0x2001),
    (0x504c, 0x00000000, 0x2001),
    (0x5064, 0x00000000, 0x2001),
    (0x5068, 0x01010101, 0x2001),
    (0x506c, 0x00000000, 0x2001),
    (0x8028, 0x0000000c, 0x0401),
    (0x802c, 0xffffffff, 0x0401),
];

/// Tail of a task program: link to the next program (or end the chain), the
/// zero descriptor and the engine enable word.
const TAIL_WORDS: usize = 4;
/// Control value of a terminal tail (no successor).
const TERMINAL_CONTROL: u32 = 0x28;

/// Words in one task program, tail included.
pub const PROGRAM_WORDS: usize = PROFILE.len() + TAIL_WORDS;
/// Bytes one task program occupies, rounded so every program starts 64-byte aligned.
pub const PROGRAM_BYTES: usize = (PROGRAM_WORDS * 8).next_multiple_of(64);

/// The register values of one task, starting from [`PROFILE`].
#[derive(Clone)]
pub struct Program {
    values: [u32; PROFILE.len()],
}

impl Default for Program {
    fn default() -> Self {
        Program { values: PROFILE.map(|(_, v, _)| v) }
    }
}

impl Program {
    /// Override `reg`. Panics if the profile has no such register: a typo in
    /// an emitter must not silently become a write nobody performs.
    pub fn set(&mut self, reg: u16, value: u32) -> &mut Self {
        let mut found = false;
        for (slot, (r, _, _)) in self.values.iter_mut().zip(PROFILE.iter()) {
            if *r == reg {
                *slot = value;
                found = true;
            }
        }
        assert!(found, "register {reg:#06x} is not in the NPU task profile");
        self
    }

    pub fn get(&self, reg: u16) -> u32 {
        PROFILE.iter().position(|(r, _, _)| *r == reg).map(|i| self.values[i]).unwrap_or_else(|| panic!("register {reg:#06x} is not in the NPU task profile"))
    }

    /// Encode into `out` (at least [`PROGRAM_WORDS`] words): the profile words,
    /// then a tail linking to the successor program at `next` (an address
    /// relative to the job base) of `next_words` profile words, or a terminal
    /// tail when `next` is `None`. `enable` is the task's engine enable mask.
    pub fn encode(&self, out: &mut [u64], next: Option<(u32, usize)>, enable: u32) {
        for (i, ((reg, _, target), value)) in PROFILE.iter().zip(self.values.iter()).enumerate() {
            out[i] = word(*target, *value, *reg);
        }
        let (link, control) = match next {
            // The successor's fetch amount as the driver itself would program
            // it: (words + 4 extra, in units of 2) - 1 on this NPU.
            Some((addr, words)) => (addr, ((words + 5) / 2 - 1) as u32),
            None => (0, TERMINAL_CONTROL),
        };
        let n = PROFILE.len();
        out[n] = word(TARGET_PC, link, 0x10);
        out[n + 1] = word(TARGET_PC, control, 0x14);
        out[n + 2] = word(TARGET_NONE, 0, 0);
        out[n + 3] = word(TARGET_PC_OP, enable, 0x08);
    }
}

fn word(target: u16, value: u32, reg: u16) -> u64 {
    (u64::from(target) << 48) | (u64::from(value) << 16) | u64::from(reg)
}
