// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements serving of NVFP4 and MXFP8 mixture-of-experts releases exactly as published for its
// clients. If your team needs expertise in microscaled expert banks then you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec: the experts of a release that stacks them in one tensor per projection (`[experts, rows, cols]`, declared per
//! tensor in `hf_quant_config.json`'s `parameter_groups`) load into one resident [`MicroBank`] per projection through
//! per-expert views, in their stored encoding:
//! - an MXFP8 bank (E4M3 elements, U8-stored E8M0 scale per 32) whose gather and grouped products equal the f64
//!   product of the decoded weights and agree with each other to the bit where both run the WGSL tier;
//! - an NVFP4 bank whose per-expert second level is the release's `amax / outer_scale_denominator`, folded once at
//!   open, carried into the bank's `global`, whose gather and grouped products likewise equal the decoded product and
//!   agree with each other to the bit on the WGSL tier.
//!
//! A product a native kernel serves (`gpu_core::native_upgrade`: on a CUDA device both NVFP4 products go to native
//! twins that sum in other orders) is held to the decoded product alone, as `gpu-core`'s `moe_nvfp4` gate holds them.
//!
//! `Ops::micro_grouped` is the grouped product at the handle's arithmetic: pinned to binary32 it is the bank's own
//! kernel to the bit; at the default tier, on a device offered the tensor-core NVFP4 kernel, it is the f64 product of
//! the decoded weights and the activations rounded to bf16 (the operand rounding of a tensor-core tier, nothing else),
//! over several tiles of one expert and a bank narrower than the kernel's 64-row block.

use checkpoint::packed::{BoundedWorkspace, CodecRegistry, MemoryBudget, PackedReader, PackedSource, PackedTensorReader, SlicedReader, Tile, TileBuf};
use data::rng::Lcg;
use gpu_core::Gpu;
use gpu_core::provider::library::GemmPrecision;
use model::ops::{micro_bank_kernels, training_kernel_list, upload_micro_bank, MicroBank, MicroRoute, MicroScaled, Ops, MICRO_GROUPED_SLOTS};

const E: usize = 3;
const N: usize = 32;
const K: usize = 256;

fn write_safetensors(path: &std::path::Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
    let mut header = serde_json::Map::new();
    let mut off = 0;
    for (name, dtype, shape, bytes) in tensors {
        header.insert((*name).into(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [off, off + bytes.len()]}));
        off += bytes.len();
    }
    let h = serde_json::to_vec(&header).unwrap();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    for (.., bytes) in tensors {
        out.extend(bytes);
    }
    std::fs::write(path, out).unwrap();
}

/// A release directory holding a stacked MXFP8 projection `mx_weight` and a stacked NVFP4 projection `nv_weight`.
fn release(dir: &std::path::Path, rng: &mut Lcg) -> [f32; E] {
    let fp8: Vec<u8> = (0..E * N * K).map(|_| (rng.next_u32() % 0x70) as u8 | if rng.next_u32().is_multiple_of(2) { 0x80 } else { 0 }).collect();
    let e8m0: Vec<u8> = (0..E * N * K / 32).map(|_| 118 + (rng.next_u32() % 8) as u8).collect();
    let codes: Vec<u8> = (0..E * N * K / 2).map(|_| rng.next_u32() as u8).collect();
    let e4m3: Vec<u8> = (0..E * N * K / 16).map(|_| 0x30 + (rng.next_u32() % 0x18) as u8).collect();
    let amax = [3.5f32, 0.75, 12.0];
    let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    write_safetensors(
        &dir.join("model.safetensors"),
        &[
            ("mx_weight", "F8_E4M3", vec![E, N, K], fp8),
            ("mx_weight_scale", "U8", vec![E, N, K / 32], e8m0),
            ("nv_weight", "U8", vec![E, N, K / 2], codes),
            ("nv_weight_scale", "F8_E4M3", vec![E, N, K / 16], e4m3),
            ("nv_weight_amax", "F32", vec![E], f32s(&amax)),
            ("nv_weight_outer_scale_denominator", "F32", vec![1], f32s(&[2688.0])),
        ],
    );
    let group = |transform: &str, compute: &str, output: &str, block: [u32; 2], format: Option<&str>, tensor: &str| {
        let mut g = serde_json::json!({"compute_dtype": compute, "output_dtype": output, "required_activation_quantization": null,
            "weight_block_shape": block, "weight_transform": transform, "tensors": [tensor]});
        if let Some(f) = format {
            g["weight_scale_format"] = f.into();
        }
        g
    };
    let hf = serde_json::json!({"model": {"parameter_groups": [
        group("mxfp8_nearest", "mxfp8", "float8_e4m3fn", [32, 32], None, "mx_weight"),
        group("nvfp4_nearest", "nvfp4", "uint8", [1, 16], Some("amax_denominator"), "nv_weight")]}});
    std::fs::write(dir.join("hf_quant_config.json"), hf.to_string()).unwrap();
    amax.map(|a| a / 2688.0)
}

/// Every expert slab of `name` decoded by the checkpoint codec: `[E * N, K]` f32.
fn decoded(reader: &PackedReader, name: &str) -> Vec<f32> {
    let d = reader.descriptor(name).unwrap();
    let mut buf = TileBuf::default();
    reader.read_blocks(&d, &(0..d.numel()), &mut buf).unwrap();
    let tile = Tile { elems: 0..d.numel(), payload: &buf.payload, scales: buf.scales.iter().map(Vec::as_slice).collect() };
    let mut out = Vec::new();
    CodecRegistry::reference().decoder(&d.encoding).unwrap().decode_tile(&d, &tile, &mut out).unwrap();
    out
}

fn bank(ops: &Ops, dir: &std::path::Path, name: &str) -> MicroBank {
    let views = SlicedReader::new(Box::new(PackedReader::open(dir).unwrap())).with_slab_views(name, |e| format!("{name}.{e}")).unwrap();
    let src = PackedSource::identity(Box::new(views), CodecRegistry::reference()).unwrap();
    let names: Vec<String> = (0..E).map(|e| format!("{name}.{e}")).collect();
    upload_micro_bank(ops, &src, &names, N, K, &BoundedWorkspace::new(MemoryBudget::new(1 << 20, 0))).unwrap()
}

/// The routing table and slot order the grouped product reads (what `moe_route_scan` / `moe_route_emit` write).
fn route(ids: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mr = MICRO_GROUPED_SLOTS;
    let counts: Vec<u32> = (0..E as u32).map(|e| ids.iter().filter(|&&i| i == e).count() as u32).collect();
    let (mut slots, mut tiles) = (vec![0u32], vec![0u32]);
    for c in &counts {
        slots.push(slots.last().unwrap() + c);
        tiles.push(tiles.last().unwrap() + c.div_ceil(mr));
    }
    let perm = (0..E as u32).flat_map(|e| (0..ids.len() as u32).filter(move |&s| ids[s as usize] == e)).collect();
    (slots.into_iter().chain(tiles).collect(), perm)
}

/// Whether a native kernel serves `bank`'s gather or grouped product for these slots: then the two are not one order
/// of arithmetic.
fn native_product(ops: &Ops, bank: &MicroBank, ids: &[u32], xdiv: u32) -> bool {
    let g = ops.gpu();
    let fmt = if bank.kind == MicroScaled::Nvfp4 { "nvfp4" } else { "mxfp8" };
    let slots = ids.len() as u32;
    let tiles = E as u32 + slots.div_ceil(MICRO_GROUPED_SLOTS);
    let native = |op: &str, params: &[u32]| g.kernel_index(&format!("moe_{fmt}_{op}")).and_then(|i| g.native_kernel_for(i, params)).is_some();
    native("gemv_gather", &[slots, bank.k, bank.n, xdiv]) || native("grouped", &[tiles, bank.k, bank.n, xdiv, E as u32])
}

fn products(ops: &Ops, bank: &MicroBank, x: &[f32], ids: &[u32], xdiv: u32) -> (Vec<f32>, Vec<f32>) {
    let g = ops.gpu();
    let slots = ids.len() as u32;
    let (tab, perm) = route(ids);
    let upload = |v: &[u32]| {
        let b = g.storage(v.len() as u64);
        g.write(&b, v);
        b
    };
    let (xb, idb, tb, pb) = (g.storage_init("x", x), upload(ids), upload(&tab), upload(&perm));
    let (gather, grouped) = (g.storage(slots as u64 * N as u64), g.storage(slots as u64 * N as u64));
    let tiles = E as u32 + slots.div_ceil(MICRO_GROUPED_SLOTS);
    let steps = [bank.gather(g, &xb, &idb, &gather, slots, xdiv).unwrap(), bank.grouped(g, &xb, &tb, &pb, &grouped, tiles, xdiv, E as u32).unwrap()];
    g.submit(&[], &steps);
    g.poll_wait();
    // The device-built table is the host's, word for word.
    let route = MicroRoute::new(g, slots, E as u32);
    assert_eq!(route.tiles(), tiles);
    g.submit(&[], &route.steps(g, &idb).unwrap());
    g.poll_wait();
    let words = |b: &gpu_core::DeviceBuffer, n: usize| g.read(b, n).into_iter().map(f32::to_bits).collect::<Vec<u32>>();
    assert_eq!((words(&route.tab, tab.len()), words(&route.perm, perm.len())), (tab.clone(), perm.clone()), "MicroRoute builds the grouped product's table");
    (g.read(&gather, slots as usize * N), g.read(&grouped, slots as usize * N))
}

fn check(label: &str, got: &[f32], want_w: &[f32], x: &[f32], ids: &[u32], xdiv: usize) {
    for (s, &e) in ids.iter().enumerate() {
        for c in 0..N {
            let row = (e as usize * N + c) * K;
            let terms: Vec<f64> = (0..K).map(|i| f64::from(want_w[row + i]) * f64::from(x[(s / xdiv) * K + i])).collect();
            let (want, scale) = (terms.iter().sum::<f64>(), terms.iter().map(|t| t * t).sum::<f64>().sqrt());
            let g = f64::from(got[s * N + c]);
            assert!((g - want).abs() <= 2e-5 * scale + 1e-9, "{label}: slot {s} col {c} is {g}, oracle {want}");
        }
    }
}

#[test]
fn stacked_expert_projections_load_into_banks_and_multiply_as_decoded() {
    let mut list = training_kernel_list().to_vec();
    list.extend(micro_bank_kernels());
    let ops = Ops::new(Gpu::new(Box::leak(list.into_boxed_slice()))).expect("Ops::new");
    let mut rng = Lcg::random();
    let dir = std::env::temp_dir().join(format!("micro_bank_stacked_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let globals = release(&dir, &mut rng);
    let reader = PackedReader::open(&dir).unwrap();
    let ids: Vec<u32> = vec![2, 0, 1, 2, 2, 0];
    let xdiv = 2usize;
    let x = rng.vec_scaled(ids.len().div_ceil(xdiv) * K, 1.0);

    let mx = bank(&ops, &dir, "mx_weight");
    assert_eq!((mx.kind, mx.blocks, mx.n, mx.k), (MicroScaled::Mxfp8E4m3, E as u32, N as u32, K as u32));
    let (gather, grouped) = products(&ops, &mx, &x, &ids, xdiv as u32);
    let want = decoded(&reader, "mx_weight");
    check("mxfp8 gather", &gather, &want, &x, &ids, xdiv);
    check("mxfp8 grouped", &grouped, &want, &x, &ids, xdiv);
    if !native_product(&ops, &mx, &ids, xdiv as u32) {
        assert_eq!(gather.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), grouped.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "grouped is the gather's arithmetic");
    }

    let nv = bank(&ops, &dir, "nv_weight");
    assert_eq!(nv.kind, MicroScaled::Nvfp4);
    let got: Vec<u32> = ops.gpu().read(&nv.global, E).iter().map(|v| v.to_bits()).collect();
    assert_eq!(got, globals.map(f32::to_bits), "each expert's global is its amax over the denominator, folded once");
    let (gather, grouped) = products(&ops, &nv, &x, &ids, xdiv as u32);
    let want = decoded(&reader, "nv_weight");
    check("nvfp4 gather", &gather, &want, &x, &ids, xdiv);
    check("nvfp4 grouped", &grouped, &want, &x, &ids, xdiv);
    if !native_product(&ops, &nv, &ids, xdiv as u32) {
        assert_eq!(gather.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), grouped.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "nvfp4 grouped is the gather's arithmetic");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// `x` rounded to bf16 (to nearest, ties to even) and widened back: what a bf16 tensor-core operand holds.
fn bf16(x: f32) -> f32 {
    let b = x.to_bits();
    f32::from_bits((b.wrapping_add(0x7FFF + ((b >> 16) & 1))) & 0xFFFF_0000)
}

#[test]
fn the_grouped_product_at_the_default_tier_rounds_only_the_activations() {
    let mut list = training_kernel_list().to_vec();
    list.extend(micro_bank_kernels());
    let mut ops = Ops::new(Gpu::new(Box::leak(list.into_boxed_slice()))).expect("Ops::new");
    let mut rng = Lcg::random();
    let dir = std::env::temp_dir().join(format!("micro_bank_grouped_tier_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    release(&dir, &mut rng);
    let reader = PackedReader::open(&dir).unwrap();
    // 19 slots of expert 1 (three tiles), a few of the others, interleaved.
    let ids: Vec<u32> = (0..27u32).map(|s| if s % 4 == 3 { s % 3 } else { 1 }).collect();
    let xdiv = 3usize;
    let x = rng.vec_scaled(ids.len().div_ceil(xdiv) * K, 1.0);
    let slots = ids.len() as u32;
    for (name, fused) in [("nv_weight", gpu_core::Fused::MoeNvfp4GroupedMma), ("mx_weight", gpu_core::Fused::MoeMxfp8GroupedMma)] {
        let b = bank(&ops, &dir, name);
        let want_w = decoded(&reader, name);
        let run = |ops: &Ops| {
            let g = ops.gpu();
            let idb = g.storage(slots as u64);
            g.write(&idb, &ids);
            let xb = g.storage_init("x", &x);
            let route = MicroRoute::new(g, slots, E as u32);
            let out = g.storage(slots as u64 * N as u64);
            let mut s = route.steps(g, &idb).unwrap();
            ops.micro_grouped(&mut s, &b, &xb, ids.len().div_ceil(xdiv) as u32, &route, &out, xdiv as u32).unwrap();
            g.submit(&[], &s);
            g.read(&out, slots as usize * N)
        };
        ops.set_gemm_precision(GemmPrecision::Fp32Reference);
        let pinned = run(&ops);
        check(&format!("{name} at the binary32 tier"), &pinned, &want_w, &x, &ids, xdiv);
        ops.set_gemm_precision(GemmPrecision::Auto);
        let auto = run(&ops);
        if !ops.gpu().has_fused(fused) {
            assert_eq!(auto.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), pinned.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "{name}: without the tensor-core kernel both tiers run the bank's kernel");
            brain_testutil::skip_unavailable(&format!("the tensor-core grouped product of {name} (no CUDA device offering it)"));
            continue;
        }
        let xr: Vec<f32> = x.iter().map(|&v| bf16(v)).collect();
        check(&format!("{name} at the tensor-core tier, against the bf16-rounded activations"), &auto, &want_w, &xr, &ids, xdiv);
        assert_ne!(auto, pinned, "{name}: the default tier rounds the activations");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

