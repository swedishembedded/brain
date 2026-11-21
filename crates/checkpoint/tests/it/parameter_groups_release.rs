// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements serving of mixed-precision mixture-of-experts releases exactly as published for its
// clients. If your team needs expertise in NVFP4 and MXFP8 checkpoints then you can procure our services by sending an
// email to info@swedishembedded.com.

//! A release that declares every tensor in `hf_quant_config.json`'s `parameter_groups` (a cast group per float dtype,
//! NVFP4 experts whose second scale level is `amax / outer_scale_denominator` per expert, MXFP8 experts with an E8M0
//! scale stored as `U8`) reads as exactly what it declares, against the real published header: the global
//! `quantization_config` in `config.json` (which names one algorithm for everything) does not apply, sidecars are
//! not parameters, and a stored dtype or scale shape that contradicts a declaration is an error.

use checkpoint::packed::{FloatFormat, PackedReader, PackedTensorReader, QuantEncoding, ScaleGranularity};

const REPO: &str = "thinkingmachines/Inkling-2-Mini";

fn published() -> Option<std::path::PathBuf> {
    let dir = brain_testutil::model_dir(REPO).map(std::path::PathBuf::from).filter(|d| d.join("hf_quant_config.json").is_file());
    if dir.is_none() {
        brain_testutil::skip_unavailable(&format!("{REPO} is not in the model store"));
    }
    dir
}

/// Spec: each tensor of the published checkpoint has the encoding its parameter group declares, its quantization
/// sidecars are not listed, and every listed tensor has a readable descriptor whose sizes match the stored bytes.
#[test]
fn a_parameter_groups_release_reads_as_declared_per_tensor() {
    let Some(dir) = published() else { return };
    let r = PackedReader::open(&dir).unwrap();
    let names = r.names();
    for suffix in ["_scale", "_amax", "_outer_scale_denominator"] {
        assert!(!names.iter().any(|n| n.ends_with(suffix)), "a {suffix} sidecar is listed as a parameter");
    }
    for n in &names {
        r.descriptor(n).unwrap_or_else(|e| panic!("{n}: {e}"));
    }

    let nvfp4 = r.descriptor("model.llm.layers.0.mlp.experts.w13_weight").unwrap();
    assert_eq!(nvfp4.shape, [128, 2304, 2304], "two E2M1 codes per stored byte");
    assert_eq!(nvfp4.encoding, QuantEncoding::nvfp4_per_slab());
    let w2 = r.descriptor("model.llm.layers.19.mlp.experts.w2_weight").unwrap();
    assert_eq!((w2.shape.clone(), w2.encoding.clone()), (vec![128, 2304, 1152], QuantEncoding::nvfp4_per_slab()));

    let mxfp8 = r.descriptor("model.llm.layers.20.mlp.experts.w13_weight").unwrap();
    assert_eq!((mxfp8.shape.clone(), mxfp8.encoding.clone()), (vec![128, 2304, 2304], QuantEncoding::mxfp8()));
    assert_eq!(mxfp8.encoding.scales[0].granularity, ScaleGranularity::Block(32));

    assert_eq!(r.descriptor("model.llm.layers.0.mlp.gate.bias").unwrap().encoding, QuantEncoding::f32());
    assert_eq!(r.descriptor("model.llm.attn_res.output_query").unwrap().encoding, QuantEncoding::f32());
    assert_eq!(r.descriptor("model.llm.embed.weight").unwrap().encoding, QuantEncoding::bf16());
    assert_eq!(r.descriptor("model.llm.unembed.weight").unwrap().encoding.element, checkpoint::packed::Element::Float(FloatFormat::Bf16));

    // Each expert of a stacked tensor is a matrix of its own in the single-expert encoding (rows as stored).
    let views = checkpoint::packed::SlicedReader::new(Box::new(r))
        .with_slab_views("model.llm.layers.0.mlp.experts.w13_weight", |e| format!("l0.w13.{e}"))
        .and_then(|v| v.with_slab_views("model.llm.layers.20.mlp.experts.w2_weight", |e| format!("l20.w2.{e}")))
        .unwrap();
    let e = views.descriptor("l0.w13.127").unwrap();
    assert_eq!((e.shape.clone(), e.encoding.clone()), (vec![2304, 2304], QuantEncoding::nvfp4()));
    let e = views.descriptor("l20.w2.0").unwrap();
    assert_eq!((e.shape.clone(), e.encoding.clone()), (vec![2304, 1152], QuantEncoding::mxfp8()));
}

/// Spec: the declarations themselves are parsed from the release's own sidecar: the transform of each group, the
/// forward output dtype a group declares, and the activation quantization the vendor's runtime used (metadata only).
#[test]
fn the_published_sidecar_declares_each_tensor_once() {
    use checkpoint::packed::{DeclaredWeight, HfQuant, Nvfp4Global};
    let Some(dir) = published() else { return };
    let q = HfQuant::from_dir(&dir).unwrap().expect("the release declares its quantization");
    assert_eq!(q.declared("model.llm.layers.3.mlp.experts.w2_weight"), DeclaredWeight::Nvfp4 { group: 16, global: Nvfp4Global::AmaxOverDenominator });
    assert_eq!(q.declared("model.llm.layers.23.mlp.experts.w13_weight"), DeclaredWeight::Mxfp8 { block: 32 });
    assert_eq!(q.declared("model.llm.embed.weight"), DeclaredWeight::Cast(FloatFormat::Bf16));
    assert_eq!(q.declared("model.llm.middle_cycle_cross_pass_queries"), DeclaredWeight::Cast(FloatFormat::F32));
    assert_eq!(q.declared("fuse_weight"), DeclaredWeight::Cast(FloatFormat::Bf16), "the mtp section declares the drafter's unprefixed names");
    assert_eq!(q.declared("model.llm.not_a_tensor"), DeclaredWeight::Undeclared);
    let unembed = q.declaration("model.llm.unembed.weight").unwrap();
    assert_eq!(unembed.forward_output, Some(FloatFormat::F32));
    let act = q.declaration("model.llm.layers.0.mlp.experts.w13_weight").unwrap().activation.clone().expect("NVFP4 experts declare activation quantization");
    assert_eq!((act.format.as_str(), act.block_size), ("nvfp4_e2m1", 16));
    assert_eq!(q.declared_tensors().count(), 74 + 993 + 40 + 8 + 1 + 54);
}

/// Spec: decoding one expert slab of the published checkpoint (an NVFP4 layer and an MXFP8 layer, both projections)
/// gives, bit for bit in f32, what the reference loader's dequantization gives (`Inkling2LoopDequantizeExperts`,
/// dumped by the tool named below): `e2m1 * e4m3 * (amax / denominator)` and `e4m3 * 2^(e8m0 - 127)`.
#[test]
fn a_decoded_expert_slab_equals_the_reference_dequantization_bit_for_bit() {
    use checkpoint::packed::{CodecRegistry, Tile, TileBuf};
    const DUMPER: &str = "tools/goldens/inkling2_expert_dequant_reference.py";
    let Some(dir) = published() else { return };
    let golden_dir = brain_testutil::testdata_path("golden/inkling2/experts");
    let Some(src) = brain_testutil::golden::Source::open(&golden_dir, DUMPER) else { return };
    if !src.require(&[("hidden_size", 2304), ("intermediate_size", 1152), ("n_routed_experts", 128)]) {
        return;
    }
    let golden = checkpoint::weightio::WeightReader::open(golden_dir.join("experts.safetensors").to_str().unwrap()).unwrap();
    let r = PackedReader::open(&dir).unwrap();
    let reg = CodecRegistry::reference();
    let mut seen = Vec::new();
    for key in golden.names().map(str::to_string).collect::<Vec<_>>() {
        let (format, rest) = key.split_once('.').unwrap();
        let (tensor, expert) = rest.rsplit_once('.').unwrap();
        let expert: u64 = expert.parse().unwrap();
        let want = golden.tensor(&key).unwrap();
        let d = r.descriptor(tensor).unwrap();
        let slab = d.slab_numel();
        let elems = expert * slab..(expert + 1) * slab;
        let mut buf = TileBuf::default();
        r.read_blocks(&d, &elems, &mut buf).unwrap();
        let tile = Tile { elems, payload: &buf.payload, scales: buf.scales.iter().map(Vec::as_slice).collect() };
        let mut got = Vec::new();
        reg.decoder(&d.encoding).unwrap().decode_tile(&d, &tile, &mut got).unwrap();
        assert_eq!(got.len(), want.len(), "{key}");
        let diff = got.iter().zip(want.iter()).filter(|(g, w)| g.to_bits() != w.to_bits()).count();
        assert_eq!(diff, 0, "{key}: {diff} of {} values differ from the reference bits", want.len());
        seen.push(format.to_string());
    }
    seen.sort();
    seen.dedup();
    assert_eq!(seen, ["mxfp8", "nvfp4"], "the golden covers both expert formats");
}
