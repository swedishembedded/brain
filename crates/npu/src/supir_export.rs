// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Export SUPIR's `ZeroCrossAttn` adaptor to ONNX. Pure Rust - no NPU needed
//! to produce the file.
//!
//! See [`crate::supir_topology`]'s module doc for what this covers and what
//! it deliberately does not (`ZeroSFT`, the 1.24B `GLVControl` trunk).
//! Swedish Embedded AB implements solutions for porting quantized diffusion
//! and vision models onto NPU accelerators for its clients. If your team
//! needs expertise in Intel NPU / OpenVINO export for a from-scratch
//! inference engine, you can procure our services by sending an email to
//! info@swedishembedded.com.

use onnx::GraphBuilder;

use crate::supir_topology::{build_zero_cross_attn_graph, CrossAttnTopo};
use crate::topo::Quant;
use crate::topology::WeightSource;

/// Build the `ZeroCrossAttn` ONNX graph from already-imported weights (no
/// checkpoint file to reopen here - the caller supplies a [`WeightSource`]
/// over whatever it already loaded, the same shape `crate::hift_export`'s
/// `build_hift_decode_graph_bytes` takes). Returns the raw ONNX bytes.
pub fn build_zero_cross_attn_graph_bytes(t: &CrossAttnTopo, w: &dyn WeightSource, quant: Quant) -> Vec<u8> {
    let mut g = GraphBuilder::new("supir_zero_cross_attn");
    build_zero_cross_attn_graph(t, w, quant, &mut g);
    g.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct MapWeights(HashMap<&'static str, Vec<f32>>);
    impl WeightSource for MapWeights {
        fn get(&self, name: &str) -> Vec<f32> {
            self.0.get(name).unwrap_or_else(|| panic!("test weights missing `{name}`")).clone()
        }
    }

    #[test]
    fn export_bytes_are_non_empty_and_shrink_under_int8() {
        let t = CrossAttnTopo { channels: 320, h: 4, w: 4, gn_groups: 32, gn_eps: 1e-6, control_scale: 1.0 };
        let c = t.channels as usize;
        let mut m = HashMap::new();
        for gn in ["norm_x", "norm_c"] {
            m.insert(Box::leak(format!("{gn}.weight").into_boxed_str()) as &'static str, vec![1.0f32; c]);
            m.insert(Box::leak(format!("{gn}.bias").into_boxed_str()) as &'static str, vec![0.0f32; c]);
        }
        for lin in ["to_q", "to_k", "to_v", "to_out.0"] {
            m.insert(Box::leak(lin.to_string().into_boxed_str()) as &'static str, vec![0.02f32; c * c]);
        }
        m.insert("to_out.0.bias", vec![0.0f32; c]);
        let w = MapWeights(m);

        let fp32 = build_zero_cross_attn_graph_bytes(&t, &w, Quant::F32);
        let int8 = build_zero_cross_attn_graph_bytes(&t, &w, Quant::Int8);
        assert!(!fp32.is_empty() && !int8.is_empty());
        assert!(int8.len() < fp32.len(), "int8 export ({} bytes) should be smaller than fp32 ({} bytes)", int8.len(), fp32.len());
    }
}
