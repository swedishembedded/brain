// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Convert ANY named-tensor source into a quantized GGUF, one tensor at a
//! time, under an explicit per-tensor policy.
//!
//! Swedish Embedded AB implements portable weight-quantization and
//! checkpoint-conversion pipelines for its clients. If your team needs
//! expertise in on-device model quantization then you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! # What this is
//!
//! Every quantized checkpoint in this repo used to be produced by a
//! per-model converter that decided, in its own code, which of its own
//! tensors to quantize and to what. That decision - the POLICY - is the only
//! part that is genuinely model-specific; reading a tensor, encoding a block,
//! and writing a GGUF are not. This module owns the generic three and takes
//! the policy as data, so converting a new architecture is a [`Policy`] and a
//! KV list, not a new converter.
//!
//! The pieces it composes already existed and are not re-implemented here:
//! [`crate::TensorSource`] for reading (so safetensors, an existing GGUF and
//! a plain in-memory map all work with no format-specific code),
//! [`crate::quant::quantize_par`] for the block encoding, and
//! [`crate::gguf_write::Writer`] for the container.
//!
//! # Bounded memory
//!
//! [`convert`] never holds more than one tensor. The GGUF header commits to
//! every tensor's byte offset up front, but those offsets are computable from
//! the shapes and the target type alone - so the whole file is PLANNED first
//! ([`plan`]) and the bodies are then streamed in plan order. A 26 GB source
//! converts in a working set of one tensor, not of the output.
//!
//! # The policy, and what is NOT negotiable
//!
//! Two of the three rules are structural - they come from the block format
//! itself, not from taste, and are therefore fixed:
//!
//! * **Rank 2 only.** A quantized GEMM operand is a matrix. A rank-1 tensor
//!   is a bias, a norm gain or a scalar table, and a rank-3+ tensor is a
//!   convolution kernel or a positional table - none of which any int8 GEMM
//!   in this workspace consumes, and all of which are small enough that
//!   quantizing them trades real precision for no meaningful bytes.
//! * **The fastest-varying dimension must be a whole number of blocks.** A
//!   ggml block carries ONE scale for its 32 (or 256) contiguous elements. In
//!   row-major storage a block that straddles a row boundary would share one
//!   scale across two different output channels, so a row length that is not
//!   a block multiple is not quantizable at this type at all.
//!
//! The third is the caller's, and is why this takes a [`Policy`] rather than
//! a bool: which NAMED tensors a given architecture must never quantize
//! regardless of shape. `ltxv::int8::is_never_quantized` is the worked
//! example - modulation/conditioning tables and the first/last projections,
//! which set a diffusion transformer's numeric scale and whose precision the
//! whole per-token combine rides on. That list is knowledge about one model
//! and cannot be derived from a tensor's shape, so it is supplied, never
//! guessed.
//!
//! Nothing is ever silently dropped or silently quantized: [`plan`] returns a
//! row for EVERY tensor in the source with the reason for its fate, and
//! [`convert`] writes every one of those rows. That is the same two-way
//! coverage discipline `ltxv::import`'s `validate_manifest` applies to an
//! import, applied to an export.

use std::collections::BTreeSet;

use crate::gguf::GgufValue;
use crate::gguf_write::{TensorPlan, Writer};

/// The on-disk tier a conversion targets.
///
/// The planner, the encoder (`crate::quant::quantize`/`quantize_par`) and the
/// writer are generic over block geometry, so a variant here is a row in each
/// accessor, not new machinery. `Q4K`/`Q5K`/`Q6K` drop the `_` before `K`
/// solely to satisfy `non_camel_case_types`, matching `GgmlType`'s naming
/// (see its doc); [`Tier::name`] still spells them the conventional way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// 32 elements per block, one f16 scale + 32 signed bytes (34 B/block,
    /// 1.0625 B/weight). The tier brain's own int8 GEMM path is fed from.
    Q8_0,
    /// 32 elements per block, one f16 scale + 16 packed nibbles (18 B/block,
    /// 4.5 bits/weight). Legacy symmetric-with-bias-fold format.
    Q4_0,
    /// 32 elements per block, one f16 scale + 32 packed 5-bit codes
    /// (22 B/block, 5.5 bits/weight).
    Q5_0,
    /// 256-element super-block, affine (`d*q - dmin*m`) with 8 sub-blocks of
    /// 32 sharing packed 6-bit scale/min pairs (144 B/block, 4.5 bits/weight).
    Q4K,
    /// 256-element super-block, affine, 8 sub-blocks of 32
    /// (176 B/block, 5.5 bits/weight).
    Q5K,
    /// 256-element super-block, symmetric, 16 sub-blocks of 16
    /// (210 B/block, 6.5625 bits/weight).
    Q6K,
}

impl Tier {
    /// The ggml type id written into the tensor info.
    pub fn ggml_type(self) -> u32 {
        match self {
            Tier::Q8_0 => crate::gguf::T_Q8_0,
            Tier::Q4_0 => crate::gguf::T_Q4_0,
            Tier::Q5_0 => crate::gguf::T_Q5_0,
            Tier::Q4K => crate::gguf::T_Q4_K,
            Tier::Q5K => crate::gguf::T_Q5_K,
            Tier::Q6K => crate::gguf::T_Q6_K,
        }
    }

    /// Elements per block - also the divisor the fastest-varying dimension
    /// must satisfy.
    pub fn block_elems(self) -> usize {
        match self {
            Tier::Q8_0 | Tier::Q4_0 | Tier::Q5_0 => 32,
            Tier::Q4K | Tier::Q5K | Tier::Q6K => crate::gguf::QK_K,
        }
    }

    /// The name this tier appears under in a GGUF's type strings.
    pub fn name(self) -> &'static str {
        match self {
            Tier::Q8_0 => "Q8_0",
            Tier::Q4_0 => "Q4_0",
            Tier::Q5_0 => "Q5_0",
            Tier::Q4K => "Q4_K",
            Tier::Q5K => "Q5_K",
            Tier::Q6K => "Q6_K",
        }
    }
}

/// Why a tensor was written through unquantized. Every variant is a distinct,
/// checkable reason - "it was skipped" with no reason is what lets a real
/// weight matrix silently miss the quantized path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kept {
    /// Not a matrix (rank != 2): a bias, a norm gain, a scalar, a
    /// convolution kernel, a positional table.
    NotRank2 { rank: usize },
    /// The fastest-varying dimension is not a whole number of blocks, so no
    /// block can be encoded without straddling a row boundary.
    RowNotBlockAligned { row: usize, block: usize },
    /// Below the policy's minimum element count.
    TooSmall { numel: usize, min: usize },
    /// Matched one of the policy's never-quantize name patterns.
    NeverQuantize { pattern: String },
}

/// What [`plan`] decided for one tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    Quantize,
    Keep(Kept),
}

/// The per-tensor rules a given architecture needs on top of the structural
/// ones. Built with the chaining setters so a caller reads as a declaration.
#[derive(Clone, Debug, Default)]
pub struct Policy {
    never_quantize: Vec<String>,
    min_elems: usize,
}

impl Policy {
    /// Structural rules only: every rank-2, block-aligned tensor is
    /// quantized. Correct for a source whose every matrix is a GEMM operand,
    /// and the honest starting point for one that is not - add the model's
    /// own exclusions rather than inheriting a guess.
    pub fn new() -> Policy {
        Policy::default()
    }

    /// Never quantize a tensor whose name CONTAINS any of `patterns`.
    /// Substring matching, matching the convention `ltxv::int8::
    /// is_never_quantized` established: one pattern covers a tensor and every
    /// per-layer/per-stream variant of it without enumerating indices.
    pub fn never_quantize(mut self, patterns: &[&str]) -> Policy {
        self.never_quantize.extend(patterns.iter().map(|p| p.to_string()));
        self
    }

    /// Keep any tensor with fewer than `min` elements. Off (`0`) by default,
    /// deliberately: there is no size below which quantizing is universally
    /// wrong, so a threshold here is a claim about ONE model's tensors and
    /// belongs to whoever can justify it for that model.
    pub fn min_elems(mut self, min: usize) -> Policy {
        self.min_elems = min;
        self
    }

    /// Decide one tensor's fate. Structural rules first (they are about
    /// whether the encoding is possible at all), then the caller's.
    pub fn decide(&self, name: &str, shape: &[usize], tier: Tier) -> Decision {
        if shape.len() != 2 {
            return Decision::Keep(Kept::NotRank2 { rank: shape.len() });
        }
        let block = tier.block_elems();
        let row = shape[1];
        if !row.is_multiple_of(block) {
            return Decision::Keep(Kept::RowNotBlockAligned { row, block });
        }
        if let Some(p) = self.never_quantize.iter().find(|p| name.contains(p.as_str())) {
            return Decision::Keep(Kept::NeverQuantize { pattern: p.clone() });
        }
        let numel: usize = shape.iter().product();
        if numel < self.min_elems {
            return Decision::Keep(Kept::TooSmall { numel, min: self.min_elems });
        }
        Decision::Quantize
    }
}

/// A source that can enumerate itself. [`crate::TensorSource`] answers "give
/// me this tensor" - a converter also has to ask "what tensors are there, and
/// what shape", which is the one thing every implementor already knows and
/// the trait had no way to say.
///
/// Kept as a separate trait rather than defaulted methods on `TensorSource`:
/// a default would let an implementor silently report an empty checkpoint,
/// and a converter cannot tell that apart from a genuinely empty one.
pub trait TensorManifest: crate::TensorSource {
    /// Every tensor name, in whatever order the source considers canonical
    /// (a GGUF's declared order, a safetensors header's key order). The
    /// output preserves it.
    fn tensor_names(&self) -> Vec<String>;

    /// `name`'s shape in torch order (outermost first), or `None` if absent.
    fn tensor_shape(&self, name: &str) -> Option<Vec<usize>>;
}

#[cfg(not(target_arch = "wasm32"))]
impl TensorManifest for crate::weightio::WeightReader {
    fn tensor_names(&self) -> Vec<String> {
        self.names().map(|n| n.to_string()).collect()
    }
    fn tensor_shape(&self, name: &str) -> Option<Vec<usize>> {
        self.shape(name).map(|s| s.iter().map(|&d| d as usize).collect())
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl TensorManifest for crate::gguf::MmapGguf {
    fn tensor_names(&self) -> Vec<String> {
        self.names().to_vec()
    }
    fn tensor_shape(&self, name: &str) -> Option<Vec<usize>> {
        self.shape(name).map(|s| s.to_vec())
    }
}

impl TensorManifest for std::collections::HashMap<String, (Vec<usize>, Vec<f32>)> {
    fn tensor_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.keys().cloned().collect();
        // A `HashMap` has no canonical order and a GGUF's tensor order is
        // part of its bytes, so sort: the same map must convert to the same
        // file.
        names.sort();
        names
    }
    fn tensor_shape(&self, name: &str) -> Option<Vec<usize>> {
        self.get(name).map(|(s, _)| s.clone())
    }
}

/// One planned tensor: what it is, what was decided, and exactly how many
/// output bytes that costs.
#[derive(Clone, Debug)]
pub struct Row {
    pub name: String,
    pub shape: Vec<usize>,
    pub numel: usize,
    pub decision: Decision,
    /// The ggml type this row is written as.
    pub ty: u32,
    /// Encoded length of this row's body in the output.
    pub nbytes: usize,
}

impl Row {
    /// Whether this row goes through the quantizer.
    pub fn quantized(&self) -> bool {
        self.decision == Decision::Quantize
    }
}

/// What a conversion did, in enough detail to assert on.
#[derive(Clone, Debug)]
pub struct Report {
    pub tier: Tier,
    pub rows: Vec<Row>,
}

impl Report {
    pub fn quantized(&self) -> usize {
        self.rows.iter().filter(|r| r.quantized()).count()
    }
    pub fn kept(&self) -> usize {
        self.rows.len() - self.quantized()
    }
    /// Total output body bytes (excluding the header and inter-tensor
    /// padding, both negligible against a real checkpoint).
    pub fn output_bytes(&self) -> u64 {
        self.rows.iter().map(|r| r.nbytes as u64).sum()
    }
    /// What the same tensors would cost written as plain f32 - the
    /// denominator for an honest compression ratio, since the SOURCE's own
    /// size depends on a dtype (bf16, fp8) this module never sees.
    pub fn f32_bytes(&self) -> u64 {
        self.rows.iter().map(|r| r.numel as u64 * 4).sum()
    }
    /// Parameters that went through the quantizer.
    pub fn quantized_params(&self) -> u64 {
        self.rows.iter().filter(|r| r.quantized()).map(|r| r.numel as u64).sum()
    }
}

/// Plan the whole output: one [`Row`] per source tensor, in the source's own
/// order, with no tensor omitted for any reason. Errors if a name the source
/// enumerates has no shape (a source that cannot describe its own contents),
/// or if two names collide.
pub fn plan(src: &dyn TensorManifest, tier: Tier, policy: &Policy) -> Result<Vec<Row>, String> {
    let names = src.tensor_names();
    let mut seen = BTreeSet::new();
    let mut rows = Vec::with_capacity(names.len());
    for name in names {
        if !seen.insert(name.clone()) {
            return Err(format!("quantize: source enumerates '{name}' twice"));
        }
        let shape = src.tensor_shape(&name).ok_or_else(|| format!("quantize: source enumerates '{name}' but has no shape for it"))?;
        let numel: usize = shape.iter().product();
        let decision = policy.decide(&name, &shape, tier);
        let (ty, nbytes) = match decision {
            Decision::Quantize => {
                let (be, bb) = (tier.block_elems(), crate::gguf::block_geometry(tier.ggml_type()).expect("tier has a block geometry").1);
                (tier.ggml_type(), numel / be * bb)
            }
            // Kept tensors are written as f32. The source's own dtype is not
            // preserved: `TensorSource` hands out f32 and nothing else, so
            // "keep the original dtype" would mean re-encoding a decoded
            // value and claiming it was untouched. f32 is what was actually
            // read, and it round-trips bit-exactly.
            Decision::Keep(_) => (crate::gguf::T_F32, numel * 4),
        };
        rows.push(Row { name, shape, numel, decision, ty, nbytes });
    }
    Ok(rows)
}

/// Convert `src` to a GGUF at `out_path` under `policy`, writing `kv` as the
/// file's metadata. Streams: one tensor is resident at a time.
///
/// `on_tensor` is called after each tensor is written, with its index and its
/// row - a converter over a multi-gigabyte checkpoint that reports nothing
/// for twenty minutes is indistinguishable from a hung one.
///
/// Errors name the tensor. A source that cannot produce a tensor it
/// enumerated, or produces the wrong number of elements for the shape it
/// declared, is a hard error - never a zero fill.
#[cfg(not(target_arch = "wasm32"))]
pub fn convert(
    src: &dyn TensorManifest,
    tier: Tier,
    policy: &Policy,
    kv: &[(String, GgufValue)],
    out_path: &str,
    on_tensor: &mut dyn FnMut(usize, &Row),
) -> Result<Report, String> {
    let rows = plan(src, tier, policy)?;
    let file_plan: Vec<TensorPlan> =
        rows.iter().map(|r| TensorPlan { name: r.name.clone(), shape: r.shape.clone(), ty: r.ty, nbytes: r.nbytes }).collect();
    let mut w = Writer::create(out_path, kv, file_plan, 32).map_err(|e| format!("quantize: creating {out_path}: {e}"))?;

    for (i, row) in rows.iter().enumerate() {
        let mut encoded: Option<Result<Vec<u8>, String>> = None;
        let found = src.with_tensor(&row.name, &mut |data| {
            if data.len() != row.numel {
                encoded = Some(Err(format!("quantize: '{}' declared {} elements, source produced {}", row.name, row.numel, data.len())));
                return;
            }
            encoded = Some(match row.decision {
                Decision::Quantize => crate::quant::quantize_par(tier.ggml_type(), data).map_err(|e| format!("quantize: '{}': {e}", row.name)),
                Decision::Keep(_) => Ok(data.iter().flat_map(|v| v.to_le_bytes()).collect()),
            });
        });
        if !found {
            return Err(format!("quantize: source enumerated '{}' but cannot produce it", row.name));
        }
        let bytes = encoded.expect("with_tensor reported found, so the callback ran")?;
        w.write_tensor(&row.name, &bytes).map_err(|e| format!("quantize: writing '{}': {e}", row.name))?;
        drop(bytes);
        on_tensor(i, row);
    }

    w.finish().map_err(|e| format!("quantize: finishing {out_path}: {e}"))?;
    Ok(Report { tier, rows })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structural_rules_are_not_negotiable_and_the_name_list_is() {
        let p = Policy::new().never_quantize(&["scale_shift_table"]);
        assert_eq!(p.decide("mlp.weight", &[8, 64], Tier::Q8_0), Decision::Quantize);
        assert_eq!(p.decide("norm.weight", &[64], Tier::Q8_0), Decision::Keep(Kept::NotRank2 { rank: 1 }));
        assert_eq!(p.decide("pos", &[2, 4, 64], Tier::Q8_0), Decision::Keep(Kept::NotRank2 { rank: 3 }));
        assert_eq!(p.decide("odd.weight", &[8, 20], Tier::Q8_0), Decision::Keep(Kept::RowNotBlockAligned { row: 20, block: 32 }));
        assert_eq!(
            p.decide("blocks.0.scale_shift_table", &[8, 64], Tier::Q8_0),
            Decision::Keep(Kept::NeverQuantize { pattern: "scale_shift_table".to_string() })
        );
        let small = Policy::new().min_elems(1024);
        assert_eq!(small.decide("tiny.weight", &[2, 32], Tier::Q8_0), Decision::Keep(Kept::TooSmall { numel: 64, min: 1024 }));
    }

    /// Every K-quant/legacy tier this milestone adds has geometry consistent
    /// with `GgmlType`'s own table, and Q4_K in particular actually runs
    /// end-to-end through `plan`/`convert`/`MmapGguf` unchanged - the
    /// planner, encoder (`crate::quant::quantize_par`) and writer are
    /// GENERIC over block geometry, so a new `Tier` variant is a row in each
    /// accessor, not new machinery, and this is the proof of that rather than
    /// an assumption of it.
    #[test]
    fn every_new_tier_has_consistent_geometry_and_q4k_round_trips() {
        use std::collections::HashMap;

        for (tier, want_id, want_name, want_elems) in [
            (Tier::Q4_0, crate::gguf::T_Q4_0, "Q4_0", 32),
            (Tier::Q5_0, crate::gguf::T_Q5_0, "Q5_0", 32),
            (Tier::Q4K, crate::gguf::T_Q4_K, "Q4_K", 256),
            (Tier::Q5K, crate::gguf::T_Q5_K, "Q5_K", 256),
            (Tier::Q6K, crate::gguf::T_Q6_K, "Q6_K", 256),
        ] {
            assert_eq!(tier.ggml_type(), want_id, "{want_name} ggml_type");
            assert_eq!(tier.name(), want_name);
            assert_eq!(tier.block_elems(), want_elems, "{want_name} block_elems");
        }

        // k = 512 = two 256-element super-blocks, so a real k-loop with two
        // independently-scaled super-blocks runs, not a degenerate single one.
        let (n_rows, k) = (2usize, 512usize);
        let data: Vec<f32> = (0..n_rows * k)
            .map(|i| ((i as i64).wrapping_mul(2_654_435_761) % 2003 - 1001) as f32 / 97.0)
            .collect();
        let mut src: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
        src.insert("w".to_string(), (vec![n_rows, k], data.clone()));

        let path = std::env::temp_dir().join(format!("brain-quantize-tier-q4k-{}.gguf", std::process::id()));
        let path = path.to_str().unwrap().to_string();
        let report = convert(&src, Tier::Q4K, &Policy::new(), &[], &path, &mut |_, _| {}).unwrap();
        assert_eq!(report.quantized(), 1);
        assert_eq!(report.rows[0].ty, crate::gguf::T_Q4_K);

        let mg = crate::gguf::MmapGguf::open(&path).unwrap();
        assert_eq!(mg.dtype("w"), Some("Q4_K"));
        let got = mg.tensor("w").unwrap().unwrap();
        assert_eq!(got.len(), data.len());

        // Cosine AND rel_l2 - lesson #2: cosine alone cannot see a dropped
        // scale factor, so a quantized-quality gate must assert both.
        let dot: f64 = data.iter().zip(&got).map(|(&a, &b)| a as f64 * b as f64).sum();
        let na: f64 = data.iter().map(|&a| (a as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = got.iter().map(|&b| (b as f64).powi(2)).sum::<f64>().sqrt();
        let cosine = dot / (na * nb);
        let err: f64 = data.iter().zip(&got).map(|(&a, &b)| (a as f64 - b as f64).powi(2)).sum::<f64>().sqrt();
        let rel_l2 = err / na;
        // A loose floor - this proves the PIPELINE wires a K-quant tier
        // through correctly, not a claim about Q4_K's numeric quality (that
        // is `crate::quant`'s own `every_type_meets_its_quality_floor`, and
        // the M8 relayout gate's job, not this one's).
        assert!(cosine > 0.99, "cosine {cosine} too low for a working Q4_K round trip");
        assert!(rel_l2 < 0.15, "rel_l2 {rel_l2} too high for a working Q4_K round trip");

        std::fs::remove_file(&path).ok();
    }
}
