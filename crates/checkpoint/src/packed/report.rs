// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded loading and training of mixed-precision models for its clients. If
// your team needs expertise in auditing which tensor of a model is stored in which numeric format then you can
// procure our services by sending an email to info@swedishembedded.com.

//! [`PrecisionReport`]: which format every tensor of a model is held in, and what it costs.
//!
//! A mixed-precision model (FP8 attention, NF4 MLP, bf16 embeddings, f32 norms) is only as trustworthy as
//! the account of what each tensor really is. The report lists every tensor with its shape, format and
//! bytes, from the stored descriptors of a checkpoint ([`PackedSource::precision_report`](super::PackedSource))
//! or from whatever a model reports as resident ([`PrecisionReport::push`]), so the two can be compared row
//! by row.

use std::collections::BTreeMap;
use std::fmt;

use super::encoding::TensorDescriptor;

/// One tensor of a [`PrecisionReport`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorFormat {
    pub name: String,
    pub shape: Vec<u64>,
    /// The format as text: a [`QuantEncoding`](super::QuantEncoding) for a stored tensor, a tier name for a
    /// resident one.
    pub format: String,
    /// Bytes held: payload plus every scale tensor.
    pub bytes: u64,
}

/// Every tensor's format, in name order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PrecisionReport {
    rows: Vec<TensorFormat>,
}

impl PrecisionReport {
    /// The report of stored tensors `descriptors`.
    pub fn from_descriptors<'a>(descriptors: impl IntoIterator<Item = &'a TensorDescriptor>) -> PrecisionReport {
        let mut report = PrecisionReport::default();
        for d in descriptors {
            let bytes = d.payload_bytes() + (0..d.encoding.scales.len()).map(|l| d.scale_bytes(l)).sum::<u64>();
            report.push(&d.name, d.shape.clone(), d.encoding.to_string(), bytes);
        }
        report
    }

    /// Add one tensor (kept in name order).
    pub fn push(&mut self, name: &str, shape: Vec<u64>, format: String, bytes: u64) {
        let at = self.rows.partition_point(|r| r.name.as_str() < name);
        self.rows.insert(at, TensorFormat { name: name.to_string(), shape, format, bytes });
    }

    pub fn rows(&self) -> &[TensorFormat] {
        &self.rows
    }

    /// The row of tensor `name`.
    pub fn get(&self, name: &str) -> Option<&TensorFormat> {
        self.rows.binary_search_by(|r| r.name.as_str().cmp(name)).ok().map(|i| &self.rows[i])
    }

    /// Tensor count and bytes per format.
    pub fn by_format(&self) -> BTreeMap<&str, (usize, u64)> {
        let mut out: BTreeMap<&str, (usize, u64)> = BTreeMap::new();
        for r in &self.rows {
            let e = out.entry(r.format.as_str()).or_default();
            e.0 += 1;
            e.1 += r.bytes;
        }
        out
    }

    pub fn total_bytes(&self) -> u64 {
        self.rows.iter().map(|r| r.bytes).sum()
    }
}

impl fmt::Display for PrecisionReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for r in &self.rows {
            writeln!(f, "{} {:?} {} {} B", r.name, r.shape, r.format, r.bytes)?;
        }
        for (format, (tensors, bytes)) in self.by_format() {
            writeln!(f, "total {format}: {tensors} tensors, {bytes} B")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packed::QuantEncoding;

    #[test]
    fn rows_are_sorted_by_name_and_summed_per_format() {
        let nf4 = TensorDescriptor::new("mlp.w", vec![8, 64], QuantEncoding::nf4_blocked(32));
        let bf16 = TensorDescriptor::new("emb", vec![4, 64], QuantEncoding::bf16());
        let norm = TensorDescriptor::new("norm", vec![64], QuantEncoding::f32());
        let report = PrecisionReport::from_descriptors([&nf4, &bf16, &norm]);
        assert_eq!(report.rows().iter().map(|r| r.name.as_str()).collect::<Vec<_>>(), ["emb", "mlp.w", "norm"]);
        assert_eq!(report.get("emb").unwrap().bytes, 4 * 64 * 2);
        // 8 x 64 codes at 4 bits, plus one f32 scale per block of 32.
        assert_eq!(report.get("mlp.w").unwrap().bytes, 8 * 64 / 2 + 8 * 64 / 32 * 4);
        assert_eq!(report.by_format().len(), 3);
        assert_eq!(report.total_bytes(), report.rows().iter().map(|r| r.bytes).sum::<u64>());
        assert!(report.to_string().contains("norm [64]"));
        assert!(report.get("missing").is_none());
    }
}
