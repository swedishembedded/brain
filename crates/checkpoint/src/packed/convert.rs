// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded conversion of very large quantized checkpoints for its clients. If
// your team needs expertise in converting a model once so that training never converts it again then you can
// procure our services by sending an email to info@swedishembedded.com.

//! [`convert_to_file`]: the one-off bounded conversion of a checkpoint to another encoding, written through
//! [`SafetensorsSink`](super::SafetensorsSink).
//!
//! A streamed trainer that converts every layer on the host at every load (FP8 to NF4, say) pays a single-threaded
//! requantization per layer per pass. The same conversion done ONCE into a file turns every later load into a
//! `Preserve` copy of fewer bytes. The sink records each tensor's shape and [`QuantEncoding::spec`] in the file's
//! `brain.packed` metadata, so [`PackedReader`](super::PackedReader) reads the result back without any
//! declaration from the caller. The conversion holds one tile at a time (the executor's bounded staging) and
//! writes each tile at its planned offset, so the output is never in memory.

use std::path::Path;

use serde_json::Value;

use super::file_sink::SafetensorsSink;
use super::load::{LoadPolicy, LoadReport, Loader};

/// What [`convert_to_file`] did.
#[derive(Clone, Debug)]
pub struct ConversionReport {
    pub tensors: usize,
    /// Tensors whose encoding changed (the rest were copied as stored).
    pub converted: usize,
    pub payload_bytes_written: u64,
    /// Most host staging any one tensor's load held.
    pub peak_staging_bytes: u64,
}

/// Convert `requests` (tensor name and policy) through `loader` into the packed safetensors file `path`, one tile
/// of host staging at a time; `config` is stored as the file's brain config. The output is complete or absent: it
/// is renamed into place only after every tensor was written, and a failure leaves no file behind.
pub fn convert_to_file(loader: &Loader<'_>, requests: &[(String, LoadPolicy)], path: &Path, config: &Value) -> Result<ConversionReport, String> {
    let plan = loader.plan_all(requests)?;
    let dsts: Vec<_> = plan.steps.iter().map(|s| s.dst.clone()).collect();
    let mut sink = SafetensorsSink::create(path, &dsts, config).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut report = ConversionReport { tensors: 0, converted: 0, payload_bytes_written: 0, peak_staging_bytes: 0 };
    for step in &plan.steps {
        let r: LoadReport = loader.execute(step, &mut sink)?;
        report.tensors += 1;
        report.converted += usize::from(step.src.encoding != step.dst.encoding);
        report.payload_bytes_written += step.dst.payload_bytes();
        report.peak_staging_bytes = report.peak_staging_bytes.max(r.peak_staging_bytes);
    }
    sink.commit().map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packed::testfix::{bf16_bytes, e4m3_bytes, write_safetensors, RawTensor, TmpDir};
    use crate::packed::{BoundedWorkspace, CodecRegistry, LoadConfig, MemorySink, MemoryBudget, PackedReader, PackedTensorReader, QuantEncoding};

    fn loader<'a>(reader: &'a PackedReader, codecs: &'a CodecRegistry) -> Loader<'a> {
        Loader { reader, codecs, workspace: BoundedWorkspace::new(MemoryBudget::new(1 << 20, 0)), config: LoadConfig { tile_bytes: 4096 } }
    }

    fn fp8_checkpoint(dir: &TmpDir) -> PackedReader {
        write_safetensors(
            &dir.join("m.safetensors"),
            &[
                RawTensor { name: "a.weight", dtype: "F8_E4M3", shape: vec![8, 128], bytes: e4m3_bytes(3, 8 * 128) },
                RawTensor { name: "a.weight_scale_inv", dtype: "BF16", shape: vec![1, 1], bytes: bf16_bytes(&[0.5]) },
                RawTensor { name: "norm", dtype: "BF16", shape: vec![16], bytes: bf16_bytes(&[1.0; 16]) },
            ],
        );
        PackedReader::open(&dir.join("m.safetensors")).unwrap()
    }

    /// The file a conversion writes holds exactly the bytes an in-memory conversion of the same tensor produces,
    /// is read back as NF4 by the ordinary packed reader with no declaration from the caller, is smaller than the
    /// FP8 it came from, and leaves the untouched tensor as stored.
    #[test]
    fn a_converted_file_reads_back_as_the_conversion_and_is_smaller() {
        let dir = TmpDir::new("convert");
        let src = fp8_checkpoint(&dir);
        let codecs = CodecRegistry::reference();
        let l = loader(&src, &codecs);
        let nf4 = QuantEncoding::nf4_blocked(32);
        let out = dir.join("out.safetensors");
        let requests = [("a.weight".to_string(), LoadPolicy::Convert(nf4.clone())), ("norm".to_string(), LoadPolicy::Preserve)];
        let report = convert_to_file(&l, &requests, &out, &serde_json::json!({})).unwrap();
        assert_eq!((report.tensors, report.converted), (2, 1));
        assert!(report.peak_staging_bytes <= 1 << 20, "staging stayed inside the budget");

        let back = PackedReader::open(&out).unwrap();
        let mut names = back.names();
        names.sort();
        assert_eq!(names, ["a.weight", "norm"], "scale sidecars stay hidden behind their payload");
        let d = back.descriptor("a.weight").unwrap();
        assert_eq!((d.shape.clone(), d.encoding.clone()), (vec![8, 128], nf4));
        assert_eq!(back.descriptor("norm").unwrap().encoding, QuantEncoding::bf16());

        // Byte-for-byte what the in-memory conversion yields.
        let mut want = MemorySink::default();
        l.load("a.weight", &LoadPolicy::Convert(QuantEncoding::nf4_blocked(32)), &mut want).unwrap();
        let mut tile = crate::packed::TileBuf::default();
        back.read_blocks(&d, &(0..d.numel()), &mut tile).unwrap();
        assert_eq!(tile.payload, want.payload);
        assert_eq!(tile.scales, want.scales);
        let fp8 = src.descriptor("a.weight").unwrap();
        assert!(d.payload_bytes() + d.scale_bytes(0) < fp8.payload_bytes() + fp8.scale_bytes(0), "NF4 stores fewer bytes than the FP8 it came from");
    }

    /// A request the loader cannot plan writes nothing: no file appears at the destination.
    #[test]
    fn a_conversion_that_cannot_be_planned_leaves_no_file() {
        let dir = TmpDir::new("noplan");
        let src = fp8_checkpoint(&dir);
        let codecs = CodecRegistry::reference();
        let out = dir.join("out.safetensors");
        let requests = [("missing.weight".to_string(), LoadPolicy::Preserve)];
        assert!(convert_to_file(&loader(&src, &codecs), &requests, &out, &serde_json::json!({})).is_err());
        assert!(!out.exists());
    }
}
