#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Run an ONNX model on one f32 input and write the first output to stdout, raw.

A dev tool, not a build dependency: it lets a Rust test check that an emitted
graph computes the right NUMBERS, not just that it has the right nodes. Tests
that use it skip loudly when onnxruntime is absent rather than passing silently.

  onnx_eval.py <model.onnx> <input.bin> --shape N,C,H,W [--input-name x]

`input.bin` is little-endian f32 in the given shape; stdout is little-endian f32
of the model's first output.
"""

import argparse
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("input")
    ap.add_argument("--shape", required=True, help="comma-separated dims of the input")
    ap.add_argument("--input-name", default=None, help="defaults to the model's first input")
    a = ap.parse_args()

    import onnxruntime as ort

    shape = tuple(int(v) for v in a.shape.split(","))
    x = np.fromfile(a.input, dtype=np.float32).reshape(shape)

    # CPU only and single-threaded: this is an oracle, not a benchmark, and a
    # deterministic one is worth more than a fast one.
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(a.model, opts, providers=["CPUExecutionProvider"])

    name = a.input_name or sess.get_inputs()[0].name
    out = sess.run(None, {name: x})[0].astype(np.float32)
    sys.stdout.buffer.write(out.tobytes())
    return 0


if __name__ == "__main__":
    sys.exit(main())
