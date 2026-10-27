#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for getting trained detectors onto
# edge NPUs with verified numerics for its clients. If your team needs
# expertise in model conversion for accelerator deployment then you can
# procure our services by sending an email to info@swedishembedded.com.

"""Convert a YOLOv5 ONNX export into a brain-format `.safetensors` checkpoint.

The reference input is Rockchip's NPU-oriented ReLU export (`yolov5s_relu.onnx`:
opset 12, input `images` [1,3,H,W], three post-sigmoid head outputs, batch norm
already folded into each Conv's bias). The converter does not interpret the
network: it checks that the graph is made only of the operators brain's YOLOv5
graph (`crates/yolov5/src/graph.rs`) is made of, takes every Conv's weight and
bias under the name of its module path, and writes them with a `brain.config`
header. The Rust loader (`yolov5::weights::open`) then validates the result
against the graph the header implies, in both directions, so a topology this
script does not understand fails there by tensor name rather than silently here.

Names. A Conv whose output tensor is `/model.2/m/m.0/cv1/conv/Conv_output_0`
becomes `model.2.m.0.cv1.conv.weight` / `.bias`; the head's
`/model.24/m.0/Conv_output_0` becomes `model.24.m.0.*`. (The ONNX leaves some
initializers unnamed, `onnx::Conv_290`, which is why the name comes from the
node, not the initializer.)

Usage:
    python tools/yolov5_export/export_yolov5.py --onnx yolov5s_relu.onnx \\
        [--out model.safetensors] [--width-multiple 0.5] [--depth-multiple 0.33]

`--out` defaults to `<models-dir>/local/yolov5s-relu/model.safetensors`, the
model-store location `brain_testutil::model_dir("local/yolov5s-relu")` and
`brain_modelstore` scan (`$BRAIN_MODELS_DIR`, else `$XDG_DATA_HOME/brain/models`,
else `~/.local/share/brain/models`).

Licensing. The weights of a stock YOLOv5 model are derived from Ultralytics'
AGPL-3.0 release. This tool does not fetch anything and the output file carries
a `license` note.

Requires: numpy, onnx, safetensors.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file

# The operators of the graph in `crates/yolov5/src/graph.rs`.
ALLOWED_OPS = {"Conv", "Relu", "Add", "Concat", "MaxPool", "Resize", "Sigmoid", "Constant"}
# COCO anchors, pixels, P3 first (the values every released YOLOv5 uses).
COCO_ANCHORS = [
    [10, 13, 16, 30, 33, 23],
    [30, 61, 62, 45, 59, 119],
    [116, 90, 156, 198, 373, 326],
]
NAME_RE = re.compile(r"^model\.\d+(\.[A-Za-z0-9_]+)*$")


def models_dir():
    explicit = os.environ.get("BRAIN_MODELS_DIR")
    if explicit:
        return explicit
    xdg = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(xdg, "brain", "models")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def module_path(output_name):
    """`/model.2/m/m.0/cv1/conv/Conv_output_0` -> `model.2.m.0.cv1.conv`."""
    trimmed = output_name.lstrip("/")
    if not trimmed.endswith("/Conv_output_0"):
        raise SystemExit(f"cannot derive a module path from Conv output {output_name!r}")
    name = trimmed[: -len("/Conv_output_0")].replace("/", ".")
    # The unrolled bottleneck list is `m/m.0`; the module path is `m.0`.
    name = name.replace(".m.m.", ".m.")
    if not NAME_RE.match(name):
        raise SystemExit(f"derived tensor prefix {name!r} is not a model module path")
    return name


def convert(onnx_path, width_multiple, depth_multiple, anchors):
    model = onnx.load(onnx_path)
    graph = model.graph

    ops = {n.op_type for n in graph.node}
    unsupported = sorted(ops - ALLOWED_OPS)
    if unsupported:
        raise SystemExit(f"unsupported operator(s) {unsupported}: only {sorted(ALLOWED_OPS)} are part of the brain YOLOv5 graph")
    if "Relu" not in ops:
        raise SystemExit("no Relu in the graph: only the ReLU variant is supported by this converter")

    (inp,) = graph.input
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    if len(dims) != 4 or dims[0] != 1 or dims[1] != 3 or dims[2] % 32 or dims[3] % 32:
        raise SystemExit(f"input {inp.name!r} has shape {dims}; expected [1, 3, H, W] with H, W multiples of 32")
    if dims[2] != dims[3]:
        raise SystemExit(f"non-square export {dims}: the config records one input side")

    init = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    tensors = {}
    head_channels = []
    for node in graph.node:
        if node.op_type != "Conv":
            continue
        prefix = module_path(node.output[0])
        w = init[node.input[1]].astype(np.float32)
        group = [a.i for a in node.attribute if a.name == "group"]
        if group and group[0] != 1:
            raise SystemExit(f"{prefix}: grouped convolution (group={group[0]}) is not part of the brain YOLOv5 graph")
        tensors[f"{prefix}.weight"] = np.ascontiguousarray(w)
        if len(node.input) > 2:
            tensors[f"{prefix}.bias"] = np.ascontiguousarray(init[node.input[2]].astype(np.float32))
        if re.fullmatch(r"model\.\d+\.m\.\d", prefix) and prefix.startswith("model.24."):
            head_channels.append(w.shape[0])
    if len(head_channels) != 3 or len(set(head_channels)) != 1:
        raise SystemExit(f"expected three equal-width head convolutions, found {head_channels}")
    nc, rem = divmod(head_channels[0], 3)
    nc -= 5
    if rem or nc < 1:
        raise SystemExit(f"head width {head_channels[0]} is not 3 * (5 + nc)")

    # Light consistency checks of the claimed scaling; the Rust loader does the
    # authoritative per-tensor check.
    first = tensors["model.0.conv.conv.weight"].shape[0]
    want = math.ceil(math.ceil(64 * width_multiple) / 8) * 8
    if first != want:
        raise SystemExit(f"--width-multiple {width_multiple} implies {want} channels after the stem, the graph has {first}")
    bottlenecks = len({k.split(".")[3] for k in tensors if k.startswith("model.6.m.")})
    if bottlenecks != max(round(9 * depth_multiple), 1):
        raise SystemExit(f"--depth-multiple {depth_multiple} implies {max(round(9 * depth_multiple), 1)} bottlenecks in model.6, the graph has {bottlenecks}")

    config = {
        "arch": "yolov5",
        "nc": int(nc),
        "depth_multiple": depth_multiple,
        "width_multiple": width_multiple,
        "activation": "relu",
        "anchors": anchors,
        "input": int(dims[2]),
    }
    return tensors, config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, help="the ONNX export to convert")
    ap.add_argument("--out", help="output .safetensors (default: the local model-store location)")
    ap.add_argument("--width-multiple", type=float, default=0.5, help="yolov5s = 0.5 (default)")
    ap.add_argument("--depth-multiple", type=float, default=0.33, help="yolov5s = 0.33 (default)")
    ap.add_argument("--anchors", help="JSON [[w,h]*3]*3 in pixels, P3 first (default: COCO)")
    args = ap.parse_args()

    anchors = json.loads(args.anchors) if args.anchors else COCO_ANCHORS
    tensors, config = convert(args.onnx, args.width_multiple, args.depth_multiple, anchors)

    out = args.out or os.path.join(models_dir(), "local", "yolov5s-relu", "model.safetensors")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    metadata = {
        "brain.config": json.dumps(config),
        "source_sha256": sha256_of(args.onnx),
        "license": "derived from Ultralytics YOLOv5 weights (AGPL-3.0)",
    }
    save_file(tensors, out, metadata=metadata)
    n = sum(t.size for t in tensors.values())
    print(f"wrote {out}: {len(tensors)} tensors, {n} parameters, nc={config['nc']}, input={config['input']}", file=sys.stderr)


if __name__ == "__main__":
    main()
