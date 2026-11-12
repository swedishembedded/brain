<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 450. safetensors' numpy writer saves a strided array's raw buffer, so a transposed golden input was saved scrambled

**Lesson:** a golden dumper must make every tensor contiguous before
`safetensors.numpy.save_file`. The writer copies the array's underlying
buffer, so a transpose or a channel flip (`[:, :, ::-1]`, `.transpose(...)`)
is written in memory order, not index order - and the reference run that
used the same array was correct, so the golden disagrees with its own
outputs.

## What happened

The OpenSeeFace dumper fed onnxruntime an input built as
`(rgb * scale + offset).transpose(2, 0, 1)[None]`. The network outputs it
saved were right; the input it saved was the strided view's raw buffer.
Replaying that input in the Rust parity test diverged at the very first
stage (stem cosine 0.546, max_abs 9.9) on the photo, while the seeded
input - `standard_normal`, contiguous - matched at cosine 1.0000000
everywhere. The asymmetry between two inputs through one network is what
pointed at the fixture rather than the model.

A three-line check settles it: save a `[:, ::-1].transpose(...)` view and
load it back; `np.array_equal` is `False` (safetensors 0.8.0, numpy 2.5.4).

## Rule

Route every saved dict through one `np.ascontiguousarray(v, dtype=np.float32)`
helper (`tools/goldens/openseeface_dump_reference.py::saved`). The other
numpy-writing dumpers (`instantid`, `yolov5`, `deepseek_ocr_convert_llamacpp`)
already call `ascontiguousarray`; a new one must too.
