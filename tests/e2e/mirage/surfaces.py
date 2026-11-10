#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
"""Send one Mirage request over HTTP (`/v1/run`) or D-Bus (`brain_py`) and print
the action's JSON outputs (one line, key-sorted) for tests/e2e/mirage.bats to compare.

    surfaces.py http  PORT KEY ACTION IMAGE.ppm [FORM]
    surfaces.py dbus  ACTION IMAGE.ppm [FORM]
"""
import base64
import json
import sys
import urllib.request

from brain_py.base import MIRAGE_MODEL
from brain_py.dbus import BrainDBus
from brain_py.image import load_ppm

OPTIONS = {"max_tokens": 600, "mask_format": "bitmap"}


def params(action, form):
    return {"form": form} if action == "extract" else dict(OPTIONS)


def http(port, key, action, image, form):
    data, w, h = load_ppm(image)
    body = {
        "model": MIRAGE_MODEL,
        "action": action,
        "params": params(action, form),
        "blobs": {"image": {"media": "image", "data": base64.b64encode(data).decode(), "meta": {"w": w, "h": h, "c": 3}}},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/run",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["outputs"]


def dbus(action, image, form):
    data, w, h = load_ppm(image)
    with BrainDBus() as brain:
        out = brain.run(MIRAGE_MODEL, action, params(action, form), blobs={"image": data}, meta={"image": {"media": "image", "w": w, "h": h, "c": 3}})
        return out.outputs


def main():
    kind, rest = sys.argv[1], sys.argv[2:]
    if kind == "http":
        port, key, action, image = rest[:4]
        outputs = http(port, key, action, image, rest[4] if len(rest) > 4 else None)
    else:
        action, image = rest[:2]
        outputs = dbus(action, image, rest[2] if len(rest) > 2 else None)
    outputs.pop("usage", None)
    print(json.dumps(outputs, sort_keys=True))


if __name__ == "__main__":
    main()
