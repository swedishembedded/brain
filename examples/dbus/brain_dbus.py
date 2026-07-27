#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Python client for brain's D-Bus control surface (com.swedishembedded.Brain1).

Uses **jeepney** (pure-Python, no libdbus) with Unix-FD passing, so this doubles as
protocol documentation:

  Run(model, action, params_json, in_fds{name:fd}, in_meta_json, transport)
      -> (result_json, out_fds{name:fd}, out_meta_json)
  Subscribe(model, action, params_json, in_fds, in_meta) -> (job, event_fd)
      event_fd streams JSON frames over SOCK_SEQPACKET; a "blob" frame carries an
      out-of-band memfd via SCM_RIGHTS.

FD passing requires `open_dbus_connection(..., enable_fds=True)`; received fds come
back as jeepney `FileDescriptor` objects.

Run it under a private session bus so it needs no system config:

    dbus-run-session -- python3 examples/dbus/brain_dbus.py

With brain's z-image weights exported (BRAIN_ZIMAGE_*) it also generates an image
over D-Bus (streaming); otherwise it runs the no-GPU demo/imageops paths.
"""
import array
import json
import mmap
import os
import socket
import struct
import sys

from jeepney import DBusAddress, new_method_call
from jeepney.fds import FileDescriptor
from jeepney.io.blocking import open_dbus_connection

ADDR = DBusAddress(
    "/com/swedishembedded/Brain1",
    bus_name="com.swedishembedded.Brain1",
    interface="com.swedishembedded.Brain1.Manager",
)


def read_fd(fdobj) -> bytes:
    """Copy the bytes behind a returned fd (mmap, offset-independent). Consumes it."""
    raw = fdobj.to_raw_fd() if isinstance(fdobj, FileDescriptor) else int(fdobj)
    try:
        size = os.fstat(raw).st_size
        if size == 0:
            return b""
        with mmap.mmap(raw, size, prot=mmap.PROT_READ) as m:
            return bytes(m)
    finally:
        os.close(raw)


def memfd(data: bytes, name: str = "brain-in") -> FileDescriptor:
    """A sealed memfd holding `data`, wrapped for sending as an input fd."""
    fd = os.memfd_create(name, os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    os.write(fd, data)
    import fcntl

    fcntl.fcntl(fd, 1033, 0x1 | 0x2 | 0x4)  # F_ADD_SEALS: SHRINK|GROW|WRITE
    return FileDescriptor(fd)


def run(conn, model, action, params, in_fds=None, in_meta=None, transport="memfd"):
    """Manager.Run -> (result_dict, {name: FileDescriptor}, out_meta_dict)."""
    msg = new_method_call(
        ADDR, "Run", "sssa{sh}ss",
        (model, action, json.dumps(params), in_fds or {}, json.dumps(in_meta or {}), transport),
    )
    result_json, out_fds, out_meta_json = conn.send_and_get_reply(msg).body
    return json.loads(result_json), out_fds, json.loads(out_meta_json)


def save_ppm(path, data: bytes, w, h, c):
    """brain images are HWC f32 in [0,1] → write a viewable binary PPM."""
    flt = array.array("f")
    flt.frombytes(data)
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(bytes(max(0, min(255, int(flt[i * c + (ch if c >= 3 else 0)] * 255 + 0.5)))
                      for i in range(w * h) for ch in range(3)))


def subscribe(conn, model, action, params, timeout=180):
    """Manager.Subscribe -> yield (frame_dict, [raw_fds]) from the event stream."""
    msg = new_method_call(ADDR, "Subscribe", "sssa{sh}s",
                          (model, action, json.dumps(params), {}, "{}"))
    job, event_fd = conn.send_and_get_reply(msg).body
    sock = socket.socket(fileno=event_fd.to_raw_fd())
    sock.settimeout(timeout)
    print(f"  subscribed job={job}")
    while True:
        data, anc, _flags, _ = sock.recvmsg(1 << 16, socket.CMSG_SPACE(4 * 8))
        if not data:
            break
        fds = []
        for lvl, typ, cdata in anc:
            if lvl == socket.SOL_SOCKET and typ == socket.SCM_RIGHTS:
                n = len(cdata) // 4
                fds = list(struct.unpack(f"{n}i", cdata[: n * 4]))
        frame = json.loads(data)
        yield frame, fds
        if frame.get("type") in ("done", "error"):
            break
    sock.close()


def main():
    conn = open_dbus_connection(bus="SESSION", enable_fds=True)

    # ---- discovery ----
    manifests = json.loads(conn.send_and_get_reply(new_method_call(ADDR, "Manifests")).body[0])
    models = [m["model"] for m in manifests]
    print("models:", models)

    # ---- FD return: demo.echo → result as a memfd ----
    result, out_fds, out_meta = run(
        conn, "demo", "echo", {"text": "hello from python over dbus ", "times": 2, "mode": "upper"})
    print(f"demo.echo -> {result}, meta={out_meta['result']}, fd={read_fd(out_fds['result']).decode()!r}")

    # ---- FD return of a real image: imageops.gradient → memfd → PPM ----
    if "imageops" in models:
        result, out_fds, out_meta = run(conn, "imageops", "gradient",
                                        {"width": 128, "height": 128, "style": "aurora"})
        m = out_meta["image"]
        bm = m.get("meta") or {}
        w, h, c = bm.get("w", 128), bm.get("h", 128), bm.get("c", 3)
        img = read_fd(out_fds["image"])
        save_ppm("/tmp/brain_dbus_gradient.ppm", img, w, h, c)
        print(f"imageops.gradient -> {len(img)} bytes ({m['transport']}, {w}x{h}x{c}) -> /tmp/brain_dbus_gradient.ppm")

    # ---- FD send + streaming + FD return: z-image (needs weights + GPU) ----
    if "z-image" in models and os.environ.get("BRAIN_ZIMAGE_DIT"):
        print("z-image text2image (streaming over dbus)...")
        for frame, fds in subscribe(conn, "z-image", "text2image",
                                    {"prompt": "a red apple on a wooden table", "width": 256, "height": 256, "steps": 8}):
            t = frame["type"]
            if t == "progress":
                print(f"  [{frame['step']}/{frame['total']}] {frame['message']}")
            elif t == "blob" and fds:
                data = read_fd(fds[0])
                save_ppm("/tmp/brain_dbus_image.ppm", data, 256, 256, 3)
                print("  saved image -> /tmp/brain_dbus_image.ppm")
            elif t == "done":
                print("  done:", frame.get("result"))
            elif t == "error":
                print("  error:", frame["message"])
        # (fd send example: run(conn, "z-image", "image2image", {...},
        #  in_fds={"image": memfd(rgb_f32_bytes)}, in_meta={"image": {"media": "image", "w": W, "h": H, "c": 3}}))
    else:
        print("z-image streaming demo skipped (export BRAIN_ZIMAGE_* to enable)")

    conn.close()
    print("OK")


if __name__ == "__main__":
    sys.exit(main())
