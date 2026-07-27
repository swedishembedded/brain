#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""End-to-end scheduler + multi-model validation over brain's D-Bus surface.

Drives `brain serve --dbus` through the residency Executor and validates, with real
numbers:

  1. generate — z-image text2image "two dogs" (streaming), timed + saved;
  2. detect   — yolo `detect` on that image (when BRAIN_YOLO is configured), boxes
                drawn client-side over the image buffer, timed + saved;
  3. batch    — several text2image requests fired concurrently must coalesce into one
                scheduler group (Stats.max_batch >= 2);
  4. evict    — requesting more distinct instance sizes than fit forces LRU eviction
                (Stats.evictions >= 1).

Exit 0 iff the scheduler validations (batch + evict) pass. Prints a PROFILE block.
Run under `brain serve --dbus` on a session bus (see the .bats wrapper).
"""
import json
import mmap
import os
import struct
import sys
import threading
import time

from jeepney import DBusAddress, new_method_call
from jeepney.fds import FileDescriptor
from jeepney.io.blocking import open_dbus_connection

ADDR = DBusAddress("/com/swedishembedded/Brain1", bus_name="com.swedishembedded.Brain1",
                   interface="com.swedishembedded.Brain1.Manager")
OUT = os.environ.get("OUT", "/tmp/brain_e2e")
os.makedirs(OUT, exist_ok=True)


def conn():
    return open_dbus_connection(bus="SESSION", enable_fds=True)


def read_fd(fdobj) -> bytes:
    raw = fdobj.to_raw_fd() if isinstance(fdobj, FileDescriptor) else int(fdobj)
    try:
        n = os.fstat(raw).st_size
        return b"" if n == 0 else bytes(mmap.mmap(raw, n, prot=mmap.PROT_READ))
    finally:
        os.close(raw)


def memfd(data: bytes) -> FileDescriptor:
    fd = os.memfd_create("e2e", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    os.write(fd, data)
    import fcntl
    fcntl.fcntl(fd, 1033, 0x1 | 0x2 | 0x4)
    return FileDescriptor(fd)


def stats(c) -> dict:
    return json.loads(c.send_and_get_reply(new_method_call(ADDR, "Stats")).body[0])


def run(c, model, action, params, in_fds=None, in_meta=None):
    msg = new_method_call(ADDR, "Run", "sssa{sh}ss",
                          (model, action, json.dumps(params), in_fds or {}, json.dumps(in_meta or {}), "memfd"))
    result, out_fds, out_meta = c.send_and_get_reply(msg).body
    return json.loads(result), out_fds, json.loads(out_meta)


def subscribe_text2image(c, prompt, w, h, steps, seed):
    """text2image via Subscribe; returns (image_bytes, meta) after the done frame."""
    msg = new_method_call(ADDR, "Subscribe", "sssa{sh}s",
                          ("z-image", "text2image",
                           json.dumps({"prompt": prompt, "width": w, "height": h, "steps": steps, "seed": seed}), {}, "{}"))
    _job, event_fd = c.send_and_get_reply(msg).body
    import socket
    sock = socket.socket(fileno=event_fd.to_raw_fd())
    sock.settimeout(600)
    img, meta = None, {}
    while True:
        data, anc, _f, _ = sock.recvmsg(1 << 16, socket.CMSG_SPACE(4 * 8))
        if not data:
            break
        fds = []
        for lvl, typ, cd in anc:
            if lvl == socket.SOL_SOCKET and typ == socket.SCM_RIGHTS:
                fds = list(struct.unpack(f"{len(cd) // 4}i", cd[: len(cd) // 4 * 4]))
        fr = json.loads(data)
        if fr["type"] == "blob" and fds:
            img, meta = read_fd(fds[0]), fr.get("meta") or {}
        elif fr["type"] in ("done", "error"):
            if fr["type"] == "error":
                raise RuntimeError(fr["message"])
            break
    sock.close()
    return img, meta


def save_ppm(path, data, w, h, c=3):
    import array
    flt = array.array("f"); flt.frombytes(data)
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(bytes(max(0, min(255, int(flt[i * c + (k if c >= 3 else 0)] * 255 + 0.5)))
                      for i in range(w * h) for k in range(3)))


def draw_boxes(rgb_f32: bytes, w, h, dets):
    """Draw detection boxes into the HWC-f32 image buffer (thick red rectangles)."""
    import array
    a = array.array("f"); a.frombytes(rgb_f32)

    def px(x, y, r, g, b):
        if 0 <= x < w and 0 <= y < h:
            i = (y * w + x) * 3
            a[i], a[i + 1], a[i + 2] = r, g, b

    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d["bbox"])
        for t in range(2):  # 2px border
            for x in range(max(0, x1), min(w, x2)):
                px(x, y1 + t, 1.0, 0.0, 0.0); px(x, y2 - t, 1.0, 0.0, 0.0)
            for y in range(max(0, y1), min(h, y2)):
                px(x1 + t, y, 1.0, 0.0, 0.0); px(x2 - t, y, 1.0, 0.0, 0.0)
    return a.tobytes()


def main():
    prof = {}
    c = conn()
    models = json.loads(c.send_and_get_reply(new_method_call(ADDR, "Manifests")).body[0])
    names = [m["model"] for m in models]
    print("models:", names, flush=True)
    if "z-image" not in names:
        print("FATAL: z-image not served (set BRAIN_ZIMAGE_*)", file=sys.stderr)
        return 2

    W = H = int(os.environ.get("SIZE", "256"))
    STEPS = int(os.environ.get("STEPS", "8"))

    # 1. generate two dogs -------------------------------------------------------
    t = time.time()
    img, meta = subscribe_text2image(c, "two dogs sitting side by side on grass, photo", W, H, STEPS, 7)
    prof["generate_s"] = round(time.time() - t, 2)
    save_ppm(f"{OUT}/dogs.ppm", img, W, H)
    print(f"[generate] {W}x{H} in {prof['generate_s']}s -> {OUT}/dogs.ppm", flush=True)

    # 2. detect + draw boxes (if yolo configured) --------------------------------
    dets = []
    if "yolo" in names:
        t = time.time()
        res, _fds, _m = run(c, "yolo", "detect", {"conf": 0.25},
                            in_fds={"image": memfd(img)}, in_meta={"image": {"media": "image", "w": W, "h": H, "c": 3}})
        prof["detect_s"] = round(time.time() - t, 2)
        dets = res.get("detections", [])
        labels = [f"{d['label']}({d['conf']:.2f})" for d in dets]
        print(f"[detect] {res.get('count', 0)} objects in {prof['detect_s']}s: {labels}", flush=True)
        annotated = draw_boxes(img, W, H, dets)
        save_ppm(f"{OUT}/dogs_boxes.ppm", annotated, W, H)
        print(f"[detect] boxes drawn -> {OUT}/dogs_boxes.ppm", flush=True)
    else:
        print("[detect] SKIPPED — no yolo model (set BRAIN_YOLO to a brain YOLOv8 checkpoint)", flush=True)

    # 3. batching: fire N same-size text2image concurrently ----------------------
    before = stats(c)
    N = int(os.environ.get("BATCH_N", "4"))
    t = time.time()

    def fire(i):
        cc = conn()
        run(cc, "z-image", "text2image", {"prompt": f"two dogs, variation {i}", "width": W, "height": H, "steps": STEPS, "seed": 100 + i})
        cc.close()

    ths = [threading.Thread(target=fire, args=(i,)) for i in range(N)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    prof["batch_wall_s"] = round(time.time() - t, 2)
    after = stats(c)
    print(f"[batch] {N} concurrent requests in {prof['batch_wall_s']}s; "
          f"max_batch={after['max_batch']} batches={after['batches']} builds={after['builds']}", flush=True)

    # 4. eviction: request 2 more distinct sizes to overflow the GPU budget ------
    t = time.time()
    for (w, h) in [(W + 32, H + 32), (W + 64, H + 64)]:
        run(c, "z-image", "text2image", {"prompt": "two dogs", "width": w, "height": h, "steps": STEPS, "seed": 5})
    prof["evict_builds_s"] = round(time.time() - t, 2)
    final = stats(c)
    print(f"[evict] +2 sizes in {prof['evict_builds_s']}s; evictions={final['evictions']} "
          f"builds={final['builds']} resident={final['resident']}", flush=True)

    c.close()

    # ---- PROFILE + validation --------------------------------------------------
    print("\n=== PROFILE ===")
    for k, v in prof.items():
        print(f"  {k:16} {v}")
    print("=== FINAL STATS ===")
    print(" ", json.dumps(final))

    ok_batch = after["max_batch"] >= 2
    ok_evict = final["evictions"] >= 1
    print(f"\nVALIDATION: batching={'PASS' if ok_batch else 'FAIL'} (max_batch={after['max_batch']}) | "
          f"eviction={'PASS' if ok_evict else 'FAIL'} (evictions={final['evictions']})", flush=True)
    return 0 if (ok_batch and ok_evict) else 1


if __name__ == "__main__":
    sys.exit(main())
