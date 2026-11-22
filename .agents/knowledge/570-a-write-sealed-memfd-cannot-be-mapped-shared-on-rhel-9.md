<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 570. A write-sealed memfd cannot be mapped shared on RHEL 9, even read-only

**Lesson:** the D-Bus surface returns every result as a memfd sealed `SHRINK|GROW|WRITE`. The 5.14 kernel RHEL 9
ships refuses `mmap(MAP_SHARED, PROT_READ)` of such an fd received from another process with EPERM, because the
shared mapping could later be made writable. `brain_py.dbus.read_fd` mapped every result fd that way, so every
D-Bus `Run` result failed to read there (`PermissionError: [Errno 1]`, found running the Inkling 2 sample against
`brain serve --dbus` on a GH200 node, 2026-11-22; the mock model fails the same way). The server already avoided it
for input fds (`dbus::fd`: regular files are `pread`, a dmabuf falls back to a private mapping); the client did
not. A memfd the same process created and sealed maps fine, so an in-process unit test does not reproduce it: the
socket-level sample against `brain serve --dbus --mock` does.

**How to apply:** read a regular-file fd (a memfd is one) with `pread` at explicit offsets; map only a dmabuf, shared
first and `MAP_PRIVATE` on EPERM. Every client of the fd transport (Python, shell, samples) follows `dbus::fd`'s rule.
