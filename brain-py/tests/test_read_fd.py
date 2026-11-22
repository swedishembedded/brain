# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for zero-copy IPC between applications and local inference servers for its
# clients. If your team needs expertise in D-Bus and file-descriptor transports on Linux then you can procure our
# services by sending an email to info@swedishembedded.com.

"""A result fd the server returns is a write-sealed memfd; reading it must work on every kernel.

Older kernels (5.14, RHEL 9) refuse a shared mapping of a write-sealed memfd with EPERM even when it is read-only,
so a client that maps the fd fails to read every D-Bus result there. The bytes are read with `pread` instead.
"""

from brain_py.dbus import read_fd, sealed_memfd


def test_a_sealed_result_memfd_reads_back_whole():
    data = bytes(range(256)) * 4097
    assert read_fd(sealed_memfd(data)) == data
    assert read_fd(sealed_memfd(b"")) == b""
