<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 492. A digest cache keyed on size and modification time served the old digest after a same-size rewrite inside one second of a whole-second filesystem

**Lesson:** A stat key proves a file unchanged only if a change is guaranteed to move it. Key a verification cache
on (device, inode, size, mtime ns, ctime ns), record when hashing began, and trust an entry only once it began more
than one timestamp tick after the file's last change; a record younger than that is re-hashed, because a rewrite
inside the tick leaves the identical stamp.

## What happened

`llmserve`'s `an_adapter_verifies_against_a_served_hugging_face_directory` writes a base, verifies an adapter, then
rewrites `model.safetensors` with different bytes of the same size and expects the adapter refused. It passed with
`TMPDIR` on a local disk and failed with `TMPDIR` on the project's Lustre mount. Logging the cache key on the failing
run showed the same `(path, 128, 1794816457000000000)` before and after the rewrite: Lustre reported whole-second
modification times, so a rewrite in the same second kept path, size and mtime, the entry hit, and the server named
the rewritten file with the old bytes' digest. The `<file>.sha256` sidecar the fine-tune save paths use had the
same key and the same hole.

## What changed

`crates/modelstore/src/fetch.rs` keys both the `base_digest_cached` cache file and the sidecar on a `Stamp` (device,
inode, size, nanosecond mtime and ctime). A rewrite that restores mtime still moves ctime on a sub-second
filesystem. Each record also carries the host time at which hashing began, and `Stamp::settled_before` trusts it
only when that is more than one tick after max(mtime, ctime): two seconds when both times are whole seconds
(Lustre or NFS without sub-second times, FAT), one second otherwise, which also absorbs coarse kernel clocks and
modest host-to-server clock skew. An unsettled entry is re-hashed and refreshed, so a base that was written long
before the server starts (every real deployment) still costs one read, ever. A file whose stamp moves while it is
being read gets no record, and the cache keeps one entry per path. Old-layout lines and sidecars no longer parse
and cost one re-hash. Spec test: `fetch::tests::a_same_size_rewrite_within_one_second_is_detected`.
