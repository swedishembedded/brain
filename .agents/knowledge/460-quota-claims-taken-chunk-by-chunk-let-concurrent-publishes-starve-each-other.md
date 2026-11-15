<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 460. Quota claims taken chunk by chunk let concurrent publishes starve each other

**Lesson:** A shared quota that concurrent writers claim a piece at a time can be split among them so that none can
finish, although any one alone fits: each holds part of the room and every next claim is refused. Claim what is
known up front (a file's size), and grow the claim only for what the estimate missed. A concurrency test that only
races threads is a coin toss; force the interleaving that breaks the invariant.

## What happened

The artifact store quotas (61ce6f38d) claimed each 1 MiB chunk of a publish against the store maximum as it was
copied. `store::tests::concurrent_publishes_cannot_jointly_exceed_the_store_maximum` (four 2 MiB artifacts racing for
3 MiB, exactly one expected) failed in the full CPU suite on a loaded node (2026-11-15) with all four refused
(`StoreQuota`), and passed 6 of 6 times when run alone. A training job's adapter, record and evaluation publishes go
through this path, so near the limit two jobs finishing together could both lose their outputs.

## Fix and evidence

`ArtifactStore::publish_from` takes the source's expected size (`publish`: the file's length; `publish_bytes`: the
slice's) and claims it whole before copying; a source that yields more claims the rest chunk by chunk.
`store::tests::of_two_publishes_racing_for_room_for_one_exactly_one_is_published` holds two 2 MiB publishes after their
first chunk until both have read it (a rendezvous with a deadline): red before the fix (both refused), green after.
