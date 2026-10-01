<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 198. A guard test must enumerate the enum, not a copy of it

`ArchDesc` kept one tier slot per `DType` in an array whose length was a
hand-maintained constant (`DTYPE_COUNT = 7`) and indexed it with `dt as usize`.
`DType` grew to eleven variants (NF4, FP4, FP8 E4M3, FP8 E5M2); the array did
not. The test whose comment said it guarded exactly this (`every_dtype_round_
trips_its_own_tier`) hard-coded the original seven variants and compared their
count to the constant, so it kept passing. Any capability query for a newer
dtype would have panicked with an index out of bounds - latent only because
every production caller used the original seven.

Two properties made it invisible: the guard enumerated a copy of the enum
rather than the enum, and the index was an `as` cast, which accepts any
variant silently.

Now `DType::ALL` is the one list, `DType::index` is an exhaustive `match` (a
new variant does not compile until it has an index), the tier array is sized
`DType::ALL.len()`, and the round-trip test iterates `DType::ALL`. A second
test pins `ALL[i].index() == i`, so a variant added to the match but not to
`ALL` fails by name.

Rule: a per-variant table is sized from, and a guard test iterates, the same
list the type exports. A number copied into a constant and a test asserting the
copy agrees with itself prove nothing.
