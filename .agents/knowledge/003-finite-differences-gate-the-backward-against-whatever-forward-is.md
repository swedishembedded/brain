<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 3. Finite differences gate the backward against whatever forward is emitted

So a **mis-weighted objective is self-consistent and passes**. `check_vqgan`
cannot see which term `beta` multiplies; that is pinned by reading the reference
implementation's source, where `beta` sits on the **codebook** term - not on the
commitment term a stale local comment claimed. Finite differences prove the
derivative, never the objective.
