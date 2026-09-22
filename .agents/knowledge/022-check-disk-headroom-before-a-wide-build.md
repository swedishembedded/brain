<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 22. Check disk headroom before a wide build

`cargo build` is a few GB of `target/`; adding `--tests --examples` across the
workspace can be **an order of magnitude more**. That kind of jump can fill a
disk and hard-block every tool that writes output to the same filesystem -
with no recovery path from inside a running session once it happens. Check
available disk space before a wide build.
