<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 125. Rust drops struct fields in declaration order, and a device context is a resource's parent

A handle that owns a driver context plus the allocations, modules and graphs
made ON that context has a drop order obligation that no type signature
states. With the context declared first it is released first, and the driver
then tears down a context whose resources are still live - which faults inside
the driver rather than returning a code anything could report or log.

Nothing warns. The order is only visible as a segfault during teardown, often
only once something new (here, an instantiated graph) joins the set of
resources that outlive the release. A parent resource belongs in the LAST
field, with the reason written next to it, because the ordering is otherwise
invisible at the point someone adds field number eleven.
