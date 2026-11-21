<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 550. A `tojson` filter that does not take its keywords cannot render a template that sorts and compacts

**Lesson:** `data::chat_template` overrode minijinja's `tojson` with a one-argument function matching Python's default
`json.dumps`. Inkling 2 Mini's template writes `tojson(sort_keys=True, separators=(",", ":"))` for its tool
declaration and `tojson(separators=(",", ":"))` for call arguments, and `transformers` registers `tojson` as
`json.dumps(x, ensure_ascii=False, indent=indent, separators=separators, sort_keys=sort_keys)`. The override refused the
keywords outright (`too many arguments`), so the checkpoint's own template could not render at all; a version that
ignored them would have rendered, silently, with the wrong separators and key order. A whole-number float also printed
as `1` where Python writes `1.0`.

**How to apply:** `tojson` takes `indent`, `separators`, `sort_keys` and `ensure_ascii`, refuses any other keyword by
name, and writes floats as Python's `repr`. Pin a new template against `apply_chat_template` output with nested objects,
whole floats, non-ASCII and `None` in the arguments, and when a golden's messages are read, parse them straight into
`minijinja::Value` (`parse_json_ordered`): a `serde_json::Value` intermediate sorts the keys and hides exactly the
ordering the template is meant to preserve.
