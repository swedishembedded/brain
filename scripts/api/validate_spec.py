#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Validate a JSON instance against one schema in a vendored OpenAPI spec.

    validate_spec.py <provider-spec.json> <SchemaName> <body.json|->

Loads the vendored OpenAPI 3.x document at <provider-spec.json>, sanitizes it into a
strict JSON-Schema Draft 2020-12 document the SAME way `crates/apiserve/tests/api.rs`'s
`spec()`/`sanitize()`/`allow_null()` do, then validates the instance in <body.json>
(or stdin when the path is "-") against `#/components/schemas/<SchemaName>`.

Exit code 0 on success; 1 (with the errors printed to stderr) on a validation failure;
2 on a usage/IO error. If `jsonschema` is not importable the script exits 3 so callers
can degrade to structural jq checks.

The sanitize step is a faithful mirror of the Rust test harness: the vendored docs use
OpenAPI's `nullable: true` (which Draft 2020-12 ignores, so a legitimately-null value
like a streaming `finish_reason` would be rejected) and one carries an invalid
`"type": null` (Anthropic's `Model`, which won't even compile). We rewrite both into
standard 2020-12 (widen `type`/`enum` to admit null, wrap a nullable `$ref`, drop the
bad `type`).
"""
import json
import sys


def allow_null(node):
    """Extend a schema object so JSON `null` is a valid instance (mirror of Rust `allow_null`)."""
    enum = node.get("enum")
    if isinstance(enum, list) and not any(e is None for e in enum):
        enum.append(None)

    ty = node.get("type")
    if isinstance(ty, str):
        node["type"] = [ty, "null"]
    elif isinstance(ty, list):
        if "null" not in ty:
            ty.append("null")
        node["type"] = ty
    elif "$ref" in node:
        ref = node.pop("$ref")
        node["anyOf"] = [{"$ref": ref}, {"type": "null"}]
    else:
        for key in ("anyOf", "oneOf"):
            arr = node.get(key)
            if isinstance(arr, list):
                arr.append({"type": "null"})
                break


def sanitize(node):
    """Recursively rewrite OpenAPI 3.x nullable/`type: null` into Draft 2020-12 (mirror of Rust `sanitize`)."""
    if isinstance(node, dict):
        if "type" in node and node["type"] is None:
            del node["type"]
        nullable = node.pop("nullable", None) is True
        if nullable:
            allow_null(node)
        for child in node.values():
            sanitize(child)
    elif isinstance(node, list):
        for child in node:
            sanitize(child)


def main(argv):
    if len(argv) != 4:
        sys.stderr.write("usage: validate_spec.py <spec.json> <SchemaName> <body.json|->\n")
        return 2
    spec_path, schema_name, body_path = argv[1], argv[2], argv[3]

    try:
        import jsonschema  # noqa: F401
        from jsonschema import Draft202012Validator
    except Exception as e:  # pragma: no cover - env-dependent
        sys.stderr.write(f"validate_spec: jsonschema unavailable ({e}); caller should degrade to jq\n")
        return 3

    try:
        with open(spec_path) as f:
            root = json.load(f)
    except (OSError, ValueError) as e:
        sys.stderr.write(f"validate_spec: cannot read spec {spec_path}: {e}\n")
        return 2

    try:
        raw = sys.stdin.read() if body_path == "-" else open(body_path).read()
        instance = json.loads(raw)
    except (OSError, ValueError) as e:
        sys.stderr.write(f"validate_spec: cannot read/parse body {body_path}: {e}\n")
        return 2

    sanitize(root)
    # Point the document's own root at the target schema so internal
    # `#/components/schemas/...` refs resolve within the same document (the
    # sibling-`$ref` trick the Rust harness uses).
    root["$ref"] = f"#/components/schemas/{schema_name}"

    try:
        validator = Draft202012Validator(root)
    except Exception as e:
        sys.stderr.write(f"validate_spec: compiling {spec_path}#{schema_name}: {e}\n")
        return 2

    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.path))
    if errors:
        sys.stderr.write(f"{spec_path}#{schema_name} rejected the instance:\n")
        for err in errors:
            loc = "/".join(str(p) for p in err.path) or "<root>"
            sys.stderr.write(f"  - {err.message} (at {loc})\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
